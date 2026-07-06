// SPDX-License-Identifier: Apache-2.0
//
// Yarn collision detection kernels, ported from YarnBall collision.cu.
// Uses ChysX QuantBVH for broadphase instead of YarnBall's custom LBVH.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#include "yarn_types.cuh"

namespace chysx {
namespace yarn {

// Build per-segment AABBs from positions + padding.
__global__ void kernel_build_aabbs(YarnMetaData* data) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= data->numVerts) return;

    auto verts = data->d_lastPos;
    auto flags = data->d_lastFlags[tid];
    collision::Aabb aabb;  // default: min=+inf, max=-inf

    Vec3f p0 = verts[tid];
    if (flags & static_cast<uint32_t>(VertexFlags::hasNext)) {
        Vec3f p1 = verts[tid + 1];
        aabb.add(p0);
        aabb.add(p1);

        auto pads = data->d_paddingSize;
        float pad = fmaxf(pads[tid], pads[tid + 1]);
        aabb.enlarge(pad);
    }

    data->d_bounds[tid] = aabb;
}

// Narrowphase: for each broadphase pair, compute segment–segment
// closest points and register collision if within padded distance.
__global__ void kernel_build_collision_list(YarnMetaData* data,
                                             const Vec2i* pairs,
                                             int n_pairs) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_pairs) return;

    const int numVerts = data->numVerts;
    Vec2i ids = pairs[tid];

    // ids are original leaf indices = segment vertex indices
    auto cids = data->d_lastCID;
    int c0 = cids[ids.x];
    int c1 = cids[ids.x + 1];

    // Skip self-collisions due to glueing
    if (c0 == ids.y || c1 == ids.y || c0 == ids.y + 1 || c1 == ids.y + 1) return;
    // Skip neighboring segments
    if (abs(ids.y - ids.x) <= 2) return;

    auto verts = data->d_lastPos;
    Vec3f a0 = verts[ids.x];
    Vec3f a1 = verts[ids.x + 1] - a0;
    Vec3f b0 = verts[ids.y] - a0;
    Vec3f b1 = verts[ids.y + 1] - a0;

    Vec2f uv = segment_closest_points(Vec3f(0.f, 0.f, 0.f), a1, b0, b1);

    Vec3f normal = uv.x * a1 - (b0 * (1.f - uv.y) + b1 * uv.y);
    float d2 = dot(normal, normal);

    auto pads = data->d_paddingSize;
    float r = fmaxf(pads[ids.x], pads[ids.x + 1])
            + fmaxf(pads[ids.y], pads[ids.y + 1]);

    if (d2 < r * r) {
        auto nCols = data->d_numCols;
        const auto collisions = data->d_collisions;

        int nc0 = atomicAdd(&nCols[ids.x], 1);
        if (nc0 < MAX_COLLISIONS_PER_SEGMENT)
            collisions[ids.x + numVerts * nc0] = ids.y;

        int nc1 = atomicAdd(&nCols[ids.y], 1);
        if (nc1 < MAX_COLLISIONS_PER_SEGMENT)
            collisions[ids.y + numVerts * nc1] = ids.x;
    }
}

// Step-size limiter kernel.
__global__ void kernel_recompute_step_limit(YarnMetaData* data) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int numVerts = data->numVerts;
    if (tid >= numVerts) return;

    float minDist = INFINITY;
    if (data->d_lastFlags[tid] & static_cast<uint32_t>(VertexFlags::hasNext)) {
        constexpr float SAFETY_MARGIN = 0.1f;

        const auto verts = data->d_lastPos;
        Vec3f a0 = verts[tid];
        Vec3f a1 = verts[tid + 1] - a0;

        minDist = data->d_paddingSize[tid] - data->detectionRadius;

        const int numCols = data->d_numCols[tid];
        const auto collisions = data->d_collisions + tid;
        for (int i = 0; i < numCols; ++i) {
            int oid = collisions[i * numVerts];
            Vec3f b0 = verts[oid] - a0;
            Vec3f b1 = verts[oid + 1] - a0;

            Vec2f uv = segment_closest_points(Vec3f(0.f, 0.f, 0.f), a1, b0, b1);
            Vec3f normal = uv.x * a1 - (b0 * (1.f - uv.y) + b1 * uv.y);
            float l = length(normal);
            minDist = fminf(minDist, ((1.f - SAFETY_MARGIN) * 0.5f) * l);
        }
    }
    data->d_maxStepSize[tid] = minDist;
}

// Compute AABB centers entirely on GPU (avoids GPU→CPU→GPU round-trip).
__global__ void kernel_compute_aabb_centers(const collision::Aabb* bounds,
                                             math::Vec3f* centers, int n) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    centers[tid] = bounds[tid].center();
}

// Host-callable wrappers.
void launch_build_aabbs(YarnMetaData* d_meta, int n_verts, cudaStream_t stream) {
    int blocks = (n_verts + 255) / 256;
    kernel_build_aabbs<<<blocks, 256, 0, stream>>>(d_meta);
}

void launch_compute_aabb_centers(const collision::Aabb* bounds,
                                  math::Vec3f* centers, int n,
                                  cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    kernel_compute_aabb_centers<<<blocks, 256, 0, stream>>>(bounds, centers, n);
}

void launch_build_collision_list(YarnMetaData* d_meta, const Vec2i* pairs,
                                  int n_pairs, cudaStream_t stream) {
    if (n_pairs <= 0) return;
    int blocks = (n_pairs + 127) / 128;
    kernel_build_collision_list<<<blocks, 128, 0, stream>>>(d_meta, pairs, n_pairs);
}

void launch_recompute_step_limit(YarnMetaData* d_meta, int n_verts, cudaStream_t stream) {
    int blocks = (n_verts + 127) / 128;
    kernel_recompute_step_limit<<<blocks, 128, 0, stream>>>(d_meta);
}

}  // namespace yarn
}  // namespace chysx
