// SPDX-License-Identifier: Apache-2.0
// CUDA kernels for the OptiX EF broadphase (AABB construction).

#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

#include "../../math/vec.cuh"

namespace chysx {
namespace collision {

// Build per-triangle AABBs (6 floats each: min_x, min_y, min_z, max_x, max_y, max_z)
__global__ void optix_ef_build_aabb_kernel(
    float* __restrict__ aabb_buffer,
    const float3* __restrict__ verts,
    const int3* __restrict__ tris,
    float thickness,
    int n_faces) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_faces) return;

    int3 f = tris[i];
    float3 v0 = verts[f.x], v1 = verts[f.y], v2 = verts[f.z];

    float mn_x = fminf(fminf(v0.x, v1.x), v2.x) - thickness;
    float mn_y = fminf(fminf(v0.y, v1.y), v2.y) - thickness;
    float mn_z = fminf(fminf(v0.z, v1.z), v2.z) - thickness;
    float mx_x = fmaxf(fmaxf(v0.x, v1.x), v2.x) + thickness;
    float mx_y = fmaxf(fmaxf(v0.y, v1.y), v2.y) + thickness;
    float mx_z = fmaxf(fmaxf(v0.z, v1.z), v2.z) + thickness;

    aabb_buffer[i * 6 + 0] = mn_x;
    aabb_buffer[i * 6 + 1] = mn_y;
    aabb_buffer[i * 6 + 2] = mn_z;
    aabb_buffer[i * 6 + 3] = mx_x;
    aabb_buffer[i * 6 + 4] = mx_y;
    aabb_buffer[i * 6 + 5] = mx_z;
}

void optix_ef_build_aabbs(float* aabb_buffer,
                           const float3* verts,
                           const int3* tris,
                           float thickness,
                           int n_faces,
                           cudaStream_t stream) {
    int block = 128;
    int grid = (n_faces + block - 1) / block;
    optix_ef_build_aabb_kernel<<<grid, block, 0, stream>>>(
        aabb_buffer, verts, tris, thickness, n_faces);
}

__global__ void optix_ef_flatten_hits_kernel(
    const int* __restrict__ hits_buffer,
    const int* __restrict__ hit_counts,
    int n_edges,
    int max_hits_per_edge,
    math::Vec2i* __restrict__ ef_pairs,
    int* __restrict__ pair_count,
    int max_pairs,
    int* __restrict__ dropped_hits) {
    using BlockScan = cub::BlockScan<int, 128>;
    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ int block_base;

    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    const int raw_n = eid < n_edges ? hit_counts[eid] : 0;
    const int n = min(raw_n, max_hits_per_edge);
    if (raw_n > max_hits_per_edge) {
        atomicAdd(dropped_hits, raw_n - max_hits_per_edge);
    }
    int local_offset = 0;
    int block_total = 0;
    BlockScan(scan_storage).ExclusiveSum(n, local_offset, block_total);
    if (threadIdx.x == 0) {
        block_base = atomicAdd(pair_count, block_total);
    }
    __syncthreads();

    if (eid >= n_edges) return;
    const int out_base = block_base + local_offset;
    for (int j = 0; j < n; ++j) {
        const int out = out_base + j;
        if (out < max_pairs) {
            ef_pairs[out] =
                math::Vec2i(eid, hits_buffer[eid * max_hits_per_edge + j]);
        }
    }
}

void optix_ef_flatten_hits(const int* hits_buffer,
                           const int* hit_counts,
                           int n_edges,
                           int max_hits_per_edge,
                           math::Vec2i* ef_pairs,
                           int* pair_count,
                           int max_pairs,
                           int* dropped_hits,
                           cudaStream_t stream) {
    cudaMemsetAsync(pair_count, 0, sizeof(int), stream);
    cudaMemsetAsync(dropped_hits, 0, sizeof(int), stream);
    int block = 128;
    int grid = (n_edges + block - 1) / block;
    optix_ef_flatten_hits_kernel<<<grid, block, 0, stream>>>(
        hits_buffer, hit_counts, n_edges, max_hits_per_edge,
        ef_pairs, pair_count, max_pairs, dropped_hits);
}

}  // namespace collision
}  // namespace chysx
