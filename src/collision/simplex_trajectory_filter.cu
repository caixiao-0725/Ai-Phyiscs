// SPDX-License-Identifier: Apache-2.0
// See simplex_trajectory_filter.h. Port of libuipc lbvh_simplex_trajectory_filter.

#include "simplex_trajectory_filter.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>

namespace chysx {
namespace collision {

using math::Vec2i;
using math::Vec3i;
using math::Vec3f;

namespace {

constexpr int kBlock = 256;
inline int grid(int n) { return (n + kBlock - 1) / kBlock; }

// Build a swept AABB for a single point: bounds {pos, pos+dx*alpha}, inflated.
__global__ void build_point_aabbs(const Vec3f* __restrict__ pos,
                                   const Vec3f* __restrict__ dx,
                                   float alpha, float expand, int n,
                                   Aabb* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Vec3f p = pos[i];
    Aabb a;
    a.add(p);
    if (dx) a.add(p + dx[i] * alpha);
    a.enlarge(expand);
    out[i] = a;
}

__global__ void build_edge_aabbs(const Vec3f* __restrict__ pos,
                                 const Vec3f* __restrict__ dx,
                                 float alpha, float expand,
                                 const Vec2i* __restrict__ edges, int n,
                                 Aabb* __restrict__ out, Vec3f* __restrict__ center) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Vec2i e = edges[i];
    Vec3f p0 = pos[e.x], p1 = pos[e.y];
    Aabb a;
    a.add(p0); a.add(p1);
    if (dx) { a.add(p0 + dx[e.x] * alpha); a.add(p1 + dx[e.y] * alpha); }
    a.enlarge(expand);
    out[i] = a;
    center[i] = a.center();
}

__global__ void build_tri_aabbs(const Vec3f* __restrict__ pos,
                                const Vec3f* __restrict__ dx,
                                float alpha, float expand,
                                const Vec3i* __restrict__ tris, int n,
                                Aabb* __restrict__ out, Vec3f* __restrict__ center) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Vec3i f = tris[i];
    Vec3f p0 = pos[f.x], p1 = pos[f.y], p2 = pos[f.z];
    Aabb a;
    a.add(p0); a.add(p1); a.add(p2);
    if (dx) {
        a.add(p0 + dx[f.x] * alpha);
        a.add(p1 + dx[f.y] * alpha);
        a.add(p2 + dx[f.z] * alpha);
    }
    a.enlarge(expand);
    out[i] = a;
    center[i] = a.center();
}

}  // namespace

void SimplexTrajectoryFilter::setup(int n_verts,
                                    const std::vector<Vec2i>& edges,
                                    const std::vector<Vec3i>& tris,
                                    const std::vector<int>& vert_body) {
    n_verts_ = n_verts;
    n_edges_ = static_cast<int>(edges.size());
    n_tris_  = static_cast<int>(tris.size());

    d_edges_.resize(n_edges_);
    std::memcpy(d_edges_.cpu_data(), edges.data(), n_edges_ * sizeof(Vec2i));
    d_edges_.copy_to_device();

    d_tris_.resize(n_tris_);
    std::memcpy(d_tris_.cpu_data(), tris.data(), n_tris_ * sizeof(Vec3i));
    d_tris_.copy_to_device();

    d_vert_body_.resize(n_verts_);
    std::memcpy(d_vert_body_.cpu_data(), vert_body.data(), n_verts_ * sizeof(int));
    d_vert_body_.copy_to_device();

    point_aabbs_.resize(n_verts_);
    edge_aabbs_.resize(n_edges_);
    tri_aabbs_.resize(n_tris_);
    edge_centers_.resize(n_edges_);
    tri_centers_.resize(n_tris_);

    // Generous candidate capacity. PT candidates scale with the proximity
    // overlap, which is bounded for d_hat-scale margins; size for the worst
    // contact-rich case and clamp inside the BVH query if exceeded.
    int cap = std::max({n_verts_, n_edges_, n_tris_}) * 64;
    cap = std::max(cap, 1 << 18);

    tri_bvh_.build(n_tris_, cap);
    edge_bvh_.build(n_edges_, cap);

    ready_ = true;
}

void SimplexTrajectoryFilter::detect(const Vec3f* positions_dev,
                                     const Vec3f* dx_dev,
                                     float alpha,
                                     float expand,
                                     std::uintptr_t cuda_stream) {
    if (!ready_) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    build_point_aabbs<<<grid(n_verts_), kBlock, 0, stream>>>(
        positions_dev, dx_dev, alpha, expand, n_verts_, point_aabbs_.gpu_data());
    build_edge_aabbs<<<grid(n_edges_), kBlock, 0, stream>>>(
        positions_dev, dx_dev, alpha, expand, d_edges_.gpu_data(), n_edges_,
        edge_aabbs_.gpu_data(), edge_centers_.gpu_data());
    build_tri_aabbs<<<grid(n_tris_), kBlock, 0, stream>>>(
        positions_dev, dx_dev, alpha, expand, d_tris_.gpu_data(), n_tris_,
        tri_aabbs_.gpu_data(), tri_centers_.gpu_data());

    // Triangle BVH <- point query gives point-triangle candidates.
    tri_bvh_.refit(tri_aabbs_.gpu_data(), tri_centers_.gpu_data(), cuda_stream);
    tri_bvh_.query_cross(point_aabbs_.gpu_data(), n_verts_, cuda_stream);

    // Edge BVH self-query gives edge-edge candidates.
    edge_bvh_.refit(edge_aabbs_.gpu_data(), edge_centers_.gpu_data(), cuda_stream);
    edge_bvh_.query_self_aabb(edge_aabbs_.gpu_data(), cuda_stream);

    cudaStreamSynchronize(stream);
    cudaMemcpy(&n_pt_pairs_, tri_bvh_.query_count_dev(), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&n_ee_pairs_, edge_bvh_.query_count_dev(), sizeof(int), cudaMemcpyDeviceToHost);
    if (n_pt_pairs_ > tri_bvh_.max_query_pairs())  n_pt_pairs_ = tri_bvh_.max_query_pairs();
    if (n_ee_pairs_ > edge_bvh_.max_query_pairs()) n_ee_pairs_ = edge_bvh_.max_query_pairs();
}

void SimplexTrajectoryFilter::download_pt(std::vector<Vec2i>& out) const {
    out.resize(n_pt_pairs_);
    if (n_pt_pairs_ > 0)
        cudaMemcpy(out.data(), tri_bvh_.query_pairs_dev(),
                   n_pt_pairs_ * sizeof(Vec2i), cudaMemcpyDeviceToHost);
}

void SimplexTrajectoryFilter::download_ee(std::vector<Vec2i>& out) const {
    out.resize(n_ee_pairs_);
    if (n_ee_pairs_ > 0)
        cudaMemcpy(out.data(), edge_bvh_.query_pairs_dev(),
                   n_ee_pairs_ * sizeof(Vec2i), cudaMemcpyDeviceToHost);
}

}  // namespace collision
}  // namespace chysx
