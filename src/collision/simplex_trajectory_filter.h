// SPDX-License-Identifier: Apache-2.0
//
// chysx::collision::SimplexTrajectoryFilter
//
// Faithful port of libuipc's cuda backend
//   collision_detection/filters/lbvh_simplex_trajectory_filter.cu
// adapted to ChysX's LinearBvh and CudaArray.
//
// Pipeline (per detect()):
//   1. Build swept AABBs for surface vertices / edges / triangles. Each AABB
//      bounds the primitive over the whole step (position .. position+dx*alpha)
//      and is expanded by (d_hat + thickness), so the candidate set is
//      COMPLETE for both discrete (d_hat-proximity) and continuous (swept)
//      collision — unlike the edge-face "vert_in_edge" broadphase which only
//      represents one vertex per edge.
//   2. Build a LinearBvh over triangle AABBs and over edge AABBs.
//   3. Query point AABBs vs the triangle BVH  -> point-triangle candidates.
//      Self-query the edge BVH                -> edge-edge candidates.
//   4. Cross-body / shared-vertex filtering on each candidate pair.
//
// The narrow phase (exact distance for DCD, time-of-impact for CCD) is left to
// the caller, which already owns the point-triangle / edge-edge distance and
// CCD routines. This module only delivers the complete candidate pair lists,
// matching the broadphase role of libuipc's filter.detect().

#pragma once

#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "bvh/lbvh.h"
#include "bvh/aabb.cuh"

#include <vector>

namespace chysx {
namespace collision {

class SimplexTrajectoryFilter {
public:
    SimplexTrajectoryFilter() = default;

    // One-time topology upload. `vert_body[i]` is the body id of vertex i,
    // used to drop intra-body candidate pairs (self-collision off).
    void setup(int n_verts,
               const std::vector<math::Vec2i>& edges,
               const std::vector<math::Vec3i>& tris,
               const std::vector<int>& vert_body);

    // Build swept AABBs from `positions_dev` (+ optional `dx_dev`*alpha) and
    // run the LBVH broadphase. `dx_dev` may be null for a pure discrete query
    // (alpha ignored). `expand` = d_hat (+ thickness) inflation radius.
    // After this call the candidate pairs live on the GPU; use the accessors
    // or download_*() to read them.
    void detect(const math::Vec3f* positions_dev,
                const math::Vec3f* dx_dev,
                float alpha,
                float expand,
                std::uintptr_t cuda_stream = 0);

    // ---- candidate accessors (valid after detect) ------------------------

    int num_pt_pairs() const noexcept { return n_pt_pairs_; }
    int num_ee_pairs() const noexcept { return n_ee_pairs_; }

    // (point_id, triangle_id) pairs on device.
    const math::Vec2i* pt_pairs_dev() const noexcept { return tri_bvh_.query_pairs_dev(); }
    // (edge_id_a, edge_id_b) pairs on device.
    const math::Vec2i* ee_pairs_dev() const noexcept { return edge_bvh_.query_pairs_dev(); }

    // Download candidate pairs to host vectors (resized to the counts).
    void download_pt(std::vector<math::Vec2i>& out) const;
    void download_ee(std::vector<math::Vec2i>& out) const;

    // Topology accessors (device).
    const math::Vec2i* edges_dev() const noexcept { return d_edges_.gpu_data(); }
    const math::Vec3i* tris_dev() const noexcept { return d_tris_.gpu_data(); }
    const int* vert_body_dev() const noexcept { return d_vert_body_.gpu_data(); }
    int n_edges() const noexcept { return n_edges_; }
    int n_tris() const noexcept { return n_tris_; }
    int n_verts() const noexcept { return n_verts_; }

private:
    int n_verts_ = 0, n_edges_ = 0, n_tris_ = 0;
    bool ready_ = false;

    CudaArray<math::Vec2i> d_edges_;
    CudaArray<math::Vec3i> d_tris_;
    CudaArray<int> d_vert_body_;

    CudaArray<Aabb> point_aabbs_;
    CudaArray<Aabb> edge_aabbs_;
    CudaArray<Aabb> tri_aabbs_;
    CudaArray<math::Vec3f> edge_centers_;
    CudaArray<math::Vec3f> tri_centers_;

    LinearBvh tri_bvh_;
    LinearBvh edge_bvh_;

    int n_pt_pairs_ = 0;
    int n_ee_pairs_ = 0;
};

}  // namespace collision
}  // namespace chysx
