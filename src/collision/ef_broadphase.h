// SPDX-License-Identifier: Apache-2.0
//
// chysx::collision::EFBroadphase
//
// GPU-accelerated edge-face broadphase using QuantBvh.
// Given a triangle surface mesh (vertices, tris, edges), builds a BVH
// over face AABBs and queries it with edge AABBs to produce candidate
// (edge_id, face_id) pairs.  These can be decomposed into vertex-face
// and edge-edge contacts by the caller.
//
// This is extracted from SelfCollisionDetector's BVH pipeline to be
// reusable by any solver that needs triangle-level broadphase.

#pragma once

#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "bvh/aabb.cuh"
#include "bvh/quant_bvh.h"
#include "mesh_topology.h"

namespace chysx {
namespace collision {

enum class BroadphaseBackend {
    QuantBvh,
    OptiX
};

class EFBroadphase {
public:
    EFBroadphase() = default;

    EFBroadphase(const EFBroadphase&) = delete;
    EFBroadphase& operator=(const EFBroadphase&) = delete;
    EFBroadphase(EFBroadphase&&) noexcept = default;
    EFBroadphase& operator=(EFBroadphase&&) noexcept = default;

    // Build topology tables and allocate BVH for the given mesh.
    // `tris` and `n_verts` define the mesh connectivity; edges, vert_in_edge,
    // and edge_in_face are derived internally via MeshTopology.
    // `max_ef_candidates` caps the output pair list; 8*n_edges is typical.
    void setup(const std::vector<math::Vec3i>& tris,
               int n_verts,
               int max_ef_candidates = -1);

    // Run the broadphase: build face AABBs (enlarged by `thickness`),
    // refit the BVH, and query all edges against it.
    //
    // `positions` must be a device pointer to n_verts Vec3f entries
    // holding current (deformed) vertex positions.
    void query(const math::Vec3f* positions_dev,
               float thickness,
               std::uintptr_t cuda_stream = 0);

    // Number of EF candidates found by the last query().
    // Reads the device counter synchronously.
    int ef_count(std::uintptr_t cuda_stream = 0);

    // Download the EF candidate pairs to a CPU vector.
    // Each entry is (edge_id, face_id).
    void download_pairs(std::vector<math::Vec2i>& out,
                        std::uintptr_t cuda_stream = 0);

    // ---- accessors --------------------------------------------------------

    bool valid() const noexcept { return topology_.valid(); }

    const MeshTopology& topology() const noexcept { return topology_; }

    int max_ef_candidates() const noexcept { return max_ef_cand_; }

    // GPU-side pair list and count pointer (for direct GPU narrow phase).
    const math::Vec2i* ef_pairs_dev() const noexcept {
        return bvh_.query_pairs_dev();
    }
    const int* ef_count_dev() const noexcept {
        return bvh_.query_count_dev();
    }

private:
    MeshTopology topology_;
    QuantBvh bvh_;

    CudaArray<Aabb>        face_aabbs_;
    CudaArray<math::Vec3f> face_centers_;

    int max_ef_cand_ = 0;
};

}  // namespace collision
}  // namespace chysx
