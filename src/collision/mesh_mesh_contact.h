// SPDX-License-Identifier: Apache-2.0
//
// Mesh-vs-mesh discrete contact detection.
//
// Pipeline:
//   1. concatenate all participating triangle meshes into one surface mesh
//   2. EF broadphase on GPU -> (edge_id, face_id)
//   3. EF narrow phase on GPU -> point-face + edge-edge contacts
//
// This is the PeriDyno/cuda-cloth style EF decomposition kept solver-neutral:
// no point-edge/point-point output is produced, and parallel/degenerated EE
// pairs are discarded.

#pragma once

#include <cstdint>
#include <vector>

#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "ef_broadphase.h"

#ifdef CHYSX_HAS_OPTIX
#include "optix/optix_ef_broadphase.h"
#endif

#include <memory>

namespace chysx {
namespace collision {

enum class MeshMeshContactType : int {
    PointFace = 0,
    EdgeEdge = 1
};

enum class MeshCollisionCategory : int {
    Self = 0,
    InterObject = 1
};

using MeshCollisionMask = std::uint32_t;
inline constexpr MeshCollisionMask kMeshCollisionSelf = 1u << 0;
inline constexpr MeshCollisionMask kMeshCollisionInterObject = 1u << 1;
inline constexpr MeshCollisionMask kMeshCollisionAll =
    kMeshCollisionSelf | kMeshCollisionInterObject;

struct alignas(16) MeshMeshContact {
    int type;                    // MeshMeshContactType as int
    math::Vec4i vertices;         // PF: (p, f0, f1, f2), EE: (a0, a1, b0, b1)
    math::Vec4f weights;          // signed point/primitive weights
    math::Vec3f normal;           // from second primitive toward first
    float distance;               // closest-point distance
    math::Vec2i mesh_pair;        // mesh ids of the two primitives
    math::Vec2i source_ef;        // broadphase (edge_id, face_id), or (-1,-1)
    int category;                 // MeshCollisionCategory as int
    float separation;             // signed along normal; distance for unoriented meshes
};

static_assert(sizeof(MeshMeshContact) == 80,
              "MeshMeshContact must keep its compact 80-byte layout");

class MeshMeshContactDetector {
public:
    MeshMeshContactDetector() = default;

    MeshMeshContactDetector(const MeshMeshContactDetector&) = delete;
    MeshMeshContactDetector& operator=(const MeshMeshContactDetector&) = delete;
    MeshMeshContactDetector(MeshMeshContactDetector&&) noexcept = default;
    MeshMeshContactDetector& operator=(MeshMeshContactDetector&&) noexcept = default;

    // `vertex_mesh_ids[v]` is the mesh/body owner for vertex v. `collision_mask`
    // selects same-owner self collision, different-owner inter-object
    // collision, or both. The topology pre-pass is used only by Self;
    // disconnected inter-object meshes do not need adjacent supplements.
    void setup(const std::vector<math::Vec3i>& triangles,
               const std::vector<int>& vertex_mesh_ids,
               int max_contacts = -1,
               int max_ef_candidates = -1,
               BroadphaseBackend backend = BroadphaseBackend::QuantBvh,
               MeshCollisionMask collision_mask = kMeshCollisionInterObject,
               const std::vector<math::Vec3f>* reference_positions = nullptr);

    bool valid() const noexcept;

    // CPU convenience path: upload positions, then run the GPU detector.
    void detect(const math::Vec3f* positions_cpu,
                int n_verts,
                float thickness,
                std::uintptr_t cuda_stream = 0);

    // Main path: positions already live on the GPU.
    void detect_gpu(const math::Vec3f* positions_dev,
                    float thickness,
                    std::uintptr_t cuda_stream = 0);

    // Reuse a fat broadphase candidate list for at most `max_interval`
    // frames. The list is rebuilt early when any surface vertex has moved
    // more than half of `skin` from the refresh positions.
    void configure_broadphase_cache(int max_interval, float skin) noexcept;
    void invalidate_broadphase_cache() noexcept;

    int count(std::uintptr_t cuda_stream = 0);
    void download(std::vector<MeshMeshContact>& out,
                  std::uintptr_t cuda_stream = 0);

    int max_contacts() const noexcept { return max_contacts_; }
    int last_ef_count() const noexcept;
    int max_ef_candidates() const noexcept { return max_ef_candidates_; }
    BroadphaseBackend backend() const noexcept { return backend_; }
    MeshCollisionMask collision_mask() const noexcept { return collision_mask_; }
    bool last_broadphase_refreshed() const noexcept {
        return last_broadphase_refreshed_;
    }
    int broadphase_cache_age() const noexcept { return broadphase_cache_age_; }
    int broadphase_refresh_count() const noexcept {
        return broadphase_refresh_count_;
    }
    float last_broadphase_max_displacement() const noexcept {
        return last_broadphase_max_displacement_;
    }
    int last_broadphase_dropped_hits() const noexcept;

    const MeshTopology& topology() const noexcept;
    const EFBroadphase& broadphase() const noexcept { return broadphase_; }

    const CudaArray<MeshMeshContact>& contacts() const noexcept { return contacts_; }
    CudaArray<MeshMeshContact>& contacts() noexcept { return contacts_; }
    const CudaArray<int>& count_array() const noexcept { return count_; }
    CudaArray<int>& count_array() noexcept { return count_; }

private:
    BroadphaseBackend backend_ = BroadphaseBackend::QuantBvh;
    EFBroadphase broadphase_;
#ifdef CHYSX_HAS_OPTIX
    std::unique_ptr<OptixEFBroadphase> optix_broadphase_;
#endif
    CudaArray<int> vertex_mesh_ids_;
    CudaArray<float> vertex_orientation_signs_;
    CudaArray<MeshMeshContact> contacts_;
    CudaArray<int> count_;
    CudaArray<int> ef_count_snapshot_;
    CudaArray<int> broadphase_dropped_hits_snapshot_;
    CudaArray<math::Vec3f> broadphase_reference_positions_;
    CudaArray<std::uint32_t> broadphase_max_displacement_sq_bits_;
    CudaArray<math::Vec3f> upload_positions_;

    int n_verts_ = 0;
    int max_contacts_ = 0;
    int max_ef_candidates_ = 0;
    MeshCollisionMask collision_mask_ = kMeshCollisionInterObject;
    bool run_self_prepass_ = false;
    int broadphase_max_interval_ = 1;
    float broadphase_skin_ = 0.0f;
    float broadphase_cached_narrow_thickness_ = -1.0f;
    bool broadphase_cache_valid_ = false;
    int broadphase_cache_age_ = 0;
    int broadphase_refresh_count_ = 0;
    bool last_broadphase_refreshed_ = true;
    float last_broadphase_max_displacement_ = 0.0f;
};

}  // namespace collision
}  // namespace chysx
