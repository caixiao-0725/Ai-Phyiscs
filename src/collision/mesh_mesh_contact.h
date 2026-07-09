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

struct alignas(16) MeshMeshContact {
    int type;                    // MeshMeshContactType as int
    math::Vec4i vertices;         // PF: (p, f0, f1, f2), EE: (a0, a1, b0, b1)
    math::Vec4f weights;          // signed point/primitive weights
    math::Vec3f normal;           // from second primitive toward first
    float distance;               // closest-point distance
    math::Vec2i mesh_pair;        // mesh ids of the two primitives
    math::Vec2i source_ef;        // broadphase (edge_id, face_id), or (-1,-1)
};

class MeshMeshContactDetector {
public:
    MeshMeshContactDetector() = default;

    MeshMeshContactDetector(const MeshMeshContactDetector&) = delete;
    MeshMeshContactDetector& operator=(const MeshMeshContactDetector&) = delete;
    MeshMeshContactDetector(MeshMeshContactDetector&&) noexcept = default;
    MeshMeshContactDetector& operator=(MeshMeshContactDetector&&) noexcept = default;

    // `vertex_mesh_ids[v]` is the mesh/body owner for vertex v. Contacts whose
    // primitives belong entirely to the same mesh id are skipped.
    void setup(const std::vector<math::Vec3i>& triangles,
               const std::vector<int>& vertex_mesh_ids,
               int max_contacts = -1,
               int max_ef_candidates = -1,
               BroadphaseBackend backend = BroadphaseBackend::QuantBvh);

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

    int count(std::uintptr_t cuda_stream = 0);
    void download(std::vector<MeshMeshContact>& out,
                  std::uintptr_t cuda_stream = 0);

    int max_contacts() const noexcept { return max_contacts_; }
    int last_ef_count() const noexcept { return last_ef_count_; }
    int max_ef_candidates() const noexcept { return max_ef_candidates_; }
    BroadphaseBackend backend() const noexcept { return backend_; }

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
    CudaArray<MeshMeshContact> contacts_;
    CudaArray<int> count_;
    CudaArray<math::Vec3f> upload_positions_;

    int n_verts_ = 0;
    int max_contacts_ = 0;
    int max_ef_candidates_ = 0;
    int last_ef_count_ = 0;
};

}  // namespace collision
}  // namespace chysx
