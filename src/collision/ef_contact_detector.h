// SPDX-License-Identifier: Apache-2.0
//
// chysx::collision::EFContactDetector
//
// Complete EF-based contact detection pipeline:
//
//   1. EF broadphase (selectable backend: QuantBvh or OptiX)
//      → (edge_id, face_id) candidate pairs
//   2. EF → VF + EE candidate decomposition
//      - VF: vert_in_edge[edge_id] vs face (one VF per EF hit)
//      - EE: edge vs each of the face's 3 edges (deduplicated)
//   3. Narrow phase distance filter (thickness-based)
//      - VF: point-triangle closest distance < thickness
//      - EE: segment-segment closest distance < thickness,
//            with interior check (0 < s,t < 1) and degeneracy filter
//
// Input:  triangle mesh (vertices on GPU, topology built once)
// Output: ContactResult (int4 per contact + normal + distance)
//
// Reference: PeriDyno's OptixCollision/VFAndEEQuery pipeline, adapted
// to use ChysX's QuantBvh (default) or OptiX as broadphase backend.

#pragma once

#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "ef_broadphase.h"

#ifdef CHYSX_HAS_OPTIX
#include "optix/optix_ef_broadphase.h"
#endif

#include <memory>
#include <vector>

namespace chysx {
namespace collision {

enum class BroadphaseBackend {
    QuantBvh,   // default: ChysX native stackless LBVH
    OptiX       // NVIDIA OptiX ray-tracing broadphase
};

// Per-contact output: 4 vertex indices + normal + signed distance.
//
// VF (point-triangle): v[0] = point, v[1..3] = triangle vertices
// EE (edge-edge):      v[0..1] = edge A endpoints, v[2..3] = edge B endpoints
struct ContactResult {
    enum Type : int { VF = 0, EE = 1 };
    Type type;
    int v[4];
    math::Vec3f normal;
    float distance;
};

class EFContactDetector {
public:
    EFContactDetector() = default;

    EFContactDetector(const EFContactDetector&) = delete;
    EFContactDetector& operator=(const EFContactDetector&) = delete;
    EFContactDetector(EFContactDetector&&) noexcept = default;
    EFContactDetector& operator=(EFContactDetector&&) noexcept = default;

    // One-time setup: build topology and broadphase from the triangle list.
    // `backend` selects between QuantBvh (default) and OptiX.
    // OptiX requires CHYSX_HAS_OPTIX to be defined at compile time.
    void setup(const std::vector<math::Vec3i>& tris,
               int n_verts,
               int max_ef_candidates = -1,
               BroadphaseBackend backend = BroadphaseBackend::QuantBvh);

    // Full detection pass:
    //   1. Upload `positions` (CPU Vec3f array, n_verts entries) to GPU
    //   2. EF broadphase on GPU
    //   3. Download EF pairs
    //   4. Narrow phase on CPU: EF → VF + EE with distance filter
    //
    // Results are appended to `contacts`.
    void detect(const math::Vec3f* positions_cpu,
                int n_verts,
                float thickness,
                std::vector<ContactResult>& contacts);

    // GPU variant: positions are already on device.
    void detect_gpu(const math::Vec3f* positions_dev,
                    const math::Vec3f* positions_cpu,
                    int n_verts,
                    float thickness,
                    std::vector<ContactResult>& contacts);

    // ---- accessors --------------------------------------------------------

    bool valid() const noexcept;
    const MeshTopology& topology() const noexcept;
    const EFBroadphase& broadphase() const noexcept { return qbvh_broadphase_; }
    EFBroadphase& broadphase() noexcept { return qbvh_broadphase_; }
    BroadphaseBackend backend() const noexcept { return backend_; }

private:
    BroadphaseBackend backend_ = BroadphaseBackend::QuantBvh;
    EFBroadphase qbvh_broadphase_;
#ifdef CHYSX_HAS_OPTIX
    std::unique_ptr<OptixEFBroadphase> optix_broadphase_;
#endif
    CudaArray<math::Vec3f> verts_gpu_;

    // Decompose EF pairs into VF + EE contacts with distance filter.
    void narrow_phase(const std::vector<math::Vec2i>& ef_pairs,
                      const math::Vec3f* positions,
                      float thickness,
                      std::vector<ContactResult>& contacts);

    // Supplementary adjacent-pair culling (missed by vert_in_edge trick).
    void cull_adjacent_vf(const math::Vec3f* positions,
                          float thickness,
                          std::vector<ContactResult>& contacts);
    void cull_adjacent_ee(const math::Vec3f* positions,
                          float thickness,
                          std::vector<ContactResult>& contacts);
};

}  // namespace collision
}  // namespace chysx
