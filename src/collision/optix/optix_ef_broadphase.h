// SPDX-License-Identifier: Apache-2.0
//
// chysx::collision::OptixEFBroadphase
//
// OptiX-accelerated EF broadphase: cast one ray per surface edge
// through a custom-AABB BVH built over triangle faces (enlarged by
// thickness).  Produces (edge_id, face_id) candidate pairs — same
// output contract as EFBroadphase (QuantBvh variant).
//
// Requires: NVIDIA driver with OptiX support (nvoptix.dll).
//           No OptiX SDK install needed — headers are vendored.

#pragma once

#include "optix_ef_params.h"
#include "../../math/vec.cuh"
#include "../../memory/cuda_array.h"
#include "../mesh_topology.h"

#include <optix.h>
#include <vector>

namespace chysx {
namespace collision {

class OptixEFBroadphase {
public:
    OptixEFBroadphase();
    ~OptixEFBroadphase();

    OptixEFBroadphase(const OptixEFBroadphase&) = delete;
    OptixEFBroadphase& operator=(const OptixEFBroadphase&) = delete;

    // Build topology tables and initialize OptiX pipeline.
    void setup(const std::vector<math::Vec3i>& tris,
               int n_verts,
               int max_hits_per_edge = 64);

    // Build/refit the AABB BVH from current positions + thickness,
    // then launch the EF query.
    void query(const math::Vec3f* positions_dev,
               float thickness,
               std::uintptr_t cuda_stream = 0);

    // Download EF candidate pairs (edge_id, face_id) to CPU.
    void download_pairs(std::vector<math::Vec2i>& out,
                        std::uintptr_t cuda_stream = 0);
    int ef_count(std::uintptr_t cuda_stream = 0);

    bool valid() const noexcept { return initialized_; }
    const MeshTopology& topology() const noexcept { return topology_; }
    int max_ef_candidates() const noexcept { return max_ef_candidates_; }
    const math::Vec2i* ef_pairs_dev() const noexcept {
        return ef_pairs_.gpu_data();
    }
    const int* ef_count_dev() const noexcept {
        return ef_count_.gpu_data();
    }

private:
    MeshTopology topology_;
    bool initialized_ = false;
    int max_hits_per_edge_ = 64;
    int max_ef_candidates_ = 0;
    int n_edges_ = 0;
    int n_faces_ = 0;
    int rebuild_counter_ = 0;

    // OptiX handles
    OptixDeviceContext context_ = nullptr;
    OptixModule module_ = nullptr;
    OptixProgramGroup raygen_pg_ = nullptr;
    OptixProgramGroup miss_pg_ = nullptr;
    OptixProgramGroup hitgroup_pg_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixShaderBindingTable sbt_{};

    // Accel structure
    CUdeviceptr gas_output_ = 0;
    size_t gas_output_size_ = 0;
    CUdeviceptr gas_temp_ = 0;
    size_t gas_temp_size_ = 0;
    OptixTraversableHandle gas_handle_ = 0;
    OptixAccelBuildOptions accel_options_{};

    // AABB buffer (one per face)
    CudaArray<float> aabb_buffer_;  // 6 floats per AABB: mn.xyz, mx.xyz

    // Per-edge output
    CudaArray<int> hits_buffer_;
    CudaArray<int> hit_counts_;
    CudaArray<math::Vec2i> ef_pairs_;
    CudaArray<int> ef_count_;

    // Launch params
    optix_ef::Params cpu_params_{};
    CUdeviceptr gpu_params_ = 0;

    // SBT record buffers
    CUdeviceptr sbt_raygen_ = 0;
    CUdeviceptr sbt_miss_ = 0;
    CUdeviceptr sbt_hitgroup_ = 0;

    void create_context();
    void create_pipeline();
    void build_gas(const math::Vec3f* positions_dev, float thickness,
                   std::uintptr_t cuda_stream);
    void refit_gas(const math::Vec3f* positions_dev, float thickness,
                   std::uintptr_t cuda_stream);
};

}  // namespace collision
}  // namespace chysx
