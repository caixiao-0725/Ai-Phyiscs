// SPDX-License-Identifier: Apache-2.0
//
// CUDA Projective Dynamics / PABD tet solver wired to ChysX's existing
// BlockCSR3 PCG.  Scenes provide mesh data; this class owns only the
// reusable simulation state and solve path.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "../../collision/mesh_topology.h"
#include "../../collision/self_collision.h"
#include "../../constraint/collision/self_collision_constraint.h"
#include "../../math/matrix.cuh"
#include "../../math/vec.cuh"
#include "../../memory/cuda_array.h"
#include "../../solver/pcg_solver.h"
#include "../../sparse/block_csr.h"

namespace chysx {
namespace rigid {
namespace pabd_cuda {

struct PabdCudaParams {
    float dt = 0.0033f;
    int substeps = 1;
    int iterations = 1;       // global PD solve passes per substep
    float gravity = -9.8f;
    float stiffness = 1200.0f;
    float density = 1.0f;
    float damping = 1.0f;
    float fixed_weight = 1.0e7f;
    float ground_y = 0.0f;
    float contact_gap = 0.035f;
    float ground_stiffness = 0.0f;
    float self_collision_thickness = 0.0f;
    float self_collision_stiffness = 0.0f;
    int self_collision_max_contacts = 256;
    int pcg_iterations = 40;
};

struct PabdCudaMesh {
    std::vector<math::Vec3f> rest_positions;
    std::vector<math::Vec3f> initial_velocities;
    std::vector<std::array<int, 4>> tets;
    std::vector<unsigned char> fixed;
    std::vector<std::array<int, 3>> surface_triangles;
};

class PabdCudaSolver {
public:
    struct TetData {
        int v[4];
        math::Mat3f inv_dm;
        float volume;
        math::Vec3f grad[4];
        int hessian_slot[16];
    };

    PabdCudaSolver() = default;
    ~PabdCudaSolver();

    PabdCudaSolver(const PabdCudaSolver&) = delete;
    PabdCudaSolver& operator=(const PabdCudaSolver&) = delete;

    void setup(const PabdCudaMesh& mesh,
               const PabdCudaParams& params = PabdCudaParams{});
    void set_params(const PabdCudaParams& params) { params_ = params; }
    const PabdCudaParams& params() const noexcept { return params_; }
    void step(float dt);

    int num_vertices() const noexcept { return num_vertices_; }
    int num_tets() const noexcept {
        return static_cast<int>(tets_.gpu_size());
    }
    int num_surface_triangles() const noexcept {
        return static_cast<int>(surface_triangles_.size());
    }

    const std::vector<float>& flat_positions() const noexcept { return flat_positions_; }
    const std::vector<int>& flat_triangles() const noexcept { return flat_triangles_; }

    int last_pcg_iterations() const noexcept { return last_pcg_iterations_; }
    float last_residual() const noexcept { return last_residual_; }
    int last_ground_contacts() const noexcept { return last_ground_contacts_; }
    int last_self_contacts() const noexcept { return last_self_contacts_; }
    float min_y() const noexcept { return min_y_; }

private:
    void release_stream() noexcept;
    void build_host_data(const PabdCudaMesh& mesh);
    void upload_static_data();
    void update_host_positions();

    PabdCudaParams params_{};
    bool initialized_ = false;

    int num_vertices_ = 0;
    std::vector<std::array<int, 3>> surface_triangles_;

    CUstream_st* stream_ = nullptr;

    sparse::BlockCSR3 H_;
    solver::PCGSolver pcg_;
    collision::MeshTopology mesh_topology_;
    collision::SelfCollisionDetector self_collision_detector_;
    constraint::SelfCollisionConstraint self_collision_;

    CudaArray<math::Vec3f> rest_;
    CudaArray<math::Vec3f> x_;
    CudaArray<math::Vec3f> old_x_;
    CudaArray<math::Vec3f> y_;
    CudaArray<math::Vec3f> velocity_;
    CudaArray<math::Vec3f> rhs_;
    CudaArray<float> mass_;
    CudaArray<unsigned char> fixed_;
    CudaArray<TetData> tets_;
    CudaArray<math::Mat3f> rotations_;
    CudaArray<int> ground_contacts_;

    std::vector<float> flat_positions_;
    std::vector<int> flat_triangles_;

    int last_pcg_iterations_ = 0;
    int last_ground_contacts_ = 0;
    int last_self_contacts_ = 0;
    float last_residual_ = 0.0f;
    float min_y_ = 0.0f;
};

PabdCudaMesh make_single_tetra_pinned_mesh();
PabdCudaMesh make_hex_drop_mesh();
PabdCudaMesh make_twenty_tets_mesh();
PabdCudaMesh make_stacked_blocks_mesh(int count,
                                      float half_extent = 0.42f,
                                      float ground_y = 0.0f,
                                      float block_gap = 0.0f);

constexpr int kTwentyTetsCount = 20;
constexpr int kHexBlockSurfaceTris = 12;
constexpr int kDefaultStackCount = 10;

}  // namespace pabd_cuda
}  // namespace rigid
}  // namespace chysx
