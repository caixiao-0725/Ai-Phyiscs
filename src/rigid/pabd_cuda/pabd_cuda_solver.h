// SPDX-License-Identifier: Apache-2.0
//
// CUDA Projective Dynamics / PABD tet solver wired to ChysX's existing
// BlockCSR3 PCG.  Scenes provide mesh data; this class owns only the
// reusable simulation state and solve path.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "../../collision/mesh_mesh_contact.h"
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

struct PabdSurfaceMap {
    std::array<int, 4> index = {-1, -1, -1, -1};
    std::array<float, 4> weight = {0.0f, 0.0f, 0.0f, 0.0f};
    math::Vec3f normal = math::Vec3f(0.0f, 0.0f, 0.0f);
    int body = 0;
    int collide = 1;
    int self_collide = 1;
    int ground_collide = 1;
};

struct PabdCudaMesh {
    std::vector<math::Vec3f> rest_positions;
    std::vector<math::Vec3f> initial_velocities;
    std::vector<std::array<int, 4>> tets;
    std::vector<unsigned char> fixed;
    std::vector<PabdSurfaceMap> surface_maps;
    std::vector<std::array<int, 3>> surface_triangles;
    std::vector<std::array<int, 2>> surface_edges;
    std::vector<float> tet_volume_overrides;
    std::vector<std::array<float, 16>> tet_mass_blocks;
};

struct PabdSurfaceMapDevice {
    int index[4]{-1, -1, -1, -1};
    float weight[4]{0.0f, 0.0f, 0.0f, 0.0f};
    math::Vec3f normal = math::Vec3f(0.0f, 0.0f, 0.0f);
    int body = 0;
    int collide = 1;
    int self_collide = 1;
    int ground_collide = 1;
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
    void set_auto_download_positions(bool enabled) noexcept {
        auto_download_positions_ = enabled;
    }
    void step(float dt);

    int num_vertices() const noexcept { return num_vertices_; }
    int num_tets() const noexcept {
        return static_cast<int>(tets_.gpu_size());
    }
    int num_surface_triangles() const noexcept {
        return static_cast<int>(surface_triangles_.size());
    }
    int num_surface_vertices() const noexcept { return num_surface_vertices_; }

    const std::vector<float>& flat_positions() const noexcept { return flat_positions_; }
    const std::vector<int>& flat_triangles() const noexcept { return flat_triangles_; }
    const math::Vec3f* interpolated_surface_positions_device() const noexcept {
        return interpolated_surface_positions_dev_.gpu_data();
    }
    const math::Vec3i* surface_triangles_device() const noexcept {
        return surface_triangles_dev_.gpu_data();
    }

    int last_pcg_iterations() const noexcept { return last_pcg_iterations_; }
    float last_residual() const noexcept { return last_residual_; }
    int last_ground_contacts() const noexcept { return last_ground_contacts_; }
    int last_self_contacts() const noexcept { return last_self_contacts_; }
    int last_self_point_face_contacts() const noexcept { return last_self_point_face_contacts_; }
    int last_self_edge_edge_contacts() const noexcept { return last_self_edge_edge_contacts_; }
    int last_self_vertical_contacts() const noexcept { return last_self_vertical_contacts_; }
    int last_self_horizontal_contacts() const noexcept { return last_self_horizontal_contacts_; }
    int last_self_broadphase_pairs() const noexcept { return last_self_broadphase_pairs_; }
    int last_self_broadphase_capacity() const noexcept {
        return last_self_broadphase_capacity_;
    }
    bool last_self_broadphase_overflow() const noexcept {
        return last_self_broadphase_overflow_;
    }
    int last_self_raw_contacts() const noexcept { return last_self_raw_contacts_; }
    int interpolated_contact_capacity() const noexcept {
        return interpolated_contact_capacity_;
    }
    bool last_self_contact_overflow() const noexcept {
        return last_self_contact_overflow_;
    }
    math::Vec3f last_self_normal_sum() const noexcept { return last_self_normal_sum_; }
    float min_y() const noexcept { return min_y_; }

private:
    void release_stream() noexcept;
    void build_host_data(const PabdCudaMesh& mesh);
    void upload_static_data();
    void update_host_positions(bool x_already_on_host = false);
    void step_interpolated(float dt);
    void assemble_interpolated_system_gpu(float h);
    void update_surface_positions(const std::vector<math::Vec3f>& controls,
                                  std::vector<math::Vec3f>& surface) const;
    void update_interpolated_surface_positions_gpu(const math::Vec3f* controls_dev,
                                                   bool update_min_y = false);
    void detect_interpolated_contacts_gpu();

    PabdCudaParams params_{};
    bool initialized_ = false;
    bool auto_download_positions_ = true;

    int num_vertices_ = 0;
    int num_surface_vertices_ = 0;
    bool interpolated_surface_ = false;
    std::vector<std::array<int, 3>> surface_triangles_;
    std::vector<std::array<int, 2>> surface_edges_;
    std::vector<PabdSurfaceMap> surface_maps_;
    std::vector<TetData> host_tets_;
    std::vector<std::array<float, 16>> host_mass_blocks_;
    std::vector<math::Vec3f> host_surface_positions_;
    int interpolated_body_count_ = 0;
    int interpolated_contact_capacity_ = 0;

    CUstream_st* stream_ = nullptr;

    sparse::BlockCSR3 H_;
    solver::PCGSolver pcg_;
    collision::MeshTopology mesh_topology_;
    collision::SelfCollisionDetector self_collision_detector_;
    collision::MeshMeshContactDetector mesh_mesh_contact_detector_;
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
    CudaArray<PabdSurfaceMapDevice> surface_maps_dev_;
    CudaArray<math::Vec3i> surface_triangles_dev_;
    CudaArray<math::Vec2i> surface_edges_dev_;
    CudaArray<math::Vec3f> interpolated_surface_positions_dev_;
    CudaArray<float> interpolated_surface_min_y_dev_;
    CudaArray<collision::WideContact> interpolated_wide_contacts_;
    CudaArray<int> interpolated_wide_contact_count_;
    CudaArray<float> tet_mass_blocks_dev_;
    CudaArray<int> interpolated_debug_counts_;
    CudaArray<math::Vec3f> interpolated_normal_sum_;

    std::vector<float> flat_positions_;
    std::vector<int> flat_triangles_;

    int last_pcg_iterations_ = 0;
    int last_ground_contacts_ = 0;
    int last_self_contacts_ = 0;
    int last_self_point_face_contacts_ = 0;
    int last_self_edge_edge_contacts_ = 0;
    int last_self_vertical_contacts_ = 0;
    int last_self_horizontal_contacts_ = 0;
    int last_self_broadphase_pairs_ = 0;
    int last_self_broadphase_capacity_ = 0;
    bool last_self_broadphase_overflow_ = false;
    int last_self_raw_contacts_ = 0;
    bool last_self_contact_overflow_ = false;
    math::Vec3f last_self_normal_sum_ = math::Vec3f(0.0f, 0.0f, 0.0f);
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
PabdCudaMesh make_pd_abd_boxes_mesh(int count,
                                    float ground_y = 0.0f,
                                    float contact_gap = 0.035f,
                                    float half_extent = 0.42f,
                                    float initial_lift = 0.0f);
PabdCudaMesh make_pd_abd_tetra_edge_edge_mesh(float lower_y = 1.10f,
                                              float upper_y = 1.55f,
                                              float scale = 0.36f);

constexpr int kTwentyTetsCount = 20;
constexpr int kHexBlockSurfaceTris = 12;
constexpr int kDefaultStackCount = 10;
constexpr int kPdAbdDefaultBoxCount = 2;
constexpr int kPdAbdBoxSurfaceVertices = 8;
constexpr int kPdAbdBoxSurfaceTris = 12;

}  // namespace pabd_cuda
}  // namespace rigid
}  // namespace chysx
