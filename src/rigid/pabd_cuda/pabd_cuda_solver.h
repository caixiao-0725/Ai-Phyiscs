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
#include "pabd_elastic_curvature.cuh"

namespace chysx {
namespace rigid {
namespace pabd_cuda {

enum class PabdGlobalSolverMode {
    PCG = 0,
    BlockJacobi12 = 1,
};

enum class PabdSelfContactMeasureMode {
    EffectiveMass = 0,
    TetrahedronVolume = 1,
};

struct PabdCudaParams {
    float dt = 0.0033f;
    int substeps = 1;
    int iterations = 1;       // global PD solve passes per substep
    float gravity = -9.8f;
    float stiffness = 1200.0f;
    bool use_arap_beta = false;
    // Dimensionless mean deformation stiffness relative to M / h^2.
    float arap_beta = 1.0f;
    PabdElasticCurvatureMode elastic_curvature =
        PabdElasticCurvatureMode::ProjectiveDynamics;
    PabdPolarGnBackend polar_gn_backend = PabdPolarGnBackend::Assembled12;
    bool elastic_curvature_diagnostics = false;
    float density = 1.0f;
    float damping = 1.0f;
    float fixed_weight = 1.0e7f;
    // Dimensionless endpoint contraction strength. The hinge energy uses the
    // endpoint effective mass, so beta has the same meaning across body mass
    // and control parameterizations.
    float hinge_beta = 0.0f;
    float motor_torque = 0.0f;
    float motor_damping = 0.0f;
    float ground_y = 0.0f;
    float contact_gap = 0.035f;
    bool use_contact_beta = false;
    // Dimensionless contact strengths. Ground contacts use their nodal
    // effective mass. PF/EE contacts can use either their effective mass or
    // rho*V_hat, where V_hat is the virtual four-point contact tetrahedron
    // volume at the activation distance.
    float ground_contact_beta = 0.0f;
    float ground_stiffness = 0.0f;
    float ground_friction = 0.0f;
    float self_collision_thickness = 0.0f;
    float self_collision_beta = 0.0f;
    PabdSelfContactMeasureMode self_contact_measure =
        PabdSelfContactMeasureMode::EffectiveMass;
    // Multiplicative PF/VF stiffness relative to EE after the base contact
    // measure is evaluated. EE always has scale 1.
    float point_face_stiffness_scale = 1.0f;
    // MeshMesh research filter. Production scenes keep both enabled; isolated
    // fixtures can disable one feature without changing narrow-phase output.
    bool enable_point_face_contacts = true;
    bool enable_edge_edge_contacts = true;
    float self_collision_stiffness = 0.0f;
    float self_collision_normal_damping = 0.0f;
    float self_collision_friction = 0.0f;
    float friction_epsilon = 1.0e-3f;
    int self_collision_max_contacts = 256;
    int mesh_broadphase_interval = 1;
    float mesh_broadphase_skin = 0.0f;
    int pcg_iterations = 40;
    bool pcg_body_preconditioner = true;
    bool pcg_true_residual_diagnostics = false;
    PabdGlobalSolverMode global_solver = PabdGlobalSolverMode::PCG;
    float block_jacobi_omega = 1.0f;
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

struct alignas(16) PabdEndpointHinge {
    int body = -1;
    int motor = 0;
    int pad0 = 0;
    int pad1 = 0;
    math::Vec4f weights0 = math::Vec4f(0.0f);
    math::Vec4f weights1 = math::Vec4f(0.0f);
    math::Vec4f endpoint0 = math::Vec4f(0.0f);
    math::Vec4f endpoint1 = math::Vec4f(0.0f);
    math::Vec4f axis = math::Vec4f(0.0f);
    // Packed symmetric endpoint effective mass: (m00, m01, m11, unused).
    math::Vec4f effective_mass = math::Vec4f(1.0f, 0.0f, 1.0f, 0.0f);
};

static_assert(sizeof(PabdEndpointHinge) == 112,
              "PabdEndpointHinge must keep its 112-byte GPU layout");

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
    std::vector<PabdEndpointHinge> endpoint_hinges;
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
    float last_true_relative_residual() const noexcept {
        return last_true_relative_residual_;
    }
    int last_ground_contacts() const noexcept { return last_ground_contacts_; }
    int last_self_contacts() const noexcept { return last_self_contacts_; }
    int last_self_point_face_contacts() const noexcept { return last_self_point_face_contacts_; }
    int last_self_edge_edge_contacts() const noexcept { return last_self_edge_edge_contacts_; }
    int last_self_active_point_face_contacts() const noexcept {
        return last_self_active_point_face_contacts_;
    }
    int last_self_active_edge_edge_contacts() const noexcept {
        return last_self_active_edge_edge_contacts_;
    }
    float last_self_max_penetration() const noexcept {
        return last_self_max_penetration_;
    }
    float last_self_rms_penetration() const noexcept {
        return last_self_rms_penetration_;
    }
    float last_self_mean_stiffness() const noexcept {
        return last_self_mean_stiffness_;
    }
    float last_self_point_face_mean_stiffness() const noexcept {
        return last_self_point_face_mean_stiffness_;
    }
    float last_self_edge_edge_mean_stiffness() const noexcept {
        return last_self_edge_edge_mean_stiffness_;
    }
    float last_self_point_face_mean_volume() const noexcept {
        return last_self_point_face_mean_volume_;
    }
    float last_self_edge_edge_mean_volume() const noexcept {
        return last_self_edge_edge_mean_volume_;
    }
    int last_self_friction_contacts() const noexcept {
        return last_self_friction_contacts_;
    }
    int last_self_vertical_contacts() const noexcept { return last_self_vertical_contacts_; }
    int last_self_horizontal_contacts() const noexcept { return last_self_horizontal_contacts_; }
    int last_self_broadphase_pairs() const noexcept { return last_self_broadphase_pairs_; }
    int last_self_broadphase_capacity() const noexcept {
        return last_self_broadphase_capacity_;
    }
    bool last_self_broadphase_overflow() const noexcept {
        return last_self_broadphase_overflow_;
    }
    bool last_mesh_broadphase_refreshed() const noexcept {
        return last_mesh_broadphase_refreshed_;
    }
    int mesh_broadphase_cache_age() const noexcept {
        return mesh_broadphase_cache_age_;
    }
    int mesh_broadphase_refresh_count() const noexcept {
        return mesh_broadphase_refresh_count_;
    }
    float last_mesh_broadphase_max_displacement() const noexcept {
        return last_mesh_broadphase_max_displacement_;
    }
    int last_mesh_broadphase_dropped_hits() const noexcept {
        return last_mesh_broadphase_dropped_hits_;
    }
    int last_self_raw_contacts() const noexcept { return last_self_raw_contacts_; }
    int interpolated_contact_capacity() const noexcept {
        return interpolated_contact_capacity_;
    }
    bool last_self_contact_overflow() const noexcept {
        return last_self_contact_overflow_;
    }
    math::Vec3f last_self_normal_sum() const noexcept { return last_self_normal_sum_; }
    int last_block_jacobi_failures() const noexcept {
        return last_block_jacobi_failures_;
    }
    int block_jacobi_contact_ell_width() const noexcept {
        return block_jacobi_contact_ell_width_;
    }
    int last_block_jacobi_contact_ell_overflow() const noexcept {
        return last_block_jacobi_contact_ell_overflow_;
    }
    int last_pcg_body_preconditioner_failures() const noexcept {
        return last_pcg_body_preconditioner_failures_;
    }
    bool pcg_body_preconditioner_active() const noexcept {
        return params_.pcg_body_preconditioner &&
               pcg_body_preconditioner_valid_;
    }
    bool matrix_free_polar_gn_active() const noexcept;
    float last_motor_axis_angular_velocity() const noexcept {
        return last_motor_axis_angular_velocity_;
    }
    float last_hinge_endpoint_error() const noexcept {
        return last_hinge_endpoint_error_;
    }
    const std::vector<float>& hinge_axis_angular_velocities() const noexcept {
        return hinge_axis_angular_velocities_;
    }
    const std::vector<float>& hinge_endpoint_errors() const noexcept {
        return hinge_endpoint_errors_;
    }
    float last_max_stretch_error() const noexcept {
        return last_max_stretch_error_;
    }
    float last_max_orthogonality_error() const noexcept {
        return last_max_orthogonality_error_;
    }
    float last_max_volume_error() const noexcept {
        return last_max_volume_error_;
    }
    float last_max_rotational_curvature() const noexcept {
        return last_max_rotational_curvature_;
    }
    float min_y() const noexcept { return min_y_; }

private:
    void release_stream() noexcept;
    void build_host_data(const PabdCudaMesh& mesh);
    void upload_static_data();
    void update_host_positions(bool x_already_on_host = false);
    void step_interpolated(float dt);
    void assemble_interpolated_system_gpu(float h);
    void update_block_jacobi_base_k_gpu(float h,
                                        bool local_rest_frame = false);
    void build_body_contact_ell_gpu();
    void build_interpolated_pcg_body_preconditioner_gpu(float h);
    void solve_interpolated_block_jacobi_gpu(float h, float omega);
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
    std::vector<PabdEndpointHinge> host_endpoint_hinges_;
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
    CudaArray<math::Vec3f> delta_;
    CudaArray<float> mass_;
    CudaArray<unsigned char> fixed_;
    CudaArray<TetData> tets_;
    CudaArray<float> tet_arap_effective_inertias_;
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
    CudaArray<float> tet_contact_inverse_mass_blocks_;
    CudaArray<float> particle_contact_stiffnesses_;
    CudaArray<PabdEndpointHinge> endpoint_hinges_dev_;
    CudaArray<math::Vec3f> endpoint_hinge_diagnostics_dev_;
    CudaArray<float> block_jacobi_base_k_;
    CudaArray<math::Vec3f> block_jacobi_x_lagged_;
    CudaArray<int> block_jacobi_failure_count_;
    CudaArray<int> block_jacobi_contact_ell_counts_;
    CudaArray<std::uint32_t> block_jacobi_contact_ell_refs_;
    CudaArray<int> block_jacobi_contact_ell_overflow_;
    CudaArray<float> pcg_body_preconditioner_lower_;
    CudaArray<math::Vec4i> pcg_body_preconditioner_rows_;
    CudaArray<int> pcg_body_preconditioner_failure_count_;
    CudaArray<math::Vec3f> matrix_free_body_gradients_;
    CudaArray<math::Mat3f> matrix_free_body_rotations_;
    CudaArray<math::Mat3f> matrix_free_body_rotational_curvatures_;
    CudaArray<float> matrix_free_body_scales_;
    CudaArray<int> interpolated_debug_counts_;
    CudaArray<float> interpolated_contact_metrics_;
    CudaArray<math::Vec3f> interpolated_normal_sum_;
    CudaArray<float> elastic_curvature_diagnostics_dev_;

    std::vector<float> flat_positions_;
    std::vector<int> flat_triangles_;

    int last_pcg_iterations_ = 0;
    int last_ground_contacts_ = 0;
    int last_self_contacts_ = 0;
    int last_self_point_face_contacts_ = 0;
    int last_self_edge_edge_contacts_ = 0;
    int last_self_active_point_face_contacts_ = 0;
    int last_self_active_edge_edge_contacts_ = 0;
    float last_self_max_penetration_ = 0.0f;
    float last_self_rms_penetration_ = 0.0f;
    float last_self_mean_stiffness_ = 0.0f;
    float last_self_point_face_mean_stiffness_ = 0.0f;
    float last_self_edge_edge_mean_stiffness_ = 0.0f;
    float last_self_point_face_mean_volume_ = 0.0f;
    float last_self_edge_edge_mean_volume_ = 0.0f;
    int last_self_friction_contacts_ = 0;
    int last_self_vertical_contacts_ = 0;
    int last_self_horizontal_contacts_ = 0;
    int last_self_broadphase_pairs_ = 0;
    int last_self_broadphase_capacity_ = 0;
    bool last_self_broadphase_overflow_ = false;
    bool last_mesh_broadphase_refreshed_ = true;
    int mesh_broadphase_cache_age_ = 0;
    int mesh_broadphase_refresh_count_ = 0;
    float last_mesh_broadphase_max_displacement_ = 0.0f;
    int last_mesh_broadphase_dropped_hits_ = 0;
    int last_self_raw_contacts_ = 0;
    bool last_self_contact_overflow_ = false;
    math::Vec3f last_self_normal_sum_ = math::Vec3f(0.0f, 0.0f, 0.0f);
    int last_block_jacobi_failures_ = 0;
    int block_jacobi_contact_ell_width_ = 0;
    int last_block_jacobi_contact_ell_overflow_ = 0;
    int last_pcg_body_preconditioner_failures_ = 0;
    bool pcg_body_preconditioner_valid_ = false;
    bool matrix_free_body_elastic_valid_ = false;
    float block_jacobi_base_inv_h2_ = -1.0f;
    float block_jacobi_base_stiffness_ = -1.0f;
    float block_jacobi_base_arap_beta_ = -1.0f;
    bool block_jacobi_base_use_arap_beta_ = false;
    float block_jacobi_base_fixed_weight_ = -1.0f;
    float block_jacobi_base_hinge_beta_ = -1.0f;
    PabdElasticCurvatureMode block_jacobi_base_curvature_ =
        PabdElasticCurvatureMode::ProjectiveDynamics;
    bool block_jacobi_base_local_rest_frame_ = false;
    bool block_jacobi_base_k_valid_ = false;
    float last_residual_ = 0.0f;
    float last_true_relative_residual_ = -1.0f;
    float last_motor_axis_angular_velocity_ = 0.0f;
    float last_hinge_endpoint_error_ = 0.0f;
    float last_max_stretch_error_ = 0.0f;
    float last_max_orthogonality_error_ = 0.0f;
    float last_max_volume_error_ = 0.0f;
    float last_max_rotational_curvature_ = 0.0f;
    std::vector<float> hinge_axis_angular_velocities_;
    std::vector<float> hinge_endpoint_errors_;
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
constexpr int kDefaultStackCount = 10;
constexpr int kPdAbdDefaultBoxCount = 2;
constexpr int kPdAbdBoxSurfaceVertices = 8;
constexpr int kPdAbdBoxSurfaceTris = 12;

}  // namespace pabd_cuda
}  // namespace rigid
}  // namespace chysx
