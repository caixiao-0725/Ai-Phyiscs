// SPDX-License-Identifier: MIT
// Rigid-IPC solver: GPU-accelerated Newton iteration with BDF1 time
// integration for 6-DOF rigid bodies with IPC contact barriers.
//
// Data layout:
//   - Vertex positions, rest-frame positions (x_bar), body IDs live on GPU
//   - Body states (q, q_prev, q_tilde) are tiny (2 bodies = 12 floats) → CPU
//     with small H2D uploads per Newton step
//   - Contacts detected via GPU broadphase, uploaded to GPU for assembly
//   - Contact gradient/Hessian accumulated on GPU via atomicAdd
//   - PCG runs on CPU (only 2 bodies = 12 DOF, not worth GPU overhead)
//   - CCD and barrier energy evaluated on GPU per contact, reduced on GPU

#pragma once

#include "rigid_ipc_types.cuh"
#include "rigid_ipc_rodrigues.cuh"
#include "rigid_ipc_mass.cuh"
#include "rigid_ipc_energy.cuh"
#include "rigid_ipc_assembly.cuh"
#include "rigid_ipc_pcg.cuh"
#include "rigid_ipc_contact.h"
#include "rigid_ipc_kernels.cuh"
#include "../../rigid/abd_ipc/abd_ipc_ccd.cuh"
#include "../../geometry/triangle_mesh.h"
#include "../../memory/cuda_array.h"
#include "../../io/obj_io.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace chysx {
namespace rigid_ipc {

using abd_ipc::ccd_pt;
using abd_ipc::ccd_ee;

// PSD projection for 12×12 symmetric matrix via Jacobi eigenvalue decomposition.
// Clamps negative eigenvalues to 0: A → V·max(Λ,0)·V^T.
inline void project_to_psd_12(Mat12f& A) {
    constexpr int N = 12;
    double M[N][N], V[N][N];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            M[i][j] = 0.5 * ((double)A(i, j) + (double)A(j, i));
            V[i][j] = (i == j) ? 1.0 : 0.0;
        }

    // Check if already PSD (all diagonals non-negative and dominant)
    double min_diag = M[0][0];
    for (int i = 1; i < N; ++i)
        if (M[i][i] < min_diag) min_diag = M[i][i];

    // Jacobi rotation sweeps
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0;
        for (int i = 0; i < N; ++i)
            for (int j = i + 1; j < N; ++j)
                off += M[i][j] * M[i][j];
        if (off < 1e-20) break;

        for (int p = 0; p < N; ++p) {
            for (int q = p + 1; q < N; ++q) {
                if (std::abs(M[p][q]) < 1e-15) continue;
                double tau = (M[q][q] - M[p][p]) / (2.0 * M[p][q]);
                double t = (tau >= 0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;

                double app = M[p][p], aqq = M[q][q], apq = M[p][q];
                M[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
                M[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
                M[p][q] = M[q][p] = 0;

                for (int r = 0; r < N; ++r) {
                    if (r == p || r == q) continue;
                    double mrp = M[r][p], mrq = M[r][q];
                    M[r][p] = M[p][r] = c * mrp - s * mrq;
                    M[r][q] = M[q][r] = s * mrp + c * mrq;
                }
                for (int r = 0; r < N; ++r) {
                    double vrp = V[r][p], vrq = V[r][q];
                    V[r][p] = c * vrp - s * vrq;
                    V[r][q] = s * vrp + c * vrq;
                }
            }
        }
    }

    // Clamp negative eigenvalues to 0
    bool any_neg = false;
    for (int i = 0; i < N; ++i)
        if (M[i][i] < 0) { any_neg = true; break; }
    if (!any_neg) return;

    // Reconstruct: A = V · diag(max(eigenval, 0)) · V^T
    double D[N];
    for (int i = 0; i < N; ++i) D[i] = std::max(0.0, M[i][i]);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double val = 0;
            for (int k = 0; k < N; ++k)
                val += V[i][k] * D[k] * V[j][k];
            A(i, j) = (float)val;
        }
}

// Finite-difference distance² Hessian for PE/PT/EE contacts.
// Computes ∇²D w.r.t. the nv vertex positions (nv*3 × nv*3 matrix).
// Uses central differences on the analytic gradient.
inline void dist2_hessian_fd(int type, const Vec3f verts[], int nv,
                              float* H_out) {
    const int dim = nv * 3;
    const float eps = 1e-5f;

    // Flatten vertices
    float x[12]; // max 4 verts * 3
    for (int i = 0; i < nv; ++i)
        for (int j = 0; j < 3; ++j)
            x[i * 3 + j] = verts[i].data[j];

    // For each DOF, perturb and compute gradient
    for (int col = 0; col < dim; ++col) {
        float xp[12], xm[12];
        for (int i = 0; i < dim; ++i) xp[i] = xm[i] = x[i];
        xp[col] += eps;
        xm[col] -= eps;

        Vec3f vp[4], vm[4];
        for (int i = 0; i < nv; ++i) {
            vp[i] = Vec3f{xp[i*3+0], xp[i*3+1], xp[i*3+2]};
            vm[i] = Vec3f{xm[i*3+0], xm[i*3+1], xm[i*3+2]};
        }

        Vec3f gp[4] = {}, gm[4] = {};
        if (type == 1) { // PE
            abd_ipc::dist2_pe_grad(vp[0], vp[1], vp[2], gp[0], gp[1], gp[2]);
            abd_ipc::dist2_pe_grad(vm[0], vm[1], vm[2], gm[0], gm[1], gm[2]);
        } else if (type == 2) { // PT
            abd_ipc::dist2_pt_grad(vp[0], vp[1], vp[2], vp[3], gp[0], gp[1], gp[2], gp[3]);
            abd_ipc::dist2_pt_grad(vm[0], vm[1], vm[2], vm[3], gm[0], gm[1], gm[2], gm[3]);
        } else { // EE
            abd_ipc::dist2_ee_grad(vp[0], vp[1], vp[2], vp[3], gp[0], gp[1], gp[2], gp[3]);
            abd_ipc::dist2_ee_grad(vm[0], vm[1], vm[2], vm[3], gm[0], gm[1], gm[2], gm[3]);
        }

        float gp_flat[12], gm_flat[12];
        for (int i = 0; i < nv; ++i)
            for (int j = 0; j < 3; ++j) {
                gp_flat[i*3+j] = gp[i].data[j];
                gm_flat[i*3+j] = gm[i].data[j];
            }

        float inv_2eps = 1.0f / (2.0f * eps);
        for (int row = 0; row < dim; ++row)
            H_out[row * dim + col] = (gp_flat[row] - gm_flat[row]) * inv_2eps;
    }

    // Symmetrize
    for (int i = 0; i < dim; ++i)
        for (int j = i + 1; j < dim; ++j) {
            float avg = 0.5f * (H_out[i*dim+j] + H_out[j*dim+i]);
            H_out[i*dim+j] = H_out[j*dim+i] = avg;
        }
}

class RigidIPCSolver {
public:
    RigidIPCConfig config;

    // ---- bodies and meshes ------------------------------------------------

    void add_body(geometry::TriangleMeshf& mesh, float density,
                  Vec3f position = Vec3f(0, 0, 0),
                  Vec3f rotation_deg = Vec3f(0, 0, 0),
                  float scale = 1.0f,
                  bool fixed = false,
                  const bool* dof_fixed_6 = nullptr) {

        if (fabsf(scale - 1.0f) > 1e-6f) {
            int nv = static_cast<int>(mesh.num_vertices());
            Vec3f* verts = mesh.vertices().cpu_data();
            for (int i = 0; i < nv; ++i) {
                verts[i].x *= scale;
                verts[i].y *= scale;
                verts[i].z *= scale;
            }
        }

        float rx = rotation_deg.x * 3.14159265358979f / 180.0f;
        float ry = rotation_deg.y * 3.14159265358979f / 180.0f;
        float rz = rotation_deg.z * 3.14159265358979f / 180.0f;
        if (fabsf(rx) + fabsf(ry) + fabsf(rz) > 1e-8f) {
            Mat3f Rx = rodrigues(Vec3f(rx, 0, 0));
            Mat3f Ry = rodrigues(Vec3f(0, ry, 0));
            Mat3f Rz = rodrigues(Vec3f(0, 0, rz));
            Mat3f R_init = Rz * Ry * Rx;
            int nv = static_cast<int>(mesh.num_vertices());
            Vec3f* verts = mesh.vertices().cpu_data();
            for (int i = 0; i < nv; ++i)
                verts[i] = R_init * verts[i];
        }

        mesh.build_edges();

        MassProperties props;
        if (!compute_and_align_mass(mesh, density, props)) {
            printf("[RigidIPCSolver] ERROR: failed to compute mass for body %d\n",
                   static_cast<int>(bodies_.size()));
            return;
        }

        RigidBody body{};
        body.mass = props.mass;
        body.moment_of_inertia = props.principal_inertia;
        body.is_fixed = fixed;
        body.is_dynamic = !fixed;
        body.friction = config.default_friction;

        if (fixed) {
            for (int d = 0; d < 6; ++d) body.dof_fixed[d] = true;
        } else if (dof_fixed_6) {
            for (int d = 0; d < 6; ++d) body.dof_fixed[d] = dof_fixed_6[d];
            bool all_fixed = true;
            for (int d = 0; d < 6; ++d) if (!body.dof_fixed[d]) all_fixed = false;
            if (all_fixed) { body.is_fixed = true; body.is_dynamic = false; }
        }

        // Initial pose: principal alignment R0 compensated in theta
        // x_world = R(theta) * x_bar + p, where x_bar is in principal frame
        // To recover original world-space positions: R(theta_init) = R0
        Vec3f theta_init = rotation_matrix_to_axis_angle(props.principal_axes);
        Vec3f world_com = position + props.center;
        body.q = make_pose(world_com, theta_init);
        body.q_prev = body.q;
        body.q_tilde = body.q;
        body.q_v = Vec6f::zero();
        body.R = rodrigues(theta_init);
        body.R_prev = body.R;
        body.R0 = body.R;
        body.Qdot_prev = Mat3f::zero();

        printf("[RigidIPCSolver] Body %d: mass=%.4f, I=(%.6e, %.6e, %.6e), "
               "com=(%.8f, %.8f, %.8f), theta_init=(%.8f, %.8f, %.8f), %s\n",
               static_cast<int>(bodies_.size()), body.mass,
               body.moment_of_inertia.x, body.moment_of_inertia.y,
               body.moment_of_inertia.z,
               world_com.x, world_com.y, world_com.z,
               theta_init.x, theta_init.y, theta_init.z,
               fixed ? "STATIC" : "DYNAMIC");

        body.surface_vert_offset = static_cast<int>(h_verts_.size());
        int nv = static_cast<int>(mesh.num_vertices());
        int nt = static_cast<int>(mesh.num_triangles());
        int ne = static_cast<int>(mesh.num_edges());

        body.surface_vert_count = nv;
        body.surface_tri_offset = static_cast<int>(h_tris_.size());
        body.surface_tri_count = nt;
        body.surface_edge_offset = static_cast<int>(h_edges_.size());
        body.surface_edge_count = ne;

        int body_idx = static_cast<int>(bodies_.size());

        const Vec3f* verts = mesh.vertices().cpu_data();
        for (int i = 0; i < nv; ++i) {
            h_verts_.push_back(verts[i]);
            h_x_bar_.push_back(verts[i]);
            h_vert_body_.push_back(body_idx);
        }

        const auto* tris = mesh.triangles().cpu_data();
        int v_off = body.surface_vert_offset;
        for (int i = 0; i < nt; ++i)
            h_tris_.push_back(Vec3i(tris[i].x + v_off,
                                    tris[i].y + v_off,
                                    tris[i].z + v_off));

        const auto* edges = mesh.edges().cpu_data();
        for (int i = 0; i < ne; ++i)
            h_edges_.push_back(math::Vec2i(edges[i].x + v_off,
                                            edges[i].y + v_off));

        SurfaceData::BodyRange range;
        range.vert_begin = body.surface_vert_offset;
        range.vert_end   = body.surface_vert_offset + nv;
        range.tri_begin  = body.surface_tri_offset;
        range.tri_end    = body.surface_tri_offset + nt;
        range.edge_begin = body.surface_edge_offset;
        range.edge_end   = body.surface_edge_offset + ne;
        surface_.body_ranges.push_back(range);

        bodies_.push_back(body);
    }

    bool add_body_from_obj(const std::string& path, float density,
                           Vec3f position = Vec3f(0, 0, 0),
                           Vec3f rotation_deg = Vec3f(0, 0, 0),
                           float scale = 1.0f,
                           bool fixed = false,
                           const bool* dof_fixed_6 = nullptr) {
        io::ObjMesh obj;
        if (!io::load_obj(path, obj)) {
            printf("[RigidIPCSolver] ERROR: failed to load OBJ %s\n", path.c_str());
            return false;
        }
        int nv = static_cast<int>(obj.positions.size()) / 3;
        int nt = static_cast<int>(obj.triangles.size()) / 3;

        geometry::TriangleMeshf mesh(nv, nt);
        auto* verts = mesh.vertices().cpu_data();
        for (int i = 0; i < nv; ++i)
            verts[i] = Vec3f(obj.positions[i*3], obj.positions[i*3+1], obj.positions[i*3+2]);
        auto* tris = mesh.triangles().cpu_data();
        for (int i = 0; i < nt; ++i)
            tris[i] = Vec3i(obj.triangles[i*3], obj.triangles[i*3+1], obj.triangles[i*3+2]);

        add_body(mesh, density, position, rotation_deg, scale, fixed, dof_fixed_6);
        return true;
    }

    // Upload all host-side geometry to GPU. Call once after all add_body().
    void finalize() {
        int nv = static_cast<int>(h_verts_.size());
        int nt = static_cast<int>(h_tris_.size());
        int ne = static_cast<int>(h_edges_.size());
        int nb = static_cast<int>(bodies_.size());

        // Persistent GPU arrays (never re-uploaded unless bodies change)
        d_x_bar_.resize(nv);
        memcpy(d_x_bar_.cpu_data(), h_x_bar_.data(), nv * sizeof(Vec3f));
        d_x_bar_.copy_to_device();

        d_vert_body_.resize(nv);
        memcpy(d_vert_body_.cpu_data(), h_vert_body_.data(), nv * sizeof(int));
        d_vert_body_.copy_to_device();

        // Working GPU arrays
        d_verts_.resize(nv);
        d_verts_trial_.resize(nv);
        d_body_q_.resize(nb);
        d_body_dq_.resize(nb);

        // Contact assembly output (on GPU)
        d_grad_.resize(nb);
        d_hess_diag_.resize(nb);

        // Build SurfaceData for broadphase (needs CPU-side tris/verts/edges)
        surface_.verts = h_verts_;
        surface_.tris = h_tris_;
        surface_.edges = h_edges_;
        surface_.vert_body = h_vert_body_;

        n_verts_ = nv;
        n_bodies_ = nb;

        // Setup broadphase eagerly so topology is available for GPU narrow phase
        {
            collision::BroadphaseBackend backend = collision::BroadphaseBackend::QuantBvh;
            if (config.broadphase_type == RigidIPCConfig::BroadphaseType::OptiX)
                backend = collision::BroadphaseBackend::OptiX;
            ef_detector_.setup(h_tris_, nv, -1, backend);
            bvh_initialized_ = true;
        }

        // Pre-allocate contact output buffer (worst case ~4x edges)
        max_contacts_ = std::max(ne * 4, 8192);
        d_contacts_.resize(max_contacts_);
        d_contact_buf_.resize(max_contacts_);
        d_contact_count_.resize(1);
        d_hess_offdiag_.resize(max_contacts_ * 2);
        d_offdiag_pairs_.resize(max_contacts_);

        // GPU PCG scratch
        d_pcg_rhs_.resize(nb);
        d_pcg_dx_.resize(nb);
        d_pcg_r_.resize(nb);
        d_pcg_z_.resize(nb);
        d_pcg_p_.resize(nb);
        d_pcg_Hp_.resize(nb);

        // DOF mask: 0=locked, 1=free
        d_dof_mask_.resize(nb);
        for (int i = 0; i < nb; ++i) {
            Vec6f mask = Vec6f::zero();
            for (int d = 0; d < 6; ++d)
                mask[d] = bodies_[i].dof_fixed[d] ? 0.0f : 1.0f;
            d_dof_mask_.cpu_data()[i] = mask;
        }
        d_dof_mask_.copy_to_device();

        // Compute world-space bbox and average mass for adaptive kappa
        compute_bbox_and_mass();
        if (config.contact_kappa <= 0) {
            kappa_ = compute_initial_kappa();
        } else {
            kappa_ = config.contact_kappa;
            max_kappa_ = kappa_ * 100.0f;
        }
        kappa_initialized_ = true;

        finalized_ = true;

        printf("[RigidIPCSolver] Finalized: %d verts, %d tris, %d edges on GPU (max_contacts=%d)\n",
               nv, nt, ne, max_contacts_);

        // Upload initial verts and run initial contact check
        {
            upload_body_q();
            gpu_update_surface_verts();
            cudaDeviceSynchronize();
            int nc_init = detect_all_contacts_gpu();
            printf("[RigidIPCSolver] init contact count = %d\n", nc_init);
        }

        // Diagnostic: compute initial world-space bboxes per body
        for (int b = 0; b < nb; ++b) {
            Vec3f p = pose_position(bodies_[b].q);
            Vec3f theta = pose_rotation(bodies_[b].q);
            Mat3f R = rodrigues(theta);
            int off = bodies_[b].surface_vert_offset;
            int cnt = bodies_[b].surface_vert_count;
            Vec3f lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
            for (int vi = 0; vi < cnt; ++vi) {
                Vec3f w = R * h_x_bar_[off + vi] + p;
                lo.x = std::min(lo.x, w.x); lo.y = std::min(lo.y, w.y); lo.z = std::min(lo.z, w.z);
                hi.x = std::max(hi.x, w.x); hi.y = std::max(hi.y, w.y); hi.z = std::max(hi.z, w.z);
            }
            printf("[RigidIPCSolver] Body %d world bbox: (%.6f,%.6f,%.6f) - (%.6f,%.6f,%.6f)\n",
                   b, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
        }
    }

    // ---- simulation -------------------------------------------------------

    void step() {
        if (!finalized_) finalize();

        int n = n_bodies_;
        float dt = config.dt;
        float inv_dt = 1.0f / dt;

        for (int i = 0; i < n; ++i) {
            bodies_[i].q_prev = bodies_[i].q;
            bodies_[i].R_prev = bodies_[i].R;
        }

        for (int i = 0; i < n; ++i) {
            if (bodies_[i].is_fixed) {
                bodies_[i].q_tilde = bodies_[i].q;
                continue;
            }
            Vec3f p = pose_position(bodies_[i].q);
            Vec3f theta = pose_rotation(bodies_[i].q);
            Vec3f v_lin = pose_position(bodies_[i].q_v);
            Vec3f omega = pose_rotation(bodies_[i].q_v);

            Vec3f p_pred = p + v_lin * dt + config.gravity * (dt * dt);
            Vec3f theta_pred = theta + omega * dt;
            bodies_[i].q_tilde = make_pose(p_pred, theta_pred);

            // Clamp locked DOFs to their current value
            for (int d = 0; d < 6; ++d) {
                if (bodies_[i].dof_fixed[d])
                    bodies_[i].q_tilde[d] = bodies_[i].q[d];
            }

            Mat3f R_cur = rodrigues(theta);
            bodies_[i].R = R_cur;
        }

        // Upload body q to GPU and update surface verts
        upload_body_q();
        gpu_update_surface_verts();

        // Newton iteration
        newton_solve();

        // Velocity update — match reference: omega = AngleAxis(R_new * R_prev^T)
        for (int i = 0; i < n; ++i) {
            if (bodies_[i].is_fixed) continue;
            Vec3f p_new = pose_position(bodies_[i].q);
            Vec3f p_old = pose_position(bodies_[i].q_prev);
            Vec3f theta_new = pose_rotation(bodies_[i].q);

            Vec3f v_lin = (p_new - p_old) * inv_dt;
            Mat3f R_new = rodrigues(theta_new);
            bodies_[i].R = R_new;

            Mat3f Qdot;
            for (int j = 0; j < 9; ++j)
                Qdot.data[j] = (R_new.data[j] - bodies_[i].R_prev.data[j]) * inv_dt;
            bodies_[i].Qdot_prev = Qdot;

            // Reference: R = R_t1 * R_t0^T → AngleAxis → omega_body = R0^T * omega
            Mat3f dR = R_new * transpose(bodies_[i].R_prev);
            Vec3f omega_aa = rotation_matrix_to_axis_angle(dR);
            Vec3f omega_world = omega_aa * inv_dt;
            Vec3f omega_body = transpose(bodies_[i].R0) * omega_world;

            bodies_[i].q_v = make_pose(v_lin, omega_body);

            for (int d = 0; d < 6; ++d)
                if (bodies_[i].dof_fixed[d]) bodies_[i].q_v[d] = 0.0f;
        }

        // Download verts for OBJ export / display
        download_verts();
    }

    // ---- accessors --------------------------------------------------------

    int num_bodies() const { return n_bodies_; }
    const RigidBody& body(int i) const { return bodies_[i]; }

    std::vector<Vec3f> body_surface_verts(int body_idx) const {
        auto& r = surface_.body_ranges[body_idx];
        return {h_verts_.begin() + r.vert_begin,
                h_verts_.begin() + r.vert_end};
    }

    std::vector<Vec3i> body_surface_tris(int body_idx) const {
        auto& r = surface_.body_ranges[body_idx];
        int offset = r.vert_begin;
        std::vector<Vec3i> tris;
        for (int i = r.tri_begin; i < r.tri_end; ++i) {
            Vec3i t = h_tris_[i];
            tris.push_back(Vec3i(t.x - offset, t.y - offset, t.z - offset));
        }
        return tris;
    }

    struct StepInfo {
        int newton_iters;
        int contact_count;
        float residual;
        float min_distance;
    };
    StepInfo last_step_info() const { return last_info_; }

private:
    // ---- Host-side data ---------------------------------------------------
    std::vector<RigidBody> bodies_;
    std::vector<Vec3f> h_verts_;        // current world verts (CPU mirror)
    std::vector<Vec3f> h_x_bar_;        // rest-frame verts
    std::vector<int> h_vert_body_;
    std::vector<Vec3i> h_tris_;
    std::vector<math::Vec2i> h_edges_;
    SurfaceData surface_;               // for broadphase (CPU narrow phase)
    StepInfo last_info_{};
    collision::EFContactDetector ef_detector_;
    bool bvh_initialized_ = false;
    bool finalized_ = false;
    int n_verts_ = 0;
    int n_bodies_ = 0;

    // Adaptive kappa state
    float kappa_ = 0.0f;
    float min_kappa_ = 0.0f;
    float max_kappa_ = 0.0f;
    float bbox_diagonal_ = 0.0f;
    float average_mass_ = 0.0f;
    bool kappa_initialized_ = false;

    // ---- GPU arrays -------------------------------------------------------
    CudaArray<Vec3f> d_x_bar_;          // rest-frame verts [n_verts]
    CudaArray<int> d_vert_body_;        // body index per vert [n_verts]
    CudaArray<Vec3f> d_verts_;          // current world verts [n_verts]
    CudaArray<Vec3f> d_verts_trial_;    // trial verts for line search [n_verts]
    CudaArray<Vec6f> d_body_q_;         // body poses [n_bodies]
    CudaArray<Vec6f> d_body_dq_;        // Newton step [n_bodies]
    CudaArray<Vec6f> d_grad_;           // gradient accumulator [n_bodies]
    CudaArray<Mat6f> d_hess_diag_;      // Hessian diagonal accumulator [n_bodies]

    // Per-contact GPU buffers
    CudaArray<GPUContactPair> d_contacts_;
    CudaArray<float> d_contact_buf_;    // scratch for energies / alphas
    CudaArray<int> d_contact_count_;    // atomic counter [1]
    CudaArray<Mat6f> d_hess_offdiag_;   // [max_contacts * 2] off-diagonal blocks
    CudaArray<math::Vec2i> d_offdiag_pairs_; // [max_contacts] body pairs
    int max_contacts_ = 0;             // capacity of d_contacts_

    // GPU PCG scratch
    CudaArray<Vec6f> d_pcg_rhs_;
    CudaArray<Vec6f> d_pcg_dx_;
    CudaArray<Vec6f> d_pcg_r_, d_pcg_z_, d_pcg_p_, d_pcg_Hp_;
    CudaArray<Vec6f> d_dof_mask_;  // 0=locked, 1=free per DOF per body

    // ---- GPU helper methods -----------------------------------------------

    void upload_body_q() {
        for (int i = 0; i < n_bodies_; ++i)
            d_body_q_.cpu_data()[i] = bodies_[i].q;
        d_body_q_.copy_to_device();
    }

    void gpu_update_surface_verts() {
        launch_update_verts(d_verts_.gpu_data(), d_x_bar_.gpu_data(),
                            d_vert_body_.gpu_data(), d_body_q_.gpu_data(),
                            n_verts_);
    }

    // Download GPU verts to CPU for display/export.
    void download_verts() {
        d_verts_.copy_to_host();
        memcpy(h_verts_.data(), d_verts_.cpu_data(), n_verts_ * sizeof(Vec3f));
        surface_.verts = h_verts_;
    }

    // ---- Contact detection (fully on GPU) -----------------------------------

    // Returns number of contacts found (already in d_contacts_ on GPU).
    int detect_all_contacts_gpu() {
        const auto& topo = ef_detector_.topology();
        auto& bp = ef_detector_.broadphase();
        float d_hat_sq = config.d_hat * config.d_hat;

        // 1. Broadphase: build AABBs + BVH refit + EF query (all on GPU)
        bp.query(d_verts_.gpu_data(), config.d_hat);

        // 2. Read EF pair count
        int n_ef = bp.ef_count();
        if (n_ef == 0 && topo.n_adj_vf() == 0 && topo.n_adj_ee_pre() == 0)
            return 0;

        // 3. Zero atomic counter
        launch_zero_int(d_contact_count_.gpu_data());

        // 4. GPU narrow phase: EF → VF + EE with distance filter
        if (n_ef > 0) {
            launch_ef_to_contacts(
                bp.ef_pairs_dev(), n_ef,
                d_verts_.gpu_data(),
                topo.faces().gpu_data(), topo.edges().gpu_data(),
                topo.vert_in_edge().gpu_data(), topo.edge_in_face().gpu_data(),
                d_vert_body_.gpu_data(),
                d_hat_sq,
                d_contacts_.gpu_data(), d_contact_count_.gpu_data(),
                max_contacts_);
        }

        // 5. Supplementary adjacent VF
        if (topo.n_adj_vf() > 0) {
            launch_adj_vf_contacts(
                topo.pre_adj_vf().gpu_data(), topo.n_adj_vf(),
                d_verts_.gpu_data(), d_vert_body_.gpu_data(),
                d_hat_sq,
                d_contacts_.gpu_data(), d_contact_count_.gpu_data(),
                max_contacts_);
        }

        // 6. Supplementary adjacent EE
        if (topo.n_adj_ee_pre() > 0) {
            launch_adj_ee_contacts(
                topo.pre_adj_ee().gpu_data(), topo.n_adj_ee_pre(),
                d_verts_.gpu_data(),
                topo.edges().gpu_data(), d_vert_body_.gpu_data(),
                d_hat_sq,
                d_contacts_.gpu_data(), d_contact_count_.gpu_data(),
                max_contacts_);
        }

        // 7. Read contact count (single int DtoH)
        int nc = read_contact_count(d_contact_count_.gpu_data());
        if (nc > max_contacts_) nc = max_contacts_;

        static int dbg_ct = 0;
        if (nc > 0 && dbg_ct < 3) {
            dbg_ct++;
            printf("  [DETECT] ef_pairs=%d adj_vf=%d adj_ee=%d -> contacts=%d\n",
                   n_ef, topo.n_adj_vf(), topo.n_adj_ee_pre(), nc);
            if (nc > 0 && nc <= 200) {
                std::vector<GPUContactPair> h_ct(nc);
                cudaMemcpy(h_ct.data(), d_contacts_.gpu_data(),
                           nc * sizeof(GPUContactPair), cudaMemcpyDeviceToHost);
                int n_pp=0, n_pe=0, n_pt=0, n_ee=0;
                for (auto& c : h_ct) {
                    if (c.type == 0) n_pp++;
                    else if (c.type == 1) n_pe++;
                    else if (c.type == 2) n_pt++;
                    else if (c.type == 3) n_ee++;
                }
                printf("  [DETECT] types: PP=%d PE=%d PT=%d EE=%d\n", n_pp, n_pe, n_pt, n_ee);
                for (int i = 0; i < std::min(nc,5); ++i) {
                    printf("    ct[%d]: type=%d v=(%d,%d,%d,%d) body=(%d,%d,%d,%d)\n",
                           i, h_ct[i].type,
                           h_ct[i].v[0], h_ct[i].v[1], h_ct[i].v[2], h_ct[i].v[3],
                           h_ct[i].body[0], h_ct[i].body[1], h_ct[i].body[2], h_ct[i].body[3]);
                }
            }
        }

        // Resize scratch buffer if needed
        if (nc > static_cast<int>(d_contact_buf_.gpu_size()))
            d_contact_buf_.resize(nc);

        return nc;
    }

    // ---- Adaptive kappa (IPC paper) ----------------------------------------

    void compute_bbox_and_mass() {
        // Compute bbox from world-space verts (using initial pose)
        Vec3f mn(1e30f, 1e30f, 1e30f), mx(-1e30f, -1e30f, -1e30f);
        for (auto& b : bodies_) {
            Vec3f p = pose_position(b.q);
            Vec3f theta = pose_rotation(b.q);
            Mat3f R = rodrigues(theta);
            for (int vi = 0; vi < b.surface_vert_count; ++vi) {
                Vec3f w = R * h_x_bar_[b.surface_vert_offset + vi] + p;
                mn.x = std::min(mn.x, w.x); mn.y = std::min(mn.y, w.y); mn.z = std::min(mn.z, w.z);
                mx.x = std::max(mx.x, w.x); mx.y = std::max(mx.y, w.y); mx.z = std::max(mx.z, w.z);
            }
        }
        Vec3f d = mx - mn;
        bbox_diagonal_ = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);

        // average_mass: mean of free-DOF mass-matrix diagonal [m,m,m,I1,I2,I3]
        float total = 0;
        int n_free_dof = 0;
        float diag[6];
        for (auto& b : bodies_) {
            if (b.is_fixed) continue;
            diag[0] = diag[1] = diag[2] = b.mass;
            diag[3] = b.moment_of_inertia.x;
            diag[4] = b.moment_of_inertia.y;
            diag[5] = b.moment_of_inertia.z;
            for (int d = 0; d < 6; ++d) {
                if (!b.dof_fixed[d]) {
                    total += diag[d];
                    n_free_dof++;
                }
            }
        }
        average_mass_ = (n_free_dof > 0) ? total / n_free_dof : 1.0f;
    }

    float compute_initial_kappa() {
        // Exact match: ipc_toolkit initial_barrier_stiffness() with dmin=0
        // d0 = (1e-8 * bbox_diagonal)^2
        // min_barrier_stiffness = scale * avg_mass / (4 * d0 * b''(d0, dhat^2))
        // kappa = clamp(-grad_E·grad_B / ||grad_B||^2, min, max)
        // Without gradients at init, kappa_init = 1 → clamped to min
        double dhat = (double)config.d_hat;
        double dhat_sq = dhat * dhat;
        double d0 = 1e-8 * (double)bbox_diagonal_;
        d0 = d0 * d0;
        if (d0 >= dhat_sq) d0 = 0.5 * dhat_sq;

        // barrier_hessian in double: (dhat/d + 2)*(dhat/d) - 2*log(d/dhat) - 3
        double bh;
        if (d0 <= 0.0 || d0 >= dhat_sq) { bh = 0.0; }
        else {
            double dhat_d = dhat_sq / d0;
            bh = (dhat_d + 2.0) * dhat_d - 2.0 * std::log(d0 / dhat_sq) - 3.0;
        }
        double denom = 4.0 * d0 * bh;
        double scale = (double)config.kappa_min_scale;  // 1e11
        double min_kappa = (std::abs(denom) > 1e-30)
            ? scale * (double)average_mass_ / denom : 1.0;
        if (!std::isfinite(min_kappa) || min_kappa <= 0) min_kappa = 1.0;

        max_kappa_ = (float)(100.0 * min_kappa);

        // kappa_init = 1.0 → will be clamped to min by gradient-ratio or directly
        float kappa = 1.0f;
        min_kappa_ = (float)min_kappa;  // store for gradient-ratio clamp
        printf("[RigidIPCSolver] kappa: min=%.4e max=%.4e init=%.4e "
               "bbox=%.6f avg_mass=%.4f\n",
               min_kappa_, max_kappa_, kappa, bbox_diagonal_, average_mass_);
        return kappa;
    }

    // Compute true min distance (linear) across all active contacts
    float gpu_min_distance(int n_contacts) {
        if (n_contacts == 0) return -1.0f;
        float D_hat = config.d_hat * config.d_hat;
        launch_compute_contact_distances(
            d_contacts_.gpu_data(), d_verts_.gpu_data(),
            d_contact_buf_.gpu_data(), D_hat, n_contacts);
        float min_D = gpu_reduce_min(d_contact_buf_.gpu_data(), n_contacts);
        return sqrtf(min_D);  // linear distance
    }

    void update_kappa(float prev_min_dist, float cur_min_dist) {
        // Exact match: ipc_toolkit update_barrier_stiffness() with dmin=0
        // dhat_epsilon = (dhat_epsilon_scale * bbox_diagonal)  [linear]
        // dhat_epsilon *= dhat_epsilon;  [squared]
        // Compare LINEAR distances against SQUARED threshold
        double eps = (double)config.kappa_dhat_eps_scale * (double)bbox_diagonal_;
        eps = eps * eps;  // squared threshold
        if (prev_min_dist >= 0 && cur_min_dist >= 0 &&
            (double)prev_min_dist < eps && (double)cur_min_dist < eps &&
            cur_min_dist < prev_min_dist) {
            kappa_ = std::min(max_kappa_, 2.0f * kappa_);
        }
    }

    // ---- GPU total energy -------------------------------------------------

    float gpu_total_barrier_energy(int n_contacts) {
        if (n_contacts == 0) return 0;
        float D_hat = config.d_hat * config.d_hat;
        launch_barrier_energy(d_contacts_.gpu_data(), d_verts_.gpu_data(),
                              d_contact_buf_.gpu_data(),
                              kappa_, D_hat,
                              n_contacts);
        return gpu_reduce_sum(d_contact_buf_.gpu_data(), n_contacts);
    }

    double gpu_total_energy(int n_contacts) {
        double inv_am = (average_mass_ > 0) ? 1.0 / average_mass_ : 1.0;
        double E = 0;
        for (auto& body : bodies_) {
            if (body.is_fixed) continue;
            E += body_energy_d(body, config.dt, config.gravity);
        }
        E *= inv_am;
        E += static_cast<double>(gpu_total_barrier_energy(n_contacts)) * inv_am;
        return E;
    }

    // Reference parity: rigid-ipc reconstructs the contact constraint set in
    // every compute_objective() call (each Newton iter AND each line-search
    // trial). Detect contacts live at the current GPU verts, then evaluate the
    // total energy over that freshly-detected set. Returns the energy and
    // writes the live contact count to *nc_out.
    double detect_and_total_energy(int* nc_out = nullptr) {
        int nc_live = detect_all_contacts_gpu();
        if (nc_live > static_cast<int>(d_contact_buf_.gpu_size()))
            d_contact_buf_.resize(nc_live);
        if (nc_out) *nc_out = nc_live;
        return gpu_total_energy(nc_live);
    }

    // ---- GPU CCD ----------------------------------------------------------

    float gpu_ccd_from_contacts(int n_contacts, float d_min) {
        if (n_contacts == 0) return 1.0f;
        launch_ccd_contacts(d_contacts_.gpu_data(),
                            d_verts_.gpu_data(), d_verts_trial_.gpu_data(),
                            d_contact_buf_.gpu_data(), d_min,
                            n_contacts);
        return gpu_reduce_min(d_contact_buf_.gpu_data(), n_contacts);
    }

    // ---- Newton solve (main loop) -----------------------------------------

    // Timing accumulators (microseconds) across all Newton iters in one step
    struct ProfileTimers {
        double detect_us = 0;
        double body_assemble_us = 0;
        double contact_assemble_us = 0;
        double pcg_us = 0;
        double ccd_us = 0;
        double line_search_us = 0;
        double upload_us = 0;
        int iters = 0;
    };
    ProfileTimers prof_;

    static double us_since(std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count();
    }

    void print_profile() const {
        double total = prof_.detect_us + prof_.body_assemble_us +
                       prof_.contact_assemble_us + prof_.pcg_us +
                       prof_.ccd_us + prof_.line_search_us + prof_.upload_us;
        if (total < 1.0) return;
        auto pct = [&](double v) { return v * 100.0 / total; };
        printf("  [PROFILE %d iters, %.1f ms total]\n", prof_.iters, total / 1000.0);
        printf("    detect:        %7.1f ms  (%4.1f%%)\n", prof_.detect_us / 1000.0, pct(prof_.detect_us));
        printf("    body_assemble: %7.1f ms  (%4.1f%%)\n", prof_.body_assemble_us / 1000.0, pct(prof_.body_assemble_us));
        printf("    ct_assemble:   %7.1f ms  (%4.1f%%)\n", prof_.contact_assemble_us / 1000.0, pct(prof_.contact_assemble_us));
        printf("    pcg:           %7.1f ms  (%4.1f%%)\n", prof_.pcg_us / 1000.0, pct(prof_.pcg_us));
        printf("    ccd:           %7.1f ms  (%4.1f%%)\n", prof_.ccd_us / 1000.0, pct(prof_.ccd_us));
        printf("    line_search:   %7.1f ms  (%4.1f%%)\n", prof_.line_search_us / 1000.0, pct(prof_.line_search_us));
        printf("    upload/sync:   %7.1f ms  (%4.1f%%)\n", prof_.upload_us / 1000.0, pct(prof_.upload_us));
    }

    // Max vertex displacement L-inf for a given dq
    float compute_max_vertex_disp(const std::vector<Vec6f>& dq, float alpha = 1.0f) {
        float max_disp = 0;
        for (int b = 0; b < n_bodies_; ++b) {
            if (bodies_[b].is_fixed) continue;
            Vec3f p = pose_position(bodies_[b].q);
            Vec3f theta = pose_rotation(bodies_[b].q);
            Vec3f dp = pose_position(dq[b]) * alpha;
            Vec3f dtheta = pose_rotation(dq[b]) * alpha;
            Mat3f R_cur = rodrigues(theta);
            Mat3f R_new = rodrigues(theta + dtheta);

            int off = bodies_[b].surface_vert_offset;
            int cnt = bodies_[b].surface_vert_count;
            for (int vi = 0; vi < cnt; ++vi) {
                Vec3f xb = h_x_bar_[off + vi];
                Vec3f v_cur = R_cur * xb + p;
                Vec3f v_new = R_new * xb + (p + dp);
                Vec3f diff = v_new - v_cur;
                float d = std::max(fabsf(diff.x), std::max(fabsf(diff.y), fabsf(diff.z)));
                if (d > max_disp) max_disp = d;
            }
        }
        return max_disp;
    }

    float compute_step_vertex_speed(const std::vector<Vec6f>& dq) {
        return compute_max_vertex_disp(dq) / config.dt;
    }

    void newton_solve() {
        int n = n_bodies_;
        prof_ = {};
        float prev_min_dist = 1e30f;

        upload_body_q();
        gpu_update_surface_verts();

        float conv_tol = config.velocity_conv_tol;
        if (!config.velocity_conv_tol_abs)
            conv_tol *= bbox_diagonal_;

        float inv_am = (average_mass_ > 0) ? 1.0f / average_mass_ : 1.0f;
        float kappa_over_am = kappa_ * inv_am;

        

        int nc = 0;
        bool contacts_dirty = true;

        for (int iter = 0; iter < config.newton_max_iter; ++iter) {
            prof_.iters = iter + 1;

            // 1. Detect contacts — only when needed
            cudaDeviceSynchronize();
            auto t0 = std::chrono::steady_clock::now();

            if (contacts_dirty) {
                nc = detect_all_contacts_gpu();
                last_info_.contact_count = nc;
                last_info_.min_distance = -1.0f;
                contacts_dirty = false;
            }

            prof_.detect_us += us_since(t0);

            // 2. Assemble body energy (CPU), scaled by 1/average_mass
            t0 = std::chrono::steady_clock::now();

            BlockSystem6 sys;
            sys.resize(n);
            for (int i = 0; i < n; ++i) {
                if (bodies_[i].is_fixed) {
                    sys.H_diag[i] = Mat6f::identity() * 1e10f;
                    sys.rhs[i] = Vec6f::zero();
                    continue;
                }
                Mat6f H_body = body_hessian(bodies_[i], config.dt, config.gravity);
                Vec6f g_body = body_gradient(bodies_[i], config.dt, config.gravity);
                sys.H_diag[i] = H_body * inv_am;
                sys.rhs[i] = g_body * (-inv_am);
            }

            prof_.body_assemble_us += us_since(t0);

            // 3. Kappa initialization (first Newton iter with contacts)
            // Compute kappa from IPC gradient ratio: κ = -∇E·∇B / ||∇B||²
            // Must be done BEFORE contact assembly so we use the final kappa.
            t0 = std::chrono::steady_clock::now();

            if (iter == 0 && nc > 0 && kappa_ <= 1.0f) {
                // Use GPU to get barrier gradient with kappa=1
                launch_zero_vec6(d_grad_.gpu_data(), n);
                float D_hat_k = config.d_hat * config.d_hat;
                float temp_koa = kappa_ * inv_am;
                launch_assemble_contacts(
                    d_contacts_.gpu_data(), d_verts_.gpu_data(),
                    d_x_bar_.gpu_data(), d_vert_body_.gpu_data(),
                    d_body_q_.gpu_data(),
                    d_grad_.gpu_data(), d_hess_diag_.gpu_data(),
                    d_hess_offdiag_.gpu_data(), d_offdiag_pairs_.gpu_data(),
                    d_contact_buf_.gpu_data(),
                    temp_koa, D_hat_k, nc);
                cudaDeviceSynchronize();
                d_grad_.copy_to_host();

                double dot_EB = 0, dot_BB = 0;
                float ka = (kappa_ > 1e-30f) ? kappa_ : 1e-30f;
                for (int i = 0; i < n; ++i) {
                    if (bodies_[i].is_fixed) continue;
                    Vec6f g_E = body_gradient(bodies_[i], config.dt, config.gravity);
                    for (int d = 0; d < 6; ++d) {
                        double gB = -(double)d_grad_.cpu_data()[i][d]
                                    * (double)average_mass_ / (double)ka;
                        dot_EB += (double)g_E[d] * gB;
                        dot_BB += gB * gB;
                    }
                }
                float kappa_grad = (dot_BB > 1e-30) ? (float)(-dot_EB / dot_BB) : 1.0f;
                float old_kappa = kappa_;
                kappa_ = std::min(max_kappa_, std::max(min_kappa_, kappa_grad));
                printf("[RigidIPCSolver] kappa from gradient: %.4e -> %.4e (ratio=%.4e)\n",
                       old_kappa, kappa_, kappa_grad);
            } else if (iter == 0 && kappa_ <= 1.0f) {
                // Reference parity: ipc-toolkit initial_barrier_stiffness clamps
                // kappa to [min_kappa, max_kappa] even with no active barriers
                // (grad_B == 0 -> kappa = min_kappa). Without this, the barrier
                // is ~1e9x too weak on the first step and bodies free-fall into
                // a jam before the set becomes active.
                float old_kappa = kappa_;
                kappa_ = std::min(max_kappa_, std::max(min_kappa_, 1.0f));
                printf("[RigidIPCSolver] kappa clamp-to-min (no active barriers): %.4e -> %.4e\n",
                       old_kappa, kappa_);
            }

            kappa_over_am = kappa_ * inv_am;
            bool use_cpu_full_hessian = (n <= 32);

            // 4. Assemble contact contributions
            if (nc > 0 && use_cpu_full_hessian) {
                // CPU path: full contact Hessian with distance Hessian + PSD projection
                d_verts_.copy_to_host();
                d_contacts_.copy_to_host();
                sys.H_offdiag.clear();
                float D_hat = config.d_hat * config.d_hat;

                for (int ci = 0; ci < nc; ++ci) {
                    GPUContactPair cp = d_contacts_.cpu_data()[ci];
                    const Vec3f* V = d_verts_.cpu_data();
                    int nv_cp;
                    float D;
                    Vec3f gv[4] = {};
                    Vec3f V_cp[4];

                    if (cp.type == 0) {
                        nv_cp = 2;
                        V_cp[0] = V[cp.v[0]]; V_cp[1] = V[cp.v[1]];
                        D = abd_ipc::dist2_pp(V_cp[0], V_cp[1]);
                        if (D >= D_hat) continue;
                        abd_ipc::dist2_pp_grad(V_cp[0], V_cp[1], gv[0], gv[1]);
                    } else if (cp.type == 1) {
                        nv_cp = 3;
                        V_cp[0] = V[cp.v[0]]; V_cp[1] = V[cp.v[1]]; V_cp[2] = V[cp.v[2]];
                        D = abd_ipc::dist2_pe(V_cp[0], V_cp[1], V_cp[2]);
                        if (D >= D_hat) continue;
                        abd_ipc::dist2_pe_grad(V_cp[0], V_cp[1], V_cp[2], gv[0], gv[1], gv[2]);
                    } else if (cp.type == 2) {
                        nv_cp = 4;
                        V_cp[0] = V[cp.v[0]]; V_cp[1] = V[cp.v[1]];
                        V_cp[2] = V[cp.v[2]]; V_cp[3] = V[cp.v[3]];
                        D = abd_ipc::dist2_pt(V_cp[0], V_cp[1], V_cp[2], V_cp[3]);
                        if (D >= D_hat) continue;
                        abd_ipc::dist2_pt_grad(V_cp[0], V_cp[1], V_cp[2], V_cp[3], gv[0], gv[1], gv[2], gv[3]);
                    } else {
                        nv_cp = 4;
                        V_cp[0] = V[cp.v[0]]; V_cp[1] = V[cp.v[1]];
                        V_cp[2] = V[cp.v[2]]; V_cp[3] = V[cp.v[3]];
                        D = abd_ipc::dist2_ee(V_cp[0], V_cp[1], V_cp[2], V_cp[3]);
                        if (D >= D_hat) continue;
                        abd_ipc::dist2_ee_grad(V_cp[0], V_cp[1], V_cp[2], V_cp[3], gv[0], gv[1], gv[2], gv[3]);
                    }

                    float dBdD = abd_ipc::barrier_gradient(D, D_hat);
                    float d2BdD = abd_ipc::barrier_hessian(D, D_hat);

                    // Gradient
                    RigidJacobi Jv[4];
                    int bids[4];
                    for (int k = 0; k < nv_cp; ++k) {
                        bids[k] = cp.body[k];
                        Jv[k] = RigidJacobi(d_x_bar_.cpu_data()[cp.v[k]]);
                        Vec6f gc = Jv[k].mul_JT(gv[k] * (kappa_over_am * dBdD), bodies_[bids[k]].q);
                        sys.rhs[bids[k]] = sys.rhs[bids[k]] + gc * (-1.0f);
                    }

                    // Find 2 distinct body IDs
                    int bodyA = bids[0], bodyB = -1;
                    for (int k = 1; k < nv_cp; ++k)
                        if (bids[k] != bodyA) { bodyB = bids[k]; break; }

                    // 2-body (12×12) layout: [bodyA:0-5, bodyB:6-11]
                    int local_bid[4];
                    for (int k = 0; k < nv_cp; ++k)
                        local_bid[k] = (bids[k] == bodyA) ? 0 : 1;

                    Mat12f H12 = Mat12f::zero();

                    // Compute distance² Hessian ∇²D
                    // PP: analytic (2I, -2I blocks)
                    // PE/PT/EE: finite difference
                    float distH[144] = {}; // max 12×12
                    int dim_v = nv_cp * 3;
                    if (cp.type == 0) {
                        // PP: ∇²D[0,0]=2I, ∇²D[0,1]=-2I, etc.
                        for (int d = 0; d < 3; ++d) {
                            distH[d*6+d] = 2.0f;     // (0,0)
                            distH[d*6+3+d] = -2.0f;  // (0,1)
                            distH[(3+d)*6+d] = -2.0f; // (1,0)
                            distH[(3+d)*6+3+d] = 2.0f; // (1,1)
                        }
                    } else {
                        dist2_hessian_fd(cp.type, V_cp, nv_cp, distH);
                    }

                    for (int a = 0; a < nv_cp; ++a) {
                        for (int b = 0; b < nv_cp; ++b) {
                            Mat3f H3 = Mat3f::zero();
                            for (int r = 0; r < 3; ++r)
                                for (int c = 0; c < 3; ++c) {
                                    H3(r, c) = d2BdD * gv[a].data[r] * gv[b].data[c]
                                              + dBdD * distH[(a*3+r)*dim_v + (b*3+c)];
                                }

                            H3 = H3 * kappa_over_am;

                            Vec6f qa = bodies_[bids[a]].q;
                            Vec6f qb = bodies_[bids[b]].q;
                            Mat6f H6 = RigidJacobi::JT_H_J(Jv[a], H3, Jv[b], qa, qb);

                            int r0 = local_bid[a] * 6, c0 = local_bid[b] * 6;
                            for (int r = 0; r < 6; ++r)
                                for (int c = 0; c < 6; ++c)
                                    H12(r0 + r, c0 + c) += H6(r, c);
                        }
                    }

                    // Vertex Hessian correction
                    for (int k = 0; k < nv_cp; ++k) {
                        Vec3f gB = gv[k] * (kappa_over_am * dBdD);
                        Mat6f corr = Jv[k].vertex_hessian_correction(gB, bodies_[bids[k]].q);
                        int r0 = local_bid[k] * 6;
                        for (int r = 0; r < 6; ++r)
                            for (int c = 0; c < 6; ++c)
                                H12(r0 + r, r0 + c) += corr(r, c);
                    }

                    project_to_psd_12(H12);

                    // Scatter to block-sparse system
                    if (bodyB < 0) bodyB = bodyA;
                    int bid2[2] = { bodyA, bodyB };
                    for (int bi = 0; bi < 2; ++bi) {
                        for (int bj = 0; bj < 2; ++bj) {
                            Mat6f blk = Mat6f::zero();
                            for (int r = 0; r < 6; ++r)
                                for (int c = 0; c < 6; ++c)
                                    blk(r, c) = H12(bi * 6 + r, bj * 6 + c);
                            if (bid2[bi] == bid2[bj]) {
                                sys.H_diag[bid2[bi]] = sys.H_diag[bid2[bi]] + blk;
                            } else {
                                sys.add_offdiag(bid2[bi], bid2[bj], blk);
                            }
                        }
                    }
                }
            } else if (nc > 0) {
                // GPU path: rank-1 Gauss-Newton Hessian
                launch_zero_vec6(d_grad_.gpu_data(), n);
                launch_zero_mat6(d_hess_diag_.gpu_data(), n);

                float D_hat = config.d_hat * config.d_hat;
                launch_assemble_contacts(
                    d_contacts_.gpu_data(), d_verts_.gpu_data(),
                    d_x_bar_.gpu_data(), d_vert_body_.gpu_data(),
                    d_body_q_.gpu_data(),
                    d_grad_.gpu_data(), d_hess_diag_.gpu_data(),
                    d_hess_offdiag_.gpu_data(), d_offdiag_pairs_.gpu_data(),
                    d_contact_buf_.gpu_data(),
                    kappa_over_am, D_hat, nc);
                cudaDeviceSynchronize();
                d_grad_.copy_to_host();
                d_hess_diag_.copy_to_host();
                for (int i = 0; i < n; ++i) {
                    sys.rhs[i] = sys.rhs[i] + d_grad_.cpu_data()[i];
                    sys.H_diag[i] = sys.H_diag[i] + d_hess_diag_.cpu_data()[i];
                }
            }

            prof_.contact_assemble_us += us_since(t0);

            // Clamp locked DOFs: set rhs=0, H_diag=large for per-DOF constraints
            for (int i = 0; i < n; ++i) {
                for (int d = 0; d < 6; ++d) {
                    if (bodies_[i].dof_fixed[d]) {
                        sys.rhs[i][d] = 0.0f;
                        for (int r = 0; r < 6; ++r)
                            sys.H_diag[i](r, d) = sys.H_diag[i](d, r) = 0.0f;
                        sys.H_diag[i](d, d) = 1e10f;
                    }
                }
            }

            // 5. Solve the linear system
            t0 = std::chrono::steady_clock::now();
            std::vector<Vec6f> dq(n, Vec6f::zero());
            int pcg_iters = 0;

            if (n <= 32) {
                solve_dense_ldlt6(sys, dq);
                pcg_iters = 1;
            } else {
                PCGResult pr = solve_pcg6(sys, dq, 1e-4f, 3000);
                pcg_iters = pr.iterations;
            }

            // Negate so dq is a descent direction: H·dq = -g → dq = -H^{-1}·g
            // (rhs already has the correct sign from body_gradient)

            prof_.pcg_us += us_since(t0);

            last_info_.newton_iters = iter + 1;

            // --- Reference: convergence check BEFORE line search ---
            float max_vspeed = compute_step_vertex_speed(dq);
            last_info_.residual = max_vspeed;

            if (iter < 5 || (iter % 50 == 0)) {
                float max_rhs = 0, max_dq_val = 0;
                float max_diag = 0;
                for (int i = 0; i < n; ++i) {
                    if (bodies_[i].is_fixed) continue;
                    for (int d = 0; d < 6; ++d) {
                        max_rhs = fmaxf(max_rhs, fabsf(sys.rhs[i][d]));
                        max_dq_val = fmaxf(max_dq_val, fabsf(dq[i][d]));
                        max_diag = fmaxf(max_diag, fabsf(sys.H_diag[i](d, d)));
                    }
                }
                float offdiag_mag = 0;
                for (auto& od : sys.H_offdiag) {
                    for (int r = 0; r < 6; ++r)
                        for (int c = 0; c < 6; ++c)
                            offdiag_mag = fmaxf(offdiag_mag, fabsf(od.block(r, c)));
                }
                printf("  [NI %d] nc=%d rhs=%.3e dq=%.3e vsp=%.3e k=%.3e Hd=%.3e Hod=%.3e tol=%.3e\n",
                       iter, nc, max_rhs, max_dq_val, max_vspeed, kappa_, max_diag,
                       offdiag_mag, conv_tol);
            }

            if (iter >= 1 && max_vspeed <= conv_tol)
                break;

            // --- Reference: CCD step-size cap (no displacement cap) ---
            t0 = std::chrono::steady_clock::now();

            for (int i = 0; i < n; ++i)
                d_body_dq_.cpu_data()[i] = dq[i];
            d_body_dq_.copy_to_device();

            float max_step_size = 1.0f;
            {
                // CCD with trial vertices: always run regardless of nc
                launch_trial_verts(d_verts_trial_.gpu_data(),
                                   d_x_bar_.gpu_data(), d_vert_body_.gpu_data(),
                                   d_body_q_.gpu_data(), d_body_dq_.gpu_data(),
                                   1.0f, n_verts_);

                // Use existing contacts if available; otherwise detect new ones
                int ccd_nc = nc;
                if (ccd_nc == 0) {
                    // No barrier contacts yet — detect with trial verts to find
                    // potential collisions along the trajectory
                    cudaDeviceSynchronize();

                    // Save current verts, swap in trial verts for detection
                    std::vector<Vec3f> verts_save(n_verts_);
                    cudaMemcpy(verts_save.data(), d_verts_.gpu_data(),
                               n_verts_ * sizeof(Vec3f), cudaMemcpyDeviceToHost);
                    cudaMemcpy(d_verts_.gpu_data(), d_verts_trial_.gpu_data(),
                               n_verts_ * sizeof(Vec3f), cudaMemcpyDeviceToDevice);

                    ccd_nc = detect_all_contacts_gpu();

                    // Restore original verts
                    cudaMemcpy(d_verts_.gpu_data(), verts_save.data(),
                               n_verts_ * sizeof(Vec3f), cudaMemcpyHostToDevice);
                }

                if (ccd_nc > 0) {
                    float ccd_toi = gpu_ccd_from_contacts(ccd_nc, 0.0f);
                    if (iter < 3)
                        printf("    CCD: nc=%d ccd_nc=%d toi=%.6e\n", nc, ccd_nc, ccd_toi);
                    max_step_size = std::min(max_step_size, ccd_toi);
                } else if (iter < 3) {
                    printf("    CCD: nc=%d ccd_nc=0 (skip)\n", nc);
                }
            }

            cudaDeviceSynchronize();
            prof_.ccd_us += us_since(t0);

            // --- Reference: line search with adaptive lower bound ---
            t0 = std::chrono::steady_clock::now();

            // Compute grad·dir for lower bound (grad = -rhs, dir = dq)
            double grad_dot_dir = 0;
            for (int i = 0; i < n; ++i)
                for (int d = 0; d < 6; ++d)
                    grad_dot_dir += (double)(-sys.rhs[i][d]) * (double)dq[i][d];
            double ls_lower_bound = 1e-12 / std::sqrt(std::max(std::abs(grad_dot_dir), 1e-30));
            ls_lower_bound = std::min(ls_lower_bound, 0.1);

            std::vector<Vec6f> q_save(n);
            for (int i = 0; i < n; ++i) q_save[i] = bodies_[i].q;
            // Baseline energy at q_save with a LIVE contact set (reference rebuilds
            // the set every objective eval). Verts are already at q_save here.
            upload_body_q();
            gpu_update_surface_verts();
            cudaDeviceSynchronize();
            int nc_ls = nc;
            double E0 = detect_and_total_energy(&nc_ls);

            float alpha = max_step_size;
            bool found_newton_step = false;
            if (iter < 5 || (iter % 100 == 0))
                printf("    LS: max_step=%.3e gdd=%.3e lb=%.3e E0=%.6e\n",
                       max_step_size, grad_dot_dir, ls_lower_bound, E0);
            while (alpha >= (float)ls_lower_bound) {
                for (int i = 0; i < n; ++i) {
                    if (bodies_[i].is_fixed) continue;
                    bodies_[i].q = q_save[i] + dq[i] * alpha;
                }
                upload_body_q();
                gpu_update_surface_verts();
                cudaDeviceSynchronize();
                double E_trial = detect_and_total_energy(&nc_ls);
                if (E_trial < E0) {
                    found_newton_step = true;
                    break;
                }
                alpha *= 0.5f;
            }

            if (found_newton_step && (iter < 5 || (iter % 100 == 0)))
                printf("    LS: newton alpha=%.3e\n", alpha);
            if (!found_newton_step) {
                // Gradient descent fallback (reference: direction = -gradient)
                for (int i = 0; i < n; ++i) bodies_[i].q = q_save[i];
                upload_body_q();
                gpu_update_surface_verts();
                cudaDeviceSynchronize();

                // GD direction = rhs (= -gradient / am, already scaled)
                // Recompute lower bound for GD direction
                double gd_grad_dot = 0;
                for (int i = 0; i < n; ++i)
                    for (int d = 0; d < 6; ++d)
                        gd_grad_dot += (double)(-sys.rhs[i][d]) * (double)sys.rhs[i][d];
                double gd_lower = 1e-12 / std::sqrt(std::max(std::abs(gd_grad_dot), 1e-30));
                gd_lower = std::min(gd_lower, 0.1);

                alpha = max_step_size;
                bool found_gd_step = false;
                while (alpha >= (float)gd_lower) {
                    for (int i = 0; i < n; ++i) {
                        if (bodies_[i].is_fixed) continue;
                        bodies_[i].q = q_save[i] + sys.rhs[i] * alpha;
                    }
                    upload_body_q();
                    gpu_update_surface_verts();
                    cudaDeviceSynchronize();
                    double E_trial = detect_and_total_energy(&nc_ls);
                    if (E_trial < E0) {
                        found_gd_step = true;
                        break;
                    }
                    alpha *= 0.5f;
                }

                if (!found_gd_step) {
                    // Both failed — restore and break if iter > 0
                    for (int i = 0; i < n; ++i) bodies_[i].q = q_save[i];
                    if (iter > 0) break;
                }
            }

            prof_.line_search_us += us_since(t0);

            // Upload final q, update verts
            t0 = std::chrono::steady_clock::now();
            upload_body_q();
            gpu_update_surface_verts();
            cudaDeviceSynchronize();
            prof_.upload_us += us_since(t0);

            // Re-detect contacts at the accepted pose so the kappa update and
            // next iteration see a consistent live set.
            nc = detect_all_contacts_gpu();
            last_info_.contact_count = nc;
            contacts_dirty = false;

            // Adaptive kappa update — after each Newton step (reference: post_step_update)
            if (nc > 0) {
                float cur_min_dist = gpu_min_distance(nc);
                last_info_.min_distance = cur_min_dist;
                update_kappa(prev_min_dist, cur_min_dist);
                prev_min_dist = cur_min_dist;
            }
        }

        print_profile();
    }
};

}  // namespace rigid_ipc
}  // namespace chysx
