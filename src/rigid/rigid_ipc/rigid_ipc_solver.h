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

class RigidIPCSolver {
public:
    RigidIPCConfig config;

    // ---- bodies and meshes ------------------------------------------------

    void add_body(geometry::TriangleMeshf& mesh, float density,
                  Vec3f position = Vec3f(0, 0, 0),
                  Vec3f rotation_deg = Vec3f(0, 0, 0),
                  float scale = 1.0f,
                  bool fixed = false) {

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
                           bool fixed = false) {
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

        add_body(mesh, density, position, rotation_deg, scale, fixed);
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

            Mat3f R_cur = rodrigues(theta);
            bodies_[i].R = R_cur;
        }

        // Upload body q to GPU and update surface verts
        upload_body_q();
        gpu_update_surface_verts();

        // Newton iteration
        newton_solve();

        // Velocity update
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

            Vec3f theta_old = pose_rotation(bodies_[i].q_prev);
            Vec3f omega = (theta_new - theta_old) * inv_dt;
            bodies_[i].q_v = make_pose(v_lin, omega);
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
    int max_contacts_ = 0;             // capacity of d_contacts_

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
        for (auto& b : bodies_) {
            if (b.is_fixed) continue;
            total += b.mass * 3.0f
                  + b.moment_of_inertia.x + b.moment_of_inertia.y + b.moment_of_inertia.z;
            n_free_dof += 6;
        }
        average_mass_ = (n_free_dof > 0) ? total / n_free_dof : 1.0f;
    }

    float compute_initial_kappa() {
        float dhat = config.d_hat;
        float D_hat = dhat * dhat;
        float d0 = 1e-8f * bbox_diagonal_;
        d0 *= d0;
        if (d0 >= D_hat) d0 = 0.5f * D_hat;
        float b_hess = abd_ipc::barrier_hessian(d0, D_hat);
        float min_kappa_denom = 4.0f * d0 * b_hess;
        float min_kappa = (fabsf(min_kappa_denom) > 1e-30f)
            ? config.kappa_min_scale * average_mass_ / min_kappa_denom
            : 1.0f;
        if (min_kappa < 0) min_kappa = -min_kappa;
        max_kappa_ = 100.0f * min_kappa;
        float kappa = std::max(min_kappa, std::min(max_kappa_, 1.0f));
        printf("[RigidIPCSolver] adaptive kappa: min=%.4e max=%.4e init=%.4e "
               "bbox=%.6f avg_mass=%.4f\n",
               min_kappa, max_kappa_, kappa, bbox_diagonal_, average_mass_);
        return kappa;
    }

    void update_kappa(float prev_min_dist, float cur_min_dist) {
        float eps = config.kappa_dhat_eps_scale * bbox_diagonal_;
        eps *= eps;
        if (prev_min_dist < eps && cur_min_dist < eps &&
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

    float gpu_total_energy(int n_contacts) {
        float inv_am = (average_mass_ > 0) ? 1.0f / average_mass_ : 1.0f;
        float E = 0;
        for (auto& body : bodies_) {
            if (body.is_fixed) continue;
            E += body_energy(body, config.dt, config.gravity);
        }
        E *= inv_am;
        E += gpu_total_barrier_energy(n_contacts) * inv_am;
        return E;
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

        // Start Newton from current pose (q_prev), NOT predictor (q_tilde).
        upload_body_q();
        gpu_update_surface_verts();

        float conv_tol = config.velocity_conv_tol;
        if (!config.velocity_conv_tol_abs)
            conv_tol *= bbox_diagonal_;

        float inv_am = (average_mass_ > 0) ? 1.0f / average_mass_ : 1.0f;
        float kappa_over_am = kappa_ * inv_am;

        for (int iter = 0; iter < config.newton_max_iter; ++iter) {
            prof_.iters = iter + 1;

            // 1. Detect contacts (fully on GPU, every iteration)
            cudaDeviceSynchronize();
            auto t0 = std::chrono::steady_clock::now();

            int nc = detect_all_contacts_gpu();
            last_info_.contact_count = nc;
            last_info_.min_distance = -1.0f;

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

            // 3. Assemble contact contributions (GPU) using kappa/average_mass
            t0 = std::chrono::steady_clock::now();

            kappa_over_am = kappa_ * inv_am;

            if (nc > 0) {
                launch_zero_vec6(d_grad_.gpu_data(), n);
                launch_zero_mat6(d_hess_diag_.gpu_data(), n);

                float D_hat = config.d_hat * config.d_hat;
                launch_assemble_contacts(
                    d_contacts_.gpu_data(), d_verts_.gpu_data(),
                    d_x_bar_.gpu_data(), d_vert_body_.gpu_data(),
                    d_body_q_.gpu_data(),
                    d_grad_.gpu_data(), d_hess_diag_.gpu_data(),
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

            // 4. PCG solve (CPU)
            t0 = std::chrono::steady_clock::now();

            std::vector<Vec6f> dq(n);
            auto pcg_result = solve_pcg6(sys, dq, config.pcg_tol,
                                          n * 6 * config.pcg_max_iter_ratio);

            prof_.pcg_us += us_since(t0);

            // Convergence: max vertex speed
            float max_vspeed = compute_step_vertex_speed(dq);
            float max_disp = compute_max_vertex_disp(dq);
            last_info_.newton_iters = iter + 1;
            last_info_.residual = max_vspeed;

            if (iter >= config.newton_min_iter && max_vspeed <= conv_tol)
                break;

            // 5. CCD (GPU) — use detected contacts for CCD
            t0 = std::chrono::steady_clock::now();

            for (int i = 0; i < n; ++i)
                d_body_dq_.cpu_data()[i] = dq[i];
            d_body_dq_.copy_to_device();

            float alpha_ccd = 1.0f;
            if (nc > 0) {
                launch_trial_verts(d_verts_trial_.gpu_data(),
                                   d_x_bar_.gpu_data(), d_vert_body_.gpu_data(),
                                   d_body_q_.gpu_data(), d_body_dq_.gpu_data(),
                                   1.0f, n_verts_);
                alpha_ccd = gpu_ccd_from_contacts(nc, config.d_hat * 0.1f);
            }

            // Limit max vertex displacement: prevent jumping over d_hat gap
            // This is a conservative replacement for global CCD. The original
            // rigid-ipc uses broadphase CCD along the full motion trajectory.
            {
                float disp_limit = config.d_hat;
                if (max_disp > disp_limit) {
                    float alpha_disp = disp_limit / max_disp;
                    if (alpha_disp < alpha_ccd)
                        alpha_ccd = alpha_disp;
                }
            }

            cudaDeviceSynchronize();
            prof_.ccd_us += us_since(t0);

            // 6. Line search — strict decrease: f(x+αd) < f(x)
            t0 = std::chrono::steady_clock::now();

            {
                float E0 = gpu_total_energy(nc);

                std::vector<Vec6f> q_save(n);
                for (int i = 0; i < n; ++i) q_save[i] = bodies_[i].q;

                float alpha = alpha_ccd;
                bool ls_success = false;
                for (int ls = 0; ls < config.line_search_max_iter; ++ls) {
                    for (int i = 0; i < n; ++i) {
                        if (bodies_[i].is_fixed) continue;
                        bodies_[i].q = q_save[i] + dq[i] * alpha;
                    }
                    upload_body_q();
                    gpu_update_surface_verts();
                    cudaDeviceSynchronize();

                    float E_trial = gpu_total_energy(nc);
                    if (E_trial < E0) {
                        ls_success = true;
                        break;
                    }
                    alpha *= 0.5f;
                }

                if (!ls_success) {
                    // Revert to q_save — no improvement found
                    for (int i = 0; i < n; ++i) bodies_[i].q = q_save[i];
                } else {
                    for (int i = 0; i < n; ++i) {
                        if (bodies_[i].is_fixed) continue;
                        bodies_[i].q = q_save[i] + dq[i] * alpha;
                    }
                }
            }

            prof_.line_search_us += us_since(t0);

            // 7. Upload updated q and update verts for next iter
            t0 = std::chrono::steady_clock::now();
            upload_body_q();
            gpu_update_surface_verts();
            cudaDeviceSynchronize();
            prof_.upload_us += us_since(t0);

            // Adaptive kappa update
            if (nc > 0) {
                float cur_min_dist = gpu_reduce_min(d_contact_buf_.gpu_data(), nc);
                update_kappa(prev_min_dist, cur_min_dist);
                prev_min_dist = cur_min_dist;
            }
        }

        print_profile();
    }
};

}  // namespace rigid_ipc
}  // namespace chysx
