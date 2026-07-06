// SPDX-License-Identifier: MIT
// ABD IPC solver: Newton iteration with BDF1 time integration.

#pragma once

#include "abd_ipc_types.cuh"
#include "abd_ipc_mesh.h"
#include "abd_ipc_energy.cuh"
#include "abd_ipc_assembly.cuh"
#include "abd_ipc_pcg.cuh"
#include "abd_ipc_contact.h"
#include "abd_ipc_ccd.cuh"
#include "abd_ipc_ccd_kernels.cuh"               // GPU full-primitive CCD candidates
#include "abd_ipc_gpu_kernels.cuh"               // GPU full-Hessian contact assembly
#include "../rigid_ipc/rigid_ipc_kernels.cuh"   // GPU broadphase + CCD kernels
#include "../../collision/simplex_trajectory_filter.h"  // libuipc-style LBVH detection
#include "../../geometry/tet_mesh.h"
#include "../../geometry/triangle_mesh.h"
#include "../../memory/cuda_array.h"
#include "../../io/obj_io.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace chysx {
namespace abd_ipc {

// Axis-angle -> 3x3 rotation matrix (Rodrigues). `axis` must be unit length.
inline Mat3f axis_angle_to_mat3(Vec3f axis, float theta) {
    float c = std::cos(theta), s = std::sin(theta);
    float t = 1.0f - c;
    float x = axis.x, y = axis.y, z = axis.z;
    Mat3f R;
    R(0,0) = c + x*x*t;   R(0,1) = x*y*t - z*s; R(0,2) = x*z*t + y*s;
    R(1,0) = y*x*t + z*s; R(1,1) = c + y*y*t;   R(1,2) = y*z*t - x*s;
    R(2,0) = z*x*t - y*s; R(2,1) = z*y*t + x*s; R(2,2) = c + z*z*t;
    return R;
}

class ABDSolver {
public:
    ABDConfig config;

    // ---- bodies and meshes ------------------------------------------------

    void add_body(geometry::TetMeshf& mesh, float density, float kappa,
                  Vec3f translation = Vec3f(0, 0, 0), bool fixed = false) {
        ABDBody body = init_body(mesh, density, kappa, config.gravity, translation, fixed);
        body.surface_vert_offset = static_cast<int>(surface_.verts.size());

        // Extract surface for this mesh
        mesh.extract_surface();
        int nv = static_cast<int>(mesh.num_vertices());
        int nt = static_cast<int>(mesh.num_surface_tris());
        int ne = static_cast<int>(mesh.num_surface_edges());

        body.surface_vert_count = nv;
        body.surface_tri_offset = static_cast<int>(surface_.tris.size());
        body.surface_tri_count = nt;
        body.surface_edge_offset = static_cast<int>(surface_.edges.size());
        body.surface_edge_count = ne;

        int body_idx = static_cast<int>(bodies_.size());

        // Copy vertices (rest-frame + Jacobi)
        const Vec3f* verts = mesh.vertices().cpu_data();
        auto jacobi = build_jacobi_array(mesh);
        for (int i = 0; i < nv; ++i) {
            surface_.verts.push_back(verts[i]);
            surface_.jacobi.push_back(jacobi[i]);
            surface_.vert_body.push_back(body_idx);
        }

        // Copy triangles (offset indices to global)
        const auto* tris = mesh.surface_tris().cpu_data();
        int v_off = body.surface_vert_offset;
        for (int i = 0; i < nt; ++i) {
            surface_.tris.push_back(Vec3i(tris[i].x + v_off,
                                          tris[i].y + v_off,
                                          tris[i].z + v_off));
        }

        // Copy edges (offset indices)
        const auto* edges = mesh.surface_edges().cpu_data();
        for (int i = 0; i < ne; ++i) {
            surface_.edges.push_back(math::Vec2i(edges[i].x + v_off,
                                                  edges[i].y + v_off));
        }

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

    // Add a body from a closed triangle mesh (no tets needed).
    void add_body(geometry::TriangleMeshf& mesh, float density, float kappa,
                  Vec3f translation = Vec3f(0, 0, 0), bool fixed = false) {
        mesh.build_edges();

        ABDBody body = init_body(mesh, density, kappa, config.gravity, translation, fixed);
        body.surface_vert_offset = static_cast<int>(surface_.verts.size());

        int nv = static_cast<int>(mesh.num_vertices());
        int nt = static_cast<int>(mesh.num_triangles());
        int ne = static_cast<int>(mesh.num_edges());

        body.surface_vert_count = nv;
        body.surface_tri_offset = static_cast<int>(surface_.tris.size());
        body.surface_tri_count = nt;
        body.surface_edge_offset = static_cast<int>(surface_.edges.size());
        body.surface_edge_count = ne;

        int body_idx = static_cast<int>(bodies_.size());

        const Vec3f* verts = mesh.vertices().cpu_data();
        auto jacobi = build_jacobi_array(mesh);
        for (int i = 0; i < nv; ++i) {
            surface_.verts.push_back(verts[i]);
            surface_.jacobi.push_back(jacobi[i]);
            surface_.vert_body.push_back(body_idx);
        }

        const auto* tris = mesh.triangles().cpu_data();
        int v_off = body.surface_vert_offset;
        for (int i = 0; i < nt; ++i)
            surface_.tris.push_back(Vec3i(tris[i].x + v_off,
                                          tris[i].y + v_off,
                                          tris[i].z + v_off));

        const auto* edges = mesh.edges().cpu_data();
        for (int i = 0; i < ne; ++i)
            surface_.edges.push_back(math::Vec2i(edges[i].x + v_off,
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

    // Load an OBJ file and add it as a body (convenience wrapper).
    bool add_body_from_obj(const std::string& path, float density, float kappa,
                           Vec3f translation = Vec3f(0, 0, 0), bool fixed = false) {
        io::ObjMesh obj;
        if (!io::load_obj(path, obj)) return false;
        int nv = static_cast<int>(obj.positions.size()) / 3;
        int nt = static_cast<int>(obj.triangles.size()) / 3;

        geometry::TriangleMeshf mesh(nv, nt);
        auto* verts = mesh.vertices().cpu_data();
        for (int i = 0; i < nv; ++i)
            verts[i] = Vec3f(obj.positions[i*3], obj.positions[i*3+1], obj.positions[i*3+2]);
        auto* tris = mesh.triangles().cpu_data();
        for (int i = 0; i < nt; ++i)
            tris[i] = Vec3i(obj.triangles[i*3], obj.triangles[i*3+1], obj.triangles[i*3+2]);

        add_body(mesh, density, kappa, translation, fixed);
        return true;
    }

    // Attach a rotating motor to a body (libuipc RotatingMotor). The motor
    // soft-drives the body's rotation about `axis` at `omega` rad/s with the
    // given `strength`; translation stays free so threaded contact converts the
    // rotation into axial motion.
    void set_motor(int body_idx, Vec3f axis, float omega, float strength) {
        ABDBody& b = bodies_[body_idx];
        b.is_motor = true;
        b.motor_axis = math::normalize(axis);
        b.motor_omega = omega;
        b.motor_strength = strength;
        b.motor_q_aim = b.q;
    }

    // ---- simulation -------------------------------------------------------

    // Advance one time step.
    void step() {
        int n = static_cast<int>(bodies_.size());
        float dt = config.dt;
        float inv_dt = 1.0f / dt;

        // Adaptive barrier stiffness: start soft, ramp toward contact_kappa.
        if (!kappa_init_) {
            kappa_ = config.adaptive_kappa ? config.contact_kappa_init
                                           : config.contact_kappa;
            kappa_init_ = true;
        }

        // 0. Save q_prev at start of step (used by CCD predictor clamping)
        for (int i = 0; i < n; ++i)
            bodies_[i].q_prev = bodies_[i].q;

        // Motor: advance the aim rotation by omega*dt about the axis. Matches
        // libuipc RotatingMotor::animate (prerotate current transform), single
        // substep so q_aim == aim_transform.
        for (int i = 0; i < n; ++i) {
            if (!bodies_[i].is_motor) continue;
            Mat3f R_delta = axis_angle_to_mat3(bodies_[i].motor_axis,
                                               bodies_[i].motor_omega * dt);
            Mat3f A_cur = q_to_A(bodies_[i].q);
            Mat3f A_aim = R_delta * A_cur;
            bodies_[i].motor_q_aim = make_q(q_translation(bodies_[i].q), A_aim);
        }

        // 1. BDF1 predictor: q_tilde = q_prev + q_v * dt + g * dt²
        // g = abd_gravity is the pre-computed generalized gravity acceleration
        for (int i = 0; i < n; ++i) {
            if (bodies_[i].is_fixed) {
                bodies_[i].q_tilde = bodies_[i].q;
                continue;
            }
            bodies_[i].q_tilde = bodies_[i].q
                               + bodies_[i].q_v * dt
                               + bodies_[i].abd_gravity * (dt * dt);
        }

        // 2. Update surface vertices from current q
        update_surface_verts();

        // 3. Newton iteration
        if (config.use_gpu_newton)
            newton_solve_gpu();
        else
            newton_solve();

        // 4. Velocity update: q_v = (q_new - q_old) / dt
        for (int i = 0; i < n; ++i) {
            if (bodies_[i].is_fixed) continue;
            bodies_[i].q_v = (bodies_[i].q - bodies_[i].q_prev) * inv_dt;
            bodies_[i].q_prev = bodies_[i].q;
        }

        // 5. Adaptive kappa update (IPC): while contacts are active, double the
        // barrier stiffness toward the maximum so the now-engaged bodies are
        // held firmly. The soft start let them slide into engagement first.
        if (config.adaptive_kappa && kappa_ < config.contact_kappa) {
            bool active = false;
            if (config.use_gpu_newton) {
                active = last_info_.contact_count > 0;  // set during newton_solve_gpu
            } else {
                update_surface_verts();
                auto contacts = detect_all_contacts();
                float D_hat = config.d_hat * config.d_hat;
                for (auto& cp : contacts) {
                    float D = (cp.type == ContactType::PT)
                        ? dist2_pt(surface_.verts[cp.v[0]], surface_.verts[cp.v[1]],
                                   surface_.verts[cp.v[2]], surface_.verts[cp.v[3]])
                        : dist2_ee(surface_.verts[cp.v[0]], surface_.verts[cp.v[1]],
                                   surface_.verts[cp.v[2]], surface_.verts[cp.v[3]]);
                    if (D < D_hat) { active = true; break; }
                }
            }
            if (active)
                kappa_ = std::min(config.contact_kappa, kappa_ * 2.0f);
        }
    }

    // ---- accessors --------------------------------------------------------

    int num_bodies() const { return static_cast<int>(bodies_.size()); }
    const ABDBody& body(int i) const { return bodies_[i]; }
    const SurfaceData& surface() const { return surface_; }
    const std::vector<Vec3f>& current_verts() const { return surface_.verts; }

    // Get surface vertices for a specific body (for OBJ export etc.)
    std::vector<Vec3f> body_surface_verts(int body_idx) const {
        auto& r = surface_.body_ranges[body_idx];
        return {surface_.verts.begin() + r.vert_begin,
                surface_.verts.begin() + r.vert_end};
    }

    std::vector<Vec3i> body_surface_tris(int body_idx) const {
        auto& r = surface_.body_ranges[body_idx];
        int offset = r.vert_begin;
        std::vector<Vec3i> tris;
        for (int i = r.tri_begin; i < r.tri_end; ++i) {
            Vec3i t = surface_.tris[i];
            tris.push_back(Vec3i(t.x - offset, t.y - offset, t.z - offset));
        }
        return tris;
    }

    // Diagnostic: last Newton iteration info
    struct StepInfo {
        int newton_iters;
        int contact_count;
        float residual;
    };
    StepInfo last_step_info() const { return last_info_; }

private:
    std::vector<ABDBody> bodies_;
    SurfaceData surface_;
    StepInfo last_info_{};
    collision::EFContactDetector ef_detector_;
    bool bvh_initialized_ = false;

    // Runtime (adaptive) barrier stiffness. Equals config.contact_kappa when
    // adaptive_kappa is disabled.
    float kappa_ = 0.0f;
    bool  kappa_init_ = false;

    // ---- GPU CCD state ----------------------------------------------------
    // World-space verts and candidate contacts live on GPU so the swept
    // time-of-impact reuses the rigid_ipc broadphase + CCD kernels instead of
    // the O(nv*nt + ne^2) CPU brute force.
    CudaArray<Vec3f> d_verts_cur_;
    CudaArray<Vec3f> d_verts_next_;
    CudaArray<int> d_vert_body_;
    CudaArray<Vec3i> d_tris_;
    CudaArray<math::Vec2i> d_edges_;
    CudaArray<float> d_alpha_;
    int gpu_nv_ = 0, gpu_nt_ = 0, gpu_ne_ = 0;
    bool gpu_ccd_ready_ = false;

    void gpu_ccd_setup() {
        gpu_nv_ = static_cast<int>(surface_.verts.size());
        gpu_nt_ = static_cast<int>(surface_.tris.size());
        gpu_ne_ = static_cast<int>(surface_.edges.size());

        d_verts_cur_.resize(gpu_nv_);
        d_verts_next_.resize(gpu_nv_);
        d_vert_body_.resize(gpu_nv_);
        std::memcpy(d_vert_body_.cpu_data(), surface_.vert_body.data(), gpu_nv_ * sizeof(int));
        d_vert_body_.copy_to_device();

        d_tris_.resize(gpu_nt_);
        std::memcpy(d_tris_.cpu_data(), surface_.tris.data(), gpu_nt_ * sizeof(Vec3i));
        d_tris_.copy_to_device();

        d_edges_.resize(gpu_ne_);
        std::memcpy(d_edges_.cpu_data(), surface_.edges.data(), gpu_ne_ * sizeof(math::Vec2i));
        d_edges_.copy_to_device();

        d_alpha_.resize(1);
        gpu_ccd_ready_ = true;
    }

    // GPU brute-force time-of-impact: parallelizes the exact CPU compute_toi
    // sweep (every inter-body point-triangle and edge-edge pair). Complete, so
    // it matches the CPU result and cannot tunnel.
    float compute_toi_gpu(const std::vector<Vec3f>& verts_cur,
                          const std::vector<Vec3f>& verts_next,
                          float d_min) {
        if (!gpu_ccd_ready_) gpu_ccd_setup();
        int nv = static_cast<int>(verts_cur.size());

        std::memcpy(d_verts_cur_.cpu_data(), verts_cur.data(), nv * sizeof(Vec3f));
        d_verts_cur_.copy_to_device();
        std::memcpy(d_verts_next_.cpu_data(), verts_next.data(), nv * sizeof(Vec3f));
        d_verts_next_.copy_to_device();

        float one = 1.0f;
        std::memcpy(d_alpha_.cpu_data(), &one, sizeof(float));
        d_alpha_.copy_to_device();

        launch_ccd_brute_pt(d_verts_cur_.gpu_data(), d_verts_next_.gpu_data(),
                            d_tris_.gpu_data(), d_vert_body_.gpu_data(),
                            gpu_nv_, gpu_nt_, d_min, d_alpha_.gpu_data());
        launch_ccd_brute_ee(d_verts_cur_.gpu_data(), d_verts_next_.gpu_data(),
                            d_edges_.gpu_data(), d_vert_body_.gpu_data(),
                            gpu_ne_, d_min, d_alpha_.gpu_data());
        cudaDeviceSynchronize();
        d_alpha_.copy_to_host();
        float alpha = d_alpha_.cpu_data()[0];
        if (alpha < 0.0f) alpha = 0.0f;
        if (alpha > 1.0f) alpha = 1.0f;
        return alpha;
    }

    // ---- GPU Newton state -------------------------------------------------
    CudaArray<Vec3f>  d_x_bar_;          // rest verts [nv]
    CudaArray<Vec12f> d_body_q_;         // body q [nb]
    CudaArray<Vec12f> d_ngrad_;          // contact gradient accumulator [nb]
    CudaArray<Mat12f> d_nhess_;          // contact Hessian diagonal [nb]
    CudaArray<rigid_ipc::GPUContactPair> d_ncontacts_;
    CudaArray<int>    d_nccount_;
    CudaArray<float>  d_nebuf_;
    int gpu_newton_maxc_ = 0;
    bool gpu_newton_ready_ = false;

    void gpu_newton_setup() {
        if (!gpu_ccd_ready_) gpu_ccd_setup();
        int nv = gpu_nv_;
        int nb = static_cast<int>(bodies_.size());
        if (!simplex_ready_) {
            simplex_filter_.setup(nv, surface_.edges, surface_.tris, surface_.vert_body);
            simplex_ready_ = true;
        }
        d_x_bar_.resize(nv);
        for (int i = 0; i < nv; ++i)
            d_x_bar_.cpu_data()[i] = surface_.jacobi[i].x_bar;
        d_x_bar_.copy_to_device();

        d_body_q_.resize(nb);
        d_ngrad_.resize(nb);
        d_nhess_.resize(nb);
        gpu_newton_maxc_ = std::max(gpu_ne_ * 16, 1 << 20);
        d_ncontacts_.resize(gpu_newton_maxc_);
        d_nccount_.resize(1);
        d_nebuf_.resize(gpu_newton_maxc_);
        gpu_newton_ready_ = true;
    }

    // Upload current world verts (from CPU surface_.verts) to the GPU buffer.
    void gpu_upload_verts() {
        std::memcpy(d_verts_cur_.cpu_data(), surface_.verts.data(),
                    gpu_nv_ * sizeof(Vec3f));
        d_verts_cur_.copy_to_device();
    }

    // GPU detect + filter: builds the active PT/EE contact set on GPU from the
    // current world verts. Returns the contact count.
    int gpu_detect_filter() {
        simplex_filter_.detect(d_verts_cur_.gpu_data(), nullptr, 0.0f, config.d_hat);
        float D_hat = config.d_hat * config.d_hat;
        rigid_ipc::launch_zero_int(d_nccount_.gpu_data());
        launch_abd_filter_pt(simplex_filter_.pt_pairs_dev(), simplex_filter_.num_pt_pairs(),
                             d_verts_cur_.gpu_data(), simplex_filter_.tris_dev(),
                             simplex_filter_.vert_body_dev(), D_hat,
                             d_ncontacts_.gpu_data(), d_nccount_.gpu_data(), gpu_newton_maxc_);
        launch_abd_filter_ee(simplex_filter_.ee_pairs_dev(), simplex_filter_.num_ee_pairs(),
                             d_verts_cur_.gpu_data(), simplex_filter_.edges_dev(),
                             simplex_filter_.vert_body_dev(), D_hat,
                             d_ncontacts_.gpu_data(), d_nccount_.gpu_data(), gpu_newton_maxc_);
        int nc = rigid_ipc::read_contact_count(d_nccount_.gpu_data());
        if (nc > gpu_newton_maxc_) nc = gpu_newton_maxc_;
        return nc;
    }

    // Total energy = CPU body/motor energy + GPU barrier energy over the fixed
    // contact set `nc` evaluated at the current (already uploaded) GPU verts.
    double total_energy_gpu(int nc) {
        double E = 0;
        for (auto& body : bodies_) {
            if (body.is_fixed) continue;
            E += assemble_body_energy(body, config.dt);
            if (body.is_motor) {
                Mat12f Ms = motor_metric(body);
                Vec12f dq = body.q - body.motor_q_aim;
                E += 0.5 * (double)math::dot(dq, Ms * dq);
            }
        }
        if (nc > 0) {
            float D_hat = config.d_hat * config.d_hat;
            rigid_ipc::launch_barrier_energy(d_ncontacts_.gpu_data(), d_verts_cur_.gpu_data(),
                                             d_nebuf_.gpu_data(), kappa_, D_hat, nc);
            E += (double)rigid_ipc::gpu_reduce_sum(d_nebuf_.gpu_data(), nc);
        }
        return E;
    }

    // GPU Newton: dense-contact IPC solve. GPU detection + full-Hessian
    // assembly (PSD-projected) + GPU barrier-energy line search; the tiny
    // (<=24 DOF) linear system is solved on the CPU.
    void newton_solve_gpu() {
        int n = static_cast<int>(bodies_.size());
        if (!gpu_newton_ready_) gpu_newton_setup();

        // Predictor + CCD clamp (reuse the brute GPU CCD via the dispatcher).
        // CCD only guards against actual penetration; the separation gap is
        // maintained by the barrier, so keep d_min a small fraction of d_hat
        // (a large d_min would cage the bodies and freeze all motion).
        float ccd_dmin = config.d_hat * 0.01f;

        std::vector<Vec3f> verts_pre(surface_.verts);
        for (int i = 0; i < n; ++i)
            if (!bodies_[i].is_fixed) bodies_[i].q = bodies_[i].q_tilde;
        update_surface_verts();
        float toi = compute_toi(verts_pre, surface_.verts, ccd_dmin);
        if (toi < 1.0f) {
            for (int i = 0; i < n; ++i) {
                if (bodies_[i].is_fixed) continue;
                Vec12f q0 = bodies_[i].q_prev;
                bodies_[i].q = q0 + (bodies_[i].q_tilde - q0) * toi;
                bodies_[i].q_tilde = bodies_[i].q;
            }
            update_surface_verts();
        }

        for (int iter = 0; iter < config.newton_max_iter; ++iter) {
            update_surface_verts();
            gpu_upload_verts();

            int nc = gpu_detect_filter();
            last_info_.contact_count = nc;

            // GPU full-Hessian contact assembly.
            launch_abd_zero_vec12(d_ngrad_.gpu_data(), n);
            launch_abd_zero_mat12(d_nhess_.gpu_data(), n);
            launch_abd_assemble(d_ncontacts_.gpu_data(), nc,
                                d_verts_cur_.gpu_data(), d_x_bar_.gpu_data(),
                                simplex_filter_.vert_body_dev(),
                                kappa_, config.d_hat * config.d_hat,
                                d_ngrad_.gpu_data(), d_nhess_.gpu_data(), n);
            cudaDeviceSynchronize();
            d_ngrad_.copy_to_host();
            d_nhess_.copy_to_host();

            // Assemble the (tiny) block system on the CPU.
            BlockSystem sys;
            sys.resize(n);
            for (int i = 0; i < n; ++i) {
                if (bodies_[i].is_fixed) {
                    sys.H_diag[i] = Mat12f::identity() * 1e10f;
                    sys.rhs[i] = Vec12f::zero();
                    continue;
                }
                Mat12f H = assemble_body_hessian(bodies_[i], config.dt);
                Vec12f g = assemble_body_gradient(bodies_[i], config.dt);
                if (bodies_[i].is_motor) {
                    Mat12f Ms = motor_metric(bodies_[i]);
                    H = H + Ms;
                    g = g + Ms * (bodies_[i].q - bodies_[i].motor_q_aim);
                }
                H = H + d_nhess_.cpu_data()[i];
                sys.H_diag[i] = H;
                sys.rhs[i] = (g + d_ngrad_.cpu_data()[i]) * (-1.0f);
            }

            std::vector<Vec12f> dq(n);
            solve_pcg(sys, dq, config.pcg_tol, n * 12 * config.pcg_max_iter_ratio);

            float max_dq = 0;
            for (int i = 0; i < n; ++i) {
                if (bodies_[i].is_fixed) continue;
                float m = math::max_abs(dq[i]);
                if (m > max_dq) max_dq = m;
            }
            last_info_.newton_iters = iter + 1;
            last_info_.residual = max_dq;
            if (iter >= config.newton_min_iter && max_dq < config.velocity_tol * config.dt)
                break;

            // CCD step-size cap.
            std::vector<Vec3f> verts_trial(surface_.verts.size());
            for (int b = 0; b < n; ++b) {
                auto& range = surface_.body_ranges[b];
                Vec12f q_trial = bodies_[b].q + dq[b];
                for (int vi = range.vert_begin; vi < range.vert_end; ++vi)
                    verts_trial[vi] = surface_.jacobi[vi].mul_q(q_trial);
            }
            float alpha_ccd = compute_toi(surface_.verts, verts_trial, ccd_dmin);

            // Energy line search on the fixed contact set (GPU barrier energy).
            double E0 = total_energy_gpu(nc);
            double dir_deriv = 0;
            for (int i = 0; i < n; ++i)
                dir_deriv += (double)math::dot(sys.rhs[i] * (-1.0f), dq[i]);

            std::vector<Vec12f> q_save(n);
            for (int i = 0; i < n; ++i) q_save[i] = bodies_[i].q;

            float alpha = alpha_ccd;
            for (int ls = 0; ls < config.line_search_max_iter; ++ls) {
                for (int i = 0; i < n; ++i) {
                    if (bodies_[i].is_fixed) continue;
                    bodies_[i].q = q_save[i] + dq[i] * alpha;
                }
                update_surface_verts();
                gpu_upload_verts();
                double E_trial = total_energy_gpu(nc);
                if (E_trial <= E0 + 1e-4 * alpha * dir_deriv) break;
                alpha *= 0.5f;
            }
            for (int i = 0; i < n; ++i) {
                if (bodies_[i].is_fixed) continue;
                bodies_[i].q = q_save[i] + dq[i] * alpha;
            }
        }
    }

    // libuipc-style LBVH broadphase (swept AABB) -> complete PT/EE candidates,
    // then narrow-phase distance filter to active contacts within d_hat.
    collision::SimplexTrajectoryFilter simplex_filter_;
    bool simplex_ready_ = false;

    std::vector<ContactPair> detect_all_contacts_lbvh() {
        int nv = static_cast<int>(surface_.verts.size());
        if (!simplex_ready_) {
            simplex_filter_.setup(nv, surface_.edges, surface_.tris, surface_.vert_body);
            simplex_ready_ = true;
        }
        // Reuse the GPU CCD vertex buffer for the current world positions.
        if (!gpu_ccd_ready_) gpu_ccd_setup();
        std::memcpy(d_verts_cur_.cpu_data(), surface_.verts.data(), nv * sizeof(Vec3f));
        d_verts_cur_.copy_to_device();

        // Discrete query: dx = null, alpha = 0, inflate by d_hat.
        simplex_filter_.detect(d_verts_cur_.gpu_data(), nullptr, 0.0f, config.d_hat);

        std::vector<math::Vec2i> pt, ee;
        simplex_filter_.download_pt(pt);
        simplex_filter_.download_ee(ee);

        float D_hat = config.d_hat * config.d_hat;
        std::vector<ContactPair> out;
        out.reserve(pt.size() + ee.size());

        // Point-triangle candidates: (point_id, tri_id)
        for (auto& c : pt) {
            int vi = c.x;
            Vec3i f = surface_.tris[c.y];
            int bv = surface_.vert_body[vi];
            if (bv == surface_.vert_body[f.x] && bv == surface_.vert_body[f.y] &&
                bv == surface_.vert_body[f.z]) continue;
            if (vi == f.x || vi == f.y || vi == f.z) continue;
            float D = dist2_pt(surface_.verts[vi], surface_.verts[f.x],
                               surface_.verts[f.y], surface_.verts[f.z]);
            if (D >= D_hat) continue;
            ContactPair cp;
            cp.type = ContactType::PT;
            cp.v[0] = vi; cp.v[1] = f.x; cp.v[2] = f.y; cp.v[3] = f.z;
            cp.body[0] = bv; cp.body[1] = surface_.vert_body[f.x];
            cp.body[2] = surface_.vert_body[f.y]; cp.body[3] = surface_.vert_body[f.z];
            cp.D = D;
            out.push_back(cp);
        }
        // Edge-edge candidates: (edge_a, edge_b)
        for (auto& c : ee) {
            math::Vec2i ea = surface_.edges[c.x];
            math::Vec2i eb = surface_.edges[c.y];
            int ba0 = surface_.vert_body[ea.x], ba1 = surface_.vert_body[ea.y];
            int bb0 = surface_.vert_body[eb.x], bb1 = surface_.vert_body[eb.y];
            if (ba0 == bb0 && ba0 == bb1 && ba1 == bb0 && ba1 == bb1) continue;
            if (ea.x == eb.x || ea.x == eb.y || ea.y == eb.x || ea.y == eb.y) continue;
            float D = dist2_ee(surface_.verts[ea.x], surface_.verts[ea.y],
                               surface_.verts[eb.x], surface_.verts[eb.y]);
            if (D >= D_hat) continue;
            ContactPair cp;
            cp.type = ContactType::EE;
            cp.v[0] = ea.x; cp.v[1] = ea.y; cp.v[2] = eb.x; cp.v[3] = eb.y;
            cp.body[0] = ba0; cp.body[1] = ba1; cp.body[2] = bb0; cp.body[3] = bb1;
            cp.D = D;
            out.push_back(cp);
        }
        return out;
    }

    // Unified contact detection: dispatches to LBVH simplex filter,
    // EFContactDetector, or brute-force.
    std::vector<ContactPair> detect_all_contacts() {
        if (config.use_lbvh_detection) {
            return detect_all_contacts_lbvh();
        }
        if (config.use_bvh_broadphase) {
            if (!bvh_initialized_) {
                int nv = static_cast<int>(surface_.verts.size());
                collision::BroadphaseBackend backend = collision::BroadphaseBackend::QuantBvh;
                if (config.broadphase_type == ABDConfig::BroadphaseType::OptiX)
                    backend = collision::BroadphaseBackend::OptiX;
                ef_detector_.setup(surface_.tris, nv, -1, backend);
                bvh_initialized_ = true;
            }

            // EF broadphase + narrow phase (VF + EE with distance filter)
            std::vector<collision::ContactResult> raw;
            int nv = static_cast<int>(surface_.verts.size());
            ef_detector_.detect(surface_.verts.data(), nv, config.d_hat, raw);

            // Convert to ABD ContactPairs with cross-body filtering
            return convert_contacts(raw, surface_);
        }
        return detect_contacts(surface_, config.d_hat);
    }

    // Update surface vertex positions from body q states.
    void update_surface_verts() {
        for (int b = 0; b < static_cast<int>(bodies_.size()); ++b) {
            auto& range = surface_.body_ranges[b];
            for (int vi = range.vert_begin; vi < range.vert_end; ++vi)
                surface_.verts[vi] = surface_.jacobi[vi].mul_q(bodies_[b].q);
        }
    }

    // Motor metric M_scaled = body mass matrix with the affine (rotation)
    // block scaled by the motor strength and the translation block zeroed
    // (libuipc RotatingMotor uses strength_ratio = {0, strength}).
    Mat12f motor_metric(const ABDBody& b) const {
        Mat12f Ms = Mat12f::zero();
        for (int r = 3; r < 12; ++r)
            for (int c = 3; c < 12; ++c)
                Ms(r, c) = b.motor_strength * b.M(r, c);
        return Ms;
    }

    // Compute total energy for all bodies at current q, re-detecting contacts.
    float total_energy() {
        float E = 0;
        for (auto& body : bodies_) {
            if (body.is_fixed) continue;
            E += assemble_body_energy(body, config.dt);
            if (body.is_motor) {
                Mat12f Ms = motor_metric(body);
                Vec12f dq = body.q - body.motor_q_aim;
                E += 0.5f * math::dot(dq, Ms * dq);
            }
        }
        auto contacts = detect_all_contacts();
        float D_hat = config.d_hat * config.d_hat;
        for (auto& cp : contacts) {
            float D = 0;
            if (cp.type == ContactType::PT)
                D = dist2_pt(surface_.verts[cp.v[0]], surface_.verts[cp.v[1]],
                             surface_.verts[cp.v[2]], surface_.verts[cp.v[3]]);
            else if (cp.type == ContactType::EE)
                D = dist2_ee(surface_.verts[cp.v[0]], surface_.verts[cp.v[1]],
                             surface_.verts[cp.v[2]], surface_.verts[cp.v[3]]);
            E += kappa_ * barrier(D, D_hat);
        }
        return E;
    }

    // Dispatch CCD to the GPU candidate+CCD path or the CPU brute force.
    float compute_toi(const std::vector<Vec3f>& verts_cur,
                      const std::vector<Vec3f>& verts_next,
                      float d_min_override = -1.0f) {
        if (config.use_gpu_ccd) {
            float d_min = d_min_override > 0 ? d_min_override : (config.d_hat * 0.5f);
            return compute_toi_gpu(verts_cur, verts_next, d_min);
        }
        return compute_toi_cpu(verts_cur, verts_next, d_min_override);
    }

    // Brute-force CCD across all PT/EE pairs between different bodies.
    // Unlike detect_contacts (which filters by d_hat), this checks ALL
    // inter-body primitive pairs to prevent tunneling.
    float compute_toi_cpu(const std::vector<Vec3f>& verts_cur,
                          const std::vector<Vec3f>& verts_next,
                          float d_min_override = -1.0f) {
        float alpha = 1.0f;
        float d_min = d_min_override > 0 ? d_min_override : (config.d_hat * 0.5f);
        int nv = static_cast<int>(verts_cur.size());
        int nt = static_cast<int>(surface_.tris.size());
        int ne = static_cast<int>(surface_.edges.size());

        // PT pairs
        for (int vi = 0; vi < nv; ++vi) {
            int body_v = surface_.vert_body[vi];
            for (int ti = 0; ti < nt; ++ti) {
                const Vec3i& tri = surface_.tris[ti];
                if (body_v == surface_.vert_body[tri.x] &&
                    body_v == surface_.vert_body[tri.y] &&
                    body_v == surface_.vert_body[tri.z])
                    continue;
                if (vi == tri.x || vi == tri.y || vi == tri.z) continue;

                Vec3f dp  = verts_next[vi]    - verts_cur[vi];
                Vec3f dt0 = verts_next[tri.x] - verts_cur[tri.x];
                Vec3f dt1 = verts_next[tri.y] - verts_cur[tri.y];
                Vec3f dt2 = verts_next[tri.z] - verts_cur[tri.z];

                float a = ccd_pt(verts_cur[vi], dp,
                                  verts_cur[tri.x], dt0,
                                  verts_cur[tri.y], dt1,
                                  verts_cur[tri.z], dt2,
                                  d_min, 20);
                if (a < alpha) alpha = a;
            }
        }

        // EE pairs
        for (int i = 0; i < ne; ++i) {
            int a0 = surface_.edges[i].x, a1 = surface_.edges[i].y;
            int ba0 = surface_.vert_body[a0], ba1 = surface_.vert_body[a1];
            for (int j = i + 1; j < ne; ++j) {
                int b0 = surface_.edges[j].x, b1 = surface_.edges[j].y;
                int bb0 = surface_.vert_body[b0], bb1 = surface_.vert_body[b1];
                if (ba0 == bb0 && ba0 == bb1 && ba1 == bb0 && ba1 == bb1)
                    continue;
                if (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1) continue;

                Vec3f da0 = verts_next[a0] - verts_cur[a0];
                Vec3f da1 = verts_next[a1] - verts_cur[a1];
                Vec3f db0 = verts_next[b0] - verts_cur[b0];
                Vec3f db1 = verts_next[b1] - verts_cur[b1];

                float a = ccd_ee(verts_cur[a0], da0,
                                  verts_cur[a1], da1,
                                  verts_cur[b0], db0,
                                  verts_cur[b1], db1,
                                  d_min, 20);
                if (a < alpha) alpha = a;
            }
        }
        return alpha;
    }

    void newton_solve() {
        int n = static_cast<int>(bodies_.size());

        // Step 0: initialize q to q_tilde (the predicted position)
        // and then immediately clamp with CCD against the pre-step state.
        std::vector<Vec3f> verts_pre(surface_.verts);  // pre-step positions
        for (int i = 0; i < n; ++i) {
            if (bodies_[i].is_fixed) continue;
            bodies_[i].q = bodies_[i].q_tilde;
        }
        update_surface_verts();

        // CCD: clamp the predictor step so we don't tunnel
        float toi = compute_toi(verts_pre, surface_.verts);
        if (toi < 1.0f) {
            // Interpolate: q = q_prev_step + toi * (q_tilde - q_prev_step)
            // where q_prev_step was stored in q_prev at start of step()
            for (int i = 0; i < n; ++i) {
                if (bodies_[i].is_fixed) continue;
                Vec12f q_start = bodies_[i].q_prev;
                bodies_[i].q = q_start + (bodies_[i].q_tilde - q_start) * toi;
                // Also update q_tilde so kinetic target is feasible
                bodies_[i].q_tilde = bodies_[i].q;
            }
            update_surface_verts();
        }

        for (int iter = 0; iter < config.newton_max_iter; ++iter) {
            // Update surface
            update_surface_verts();

            auto contacts = detect_all_contacts();
            last_info_.contact_count = static_cast<int>(contacts.size());

            // Assemble system
            BlockSystem sys;
            sys.resize(n);

            // Per-body: kinetic + shape
            for (int i = 0; i < n; ++i) {
                if (bodies_[i].is_fixed) {
                    sys.H_diag[i] = Mat12f::identity() * 1e10f;
                    sys.rhs[i] = Vec12f::zero();
                    continue;
                }
                sys.H_diag[i] = assemble_body_hessian(bodies_[i], config.dt);
                sys.rhs[i] = assemble_body_gradient(bodies_[i], config.dt) * (-1.0f);

                // RotatingMotor soft constraint (libuipc soft_transform_constraint):
                // E = 0.5 * dq^T (strength * M_affine) dq, dq = q - q_aim.
                if (bodies_[i].is_motor) {
                    Mat12f Ms = motor_metric(bodies_[i]);
                    Vec12f dq = bodies_[i].q - bodies_[i].motor_q_aim;
                    sys.rhs[i] = sys.rhs[i] - Ms * dq;   // rhs = -gradient
                    sys.H_diag[i] = sys.H_diag[i] + Ms;
                }
            }

            // Contact contributions
            float D_hat = config.d_hat * config.d_hat;
            for (auto& cp : contacts) {
                if (cp.type == ContactType::PT) {
                    assemble_pt_contact(cp, sys, D_hat);
                } else if (cp.type == ContactType::EE) {
                    assemble_ee_contact(cp, sys, D_hat);
                }
            }

            // Solve H * dq = rhs (rhs = -gradient)
            std::vector<Vec12f> dq(n);
            auto pcg_result = solve_pcg(sys, dq, config.pcg_tol,
                                         n * 12 * config.pcg_max_iter_ratio);

            // Check convergence: max velocity change
            float max_dq = 0;
            for (int i = 0; i < n; ++i) {
                if (bodies_[i].is_fixed) continue;
                float m = math::max_abs(dq[i]);
                if (m > max_dq) max_dq = m;
            }

            last_info_.newton_iters = iter + 1;
            last_info_.residual = max_dq;

            if (iter >= config.newton_min_iter && max_dq < config.velocity_tol * config.dt) {
                break;
            }

            // CCD step size: check ALL inter-body pairs, not just active contacts
            std::vector<Vec3f> verts_trial(surface_.verts.size());
            for (int b = 0; b < n; ++b) {
                auto& range = surface_.body_ranges[b];
                Vec12f q_trial = bodies_[b].q + dq[b];
                for (int vi = range.vert_begin; vi < range.vert_end; ++vi)
                    verts_trial[vi] = surface_.jacobi[vi].mul_q(q_trial);
            }

            float alpha_ccd = compute_toi(surface_.verts, verts_trial, config.d_hat * 0.1f);

            // Energy-based line search
            float E0 = total_energy();
            float dir_deriv = 0;
            for (int i = 0; i < n; ++i)
                dir_deriv += math::dot(sys.rhs[i] * (-1.0f), dq[i]);

            // Save current q for line search rollback
            std::vector<Vec12f> q_save(n);
            for (int i = 0; i < n; ++i) q_save[i] = bodies_[i].q;

            float alpha = alpha_ccd;
            for (int ls = 0; ls < config.line_search_max_iter; ++ls) {
                for (int i = 0; i < n; ++i) {
                    if (bodies_[i].is_fixed) continue;
                    bodies_[i].q = q_save[i] + dq[i] * alpha;
                }
                update_surface_verts();
                float E_trial = total_energy();

                if (E_trial <= E0 + 1e-4f * alpha * dir_deriv) break;
                alpha *= 0.5f;
            }

            // Apply the accepted step
            for (int i = 0; i < n; ++i) {
                if (bodies_[i].is_fixed) continue;
                bodies_[i].q = q_save[i] + dq[i] * alpha;
            }
        }
    }

    // Assemble a PT contact pair into the block system.
    void assemble_pt_contact(const ContactPair& cp, BlockSystem& sys,
                              float D_hat) {
        int vi_p = cp.v[0], vi_t0 = cp.v[1], vi_t1 = cp.v[2], vi_t2 = cp.v[3];
        Vec3f p  = surface_.verts[vi_p];
        Vec3f t0 = surface_.verts[vi_t0];
        Vec3f t1 = surface_.verts[vi_t1];
        Vec3f t2 = surface_.verts[vi_t2];

        float D = dist2_pt(p, t0, t1, t2);
        if (D >= D_hat) return;

        float dBdD  = barrier_gradient(D, D_hat);
        float d2BdD = barrier_hessian_spd(D, D_hat);
        float kappa = kappa_;

        // PT gradient
        Vec3f gp, gt0, gt1, gt2;
        dist2_pt_grad(p, t0, t1, t2, gp, gt0, gt1, gt2);

        // Map vertices to bodies and accumulate
        struct VInfo { int vi; Vec3f grad; int body; };
        VInfo vinf[4] = {
            {vi_p,  gp,  cp.body[0]},
            {vi_t0, gt0, cp.body[1]},
            {vi_t1, gt1, cp.body[2]},
            {vi_t2, gt2, cp.body[3]},
        };

        for (auto& v : vinf) {
            Vec12f g12 = lift_contact_gradient(surface_.jacobi[v.vi], v.grad,
                                               kappa, dBdD);
            sys.rhs[v.body] = sys.rhs[v.body] - g12;  // rhs = -gradient

            // Diagonal Hessian (rank-1 + identity-scaled)
            Vec12f jt_g = surface_.jacobi[v.vi].mul_JT(v.grad);
            for (int r = 0; r < 12; ++r)
                for (int c = 0; c < 12; ++c)
                    sys.H_diag[v.body](r, c) += kappa * d2BdD * jt_g[r] * jt_g[c];
        }
    }

    // Assemble an EE contact pair into the block system.
    void assemble_ee_contact(const ContactPair& cp, BlockSystem& sys,
                              float D_hat) {
        int a0 = cp.v[0], a1 = cp.v[1], b0 = cp.v[2], b1 = cp.v[3];
        Vec3f va0 = surface_.verts[a0], va1 = surface_.verts[a1];
        Vec3f vb0 = surface_.verts[b0], vb1 = surface_.verts[b1];

        float D = dist2_ee(va0, va1, vb0, vb1);
        if (D >= D_hat) return;

        float dBdD  = barrier_gradient(D, D_hat);
        float d2BdD = barrier_hessian_spd(D, D_hat);
        float kappa = kappa_;

        Vec3f ga0, ga1, gb0, gb1;
        dist2_ee_grad(va0, va1, vb0, vb1, ga0, ga1, gb0, gb1);

        struct VInfo { int vi; Vec3f grad; int body; };
        VInfo vinf[4] = {
            {a0, ga0, cp.body[0]},
            {a1, ga1, cp.body[1]},
            {b0, gb0, cp.body[2]},
            {b1, gb1, cp.body[3]},
        };

        for (auto& v : vinf) {
            Vec12f g12 = lift_contact_gradient(surface_.jacobi[v.vi], v.grad,
                                               kappa, dBdD);
            sys.rhs[v.body] = sys.rhs[v.body] - g12;

            Vec12f jt_g = surface_.jacobi[v.vi].mul_JT(v.grad);
            for (int r = 0; r < 12; ++r)
                for (int c = 0; c < 12; ++c)
                    sys.H_diag[v.body](r, c) += kappa * d2BdD * jt_g[r] * jt_g[c];
        }
    }
};

}  // namespace abd_ipc
}  // namespace chysx
