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
#include "../../geometry/tet_mesh.h"

#include <cstdio>
#include <string>
#include <vector>

namespace chysx {
namespace abd_ipc {

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

    // ---- simulation -------------------------------------------------------

    // Advance one time step.
    void step() {
        int n = static_cast<int>(bodies_.size());
        float dt = config.dt;
        float inv_dt = 1.0f / dt;

        // 0. Save q_prev at start of step (used by CCD predictor clamping)
        for (int i = 0; i < n; ++i)
            bodies_[i].q_prev = bodies_[i].q;

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
        newton_solve();

        // 4. Velocity update: q_v = (q_new - q_old) / dt
        for (int i = 0; i < n; ++i) {
            if (bodies_[i].is_fixed) continue;
            bodies_[i].q_v = (bodies_[i].q - bodies_[i].q_prev) * inv_dt;
            bodies_[i].q_prev = bodies_[i].q;
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

    // Update surface vertex positions from body q states.
    void update_surface_verts() {
        for (int b = 0; b < static_cast<int>(bodies_.size()); ++b) {
            auto& range = surface_.body_ranges[b];
            for (int vi = range.vert_begin; vi < range.vert_end; ++vi)
                surface_.verts[vi] = surface_.jacobi[vi].mul_q(bodies_[b].q);
        }
    }

    // Compute total energy for all bodies at current q, re-detecting contacts.
    float total_energy() {
        float E = 0;
        for (auto& body : bodies_) {
            if (body.is_fixed) continue;
            E += assemble_body_energy(body, config.dt);
        }
        // Re-detect contacts at current surface positions
        auto contacts = detect_contacts(surface_, config.d_hat);
        float D_hat = config.d_hat * config.d_hat;
        for (auto& cp : contacts) {
            float D = 0;
            if (cp.type == ContactType::PT)
                D = dist2_pt(surface_.verts[cp.v[0]], surface_.verts[cp.v[1]],
                             surface_.verts[cp.v[2]], surface_.verts[cp.v[3]]);
            else if (cp.type == ContactType::EE)
                D = dist2_ee(surface_.verts[cp.v[0]], surface_.verts[cp.v[1]],
                             surface_.verts[cp.v[2]], surface_.verts[cp.v[3]]);
            E += config.contact_kappa * barrier(D, D_hat);
        }
        return E;
    }

    // Brute-force CCD across all PT/EE pairs between different bodies.
    // Unlike detect_contacts (which filters by d_hat), this checks ALL
    // inter-body primitive pairs to prevent tunneling.
    float compute_toi(const std::vector<Vec3f>& verts_cur,
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

            // Detect contacts
            auto contacts = detect_contacts(surface_, config.d_hat);
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
        float kappa = config.contact_kappa;

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
        float kappa = config.contact_kappa;

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
