// SPDX-License-Identifier: MIT
// Linear system assembly for the ABD IPC solver.
//
// Three energy terms are combined per Newton step:
//   E_total = E_kinetic + E_shape + E_contact
//
// E_kinetic = 1/2 (q − q̃)^T M (q − q̃)
// E_shape   = κ V Ψ(A)
// E_contact = κ_c Σ_k B(D_k)
//
// The per-body gradient/Hessian from kinetic+shape are diagonal blocks.
// Contact pairs couple two bodies via J^T H_contact J.

#pragma once

#include "abd_ipc_types.cuh"
#include "abd_ipc_energy.cuh"
#include "abd_ipc_barrier.cuh"
#include "abd_ipc_distance.cuh"

namespace chysx {
namespace abd_ipc {

// ============================================================================
// Per-body assembly: kinetic + shape (no contact)
// ============================================================================

// Gradient contribution: M(q − q̃) + ∇E_shape
// Vdt2 = volume * dt² (time-step scaling for shape energy)
CHYSX_HDI Vec12f assemble_body_gradient(const ABDBody& body, float dt) {
    Vec12f diff = body.q - body.q_tilde;
    Vec12f g_kin = body.M * diff;
    float Vdt2 = body.volume * dt * dt;
    Vec12f g_shape = ortho_gradient(body.q, body.kappa, Vdt2);
    return g_kin + g_shape;
}

// Hessian contribution: M + ∇²E_shape
CHYSX_HDI Mat12f assemble_body_hessian(const ABDBody& body, float dt) {
    Mat12f H_kin = body.M;
    float Vdt2 = body.volume * dt * dt;
    Mat12f H_shape = ortho_hessian_12(body.q, body.kappa, Vdt2, true);
    return H_kin + H_shape;
}

// Total body energy: kinetic + shape
CHYSX_HDI float assemble_body_energy(const ABDBody& body, float dt) {
    Vec12f diff = body.q - body.q_tilde;
    float E_kin = 0.5f * math::dot(diff, body.M * diff);
    float Vdt2 = body.volume * dt * dt;
    float E_shape = ortho_energy(body.q, body.kappa, Vdt2);
    return E_kin + E_shape;
}

// ============================================================================
// Contact contribution: lift vertex-level ∂D/∂x to body-level ∂E/∂q
// ============================================================================

// Single-vertex contact gradient lifted to body:
//   ∂E/∂q_body += κ · dB/dD · J_v^T · ∂D/∂x_v
CHYSX_HDI Vec12f lift_contact_gradient(const ABDJacobi& J, Vec3f dD_dx,
                                        float kappa, float dBdD) {
    return J.mul_JT(dD_dx * (kappa * dBdD));
}

// Single-vertex pair Hessian block lifted to body:
//   ∂²E/∂q_i∂q_j += κ · [ d²B/dD² · (J_i^T ∂D/∂x_i)(J_j^T ∂D/∂x_j)^T
//                         + dB/dD  · J_i^T ∂²D/∂x_i∂x_j J_j ]
// For the rank-1 outer part, we use gg^T directly in q-space.
CHYSX_HDI Mat12f lift_contact_hessian_rank1(const ABDJacobi& Ji,
                                             const ABDJacobi& Jj,
                                             Vec3f dDdx_i, Vec3f dDdx_j,
                                             float kappa, float d2BdD2) {
    Vec12f gi = Ji.mul_JT(dDdx_i);
    Vec12f gj = Jj.mul_JT(dDdx_j);
    // outer product gi * gj^T
    Mat12f R = Mat12f::zero();
    float coeff = kappa * d2BdD2;
    for (int r = 0; r < 12; ++r)
        for (int c = 0; c < 12; ++c)
            R(r, c) = coeff * gi[r] * gj[c];
    return R;
}

// Full contact Hessian block from a 3x3 spatial Hessian ∂²D/∂x_i∂x_j
CHYSX_HDI Mat12f lift_contact_hessian_spatial(const ABDJacobi& Ji,
                                               const ABDJacobi& Jj,
                                               const Mat3f& d2D_dxidxj,
                                               float kappa, float dBdD) {
    return ABDJacobi::JT_H_J(Ji, d2D_dxidxj * (kappa * dBdD), Jj);
}

// ============================================================================
// PP contact: accumulate into body gradient and Hessian
// ============================================================================

struct ContactGradHess {
    Vec12f g[2];    // per-body gradient contributions
    Mat12f H[4];    // H[0]=diag_0, H[1]=off_01, H[2]=off_10, H[3]=diag_1
};

CHYSX_HDI ContactGradHess assemble_pp_contact(
        Vec3f p0, Vec3f p1,
        const ABDJacobi& J0, const ABDJacobi& J1,
        int body0, int body1,
        float kappa, float d_hat) {
    ContactGradHess cgh;
    float D = dist2_pp(p0, p1);
    float D_hat = d_hat * d_hat;
    if (D >= D_hat) {
        for (int i = 0; i < 2; ++i) cgh.g[i] = Vec12f::zero();
        for (int i = 0; i < 4; ++i) cgh.H[i] = Mat12f::zero();
        return cgh;
    }

    float dBdD  = barrier_gradient(D, D_hat);
    float d2BdD = barrier_hessian_spd(D, D_hat);

    Vec3f g0, g1;
    dist2_pp_grad(p0, p1, g0, g1);

    cgh.g[0] = lift_contact_gradient(J0, g0, kappa, dBdD);
    cgh.g[1] = lift_contact_gradient(J1, g1, kappa, dBdD);

    // Hessian: rank-1 (barrier curvature) + spatial (distance curvature)
    Mat3f H00, H01, H10, H11;
    dist2_pp_hess(H00, H01, H10, H11);

    cgh.H[0] = lift_contact_hessian_rank1(J0, J0, g0, g0, kappa, d2BdD)
             + lift_contact_hessian_spatial(J0, J0, H00, kappa, dBdD);
    cgh.H[1] = lift_contact_hessian_rank1(J0, J1, g0, g1, kappa, d2BdD)
             + lift_contact_hessian_spatial(J0, J1, H01, kappa, dBdD);
    cgh.H[2] = lift_contact_hessian_rank1(J1, J0, g1, g0, kappa, d2BdD)
             + lift_contact_hessian_spatial(J1, J0, H10, kappa, dBdD);
    cgh.H[3] = lift_contact_hessian_rank1(J1, J1, g1, g1, kappa, d2BdD)
             + lift_contact_hessian_spatial(J1, J1, H11, kappa, dBdD);
    return cgh;
}

}  // namespace abd_ipc
}  // namespace chysx
