// SPDX-License-Identifier: MIT
// Linear system assembly for the Rigid-IPC solver.
//
// Two energy terms per Newton step:
//   E_total = E_body (inertia) + E_contact (IPC barrier)
//
// (No shape energy needed — rigid bodies preserve shape by construction.)
//
// Contact pairs couple two bodies via the chain rule:
//   grad_q = J_V^T * grad_V
//   H_q = J_V^T * H_V * J_V + sum_i (d^2V_i/dq^2 * grad_V_i)

#pragma once

#include "rigid_ipc_types.cuh"
#include "rigid_ipc_rodrigues.cuh"
#include "rigid_ipc_energy.cuh"
#include "../../rigid/abd_ipc/abd_ipc_barrier.cuh"
#include "../../rigid/abd_ipc/abd_ipc_distance.cuh"

namespace chysx {
namespace rigid_ipc {

// Re-export barrier/distance from abd_ipc (DOF-agnostic)
using abd_ipc::barrier;
using abd_ipc::barrier_gradient;
using abd_ipc::barrier_hessian;
using abd_ipc::barrier_hessian_spd;
using abd_ipc::dist2_pp;
using abd_ipc::dist2_pe;
using abd_ipc::dist2_pt;
using abd_ipc::dist2_ee;
using abd_ipc::dist2_pt_grad;
using abd_ipc::dist2_ee_grad;
using abd_ipc::dist2_pp_grad;

// ============================================================================
// Contact lifting: vertex-space -> 6-DOF body space
// ============================================================================

// Single-vertex contact gradient lifted to body:
//   dE/dq += kappa * dB/dD * J_v^T * dD/dx_v
CHYSX_HDI Vec6f lift_contact_gradient(const RigidJacobi& J, Vec3f dD_dx,
                                       const Vec6f& q, float kappa, float dBdD) {
    return J.mul_JT(dD_dx * (kappa * dBdD), q);
}

// Single-vertex diagonal Hessian contribution (rank-1 in q-space):
//   H_diag[body] += kappa * d^2B/dD^2 * (J^T dD/dx)(J^T dD/dx)^T
CHYSX_HDI Mat6f lift_contact_hessian_rank1_diag(const RigidJacobi& J,
                                                  Vec3f dD_dx, const Vec6f& q,
                                                  float kappa, float d2BdD2) {
    Vec6f jt_g = J.mul_JT(dD_dx, q);
    Mat6f H = Mat6f::zero();
    float coeff = kappa * d2BdD2;
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
            H(r, c) = coeff * jt_g[r] * jt_g[c];
    return H;
}

// Full Hessian block including spatial curvature + vertex Hessian correction:
//   H_body_ij = kappa * [d^2B/dD^2 * gi * gj^T
//              + dB/dD * Ji^T * d^2D/dxi dxj * Jj]
//              + vertex_hessian_correction terms (from d^2R/dtheta^2)
CHYSX_HDI Mat6f lift_contact_hessian_full(const RigidJacobi& Ji,
                                            const RigidJacobi& Jj,
                                            Vec3f dDdx_i, Vec3f dDdx_j,
                                            const Mat3f& d2D_dxidxj,
                                            Vec3f grad_V_i,
                                            const Vec6f& qi, const Vec6f& qj,
                                            float kappa, float dBdD, float d2BdD2) {
    // Rank-1 part
    Vec6f gi = Ji.mul_JT(dDdx_i, qi);
    Vec6f gj = Jj.mul_JT(dDdx_j, qj);
    Mat6f H = Mat6f::zero();
    float coeff_rank1 = kappa * d2BdD2;
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
            H(r, c) = coeff_rank1 * gi[r] * gj[c];

    // Spatial curvature part: Ji^T * d2D * Jj
    Mat6f H_spatial = RigidJacobi::JT_H_J(Ji, d2D_dxidxj * (kappa * dBdD), Jj, qi);
    H = H + H_spatial;

    // Vertex Hessian correction (only for same-body diagonal blocks)
    // The correction sum_j (d^2V_ij/dq^2 * grad_V[j]) applies to vertex Vi
    // Here grad_V_i is the full barrier gradient at vertex i: kappa * dBdD * dD/dx_i
    // We only add correction for the diagonal block of body i.
    H = H + Ji.vertex_hessian_correction(grad_V_i * (kappa * dBdD), qi);

    return H;
}

// ============================================================================
// Simplified per-vertex diagonal assembly (used by solver)
// ============================================================================

// Accumulate a single vertex's contact contribution to the block system.
// Only diagonal blocks (same body) are computed — matches ABD solver pattern.
CHYSX_HDI void accumulate_vertex_contact(
    const RigidJacobi& J, Vec3f dD_dx, int body_idx,
    const Vec6f& q, float kappa, float dBdD, float d2BdD2,
    Vec6f* rhs, Mat6f* H_diag) {

    Vec6f g = lift_contact_gradient(J, dD_dx, q, kappa, dBdD);
    rhs[body_idx] = rhs[body_idx] - g;

    Mat6f H = lift_contact_hessian_rank1_diag(J, dD_dx, q, kappa, d2BdD2);
    H_diag[body_idx] = H_diag[body_idx] + H;
}

}  // namespace rigid_ipc
}  // namespace chysx
