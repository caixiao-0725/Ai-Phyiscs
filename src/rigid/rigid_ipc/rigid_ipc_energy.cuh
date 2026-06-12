// SPDX-License-Identifier: MIT
// Rigid-body inertia energy for implicit Euler time integration.
//
// Translational:
//   E_trans = 0.5 * m * p^T p - m * p^T * p_hat
//   p_hat = p_prev + dt * v_prev + dt^2 * gravity
//
// Rotational (matrix form, from rigid-ipc / Brüel-Gabrielsson 2021):
//   J = compute_J(I) = diag(J1, J2, J3)
//   E_rot = 0.5 * tr(Q*J*Q^T) - tr(Q*J*Q_target^T)
//   Q_target = Q_prev + dt * Qdot_prev
//   (Torque terms and Newmark extensions are omitted for simplicity.)
//
// All gradient/Hessian computations use the analytic Rodrigues derivatives
// from rigid_ipc_rodrigues.cuh.

#pragma once

#include "rigid_ipc_types.cuh"
#include "rigid_ipc_rodrigues.cuh"
#include "rigid_ipc_mass.cuh"

namespace chysx {
namespace rigid_ipc {

// ============================================================================
// Translational energy — trivially quadratic in p
// ============================================================================

CHYSX_HDI float energy_translation(Vec3f p, Vec3f p_hat, float mass) {
    return 0.5f * mass * math::dot(p, p) - mass * math::dot(p, p_hat);
}

CHYSX_HDI Vec3f grad_translation(Vec3f p, Vec3f p_hat, float mass) {
    return (p - p_hat) * mass;
}

// Hessian of translational energy = m * I_3 (constant)

// ============================================================================
// Rotational energy — through R(theta) matrix trace formulation
//
// E_rot = 0.5 * tr(Q*J*Q^T) - tr(Q*J*Q_target^T)
//
// where Q = R(theta), J = diag(J1,J2,J3), Q_target is given.
//
// dE/dtheta_k = 0.5 * tr(dQ/dk * J * Q^T + Q * J * dQ/dk^T)
//             - tr(dQ/dk * J * Q_target^T)
//
// d^2E/(dtheta_k dtheta_l) computed similarly via d^2R/dk dl.
// ============================================================================

CHYSX_HDI float energy_rotation(Vec3f theta, Vec3f J_diag,
                                 const Mat3f& Q_target) {
    Mat3f Q = rodrigues(theta);
    Mat3f J = Mat3f(J_diag.x, 0, 0,  0, J_diag.y, 0,  0, 0, J_diag.z);

    Mat3f QJ = Q * J;
    float E = 0.5f * math::trace(QJ * transpose(Q))
            - math::trace(QJ * transpose(Q_target));
    return E;
}

CHYSX_HDI Vec3f grad_rotation(Vec3f theta, Vec3f J_diag,
                               const Mat3f& Q_target) {
    Mat3f Q = rodrigues(theta);
    Mat3f J = Mat3f(J_diag.x, 0, 0,  0, J_diag.y, 0,  0, 0, J_diag.z);

    Vec3f g;
    for (int k = 0; k < 3; ++k) {
        Mat3f dQk = dR_dr(theta, k);
        Mat3f dQk_J = dQk * J;
        // dE/dk = 0.5*(tr(dQk*J*Q^T) + tr(Q*J*dQk^T)) - tr(dQk*J*Qt^T)
        // By symmetry of trace: tr(A*B^T) = tr(B*A^T), so the first two
        // terms combine to tr(dQk * J * Q^T)
        g[k] = math::trace(dQk_J * transpose(Q))
             - math::trace(dQk_J * transpose(Q_target));
    }
    return g;
}

CHYSX_HDI Mat3f hessian_rotation(Vec3f theta, Vec3f J_diag,
                                  const Mat3f& Q_target) {
    Mat3f Q = rodrigues(theta);
    Mat3f J = Mat3f(J_diag.x, 0, 0,  0, J_diag.y, 0,  0, 0, J_diag.z);

    // Precompute dR/dk
    Mat3f dQ[3];
    for (int k = 0; k < 3; ++k) dQ[k] = dR_dr(theta, k);

    Mat3f H;
    for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
            Mat3f d2Qkl = d2R_dr2(theta, k, l);
            Mat3f d2Qkl_J = d2Qkl * J;
            Mat3f dQk_J = dQ[k] * J;
            Mat3f dQl_J = dQ[l] * J;

            // d^2E/dk dl = tr(d^2Qkl*J*Q^T) + tr(dQk*J*dQl^T) - tr(d^2Qkl*J*Qt^T)
            float val = math::trace(d2Qkl_J * transpose(Q))
                      + math::trace(dQk_J * transpose(dQ[l]))
                      - math::trace(d2Qkl_J * transpose(Q_target));
            H(k, l) = val;
        }
    }
    return H;
}

// ============================================================================
// Combined body energy (translational + rotational)
// ============================================================================

// Compute translational prediction target
CHYSX_HDI Vec3f compute_p_hat(const RigidBody& body, float dt, Vec3f gravity) {
    Vec3f v_prev = pose_position(body.q_v);
    Vec3f p_prev = pose_position(body.q_prev);
    return p_prev + v_prev * dt + gravity * (dt * dt);
}

// Compute rotational prediction target matrix: Q_target = Q_prev + dt * Qdot_prev
CHYSX_HDI Mat3f compute_Q_target(const RigidBody& body, float dt) {
    Mat3f Q_target = body.R_prev;
    for (int i = 0; i < 9; ++i)
        Q_target.data[i] += body.Qdot_prev.data[i] * dt;
    return Q_target;
}

CHYSX_HDI float body_energy(const RigidBody& body, float dt, Vec3f gravity) {
    Vec3f p = pose_position(body.q);
    Vec3f theta = pose_rotation(body.q);
    Vec3f p_hat = compute_p_hat(body, dt, gravity);
    Vec3f J_diag = compute_J(body.moment_of_inertia);
    Mat3f Q_target = compute_Q_target(body, dt);

    float E_trans = energy_translation(p, p_hat, body.mass);
    float E_rot = energy_rotation(theta, J_diag, Q_target);
    return E_trans + E_rot;
}

CHYSX_HDI Vec6f body_gradient(const RigidBody& body, float dt, Vec3f gravity) {
    Vec3f p = pose_position(body.q);
    Vec3f theta = pose_rotation(body.q);
    Vec3f p_hat = compute_p_hat(body, dt, gravity);
    Vec3f J_diag = compute_J(body.moment_of_inertia);
    Mat3f Q_target = compute_Q_target(body, dt);

    Vec3f g_trans = grad_translation(p, p_hat, body.mass);
    Vec3f g_rot = grad_rotation(theta, J_diag, Q_target);

    Vec6f g;
    g[0] = g_trans.x; g[1] = g_trans.y; g[2] = g_trans.z;
    g[3] = g_rot.x;   g[4] = g_rot.y;   g[5] = g_rot.z;
    return g;
}

CHYSX_HDI Mat6f body_hessian(const RigidBody& body, float dt, Vec3f gravity) {
    Vec3f theta = pose_rotation(body.q);
    Vec3f J_diag = compute_J(body.moment_of_inertia);
    Mat3f Q_target = compute_Q_target(body, dt);

    Mat6f H = Mat6f::zero();

    // Translation block: m * I_3
    H(0, 0) = body.mass;
    H(1, 1) = body.mass;
    H(2, 2) = body.mass;

    // Rotation block
    Mat3f H_rot = hessian_rotation(theta, J_diag, Q_target);
    for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
            H(3 + k, 3 + l) = H_rot(k, l);

    // SPD projection of the rotation block
    math::make_spd(H);

    return H;
}

}  // namespace rigid_ipc
}  // namespace chysx
