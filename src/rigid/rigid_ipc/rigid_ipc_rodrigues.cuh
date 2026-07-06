// SPDX-License-Identifier: MIT
// Rodrigues rotation: rotation vector -> rotation matrix, with analytic
// first and second derivatives for GPU-friendly rigid-body IPC.
//
// R(r) = I + sinc(||r||) * K + half_sinc_sq(||r||) * K^2
// where K = skew(r), sinc(x) = sin(x)/x, half_sinc_sq(x) = (1-cos(x))/x^2
//
// The derivatives dR/dr_k and d^2R/(dr_k dr_l) are derived analytically
// from the Rodrigues formula. Taylor expansions are used near ||r|| = 0
// for numerical stability.

#pragma once

#include "rigid_ipc_types.cuh"
#include "../../math/quat.cuh"  // skew()

#include <cmath>

namespace chysx {
namespace rigid_ipc {

// ============================================================================
// Scalar helpers with Taylor expansions near zero
// ============================================================================

// sinc(x) = sin(x)/x, with Taylor: 1 - x^2/6 + x^4/120
CHYSX_HDI float safe_sinc(float x) {
    if (fabsf(x) < 1e-4f)
        return 1.0f - x * x * (1.0f / 6.0f) + x * x * x * x * (1.0f / 120.0f);
    return sinf(x) / x;
}

// (1 - cos(x)) / x^2, with Taylor: 1/2 - x^2/24 + x^4/720
CHYSX_HDI float safe_one_minus_cos_over_x2(float x) {
    if (fabsf(x) < 1e-4f)
        return 0.5f - x * x * (1.0f / 24.0f) + x * x * x * x * (1.0f / 720.0f);
    return (1.0f - cosf(x)) / (x * x);
}

// d/dx [sinc(x)] = (x*cos(x) - sin(x)) / x^2
// Taylor: -x/3 + x^3/30 - x^5/840
CHYSX_HDI float d_sinc(float x) {
    if (fabsf(x) < 1e-4f)
        return -x * (1.0f / 3.0f) + x * x * x * (1.0f / 30.0f);
    return (x * cosf(x) - sinf(x)) / (x * x);
}

// d/dx [(1-cos(x))/x^2] = (x*sin(x) - 2*(1-cos(x))) / x^3
// Taylor: -x/12 + x^3/180
CHYSX_HDI float d_one_minus_cos_over_x2(float x) {
    if (fabsf(x) < 1e-4f)
        return -x * (1.0f / 12.0f) + x * x * x * (1.0f / 180.0f);
    return (x * sinf(x) - 2.0f * (1.0f - cosf(x))) / (x * x * x);
}

// d^2/dx^2 [sinc(x)] = ((x^2-2)*sin(x) + 2*x*cos(x)) / x^3
// Taylor: -1/3 + x^2/10
CHYSX_HDI float d2_sinc(float x) {
    if (fabsf(x) < 1e-4f)
        return -1.0f / 3.0f + x * x * (1.0f / 10.0f);
    float s = sinf(x), c = cosf(x);
    return ((x * x - 2.0f) * s + 2.0f * x * c) / (x * x * x);
}

// d^2/dx^2 [(1-cos(x))/x^2]
// Taylor: -1/12 + x^2/60
CHYSX_HDI float d2_one_minus_cos_over_x2(float x) {
    if (fabsf(x) < 1e-4f)
        return -1.0f / 12.0f + x * x * (1.0f / 60.0f);
    float s = sinf(x), c = cosf(x);
    float x2 = x * x, x3 = x2 * x, x4 = x3 * x;
    return ((x2 - 6.0f) * c + (4.0f * x) * s + 6.0f - x2) / x4
           - 2.0f * d_one_minus_cos_over_x2(x) / x;
}

// ============================================================================
// Rodrigues: rotation vector -> rotation matrix
// ============================================================================

CHYSX_HDI Mat3f rodrigues(Vec3f r) {
    float theta = math::length(r);
    float s = safe_sinc(theta);
    float c = safe_one_minus_cos_over_x2(theta);
    Mat3f K = math::skew(r);
    Mat3f K2 = K * K;

    Mat3f R = Mat3f::identity();
    for (int i = 0; i < 9; ++i) R.data[i] += s * K.data[i] + c * K2.data[i];
    return R;
}

// ============================================================================
// Inverse Rodrigues: rotation matrix -> axis-angle vector
// ============================================================================

inline Vec3f rotation_matrix_to_axis_angle(const Mat3f& R) {
    // trace(R) = 1 + 2*cos(theta)
    float tr = R(0,0) + R(1,1) + R(2,2);
    float cos_theta = (tr - 1.0f) * 0.5f;
    cos_theta = std::max(-1.0f, std::min(1.0f, cos_theta));
    float theta = acosf(cos_theta);

    if (theta < 1e-6f) return Vec3f(0, 0, 0);

    // axis from skew-symmetric part: (R - R^T) / (2*sin(theta))
    float s = sinf(theta);
    if (fabsf(s) < 1e-8f) {
        // theta ≈ pi: use eigenvector of (R + I)/2 with largest diagonal
        float rx = sqrtf(std::max(0.0f, (R(0,0) + 1.0f) * 0.5f));
        float ry = sqrtf(std::max(0.0f, (R(1,1) + 1.0f) * 0.5f));
        float rz = sqrtf(std::max(0.0f, (R(2,2) + 1.0f) * 0.5f));
        if (R(0,1) + R(1,0) < 0) ry = -ry;
        if (R(0,2) + R(2,0) < 0) rz = -rz;
        float len = sqrtf(rx*rx + ry*ry + rz*rz);
        if (len > 1e-10f) { rx /= len; ry /= len; rz /= len; }
        return Vec3f(rx * theta, ry * theta, rz * theta);
    }

    float inv_2s = 1.0f / (2.0f * s);
    Vec3f axis;
    axis.x = (R(2,1) - R(1,2)) * inv_2s;
    axis.y = (R(0,2) - R(2,0)) * inv_2s;
    axis.z = (R(1,0) - R(0,1)) * inv_2s;
    return axis * theta;
}

// ============================================================================
// dR/dr_k  (3x3 matrix, k = 0, 1, 2)
//
// R = I + sinc(theta)*K + h(theta)*K^2
// where theta = ||r||, K = skew(r), h = (1-cos)/theta^2
//
// dR/dr_k = sinc'(theta)*(r_k/theta)*K + sinc(theta)*dK/dr_k
//         + h'(theta)*(r_k/theta)*K^2   + h(theta)*(dK/dr_k*K + K*dK/dr_k)
//
// dK/dr_k = skew(e_k) where e_k is the k-th basis vector.
// ============================================================================

CHYSX_HDI Mat3f dK_dr(int k) {
    Vec3f e = Vec3f(0, 0, 0);
    if (k == 0) e.x = 1.0f;
    else if (k == 1) e.y = 1.0f;
    else e.z = 1.0f;
    return math::skew(e);
}

CHYSX_HDI Mat3f dR_dr(Vec3f r, int k) {
    float theta = math::length(r);
    Mat3f K = math::skew(r);
    Mat3f K2 = K * K;
    Mat3f dKk = dK_dr(k);

    float s  = safe_sinc(theta);
    float h  = safe_one_minus_cos_over_x2(theta);

    Mat3f result;
    if (theta < 1e-6f) {
        // Near zero: R ≈ I + K, dR/dr_k ≈ dK/dr_k
        result = dKk;
    } else {
        float ds = d_sinc(theta);
        float dh = d_one_minus_cos_over_x2(theta);
        float rk_over_theta = r[k] / theta;

        Mat3f dKk_K = dKk * K;
        Mat3f K_dKk = K * dKk;
        Mat3f dK2 = dKk_K + K_dKk;  // d(K^2)/dr_k

        result = K * (ds * rk_over_theta) + dKk * s
               + K2 * (dh * rk_over_theta) + dK2 * h;
    }
    return result;
}

// ============================================================================
// d^2R / (dr_k dr_l)  —  3x3 matrix
//
// Used for the vertex Hessian: d^2 x_i / (dtheta_k dtheta_l) = d^2R/(dk dl) * x_bar
// ============================================================================

CHYSX_HDI Mat3f d2R_dr2(Vec3f r, int k, int l) {
    float theta = math::length(r);
    Mat3f K = math::skew(r);
    Mat3f K2 = K * K;
    Mat3f dKk = dK_dr(k);
    Mat3f dKl = dK_dr(l);

    if (theta < 1e-6f) {
        // Near zero: d^2R ≈ d(dK_k)/dr_l terms from K^2 part
        // h ≈ 0.5, dK2/dl = dKl*K + K*dKl, d(dK2_k)/dl has cross terms
        Mat3f dKk_dKl = dKk * dKl;
        Mat3f dKl_dKk = dKl * dKk;
        return (dKk_dKl + dKl_dKk) * 0.5f;
    }

    float s   = safe_sinc(theta);
    float h   = safe_one_minus_cos_over_x2(theta);
    float ds  = d_sinc(theta);
    float dh  = d_one_minus_cos_over_x2(theta);
    float d2s = d2_sinc(theta);
    float d2h = d2_one_minus_cos_over_x2(theta);

    float rk = r[k], rl = r[l];
    float inv_theta = 1.0f / theta;
    float rk_t = rk * inv_theta;
    float rl_t = rl * inv_theta;

    // Kronecker delta for k == l
    float dkl = (k == l) ? 1.0f : 0.0f;

    // Second derivative of rk/theta w.r.t. r_l:
    // d/dl (rk/theta) = (dkl*theta - rk*rl/theta) / theta^2
    //                 = (dkl - rk*rl/theta^2) / theta
    float d_rk_t_dl = (dkl - rk * rl * inv_theta * inv_theta) * inv_theta;

    // d^2(sinc)/dr_k dr_l = d2s * rk_t * rl_t + ds * d_rk_t_dl
    float coeff_K_2  = d2s * rk_t * rl_t + ds * d_rk_t_dl;
    // d^2(h)/dr_k dr_l
    float coeff_K2_2 = d2h * rk_t * rl_t + dh * d_rk_t_dl;

    Mat3f dKk_K = dKk * K;
    Mat3f K_dKk = K * dKk;
    Mat3f dKl_K = dKl * K;
    Mat3f K_dKl = K * dKl;
    Mat3f dK2_k = dKk_K + K_dKk;
    Mat3f dK2_l = dKl_K + K_dKl;
    Mat3f dKk_dKl = dKk * dKl;
    Mat3f dKl_dKk = dKl * dKk;
    Mat3f d2K2 = dKk_dKl + dKl_dKk;  // d^2(K^2)/(dr_k dr_l)

    Mat3f result = K * coeff_K_2
                 + dKk * (ds * rl_t) + dKl * (ds * rk_t)
                 + K2 * coeff_K2_2
                 + dK2_k * (dh * rl_t) + dK2_l * (dh * rk_t)
                 + d2K2 * h;

    return result;
}

// ============================================================================
// Vertex Jacobian for rigid body: x = R(theta) * x_bar + p
//
// RigidJacobi stores the rest-frame vertex position x_bar and provides
// the 3x6 Jacobian, its transpose, and the Hessian lifting for
// the 6-DOF pose vector q = [p(3), theta(3)].
//
// J = [I_3  |  dR/dtheta_0 * x_bar,  dR/dtheta_1 * x_bar,  dR/dtheta_2 * x_bar]
// ============================================================================

struct RigidJacobi {
    Vec3f x_bar;

    CHYSX_HDI RigidJacobi() : x_bar(0, 0, 0) {}
    CHYSX_HDI explicit RigidJacobi(Vec3f xb) : x_bar(xb) {}

    // World position: x = R(theta) * x_bar + p
    CHYSX_HDI Vec3f mul_q(const Vec6f& q) const {
        Vec3f p = pose_position(q);
        Vec3f theta = pose_rotation(q);
        Mat3f R = rodrigues(theta);
        return p + R * x_bar;
    }

    // J^T * g3 -> 6D generalized force
    // grad_p = g (translation part is identity)
    // grad_theta_k = (dR/dtheta_k * x_bar)^T * g  for k = 0,1,2
    CHYSX_HDI Vec6f mul_JT(Vec3f g, const Vec6f& q) const {
        Vec3f theta = pose_rotation(q);
        Vec6f result;
        result[0] = g.x; result[1] = g.y; result[2] = g.z;
        for (int k = 0; k < 3; ++k) {
            Mat3f dRk = dR_dr(theta, k);
            Vec3f dxk = dRk * x_bar;
            result[3 + k] = math::dot(dxk, g);
        }
        return result;
    }

    // Ji^T * H3x3 * Jj -> 6x6 Hessian (single body version)
    CHYSX_HDI static Mat6f JT_H_J(const RigidJacobi& Ji, const Mat3f& H,
                                    const RigidJacobi& Jj, const Vec6f& q) {
        return JT_H_J(Ji, H, Jj, q, q);
    }

    // Ji^T * H3x3 * Jj -> 6x6 Hessian (two-body version: qi for Ji, qj for Jj)
    CHYSX_HDI static Mat6f JT_H_J(const RigidJacobi& Ji, const Mat3f& H,
                                    const RigidJacobi& Jj,
                                    const Vec6f& qi, const Vec6f& qj) {
        Vec3f theta_i = pose_rotation(qi);
        Vec3f theta_j = pose_rotation(qj);
        Mat6f R = Mat6f::zero();

        Vec3f dVi[3], dVj[3];
        for (int k = 0; k < 3; ++k) {
            dVi[k] = dR_dr(theta_i, k) * Ji.x_bar;
            dVj[k] = dR_dr(theta_j, k) * Jj.x_bar;
        }

        // (0,0) block: I^T * H * I = H
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R(r, c) = H(r, c);

        // (0, 3+l) block: I^T * H * dVj[l]
        for (int l = 0; l < 3; ++l) {
            Vec3f Hv = H * dVj[l];
            for (int r = 0; r < 3; ++r)
                R(r, 3 + l) = Hv.data[r];
        }

        // (3+k, 0) block: dVi[k]^T * H * I
        for (int k = 0; k < 3; ++k) {
            for (int c = 0; c < 3; ++c) {
                float val = 0;
                for (int m = 0; m < 3; ++m) val += dVi[k].data[m] * H(m, c);
                R(3 + k, c) = val;
            }
        }

        // (3+k, 3+l) block: dVi[k]^T * H * dVj[l]
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                Vec3f Hvl = H * dVj[l];
                R(3 + k, 3 + l) = math::dot(dVi[k], Hvl);
            }
        }

        return R;
    }

    // Second-derivative correction: sum over spatial coords j of
    // (d^2 V_j / dq^2) * grad_V[j]
    // where V_j = (R * x_bar + p)[j], so d^2V_j/dp^2 = 0, d^2V_j/dp dtheta = 0,
    // d^2V_j / dtheta_k dtheta_l = (d^2R/dk dl * x_bar)[j]
    CHYSX_HDI Mat6f vertex_hessian_correction(Vec3f grad_V, const Vec6f& q) const {
        Vec3f theta = pose_rotation(q);
        Mat6f corr = Mat6f::zero();
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                Mat3f d2Rkl = d2R_dr2(theta, k, l);
                Vec3f d2v = d2Rkl * x_bar;
                corr(3 + k, 3 + l) = math::dot(d2v, grad_V);
            }
        }
        return corr;
    }
};

}  // namespace rigid_ipc
}  // namespace chysx
