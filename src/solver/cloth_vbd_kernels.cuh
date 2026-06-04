// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// Cloth VBD energy kernels — Stable Neo-Hookean membrane + dihedral bending.
// Ported 1:1 from Newton's particle_vbd_kernels.py.
//
// IMPORTANT: This header is included inside `namespace chysx { namespace solver { }`.
// Do NOT add namespace wrappers here. The math headers MUST be included
// by the .cu file BEFORE the enclosing namespace opens.

#pragma once

// Type aliases — fully qualified from global namespace
using _Vec3f = ::chysx::math::Vec3f;
using _Mat2f = ::chysx::math::Mat2f;
using _Mat3f = ::chysx::math::Mat3f;

// -----------------------------------------------------------------------
// Stable Neo-Hookean 2D membrane force + Hessian (Smith et al. 2018)
// -----------------------------------------------------------------------
__device__ __forceinline__ void cloth_vbd_membrane_force_hessian(
    int v_order,
    const _Vec3f& x0, const _Vec3f& x1, const _Vec3f& x2,
    const _Vec3f& x0_prev, const _Vec3f& x1_prev, const _Vec3f& x2_prev,
    const _Mat2f& DmInv,
    float area, float mu, float lmbd, float damping, float dt,
    _Vec3f& f_out, _Mat3f& h_out)
{
    using ::chysx::math::dot;
    using ::chysx::math::outer;

    const _Vec3f x01 = x1 - x0;
    const _Vec3f x02 = x2 - x0;

    const float D00 = DmInv(0,0), D01 = DmInv(0,1);
    const float D10 = DmInv(1,0), D11 = DmInv(1,1);

    const _Vec3f f0 = x01 * D00 + x02 * D10;
    const _Vec3f f1 = x01 * D01 + x02 * D11;

    const float f0f0 = dot(f0, f0);
    const float f1f1 = dot(f1, f1);
    const float f0f1 = dot(f0, f1);

    float J_s_sq = f0f0 * f1f1 - f0f1 * f0f1;
    J_s_sq = fmaxf(J_s_sq, 1.0e-20f);
    const float J_s = sqrtf(J_s_sq);
    const float inv_J_s = 1.0f / J_s;

    const float mu_nh = mu;
    const float lmbd_nh = lmbd + mu;
    const float lmbd_safe = copysignf(fmaxf(fabsf(lmbd_nh), 1.0e-6f), lmbd_nh);
    const float alpha = 1.0f + mu_nh / lmbd_safe;

    const _Vec3f g0 = (f1f1 * f0 - f0f1 * f1) * inv_J_s;
    const _Vec3f g1 = (f0f0 * f1 - f0f1 * f0) * inv_J_s;

    const float s = lmbd_nh * (J_s - alpha);
    const _Vec3f P0 = f0 * mu_nh + g0 * s;
    const _Vec3f P1 = f1 * mu_nh + g1 * s;

    const float m0 = (v_order == 0) ? 1.0f : 0.0f;
    const float m1 = (v_order == 1) ? 1.0f : 0.0f;
    const float m2 = (v_order == 2) ? 1.0f : 0.0f;

    const float df0_dx = D00 * (m1 - m0) + D10 * (m2 - m0);
    const float df1_dx = D01 * (m1 - m0) + D11 * (m2 - m0);

    _Vec3f force = (P0 * df0_dx + P1 * df1_dx) * (-area);

    const float s_clamp = fmaxf(0.0f, s);
    const float r = s_clamp * inv_J_s;
    const float c1 = lmbd_nh - r;
    const float df0_sq = df0_dx * df0_dx;
    const float df1_sq = df1_dx * df1_dx;

    const _Vec3f dJ_dx = g0 * df0_dx + g1 * df1_dx;
    const _Vec3f w = f1 * df0_dx - f0 * df1_dx;

    const float I_coeff = mu_nh * (df0_sq + df1_sq) + r * (
        df0_sq * f1f1 + df1_sq * f0f0 - 2.0f * df0_dx * df1_dx * f0f1);

    _Mat3f hessian = _Mat3f::identity() * I_coeff
                   + outer(dJ_dx, dJ_dx) * c1
                   - outer(w, w) * r;

    if (damping > 0.0f) {
        const float G00 = 0.5f * (f0f0 - 1.0f);
        const float G11 = 0.5f * (f1f1 - 1.0f);
        const float G01 = 0.5f * f0f1;
        const float G_frob_sq = G00*G00 + G11*G11 + 2.0f*G01*G01;

        if (G_frob_sq >= 1.0e-20f) {
            const float inv_dt = 1.0f / dt;
            const _Vec3f vel01 = (x01 - (x1_prev - x0_prev)) * inv_dt;
            const _Vec3f vel02 = (x02 - (x2_prev - x0_prev)) * inv_dt;

            const _Vec3f df0_dt = vel01 * D00 + vel02 * D10;
            const _Vec3f df1_dt = vel01 * D01 + vel02 * D11;

            const float Cmu = sqrtf(G_frob_sq);
            const float G00n = G00 / Cmu, G01n = G01 / Cmu, G11n = G11 / Cmu;

            const float dG00 = dot(f0, df0_dt);
            const float dG11 = dot(f1, df1_dt);
            const float dG01 = 0.5f * (dot(f0, df1_dt) + dot(f1, df0_dt));

            const float dCmu_dt = G00n*dG00 + G11n*dG11 + 2.0f*G01n*dG01;
            const _Vec3f dCmu_dF0 = f0*G00n + f1*G01n;
            const _Vec3f dCmu_dF1 = f0*G01n + f1*G11n;
            const _Vec3f dCmu_dx = dCmu_dF0*df0_dx + dCmu_dF1*df1_dx;

            const float kd_mu = mu * damping;
            force = force - dCmu_dx * (kd_mu * dCmu_dt * area);
            hessian = hessian + outer(dCmu_dx, dCmu_dx) * (kd_mu * inv_dt);

            const float dClmbd_dt = dG00 + dG11;
            const _Vec3f dClmbd_dx = f0*df0_dx + f1*df1_dx;

            const float kd_lmbd = lmbd * damping;
            force = force - dClmbd_dx * (kd_lmbd * dClmbd_dt * area);
            hessian = hessian + outer(dClmbd_dx, dClmbd_dx) * (kd_lmbd * inv_dt);
        }
    }

    hessian = hessian * area;
    f_out = f_out + force;
    h_out = h_out + hessian;
}

// -----------------------------------------------------------------------
// Dihedral bending force + Hessian (Grinspun discrete shells)
// -----------------------------------------------------------------------
__device__ __forceinline__ void cloth_vbd_bending_force_hessian(
    int v_order,
    const _Vec3f& x0, const _Vec3f& x1,
    const _Vec3f& x2, const _Vec3f& x3,
    const _Vec3f& x0_prev, const _Vec3f& x1_prev,
    const _Vec3f& x2_prev, const _Vec3f& x3_prev,
    float rest_angle, float rest_length,
    float stiffness, float damping, float dt,
    _Vec3f& f_out, _Mat3f& h_out)
{
    using ::chysx::math::dot;
    using ::chysx::math::cross;
    using ::chysx::math::length;
    using ::chysx::math::outer;
    using ::chysx::math::transpose;
    using ::chysx::math::skew;

    constexpr float eps = 1.0e-6f;

    const _Vec3f x02 = x2 - x0;
    const _Vec3f x03 = x3 - x0;
    const _Vec3f x13 = x3 - x1;
    const _Vec3f x12 = x2 - x1;
    const _Vec3f e = x3 - x2;

    const _Vec3f n1 = cross(x02, x03);
    const _Vec3f n2 = cross(x13, x12);

    const float n1_norm = length(n1);
    const float n2_norm = length(n2);
    const float e_norm = length(e);

    if (n1_norm < eps || n2_norm < eps || e_norm < eps) return;

    const _Vec3f n1_hat = n1 / n1_norm;
    const _Vec3f n2_hat = n2 / n2_norm;
    const _Vec3f e_hat = e / e_norm;

    const float sin_theta = dot(cross(n1_hat, n2_hat), e_hat);
    const float cos_theta = dot(n1_hat, n2_hat);
    const float theta = atan2f(sin_theta, cos_theta);

    const float k = stiffness * rest_length;
    const float dE_dtheta = k * (theta - rest_angle);

    auto compute_dtheta = [&](const _Mat3f& dn1h_dx, const _Mat3f& dn2h_dx) -> _Vec3f {
        const _Mat3f sk_n1 = skew(n1_hat);
        const _Mat3f sk_n2 = skew(n2_hat);
        const _Mat3f dsin_mat = sk_n1 * dn2h_dx - sk_n2 * dn1h_dx;
        const _Vec3f dsin_dx = transpose(dsin_mat) * e_hat;
        const _Vec3f dcos_dx = transpose(dn1h_dx) * n2_hat + transpose(dn2h_dx) * n1_hat;
        return dsin_dx * cos_theta - dcos_dx * sin_theta;
    };

    auto proj_deriv = [](float norm, const _Vec3f& hat, const _Mat3f& raw_deriv) -> _Mat3f {
        const _Mat3f P = _Mat3f::identity() - outer(hat, hat);
        return P * raw_deriv * (1.0f / norm);
    };

    const _Mat3f skew_e = skew(e);

    _Mat3f dn1h[4], dn2h[4];
    dn1h[0] = proj_deriv(n1_norm, n1_hat, skew_e);
    dn2h[0] = _Mat3f();

    dn1h[1] = _Mat3f();
    dn2h[1] = proj_deriv(n2_norm, n2_hat, skew_e * (-1.0f));

    dn1h[2] = proj_deriv(n1_norm, n1_hat, skew(x03) * (-1.0f));
    dn2h[2] = proj_deriv(n2_norm, n2_hat, skew(x13));

    dn1h[3] = proj_deriv(n1_norm, n1_hat, skew(x02));
    dn2h[3] = proj_deriv(n2_norm, n2_hat, skew(x12) * (-1.0f));

    _Vec3f dtheta[4];
    for (int i = 0; i < 4; ++i) {
        dtheta[i] = compute_dtheta(dn1h[i], dn2h[i]);
    }

    const _Vec3f& dtheta_dx = dtheta[v_order];

    _Vec3f bforce = dtheta_dx * (-dE_dtheta);
    _Mat3f bhessian = outer(dtheta_dx, dtheta_dx) * k;

    if (damping > 0.0f) {
        const float inv_dt = 1.0f / dt;
        const _Vec3f dx[4] = {x0 - x0_prev, x1 - x1_prev, x2 - x2_prev, x3 - x3_prev};
        float dtheta_dt = 0.0f;
        for (int i = 0; i < 4; ++i) dtheta_dt += dot(dtheta[i], dx[i]);
        dtheta_dt *= inv_dt;

        const float dk = damping * k;
        bforce = bforce - dtheta_dx * (dk * dtheta_dt);
        bhessian = bhessian + outer(dtheta_dx, dtheta_dx) * (dk * inv_dt);
    }

    f_out = f_out + bforce;
    h_out = h_out + bhessian;
}
