// SPDX-License-Identifier: MIT
// Squared-distance primitives for IPC contact:
//   PP  (point–point),  PE  (point–edge),
//   PT  (point–triangle), EE  (edge–edge).
//
// Each function returns D = d² and its gradient / Hessian w.r.t. the
// stacked vertex positions x = [x0; x1; ...].

#pragma once

#include "../../math/matrix.cuh"
#include "../../math/vec.cuh"

namespace chysx {
namespace abd_ipc {

using math::Vec3f;
using math::Mat3f;

// ============================================================================
// Point–Point  D(p0, p1) = |p0 − p1|²
// ============================================================================

CHYSX_HDI float dist2_pp(Vec3f p0, Vec3f p1) {
    Vec3f d = p0 - p1;
    return math::dot(d, d);
}

// Gradient w.r.t. [p0; p1] — 6-vector
CHYSX_HDI void dist2_pp_grad(Vec3f p0, Vec3f p1, Vec3f& g0, Vec3f& g1) {
    Vec3f d = p0 - p1;
    g0 = d * 2.0f;
    g1 = d * (-2.0f);
}

// Hessian w.r.t. [p0; p1] — four 3x3 blocks
CHYSX_HDI void dist2_pp_hess(Mat3f& H00, Mat3f& H01, Mat3f& H10, Mat3f& H11) {
    Mat3f I2 = Mat3f::identity() * 2.0f;
    H00 = I2; H11 = I2;
    H01 = I2 * (-1.0f); H10 = I2 * (-1.0f);
}

// ============================================================================
// Point–Edge  D(p, e0, e1) — closest point on segment [e0,e1]
// ============================================================================

CHYSX_HDI float dist2_pe(Vec3f p, Vec3f e0, Vec3f e1) {
    Vec3f e = e1 - e0;
    Vec3f w = p - e0;
    float ee = math::dot(e, e);
    float we = math::dot(w, e);

    float t = (ee > 1e-12f) ? math::clamp(we / ee, 0.0f, 1.0f) : 0.0f;
    Vec3f closest = e0 + e * t;
    Vec3f diff = p - closest;
    return math::dot(diff, diff);
}

// Gradient w.r.t. [p; e0; e1] — three Vec3f
CHYSX_HDI void dist2_pe_grad(Vec3f p, Vec3f e0, Vec3f e1,
                              Vec3f& gp, Vec3f& g0, Vec3f& g1) {
    Vec3f e = e1 - e0;
    Vec3f w = p - e0;
    float ee = math::dot(e, e);
    float we = math::dot(w, e);

    float t = (ee > 1e-12f) ? math::clamp(we / ee, 0.0f, 1.0f) : 0.0f;
    Vec3f closest = e0 + e * t;
    Vec3f diff = p - closest;

    // D = |p - e0 - t*e|^2, with t clamped
    // If t is at a boundary, grad is same as PP with the endpoint.
    gp = diff * 2.0f;
    g0 = diff * (-2.0f * (1.0f - t));
    g1 = diff * (-2.0f * t);
}

// ============================================================================
// Point–Triangle  D(p, t0, t1, t2)
// ============================================================================

CHYSX_HDI float dist2_pt(Vec3f p, Vec3f t0, Vec3f t1, Vec3f t2) {
    Vec3f e0 = t1 - t0, e1 = t2 - t0, v = p - t0;
    float d00 = math::dot(e0, e0);
    float d01 = math::dot(e0, e1);
    float d11 = math::dot(e1, e1);
    float d20 = math::dot(v, e0);
    float d21 = math::dot(v, e1);

    float denom = d00 * d11 - d01 * d01;
    if (denom < 1e-12f) {
        // Degenerate triangle — fall back to closest edge
        float d_e0 = dist2_pe(p, t0, t1);
        float d_e1 = dist2_pe(p, t1, t2);
        float d_e2 = dist2_pe(p, t2, t0);
        return math::min(d_e0, math::min(d_e1, d_e2));
    }

    float inv = 1.0f / denom;
    float u = (d11 * d20 - d01 * d21) * inv;
    float w = (d00 * d21 - d01 * d20) * inv;

    // Clamp to triangle interior
    u = math::clamp(u, 0.0f, 1.0f);
    w = math::clamp(w, 0.0f, 1.0f);
    if (u + w > 1.0f) {
        float s = u + w;
        u /= s; w /= s;
    }

    Vec3f closest = t0 + e0 * u + e1 * w;
    Vec3f diff = p - closest;
    return math::dot(diff, diff);
}

// PT gradient w.r.t. [p; t0; t1; t2]
CHYSX_HDI void dist2_pt_grad(Vec3f p, Vec3f t0, Vec3f t1, Vec3f t2,
                              Vec3f& gp, Vec3f& gt0, Vec3f& gt1, Vec3f& gt2) {
    Vec3f e0 = t1 - t0, e1 = t2 - t0, v = p - t0;
    float d00 = math::dot(e0, e0);
    float d01 = math::dot(e0, e1);
    float d11 = math::dot(e1, e1);
    float d20 = math::dot(v, e0);
    float d21 = math::dot(v, e1);

    float denom = d00 * d11 - d01 * d01;
    float u, w;
    if (denom < 1e-12f) {
        u = 0.0f; w = 0.0f;
    } else {
        float inv = 1.0f / denom;
        u = (d11 * d20 - d01 * d21) * inv;
        w = (d00 * d21 - d01 * d20) * inv;
    }

    u = math::clamp(u, 0.0f, 1.0f);
    w = math::clamp(w, 0.0f, 1.0f);
    if (u + w > 1.0f) { float s = u + w; u /= s; w /= s; }

    Vec3f closest = t0 + e0 * u + e1 * w;
    Vec3f diff = p - closest;

    gp  = diff * 2.0f;
    gt0 = diff * (-2.0f * (1.0f - u - w));
    gt1 = diff * (-2.0f * u);
    gt2 = diff * (-2.0f * w);
}

// ============================================================================
// Edge–Edge  D(a0, a1, b0, b1)
// ============================================================================

CHYSX_HDI float dist2_ee(Vec3f a0, Vec3f a1, Vec3f b0, Vec3f b1) {
    Vec3f u = a1 - a0, v = b1 - b0, w = a0 - b0;
    float uu = math::dot(u, u);
    float uv = math::dot(u, v);
    float vv = math::dot(v, v);
    float uw = math::dot(u, w);
    float vw = math::dot(v, w);

    float det = uu * vv - uv * uv;
    float s, t;

    if (det < 1e-12f) {
        // Nearly parallel edges
        s = 0.0f;
        t = (vv > 1e-12f) ? math::clamp(-vw / vv, 0.0f, 1.0f) : 0.0f;
    } else {
        float inv = 1.0f / det;
        s = math::clamp((uv * vw - vv * uw) * inv, 0.0f, 1.0f);
        t = math::clamp((uu * vw - uv * uw) * inv, 0.0f, 1.0f);
    }

    // Re-clamp to handle boundary conditions
    float tnom = uv * s + vw;
    if (tnom < 0.0f) {
        t = 0.0f;
        s = math::clamp(-uw / uu, 0.0f, 1.0f);
    } else if (tnom > vv) {
        t = 1.0f;
        s = math::clamp((uv - uw) / uu, 0.0f, 1.0f);
    } else {
        t = (vv > 1e-12f) ? tnom / vv : 0.0f;
    }

    Vec3f diff = (a0 + u * s) - (b0 + v * t);
    return math::dot(diff, diff);
}

// EE gradient w.r.t. [a0; a1; b0; b1]
CHYSX_HDI void dist2_ee_grad(Vec3f a0, Vec3f a1, Vec3f b0, Vec3f b1,
                              Vec3f& ga0, Vec3f& ga1, Vec3f& gb0, Vec3f& gb1) {
    Vec3f u = a1 - a0, v = b1 - b0, w = a0 - b0;
    float uu = math::dot(u, u);
    float uv = math::dot(u, v);
    float vv = math::dot(v, v);
    float uw = math::dot(u, w);
    float vw = math::dot(v, w);

    float det = uu * vv - uv * uv;
    float s, t;
    if (det < 1e-12f) {
        s = 0.0f;
        t = (vv > 1e-12f) ? math::clamp(-vw / vv, 0.0f, 1.0f) : 0.0f;
    } else {
        float inv = 1.0f / det;
        s = math::clamp((uv * vw - vv * uw) * inv, 0.0f, 1.0f);
        t = math::clamp((uu * vw - uv * uw) * inv, 0.0f, 1.0f);
    }

    float tnom = uv * s + vw;
    if (tnom < 0.0f) {
        t = 0.0f;
        s = math::clamp(-uw / uu, 0.0f, 1.0f);
    } else if (tnom > vv) {
        t = 1.0f;
        s = math::clamp((uv - uw) / uu, 0.0f, 1.0f);
    } else {
        t = (vv > 1e-12f) ? tnom / vv : 0.0f;
    }

    Vec3f pa = a0 + u * s;
    Vec3f pb = b0 + v * t;
    Vec3f diff = pa - pb;

    ga0 = diff * (2.0f * (1.0f - s));
    ga1 = diff * (2.0f * s);
    gb0 = diff * (-2.0f * (1.0f - t));
    gb1 = diff * (-2.0f * t);
}

// ============================================================================
// Contact type classification — determines degenerate cases
// ============================================================================

enum class PTType { PT, PE01, PE12, PE20, PP0, PP1, PP2 };
enum class EEType { EE, PE_A, PE_B, PP };

CHYSX_HDI PTType classify_pt(Vec3f p, Vec3f t0, Vec3f t1, Vec3f t2) {
    Vec3f e0 = t1 - t0, e1 = t2 - t0, v = p - t0;
    float d00 = math::dot(e0, e0);
    float d01 = math::dot(e0, e1);
    float d11 = math::dot(e1, e1);
    float d20 = math::dot(v, e0);
    float d21 = math::dot(v, e1);

    float denom = d00 * d11 - d01 * d01;
    if (denom < 1e-12f) return PTType::PE01;

    float inv = 1.0f / denom;
    float u = (d11 * d20 - d01 * d21) * inv;
    float w = (d00 * d21 - d01 * d20) * inv;

    if (u >= 0 && w >= 0 && u + w <= 1) return PTType::PT;
    if (u < 0 && w < 0) return PTType::PP0;
    if (u > 1) return PTType::PP1;
    if (w > 1) return PTType::PP2;
    if (u < 0) return PTType::PE20;
    if (w < 0) return PTType::PE01;
    return PTType::PE12; // u + w > 1
}

CHYSX_HDI EEType classify_ee(Vec3f a0, Vec3f a1, Vec3f b0, Vec3f b1) {
    Vec3f u = a1 - a0, v = b1 - b0, w = a0 - b0;
    float uu = math::dot(u, u);
    float uv = math::dot(u, v);
    float vv = math::dot(v, v);
    float uw = math::dot(u, w);
    float vw = math::dot(v, w);

    float det = uu * vv - uv * uv;
    float s, t;
    if (det < 1e-12f) {
        s = 0.0f;
        t = (vv > 1e-12f) ? math::clamp(-vw / vv, 0.0f, 1.0f) : 0.0f;
    } else {
        float inv = 1.0f / det;
        s = (uv * vw - vv * uw) * inv;
        t = (uu * vw - uv * uw) * inv;
    }

    bool s_int = s > 0.0f && s < 1.0f;
    bool t_int = t > 0.0f && t < 1.0f;

    if (s_int && t_int) return EEType::EE;
    if (!s_int && t_int) return EEType::PE_B;  // point on edge A, segment B
    if (s_int && !t_int) return EEType::PE_A;  // point on edge B, segment A
    return EEType::PP;
}

}  // namespace abd_ipc
}  // namespace chysx
