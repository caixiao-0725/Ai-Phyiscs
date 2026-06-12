// SPDX-License-Identifier: MIT
// IPC log-barrier contact energy.
//
// B(D) = -(D - D̂)² · ln(D / D̂)     for 0 < D < D̂
//      = 0                            for D ≥ D̂
//
// where D = d² is the squared distance between primitives and
// D̂ = d_hat² is the squared activation threshold.
//
// The total contact energy is κ_contact · Σ B(D_k).

#pragma once

#include "../../math/common.cuh"

#include <cmath>

namespace chysx {
namespace abd_ipc {

// ============================================================================
// Barrier function and its derivatives w.r.t. D = d²
// ============================================================================

// B(D, D_hat) — barrier energy
CHYSX_HDI float barrier(float D, float D_hat) {
    if (D >= D_hat) return 0.0f;
    if (D <= 0.0f) D = 1e-12f;
    float diff = D - D_hat;
    return -(diff * diff) * logf(D / D_hat);
}

// dB/dD
CHYSX_HDI float barrier_gradient(float D, float D_hat) {
    if (D >= D_hat) return 0.0f;
    if (D <= 0.0f) D = 1e-12f;
    float diff = D - D_hat;
    return -diff * (2.0f * logf(D / D_hat) + diff / D);
}

// d²B/dD²
CHYSX_HDI float barrier_hessian(float D, float D_hat) {
    if (D >= D_hat) return 0.0f;
    if (D <= 0.0f) D = 1e-12f;
    float inv_D = 1.0f / D;
    float ratio = D_hat * inv_D;
    return -(2.0f * logf(D / D_hat) + 3.0f - 2.0f * ratio - ratio * ratio);
}

// ============================================================================
// Convenience: barrier with d_hat (unsquared) as input
// ============================================================================

CHYSX_HDI float barrier_dhat(float D, float d_hat) {
    return barrier(D, d_hat * d_hat);
}

CHYSX_HDI float barrier_gradient_dhat(float D, float d_hat) {
    return barrier_gradient(D, d_hat * d_hat);
}

CHYSX_HDI float barrier_hessian_dhat(float D, float d_hat) {
    return barrier_hessian(D, d_hat * d_hat);
}

// Clamp d²B/dD² to be non-negative (SPD projection for a scalar).
CHYSX_HDI float barrier_hessian_spd(float D, float D_hat) {
    float h = barrier_hessian(D, D_hat);
    return h > 0.0f ? h : 0.0f;
}

}  // namespace abd_ipc
}  // namespace chysx
