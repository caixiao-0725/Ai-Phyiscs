// SPDX-License-Identifier: MIT
// Preconditioned Conjugate Gradient solver for the ABD block-sparse system.
//
// The global system H Δq = −g has 12×12 blocks on the diagonal (one per body)
// and 12×12 off-diagonal blocks from contact coupling.  The preconditioner
// is the inverse of the block-diagonal.
//
// For small body counts (e.g. hello_affine_body with 2 bodies) this runs
// entirely on the host.  A GPU kernel path can be added later.

#pragma once

#include "abd_ipc_types.cuh"

#include <cstdio>
#include <vector>

namespace chysx {
namespace abd_ipc {

// ============================================================================
// Block-sparse linear system
// ============================================================================

struct BlockSystem {
    int n_bodies;

    // Diagonal blocks: H_diag[i] is 12x12 for body i
    std::vector<Mat12f> H_diag;

    // Off-diagonal blocks: sparse list of (row, col, block)
    struct OffDiag {
        int row, col;
        Mat12f block;
    };
    std::vector<OffDiag> H_offdiag;

    // Right-hand side (negative gradient): one Vec12f per body
    std::vector<Vec12f> rhs;

    void resize(int n) {
        n_bodies = n;
        H_diag.resize(n);
        rhs.resize(n);
        H_offdiag.clear();
    }

    void zero() {
        for (int i = 0; i < n_bodies; ++i) {
            H_diag[i] = Mat12f::zero();
            rhs[i] = Vec12f::zero();
        }
        H_offdiag.clear();
    }

    void add_offdiag(int row, int col, const Mat12f& block) {
        H_offdiag.push_back({row, col, block});
    }

    // Matrix-vector product: y = H * x
    void matvec(const std::vector<Vec12f>& x, std::vector<Vec12f>& y) const {
        for (int i = 0; i < n_bodies; ++i)
            y[i] = H_diag[i] * x[i];
        for (auto& od : H_offdiag)
            y[od.row] += od.block * x[od.col];
    }
};

// ============================================================================
// PCG solver
// ============================================================================

struct PCGResult {
    int iterations;
    float residual;
    bool converged;
};

// Solve H * dx = rhs using PCG with block-diagonal preconditioner.
// dx is output (one Vec12f per body, initialized to zero).
inline PCGResult solve_pcg(const BlockSystem& sys,
                            std::vector<Vec12f>& dx,
                            float tol = 1e-3f,
                            int max_iter = 0) {
    int n = sys.n_bodies;
    if (max_iter <= 0) max_iter = n * 12 * 2;

    // Build preconditioner: P[i] = H_diag[i]^{-1}
    std::vector<Mat12f> P(n);
    for (int i = 0; i < n; ++i)
        P[i] = math::inverse_spd<float, 12>(sys.H_diag[i]);

    // Allocate work vectors
    std::vector<Vec12f> r(n), z(n), p(n), Hp(n);

    // Initialize: x = 0, r = rhs
    for (int i = 0; i < n; ++i) {
        dx[i] = Vec12f::zero();
        r[i] = sys.rhs[i];
    }

    // z = P * r
    for (int i = 0; i < n; ++i) z[i] = P[i] * r[i];

    // p = z
    for (int i = 0; i < n; ++i) p[i] = z[i];

    // rz = r · z
    float rz = 0;
    for (int i = 0; i < n; ++i) rz += math::dot(r[i], z[i]);

    float rhs_norm = 0;
    for (int i = 0; i < n; ++i) rhs_norm += math::dot(sys.rhs[i], sys.rhs[i]);
    rhs_norm = sqrtf(rhs_norm);
    if (rhs_norm < 1e-20f) {
        return {0, 0.0f, true};
    }

    float abs_tol = tol * rhs_norm;

    PCGResult result{0, 0.0f, false};

    for (int iter = 0; iter < max_iter; ++iter) {
        // Hp = H * p
        sys.matvec(p, Hp);

        // alpha = rz / (p · Hp)
        float pHp = 0;
        for (int i = 0; i < n; ++i) pHp += math::dot(p[i], Hp[i]);
        if (pHp <= 0.0f) pHp = 1e-20f;
        float alpha = rz / pHp;

        // x += alpha * p,  r -= alpha * Hp
        for (int i = 0; i < n; ++i) {
            dx[i] += p[i] * alpha;
            r[i] = r[i] - Hp[i] * alpha;
        }

        // Check convergence
        float r_norm = 0;
        for (int i = 0; i < n; ++i) r_norm += math::dot(r[i], r[i]);
        r_norm = sqrtf(r_norm);

        result.iterations = iter + 1;
        result.residual = r_norm;

        if (r_norm < abs_tol) {
            result.converged = true;
            break;
        }

        // z = P * r
        for (int i = 0; i < n; ++i) z[i] = P[i] * r[i];

        float rz_new = 0;
        for (int i = 0; i < n; ++i) rz_new += math::dot(r[i], z[i]);

        float beta = rz_new / (rz > 1e-30f ? rz : 1e-30f);
        rz = rz_new;

        // p = z + beta * p
        for (int i = 0; i < n; ++i)
            p[i] = z[i] + p[i] * beta;
    }
    return result;
}

}  // namespace abd_ipc
}  // namespace chysx
