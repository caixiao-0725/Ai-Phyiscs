// SPDX-License-Identifier: MIT
// Preconditioned Conjugate Gradient solver for the Rigid-IPC block-sparse system.
//
// The global system H dq = -g has 6x6 blocks on the diagonal (one per body)
// and 6x6 off-diagonal blocks from contact coupling. The preconditioner
// is the inverse of the block-diagonal.
//
// Adapted from abd_ipc_pcg.cuh (12x12 blocks -> 6x6 blocks).

#pragma once

#include "rigid_ipc_types.cuh"

#include <cstdio>
#include <vector>

namespace chysx {
namespace rigid_ipc {

// ============================================================================
// Block-sparse linear system (6x6 blocks)
// ============================================================================

struct BlockSystem6 {
    int n_bodies;

    std::vector<Mat6f> H_diag;

    struct OffDiag {
        int row, col;
        Mat6f block;
    };
    std::vector<OffDiag> H_offdiag;

    std::vector<Vec6f> rhs;

    void resize(int n) {
        n_bodies = n;
        H_diag.resize(n);
        rhs.resize(n);
        H_offdiag.clear();
    }

    void zero() {
        for (int i = 0; i < n_bodies; ++i) {
            H_diag[i] = Mat6f::zero();
            rhs[i] = Vec6f::zero();
        }
        H_offdiag.clear();
    }

    void add_offdiag(int row, int col, const Mat6f& block) {
        H_offdiag.push_back({row, col, block});
    }

    void matvec(const std::vector<Vec6f>& x, std::vector<Vec6f>& y) const {
        for (int i = 0; i < n_bodies; ++i)
            y[i] = H_diag[i] * x[i];
        for (auto& od : H_offdiag)
            y[od.row] = y[od.row] + od.block * x[od.col];
    }
};

// ============================================================================
// PCG solver (6x6 block version)
// ============================================================================

struct PCGResult {
    int iterations;
    float residual;
    bool converged;
};

inline PCGResult solve_pcg6(const BlockSystem6& sys,
                             std::vector<Vec6f>& dx,
                             float tol = 1e-3f,
                             int max_iter = 0) {
    int n = sys.n_bodies;
    if (max_iter <= 0) max_iter = n * 6 * 2;

    std::vector<Mat6f> P(n);
    for (int i = 0; i < n; ++i)
        P[i] = math::inverse_spd<float, 6>(sys.H_diag[i]);

    std::vector<Vec6f> r(n), z(n), p(n), Hp(n);

    for (int i = 0; i < n; ++i) {
        dx[i] = Vec6f::zero();
        r[i] = sys.rhs[i];
    }

    for (int i = 0; i < n; ++i) z[i] = P[i] * r[i];
    for (int i = 0; i < n; ++i) p[i] = z[i];

    float rz = 0;
    for (int i = 0; i < n; ++i) rz += math::dot(r[i], z[i]);

    float rhs_norm = 0;
    for (int i = 0; i < n; ++i) rhs_norm += math::dot(sys.rhs[i], sys.rhs[i]);
    rhs_norm = sqrtf(rhs_norm);
    if (rhs_norm < 1e-20f)
        return {0, 0.0f, true};

    float abs_tol = tol * rhs_norm;
    PCGResult result{0, 0.0f, false};

    for (int iter = 0; iter < max_iter; ++iter) {
        sys.matvec(p, Hp);

        float pHp = 0;
        for (int i = 0; i < n; ++i) pHp += math::dot(p[i], Hp[i]);
        if (pHp <= 0.0f) pHp = 1e-20f;
        float alpha = rz / pHp;

        for (int i = 0; i < n; ++i) {
            dx[i] = dx[i] + p[i] * alpha;
            r[i] = r[i] - Hp[i] * alpha;
        }

        float r_norm = 0;
        for (int i = 0; i < n; ++i) r_norm += math::dot(r[i], r[i]);
        r_norm = sqrtf(r_norm);

        result.iterations = iter + 1;
        result.residual = r_norm;

        if (r_norm < abs_tol) {
            result.converged = true;
            break;
        }

        for (int i = 0; i < n; ++i) z[i] = P[i] * r[i];

        float rz_new = 0;
        for (int i = 0; i < n; ++i) rz_new += math::dot(r[i], z[i]);

        float beta = rz_new / (rz > 1e-30f ? rz : 1e-30f);
        rz = rz_new;

        for (int i = 0; i < n; ++i)
            p[i] = z[i] + p[i] * beta;
    }
    return result;
}

}  // namespace rigid_ipc
}  // namespace chysx
