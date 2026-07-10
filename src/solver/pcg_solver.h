// SPDX-License-Identifier: Apache-2.0
//
// chysx::solver::PCGSolver
//
// Preconditioned Conjugate Gradient solver for the linear system
//
//     A x = b
//
// where `A` is a `chysx::sparse::BlockCSR3` (block-CSR matrix with
// 3x3 single-precision non-zeros) and `x`, `b` are arrays of
// `math::Vec3f` (one Vec3 per particle / block row).
//
// Preconditioner
// --------------
// By default the solver uses the inverse 3x3 diagonal blocks from
// `BlockCSR3::diag`. Callers whose unknowns form four-Vec3 rigid-body groups
// may instead provide packed 12x12 Cholesky factors through
// `BodyBlock12PreconditionerOp`; this changes only M^-1, not A or its SpMV.
//
// Algorithm
// ---------
// Standard right-preconditioned CG.  The initial guess `x_0` is taken
// from the contents of the caller's `x` buffer on entry, so callers
// can warm-start by leaving the previous solve's solution in place.
// (Pass an explicitly zeroed `x` for the cold-start variant.)
//
//     r_0 = b - A x_0
//     z_0 = M^-1 r_0
//     p_0 = z_0
//     rho_0 = <r_0, z_0>
//     for k = 0, 1, 2, ...
//         q_k     = A p_k
//         alpha_k = rho_k / <p_k, q_k>
//         x_{k+1} = x_k + alpha_k p_k
//         r_{k+1} = r_k - alpha_k q_k
//         z_{k+1} = M^-1 r_{k+1}
//         rho_{k+1} = <r_{k+1}, z_{k+1}>
//         beta_k  = rho_{k+1} / rho_k
//         p_{k+1} = z_{k+1} + beta_k p_k
//
// All scalar coefficients (alpha, beta, rho) live on the device; the
// solver never copies them back to the host inside the iteration so a
// full solve runs without host-device synchronisation.
//
// Launch path
// ------------
// `PCGSolver::solve()` calls `emit_pcg()` in pcg_solver.cu, which
// issues a fused PCG kernel sequence on the caller stream.  Adjacent
// vector updates and scalar reductions are merged to minimise launch
// overhead.  The contact SpMV kernel is launched unconditionally so
// the sequence stays stable for CUDA Graph capture.

#pragma once

#include <cstdint>

#include "../collision/contact_spmv.h"
#include "../math/matrix.cuh"
#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "../memory/device_span.h"
#include "../sparse/block_csr.h"

// Forward-declare the CUDA Runtime stream type so this header does
// not need <cuda_runtime.h>; the .cu file casts std::uintptr_t to
// `cudaStream_t` (which is `CUstream_st*`) at call sites.
struct CUstream_st;

// Forward-declare CUDA Graph types to avoid pulling <cuda_runtime.h>
// into every translation unit that includes this header.
struct CUgraphExec_st;

namespace chysx {
namespace solver {

struct PCGParams {
    int max_iterations = 50;
    bool compute_true_residual = false;
};

constexpr int kBodyBlock12Dofs = 12;
constexpr int kBodyBlock12PackedLowerSize =
    kBodyBlock12Dofs * (kBodyBlock12Dofs + 1) / 2;

// Optional four-Vec3 (12 scalar DoF) block preconditioner. `lower_factors`
// stores one packed lower-triangular Cholesky factor per body, while
// `body_rows` maps the body's four local controls to BlockCSR rows.
struct BodyBlock12PreconditionerOp {
    const float* lower_factors = nullptr;
    const math::Vec4i* body_rows = nullptr;
    const unsigned char* fixed_rows = nullptr;
    int num_bodies = 0;

    bool active() const noexcept {
        return lower_factors != nullptr && body_rows != nullptr &&
               num_bodies > 0;
    }
};

class PCGSolver {
public:
    PCGSolver() = default;

    PCGSolver(const PCGSolver&) = delete;
    PCGSolver& operator=(const PCGSolver&) = delete;

    PCGSolver(PCGSolver&& other) noexcept;
    PCGSolver& operator=(PCGSolver&& other) noexcept;

    ~PCGSolver();

    // Allocate (or reuse) workspace for `num_block_rows` particles.
    // Idempotent: if the size already matches no allocation happens.
    void initialize(int num_block_rows);

    // Solve A x = b in place.
    //
    //   A : block-CSR matrix. `A.diag` supplies the default per-row 3x3
    //       preconditioner when no body preconditioner is provided.
    //   b : right-hand side, length = A.num_block_rows().
    //   x : solution buffer, length = A.num_block_rows().  Used as
    //       the initial guess on entry and overwritten with the
    //       solution on exit.  Pass a zeroed buffer for cold-start
    //       behaviour, or leave the previous solve's result in place
    //       to warm-start.
    //
    // `contact` attaches a dynamic COO-style additive operator to the
    // system: every `A * x` evaluation inside the iteration becomes
    // `(A + C) * x`, where C reads the contact pairs/weights from
    // `contact`.  The static CSR topology of A is therefore never
    // modified by collision -- contact churn between frames is
    // absorbed entirely by the COO sidecar.  Pass a default-
    // constructed (inactive) op to recover the plain `A x = b`
    // solver.
    //
    // The contact SpMV kernel is launched unconditionally (even when
    // `contact.active() == false`) so the kernel sequence is
    // identical regardless of whether contacts exist — this keeps
    // a surrounding CUDA Graph capture valid across frames.
    //
    // `body_preconditioner` optionally replaces pointwise 3x3 Jacobi with
    // one packed 12x12 triangular solve per four-row rigid-body group.
    //
    // Returns the number of iterations actually performed.
    int solve(const sparse::BlockCSR3& A,
              DeviceSpan<math::Vec3f> b,
              DeviceSpan<math::Vec3f> x,
              const PCGParams& params = PCGParams{},
              std::uintptr_t cuda_stream = 0,
              collision::ContactSpMVOp contact = {},
              collision::WideContactSpMVOp wide_contact = {},
              BodyBlock12PreconditionerOp body_preconditioner = {});

    // Last solve's preconditioner-weighted residual <r, z> from the
    // final iteration, copied to host on demand.  Useful for cheap
    // convergence checks outside the main loop.
    float last_residual();

    // Queue the residual copy on an existing stream so callers that already
    // have a frame-end synchronization can avoid an extra wait.  The host
    // value is valid after that stream has completed.
    void copy_last_residual_to_host(std::uintptr_t cuda_stream);
    float host_last_residual() const noexcept;
    float host_last_true_relative_residual() const noexcept;

private:
    void destroy_graph() noexcept;

    int num_block_rows_ = 0;

    CudaArray<math::Vec3f> r_;
    CudaArray<math::Vec3f> p_;
    CudaArray<math::Vec3f> z_;
    CudaArray<math::Vec3f> Ap_;

    CudaArray<math::Mat3f> M_inv_;        // block-Jacobi preconditioner

    // Three scalar reductions live in a single 4-element buffer so we
    // can read them out with one cudaMemcpy when needed:
    //   coeff_[0] = <r, z>      (rho_k)
    //   coeff_[1] = <p, A p>    (sigma_k)
    //   coeff_[2] = <r, z>_new  (rho_{k+1})
    //   coeff_[3] = beta_k
    CudaArray<float> coeff_;

    // One partial sum per vector block.  The buffer is allocated before
    // graph capture and reused by every reduction in every iteration.
    CudaArray<float> reduction_partial_;
    CudaArray<float> true_residual_norms_;
    bool last_true_residual_valid_ = false;

    // CUDA Graph cache.  Captured on the caller's non-default stream on
    // the first solve and replayed on subsequent calls.  Invalidated
    // when the workspace is reinitialized, or when the problem size
    // (`n`) or iteration count changes.
    CUgraphExec_st* graph_exec_ = nullptr;
    int graph_n_ = 0;
    int graph_max_iter_ = 0;
    const float* graph_body_factors_ = nullptr;
    const math::Vec4i* graph_body_rows_ = nullptr;
    const unsigned char* graph_fixed_rows_ = nullptr;
    int graph_num_bodies_ = 0;
};

}  // namespace solver
}  // namespace chysx
