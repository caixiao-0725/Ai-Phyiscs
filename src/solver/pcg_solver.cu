// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::solver::PCGSolver.

#include "pcg_solver.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../collision/zero_count.h"
#include "../profile/nvtx_range.h"

namespace chysx {
namespace solver {

namespace {

constexpr int kBlockDim = 256;

inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void residual_m_inv_kernel(int n,
                                      const math::Mat3f* __restrict__ diag,
                                      const int* __restrict__ row_offsets,
                                      const int* __restrict__ col_indices,
                                      const math::Mat3f* __restrict__ values,
                                      const math::Vec3f* __restrict__ x,
                                      const math::Vec3f* __restrict__ b,
                                      math::Vec3f* __restrict__ r,
                                      math::Mat3f* __restrict__ M_inv) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    const math::Mat3f diag_block = diag[row];
    M_inv[row] = math::inverse(diag_block);

    math::Vec3f acc = diag_block * x[row];
    const int beg = row_offsets[row];
    const int end = row_offsets[row + 1];
    for (int k = beg; k < end; ++k) {
        acc += values[k] * x[col_indices[k]];
    }
    r[row] = b[row] - acc;
}

__global__ void residual_kernel(int n,
                                const math::Mat3f* __restrict__ diag,
                                const int* __restrict__ row_offsets,
                                const int* __restrict__ col_indices,
                                const math::Mat3f* __restrict__ values,
                                const math::Vec3f* __restrict__ x,
                                const math::Vec3f* __restrict__ b,
                                math::Vec3f* __restrict__ r) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    math::Vec3f acc = diag[row] * x[row];
    const int beg = row_offsets[row];
    const int end = row_offsets[row + 1];
    for (int k = beg; k < end; ++k) {
        acc += values[k] * x[col_indices[k]];
    }
    r[row] = b[row] - acc;
}

__global__ void zero_fixed_rows_kernel(
    int n,
    const unsigned char* __restrict__ fixed_rows,
    math::Vec3f* __restrict__ values) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && fixed_rows[row]) {
        values[row] = math::Vec3f(0.0f, 0.0f, 0.0f);
    }
}

__device__ __forceinline__ int packed_lower_index(int row, int col) {
    return row * (row + 1) / 2 + col;
}

__device__ __forceinline__ void solve_body_block12(
    const float* __restrict__ lower,
    const float rhs[kBodyBlock12Dofs],
    float solution[kBodyBlock12Dofs]) {
    float y[kBodyBlock12Dofs];
    #pragma unroll
    for (int i = 0; i < kBodyBlock12Dofs; ++i) {
        float sum = rhs[i];
        #pragma unroll
        for (int j = 0; j < i; ++j) {
            sum -= lower[packed_lower_index(i, j)] * y[j];
        }
        y[i] = sum / lower[packed_lower_index(i, i)];
    }

    #pragma unroll
    for (int i = kBodyBlock12Dofs - 1; i >= 0; --i) {
        float sum = y[i];
        #pragma unroll
        for (int j = i + 1; j < kBodyBlock12Dofs; ++j) {
            sum -= lower[packed_lower_index(j, i)] * solution[j];
        }
        solution[i] = sum / lower[packed_lower_index(i, i)];
    }
}

__global__ void body_block12_copy_dot_kernel(
    const float* __restrict__ lower_factors,
    const math::Vec4i* __restrict__ body_rows,
    const unsigned char* __restrict__ fixed_rows,
    int num_bodies,
    const math::Vec3f* __restrict__ r,
    math::Vec3f* __restrict__ z,
    math::Vec3f* __restrict__ p,
    float* __restrict__ rho) {
    const int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= num_bodies) return;

    const math::Vec4i rows = body_rows[body];
    float rhs[kBodyBlock12Dofs];
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int row = rows[a];
        const math::Vec3f ri = fixed_rows != nullptr && fixed_rows[row]
            ? math::Vec3f(0.0f, 0.0f, 0.0f)
            : r[row];
        rhs[3 * a + 0] = ri.x;
        rhs[3 * a + 1] = ri.y;
        rhs[3 * a + 2] = ri.z;
    }

    float solution[kBodyBlock12Dofs];
    solve_body_block12(
        lower_factors + body * kBodyBlock12PackedLowerSize,
        rhs, solution);

    float local_rho = 0.0f;
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int row = rows[a];
        const math::Vec3f zi = fixed_rows != nullptr && fixed_rows[row]
            ? math::Vec3f(0.0f, 0.0f, 0.0f)
            : math::Vec3f(solution[3 * a + 0],
                          solution[3 * a + 1],
                          solution[3 * a + 2]);
        z[row] = zi;
        p[row] = zi;
        local_rho += rhs[3 * a + 0] * zi.x +
                     rhs[3 * a + 1] * zi.y +
                     rhs[3 * a + 2] * zi.z;
    }
    atomicAdd(rho, local_rho);
}

__global__ void alpha_body_block12_dot_kernel(
    const float* __restrict__ rho,
    const float* __restrict__ sigma,
    const float* __restrict__ lower_factors,
    const math::Vec4i* __restrict__ body_rows,
    const unsigned char* __restrict__ fixed_rows,
    int num_bodies,
    const math::Vec3f* __restrict__ p,
    const math::Vec3f* __restrict__ Ap,
    math::Vec3f* __restrict__ x,
    math::Vec3f* __restrict__ r,
    math::Vec3f* __restrict__ z,
    float* __restrict__ new_rho) {
    const int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= num_bodies) return;

    const float sv = *sigma;
    const float alpha =
        (sv > 1e-37f || sv < -1e-37f) ? (*rho / sv) : 0.0f;
    const math::Vec4i rows = body_rows[body];
    float rhs[kBodyBlock12Dofs];
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int row = rows[a];
        math::Vec3f ri(0.0f, 0.0f, 0.0f);
        if (fixed_rows == nullptr || !fixed_rows[row]) {
            x[row] = x[row] + p[row] * alpha;
            ri = r[row] - Ap[row] * alpha;
            r[row] = ri;
        } else {
            r[row] = math::Vec3f(0.0f, 0.0f, 0.0f);
        }
        rhs[3 * a + 0] = ri.x;
        rhs[3 * a + 1] = ri.y;
        rhs[3 * a + 2] = ri.z;
    }

    float solution[kBodyBlock12Dofs];
    solve_body_block12(
        lower_factors + body * kBodyBlock12PackedLowerSize,
        rhs, solution);

    float local_rho = 0.0f;
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int row = rows[a];
        const math::Vec3f zi = fixed_rows != nullptr && fixed_rows[row]
            ? math::Vec3f(0.0f, 0.0f, 0.0f)
            : math::Vec3f(solution[3 * a + 0],
                          solution[3 * a + 1],
                          solution[3 * a + 2]);
        z[row] = zi;
        local_rho += rhs[3 * a + 0] * zi.x +
                     rhs[3 * a + 1] * zi.y +
                     rhs[3 * a + 2] * zi.z;
    }
    atomicAdd(new_rho, local_rho);
}

__global__ void finalize_body_rho_beta_kernel(float* rho,
                                               const float* new_rho,
                                               float* beta) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float old = *rho;
    const float next = *new_rho;
    *beta = (old > 1e-37f || old < -1e-37f) ? (next / old) : 0.0f;
    *rho = next;
}

template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[BLOCK / 32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    constexpr int kNumWarps = BLOCK / 32;
    val = (threadIdx.x < kNumWarps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) {
        #pragma unroll
        for (int offset = kNumWarps / 2; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
    }
    return val;
}

template <int BLOCK>
__global__ void dot_partial_kernel(const math::Vec3f* __restrict__ a,
                                   const math::Vec3f* __restrict__ b,
                                   float* __restrict__ partial,
                                   int n) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    float local = 0.0f;
    if (i < n) {
        const math::Vec3f av = a[i];
        const math::Vec3f bv = b[i];
        local += av.x * bv.x + av.y * bv.y + av.z * bv.z;
    }
    const float bsum = block_reduce_sum<BLOCK>(local);
    if (threadIdx.x == 0) partial[blockIdx.x] = bsum;
}

template <int BLOCK>
__global__ void dot_masked_partial_kernel(
    const math::Vec3f* __restrict__ a,
    const math::Vec3f* __restrict__ b,
    const unsigned char* __restrict__ fixed_rows,
    float* __restrict__ partial,
    int n) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    float local = 0.0f;
    if (i < n && (fixed_rows == nullptr || !fixed_rows[i])) {
        const math::Vec3f av = a[i];
        const math::Vec3f bv = b[i];
        local = av.x * bv.x + av.y * bv.y + av.z * bv.z;
    }
    const float bsum = block_reduce_sum<BLOCK>(local);
    if (threadIdx.x == 0) partial[blockIdx.x] = bsum;
}

template <int BLOCK>
__global__ void reduce_partial_kernel(const float* __restrict__ partial,
                                      int partial_count,
                                      float* __restrict__ out) {
    float local = 0.0f;
    for (int i = threadIdx.x; i < partial_count; i += BLOCK) {
        local += partial[i];
    }
    const float sum = block_reduce_sum<BLOCK>(local);
    if (threadIdx.x == 0) *out = sum;
}

template <int BLOCK>
__global__ void jacobi_copy_dot_partial_kernel(
    const math::Mat3f* __restrict__ M_inv,
    const math::Vec3f* __restrict__ r,
    math::Vec3f* __restrict__ z,
    math::Vec3f* __restrict__ p,
    float* __restrict__ partial,
    int n) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    float local = 0.0f;
    if (i < n) {
        const math::Vec3f zi = M_inv[i] * r[i];
        z[i] = zi;
        p[i] = zi;
        local += r[i].x * zi.x + r[i].y * zi.y + r[i].z * zi.z;
    }
    const float bsum = block_reduce_sum<BLOCK>(local);
    if (threadIdx.x == 0) partial[blockIdx.x] = bsum;
}

template <int BLOCK>
__global__ void alpha_jacobi_dot_partial_kernel(
    const float* __restrict__ rho,
    const float* __restrict__ sigma,
    const math::Vec3f* __restrict__ p,
    const math::Vec3f* __restrict__ Ap,
    math::Vec3f* __restrict__ x,
    math::Vec3f* __restrict__ r,
    const math::Mat3f* __restrict__ M_inv,
    math::Vec3f* __restrict__ z,
    float* __restrict__ partial,
    int n) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    float local = 0.0f;
    if (i < n) {
        const float sv = *sigma;
        const float alpha =
            (sv > 1e-37f || sv < -1e-37f) ? (*rho / sv) : 0.0f;
        const math::Vec3f xi = x[i] + p[i] * alpha;
        const math::Vec3f ri = r[i] - Ap[i] * alpha;
        const math::Vec3f zi = M_inv[i] * ri;
        x[i] = xi;
        r[i] = ri;
        z[i] = zi;
        local += ri.x * zi.x + ri.y * zi.y + ri.z * zi.z;
    }
    const float bsum = block_reduce_sum<BLOCK>(local);
    if (threadIdx.x == 0) partial[blockIdx.x] = bsum;
}

template <int BLOCK>
__global__ void reduce_rho_beta_kernel(
    const float* __restrict__ partial,
    int partial_count,
    float* __restrict__ rho,
    float* __restrict__ new_rho,
    float* __restrict__ beta) {
    float local = 0.0f;
    for (int i = threadIdx.x; i < partial_count; i += BLOCK) {
        local += partial[i];
    }
    const float sum = block_reduce_sum<BLOCK>(local);
    if (threadIdx.x == 0) {
        const float old = *rho;
        *new_rho = sum;
        *beta = (old > 1e-37f || old < -1e-37f) ? (sum / old) : 0.0f;
        *rho = sum;
    }
}

__global__ void update_p_kernel(
    const float* __restrict__ beta,
    const math::Vec3f* __restrict__ z,
    math::Vec3f* __restrict__ p,
    int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i] = z[i] + p[i] * (*beta);
}

// Issue all PCG kernels onto `stream`.
void emit_pcg(const sparse::BlockCSR3& A,
              DeviceSpan<math::Vec3f> b,
              DeviceSpan<math::Vec3f> x,
              int max_iterations,
              std::uintptr_t cuda_stream,
              const collision::ContactSpMVOp& contact,
              const collision::WideContactSpMVOp& wide_contact,
              const BodyBlock12PreconditionerOp& body_preconditioner,
              CudaArray<math::Vec3f>& r,
              CudaArray<math::Vec3f>& p,
              CudaArray<math::Vec3f>& z,
              CudaArray<math::Vec3f>& Ap,
              CudaArray<math::Mat3f>& M_inv,
              CudaArray<float>& coeff,
              CudaArray<float>& reduction_partial) {
    const int n = A.num_block_rows();
    auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    const int grid = grid_for(n);
    const bool use_body_preconditioner = body_preconditioner.active();

    if (use_body_preconditioner) {
        residual_kernel<<<grid, kBlockDim, 0, stream>>>(
            n,
            A.diag.gpu_data(),
            A.row_offsets.gpu_data(),
            A.col_indices.gpu_data(),
            A.values.gpu_data(),
            x.data(),
            b.data(),
            r.gpu_data());
    } else {
        residual_m_inv_kernel<<<grid, kBlockDim, 0, stream>>>(
            n,
            A.diag.gpu_data(),
            A.row_offsets.gpu_data(),
            A.col_indices.gpu_data(),
            A.values.gpu_data(),
            x.data(),
            b.data(),
            r.gpu_data(),
            M_inv.gpu_data());
    }
    //check_cuda(cudaGetLastError(), "residual_m_inv_kernel launch");

    collision::apply_contact_spmv(contact, x.data(), r.gpu_data(),
                                  n, -1.0f, cuda_stream);
    collision::apply_wide_contact_spmv(wide_contact, x.data(), r.gpu_data(),
                                       n, -1.0f, cuda_stream);
    if (use_body_preconditioner && body_preconditioner.fixed_rows != nullptr) {
        zero_fixed_rows_kernel<<<grid, kBlockDim, 0, stream>>>(
            n, body_preconditioner.fixed_rows, r.gpu_data());
    }

    if (use_body_preconditioner) {
        cudaMemsetAsync(&coeff.gpu_data()[0], 0, sizeof(float), stream);
        const int body_grid = grid_for(body_preconditioner.num_bodies);
        body_block12_copy_dot_kernel<<<body_grid, kBlockDim, 0, stream>>>(
            body_preconditioner.lower_factors,
            body_preconditioner.body_rows,
            body_preconditioner.fixed_rows,
            body_preconditioner.num_bodies,
            r.gpu_data(), z.gpu_data(), p.gpu_data(),
            &coeff.gpu_data()[0]);
    } else {
        jacobi_copy_dot_partial_kernel<kBlockDim><<<grid, kBlockDim, 0, stream>>>(
            M_inv.gpu_data(), r.gpu_data(), z.gpu_data(), p.gpu_data(),
            reduction_partial.gpu_data(), n);
        reduce_partial_kernel<kBlockDim><<<1, kBlockDim, 0, stream>>>(
            reduction_partial.gpu_data(), grid, &coeff.gpu_data()[0]);
    }

    for (int iter = 0; iter < max_iterations; ++iter) {
        sparse::spmv(A,
                     DeviceSpan<math::Vec3f>::from(p),
                     DeviceSpan<math::Vec3f>::from(Ap),
                     1.0f, 0.0f, cuda_stream);
        collision::apply_contact_spmv(contact, p.gpu_data(),
                                      Ap.gpu_data(), n, 1.0f, cuda_stream);
        collision::apply_wide_contact_spmv(wide_contact, p.gpu_data(),
                                           Ap.gpu_data(), n, 1.0f,
                                           cuda_stream);

        dot_partial_kernel<kBlockDim><<<grid, kBlockDim, 0, stream>>>(
            p.gpu_data(), Ap.gpu_data(), reduction_partial.gpu_data(), n);
        reduce_partial_kernel<kBlockDim><<<1, kBlockDim, 0, stream>>>(
            reduction_partial.gpu_data(), grid, &coeff.gpu_data()[1]);

        if (use_body_preconditioner) {
            cudaMemsetAsync(&coeff.gpu_data()[2], 0, sizeof(float), stream);
            const int body_grid = grid_for(body_preconditioner.num_bodies);
            alpha_body_block12_dot_kernel<<<body_grid, kBlockDim, 0, stream>>>(
                &coeff.gpu_data()[0], &coeff.gpu_data()[1],
                body_preconditioner.lower_factors,
                body_preconditioner.body_rows,
                body_preconditioner.fixed_rows,
                body_preconditioner.num_bodies,
                p.gpu_data(), Ap.gpu_data(), x.data(), r.gpu_data(),
                z.gpu_data(), &coeff.gpu_data()[2]);
            finalize_body_rho_beta_kernel<<<1, 1, 0, stream>>>(
                &coeff.gpu_data()[0], &coeff.gpu_data()[2],
                &coeff.gpu_data()[3]);
        } else {
            alpha_jacobi_dot_partial_kernel<kBlockDim><<<
                grid, kBlockDim, 0, stream>>>(
                &coeff.gpu_data()[0], &coeff.gpu_data()[1],
                p.gpu_data(), Ap.gpu_data(),
                x.data(), r.gpu_data(), M_inv.gpu_data(), z.gpu_data(),
                reduction_partial.gpu_data(), n);
            reduce_rho_beta_kernel<kBlockDim><<<1, kBlockDim, 0, stream>>>(
                reduction_partial.gpu_data(), grid,
                &coeff.gpu_data()[0], &coeff.gpu_data()[2],
                &coeff.gpu_data()[3]);
        }

        if (iter + 1 < max_iterations) {
            update_p_kernel<<<grid, kBlockDim, 0, stream>>>(
                &coeff.gpu_data()[3], z.gpu_data(), p.gpu_data(), n);
        }
    }
}

void emit_true_residual(const sparse::BlockCSR3& A,
                        DeviceSpan<math::Vec3f> b,
                        DeviceSpan<math::Vec3f> x,
                        std::uintptr_t cuda_stream,
                        const collision::ContactSpMVOp& contact,
                        const collision::WideContactSpMVOp& wide_contact,
                        const BodyBlock12PreconditionerOp& body_preconditioner,
                        CudaArray<math::Vec3f>& r,
                        CudaArray<float>& reduction_partial,
                        CudaArray<float>& true_residual_norms) {
    const int n = A.num_block_rows();
    auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    const int grid = grid_for(n);
    residual_kernel<<<grid, kBlockDim, 0, stream>>>(
        n,
        A.diag.gpu_data(),
        A.row_offsets.gpu_data(),
        A.col_indices.gpu_data(),
        A.values.gpu_data(),
        x.data(),
        b.data(),
        r.gpu_data());
    collision::apply_contact_spmv(
        contact, x.data(), r.gpu_data(), n, -1.0f, cuda_stream);
    collision::apply_wide_contact_spmv(
        wide_contact, x.data(), r.gpu_data(), n, -1.0f, cuda_stream);
    if (body_preconditioner.fixed_rows != nullptr) {
        zero_fixed_rows_kernel<<<grid, kBlockDim, 0, stream>>>(
            n, body_preconditioner.fixed_rows, r.gpu_data());
    }

    dot_masked_partial_kernel<kBlockDim><<<grid, kBlockDim, 0, stream>>>(
        r.gpu_data(), r.gpu_data(), body_preconditioner.fixed_rows,
        reduction_partial.gpu_data(), n);
    reduce_partial_kernel<kBlockDim><<<1, kBlockDim, 0, stream>>>(
        reduction_partial.gpu_data(), grid,
        &true_residual_norms.gpu_data()[0]);
    dot_masked_partial_kernel<kBlockDim><<<grid, kBlockDim, 0, stream>>>(
        b.data(), b.data(), body_preconditioner.fixed_rows,
        reduction_partial.gpu_data(), n);
    reduce_partial_kernel<kBlockDim><<<1, kBlockDim, 0, stream>>>(
        reduction_partial.gpu_data(), grid,
        &true_residual_norms.gpu_data()[1]);
}

}  // namespace

// ---------------------------------------------------------------------------
// PCGSolver
// ---------------------------------------------------------------------------

void PCGSolver::destroy_graph() noexcept {
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    graph_n_ = 0;
    graph_max_iter_ = 0;
    graph_body_factors_ = nullptr;
    graph_body_rows_ = nullptr;
    graph_fixed_rows_ = nullptr;
    graph_num_bodies_ = 0;
}

PCGSolver::~PCGSolver() { destroy_graph(); }

PCGSolver::PCGSolver(PCGSolver&& o) noexcept
    : num_block_rows_(o.num_block_rows_),
      r_(std::move(o.r_)), p_(std::move(o.p_)),
      z_(std::move(o.z_)), Ap_(std::move(o.Ap_)),
      M_inv_(std::move(o.M_inv_)), coeff_(std::move(o.coeff_)),
      reduction_partial_(std::move(o.reduction_partial_)),
      true_residual_norms_(std::move(o.true_residual_norms_)),
      last_true_residual_valid_(o.last_true_residual_valid_),
      graph_exec_(o.graph_exec_),
      graph_n_(o.graph_n_), graph_max_iter_(o.graph_max_iter_),
      graph_body_factors_(o.graph_body_factors_),
      graph_body_rows_(o.graph_body_rows_),
      graph_fixed_rows_(o.graph_fixed_rows_),
      graph_num_bodies_(o.graph_num_bodies_) {
    o.graph_exec_ = nullptr;
    o.graph_n_ = 0;
    o.graph_max_iter_ = 0;
    o.graph_body_factors_ = nullptr;
    o.graph_body_rows_ = nullptr;
    o.graph_fixed_rows_ = nullptr;
    o.graph_num_bodies_ = 0;
}

PCGSolver& PCGSolver::operator=(PCGSolver&& o) noexcept {
    if (this != &o) {
        destroy_graph();
        num_block_rows_ = o.num_block_rows_;
        r_ = std::move(o.r_); p_ = std::move(o.p_);
        z_ = std::move(o.z_); Ap_ = std::move(o.Ap_);
        M_inv_ = std::move(o.M_inv_); coeff_ = std::move(o.coeff_);
        reduction_partial_ = std::move(o.reduction_partial_);
        true_residual_norms_ = std::move(o.true_residual_norms_);
        last_true_residual_valid_ = o.last_true_residual_valid_;
        graph_exec_ = o.graph_exec_;
        graph_n_ = o.graph_n_; graph_max_iter_ = o.graph_max_iter_;
        graph_body_factors_ = o.graph_body_factors_;
        graph_body_rows_ = o.graph_body_rows_;
        graph_fixed_rows_ = o.graph_fixed_rows_;
        graph_num_bodies_ = o.graph_num_bodies_;
        o.graph_exec_ = nullptr;
        o.graph_n_ = 0; o.graph_max_iter_ = 0;
        o.graph_body_factors_ = nullptr;
        o.graph_body_rows_ = nullptr;
        o.graph_fixed_rows_ = nullptr;
        o.graph_num_bodies_ = 0;
    }
    return *this;
}

void PCGSolver::initialize(int num_block_rows) {
    if (num_block_rows < 0) {
        throw std::invalid_argument("PCGSolver::initialize: negative size");
    }
    destroy_graph();
    if (num_block_rows == num_block_rows_) return;

    r_.resize(num_block_rows);
    p_.resize(num_block_rows);
    z_.resize(num_block_rows);
    Ap_.resize(num_block_rows);
    M_inv_.resize(num_block_rows);
    coeff_.resize(4);
    reduction_partial_.resize(grid_for(num_block_rows));
    true_residual_norms_.resize(2);

    num_block_rows_ = num_block_rows;
}

int PCGSolver::solve(const sparse::BlockCSR3& A,
                     DeviceSpan<math::Vec3f> b,
                     DeviceSpan<math::Vec3f> x,
                     const PCGParams& params,
                     std::uintptr_t cuda_stream,
                     collision::ContactSpMVOp contact,
                     collision::WideContactSpMVOp wide_contact,
                     BodyBlock12PreconditionerOp body_preconditioner) {
    const int n = A.num_block_rows();
    if (n == 0) return 0;

    if (static_cast<int>(b.size()) < n || static_cast<int>(x.size()) < n) {
        throw std::invalid_argument("PCGSolver::solve: b/x shorter than A rows");
    }
    if (static_cast<int>(A.diag.gpu_size()) < n) {
        throw std::invalid_argument(
            "PCGSolver::solve: A.diag has fewer than A.num_block_rows() entries; "
            "call A.build_topology(...) before solving");
    }

    if (n != num_block_rows_) {
        initialize(n);
    }
    if (body_preconditioner.active() &&
        body_preconditioner.num_bodies * 4 != n) {
        throw std::invalid_argument(
            "PCGSolver::solve: body 12x12 preconditioner must cover exactly "
            "four BlockCSR rows per body");
    }

    CHYSX_NVTX_RANGE_COLOUR("pcg::solve", 0xfff1c40f);

    collision::zero_count_ptr();

    const int max_iter = params.max_iterations;
    last_true_residual_valid_ = params.compute_true_residual;
    auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    if (stream == nullptr) {
        emit_pcg(A, b, x, max_iter, cuda_stream, contact,
                 wide_contact, body_preconditioner,
                 r_, p_, z_, Ap_, M_inv_, coeff_,
                 reduction_partial_);
        if (params.compute_true_residual) {
            emit_true_residual(
                A, b, x, cuda_stream, contact, wide_contact,
                body_preconditioner, r_, reduction_partial_,
                true_residual_norms_);
        }
        return max_iter;
    }

    if (!graph_exec_ || graph_n_ != n || graph_max_iter_ != max_iter ||
        graph_body_factors_ != body_preconditioner.lower_factors ||
        graph_body_rows_ != body_preconditioner.body_rows ||
        graph_fixed_rows_ != body_preconditioner.fixed_rows ||
        graph_num_bodies_ != body_preconditioner.num_bodies) {
        destroy_graph();
        cudaGraph_t graph = nullptr;
        cudaError_t err = cudaStreamBeginCapture(
            stream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("PCGSolver::solve: cudaStreamBeginCapture failed: ") +
                cudaGetErrorString(err));
        }
        emit_pcg(A, b, x, max_iter, cuda_stream, contact,
                 wide_contact, body_preconditioner,
                 r_, p_, z_, Ap_, M_inv_, coeff_,
                 reduction_partial_);
        err = cudaStreamEndCapture(stream, &graph);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("PCGSolver::solve: cudaStreamEndCapture failed: ") +
                cudaGetErrorString(err));
        }
        err = cudaGraphInstantiate(&graph_exec_, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph);
        if (err != cudaSuccess) {
            graph_exec_ = nullptr;
            throw std::runtime_error(
                std::string("PCGSolver::solve: cudaGraphInstantiate failed: ") +
                cudaGetErrorString(err));
        }
        graph_n_ = n;
        graph_max_iter_ = max_iter;
        graph_body_factors_ = body_preconditioner.lower_factors;
        graph_body_rows_ = body_preconditioner.body_rows;
        graph_fixed_rows_ = body_preconditioner.fixed_rows;
        graph_num_bodies_ = body_preconditioner.num_bodies;
    }

    cudaError_t err = cudaGraphLaunch(graph_exec_, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("PCGSolver::solve: cudaGraphLaunch failed: ") +
            cudaGetErrorString(err));
    }
    if (params.compute_true_residual) {
        emit_true_residual(
            A, b, x, cuda_stream, contact, wide_contact,
            body_preconditioner, r_, reduction_partial_,
            true_residual_norms_);
    }
    return max_iter;
}

float PCGSolver::last_residual() {
    if (coeff_.gpu_size() == 0) return 0.0f;
    coeff_.copy_to_host();
    if (last_true_residual_valid_) true_residual_norms_.copy_to_host();
    return host_last_residual();
}

void PCGSolver::copy_last_residual_to_host(std::uintptr_t cuda_stream) {
    if (coeff_.gpu_size() == 0) return;
    coeff_.copy_to_host(cuda_stream);
    if (last_true_residual_valid_) {
        true_residual_norms_.copy_to_host(cuda_stream);
    }
}

float PCGSolver::host_last_residual() const noexcept {
    return coeff_.cpu_size() == 0 ? 0.0f : coeff_.cpu_data()[0];
}

float PCGSolver::host_last_true_relative_residual() const noexcept {
    if (!last_true_residual_valid_ || true_residual_norms_.cpu_size() < 2) {
        return -1.0f;
    }
    const float r2 = std::max(0.0f, true_residual_norms_.cpu_data()[0]);
    const float b2 = std::max(0.0f, true_residual_norms_.cpu_data()[1]);
    return std::sqrt(r2 / std::max(b2, 1.0e-30f));
}

}  // namespace solver
}  // namespace chysx
