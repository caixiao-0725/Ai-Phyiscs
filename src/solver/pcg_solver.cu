// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::solver::PCGSolver.

#include "pcg_solver.h"

#include <cuda_runtime.h>

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
__global__ void dot_single_block_kernel(const math::Vec3f* __restrict__ a,
                                        const math::Vec3f* __restrict__ b,
                                        float* __restrict__ out,
                                        int n) {
    float local = 0.0f;
    for (int i = threadIdx.x; i < n; i += BLOCK) {
        const math::Vec3f av = a[i];
        const math::Vec3f bv = b[i];
        local += av.x * bv.x + av.y * bv.y + av.z * bv.z;
    }
    const float bsum = block_reduce_sum<BLOCK>(local);
    if (threadIdx.x == 0) *out = bsum;
}

template <int BLOCK>
__global__ void jacobi_copy_dot_single_block_kernel(
    const math::Mat3f* __restrict__ M_inv,
    const math::Vec3f* __restrict__ r,
    math::Vec3f* __restrict__ z,
    math::Vec3f* __restrict__ p,
    float* __restrict__ out,
    int n) {
    float local = 0.0f;
    for (int i = threadIdx.x; i < n; i += BLOCK) {
        const math::Vec3f zi = M_inv[i] * r[i];
        z[i] = zi;
        p[i] = zi;
        local += r[i].x * zi.x + r[i].y * zi.y + r[i].z * zi.z;
    }
    const float bsum = block_reduce_sum<BLOCK>(local);
    if (threadIdx.x == 0) *out = bsum;
}

template <int BLOCK>
__global__ void jacobi_dot_single_block_kernel(
    const math::Mat3f* __restrict__ M_inv,
    const math::Vec3f* __restrict__ r,
    math::Vec3f* __restrict__ z,
    float* __restrict__ out,
    int n) {
    float local = 0.0f;
    for (int i = threadIdx.x; i < n; i += BLOCK) {
        const math::Vec3f zi = M_inv[i] * r[i];
        z[i] = zi;
        local += r[i].x * zi.x + r[i].y * zi.y + r[i].z * zi.z;
    }
    const float bsum = block_reduce_sum<BLOCK>(local);
    if (threadIdx.x == 0) *out = bsum;
}

__global__ void alpha_update_x_r_kernel(const float* __restrict__ rho,
                                        const float* __restrict__ sigma,
                                        const math::Vec3f* __restrict__ p,
                                        const math::Vec3f* __restrict__ Ap,
                                        math::Vec3f* __restrict__ x,
                                        math::Vec3f* __restrict__ r,
                                        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float sv = *sigma;
    const float alpha =
        (sv > 1e-37f || sv < -1e-37f) ? (*rho / sv) : 0.0f;
    x[i] = x[i] + p[i] * alpha;
    r[i] = r[i] - Ap[i] * alpha;
}

__global__ void update_p_copy_rho_single_block_kernel(
    const float* __restrict__ old_rho,
    const float* __restrict__ new_rho,
    float* __restrict__ rho,
    const math::Vec3f* __restrict__ z,
    math::Vec3f* __restrict__ p,
    int n) {
    const float ov = *old_rho;
    const float beta =
        (ov > 1e-37f || ov < -1e-37f) ? (*new_rho / ov) : 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        p[i] = z[i] + p[i] * beta;
    }
    __syncthreads();
    if (threadIdx.x == 0) *rho = *new_rho;
}

// Issue all PCG kernels onto `stream`.
void emit_pcg(const sparse::BlockCSR3& A,
              DeviceSpan<math::Vec3f> b,
              DeviceSpan<math::Vec3f> x,
              int max_iterations,
              std::uintptr_t cuda_stream,
              const collision::ContactSpMVOp& contact,
              CudaArray<math::Vec3f>& r,
              CudaArray<math::Vec3f>& p,
              CudaArray<math::Vec3f>& z,
              CudaArray<math::Vec3f>& Ap,
              CudaArray<math::Mat3f>& M_inv,
              CudaArray<float>& coeff) {
    const int n = A.num_block_rows();
    auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    const int grid = grid_for(n);

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
    //check_cuda(cudaGetLastError(), "residual_m_inv_kernel launch");

    collision::apply_contact_spmv(contact, x.data(), r.gpu_data(),
                                  n, -1.0f, cuda_stream);

    jacobi_copy_dot_single_block_kernel<kBlockDim><<<1, kBlockDim, 0, stream>>>(
        M_inv.gpu_data(), r.gpu_data(), z.gpu_data(), p.gpu_data(),
        &coeff.gpu_data()[0], n);
    //check_cuda(cudaGetLastError(), "jacobi_copy_dot_single_block_kernel launch");

    for (int iter = 0; iter < max_iterations; ++iter) {
        sparse::spmv(A,
                     DeviceSpan<math::Vec3f>::from(p),
                     DeviceSpan<math::Vec3f>::from(Ap),
                     1.0f, 0.0f, cuda_stream);
        collision::apply_contact_spmv(contact, p.gpu_data(),
                                      Ap.gpu_data(), n, 1.0f, cuda_stream);

        dot_single_block_kernel<kBlockDim><<<1, kBlockDim, 0, stream>>>(
            p.gpu_data(), Ap.gpu_data(), &coeff.gpu_data()[1], n);

        alpha_update_x_r_kernel<<<grid, kBlockDim, 0, stream>>>(
            &coeff.gpu_data()[0], &coeff.gpu_data()[1],
            p.gpu_data(), Ap.gpu_data(),
            x.data(), r.gpu_data(), n);

        jacobi_dot_single_block_kernel<kBlockDim><<<1, kBlockDim, 0, stream>>>(
            M_inv.gpu_data(), r.gpu_data(), z.gpu_data(),
            &coeff.gpu_data()[2], n);

        update_p_copy_rho_single_block_kernel<<<1, kBlockDim, 0, stream>>>(
            &coeff.gpu_data()[0], &coeff.gpu_data()[2],
            &coeff.gpu_data()[0], z.gpu_data(), p.gpu_data(), n);
    }
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
}

PCGSolver::~PCGSolver() { destroy_graph(); }

PCGSolver::PCGSolver(PCGSolver&& o) noexcept
    : num_block_rows_(o.num_block_rows_),
      r_(std::move(o.r_)), p_(std::move(o.p_)),
      z_(std::move(o.z_)), Ap_(std::move(o.Ap_)),
      M_inv_(std::move(o.M_inv_)), coeff_(std::move(o.coeff_)),
      graph_exec_(o.graph_exec_),
      graph_n_(o.graph_n_), graph_max_iter_(o.graph_max_iter_) {
    o.graph_exec_ = nullptr;
    o.graph_n_ = 0;
    o.graph_max_iter_ = 0;
}

PCGSolver& PCGSolver::operator=(PCGSolver&& o) noexcept {
    if (this != &o) {
        destroy_graph();
        num_block_rows_ = o.num_block_rows_;
        r_ = std::move(o.r_); p_ = std::move(o.p_);
        z_ = std::move(o.z_); Ap_ = std::move(o.Ap_);
        M_inv_ = std::move(o.M_inv_); coeff_ = std::move(o.coeff_);
        graph_exec_ = o.graph_exec_;
        graph_n_ = o.graph_n_; graph_max_iter_ = o.graph_max_iter_;
        o.graph_exec_ = nullptr;
        o.graph_n_ = 0; o.graph_max_iter_ = 0;
    }
    return *this;
}

void PCGSolver::initialize(int num_block_rows) {
    if (num_block_rows < 0) {
        throw std::invalid_argument("PCGSolver::initialize: negative size");
    }
    if (num_block_rows == num_block_rows_) return;

    r_.resize(num_block_rows);
    p_.resize(num_block_rows);
    z_.resize(num_block_rows);
    Ap_.resize(num_block_rows);
    M_inv_.resize(num_block_rows);
    coeff_.resize(4);

    num_block_rows_ = num_block_rows;
}

int PCGSolver::solve(const sparse::BlockCSR3& A,
                     DeviceSpan<math::Vec3f> b,
                     DeviceSpan<math::Vec3f> x,
                     const PCGParams& params,
                     std::uintptr_t cuda_stream,
                     collision::ContactSpMVOp contact) {
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
        destroy_graph();
    }

    CHYSX_NVTX_RANGE_COLOUR("pcg::solve", 0xfff1c40f);

    collision::zero_count_ptr();

    const int max_iter = params.max_iterations;

    destroy_graph();
    emit_pcg(A, b, x, max_iter, cuda_stream, contact,
             r_, p_, z_, Ap_, M_inv_, coeff_);
    return max_iter;
}

float PCGSolver::last_residual() {
    if (coeff_.gpu_size() == 0) return 0.0f;
    coeff_.copy_to_host();
    return coeff_.cpu_data()[0];
}

}  // namespace solver
}  // namespace chysx
