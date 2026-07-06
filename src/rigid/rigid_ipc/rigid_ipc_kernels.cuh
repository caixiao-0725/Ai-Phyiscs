// SPDX-License-Identifier: MIT
// CUDA kernel declarations for Rigid-IPC solver.
//
// All heavy per-vertex/per-contact work runs on GPU.
// Body-level operations (2 bodies = tiny) run on CPU.

#pragma once

#include "rigid_ipc_types.cuh"
#include "rigid_ipc_rodrigues.cuh"
#include "rigid_ipc_assembly.cuh"
#include "../../rigid/abd_ipc/abd_ipc_barrier.cuh"
#include "../../rigid/abd_ipc/abd_ipc_distance.cuh"
#include "../../rigid/abd_ipc/abd_ipc_ccd.cuh"
#include "../../math/vec.cuh"

#include <cuda_runtime.h>

namespace chysx {
namespace rigid_ipc {

// GPU-friendly contact pair (POD, no enum class)
// type: 0=PP, 1=PE, 2=PT, 3=EE  (matches IPC Toolkit conventions)
struct GPUContactPair {
    int type;
    int v[4];
    int body[4];
};

// ---- GPU Narrow Phase ------------------------------------------------------

// From EF broadphase pairs, generate VF and EE contact candidates on GPU.
// Writes compacted GPUContactPair list to `contacts_out`.
// `d_count` is an atomic counter (device int*), initialized to 0 before call.
// `max_contacts` caps the output to avoid overflow.
void launch_ef_to_contacts(
    const math::Vec2i* ef_pairs, int n_ef_pairs,
    const math::Vec3f* verts,
    const math::Vec3i* faces, const math::Vec2i* edges,
    const int* vert_in_edge, const math::Vec3i* edge_in_face,
    const int* vert_body,
    float d_hat_sq,
    GPUContactPair* contacts_out, int* d_count, int max_contacts,
    cudaStream_t stream = 0);

// Supplementary adjacent VF contacts (pre-computed pairs).
void launch_adj_vf_contacts(
    const math::Vec4i* adj_vf_pairs, int n_adj_vf,
    const math::Vec3f* verts, const int* vert_body,
    float d_hat_sq,
    GPUContactPair* contacts_out, int* d_count, int max_contacts,
    cudaStream_t stream = 0);

// Supplementary adjacent EE contacts (pre-computed edge-id pairs).
void launch_adj_ee_contacts(
    const math::Vec2i* adj_ee_pairs, int n_adj_ee,
    const math::Vec3f* verts,
    const math::Vec2i* edges, const int* vert_body,
    float d_hat_sq,
    GPUContactPair* contacts_out, int* d_count, int max_contacts,
    cudaStream_t stream = 0);

// Read the atomic contact count from device.
int read_contact_count(const int* d_count, cudaStream_t stream = 0);

// Zero a single device int (for atomic counter reset).
void launch_zero_int(int* d_val, cudaStream_t stream = 0);

// ---- Kernel launches -------------------------------------------------------

void launch_update_verts(Vec3f* verts, const Vec3f* x_bar,
                          const int* vert_body, const Vec6f* body_q,
                          int n_verts, cudaStream_t stream = 0);

void launch_trial_verts(Vec3f* verts_trial, const Vec3f* x_bar,
                         const int* vert_body, const Vec6f* body_q,
                         const Vec6f* dq, float alpha,
                         int n_verts, cudaStream_t stream = 0);

// Assemble contact barrier gradient + Hessian (diagonal and off-diagonal).
// hess_diag_out: [n_bodies]           diagonal 6×6 blocks (atomicAdd)
// hess_offdiag_out: [n_contacts * 2]  off-diagonal 6×6 blocks per contact
//   For contact c involving body_A and body_B:
//     hess_offdiag_out[2*c + 0] = H[body_A, body_B]  (6×6)
//     hess_offdiag_out[2*c + 1] = H[body_B, body_A]  (6×6)
// offdiag_body_pairs: [n_contacts]  Vec2i(body_A, body_B) per contact
//   Written by this kernel so CPU/GPU knows the body pair for each off-diag block.
void launch_assemble_contacts(
    const GPUContactPair* contacts, const Vec3f* verts,
    const Vec3f* x_bar, const int* vert_body,
    const Vec6f* body_q,
    Vec6f* grad_out, Mat6f* hess_diag_out,
    Mat6f* hess_offdiag_out, math::Vec2i* offdiag_body_pairs,
    float* barrier_energy,
    float kappa, float D_hat,
    int n_contacts, cudaStream_t stream = 0);

void launch_ccd_contacts(
    const GPUContactPair* contacts,
    const Vec3f* verts_cur, const Vec3f* verts_next,
    float* alphas, float d_min,
    int n_contacts, cudaStream_t stream = 0);

void launch_barrier_energy(
    const GPUContactPair* contacts, const Vec3f* verts,
    float* energies, float kappa, float D_hat,
    int n_contacts, cudaStream_t stream = 0);

// ---- GPU block-sparse matvec: y = H*x  ------------------------------------
// y[i] = H_diag[i]*x[i] + Σ_c H_offdiag[...]*x[...]
// offdiag blocks stored as: hess_offdiag[2*c+0] = H[A,B], hess_offdiag[2*c+1] = H[B,A]
// body_pairs[c] = (A,B)
void launch_block_sparse_matvec(
    const Mat6f* H_diag,
    const Mat6f* hess_offdiag, const math::Vec2i* body_pairs,
    int n_offdiag_pairs,
    const Vec6f* x, Vec6f* y,
    int n_bodies, cudaStream_t stream = 0);

// ---- GPU PCG solver --------------------------------------------------------
// Solves H * dx = rhs on GPU.  Returns number of iterations.
// All arrays are device pointers.  dx is zeroed and overwritten.
// Needs scratch: r, z, p, Hp — each [n_bodies] Vec6f, caller-allocated.
int launch_pcg6_gpu(
    const Mat6f* H_diag,
    const Mat6f* hess_offdiag, const math::Vec2i* body_pairs,
    int n_offdiag_pairs,
    const Vec6f* rhs, Vec6f* dx,
    Vec6f* r, Vec6f* z, Vec6f* p, Vec6f* Hp,
    int n_bodies, float tol, int max_iter,
    cudaStream_t stream = 0);

// Compute per-contact squared distance D (for min-distance tracking / kappa update).
// Writes D[ci] for each contact. Inactive contacts (D >= D_hat) get D_hat.
void launch_compute_contact_distances(
    const GPUContactPair* contacts, const Vec3f* verts,
    float* dist_sq_out, float D_hat,
    int n_contacts, cudaStream_t stream = 0);

// ---- Reductions -----------------------------------------------------------

float gpu_reduce_sum(const float* d_data, int n, cudaStream_t stream = 0);
float gpu_reduce_min(const float* d_data, int n, cudaStream_t stream = 0);

// Clamp off-diagonal Hessian blocks for locked DOFs.
// dof_mask: [n_bodies] Vec6f where mask[d]=0 for locked, 1 for free.
void launch_clamp_offdiag_dofs(
    Mat6f* hess_offdiag, const math::Vec2i* body_pairs,
    const Vec6f* dof_mask,
    int n_pairs, cudaStream_t stream = 0);

// ---- Utility kernels ------------------------------------------------------

void launch_zero_vec6(Vec6f* data, int n, cudaStream_t stream = 0);
void launch_zero_mat6(Mat6f* data, int n, cudaStream_t stream = 0);

}  // namespace rigid_ipc
}  // namespace chysx
