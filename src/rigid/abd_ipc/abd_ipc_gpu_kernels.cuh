// SPDX-License-Identifier: MIT
// GPU IPC contact pipeline for the ABD solver (handles dense contact like
// libuipc's screw-and-nut, ~10^4-10^5 contacts).
//
//   - update world verts from 12-DOF affine q
//   - turn LBVH candidate pairs into active PT/EE contacts (distance < d_hat)
//   - assemble the FULL contact Hessian (d^2B*g g^T + dB*d^2D), PSD-projected,
//     lifted to per-body 12x12 blocks via the (constant) ABD Jacobian and
//     scattered with atomicAdd. Because x = t + A x_bar is LINEAR in q, no
//     vertex-Hessian correction is needed; the q-space Hessian is J^T H_vert J.
//
// Energy / distance / CCD reductions reuse the DOF-agnostic rigid_ipc kernels.

#pragma once

#include "abd_ipc_types.cuh"
#include "../rigid_ipc/rigid_ipc_kernels.cuh"   // rigid_ipc::GPUContactPair
#include "../../math/vec.cuh"

#include <cuda_runtime.h>

namespace chysx {
namespace abd_ipc {

using rigid_ipc::GPUContactPair;

// World verts from affine q: x = t + A x_bar.
void launch_abd_update_verts(Vec3f* verts, const Vec3f* x_bar,
                             const int* vert_body, const Vec12f* body_q,
                             int n_verts, cudaStream_t stream = 0);

// (point_id, tri_id) candidates -> active PT contacts (type 2) within d_hat.
void launch_abd_filter_pt(const math::Vec2i* pt_pairs, int n_pairs,
                          const Vec3f* verts, const Vec3i* tris,
                          const int* vert_body, float D_hat,
                          GPUContactPair* out, int* count, int max_contacts,
                          cudaStream_t stream = 0);

// (edge_a, edge_b) candidates -> active EE contacts (type 3) within d_hat.
void launch_abd_filter_ee(const math::Vec2i* ee_pairs, int n_pairs,
                          const Vec3f* verts, const math::Vec2i* edges,
                          const int* vert_body, float D_hat,
                          GPUContactPair* out, int* count, int max_contacts,
                          cudaStream_t stream = 0);

// Full-Hessian contact assembly. grad_out[nb] (Vec12f) and hess_diag_out[nb]
// (Mat12f) must be zeroed before. Off-diagonal blocks to fixed bodies vanish
// (their dq=0), so only per-body diagonal blocks are produced.
void launch_abd_assemble(const GPUContactPair* contacts, int n_contacts,
                         const Vec3f* verts, const Vec3f* x_bar,
                         const int* vert_body,
                         float kappa, float D_hat,
                         Vec12f* grad_out, Mat12f* hess_diag_out,
                         int n_bodies, cudaStream_t stream = 0);

void launch_abd_zero_vec12(Vec12f* data, int n, cudaStream_t stream = 0);
void launch_abd_zero_mat12(Mat12f* data, int n, cudaStream_t stream = 0);

}  // namespace abd_ipc
}  // namespace chysx
