// SPDX-License-Identifier: MIT
// GPU CCD candidate generation for ABD-IPC.
//
// Unlike the barrier narrow phase (rigid_ipc_kernels: launch_ef_to_contacts),
// which sub-classifies every EF pair into the closest feature (PP/PE/PT/EE) at
// the *current* pose, continuous collision detection needs the FULL primitives:
//   - point vs whole triangle (type 2)
//   - edge vs whole edge       (type 3)
// because the closest feature changes during the swept motion (a vertex that is
// closest to a triangle corner now may pass through the triangle face over the
// step). Emitting full PT/EE candidates makes the swept CCD tunnel-proof, which
// matches the CPU brute-force compute_toi() it replaces.

#pragma once

#include "../rigid_ipc/rigid_ipc_kernels.cuh"  // rigid_ipc::GPUContactPair
#include "../../math/vec.cuh"

#include <cuda_runtime.h>

namespace chysx {
namespace abd_ipc {

using rigid_ipc::GPUContactPair;

// EF broadphase pairs -> full PT + EE CCD candidates (cross-body, within margin).
void launch_ef_to_ccd_candidates(
    const math::Vec2i* ef_pairs, int n_ef,
    const math::Vec3f* verts,
    const math::Vec3i* faces, const math::Vec2i* edges,
    const int* vert_in_edge, const math::Vec3i* edge_in_face,
    const int* vert_body,
    float margin_sq,
    GPUContactPair* out, int* d_count, int max_contacts,
    cudaStream_t stream = 0);

// Supplementary adjacent VF pairs -> full PT candidates.
void launch_adj_vf_ccd(
    const math::Vec4i* adj_vf, int n,
    const math::Vec3f* verts, const int* vert_body,
    float margin_sq,
    GPUContactPair* out, int* d_count, int max_contacts,
    cudaStream_t stream = 0);

// Supplementary adjacent EE pairs -> full EE candidates.
void launch_adj_ee_ccd(
    const math::Vec2i* adj_ee, int n,
    const math::Vec3f* verts, const math::Vec2i* edges, const int* vert_body,
    float margin_sq,
    GPUContactPair* out, int* d_count, int max_contacts,
    cudaStream_t stream = 0);

// ---- Brute-force GPU CCD (complete, matches the CPU compute_toi exactly) ----
//
// Parallelizes the O(nv*nt + ne^2) inter-body point-triangle / edge-edge sweep:
// one thread per primitive pair computes its time-of-impact and atomically
// min-reduces into *d_alpha (initialize to 1.0 before launch). This is the
// ground-truth path used when broadphase candidate generation is not trusted
// to be complete for CCD.
void launch_ccd_brute_pt(
    const math::Vec3f* verts_cur, const math::Vec3f* verts_next,
    const math::Vec3i* tris, const int* vert_body,
    int n_verts, int n_tris, float d_min,
    float* d_alpha, cudaStream_t stream = 0);

void launch_ccd_brute_ee(
    const math::Vec3f* verts_cur, const math::Vec3f* verts_next,
    const math::Vec2i* edges, const int* vert_body,
    int n_edges, float d_min,
    float* d_alpha, cudaStream_t stream = 0);

}  // namespace abd_ipc
}  // namespace chysx
