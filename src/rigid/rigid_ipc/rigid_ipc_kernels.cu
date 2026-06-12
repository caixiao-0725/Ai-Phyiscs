// SPDX-License-Identifier: MIT
// CUDA kernels for Rigid-IPC solver.
//
// Every per-vertex / per-contact operation runs as a GPU kernel.
// Body-level aggregation (2 bodies = tiny) stays on CPU but operates
// on device data via small H2D/D2H transfers of 6x6 blocks.

#include "rigid_ipc_kernels.cuh"

#include <cub/cub.cuh>

namespace chysx {
namespace rigid_ipc {

// ============================================================================
// GPU Narrow Phase: closest-point helpers (device-side)
// ============================================================================

__device__ float gpu_closest_point_triangle_dist2(
    Vec3f p, Vec3f a, Vec3f b, Vec3f c) {
    Vec3f ab = b - a, ac = c - a, ap = p - a;
    float d1 = math::dot(ab, ap), d2 = math::dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) { Vec3f d = p - a; return math::dot(d, d); }

    Vec3f bp = p - b;
    float d3 = math::dot(ab, bp), d4 = math::dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) { Vec3f d = p - b; return math::dot(d, d); }

    float vc = d1*d4 - d3*d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        float v = d1 / (d1 - d3);
        Vec3f cp = a + ab * v; Vec3f d = p - cp; return math::dot(d, d);
    }

    Vec3f cpp = p - c;
    float d5 = math::dot(ab, cpp), d6 = math::dot(ac, cpp);
    if (d6 >= 0 && d5 <= d6) { Vec3f d = p - c; return math::dot(d, d); }

    float vb = d5*d2 - d1*d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        float ww = d2 / (d2 - d6);
        Vec3f cp = a + ac * ww; Vec3f d = p - cp; return math::dot(d, d);
    }

    float va = d3*d6 - d5*d4;
    if (va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0) {
        float ww = (d4-d3) / ((d4-d3)+(d5-d6));
        Vec3f cp = b + (c - b) * ww; Vec3f d = p - cp; return math::dot(d, d);
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom, ww = vc * denom;
    Vec3f cp = a + ab * v + ac * ww;
    Vec3f d = p - cp;
    return math::dot(d, d);
}

__device__ void gpu_closest_point_segments(
    Vec3f p0, Vec3f p1, Vec3f q0, Vec3f q1,
    float& s, float& t) {
    Vec3f d1 = p1 - p0, d2 = q1 - q0, r = p0 - q0;
    float a = math::dot(d1, d1), e = math::dot(d2, d2);
    float f = math::dot(d2, r);
    const float eps = 1e-12f;

    if (a <= eps && e <= eps) { s = 0; t = 0; return; }
    if (a <= eps) {
        s = 0; t = fminf(fmaxf(f / e, 0.0f), 1.0f);
    } else {
        float c = math::dot(d1, r);
        if (e <= eps) {
            t = 0; s = fminf(fmaxf(-c / a, 0.0f), 1.0f);
        } else {
            float b = math::dot(d1, d2);
            float denom = a*e - b*b;
            s = (denom != 0) ? fminf(fmaxf((b*f - c*e) / denom, 0.0f), 1.0f) : 0;
            t = (b*s + f) / e;
            if (t < 0) { t = 0; s = fminf(fmaxf(-c / a, 0.0f), 1.0f); }
            else if (t > 1) { t = 1; s = fminf(fmaxf((b - c) / a, 0.0f), 1.0f); }
        }
    }
}

__device__ float gpu_segment_dist2(Vec3f p0, Vec3f p1, Vec3f q0, Vec3f q1,
                                    float& s_out, float& t_out) {
    gpu_closest_point_segments(p0, p1, q0, q1, s_out, t_out);
    Vec3f d1 = p1 - p0, d2 = q1 - q0;
    Vec3f cp = p0 + d1 * s_out;
    Vec3f cq = q0 + d2 * t_out;
    Vec3f diff = cp - cq;
    return math::dot(diff, diff);
}

// Emit one GPUContactPair atomically.
__device__ void emit_contact(
    GPUContactPair* out, int* d_count, int max_contacts,
    int type, int v0, int v1, int v2, int v3,
    int b0, int b1, int b2, int b3) {
    int idx = atomicAdd(d_count, 1);
    if (idx >= max_contacts) return;
    GPUContactPair cp;
    cp.type = type;
    cp.v[0] = v0; cp.v[1] = v1; cp.v[2] = v2; cp.v[3] = v3;
    cp.body[0] = b0; cp.body[1] = b1; cp.body[2] = b2; cp.body[3] = b3;
    out[idx] = cp;
}

// ============================================================================
// Kernel: EF → VF + EE decomposition with distance filter (one thread per EF)
// ============================================================================

__global__ void kernel_ef_to_contacts(
    const math::Vec2i* __restrict__ ef_pairs, int n_ef,
    const Vec3f* __restrict__ verts,
    const math::Vec3i* __restrict__ faces,
    const math::Vec2i* __restrict__ edges,
    const int* __restrict__ vert_in_edge,
    const math::Vec3i* __restrict__ edge_in_face,
    const int* __restrict__ vert_body,
    float d_hat_sq,
    GPUContactPair* contacts_out, int* d_count, int max_contacts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_ef) return;

    int eid = ef_pairs[i].x;
    int fid = ef_pairs[i].y;

    // ---- VF: vert_in_edge[eid] vs face[fid] ----
    int vid = vert_in_edge[eid];
    if (vid >= 0) {
        math::Vec3i f = faces[fid];
        if (vid != f.x && vid != f.y && vid != f.z) {
            int bv = vert_body[vid];
            int bf0 = vert_body[f.x], bf1 = vert_body[f.y], bf2 = vert_body[f.z];
            if (!(bv == bf0 && bv == bf1 && bv == bf2)) {
                float d2 = gpu_closest_point_triangle_dist2(
                    verts[vid], verts[f.x], verts[f.y], verts[f.z]);
                if (d2 > 1e-24f && d2 < d_hat_sq) {
                    emit_contact(contacts_out, d_count, max_contacts,
                                 0, vid, f.x, f.y, f.z,
                                 bv, bf0, bf1, bf2);
                }
            }
        }
    }

    // ---- EE: edge[eid] vs each of face[fid]'s 3 edges ----
    math::Vec3i e3 = edge_in_face[fid];
    math::Vec2i ea = edges[eid];

    for (int j = 0; j < 3; ++j) {
        int oeid = e3.data[j];
        if (oeid < 0 || oeid == eid) continue;
        if (oeid >= eid) continue;  // dedup: only emit when oeid < eid

        math::Vec2i eb = edges[oeid];
        if (ea.x == eb.x || ea.x == eb.y || ea.y == eb.x || ea.y == eb.y)
            continue;

        int ba0 = vert_body[ea.x], ba1 = vert_body[ea.y];
        int bb0 = vert_body[eb.x], bb1 = vert_body[eb.y];
        if (ba0 == bb0 && ba0 == bb1 && ba1 == bb0 && ba1 == bb1)
            continue;

        float s, t;
        float d2 = gpu_segment_dist2(verts[ea.x], verts[ea.y],
                                      verts[eb.x], verts[eb.y], s, t);

        // Interior check + degeneracy filter
        if (s > 0.0f && s < 1.0f && t > 0.0f && t < 1.0f && d2 > 1e-24f && d2 < d_hat_sq) {
            Vec3f d1v = verts[ea.y] - verts[ea.x];
            Vec3f d2v = verts[eb.y] - verts[eb.x];
            Vec3f cross_d = math::cross(d1v, d2v);
            float cross_sq = math::dot(cross_d, cross_d);
            float len_sq1 = math::dot(d1v, d1v), len_sq2 = math::dot(d2v, d2v);
            if (cross_sq >= 1e-3f * len_sq1 * len_sq2) {
                emit_contact(contacts_out, d_count, max_contacts,
                             1, ea.x, ea.y, eb.x, eb.y,
                             ba0, ba1, bb0, bb1);
            }
        }
    }
}

void launch_ef_to_contacts(
    const math::Vec2i* ef_pairs, int n_ef,
    const math::Vec3f* verts,
    const math::Vec3i* faces, const math::Vec2i* edges,
    const int* vert_in_edge, const math::Vec3i* edge_in_face,
    const int* vert_body,
    float d_hat_sq,
    GPUContactPair* contacts_out, int* d_count, int max_contacts,
    cudaStream_t stream) {
    if (n_ef == 0) return;
    int block = 256;
    int grid = (n_ef + block - 1) / block;
    kernel_ef_to_contacts<<<grid, block, 0, stream>>>(
        ef_pairs, n_ef, verts, faces, edges, vert_in_edge, edge_in_face,
        vert_body, d_hat_sq, contacts_out, d_count, max_contacts);
}

// ============================================================================
// Kernel: supplementary adjacent VF contacts
// ============================================================================

__global__ void kernel_adj_vf_contacts(
    const math::Vec4i* __restrict__ adj_vf, int n,
    const Vec3f* __restrict__ verts, const int* __restrict__ vert_body,
    float d_hat_sq,
    GPUContactPair* contacts_out, int* d_count, int max_contacts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    math::Vec4i p = adj_vf[i];
    int vid = p.x, f0 = p.y, f1 = p.z, f2 = p.w;

    float d2 = gpu_closest_point_triangle_dist2(
        verts[vid], verts[f0], verts[f1], verts[f2]);

    if (d2 > 1e-24f && d2 < d_hat_sq) {
        int bv = vert_body[vid];
        int b0 = vert_body[f0], b1 = vert_body[f1], b2 = vert_body[f2];
        if (!(bv == b0 && bv == b1 && bv == b2)) {
            emit_contact(contacts_out, d_count, max_contacts,
                         0, vid, f0, f1, f2, bv, b0, b1, b2);
        }
    }
}

void launch_adj_vf_contacts(
    const math::Vec4i* adj_vf, int n,
    const math::Vec3f* verts, const int* vert_body,
    float d_hat_sq,
    GPUContactPair* contacts_out, int* d_count, int max_contacts,
    cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_adj_vf_contacts<<<grid, block, 0, stream>>>(
        adj_vf, n, verts, vert_body, d_hat_sq, contacts_out, d_count, max_contacts);
}

// ============================================================================
// Kernel: supplementary adjacent EE contacts
// ============================================================================

__global__ void kernel_adj_ee_contacts(
    const math::Vec2i* __restrict__ adj_ee, int n,
    const Vec3f* __restrict__ verts,
    const math::Vec2i* __restrict__ edges, const int* __restrict__ vert_body,
    float d_hat_sq,
    GPUContactPair* contacts_out, int* d_count, int max_contacts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    math::Vec2i pr = adj_ee[i];
    math::Vec2i ea = edges[pr.x], eb = edges[pr.y];

    if (ea.x == eb.x || ea.x == eb.y || ea.y == eb.x || ea.y == eb.y)
        return;

    float s, t;
    float d2 = gpu_segment_dist2(verts[ea.x], verts[ea.y],
                                  verts[eb.x], verts[eb.y], s, t);

    if (s > 0.0f && s < 1.0f && t > 0.0f && t < 1.0f && d2 > 1e-24f && d2 < d_hat_sq) {
        Vec3f d1v = verts[ea.y] - verts[ea.x];
        Vec3f d2v = verts[eb.y] - verts[eb.x];
        Vec3f cross_d = math::cross(d1v, d2v);
        float cross_sq = math::dot(cross_d, cross_d);
        float len_sq1 = math::dot(d1v, d1v), len_sq2 = math::dot(d2v, d2v);
        if (cross_sq >= 1e-3f * len_sq1 * len_sq2) {
            int ba0 = vert_body[ea.x], ba1 = vert_body[ea.y];
            int bb0 = vert_body[eb.x], bb1 = vert_body[eb.y];
            if (!(ba0 == bb0 && ba0 == bb1 && ba1 == bb0 && ba1 == bb1)) {
                emit_contact(contacts_out, d_count, max_contacts,
                             1, ea.x, ea.y, eb.x, eb.y,
                             ba0, ba1, bb0, bb1);
            }
        }
    }
}

void launch_adj_ee_contacts(
    const math::Vec2i* adj_ee, int n,
    const math::Vec3f* verts,
    const math::Vec2i* edges, const int* vert_body,
    float d_hat_sq,
    GPUContactPair* contacts_out, int* d_count, int max_contacts,
    cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_adj_ee_contacts<<<grid, block, 0, stream>>>(
        adj_ee, n, verts, edges, vert_body, d_hat_sq, contacts_out, d_count, max_contacts);
}

// ============================================================================
// Utility: read atomic counter, zero counter
// ============================================================================

int read_contact_count(const int* d_count, cudaStream_t stream) {
    int count;
    cudaMemcpyAsync(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return count;
}

__global__ void kernel_zero_int(int* val) { *val = 0; }

void launch_zero_int(int* d_val, cudaStream_t stream) {
    kernel_zero_int<<<1, 1, 0, stream>>>(d_val);
}

// ============================================================================
// Kernel: update surface vertices  (one thread per vertex)
// x_world = R(theta) * x_bar + p
// ============================================================================

__global__ void kernel_update_verts(
    Vec3f* __restrict__ verts,
    const Vec3f* __restrict__ x_bar,
    const int* __restrict__ vert_body,
    const Vec6f* __restrict__ body_q,
    int n_verts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_verts) return;

    int bid = vert_body[i];
    Vec3f p = pose_position(body_q[bid]);
    Vec3f theta = pose_rotation(body_q[bid]);
    Mat3f R = rodrigues(theta);
    verts[i] = p + R * x_bar[i];
}

void launch_update_verts(Vec3f* verts, const Vec3f* x_bar,
                          const int* vert_body, const Vec6f* body_q,
                          int n_verts, cudaStream_t stream) {
    if (n_verts == 0) return;
    int block = 256;
    int grid = (n_verts + block - 1) / block;
    kernel_update_verts<<<grid, block, 0, stream>>>(
        verts, x_bar, vert_body, body_q, n_verts);
}

// ============================================================================
// Kernel: compute trial vertices for line search  (one thread per vertex)
// ============================================================================

__global__ void kernel_trial_verts(
    Vec3f* __restrict__ verts_trial,
    const Vec3f* __restrict__ x_bar,
    const int* __restrict__ vert_body,
    const Vec6f* __restrict__ body_q,
    const Vec6f* __restrict__ dq,
    float alpha,
    int n_verts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_verts) return;

    int bid = vert_body[i];
    Vec6f q_trial = body_q[bid] + dq[bid] * alpha;
    Vec3f p = pose_position(q_trial);
    Vec3f theta = pose_rotation(q_trial);
    Mat3f R = rodrigues(theta);
    verts_trial[i] = p + R * x_bar[i];
}

void launch_trial_verts(Vec3f* verts_trial, const Vec3f* x_bar,
                         const int* vert_body, const Vec6f* body_q,
                         const Vec6f* dq, float alpha,
                         int n_verts, cudaStream_t stream) {
    if (n_verts == 0) return;
    int block = 256;
    int grid = (n_verts + block - 1) / block;
    kernel_trial_verts<<<grid, block, 0, stream>>>(
        verts_trial, x_bar, vert_body, body_q, dq, alpha, n_verts);
}

// ============================================================================
// Kernel: per-contact barrier gradient + Hessian accumulation
//
// Each thread processes one ContactPair.  Gradient and Hessian
// contributions are atomically added to per-body arrays.
// ============================================================================

__device__ void atomic_add_vec6(Vec6f* dst, const Vec6f& v) {
    for (int k = 0; k < 6; ++k)
        atomicAdd(&dst->data[k], v.data[k]);
}

__device__ void atomic_add_mat6(Mat6f* dst, const Mat6f& m) {
    for (int k = 0; k < 36; ++k)
        atomicAdd(&dst->data[k], m.data[k]);
}

__global__ void kernel_assemble_contacts(
    const GPUContactPair* __restrict__ contacts,
    const Vec3f* __restrict__ verts,
    const Vec3f* __restrict__ x_bar,
    const int* __restrict__ vert_body,
    const Vec6f* __restrict__ body_q,
    Vec6f* __restrict__ grad_out,       // [n_bodies]
    Mat6f* __restrict__ hess_diag_out,  // [n_bodies]
    float* __restrict__ barrier_energy, // [n_contacts] for reduction
    float kappa, float D_hat,
    int n_contacts)
{
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= n_contacts) return;

    GPUContactPair cp = contacts[ci];

    Vec3f v0 = verts[cp.v[0]], v1 = verts[cp.v[1]];
    Vec3f v2 = verts[cp.v[2]], v3 = verts[cp.v[3]];

    float D;
    Vec3f g[4];

    if (cp.type == 0) {
        // PT contact
        D = abd_ipc::dist2_pt(v0, v1, v2, v3);
        if (D >= D_hat) { barrier_energy[ci] = 0; return; }
        abd_ipc::dist2_pt_grad(v0, v1, v2, v3, g[0], g[1], g[2], g[3]);
    } else {
        // EE contact
        D = abd_ipc::dist2_ee(v0, v1, v2, v3);
        if (D >= D_hat) { barrier_energy[ci] = 0; return; }
        abd_ipc::dist2_ee_grad(v0, v1, v2, v3, g[0], g[1], g[2], g[3]);
    }

    float dBdD  = abd_ipc::barrier_gradient(D, D_hat);
    float d2BdD = abd_ipc::barrier_hessian_spd(D, D_hat);

    barrier_energy[ci] = kappa * abd_ipc::barrier(D, D_hat);

    for (int k = 0; k < 4; ++k) {
        int vi = cp.v[k];
        int bid = cp.body[k];
        Vec6f q = body_q[bid];
        RigidJacobi J(x_bar[vi]);

        // Gradient: -kappa * dBdD * J^T * dD/dx
        Vec6f grad_contrib = J.mul_JT(g[k] * (kappa * dBdD), q);
        atomic_add_vec6(&grad_out[bid], grad_contrib * (-1.0f));

        // Hessian diagonal: kappa * d2BdD * (J^T dD/dx)(J^T dD/dx)^T
        Vec6f jt_g = J.mul_JT(g[k], q);
        Mat6f H_contrib = Mat6f::zero();
        float coeff = kappa * d2BdD;
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 6; ++c)
                H_contrib(r, c) = coeff * jt_g[r] * jt_g[c];
        atomic_add_mat6(&hess_diag_out[bid], H_contrib);
    }
}

void launch_assemble_contacts(
    const GPUContactPair* contacts, const Vec3f* verts,
    const Vec3f* x_bar, const int* vert_body,
    const Vec6f* body_q,
    Vec6f* grad_out, Mat6f* hess_diag_out,
    float* barrier_energy,
    float kappa, float D_hat,
    int n_contacts, cudaStream_t stream) {
    if (n_contacts == 0) return;
    int block = 128;
    int grid = (n_contacts + block - 1) / block;
    kernel_assemble_contacts<<<grid, block, 0, stream>>>(
        contacts, verts, x_bar, vert_body, body_q,
        grad_out, hess_diag_out, barrier_energy,
        kappa, D_hat, n_contacts);
}

// ============================================================================
// Kernel: CCD per contact pair  (parallel min reduction)
// ============================================================================

__global__ void kernel_ccd_contacts(
    const GPUContactPair* __restrict__ contacts,
    const Vec3f* __restrict__ verts_cur,
    const Vec3f* __restrict__ verts_next,
    float* __restrict__ alphas,   // [n_contacts]
    float d_min,
    int n_contacts)
{
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= n_contacts) return;

    GPUContactPair cp = contacts[ci];
    float alpha = 1.0f;

    Vec3f v0_c = verts_cur[cp.v[0]], v1_c = verts_cur[cp.v[1]];
    Vec3f v2_c = verts_cur[cp.v[2]], v3_c = verts_cur[cp.v[3]];
    Vec3f v0_n = verts_next[cp.v[0]], v1_n = verts_next[cp.v[1]];
    Vec3f v2_n = verts_next[cp.v[2]], v3_n = verts_next[cp.v[3]];

    Vec3f d0 = v0_n - v0_c, d1 = v1_n - v1_c;
    Vec3f d2 = v2_n - v2_c, d3 = v3_n - v3_c;

    if (cp.type == 0) {
        alpha = abd_ipc::ccd_pt(v0_c, d0, v1_c, d1, v2_c, d2, v3_c, d3, d_min, 20);
    } else {
        alpha = abd_ipc::ccd_ee(v0_c, d0, v1_c, d1, v2_c, d2, v3_c, d3, d_min, 20);
    }

    alphas[ci] = alpha;
}

void launch_ccd_contacts(
    const GPUContactPair* contacts,
    const Vec3f* verts_cur, const Vec3f* verts_next,
    float* alphas, float d_min,
    int n_contacts, cudaStream_t stream) {
    if (n_contacts == 0) return;
    int block = 128;
    int grid = (n_contacts + block - 1) / block;
    kernel_ccd_contacts<<<grid, block, 0, stream>>>(
        contacts, verts_cur, verts_next, alphas, d_min, n_contacts);
}

// ============================================================================
// Kernel: compute total barrier energy (per contact)
// ============================================================================

__global__ void kernel_barrier_energy(
    const GPUContactPair* __restrict__ contacts,
    const Vec3f* __restrict__ verts,
    float* __restrict__ energies,
    float kappa, float D_hat,
    int n_contacts)
{
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= n_contacts) return;

    GPUContactPair cp = contacts[ci];
    Vec3f v0 = verts[cp.v[0]], v1 = verts[cp.v[1]];
    Vec3f v2 = verts[cp.v[2]], v3 = verts[cp.v[3]];

    float D = (cp.type == 0)
        ? abd_ipc::dist2_pt(v0, v1, v2, v3)
        : abd_ipc::dist2_ee(v0, v1, v2, v3);

    energies[ci] = (D < D_hat) ? kappa * abd_ipc::barrier(D, D_hat) : 0.0f;
}

void launch_barrier_energy(
    const GPUContactPair* contacts, const Vec3f* verts,
    float* energies, float kappa, float D_hat,
    int n_contacts, cudaStream_t stream) {
    if (n_contacts == 0) return;
    int block = 256;
    int grid = (n_contacts + block - 1) / block;
    kernel_barrier_energy<<<grid, block, 0, stream>>>(
        contacts, verts, energies, kappa, D_hat, n_contacts);
}

// ============================================================================
// GPU reduce: sum of float array
// ============================================================================

float gpu_reduce_sum(const float* d_data, int n, cudaStream_t stream) {
    if (n == 0) return 0;
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    float* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(float));

    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_data, d_out, n, stream);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_data, d_out, n, stream);

    float result;
    cudaMemcpyAsync(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_temp);
    cudaFree(d_out);
    return result;
}

// ============================================================================
// GPU reduce: min of float array
// ============================================================================

float gpu_reduce_min(const float* d_data, int n, cudaStream_t stream) {
    if (n == 0) return 1.0f;
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    float* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(float));

    cub::DeviceReduce::Min(d_temp, temp_bytes, d_data, d_out, n, stream);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceReduce::Min(d_temp, temp_bytes, d_data, d_out, n, stream);

    float result;
    cudaMemcpyAsync(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_temp);
    cudaFree(d_out);
    return result;
}

// ============================================================================
// Kernel: zero Vec6f / Mat6f arrays
// ============================================================================

__global__ void kernel_zero_vec6(Vec6f* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = Vec6f::zero();
}

__global__ void kernel_zero_mat6(Mat6f* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = Mat6f::zero();
}

void launch_zero_vec6(Vec6f* data, int n, cudaStream_t stream) {
    if (n == 0) return;
    kernel_zero_vec6<<<(n+255)/256, 256, 0, stream>>>(data, n);
}

void launch_zero_mat6(Mat6f* data, int n, cudaStream_t stream) {
    if (n == 0) return;
    kernel_zero_mat6<<<(n+255)/256, 256, 0, stream>>>(data, n);
}

}  // namespace rigid_ipc
}  // namespace chysx
