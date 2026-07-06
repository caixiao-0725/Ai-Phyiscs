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
    // Classify as PP/PE/PT based on barycentric closest point
    int vid = vert_in_edge[eid];
    if (vid >= 0) {
        math::Vec3i f = faces[fid];
        if (vid != f.x && vid != f.y && vid != f.z) {
            int bv = vert_body[vid];
            int bf0 = vert_body[f.x], bf1 = vert_body[f.y], bf2 = vert_body[f.z];
            if (!(bv == bf0 && bv == bf1 && bv == bf2)) {
                Vec3f p = verts[vid], a = verts[f.x], b = verts[f.y], c = verts[f.z];
                Vec3f ab = b - a, ac = c - a, ap = p - a;
                float d1 = math::dot(ab, ap), d2 = math::dot(ac, ap);
                Vec3f bp = p - b;
                float d3 = math::dot(ab, bp), d4 = math::dot(ac, bp);
                Vec3f cp_ = p - c;
                float d5 = math::dot(ab, cp_), d6 = math::dot(ac, cp_);
                float vc = d1*d4 - d3*d2;
                float vb = d5*d2 - d1*d6;
                float va = d3*d6 - d5*d4;

                // Classify region
                int ctype; int cv[4]; int cb[4];
                if (d1 <= 0 && d2 <= 0) {
                    // PP: closest to vertex a = f.x
                    ctype = 0;
                    cv[0] = vid; cv[1] = f.x; cv[2] = -1; cv[3] = -1;
                    cb[0] = bv;  cb[1] = bf0;  cb[2] = -1; cb[3] = -1;
                } else if (d3 >= 0 && d4 <= d3) {
                    // PP: closest to vertex b = f.y
                    ctype = 0;
                    cv[0] = vid; cv[1] = f.y; cv[2] = -1; cv[3] = -1;
                    cb[0] = bv;  cb[1] = bf1;  cb[2] = -1; cb[3] = -1;
                } else if (d6 >= 0 && d5 <= d6) {
                    // PP: closest to vertex c = f.z
                    ctype = 0;
                    cv[0] = vid; cv[1] = f.z; cv[2] = -1; cv[3] = -1;
                    cb[0] = bv;  cb[1] = bf2;  cb[2] = -1; cb[3] = -1;
                } else if (vc <= 0 && d1 >= 0 && d3 <= 0) {
                    // PE: closest to edge ab = (f.x, f.y)
                    ctype = 1;
                    cv[0] = vid; cv[1] = f.x; cv[2] = f.y; cv[3] = -1;
                    cb[0] = bv;  cb[1] = bf0;  cb[2] = bf1;  cb[3] = -1;
                } else if (vb <= 0 && d2 >= 0 && d6 <= 0) {
                    // PE: closest to edge ac = (f.x, f.z)
                    ctype = 1;
                    cv[0] = vid; cv[1] = f.x; cv[2] = f.z; cv[3] = -1;
                    cb[0] = bv;  cb[1] = bf0;  cb[2] = bf2;  cb[3] = -1;
                } else if (va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0) {
                    // PE: closest to edge bc = (f.y, f.z)
                    ctype = 1;
                    cv[0] = vid; cv[1] = f.y; cv[2] = f.z; cv[3] = -1;
                    cb[0] = bv;  cb[1] = bf1;  cb[2] = bf2;  cb[3] = -1;
                } else {
                    // PT: interior of triangle
                    ctype = 2;
                    cv[0] = vid; cv[1] = f.x; cv[2] = f.y; cv[3] = f.z;
                    cb[0] = bv;  cb[1] = bf0;  cb[2] = bf1;  cb[3] = bf2;
                }

                float d2val = gpu_closest_point_triangle_dist2(p, a, b, c);
                if (d2val > 1e-24f && d2val < d_hat_sq) {
                    emit_contact(contacts_out, d_count, max_contacts,
                                 ctype, cv[0], cv[1], cv[2], cv[3],
                                 cb[0], cb[1], cb[2], cb[3]);
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
        if (d2 > 1e-24f && d2 < d_hat_sq) {
            // Classify EE contact type based on closest-point parameters
            Vec3f d1v = verts[ea.y] - verts[ea.x];
            Vec3f d2v = verts[eb.y] - verts[eb.x];
            Vec3f cross_d = math::cross(d1v, d2v);
            float cross_sq = math::dot(cross_d, cross_d);
            float len_sq1 = math::dot(d1v, d1v), len_sq2 = math::dot(d2v, d2v);
            const float eps_s = 1e-6f;

            if (s < eps_s && t < eps_s) {
                // PP: ea.x vs eb.x
                emit_contact(contacts_out, d_count, max_contacts,
                             0, ea.x, eb.x, -1, -1, ba0, bb0, -1, -1);
            } else if (s < eps_s && t > 1.0f - eps_s) {
                emit_contact(contacts_out, d_count, max_contacts,
                             0, ea.x, eb.y, -1, -1, ba0, bb1, -1, -1);
            } else if (s > 1.0f - eps_s && t < eps_s) {
                emit_contact(contacts_out, d_count, max_contacts,
                             0, ea.y, eb.x, -1, -1, ba1, bb0, -1, -1);
            } else if (s > 1.0f - eps_s && t > 1.0f - eps_s) {
                emit_contact(contacts_out, d_count, max_contacts,
                             0, ea.y, eb.y, -1, -1, ba1, bb1, -1, -1);
            } else if (s < eps_s) {
                // PE: ea.x on edge eb
                emit_contact(contacts_out, d_count, max_contacts,
                             1, ea.x, eb.x, eb.y, -1, ba0, bb0, bb1, -1);
            } else if (s > 1.0f - eps_s) {
                emit_contact(contacts_out, d_count, max_contacts,
                             1, ea.y, eb.x, eb.y, -1, ba1, bb0, bb1, -1);
            } else if (t < eps_s) {
                emit_contact(contacts_out, d_count, max_contacts,
                             1, eb.x, ea.x, ea.y, -1, bb0, ba0, ba1, -1);
            } else if (t > 1.0f - eps_s) {
                emit_contact(contacts_out, d_count, max_contacts,
                             1, eb.y, ea.x, ea.y, -1, bb1, ba0, ba1, -1);
            } else if (cross_sq >= 1e-3f * len_sq1 * len_sq2) {
                // True EE: interior of both edges, non-degenerate
                emit_contact(contacts_out, d_count, max_contacts,
                             3, ea.x, ea.y, eb.x, eb.y,
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

    math::Vec4i pr = adj_vf[i];
    int vid = pr.x, f0 = pr.y, f1 = pr.z, f2 = pr.w;

    Vec3f pv = verts[vid], a = verts[f0], b = verts[f1], c = verts[f2];
    float d2val = gpu_closest_point_triangle_dist2(pv, a, b, c);

    if (d2val > 1e-24f && d2val < d_hat_sq) {
        int bv = vert_body[vid];
        int b0 = vert_body[f0], b1 = vert_body[f1], b2 = vert_body[f2];
        if (!(bv == b0 && bv == b1 && bv == b2)) {
            Vec3f ab = b - a, ac = c - a, ap = pv - a;
            float d1_ = math::dot(ab, ap), d2_ = math::dot(ac, ap);
            Vec3f bp = pv - b;
            float d3_ = math::dot(ab, bp), d4_ = math::dot(ac, bp);
            Vec3f cp_ = pv - c;
            float d5_ = math::dot(ab, cp_), d6_ = math::dot(ac, cp_);
            float vc = d1_*d4_ - d3_*d2_;
            float vb = d5_*d2_ - d1_*d6_;
            float va = d3_*d6_ - d5_*d4_;

            if (d1_ <= 0 && d2_ <= 0) {
                emit_contact(contacts_out, d_count, max_contacts,
                             0, vid, f0, -1, -1, bv, b0, -1, -1);
            } else if (d3_ >= 0 && d4_ <= d3_) {
                emit_contact(contacts_out, d_count, max_contacts,
                             0, vid, f1, -1, -1, bv, b1, -1, -1);
            } else if (d6_ >= 0 && d5_ <= d6_) {
                emit_contact(contacts_out, d_count, max_contacts,
                             0, vid, f2, -1, -1, bv, b2, -1, -1);
            } else if (vc <= 0 && d1_ >= 0 && d3_ <= 0) {
                emit_contact(contacts_out, d_count, max_contacts,
                             1, vid, f0, f1, -1, bv, b0, b1, -1);
            } else if (vb <= 0 && d2_ >= 0 && d6_ <= 0) {
                emit_contact(contacts_out, d_count, max_contacts,
                             1, vid, f0, f2, -1, bv, b0, b2, -1);
            } else if (va <= 0 && (d4_-d3_) >= 0 && (d5_-d6_) >= 0) {
                emit_contact(contacts_out, d_count, max_contacts,
                             1, vid, f1, f2, -1, bv, b1, b2, -1);
            } else {
                emit_contact(contacts_out, d_count, max_contacts,
                             2, vid, f0, f1, f2, bv, b0, b1, b2);
            }
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

    if (d2 > 1e-24f && d2 < d_hat_sq) {
        int ba0 = vert_body[ea.x], ba1 = vert_body[ea.y];
        int bb0 = vert_body[eb.x], bb1 = vert_body[eb.y];
        if (ba0 == bb0 && ba0 == bb1 && ba1 == bb0 && ba1 == bb1)
            return;

        const float eps_s = 1e-6f;
        if (s < eps_s && t < eps_s) {
            emit_contact(contacts_out, d_count, max_contacts,
                         0, ea.x, eb.x, -1, -1, ba0, bb0, -1, -1);
        } else if (s < eps_s && t > 1.0f - eps_s) {
            emit_contact(contacts_out, d_count, max_contacts,
                         0, ea.x, eb.y, -1, -1, ba0, bb1, -1, -1);
        } else if (s > 1.0f - eps_s && t < eps_s) {
            emit_contact(contacts_out, d_count, max_contacts,
                         0, ea.y, eb.x, -1, -1, ba1, bb0, -1, -1);
        } else if (s > 1.0f - eps_s && t > 1.0f - eps_s) {
            emit_contact(contacts_out, d_count, max_contacts,
                         0, ea.y, eb.y, -1, -1, ba1, bb1, -1, -1);
        } else if (s < eps_s) {
            emit_contact(contacts_out, d_count, max_contacts,
                         1, ea.x, eb.x, eb.y, -1, ba0, bb0, bb1, -1);
        } else if (s > 1.0f - eps_s) {
            emit_contact(contacts_out, d_count, max_contacts,
                         1, ea.y, eb.x, eb.y, -1, ba1, bb0, bb1, -1);
        } else if (t < eps_s) {
            emit_contact(contacts_out, d_count, max_contacts,
                         1, eb.x, ea.x, ea.y, -1, bb0, ba0, ba1, -1);
        } else if (t > 1.0f - eps_s) {
            emit_contact(contacts_out, d_count, max_contacts,
                         1, eb.y, ea.x, ea.y, -1, bb1, ba0, ba1, -1);
        } else {
            Vec3f d1v = verts[ea.y] - verts[ea.x];
            Vec3f d2v = verts[eb.y] - verts[eb.x];
            Vec3f cross_d = math::cross(d1v, d2v);
            float cross_sq = math::dot(cross_d, cross_d);
            float len_sq1 = math::dot(d1v, d1v), len_sq2 = math::dot(d2v, d2v);
            if (cross_sq >= 1e-3f * len_sq1 * len_sq2) {
                emit_contact(contacts_out, d_count, max_contacts,
                             3, ea.x, ea.y, eb.x, eb.y,
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
    Vec6f* __restrict__ grad_out,            // [n_bodies]
    Mat6f* __restrict__ hess_diag_out,       // [n_bodies]
    Mat6f* __restrict__ hess_offdiag_out,    // [n_contacts * 2]
    math::Vec2i* __restrict__ offdiag_body_pairs, // [n_contacts]
    float* __restrict__ barrier_energy,      // [n_contacts]
    float kappa, float D_hat,
    int n_contacts)
{
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= n_contacts) return;

    GPUContactPair cp = contacts[ci];

    float D;
    Vec3f g[4] = {};
    int n_verts_in_cp;

    if (cp.type == 0) {
        n_verts_in_cp = 2;
        Vec3f v0 = verts[cp.v[0]], v1 = verts[cp.v[1]];
        D = abd_ipc::dist2_pp(v0, v1);
        if (D >= D_hat) {
            barrier_energy[ci] = 0;
            hess_offdiag_out[2*ci] = Mat6f::zero();
            hess_offdiag_out[2*ci+1] = Mat6f::zero();
            offdiag_body_pairs[ci] = math::Vec2i(-1, -1);
            return;
        }
        abd_ipc::dist2_pp_grad(v0, v1, g[0], g[1]);
    } else if (cp.type == 1) {
        n_verts_in_cp = 3;
        Vec3f v0 = verts[cp.v[0]], v1 = verts[cp.v[1]], v2 = verts[cp.v[2]];
        D = abd_ipc::dist2_pe(v0, v1, v2);
        if (D >= D_hat) {
            barrier_energy[ci] = 0;
            hess_offdiag_out[2*ci] = Mat6f::zero();
            hess_offdiag_out[2*ci+1] = Mat6f::zero();
            offdiag_body_pairs[ci] = math::Vec2i(-1, -1);
            return;
        }
        abd_ipc::dist2_pe_grad(v0, v1, v2, g[0], g[1], g[2]);
    } else if (cp.type == 2) {
        n_verts_in_cp = 4;
        Vec3f v0 = verts[cp.v[0]], v1 = verts[cp.v[1]];
        Vec3f v2 = verts[cp.v[2]], v3 = verts[cp.v[3]];
        D = abd_ipc::dist2_pt(v0, v1, v2, v3);
        if (D >= D_hat) {
            barrier_energy[ci] = 0;
            hess_offdiag_out[2*ci] = Mat6f::zero();
            hess_offdiag_out[2*ci+1] = Mat6f::zero();
            offdiag_body_pairs[ci] = math::Vec2i(-1, -1);
            return;
        }
        abd_ipc::dist2_pt_grad(v0, v1, v2, v3, g[0], g[1], g[2], g[3]);
    } else {
        n_verts_in_cp = 4;
        Vec3f v0 = verts[cp.v[0]], v1 = verts[cp.v[1]];
        Vec3f v2 = verts[cp.v[2]], v3 = verts[cp.v[3]];
        D = abd_ipc::dist2_ee(v0, v1, v2, v3);
        if (D >= D_hat) {
            barrier_energy[ci] = 0;
            hess_offdiag_out[2*ci] = Mat6f::zero();
            hess_offdiag_out[2*ci+1] = Mat6f::zero();
            offdiag_body_pairs[ci] = math::Vec2i(-1, -1);
            return;
        }
        abd_ipc::dist2_ee_grad(v0, v1, v2, v3, g[0], g[1], g[2], g[3]);
    }

    float dBdD  = abd_ipc::barrier_gradient(D, D_hat);
    float d2BdD = abd_ipc::barrier_hessian_spd(D, D_hat);
    float coeff = kappa * d2BdD;

    barrier_energy[ci] = kappa * abd_ipc::barrier(D, D_hat);

    // Compute J^T * grad_d for each participating vertex
    Vec6f jt_g[4];
    int bids[4];
    for (int k = 0; k < n_verts_in_cp; ++k) {
        int vi = cp.v[k];
        bids[k] = cp.body[k];
        Vec6f q = body_q[bids[k]];
        RigidJacobi J(x_bar[vi]);

        Vec6f grad_contrib = J.mul_JT(g[k] * (kappa * dBdD), q);
        atomic_add_vec6(&grad_out[bids[k]], grad_contrib * (-1.0f));

        jt_g[k] = J.mul_JT(g[k], q);
    }

    // Diagonal Hessian: all (k,k) pairs contribute to H_diag[body_k]
    for (int k = 0; k < n_verts_in_cp; ++k) {
        Mat6f H_kk = Mat6f::zero();
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 6; ++c)
                H_kk(r, c) = coeff * jt_g[k][r] * jt_g[k][c];
        atomic_add_mat6(&hess_diag_out[bids[k]], H_kk);
    }

    // Determine the two distinct body IDs in this contact
    int bodyA = bids[0], bodyB = -1;
    for (int k = 1; k < n_verts_in_cp; ++k) {
        if (bids[k] != bodyA) { bodyB = bids[k]; break; }
    }

    if (bodyA < 0 || bodyB < 0 || bodyA == bodyB) {
        // Single-body contact (all verts on same body) — no off-diagonal
        hess_offdiag_out[2*ci] = Mat6f::zero();
        hess_offdiag_out[2*ci+1] = Mat6f::zero();
        offdiag_body_pairs[ci] = math::Vec2i(-1, -1);

        // But we still need same-body cross-vert diagonal terms
        for (int a = 0; a < n_verts_in_cp; ++a) {
            for (int b = a+1; b < n_verts_in_cp; ++b) {
                Mat6f H_ab = Mat6f::zero();
                for (int r = 0; r < 6; ++r)
                    for (int c = 0; c < 6; ++c)
                        H_ab(r, c) = coeff * jt_g[a][r] * jt_g[b][c];
                // Symmetric: add both H_ab and H_ba to the diagonal block
                Mat6f H_sym = Mat6f::zero();
                for (int r = 0; r < 6; ++r)
                    for (int c = 0; c < 6; ++c)
                        H_sym(r, c) = H_ab(r, c) + H_ab(c, r);
                atomic_add_mat6(&hess_diag_out[bids[a]], H_sym);
            }
        }
        return;
    }

    offdiag_body_pairs[ci] = math::Vec2i(bodyA, bodyB);

    // Accumulate same-body cross-vert terms into diagonal blocks
    for (int a = 0; a < n_verts_in_cp; ++a) {
        for (int b = a+1; b < n_verts_in_cp; ++b) {
            if (bids[a] != bids[b]) continue;
            Mat6f H_ab = Mat6f::zero();
            for (int r = 0; r < 6; ++r)
                for (int c = 0; c < 6; ++c)
                    H_ab(r, c) = coeff * jt_g[a][r] * jt_g[b][c];
            Mat6f H_sym = Mat6f::zero();
            for (int r = 0; r < 6; ++r)
                for (int c = 0; c < 6; ++c)
                    H_sym(r, c) = H_ab(r, c) + H_ab(c, r);
            atomic_add_mat6(&hess_diag_out[bids[a]], H_sym);
        }
    }

    // Off-diagonal: H[A,B] and H[B,A] from cross-body vertex pairs
    Mat6f H_AB = Mat6f::zero();
    for (int a = 0; a < n_verts_in_cp; ++a) {
        for (int b = 0; b < n_verts_in_cp; ++b) {
            if (bids[a] == bodyA && bids[b] == bodyB) {
                for (int r = 0; r < 6; ++r)
                    for (int c = 0; c < 6; ++c)
                        H_AB(r, c) += coeff * jt_g[a][r] * jt_g[b][c];
            }
        }
    }

    hess_offdiag_out[2*ci]   = H_AB;
    // H[B,A] = H[A,B]^T
    Mat6f H_BA = Mat6f::zero();
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
            H_BA(r, c) = H_AB(c, r);
    hess_offdiag_out[2*ci+1] = H_BA;
}

void launch_assemble_contacts(
    const GPUContactPair* contacts, const Vec3f* verts,
    const Vec3f* x_bar, const int* vert_body,
    const Vec6f* body_q,
    Vec6f* grad_out, Mat6f* hess_diag_out,
    Mat6f* hess_offdiag_out, math::Vec2i* offdiag_body_pairs,
    float* barrier_energy,
    float kappa, float D_hat,
    int n_contacts, cudaStream_t stream) {
    if (n_contacts == 0) return;
    int block = 128;
    int grid = (n_contacts + block - 1) / block;
    kernel_assemble_contacts<<<grid, block, 0, stream>>>(
        contacts, verts, x_bar, vert_body, body_q,
        grad_out, hess_diag_out,
        hess_offdiag_out, offdiag_body_pairs,
        barrier_energy,
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

    // CCD: find earliest t in [0,1] where distance² drops below d_min².
    // When d_min==0 we use a small epsilon to detect near-zero crossings.
    float d_thresh = (d_min > 1e-10f) ? (d_min * d_min) : 1e-20f;

    if (cp.type == 0) {
        Vec3f p0_c = verts_cur[cp.v[0]], p1_c = verts_cur[cp.v[1]];
        Vec3f p0_n = verts_next[cp.v[0]], p1_n = verts_next[cp.v[1]];
        float d_cur = abd_ipc::dist2_pp(p0_c, p1_c);
        float d_next = abd_ipc::dist2_pp(p0_n, p1_n);
        if (d_next < d_thresh && d_next < d_cur) {
            float lo = 0, hi = 1.0f;
            for (int it = 0; it < 20; ++it) {
                float mid = (lo + hi) * 0.5f;
                Vec3f a = p0_c + (p0_n - p0_c) * mid;
                Vec3f b = p1_c + (p1_n - p1_c) * mid;
                float dsq = abd_ipc::dist2_pp(a, b);
                if (dsq < d_thresh) hi = mid; else lo = mid;
            }
            alpha = fmaxf(lo, 1e-3f);
        }
    } else if (cp.type == 1) {
        Vec3f p0_c = verts_cur[cp.v[0]], e0_c = verts_cur[cp.v[1]], e1_c = verts_cur[cp.v[2]];
        Vec3f p0_n = verts_next[cp.v[0]], e0_n = verts_next[cp.v[1]], e1_n = verts_next[cp.v[2]];
        float d_cur = abd_ipc::dist2_pe(p0_c, e0_c, e1_c);
        float d_next = abd_ipc::dist2_pe(p0_n, e0_n, e1_n);
        if (d_next < d_thresh && d_next < d_cur) {
            float lo = 0, hi = 1.0f;
            for (int it = 0; it < 20; ++it) {
                float mid = (lo + hi) * 0.5f;
                Vec3f p = p0_c + (p0_n - p0_c) * mid;
                Vec3f a = e0_c + (e0_n - e0_c) * mid;
                Vec3f b = e1_c + (e1_n - e1_c) * mid;
                float dsq = abd_ipc::dist2_pe(p, a, b);
                if (dsq < d_thresh) hi = mid; else lo = mid;
            }
            alpha = fmaxf(lo, 1e-3f);
        }
    } else if (cp.type == 2) {
        Vec3f v0_c = verts_cur[cp.v[0]], v1_c = verts_cur[cp.v[1]];
        Vec3f v2_c = verts_cur[cp.v[2]], v3_c = verts_cur[cp.v[3]];
        Vec3f v0_n = verts_next[cp.v[0]], v1_n = verts_next[cp.v[1]];
        Vec3f v2_n = verts_next[cp.v[2]], v3_n = verts_next[cp.v[3]];
        float d_cur = abd_ipc::dist2_pt(v0_c, v1_c, v2_c, v3_c);
        float d_next = abd_ipc::dist2_pt(v0_n, v1_n, v2_n, v3_n);
        if (d_next < d_thresh && d_next < d_cur) {
            Vec3f d0 = v0_n - v0_c, d1 = v1_n - v1_c;
            Vec3f d2 = v2_n - v2_c, d3 = v3_n - v3_c;
            float ccd_d = (d_min > 1e-10f) ? d_min : 1e-10f;
            alpha = abd_ipc::ccd_pt(v0_c, d0, v1_c, d1, v2_c, d2, v3_c, d3, ccd_d, 20);
            alpha = fmaxf(alpha, 1e-3f);
        }
    } else {
        Vec3f v0_c = verts_cur[cp.v[0]], v1_c = verts_cur[cp.v[1]];
        Vec3f v2_c = verts_cur[cp.v[2]], v3_c = verts_cur[cp.v[3]];
        Vec3f v0_n = verts_next[cp.v[0]], v1_n = verts_next[cp.v[1]];
        Vec3f v2_n = verts_next[cp.v[2]], v3_n = verts_next[cp.v[3]];
        float d_cur = abd_ipc::dist2_ee(v0_c, v1_c, v2_c, v3_c);
        float d_next = abd_ipc::dist2_ee(v0_n, v1_n, v2_n, v3_n);
        if (d_next < d_thresh && d_next < d_cur) {
            Vec3f d0 = v0_n - v0_c, d1 = v1_n - v1_c;
            Vec3f d2 = v2_n - v2_c, d3 = v3_n - v3_c;
            float ccd_d = (d_min > 1e-10f) ? d_min : 1e-10f;
            alpha = abd_ipc::ccd_ee(v0_c, d0, v1_c, d1, v2_c, d2, v3_c, d3, ccd_d, 20);
            alpha = fmaxf(alpha, 1e-3f);
        }
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

    float D;
    if (cp.type == 0) {
        D = abd_ipc::dist2_pp(v0, v1);
    } else if (cp.type == 1) {
        Vec3f v2 = verts[cp.v[2]];
        D = abd_ipc::dist2_pe(v0, v1, v2);
    } else if (cp.type == 2) {
        Vec3f v2 = verts[cp.v[2]], v3 = verts[cp.v[3]];
        D = abd_ipc::dist2_pt(v0, v1, v2, v3);
    } else {
        Vec3f v2 = verts[cp.v[2]], v3 = verts[cp.v[3]];
        D = abd_ipc::dist2_ee(v0, v1, v2, v3);
    }

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
// Kernel: compute per-contact squared distance (for kappa update)
// ============================================================================

__global__ void kernel_compute_contact_distances(
    const GPUContactPair* __restrict__ contacts,
    const Vec3f* __restrict__ verts,
    float* __restrict__ dist_sq_out,
    float D_hat, int n_contacts)
{
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= n_contacts) return;

    GPUContactPair cp = contacts[ci];
    Vec3f v0 = verts[cp.v[0]], v1 = verts[cp.v[1]];
    float D;
    if (cp.type == 0) {
        D = abd_ipc::dist2_pp(v0, v1);
    } else if (cp.type == 1) {
        Vec3f v2 = verts[cp.v[2]];
        D = abd_ipc::dist2_pe(v0, v1, v2);
    } else if (cp.type == 2) {
        Vec3f v2 = verts[cp.v[2]], v3 = verts[cp.v[3]];
        D = abd_ipc::dist2_pt(v0, v1, v2, v3);
    } else {
        Vec3f v2 = verts[cp.v[2]], v3 = verts[cp.v[3]];
        D = abd_ipc::dist2_ee(v0, v1, v2, v3);
    }
    dist_sq_out[ci] = (D < D_hat) ? D : D_hat;
}

void launch_compute_contact_distances(
    const GPUContactPair* contacts, const Vec3f* verts,
    float* dist_sq_out, float D_hat,
    int n_contacts, cudaStream_t stream) {
    if (n_contacts == 0) return;
    int block = 256;
    int grid = (n_contacts + block - 1) / block;
    kernel_compute_contact_distances<<<grid, block, 0, stream>>>(
        contacts, verts, dist_sq_out, D_hat, n_contacts);
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

// ============================================================================
// Clamp off-diagonal Hessian blocks for locked DOFs
// ============================================================================

__global__ void kernel_clamp_offdiag_dofs(
    Mat6f* __restrict__ hess_offdiag,
    const math::Vec2i* __restrict__ body_pairs,
    const Vec6f* __restrict__ dof_mask,
    int n_pairs)
{
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= n_pairs) return;
    math::Vec2i bp = body_pairs[ci];
    if (bp.x < 0) return;

    Vec6f maskA = dof_mask[bp.x];
    Vec6f maskB = dof_mask[bp.y];

    // H[A,B]: rows=A's DOFs, cols=B's DOFs
    Mat6f& HAB = hess_offdiag[2*ci];
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
            HAB(r, c) *= maskA[r] * maskB[c];

    // H[B,A]: rows=B's DOFs, cols=A's DOFs
    Mat6f& HBA = hess_offdiag[2*ci+1];
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
            HBA(r, c) *= maskB[r] * maskA[c];
}

void launch_clamp_offdiag_dofs(
    Mat6f* hess_offdiag, const math::Vec2i* body_pairs,
    const Vec6f* dof_mask,
    int n_pairs, cudaStream_t stream) {
    if (n_pairs == 0) return;
    kernel_clamp_offdiag_dofs<<<(n_pairs+127)/128, 128, 0, stream>>>(
        hess_offdiag, body_pairs, dof_mask, n_pairs);
}

// ============================================================================
// GPU block-sparse matvec:  y[i] = H_diag[i]*x[i] + off-diag contributions
// ============================================================================

__global__ void kernel_bspmv_diag(
    const Mat6f* __restrict__ H_diag,
    const Vec6f* __restrict__ x,
    Vec6f* __restrict__ y,
    int n_bodies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_bodies) y[i] = H_diag[i] * x[i];
}

__global__ void kernel_bspmv_offdiag(
    const Mat6f* __restrict__ hess_offdiag,
    const math::Vec2i* __restrict__ body_pairs,
    const Vec6f* __restrict__ x,
    Vec6f* __restrict__ y,
    int n_pairs)
{
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= n_pairs) return;
    math::Vec2i bp = body_pairs[ci];
    if (bp.x < 0) return;
    int A = bp.x, B = bp.y;
    // H[A,B] * x[B] → y[A]
    Vec6f contrib_AB = hess_offdiag[2*ci] * x[B];
    atomic_add_vec6(&y[A], contrib_AB);
    // H[B,A] * x[A] → y[B]
    Vec6f contrib_BA = hess_offdiag[2*ci+1] * x[A];
    atomic_add_vec6(&y[B], contrib_BA);
}

void launch_block_sparse_matvec(
    const Mat6f* H_diag,
    const Mat6f* hess_offdiag, const math::Vec2i* body_pairs,
    int n_offdiag_pairs,
    const Vec6f* x, Vec6f* y,
    int n_bodies, cudaStream_t stream) {
    kernel_bspmv_diag<<<(n_bodies+127)/128, 128, 0, stream>>>(
        H_diag, x, y, n_bodies);
    if (n_offdiag_pairs > 0) {
        kernel_bspmv_offdiag<<<(n_offdiag_pairs+127)/128, 128, 0, stream>>>(
            hess_offdiag, body_pairs, x, y, n_offdiag_pairs);
    }
}

// ============================================================================
// GPU PCG solver for block-sparse 6×6 system
// ============================================================================

// Helper: GPU dot product of Vec6f arrays (sum of dot(a[i], b[i]))
static __global__ void kernel_dot6(
    const Vec6f* __restrict__ a,
    const Vec6f* __restrict__ b,
    float* __restrict__ partial, int n)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0;
    if (i < n) val = math::dot(a[i], b[i]);
    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

static float gpu_dot6(const Vec6f* a, const Vec6f* b, int n,
                      float* d_partial, float* h_partial, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_dot6<<<grid, block, 0, stream>>>(a, b, d_partial, n);
    cudaMemcpyAsync(h_partial, d_partial, grid * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    float sum = 0;
    for (int i = 0; i < grid; ++i) sum += h_partial[i];
    return sum;
}

// z[i] = P[i] * r[i]  where P[i] = inv(H_diag[i])
static __global__ void kernel_precond(
    const Mat6f* __restrict__ P_inv,
    const Vec6f* __restrict__ r,
    Vec6f* __restrict__ z, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = P_inv[i] * r[i];
}

// axpy-like: a[i] = a[i] + b[i] * alpha
static __global__ void kernel_axpy6(
    Vec6f* __restrict__ a, const Vec6f* __restrict__ b,
    float alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = a[i] + b[i] * alpha;
}

// copy
static __global__ void kernel_copy6(
    Vec6f* __restrict__ dst, const Vec6f* __restrict__ src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// p = add + p * scale  (for PCG direction update: p = z + beta * p)
static __global__ void kernel_scale_and_add6(
    Vec6f* __restrict__ p, const Vec6f* __restrict__ add,
    float scale, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = add[i] + p[i] * scale;
}

int launch_pcg6_gpu(
    const Mat6f* H_diag,
    const Mat6f* hess_offdiag, const math::Vec2i* body_pairs,
    int n_offdiag_pairs,
    const Vec6f* rhs, Vec6f* dx,
    Vec6f* r, Vec6f* z, Vec6f* p, Vec6f* Hp,
    int n_bodies, float tol, int max_iter,
    cudaStream_t stream)
{
    if (n_bodies == 0) return 0;
    int grid1 = (n_bodies + 127) / 128;
    int dot_grid = (n_bodies + 255) / 256;

    // Allocate scratch for dot-product partial sums and preconditioner
    float* d_partial = nullptr;
    cudaMalloc(&d_partial, dot_grid * sizeof(float));
    float* h_partial = new float[dot_grid];

    Mat6f* d_P_inv = nullptr;
    cudaMalloc(&d_P_inv, n_bodies * sizeof(Mat6f));

    // Compute block-diagonal preconditioner P = inv(H_diag) on GPU
    // We do this on CPU since n_bodies is small
    {
        std::vector<Mat6f> h_diag(n_bodies), h_pinv(n_bodies);
        cudaMemcpy(h_diag.data(), H_diag, n_bodies * sizeof(Mat6f),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < n_bodies; ++i)
            h_pinv[i] = math::inverse_spd<float, 6>(h_diag[i]);
        cudaMemcpy(d_P_inv, h_pinv.data(), n_bodies * sizeof(Mat6f),
                   cudaMemcpyHostToDevice);
    }

    // dx = 0
    launch_zero_vec6(dx, n_bodies, stream);
    // r = rhs
    kernel_copy6<<<grid1, 128, 0, stream>>>(r, rhs, n_bodies);
    // z = P * r
    kernel_precond<<<grid1, 128, 0, stream>>>(d_P_inv, r, z, n_bodies);
    // p = z
    kernel_copy6<<<grid1, 128, 0, stream>>>(p, z, n_bodies);

    float rz = gpu_dot6(r, z, n_bodies, d_partial, h_partial, stream);

    float rhs_norm = gpu_dot6(rhs, rhs, n_bodies, d_partial, h_partial, stream);
    rhs_norm = sqrtf(rhs_norm);
    if (rhs_norm < 1e-20f) {
        cudaFree(d_partial); cudaFree(d_P_inv); delete[] h_partial;
        return 0;
    }
    float abs_tol = tol * rhs_norm;

    int iters = 0;
    for (int iter = 0; iter < max_iter; ++iter) {
        // Hp = H * p
        launch_block_sparse_matvec(H_diag, hess_offdiag, body_pairs,
                                    n_offdiag_pairs, p, Hp, n_bodies, stream);

        float pHp = gpu_dot6(p, Hp, n_bodies, d_partial, h_partial, stream);
        if (pHp <= 0.0f) pHp = 1e-20f;
        float alpha = rz / pHp;

        // dx += alpha * p
        kernel_axpy6<<<grid1, 128, 0, stream>>>(dx, p, alpha, n_bodies);
        // r -= alpha * Hp
        kernel_axpy6<<<grid1, 128, 0, stream>>>(r, Hp, -alpha, n_bodies);

        float r_norm = gpu_dot6(r, r, n_bodies, d_partial, h_partial, stream);
        r_norm = sqrtf(r_norm);
        iters = iter + 1;
        if (r_norm < abs_tol) break;

        // z = P * r
        kernel_precond<<<grid1, 128, 0, stream>>>(d_P_inv, r, z, n_bodies);
        float rz_new = gpu_dot6(r, z, n_bodies, d_partial, h_partial, stream);
        float beta = rz_new / (rz > 1e-30f ? rz : 1e-30f);
        rz = rz_new;

        // p = z + beta * p
        // scale p by beta, then add z
        kernel_scale_and_add6<<<grid1, 128, 0, stream>>>(p, z, beta, n_bodies);
    }

    cudaFree(d_partial);
    cudaFree(d_P_inv);
    delete[] h_partial;
    return iters;
}

}  // namespace rigid_ipc
}  // namespace chysx
