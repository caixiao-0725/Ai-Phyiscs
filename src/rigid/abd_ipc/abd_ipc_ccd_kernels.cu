// SPDX-License-Identifier: MIT
// GPU CCD candidate generation for ABD-IPC (see header for rationale).

#include "abd_ipc_ccd_kernels.cuh"
#include "abd_ipc_distance.cuh"
#include "abd_ipc_ccd.cuh"

namespace chysx {
namespace abd_ipc {

using math::Vec3f;
using math::Vec2i;
using math::Vec3i;
using math::Vec4i;

// ---- device closest-distance helpers (margin culling only) ----------------

__device__ static float ccd_cand_point_tri_dist2(Vec3f p, Vec3f a, Vec3f b, Vec3f c) {
    Vec3f ab = b - a, ac = c - a, ap = p - a;
    float d1 = math::dot(ab, ap), d2 = math::dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) { Vec3f d = p - a; return math::dot(d, d); }
    Vec3f bp = p - b;
    float d3 = math::dot(ab, bp), d4 = math::dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) { Vec3f d = p - b; return math::dot(d, d); }
    float vc = d1*d4 - d3*d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        float v = d1 / (d1 - d3); Vec3f cp = a + ab * v; Vec3f d = p - cp; return math::dot(d, d);
    }
    Vec3f cpp = p - c;
    float d5 = math::dot(ab, cpp), d6 = math::dot(ac, cpp);
    if (d6 >= 0 && d5 <= d6) { Vec3f d = p - c; return math::dot(d, d); }
    float vb = d5*d2 - d1*d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        float w = d2 / (d2 - d6); Vec3f cp = a + ac * w; Vec3f d = p - cp; return math::dot(d, d);
    }
    float va = d3*d6 - d5*d4;
    if (va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0) {
        float w = (d4-d3) / ((d4-d3)+(d5-d6));
        Vec3f cp = b + (c - b) * w; Vec3f d = p - cp; return math::dot(d, d);
    }
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom, w = vc * denom;
    Vec3f cp = a + ab * v + ac * w; Vec3f d = p - cp; return math::dot(d, d);
}

__device__ static float ccd_cand_seg_seg_dist2(Vec3f p0, Vec3f p1, Vec3f q0, Vec3f q1) {
    Vec3f d1 = p1 - p0, d2 = q1 - q0, r = p0 - q0;
    float a = math::dot(d1, d1), e = math::dot(d2, d2), f = math::dot(d2, r);
    const float eps = 1e-12f;
    float s, t;
    if (a <= eps && e <= eps) { s = 0; t = 0; }
    else if (a <= eps) { s = 0; t = fminf(fmaxf(f / e, 0.0f), 1.0f); }
    else {
        float c = math::dot(d1, r);
        if (e <= eps) { t = 0; s = fminf(fmaxf(-c / a, 0.0f), 1.0f); }
        else {
            float b = math::dot(d1, d2);
            float denom = a*e - b*b;
            s = (denom != 0) ? fminf(fmaxf((b*f - c*e) / denom, 0.0f), 1.0f) : 0;
            t = (b*s + f) / e;
            if (t < 0) { t = 0; s = fminf(fmaxf(-c / a, 0.0f), 1.0f); }
            else if (t > 1) { t = 1; s = fminf(fmaxf((b - c) / a, 0.0f), 1.0f); }
        }
    }
    Vec3f cp = p0 + d1 * s;
    Vec3f cq = q0 + d2 * t;
    Vec3f diff = cp - cq;
    return math::dot(diff, diff);
}

__device__ static void emit_cand(
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

// ---- EF -> full PT + EE ----------------------------------------------------

__global__ void kernel_ef_to_ccd(
    const Vec2i* __restrict__ ef_pairs, int n_ef,
    const Vec3f* __restrict__ verts,
    const Vec3i* __restrict__ faces,
    const Vec2i* __restrict__ edges,
    const int* __restrict__ vert_in_edge,
    const Vec3i* __restrict__ edge_in_face,
    const int* __restrict__ vert_body,
    float margin_sq,
    GPUContactPair* out, int* d_count, int max_contacts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_ef) return;

    int eid = ef_pairs[i].x;
    int fid = ef_pairs[i].y;
    Vec3i f = faces[fid];

    // Full point-triangle: vert_in_edge[eid] vs the whole face.
    int vid = vert_in_edge[eid];
    if (vid >= 0 && vid != f.x && vid != f.y && vid != f.z) {
        int bv = vert_body[vid];
        int b0 = vert_body[f.x], b1 = vert_body[f.y], b2 = vert_body[f.z];
        if (!(bv == b0 && bv == b1 && bv == b2)) {
            float d2 = ccd_cand_point_tri_dist2(verts[vid], verts[f.x], verts[f.y], verts[f.z]);
            if (d2 < margin_sq)
                emit_cand(out, d_count, max_contacts, 2, vid, f.x, f.y, f.z, bv, b0, b1, b2);
        }
    }

    // Full edge-edge: edge[eid] vs each of the face's 3 edges (dedup oeid<eid).
    Vec3i e3 = edge_in_face[fid];
    Vec2i ea = edges[eid];
    for (int j = 0; j < 3; ++j) {
        int oeid = e3.data[j];
        if (oeid < 0 || oeid >= eid) continue;
        Vec2i eb = edges[oeid];
        if (ea.x == eb.x || ea.x == eb.y || ea.y == eb.x || ea.y == eb.y) continue;
        int ba0 = vert_body[ea.x], ba1 = vert_body[ea.y];
        int bb0 = vert_body[eb.x], bb1 = vert_body[eb.y];
        if (ba0 == bb0 && ba0 == bb1 && ba1 == bb0 && ba1 == bb1) continue;
        float d2 = ccd_cand_seg_seg_dist2(verts[ea.x], verts[ea.y], verts[eb.x], verts[eb.y]);
        if (d2 < margin_sq)
            emit_cand(out, d_count, max_contacts, 3, ea.x, ea.y, eb.x, eb.y, ba0, ba1, bb0, bb1);
    }
}

void launch_ef_to_ccd_candidates(
    const Vec2i* ef_pairs, int n_ef,
    const Vec3f* verts,
    const Vec3i* faces, const Vec2i* edges,
    const int* vert_in_edge, const Vec3i* edge_in_face,
    const int* vert_body,
    float margin_sq,
    GPUContactPair* out, int* d_count, int max_contacts,
    cudaStream_t stream) {
    if (n_ef == 0) return;
    int block = 256, grid = (n_ef + block - 1) / block;
    kernel_ef_to_ccd<<<grid, block, 0, stream>>>(
        ef_pairs, n_ef, verts, faces, edges, vert_in_edge, edge_in_face,
        vert_body, margin_sq, out, d_count, max_contacts);
}

// ---- adjacency supplements -------------------------------------------------

__global__ void kernel_adj_vf_ccd(
    const Vec4i* __restrict__ adj_vf, int n,
    const Vec3f* __restrict__ verts, const int* __restrict__ vert_body,
    float margin_sq,
    GPUContactPair* out, int* d_count, int max_contacts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Vec4i pr = adj_vf[i];
    int vid = pr.x, f0 = pr.y, f1 = pr.z, f2 = pr.w;
    int bv = vert_body[vid];
    int b0 = vert_body[f0], b1 = vert_body[f1], b2 = vert_body[f2];
    if (bv == b0 && bv == b1 && bv == b2) return;
    float d2 = ccd_cand_point_tri_dist2(verts[vid], verts[f0], verts[f1], verts[f2]);
    if (d2 < margin_sq)
        emit_cand(out, d_count, max_contacts, 2, vid, f0, f1, f2, bv, b0, b1, b2);
}

void launch_adj_vf_ccd(
    const Vec4i* adj_vf, int n,
    const Vec3f* verts, const int* vert_body, float margin_sq,
    GPUContactPair* out, int* d_count, int max_contacts,
    cudaStream_t stream) {
    if (n == 0) return;
    int block = 256, grid = (n + block - 1) / block;
    kernel_adj_vf_ccd<<<grid, block, 0, stream>>>(
        adj_vf, n, verts, vert_body, margin_sq, out, d_count, max_contacts);
}

__global__ void kernel_adj_ee_ccd(
    const Vec2i* __restrict__ adj_ee, int n,
    const Vec3f* __restrict__ verts,
    const Vec2i* __restrict__ edges, const int* __restrict__ vert_body,
    float margin_sq,
    GPUContactPair* out, int* d_count, int max_contacts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Vec2i pr = adj_ee[i];
    Vec2i ea = edges[pr.x], eb = edges[pr.y];
    if (ea.x == eb.x || ea.x == eb.y || ea.y == eb.x || ea.y == eb.y) return;
    int ba0 = vert_body[ea.x], ba1 = vert_body[ea.y];
    int bb0 = vert_body[eb.x], bb1 = vert_body[eb.y];
    if (ba0 == bb0 && ba0 == bb1 && ba1 == bb0 && ba1 == bb1) return;
    float d2 = ccd_cand_seg_seg_dist2(verts[ea.x], verts[ea.y], verts[eb.x], verts[eb.y]);
    if (d2 < margin_sq)
        emit_cand(out, d_count, max_contacts, 3, ea.x, ea.y, eb.x, eb.y, ba0, ba1, bb0, bb1);
}

void launch_adj_ee_ccd(
    const Vec2i* adj_ee, int n,
    const Vec3f* verts, const Vec2i* edges, const int* vert_body,
    float margin_sq,
    GPUContactPair* out, int* d_count, int max_contacts,
    cudaStream_t stream) {
    if (n == 0) return;
    int block = 256, grid = (n + block - 1) / block;
    kernel_adj_ee_ccd<<<grid, block, 0, stream>>>(
        adj_ee, n, verts, edges, vert_body, margin_sq, out, d_count, max_contacts);
}

// ---- Brute-force GPU CCD ---------------------------------------------------

__device__ static void atomic_min_float(float* addr, float val) {
    // Valid for non-negative floats: their bit patterns order monotonically.
    int* iaddr = reinterpret_cast<int*>(addr);
    int old = *iaddr, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= val) break;
        old = atomicCAS(iaddr, assumed, __float_as_int(val));
    } while (assumed != old);
}

__global__ void kernel_ccd_brute_pt(
    const Vec3f* __restrict__ vc, const Vec3f* __restrict__ vn,
    const Vec3i* __restrict__ tris, const int* __restrict__ vert_body,
    int n_verts, int n_tris, float d_min, float* __restrict__ d_alpha)
{
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n_verts * n_tris;
    if (id >= total) return;
    int vi = (int)(id / n_tris);
    int ti = (int)(id % n_tris);

    Vec3i tri = tris[ti];
    int bv = vert_body[vi];
    if (bv == vert_body[tri.x] && bv == vert_body[tri.y] && bv == vert_body[tri.z]) return;
    if (vi == tri.x || vi == tri.y || vi == tri.z) return;

    Vec3f p_c = vc[vi], t0_c = vc[tri.x], t1_c = vc[tri.y], t2_c = vc[tri.z];
    Vec3f dp  = vn[vi]    - p_c;
    Vec3f dt0 = vn[tri.x] - t0_c;
    Vec3f dt1 = vn[tri.y] - t1_c;
    Vec3f dt2 = vn[tri.z] - t2_c;

    float a = ccd_pt(p_c, dp, t0_c, dt0, t1_c, dt1, t2_c, dt2, d_min, 20);
    if (a < 1.0f) atomic_min_float(d_alpha, a);
}

__global__ void kernel_ccd_brute_ee(
    const Vec3f* __restrict__ vc, const Vec3f* __restrict__ vn,
    const Vec2i* __restrict__ edges, const int* __restrict__ vert_body,
    int n_edges, float d_min, float* __restrict__ d_alpha)
{
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n_edges * n_edges;
    if (id >= total) return;
    int i = (int)(id / n_edges);
    int j = (int)(id % n_edges);
    if (j <= i) return;

    Vec2i ea = edges[i], eb = edges[j];
    int ba0 = vert_body[ea.x], ba1 = vert_body[ea.y];
    int bb0 = vert_body[eb.x], bb1 = vert_body[eb.y];
    if (ba0 == bb0 && ba0 == bb1 && ba1 == bb0 && ba1 == bb1) return;
    if (ea.x == eb.x || ea.x == eb.y || ea.y == eb.x || ea.y == eb.y) return;

    Vec3f a0 = vc[ea.x], a1 = vc[ea.y], b0 = vc[eb.x], b1 = vc[eb.y];
    Vec3f da0 = vn[ea.x] - a0, da1 = vn[ea.y] - a1;
    Vec3f db0 = vn[eb.x] - b0, db1 = vn[eb.y] - b1;

    float a = ccd_ee(a0, da0, a1, da1, b0, db0, b1, db1, d_min, 20);
    if (a < 1.0f) atomic_min_float(d_alpha, a);
}

void launch_ccd_brute_pt(
    const Vec3f* verts_cur, const Vec3f* verts_next,
    const Vec3i* tris, const int* vert_body,
    int n_verts, int n_tris, float d_min,
    float* d_alpha, cudaStream_t stream) {
    long long total = (long long)n_verts * n_tris;
    if (total == 0) return;
    int block = 256;
    long long grid = (total + block - 1) / block;
    kernel_ccd_brute_pt<<<(unsigned)grid, block, 0, stream>>>(
        verts_cur, verts_next, tris, vert_body, n_verts, n_tris, d_min, d_alpha);
}

void launch_ccd_brute_ee(
    const Vec3f* verts_cur, const Vec3f* verts_next,
    const Vec2i* edges, const int* vert_body,
    int n_edges, float d_min,
    float* d_alpha, cudaStream_t stream) {
    long long total = (long long)n_edges * n_edges;
    if (total == 0) return;
    int block = 256;
    long long grid = (total + block - 1) / block;
    kernel_ccd_brute_ee<<<(unsigned)grid, block, 0, stream>>>(
        verts_cur, verts_next, edges, vert_body, n_edges, d_min, d_alpha);
}

}  // namespace abd_ipc
}  // namespace chysx
