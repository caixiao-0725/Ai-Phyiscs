// SPDX-License-Identifier: MIT
// GPU IPC contact pipeline for the ABD solver. See header.

#include "abd_ipc_gpu_kernels.cuh"
#include "abd_ipc_distance.cuh"
#include "abd_ipc_barrier.cuh"

namespace chysx {
namespace abd_ipc {

using math::Vec2i;
using math::Vec3i;
using math::Vec3f;
using math::Mat3f;

namespace {

constexpr int kBlock = 128;
inline int grid(int n) { return (n + kBlock - 1) / kBlock; }

// ---- atomic accumulation ---------------------------------------------------

__device__ void atomic_add_vec12(Vec12f* dst, const Vec12f& v) {
    for (int k = 0; k < 12; ++k) atomicAdd(&dst->data[k], v.data[k]);
}
__device__ void atomic_add_mat12(Mat12f* dst, const Mat12f& m) {
    for (int k = 0; k < 144; ++k) atomicAdd(&dst->data[k], m.data[k]);
}

// ---- finite-difference distance Hessian (12x12 over 4 verts) ---------------
// Central differences on the analytic distance gradient (matches the CPU IPC
// reference). type 2 = PT, 3 = EE.
__device__ void abd_dist_hess_fd(int type, const Vec3f V[4], float H[144]) {
    // eps scaled to the primitive size; too small loses float precision in the
    // gradient central difference, too large biases the curvature.
    float scale = fmaxf(fmaxf(fabsf(V[0].x), fabsf(V[0].y)), fabsf(V[0].z));
    scale = fmaxf(scale, 1e-3f);
    const float eps = 1e-4f * scale;
    float x[12];
    for (int k = 0; k < 4; ++k) { x[k*3+0]=V[k].x; x[k*3+1]=V[k].y; x[k*3+2]=V[k].z; }

    for (int col = 0; col < 12; ++col) {
        float xp[12], xm[12];
        for (int i = 0; i < 12; ++i) { xp[i]=x[i]; xm[i]=x[i]; }
        xp[col] += eps; xm[col] -= eps;

        Vec3f vp[4], vm[4];
        for (int k = 0; k < 4; ++k) {
            vp[k] = Vec3f(xp[k*3+0], xp[k*3+1], xp[k*3+2]);
            vm[k] = Vec3f(xm[k*3+0], xm[k*3+1], xm[k*3+2]);
        }
        Vec3f gp[4], gm[4];
        if (type == 2) {
            dist2_pt_grad(vp[0],vp[1],vp[2],vp[3], gp[0],gp[1],gp[2],gp[3]);
            dist2_pt_grad(vm[0],vm[1],vm[2],vm[3], gm[0],gm[1],gm[2],gm[3]);
        } else {
            dist2_ee_grad(vp[0],vp[1],vp[2],vp[3], gp[0],gp[1],gp[2],gp[3]);
            dist2_ee_grad(vm[0],vm[1],vm[2],vm[3], gm[0],gm[1],gm[2],gm[3]);
        }
        float inv2e = 1.0f / (2.0f * eps);
        for (int k = 0; k < 4; ++k)
            for (int j = 0; j < 3; ++j)
                H[(k*3+j)*12 + col] = (gp[k][j] - gm[k][j]) * inv2e;
    }
    // symmetrize
    for (int i = 0; i < 12; ++i)
        for (int j = i+1; j < 12; ++j) {
            float a = 0.5f * (H[i*12+j] + H[j*12+i]);
            H[i*12+j] = H[j*12+i] = a;
        }
}

// ---- PSD projection of a 12x12 symmetric matrix (cyclic Jacobi, float) ------
__device__ void psd_project12(float M[144]) {
    const int N = 12;
    float V[144];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) V[i*N+j] = (i==j) ? 1.0f : 0.0f;

    for (int sweep = 0; sweep < 12; ++sweep) {
        float off = 0;
        for (int i = 0; i < N; ++i)
            for (int j = i+1; j < N; ++j) off += M[i*N+j]*M[i*N+j];
        if (off < 1e-18f) break;
        for (int p = 0; p < N; ++p) {
            for (int q = p+1; q < N; ++q) {
                float apq = M[p*N+q];
                if (fabsf(apq) < 1e-14f) continue;
                float app = M[p*N+p], aqq = M[q*N+q];
                float tau = (aqq - app) / (2.0f * apq);
                float t = (tau >= 0 ? 1.0f : -1.0f) / (fabsf(tau) + sqrtf(1.0f + tau*tau));
                float c = 1.0f / sqrtf(1.0f + t*t);
                float s = t * c;
                M[p*N+p] = c*c*app - 2*s*c*apq + s*s*aqq;
                M[q*N+q] = s*s*app + 2*s*c*apq + c*c*aqq;
                M[p*N+q] = M[q*N+p] = 0;
                for (int r = 0; r < N; ++r) {
                    if (r==p || r==q) continue;
                    float mrp = M[r*N+p], mrq = M[r*N+q];
                    M[r*N+p] = M[p*N+r] = c*mrp - s*mrq;
                    M[r*N+q] = M[q*N+r] = s*mrp + c*mrq;
                }
                for (int r = 0; r < N; ++r) {
                    float vrp = V[r*N+p], vrq = V[r*N+q];
                    V[r*N+p] = c*vrp - s*vrq;
                    V[r*N+q] = s*vrp + c*vrq;
                }
            }
        }
    }
    float d[12];
    for (int i = 0; i < N; ++i) d[i] = M[i*N+i] > 0 ? M[i*N+i] : 0.0f;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float val = 0;
            for (int k = 0; k < N; ++k) val += V[i*N+k]*d[k]*V[j*N+k];
            M[i*N+j] = val;
        }
}

// ---- kernels ---------------------------------------------------------------

__global__ void k_update_verts(Vec3f* verts, const Vec3f* x_bar,
                               const int* vert_body, const Vec12f* body_q,
                               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ABDJacobi J(x_bar[i]);
    verts[i] = J.mul_q(body_q[vert_body[i]]);
}

__global__ void k_filter_pt(const Vec2i* pairs, int n,
                            const Vec3f* verts, const Vec3i* tris,
                            const int* vert_body, float D_hat,
                            GPUContactPair* out, int* count, int maxc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Vec2i pr = pairs[i];
    int vi = pr.x;
    Vec3i f = tris[pr.y];
    int bv = vert_body[vi], b0 = vert_body[f.x], b1 = vert_body[f.y], b2 = vert_body[f.z];
    if (bv == b0 && bv == b1 && bv == b2) return;
    if (vi == f.x || vi == f.y || vi == f.z) return;
    float D = dist2_pt(verts[vi], verts[f.x], verts[f.y], verts[f.z]);
    if (D >= D_hat) return;
    int idx = atomicAdd(count, 1);
    if (idx >= maxc) return;
    GPUContactPair cp;
    cp.type = 2;
    cp.v[0]=vi; cp.v[1]=f.x; cp.v[2]=f.y; cp.v[3]=f.z;
    cp.body[0]=bv; cp.body[1]=b0; cp.body[2]=b1; cp.body[3]=b2;
    out[idx] = cp;
}

__global__ void k_filter_ee(const Vec2i* pairs, int n,
                            const Vec3f* verts, const Vec2i* edges,
                            const int* vert_body, float D_hat,
                            GPUContactPair* out, int* count, int maxc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Vec2i pr = pairs[i];
    Vec2i ea = edges[pr.x], eb = edges[pr.y];
    int ba0=vert_body[ea.x], ba1=vert_body[ea.y], bb0=vert_body[eb.x], bb1=vert_body[eb.y];
    if (ba0==bb0 && ba0==bb1 && ba1==bb0 && ba1==bb1) return;
    if (ea.x==eb.x || ea.x==eb.y || ea.y==eb.x || ea.y==eb.y) return;
    float D = dist2_ee(verts[ea.x], verts[ea.y], verts[eb.x], verts[eb.y]);
    if (D >= D_hat) return;
    int idx = atomicAdd(count, 1);
    if (idx >= maxc) return;
    GPUContactPair cp;
    cp.type = 3;
    cp.v[0]=ea.x; cp.v[1]=ea.y; cp.v[2]=eb.x; cp.v[3]=eb.y;
    cp.body[0]=ba0; cp.body[1]=ba1; cp.body[2]=bb0; cp.body[3]=bb1;
    out[idx] = cp;
}

__global__ void k_assemble(const GPUContactPair* contacts, int nc,
                          const Vec3f* verts, const Vec3f* x_bar,
                          float kappa, float D_hat,
                          Vec12f* grad, Mat12f* hess_diag) {
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= nc) return;
    GPUContactPair cp = contacts[ci];

    Vec3f V[4] = { verts[cp.v[0]], verts[cp.v[1]], verts[cp.v[2]], verts[cp.v[3]] };
    float D;
    Vec3f g[4];
    if (cp.type == 2) {
        D = dist2_pt(V[0],V[1],V[2],V[3]);
        if (D >= D_hat) return;
        dist2_pt_grad(V[0],V[1],V[2],V[3], g[0],g[1],g[2],g[3]);
    } else {
        D = dist2_ee(V[0],V[1],V[2],V[3]);
        if (D >= D_hat) return;
        dist2_ee_grad(V[0],V[1],V[2],V[3], g[0],g[1],g[2],g[3]);
    }
    float dB  = kappa * barrier_gradient(D, D_hat);
    float d2B = kappa * barrier_hessian(D, D_hat);

    float distH[144];
    abd_dist_hess_fd(cp.type, V, distH);

    float Hv[144];
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b)
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    int R = a*3+r, C = b*3+c;
                    Hv[R*12+C] = d2B * g[a][r] * g[b][c] + dB * distH[R*12+C];
                }
    psd_project12(Hv);

    // gradient: J^T (dB * g_k) per vertex
    for (int k = 0; k < 4; ++k) {
        ABDJacobi Jk(x_bar[cp.v[k]]);
        Vec12f gq = Jk.mul_JT(g[k] * dB);
        atomic_add_vec12(&grad[cp.body[k]], gq);
    }

    // Hessian: per-body diagonal block sum_{a,b same body} J_a^T H_vert[a,b] J_b.
    // Off-diagonal (cross-body) blocks vanish against fixed bodies, so only
    // same-body pairs are scattered to the diagonal.
    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            if (cp.body[a] != cp.body[b]) continue;
            Mat3f H3;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    H3(r,c) = Hv[(a*3+r)*12 + (b*3+c)];
            ABDJacobi Ja(x_bar[cp.v[a]]), Jb(x_bar[cp.v[b]]);
            Mat12f H12 = ABDJacobi::JT_H_J(Ja, H3, Jb);
            atomic_add_mat12(&hess_diag[cp.body[a]], H12);
        }
    }
}

__global__ void k_zero_vec12(Vec12f* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = Vec12f::zero();
}
__global__ void k_zero_mat12(Mat12f* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = Mat12f::zero();
}

}  // namespace

void launch_abd_update_verts(Vec3f* verts, const Vec3f* x_bar,
                             const int* vert_body, const Vec12f* body_q,
                             int n_verts, cudaStream_t stream) {
    if (n_verts == 0) return;
    k_update_verts<<<grid(n_verts), kBlock, 0, stream>>>(
        verts, x_bar, vert_body, body_q, n_verts);
}

void launch_abd_filter_pt(const Vec2i* pt_pairs, int n_pairs,
                          const Vec3f* verts, const Vec3i* tris,
                          const int* vert_body, float D_hat,
                          GPUContactPair* out, int* count, int max_contacts,
                          cudaStream_t stream) {
    if (n_pairs == 0) return;
    k_filter_pt<<<grid(n_pairs), kBlock, 0, stream>>>(
        pt_pairs, n_pairs, verts, tris, vert_body, D_hat, out, count, max_contacts);
}

void launch_abd_filter_ee(const Vec2i* ee_pairs, int n_pairs,
                          const Vec3f* verts, const Vec2i* edges,
                          const int* vert_body, float D_hat,
                          GPUContactPair* out, int* count, int max_contacts,
                          cudaStream_t stream) {
    if (n_pairs == 0) return;
    k_filter_ee<<<grid(n_pairs), kBlock, 0, stream>>>(
        ee_pairs, n_pairs, verts, edges, vert_body, D_hat, out, count, max_contacts);
}

void launch_abd_assemble(const GPUContactPair* contacts, int n_contacts,
                         const Vec3f* verts, const Vec3f* x_bar,
                         const int* vert_body,
                         float kappa, float D_hat,
                         Vec12f* grad_out, Mat12f* hess_diag_out,
                         int n_bodies, cudaStream_t stream) {
    if (n_contacts == 0) return;
    k_assemble<<<grid(n_contacts), kBlock, 0, stream>>>(
        contacts, n_contacts, verts, x_bar, kappa, D_hat, grad_out, hess_diag_out);
}

void launch_abd_zero_vec12(Vec12f* data, int n, cudaStream_t stream) {
    if (n == 0) return;
    k_zero_vec12<<<grid(n), kBlock, 0, stream>>>(data, n);
}
void launch_abd_zero_mat12(Mat12f* data, int n, cudaStream_t stream) {
    if (n == 0) return;
    k_zero_mat12<<<grid(n), kBlock, 0, stream>>>(data, n);
}

}  // namespace abd_ipc
}  // namespace chysx
