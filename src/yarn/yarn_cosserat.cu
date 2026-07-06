// SPDX-License-Identifier: Apache-2.0
//
// Core Cosserat rod iteration kernels, ported from YarnBall cosserat.cu.
// cosseratItr: per-vertex local Newton solve (inertia + stretch + collision + friction)
// quaternionLambdaItr: orientation update via lambda iteration

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#include "yarn_types.cuh"

namespace chysx {
namespace yarn {

#define YARN_BLOCK_SIZE        (32 * 4)
#define YARN_THREADS_PER_VERTEX 2
#define YARN_VERTEX_PER_BLOCK  (YARN_BLOCK_SIZE / YARN_THREADS_PER_VERTEX)

template <bool LIMIT>
__global__ void kernel_cosserat_itr(YarnMetaData* data) {
    const int sid  = threadIdx.x / YARN_VERTEX_PER_BLOCK;
    const int ltid = threadIdx.x - sid * YARN_VERTEX_PER_BLOCK;
    const int tid  = static_cast<int>(blockIdx.x * (YARN_VERTEX_PER_BLOCK - 1) + ltid) - 1;

    const int numVerts = data->numVerts;
    if (tid >= numVerts || tid < 0) return;

    const float h       = data->h;
    const float damping = data->damping / h;
    const auto lastPos  = data->d_lastPos;
    const auto dxs      = data->d_dx;

    Vertex v0 = data->d_verts[tid];
    Vec3f  dx = dxs[tid];

    SymHess3 H(0.f);
    Vec3f f(0.f, 0.f, 0.f);

    if (!sid) {
        // Inertia: H = (1/(m*h²)) I, f = (1/(m*h²)) (y - dx)
        H = SymHess3(1.f / (v0.invMass * h * h));
        Vec3f y = data->d_vels[tid];  // overwritten with y in initItr
        f = (1.f / (h * h * v0.invMass)) * (y - dx);

        // Special connections (glueing endpoints)
        if (v0.connectionIndex >= 0) {
            constexpr float stiffness = 4e1f;
            Vec3f p0   = lastPos[v0.connectionIndex];
            Vec3f p0dx = dxs[v0.connectionIndex];
            f = f - stiffness * ((v0.pos - p0) + (dx - p0dx) + damping * dx);
            H.diag = H.diag + Vec3f(stiffness * (1.f + damping));
        }
    }

    Vec3f p1(0.f, 0.f, 0.f);
    Vec3f p1dx(0.f, 0.f, 0.f);
    Vec3f f2(0.f, 0.f, 0.f);
    SymHess3 H2(0.f);

    if (v0.flags & static_cast<uint32_t>(VertexFlags::hasNext)) {
        p1   = lastPos[tid + 1];
        p1dx = dxs[tid + 1];

        // Cosserat stretching energy
        if (!sid) {
            float invl = 1.f / v0.lRest;
            Vec3f edge = ((p1 - v0.pos) + (p1dx - dx)) * invl;
            Vec3f ex   = math::quat_rotate(data->d_qs[tid], Vec3f(1.f, 0.f, 0.f));
            Vec3f c    = edge - ex;

            float k = v0.kStretch * invl;
            float d = k * invl;
            f  = f + k * c - (damping * d) * dx;
            f2 = f2 + (-k) * c - (damping * d) * p1dx;
            d *= (1.f + damping);
            H.diag  = H.diag + Vec3f(d);
            H2.diag = H2.diag + Vec3f(d);
        }

        const float fricK  = data->kFriction;
        const float invb   = 1.f / data->barrierThickness;
        const float radius = 2.f * data->radius;
        const float fricMu = data->frictionCoeff;
        const auto  collisions = data->d_collisions;
        const float kCol   = data->kCollision * invb;

        const int numCols = data->d_numCols[tid];
        for (int i = sid; i < numCols; i += YARN_THREADS_PER_VERTEX) {
            int colID = collisions[tid + i * numVerts];

            Vec3f b0   = lastPos[colID];
            Vec3f b1   = lastPos[colID + 1];
            Vec3f db0  = dxs[colID];
            Vec3f db1  = dxs[colID + 1];

            Vec2f uv = segment_closest_points(
                Vec3f(0.f, 0.f, 0.f),
                (p1 - v0.pos) + (p1dx - dx),
                (b0 - v0.pos) + (db0 - dx),
                (b1 - v0.pos) + (db1 - dx));

            Vec3f dpos  = v0.pos * (1.f - uv.x) + p1 * uv.x
                        - (b0 * (1.f - uv.y) + b1 * uv.y);
            Vec3f ddpos = dx * (1.f - uv.x) + p1dx * uv.x
                        - (db0 * (1.f - uv.y) + db1 * uv.y);
            Vec3f normal = dpos + ddpos;
            float dist   = length(normal);
            normal = normal * (1.f / dist);

            // Swap to match reference: uv.y=uv.x (this-segment weight), uv.x=1-uv.x
            uv.y = uv.x;
            uv.x = 1.f - uv.x;

            // Penetration depth normalized by barrier thickness
            dist = dist - radius;
            dist *= invb;
            if (dist > 1.f) continue;
            dist = fmaxf(dist, 1e-3f);

            // IPC log-barrier
            float invd = 1.f / dist;
            float logd = logf(dist);

            float dH = (-3.f + (2.f + invd) * invd - 2.f * logd) * kCol * invb;
            float ff = -(1.f - dist) * (dist - 1.f + 2.f * dist * logd) * invd * kCol;

            f  = f + (ff * uv.x - damping * dH * uv.x * uv.x * dot(normal, dx)) * normal;
            f2 = f2 + (ff * uv.y - damping * dH * uv.y * uv.y * dot(normal, p1dx)) * normal;

            dH *= (1.f + damping);
            SymHess3 op = SymHess3::outer(normal);
            H  += op * (dH * uv.x * uv.x);
            H2 += op * (dH * uv.y * uv.y);

            // Friction
            Vec3f u  = ddpos - dot(normal, ddpos) * normal;
            float ul = length(u);
            if (ul > 0.f) {
                float f1 = fminf(fricK, fricMu * ff / ul);

                // (I - n⊗n) = -op + I  →  op.diag -= 1 matches reference
                SymHess3 friction_op = op;
                friction_op.diag = friction_op.diag - Vec3f(1.f);

                f  = f - f1 * uv.x * u;
                H  -= friction_op * (uv.x * uv.x * f1);

                f2 = f2 - f1 * uv.y * u;
                H2 -= friction_op * (uv.y * uv.y * f1);
            }
        }
    }

    // --- Shared memory reduce across sectors ---
    __shared__ float sharedData[18 * YARN_VERTEX_PER_BLOCK];

    Vec3f*    f0s = reinterpret_cast<Vec3f*>(sharedData);
    Vec3f*    f1s = reinterpret_cast<Vec3f*>(sharedData + 3 * YARN_VERTEX_PER_BLOCK);
    SymHess3* h0s = reinterpret_cast<SymHess3*>(sharedData + 6 * YARN_VERTEX_PER_BLOCK);
    SymHess3* h1s = reinterpret_cast<SymHess3*>(sharedData + 12 * YARN_VERTEX_PER_BLOCK);

    if (sid) {
        f0s[ltid] = f;
        f1s[ltid] = f2;
        h0s[ltid] = H;
        h1s[ltid] = H2;
    }
    __syncthreads();

    if (!sid) {
        f  = f + f0s[ltid];
        f2 = f2 + f1s[ltid];
        H  += h0s[ltid];
        H2 += h1s[ltid];
    }
    __syncthreads();

    // Sector 0 performs the actual position update
    if (!sid) {
        // Stash f2/H2 for neighbor accumulation
        Vec4f*    forces   = reinterpret_cast<Vec4f*>(sharedData);
        SymHess3* hessians = reinterpret_cast<SymHess3*>(sharedData + 4 * YARN_VERTEX_PER_BLOCK);

        float stepLimit = INFINITY;
        if (LIMIT) stepLimit = data->d_maxStepSize[tid];
        forces[threadIdx.x]   = Vec4f(f2.x, f2.y, f2.z, stepLimit);
        hessians[threadIdx.x] = H2;

        __syncthreads();

        // Thread 0 of each block is a ghost (tid=-1), skip update
        if (!threadIdx.x) return;

        if (v0.flags & static_cast<uint32_t>(VertexFlags::hasPrev)) {
            Vec4f v = forces[threadIdx.x - 1];
            if (LIMIT) stepLimit = fminf(stepLimit, v.w);
            f = f + Vec3f(v.x, v.y, v.z);
            H += hessians[threadIdx.x - 1];
        }

        if (v0.invMass != 0.f) {
            Mat3f Hfull = static_cast<Mat3f>(H);
            Vec3f delta = data->accelerationRatio * (inverse(Hfull) * f);
            dx = dx + delta;

            if (LIMIT) {
                float l = length(dx);
                if (l > stepLimit && l > 0.f) dx = dx * (stepLimit / l);
            }

            dxs[tid] = dx;
        }
    }
}

// --- Quaternion lambda iteration (orientation update) ---

__global__ void kernel_quaternion_lambda_itr(YarnMetaData* data) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int numVerts = data->numVerts;
    if (tid >= numVerts || tid < 0) return;

    const auto verts = data->d_verts;
    const auto dxs   = data->d_dx;

    Vertex v0 = verts[tid];

    // Exact match of YarnBall's condition (note the precedence: !(bool)(...) != 0 && ...)
    if (!(bool)(v0.flags & static_cast<uint32_t>(VertexFlags::fixOrientation)) != 0
        && (v0.flags & static_cast<uint32_t>(VertexFlags::hasNext)))
    {
        Vec3f dx   = dxs[tid];
        Vec3f p1   = verts[tid + 1].pos;
        Vec3f p1dx = dxs[tid + 1];

        // YarnBall reuses v0.pos as scratch for sdir
        Vec3f sdir = ((p1 - v0.pos) + (p1dx - dx)) / v0.lRest;
        sdir = sdir * (-2.f * v0.kStretch);

        Vec4f b(0.f, 0.f, 0.f, 0.f);
        auto qs     = data->d_qs;
        auto qRests = data->d_qRests;
        Quatf q0    = qs[tid];

        if (v0.flags & static_cast<uint32_t>(VertexFlags::hasPrev)) {
            Quatf qRest = qRests[tid - 1];
            Quatf qq    = qs[tid - 1];

            // qq.inverse() * q0 → conjugate(qq) * q0
            Quatf rel = math::quat_multiply(math::quat_conjugate(qq), q0);
            float sd  = dot(rel, qRest);
            float s   = (sd > 0.f) ? 1.f : -1.f;

            Quatf prod = math::quat_multiply(qq, qRest);
            b = b + s * prod;
        }

        if (v0.flags & static_cast<uint32_t>(VertexFlags::hasNextOrientation)) {
            Quatf qRest = qRests[tid];
            Quatf qq    = qs[tid + 1];

            Quatf rel = math::quat_multiply(math::quat_conjugate(q0), qq);
            float sd  = dot(rel, qRest);
            float s   = (sd > 0.f) ? 1.f : -1.f;

            // qq * qRest.inverse() = qq * conjugate(qRest)
            Quatf prod = math::quat_multiply(qq, math::quat_conjugate(qRest));
            b = b + s * prod;
        }

        float lambda = length(sdir) + length(b);

        // Kit::Rotor(sdir) = pure quaternion (sdir.x, sdir.y, sdir.z, 0)
        Quatf r_sdir = Quatf(sdir.x, sdir.y, sdir.z, 0.f);
        // Kit::Rotor(b) = quaternion (b.x, b.y, b.z, b.w)
        Quatf r_b    = Quatf(b.x, b.y, b.z, b.w);
        // Kit::Rotor(1) = RotorX(1,0,0,0) — pure quaternion i, NOT identity!
        Quatf r_one  = Quatf(1.f, 0.f, 0.f, 0.f);

        Quatf product = math::quat_multiply(
            math::quat_multiply(r_sdir, r_b), r_one);

        // q0 = normalize(product.v + lambda * b)
        Vec4f sum = product + lambda * b;
        float n = length(sum);
        q0 = (n > 1e-12f) ? sum * (1.f / n) : Quatf(0.f, 0.f, 0.f, 1.f);

        qs[tid] = q0;
    }
}

// Host-callable wrappers.
void launch_cosserat_itr(YarnMetaData* d_meta, int n_verts, bool use_step_limit,
                          cudaStream_t stream) {
    int grid = (n_verts + YARN_VERTEX_PER_BLOCK - 2) / (YARN_VERTEX_PER_BLOCK - 1);
    if (use_step_limit)
        kernel_cosserat_itr<true><<<grid, YARN_BLOCK_SIZE, 0, stream>>>(d_meta);
    else
        kernel_cosserat_itr<false><<<grid, YARN_BLOCK_SIZE, 0, stream>>>(d_meta);
}

void launch_quaternion_lambda_itr(YarnMetaData* d_meta, int n_verts, cudaStream_t stream) {
    int blocks = (n_verts + 127) / 128;
    kernel_quaternion_lambda_itr<<<blocks, 128, 0, stream>>>(d_meta);
}

}  // namespace yarn
}  // namespace chysx
