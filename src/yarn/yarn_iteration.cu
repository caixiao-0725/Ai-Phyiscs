// SPDX-License-Identifier: Apache-2.0
//
// Yarn time integration kernels: initItr (prediction) and endItr (advection).
// Direct port of YarnBall iteration.cu.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "yarn_types.cuh"

namespace chysx {
namespace yarn {

// Velocity → initial dx guess (with gravity extrapolation).
__global__ void kernel_init_itr(YarnMetaData* data) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= data->numVerts) return;

    const float h = data->h;
    auto verts    = data->d_verts;
    auto lastVels = data->d_lastVels;

    const Vec3f g   = data->gravity;
    const Vec3f vel = data->d_vels[tid];

    Vec3f dx = h * vel;
    Vec3f lastVel = lastVels[tid];
    lastVels[tid] = vel;

    float padding = data->scaledDetectionRadius;
    Vec3f y = dx + (h * h) * g;
    padding += length(y);

    if (verts[tid].invMass != 0.f) {
        // Store y = h*v + h²g in d_vels (vel no longer needed this step)
        data->d_vels[tid] = y;

        // Improved initial guess via acceleration projection onto gravity
        float g2 = dot(g, g);
        if (g2 > 0.f) {
            Vec3f a = (vel - lastVel) / data->lastH;
            float s = math::clamp(dot(a, g) / g2, 0.f, 1.f);
            dx = dx + (h * h * s) * g;
        }
    }

    data->d_paddingSize[tid] = padding;
    data->d_dx[tid] = dx;

    Vec3f pos = verts[tid].pos;
    data->d_lastPos[tid] = pos;
}

// dx → velocity, then advect positions.
__global__ void kernel_end_itr(YarnMetaData* data) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= data->numVerts) return;

    const float h    = data->h;
    const float invH = 1.f / h;
    auto verts       = data->d_verts;

    Vec3f dx = data->d_dx[tid];
    if (verts[tid].invMass != 0.f)
        data->d_vels[tid] = dx * invH * (1.f - data->drag * h);
    verts[tid].pos = verts[tid].pos + dx;
}

// Host-callable wrappers.
void launch_init_itr(YarnMetaData* d_meta, int n_verts, cudaStream_t stream) {
    int blocks = (n_verts + 255) / 256;
    kernel_init_itr<<<blocks, 256, 0, stream>>>(d_meta);
}

void launch_end_itr(YarnMetaData* d_meta, int n_verts, cudaStream_t stream) {
    int blocks = (n_verts + 255) / 256;
    kernel_end_itr<<<blocks, 256, 0, stream>>>(d_meta);
}

}  // namespace yarn
}  // namespace chysx
