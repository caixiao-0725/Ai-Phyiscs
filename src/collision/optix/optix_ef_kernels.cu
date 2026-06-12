// SPDX-License-Identifier: Apache-2.0
// CUDA kernels for the OptiX EF broadphase (AABB construction).

#include <cuda_runtime.h>

namespace chysx {
namespace collision {

// Build per-triangle AABBs (6 floats each: min_x, min_y, min_z, max_x, max_y, max_z)
__global__ void optix_ef_build_aabb_kernel(
    float* __restrict__ aabb_buffer,
    const float3* __restrict__ verts,
    const int3* __restrict__ tris,
    float thickness,
    int n_faces) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_faces) return;

    int3 f = tris[i];
    float3 v0 = verts[f.x], v1 = verts[f.y], v2 = verts[f.z];

    float mn_x = fminf(fminf(v0.x, v1.x), v2.x) - thickness;
    float mn_y = fminf(fminf(v0.y, v1.y), v2.y) - thickness;
    float mn_z = fminf(fminf(v0.z, v1.z), v2.z) - thickness;
    float mx_x = fmaxf(fmaxf(v0.x, v1.x), v2.x) + thickness;
    float mx_y = fmaxf(fmaxf(v0.y, v1.y), v2.y) + thickness;
    float mx_z = fmaxf(fmaxf(v0.z, v1.z), v2.z) + thickness;

    aabb_buffer[i * 6 + 0] = mn_x;
    aabb_buffer[i * 6 + 1] = mn_y;
    aabb_buffer[i * 6 + 2] = mn_z;
    aabb_buffer[i * 6 + 3] = mx_x;
    aabb_buffer[i * 6 + 4] = mx_y;
    aabb_buffer[i * 6 + 5] = mx_z;
}

void optix_ef_build_aabbs(float* aabb_buffer,
                           const float3* verts,
                           const int3* tris,
                           float thickness,
                           int n_faces,
                           cudaStream_t stream) {
    int block = 128;
    int grid = (n_faces + block - 1) / block;
    optix_ef_build_aabb_kernel<<<grid, block, 0, stream>>>(
        aabb_buffer, verts, tris, thickness, n_faces);
}

}  // namespace collision
}  // namespace chysx
