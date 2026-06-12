// SPDX-License-Identifier: Apache-2.0
// CUDA implementation of chysx::collision::EFBroadphase.

#include "ef_broadphase.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace chysx {
namespace collision {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::collision::EFBroadphase: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlock = 128;
inline int grid(int n) { return (n + kBlock - 1) / kBlock; }

__global__ void ef_face_aabb_kernel(
    const math::Vec3i* __restrict__ faces, int n_faces,
    const math::Vec3f* __restrict__ pos,
    float thickness,
    Aabb*        __restrict__ aabbs,
    math::Vec3f* __restrict__ centers) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_faces) return;
    const math::Vec3i f = faces[i];
    const math::Vec3f a = pos[f.x];
    const math::Vec3f b = pos[f.y];
    const math::Vec3f c = pos[f.z];
    Aabb box;
    box.set(a, b);
    box.add(c);
    box.enlarge(thickness);
    aabbs[i] = box;
    centers[i] = math::Vec3f(0.5f * (box.mn.x + box.mx.x),
                              0.5f * (box.mn.y + box.mx.y),
                              0.5f * (box.mn.z + box.mx.z));
}

}  // namespace

void EFBroadphase::setup(const std::vector<math::Vec3i>& tris,
                          int n_verts,
                          int max_ef_candidates) {
    topology_.build(tris, n_verts);
    if (!topology_.valid()) return;

    int n_faces = topology_.n_faces();
    int n_edges = topology_.n_edges();

    max_ef_cand_ = (max_ef_candidates > 0)
                       ? max_ef_candidates
                       : std::max(n_edges * 8, 4096);

    face_aabbs_.resize(static_cast<std::size_t>(n_faces));
    face_centers_.resize(static_cast<std::size_t>(n_faces));

    bvh_.build(n_faces, max_ef_cand_);
}

void EFBroadphase::query(const math::Vec3f* positions_dev,
                          float thickness,
                          std::uintptr_t cuda_stream) {
    if (!topology_.valid()) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    const int n_faces = topology_.n_faces();
    const int n_edges = topology_.n_edges();

    // 1. Build face AABBs + centers
    ef_face_aabb_kernel<<<grid(n_faces), kBlock, 0, stream>>>(
        topology_.faces().gpu_data(), n_faces,
        positions_dev, thickness,
        face_aabbs_.gpu_data(), face_centers_.gpu_data());
    check_cuda(cudaGetLastError(), "ef_face_aabb_kernel");

    // 2. BVH refit over faces (with PackedFace for covertex filter)
    bvh_.refit(face_aabbs_.gpu_data(), face_centers_.gpu_data(),
               topology_.faces().gpu_data(),
               cuda_stream);

    // 3. Self-EF query: edge AABBs are built on the fly inside the kernel
    bvh_.query_self_ef(topology_.edges().gpu_data(),
                       positions_dev,
                       n_edges, thickness,
                       cuda_stream);
}

int EFBroadphase::ef_count(std::uintptr_t cuda_stream) {
    int count = 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    if (stream) cudaStreamSynchronize(stream);
    check_cuda(cudaMemcpy(&count, bvh_.query_count_dev(), sizeof(int),
                           cudaMemcpyDeviceToHost),
               "ef_count memcpy");
    return std::min(count, max_ef_cand_);
}

void EFBroadphase::download_pairs(std::vector<math::Vec2i>& out,
                                   std::uintptr_t cuda_stream) {
    int n = ef_count(cuda_stream);
    out.resize(n);
    if (n > 0) {
        check_cuda(cudaMemcpy(out.data(), bvh_.query_pairs_dev(),
                               n * sizeof(math::Vec2i),
                               cudaMemcpyDeviceToHost),
                   "download_pairs memcpy");
    }
}

}  // namespace collision
}  // namespace chysx
