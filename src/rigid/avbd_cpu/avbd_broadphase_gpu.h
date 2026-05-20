// SPDX-FileCopyrightText: 2026 NVIDIA Corporation
// SPDX-License-Identifier: MIT
//
// GPU-accelerated broadphase for the AVBD CPU solver.
// Uses QuantBvh to build a quantized stackless BVH over body AABBs
// and performs self-query to find overlapping pairs.

#pragma once

#include "../../collision/bvh/aabb.cuh"
#include "../../collision/bvh/quant_bvh.h"
#include "../../math/vec.cuh"
#include "../../memory/cuda_array.h"

namespace chysx {
namespace avbd {

class BroadphaseGPU {
public:
    BroadphaseGPU() = default;
    ~BroadphaseGPU() = default;

    BroadphaseGPU(const BroadphaseGPU&) = delete;
    BroadphaseGPU& operator=(const BroadphaseGPU&) = delete;

    void build(int max_bodies, int max_pairs);

    /// Upload SoA body data, compute AABBs on GPU, build QuantBvh,
    /// self-query, download results. Returns number of pairs found.
    /// pair_a / pair_b are CPU output arrays (must hold max_pairs ints).
    int query(const float* pos_x, const float* pos_y, const float* pos_z,
              const float* quat_x, const float* quat_y, const float* quat_z,
              const float* quat_w,
              const float* half_x, const float* half_y, const float* half_z,
              const float* mass,
              int n_bodies,
              int* pair_a, int* pair_b);

private:
    int max_bodies_ = 0;
    int max_pairs_  = 0;

    CudaArray<float> pos_x_, pos_y_, pos_z_;
    CudaArray<float> quat_x_, quat_y_, quat_z_, quat_w_;
    CudaArray<float> half_x_, half_y_, half_z_;
    CudaArray<float> mass_;

    CudaArray<collision::Aabb> aabbs_;
    CudaArray<math::Vec3f>     centers_;

    collision::QuantBvh bvh_;
};

}  // namespace avbd
}  // namespace chysx
