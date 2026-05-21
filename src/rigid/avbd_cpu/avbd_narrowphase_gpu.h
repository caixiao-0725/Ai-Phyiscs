// SPDX-FileCopyrightText: 2026 NVIDIA Corporation
// SPDX-License-Identifier: MIT
//
// GPU-accelerated narrowphase (OBB-OBB SAT) for the AVBD solver.
// Takes broadphase pair list + SoA body data on GPU, outputs contacts.

#pragma once

#include "../../memory/cuda_array.h"
#include "../../math/vec.cuh"

namespace chysx {
namespace avbd {

constexpr int VERTEX_TABLE_MAX_NEIGHBORS = 8;

struct GpuContact {
    int feature_key;
    float rA_x, rA_y, rA_z;
    float rB_x, rB_y, rB_z;
};

struct GpuManifold {
    int body_a;
    int body_b;
    int num_contacts;
    int contact_offset;    // index into flat contact array
    float basis[9];        // 3x3 row-major
    float friction;
};

struct VertexEntry {
    int other_body;        // which body this body collides with
    int manifold_idx;      // index into the manifold array
};

class NarrowphaseGPU {
public:
    NarrowphaseGPU() = default;
    ~NarrowphaseGPU() = default;

    NarrowphaseGPU(const NarrowphaseGPU&) = delete;
    NarrowphaseGPU& operator=(const NarrowphaseGPU&) = delete;

    void build(int max_pairs);

    /// Run SAT narrowphase on GPU for all broadphase pairs.
    /// Body SoA data must already reside on the GPU (device pointers).
    /// pair_a_dev / pair_b_dev are device arrays of broadphase pair indices.
    /// Returns the number of manifolds that had contacts (written into manifolds/contacts).
    int query(const float* pos_x_dev, const float* pos_y_dev, const float* pos_z_dev,
              const float* quat_x_dev, const float* quat_y_dev,
              const float* quat_z_dev, const float* quat_w_dev,
              const float* half_x_dev, const float* half_y_dev, const float* half_z_dev,
              const float* friction_dev,
              const int* pair_a_dev, const int* pair_b_dev,
              int n_pairs, int n_bodies,
              GpuManifold* manifolds_out, GpuContact* contacts_out,
              int& total_contacts_out);

    // Per-body neighbor count / table (downloaded after query)
    const int* vertex_counts() const { return vtx_counts_.cpu_data(); }
    const VertexEntry* vertex_table() const { return vtx_table_.cpu_data(); }
    int vertex_table_stride() const { return VERTEX_TABLE_MAX_NEIGHBORS; }

private:
    int max_pairs_  = 0;
    int max_bodies_ = 0;

    CudaArray<GpuManifold> manifolds_;
    CudaArray<GpuContact>  contacts_;
    CudaArray<int>         manifold_count_;   // single int on GPU
    CudaArray<int>         contact_count_;    // single int on GPU

    // Per-body vertex table: vtx_table_[body * 8 + slot] = {other, manifold_idx}
    CudaArray<int>         vtx_counts_;       // [n_bodies], neighbor count per body
    CudaArray<VertexEntry> vtx_table_;        // [n_bodies * 8]
};

}  // namespace avbd
}  // namespace chysx
