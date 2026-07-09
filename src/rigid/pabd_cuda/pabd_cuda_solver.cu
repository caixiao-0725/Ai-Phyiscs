// SPDX-License-Identifier: Apache-2.0

#include "pabd_cuda_solver.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "../../collision/contact_spmv.h"
#include "../../io/obj_io.h"

namespace chysx {
namespace rigid {
namespace pabd_cuda {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("PabdCudaSolver: ") +
                                 what + " failed: " + cudaGetErrorString(err));
    }
}

inline std::uintptr_t stream_handle(CUstream_st* stream) {
    return reinterpret_cast<std::uintptr_t>(stream);
}

CHYSX_HDI math::Mat3f make_columns(math::Vec3f c0, math::Vec3f c1, math::Vec3f c2) {
    return math::Mat3f(c0.x, c1.x, c2.x,
                       c0.y, c1.y, c2.y,
                       c0.z, c1.z, c2.z);
}

CHYSX_HDI math::Mat3f polar_rotation(math::Mat3f F) {
    math::Mat3f R = F;
    for (int iter = 0; iter < 10; ++iter) {
        const math::Mat3f inv_t = math::transpose(math::inverse(R));
        R = (R + inv_t) * 0.5f;
    }

    if (math::determinant(R) < 0.0f) {
        int column = 0;
        float min_len = 3.402823466e38f;
        for (int c = 0; c < 3; ++c) {
            const float len = R(0, c) * R(0, c)
                            + R(1, c) * R(1, c)
                            + R(2, c) * R(2, c);
            if (len < min_len) {
                min_len = len;
                column = c;
            }
        }
        for (int row = 0; row < 3; ++row) {
            R(row, column) = -R(row, column);
        }
    }
    return R;
}

__device__ void atomic_add_vec(math::Vec3f* dst, math::Vec3f v) {
    atomicAdd(&dst->x, v.x);
    atomicAdd(&dst->y, v.y);
    atomicAdd(&dst->z, v.z);
}

__device__ void atomic_add_scalar_identity(math::Mat3f* dst, float s) {
    atomicAdd(&dst->data[0], s);
    atomicAdd(&dst->data[4], s);
    atomicAdd(&dst->data[8], s);
}

__device__ void add_scalar_identity_by_slot_device(math::Mat3f* diag,
                                                   math::Mat3f* values,
                                                   int slot,
                                                   float value) {
    if (slot < 0) {
        atomic_add_scalar_identity(&diag[-slot - 1], value);
    } else {
        atomic_add_scalar_identity(&values[slot], value);
    }
}

__global__ void prepare_step_kernel(const math::Vec3f* rest,
                                    math::Vec3f* x,
                                    math::Vec3f* old_x,
                                    math::Vec3f* y,
                                    math::Vec3f* velocity,
                                    const unsigned char* fixed,
                                    int n,
                                    float h,
                                    float gravity) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    old_x[i] = x[i];
    if (fixed[i]) {
        y[i] = rest[i];
    } else {
        y[i] = x[i] + velocity[i] * h + math::Vec3f(0.0f, gravity * h * h, 0.0f);
    }
}

__global__ void prepare_interpolated_step_kernel(const math::Vec3f* rest,
                                                 math::Vec3f* x,
                                                 math::Vec3f* old_x,
                                                 math::Vec3f* y,
                                                 math::Vec3f* velocity,
                                                 const unsigned char* fixed,
                                                 int n,
                                                 float h,
                                                 float gravity,
                                                 float damping) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    old_x[i] = x[i];
    if (fixed[i]) {
        velocity[i] = math::Vec3f(0.0f, 0.0f, 0.0f);
        y[i] = rest[i];
        x[i] = rest[i];
    } else {
        velocity[i] = velocity[i] * damping +
                      math::Vec3f(0.0f, gravity * h, 0.0f);
        y[i] = x[i] + velocity[i] * h;
        x[i] = y[i];
    }
}

__global__ void update_interpolated_surface_positions_kernel(
    const PabdSurfaceMapDevice* maps,
    int n,
    const math::Vec3f* controls,
    math::Vec3f* surface_positions) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const PabdSurfaceMapDevice map = maps[i];
    math::Vec3f p(0.0f, 0.0f, 0.0f);
    for (int k = 0; k < 4; ++k) {
        const int id = map.index[k];
        const float w = map.weight[k];
        if (id >= 0 && w != 0.0f) {
            p += controls[id] * w;
        }
    }
    surface_positions[i] = p;
}

__global__ void assemble_interpolated_mass_elastic_kernel(
    math::Mat3f* diag,
    math::Mat3f* values,
    math::Vec3f* rhs,
    const PabdCudaSolver::TetData* tets,
    const float* mass_blocks,
    int n_tets,
    const math::Vec3f* y,
    const math::Vec3f* x,
    float inv_h2,
    float stiffness) {
    const int elem_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_id >= n_tets) return;

    const PabdCudaSolver::TetData tet = tets[elem_id];
    const float* mass = mass_blocks + elem_id * 16;

    const math::Mat3f Ds = make_columns(x[tet.v[1]] - x[tet.v[0]],
                                        x[tet.v[2]] - x[tet.v[0]],
                                        x[tet.v[3]] - x[tet.v[0]]);
    const math::Mat3f R = polar_rotation(Ds * tet.inv_dm);
    const float w_elastic = stiffness * tet.volume;

    for (int row = 0; row < 4; ++row) {
        const int row_id = tet.v[row];
        for (int col = 0; col < 4; ++col) {
            const int col_id = tet.v[col];
            const float m = mass[row * 4 + col] * inv_h2;
            const int slot = tet.hessian_slot[row * 4 + col];
            add_scalar_identity_by_slot_device(diag, values, slot, m);
            atomic_add_vec(&rhs[row_id], y[col_id] * m);

            const float k = w_elastic * math::dot(tet.grad[row], tet.grad[col]);
            add_scalar_identity_by_slot_device(diag, values, slot, k);
        }
        atomic_add_vec(&rhs[row_id], (R * tet.grad[row]) * w_elastic);
    }
}

__global__ void assemble_interpolated_fixed_kernel(
    math::Mat3f* diag,
    math::Vec3f* rhs,
    const math::Vec3f* rest,
    const unsigned char* fixed,
    int n,
    float fixed_weight) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !fixed[i]) return;
    atomic_add_scalar_identity(&diag[i], fixed_weight);
    atomic_add_vec(&rhs[i], rest[i] * fixed_weight);
}

__device__ void merge_wide_coeff(int dof,
                                 float coeff,
                                 int* dofs,
                                 float* coeffs,
                                 int* coeff_count) {
    if (dof < 0 || coeff == 0.0f) return;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (i >= *coeff_count) break;
        if (dofs[i] == dof) {
            coeffs[i] += coeff;
            return;
        }
    }
    if (*coeff_count < 8) {
        const int id = *coeff_count;
        dofs[id] = dof;
        coeffs[id] = coeff;
        *coeff_count = id + 1;
    }
}

__device__ void store_wide_contact(collision::WideContact* wide_contacts,
                                   int cid,
                                   const int* dofs,
                                   const float* coeffs,
                                   int coeff_count,
                                   math::Vec3f normal,
                                   float target,
                                   float stiffness) {
    collision::WideContact out{};
    out.ids0 = math::Vec4i(-1, -1, -1, -1);
    out.ids1 = math::Vec4i(-1, -1, -1, -1);
    out.weights0 = math::Vec4f(0.0f, 0.0f, 0.0f, 0.0f);
    out.weights1 = math::Vec4f(0.0f, 0.0f, 0.0f, 0.0f);
    out.normal_target = math::Vec4f(normal.x, normal.y, normal.z, target);
    out.stiffness = stiffness;
    for (int i = 0; i < coeff_count && i < 8; ++i) {
        if (i < 4) {
            out.ids0[i] = dofs[i];
            out.weights0[i] = coeffs[i];
        } else {
            out.ids1[i - 4] = dofs[i];
            out.weights1[i - 4] = coeffs[i];
        }
    }
    wide_contacts[cid] = out;
}

__global__ void append_interpolated_ground_contacts_kernel(
    math::Vec3f* rhs,
    const PabdSurfaceMapDevice* maps,
    const math::Vec3f* surface_positions,
    int n_surface,
    float target_y,
    float stiffness,
    int* ground_count,
    collision::WideContact* wide_contacts,
    int* wide_count,
    int wide_capacity) {
    const int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= n_surface || stiffness <= 0.0f) return;

    const PabdSurfaceMapDevice map = maps[sid];
    if (!map.ground_collide || surface_positions[sid].y >= target_y) return;
    atomicAdd(ground_count, 1);

    int dofs[8];
    float coeffs[8];
    int coeff_count = 0;
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int ia = map.index[a];
        const float wa = map.weight[a];
        if (ia < 0 || wa == 0.0f) continue;
        merge_wide_coeff(ia, wa, dofs, coeffs, &coeff_count);
        atomic_add_vec(&rhs[ia],
                       math::Vec3f(0.0f, stiffness * wa * target_y, 0.0f));
    }
    if (coeff_count > 0) {
        const int cid = atomicAdd(wide_count, 1);
        if (cid < wide_capacity) {
            store_wide_contact(wide_contacts, cid, dofs, coeffs, coeff_count,
                               math::Vec3f(0.0f, 1.0f, 0.0f),
                               target_y, stiffness);
        }
    }
}

__global__ void append_interpolated_self_contacts_kernel(
    math::Vec3f* rhs,
    const collision::MeshMeshContact* contacts,
    const int* contact_count,
    int max_contacts,
    const PabdSurfaceMapDevice* maps,
    float contact_gap,
    float stiffness,
    int* debug_counts,
    math::Vec3f* normal_sum,
    collision::WideContact* wide_contacts,
    int* wide_count,
    int wide_capacity) {
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_raw = *contact_count;
    const int n = n_raw < max_contacts ? n_raw : max_contacts;
    if (cid >= n || stiffness <= 0.0f) return;

    const collision::MeshMeshContact c = contacts[cid];
    if (c.type == static_cast<int>(collision::MeshMeshContactType::PointFace)) {
        atomicAdd(&debug_counts[3], 1);
    } else if (c.type == static_cast<int>(collision::MeshMeshContactType::EdgeEdge)) {
        atomicAdd(&debug_counts[4], 1);
    }
    if (c.distance >= contact_gap) return;

    atomicAdd(&debug_counts[0], 1);
    atomic_add_vec(normal_sum, c.normal);
    if (math::abs(c.normal.y) >= 0.75f) {
        atomicAdd(&debug_counts[1], 1);
    } else {
        atomicAdd(&debug_counts[2], 1);
    }

    int dofs[8];
    float coeffs[8];
    int coeff_count = 0;
    const int ids[4] = {c.vertices.x, c.vertices.y, c.vertices.z, c.vertices.w};
    const float ws[4] = {c.weights.x, c.weights.y, c.weights.z, c.weights.w};
    #pragma unroll
    for (int si = 0; si < 4; ++si) {
        const PabdSurfaceMapDevice map = maps[ids[si]];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float scalar = ws[si] * map.weight[k];
            merge_wide_coeff(map.index[k], scalar, dofs, coeffs,
                             &coeff_count);
        }
    }

    for (int i = 0; i < coeff_count; ++i) {
        const int di = dofs[i];
        const float ci = coeffs[i];
        atomic_add_vec(&rhs[di],
                       c.normal * (stiffness * ci * contact_gap));
    }
    if (coeff_count > 0) {
        const int out_cid = atomicAdd(wide_count, 1);
        if (out_cid < wide_capacity) {
            store_wide_contact(wide_contacts, out_cid, dofs, coeffs,
                               coeff_count, c.normal, contact_gap, stiffness);
        }
    }
}

__global__ void assemble_base_matrix_kernel(math::Mat3f* diag,
                                            const float* mass,
                                            const unsigned char* fixed,
                                            int n,
                                            float inv_h2,
                                            float fixed_weight) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float s = mass[i] * inv_h2 + (fixed[i] ? fixed_weight : 0.0f);
    diag[i] = math::Mat3f::identity() * s;
}

__global__ void add_tet_matrix_kernel(math::Mat3f* diag,
                                      math::Mat3f* values,
                                      const PabdCudaSolver::TetData* tets,
                                      int num_tets,
                                      float stiffness) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_tets * 16) return;

    const int tet_id = id / 16;
    const int local = id - tet_id * 16;
    const int a = local / 4;
    const int b = local - a * 4;
    const auto tet = tets[tet_id];

    const float s = stiffness * tet.volume * math::dot(tet.grad[a], tet.grad[b]);
    const int slot = tet.hessian_slot[local];
    if (slot < 0) {
        atomic_add_scalar_identity(&diag[-slot - 1], s);
    } else {
        atomic_add_scalar_identity(&values[slot], s);
    }
}

__global__ void add_ground_matrix_kernel(math::Mat3f* diag,
                                         const math::Vec3f* x,
                                         const unsigned char* fixed,
                                         int* contact_count,
                                         int n,
                                         float ground_y,
                                         float gap,
                                         float stiffness) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || fixed[i] || stiffness <= 0.0f) return;

    const float target = ground_y + gap;
    if (x[i].y >= target) return;

    // Vertex-ground contact: k * (x.y - target)^2 / 2.
    atomicAdd(&diag[i].data[4], stiffness);
    atomicAdd(contact_count, 1);
}

__global__ void rotation_kernel(const math::Vec3f* x,
                                const PabdCudaSolver::TetData* tets,
                                math::Mat3f* rotations,
                                int num_tets) {
    const int tet_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (tet_id >= num_tets) return;

    const auto tet = tets[tet_id];
    const math::Vec3f x0 = x[tet.v[0]];
    const math::Mat3f Ds = make_columns(x[tet.v[1]] - x0,
                                        x[tet.v[2]] - x0,
                                        x[tet.v[3]] - x0);
    rotations[tet_id] = polar_rotation(Ds * tet.inv_dm);
}

__global__ void assemble_base_rhs_kernel(math::Vec3f* rhs,
                                         const math::Vec3f* rest,
                                         const math::Vec3f* y,
                                         const float* mass,
                                         const unsigned char* fixed,
                                         int n,
                                         float inv_h2,
                                         float fixed_weight) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    math::Vec3f value = y[i] * (mass[i] * inv_h2);
    if (fixed[i]) {
        value += rest[i] * fixed_weight;
    }
    rhs[i] = value;
}

__global__ void add_tet_rhs_kernel(math::Vec3f* rhs,
                                   const PabdCudaSolver::TetData* tets,
                                   const math::Mat3f* rotations,
                                   int num_tets,
                                   float stiffness) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_tets * 4) return;

    const int tet_id = id / 4;
    const int local = id - tet_id * 4;
    const auto tet = tets[tet_id];
    const float w = stiffness * tet.volume;
    const math::Vec3f value = rotations[tet_id] * tet.grad[local] * w;
    atomic_add_vec(&rhs[tet.v[local]], value);
}

__global__ void add_ground_rhs_kernel(math::Vec3f* rhs,
                                      const math::Vec3f* x,
                                      const unsigned char* fixed,
                                      int n,
                                      float ground_y,
                                      float gap,
                                      float stiffness) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || fixed[i] || stiffness <= 0.0f) return;

    const float target = ground_y + gap;
    if (x[i].y >= target) return;
    rhs[i].y += stiffness * target;
}

__global__ void add_self_contact_rhs_kernel(
    const math::Vec4i* __restrict__ pairs,
    const collision::ContactWeights* __restrict__ weights,
    const int* __restrict__ count_ptr,
    int max_contacts,
    float stiffness,
    const math::Vec3f* __restrict__ x,
    math::Vec3f* __restrict__ rhs) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_raw = *count_ptr;
    const int n = (n_raw < max_contacts) ? n_raw : max_contacts;
    if (c >= n || stiffness <= 0.0f) return;

    const math::Vec4i ids = pairs[c];
    const collision::ContactWeights w = weights[c];
    const int idxs[4] = {ids.x, ids.y, ids.z, ids.w};
    const float ws[4] = {w.w0, w.w1, w.w2, w.w3};
    const math::Vec3f normal(w.nx, w.ny, w.nz);

    math::Vec3f g(0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        g += x[idxs[j]] * ws[j];
    }

    const float scale = stiffness * (math::dot(normal, g) + w.depth);
    const math::Vec3f value = normal * scale;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float wi = ws[i];
        if (wi == 0.0f) continue;
        atomic_add_vec(&rhs[idxs[i]], value * wi);
    }
}

__global__ void apply_fixed_kernel(math::Vec3f* x,
                                   const math::Vec3f* rest,
                                   const unsigned char* fixed,
                                   int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !fixed[i]) return;
    x[i] = rest[i];
}

__global__ void update_velocity_kernel(math::Vec3f* velocity,
                                       const math::Vec3f* x,
                                       const math::Vec3f* old_x,
                                       const unsigned char* fixed,
                                       int n,
                                       float inv_h,
                                       float damping) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (fixed[i]) {
        velocity[i] = math::Vec3f(0.0f, 0.0f, 0.0f);
    } else {
        velocity[i] = (x[i] - old_x[i]) * inv_h * damping;
    }
}

float signed_tet_volume(math::Vec3f a, math::Vec3f b,
                        math::Vec3f c, math::Vec3f d) {
    return math::dot(b - a, math::cross(c - a, d - a)) / 6.0f;
}

std::array<int, 3> sorted_face(std::array<int, 3> f) {
    std::sort(f.begin(), f.end());
    return f;
}

std::vector<std::array<int, 3>> derive_surface_triangles(
    const std::vector<std::array<int, 4>>& tets) {
    std::map<std::array<int, 3>, std::array<int, 3>> owner;
    std::map<std::array<int, 3>, int> count;

    for (const auto& tv : tets) {
        const std::array<std::array<int, 3>, 4> faces = {{
            {tv[0], tv[2], tv[1]},
            {tv[0], tv[1], tv[3]},
            {tv[1], tv[2], tv[3]},
            {tv[2], tv[0], tv[3]},
        }};
        for (const auto& face : faces) {
            const auto key = sorted_face(face);
            owner[key] = face;
            ++count[key];
        }
    }

    std::vector<std::array<int, 3>> surface;
    for (const auto& item : count) {
        if (item.second == 1) {
            surface.push_back(owner[item.first]);
        }
    }
    return surface;
}

}  // namespace

PabdCudaSolver::~PabdCudaSolver() { release_stream(); }

void PabdCudaSolver::release_stream() noexcept {
    if (stream_) {
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
        stream_ = nullptr;
    }
}

void PabdCudaSolver::setup(const PabdCudaMesh& mesh,
                           const PabdCudaParams& params) {
    params_ = params;
    release_stream();
    check_cuda(cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&stream_)),
               "cudaStreamCreate");

    build_host_data(mesh);
    upload_static_data();

    pcg_.initialize(num_vertices_);
    initialized_ = true;
    update_host_positions();
}

void PabdCudaSolver::build_host_data(const PabdCudaMesh& mesh) {
    if (mesh.rest_positions.empty()) {
        throw std::invalid_argument("PabdCudaSolver::setup: empty rest_positions");
    }
    if (mesh.tets.empty()) {
        throw std::invalid_argument("PabdCudaSolver::setup: empty tets");
    }

    num_vertices_ = static_cast<int>(mesh.rest_positions.size());
    interpolated_surface_ = !mesh.surface_maps.empty();
    surface_maps_ = mesh.surface_maps;
    surface_edges_ = mesh.surface_edges;
    num_surface_vertices_ = interpolated_surface_
        ? static_cast<int>(surface_maps_.size())
        : num_vertices_;
    host_tets_.clear();
    host_mass_blocks_.clear();
    host_surface_positions_.assign(
        static_cast<std::size_t>(std::max(0, num_surface_vertices_)),
        math::Vec3f());
    rest_.resize(num_vertices_);
    x_.resize(num_vertices_);
    old_x_.resize(num_vertices_);
    y_.resize(num_vertices_);
    velocity_.resize(num_vertices_);
    rhs_.resize(num_vertices_);
    mass_.resize(num_vertices_);
    fixed_.resize(num_vertices_);
    ground_contacts_.resize(1);
    ground_contacts_[0] = 0;

    for (int i = 0; i < num_vertices_; ++i) {
        rest_[i] = mesh.rest_positions[i];
        x_[i] = mesh.rest_positions[i];
        old_x_[i] = mesh.rest_positions[i];
        y_[i] = mesh.rest_positions[i];
        velocity_[i] = (i < static_cast<int>(mesh.initial_velocities.size()))
            ? mesh.initial_velocities[i]
            : math::Vec3f(0.0f, 0.0f, 0.0f);
        rhs_[i] = math::Vec3f(0.0f, 0.0f, 0.0f);
        mass_[i] = 0.0f;
        fixed_[i] = (i < static_cast<int>(mesh.fixed.size())) ? mesh.fixed[i] : 0;
    }

    std::vector<std::array<int, 4>> oriented_tets = mesh.tets;
    std::vector<TetData> host_tets(oriented_tets.size());

    for (std::size_t t = 0; t < oriented_tets.size(); ++t) {
        auto& tv = oriented_tets[t];
        for (int k = 0; k < 4; ++k) {
            if (tv[k] < 0 || tv[k] >= num_vertices_) {
                throw std::out_of_range("PabdCudaSolver::setup: tet index out of range");
            }
        }

        float volume = signed_tet_volume(rest_[tv[0]], rest_[tv[1]],
                                         rest_[tv[2]], rest_[tv[3]]);
        if (volume < 0.0f) {
            std::swap(tv[2], tv[3]);
            volume = -volume;
        }
        if (t < mesh.tet_volume_overrides.size() &&
            mesh.tet_volume_overrides[t] > 0.0f) {
            volume = mesh.tet_volume_overrides[t];
        }
        volume = std::max(volume, 1.0e-8f);

        TetData tet{};
        for (int k = 0; k < 4; ++k) {
            tet.v[k] = tv[k];
        }
        tet.inv_dm = math::inverse(make_columns(rest_[tv[1]] - rest_[tv[0]],
                                                rest_[tv[2]] - rest_[tv[0]],
                                                rest_[tv[3]] - rest_[tv[0]]));
        tet.volume = volume;
        tet.grad[1] = math::Vec3f(tet.inv_dm(0, 0), tet.inv_dm(0, 1), tet.inv_dm(0, 2));
        tet.grad[2] = math::Vec3f(tet.inv_dm(1, 0), tet.inv_dm(1, 1), tet.inv_dm(1, 2));
        tet.grad[3] = math::Vec3f(tet.inv_dm(2, 0), tet.inv_dm(2, 1), tet.inv_dm(2, 2));
        tet.grad[0] = -(tet.grad[1] + tet.grad[2] + tet.grad[3]);
        host_tets[t] = tet;

        if (t < mesh.tet_mass_blocks.size()) {
            for (int row = 0; row < 4; ++row) {
                float row_mass = 0.0f;
                for (int col = 0; col < 4; ++col) {
                    row_mass += mesh.tet_mass_blocks[t][row * 4 + col] *
                                params_.density;
                }
                mass_[tv[row]] += row_mass;
            }
        } else {
            for (int k = 0; k < 4; ++k) {
                mass_[tv[k]] += volume * params_.density;
            }
        }
    }

    for (int i = 0; i < num_vertices_; ++i) {
        if (mass_[i] <= 0.0f) {
            mass_[i] = params_.density;
        }
    }

    std::vector<int> rows;
    std::vector<int> cols;
    rows.reserve(host_tets.size() * 12);
    cols.reserve(host_tets.size() * 12);
    for (const auto& tet : host_tets) {
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                if (a == b) continue;
                rows.push_back(tet.v[a]);
                cols.push_back(tet.v[b]);
            }
        }
    }
    H_.build_topology(num_vertices_, rows.data(), cols.data(),
                      static_cast<int>(rows.size()));

    std::vector<int> query_rows(host_tets.size() * 16);
    std::vector<int> query_cols(host_tets.size() * 16);
    std::vector<int> slots(host_tets.size() * 16);
    for (std::size_t t = 0; t < host_tets.size(); ++t) {
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                const int id = static_cast<int>(t) * 16 + a * 4 + b;
                query_rows[id] = host_tets[t].v[a];
                query_cols[id] = host_tets[t].v[b];
            }
        }
    }
    H_.resolve_slots(query_rows.data(), query_cols.data(), slots.data(),
                     static_cast<int>(slots.size()));
    for (std::size_t t = 0; t < host_tets.size(); ++t) {
        for (int k = 0; k < 16; ++k) {
            host_tets[t].hessian_slot[k] = slots[t * 16 + k];
        }
    }

    host_tets_ = host_tets;
    host_mass_blocks_.resize(host_tets.size());
    for (std::size_t t = 0; t < host_tets.size(); ++t) {
        host_mass_blocks_[t].fill(0.0f);
        if (t < mesh.tet_mass_blocks.size()) {
            for (int i = 0; i < 16; ++i) {
                host_mass_blocks_[t][i] =
                    mesh.tet_mass_blocks[t][i] * params_.density;
            }
        } else {
            const float lumped = host_tets[t].volume * params_.density;
            for (int i = 0; i < 4; ++i) {
                host_mass_blocks_[t][i * 4 + i] = lumped;
            }
        }
    }
    if (interpolated_surface_) {
        tet_mass_blocks_dev_.resize(host_mass_blocks_.size() * 16);
        for (std::size_t t = 0; t < host_mass_blocks_.size(); ++t) {
            for (int i = 0; i < 16; ++i) {
                tet_mass_blocks_dev_[t * 16 + i] = host_mass_blocks_[t][i];
            }
        }
    } else {
        tet_mass_blocks_dev_.clear();
    }

    tets_.resize(host_tets.size());
    rotations_.resize(host_tets.size());
    for (std::size_t i = 0; i < host_tets.size(); ++i) {
        tets_[i] = host_tets[i];
        rotations_[i] = math::Mat3f::identity();
    }

    surface_triangles_ = mesh.surface_triangles.empty()
        ? derive_surface_triangles(oriented_tets)
        : mesh.surface_triangles;

    std::vector<math::Vec3i> tri_vec;
    tri_vec.reserve(surface_triangles_.size());
    for (const auto& tri : surface_triangles_) {
        tri_vec.push_back(math::Vec3i(tri[0], tri[1], tri[2]));
    }
    if (!interpolated_surface_ && !tri_vec.empty()) {
        mesh_topology_.build(tri_vec, num_vertices_);
        self_collision_detector_.bind_topology(&mesh_topology_);
        const int max_contacts = std::max(1, params_.self_collision_max_contacts);
        const int max_ef_candidates =
            std::max(max_contacts, mesh_topology_.n_edges() * 64);
        self_collision_detector_.reserve(max_contacts, max_ef_candidates);
    } else {
        self_collision_detector_.bind_topology(nullptr);
    }
    self_collision_.set_stiffness(params_.self_collision_stiffness);

    flat_positions_.assign(static_cast<std::size_t>(num_surface_vertices_) * 3, 0.0f);
    flat_triangles_.resize(surface_triangles_.size() * 3);
    for (std::size_t i = 0; i < surface_triangles_.size(); ++i) {
        flat_triangles_[3 * i + 0] = surface_triangles_[i][0];
        flat_triangles_[3 * i + 1] = surface_triangles_[i][1];
        flat_triangles_[3 * i + 2] = surface_triangles_[i][2];
    }

    interpolated_body_count_ = 0;
    for (const PabdSurfaceMap& map : surface_maps_) {
        interpolated_body_count_ = std::max(interpolated_body_count_, map.body + 1);
    }

    if (interpolated_surface_) {
        surface_maps_dev_.resize(surface_maps_.size());
        for (std::size_t i = 0; i < surface_maps_.size(); ++i) {
            PabdSurfaceMapDevice dev{};
            for (int k = 0; k < 4; ++k) {
                dev.index[k] = surface_maps_[i].index[k];
                dev.weight[k] = surface_maps_[i].weight[k];
            }
            dev.normal = surface_maps_[i].normal;
            dev.body = surface_maps_[i].body;
            dev.collide = surface_maps_[i].collide;
            dev.self_collide = surface_maps_[i].self_collide;
            dev.ground_collide = surface_maps_[i].ground_collide;
            surface_maps_dev_[i] = dev;
        }

        surface_triangles_dev_.resize(surface_triangles_.size());
        for (std::size_t i = 0; i < surface_triangles_.size(); ++i) {
            surface_triangles_dev_[i] = math::Vec3i(
                surface_triangles_[i][0],
                surface_triangles_[i][1],
                surface_triangles_[i][2]);
        }

        surface_edges_dev_.resize(surface_edges_.size());
        for (std::size_t i = 0; i < surface_edges_.size(); ++i) {
            surface_edges_dev_[i] = math::Vec2i(surface_edges_[i][0],
                                                surface_edges_[i][1]);
        }

        interpolated_surface_positions_dev_.resize(
            static_cast<std::size_t>(num_surface_vertices_));
        interpolated_debug_counts_.resize(5);
        for (int i = 0; i < 5; ++i) {
            interpolated_debug_counts_[i] = 0;
        }
        interpolated_normal_sum_.resize(1);
        interpolated_normal_sum_[0] = math::Vec3f(0.0f, 0.0f, 0.0f);
        interpolated_contact_capacity_ =
            std::max(1, params_.self_collision_max_contacts);
        const int wide_capacity =
            interpolated_contact_capacity_ +
            std::max(0, num_surface_vertices_);
        interpolated_wide_contacts_.resize(
            static_cast<std::size_t>(wide_capacity));
        interpolated_wide_contact_count_.resize(1);
        interpolated_wide_contact_count_[0] = 0;

        std::vector<int> vertex_body_ids(surface_maps_.size(), 0);
        for (std::size_t i = 0; i < surface_maps_.size(); ++i) {
            vertex_body_ids[i] = surface_maps_[i].body;
        }
        const int max_ef_candidates = std::max(
            interpolated_contact_capacity_,
            static_cast<int>(surface_edges_.size()) * 64);
        mesh_mesh_contact_detector_.setup(
            tri_vec, vertex_body_ids, interpolated_contact_capacity_,
            max_ef_candidates);
    } else {
        surface_maps_dev_.clear();
        surface_triangles_dev_.clear();
        surface_edges_dev_.clear();
        interpolated_surface_positions_dev_.clear();
        interpolated_debug_counts_.clear();
        interpolated_normal_sum_.clear();
        interpolated_wide_contacts_.clear();
        interpolated_wide_contact_count_.clear();
        interpolated_contact_capacity_ = 0;
        mesh_mesh_contact_detector_ = collision::MeshMeshContactDetector{};
    }
}

void PabdCudaSolver::upload_static_data() {
    const std::uintptr_t s = stream_handle(stream_);
    rest_.copy_to_device(s);
    x_.copy_to_device(s);
    old_x_.copy_to_device(s);
    y_.copy_to_device(s);
    velocity_.copy_to_device(s);
    rhs_.copy_to_device(s);
    mass_.copy_to_device(s);
    fixed_.copy_to_device(s);
    tets_.copy_to_device(s);
    rotations_.copy_to_device(s);
    ground_contacts_.copy_to_device(s);
    if (interpolated_surface_) {
        surface_maps_dev_.copy_to_device(s);
        surface_triangles_dev_.copy_to_device(s);
        surface_edges_dev_.copy_to_device(s);
        interpolated_surface_positions_dev_.copy_to_device(s);
        interpolated_wide_contacts_.copy_to_device(s);
        interpolated_wide_contact_count_.copy_to_device(s);
        tet_mass_blocks_dev_.copy_to_device(s);
        interpolated_debug_counts_.copy_to_device(s);
        interpolated_normal_sum_.copy_to_device(s);
    }
    check_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)),
               "upload_static_data sync");
}

void PabdCudaSolver::update_surface_positions(
    const std::vector<math::Vec3f>& controls,
    std::vector<math::Vec3f>& surface) const {
    surface.assign(surface_maps_.size(), math::Vec3f());
    for (std::size_t i = 0; i < surface_maps_.size(); ++i) {
        const PabdSurfaceMap& map = surface_maps_[i];
        math::Vec3f p(0.0f, 0.0f, 0.0f);
        for (int k = 0; k < 4; ++k) {
            if (map.index[k] >= 0 && map.weight[k] != 0.0f) {
                p += controls[map.index[k]] * map.weight[k];
            }
        }
        surface[i] = p;
    }
}

void PabdCudaSolver::update_interpolated_surface_positions_gpu(
    const math::Vec3f* controls_dev) {
    if (!interpolated_surface_ || num_surface_vertices_ <= 0 ||
        controls_dev == nullptr) {
        return;
    }

    auto* stream = reinterpret_cast<cudaStream_t>(stream_);
    constexpr int block = 128;
    const int grid = (num_surface_vertices_ + block - 1) / block;
    update_interpolated_surface_positions_kernel<<<grid, block, 0, stream>>>(
        surface_maps_dev_.gpu_data(),
        num_surface_vertices_,
        controls_dev,
        interpolated_surface_positions_dev_.gpu_data());
    check_cuda(cudaGetLastError(),
               "update_interpolated_surface_positions_kernel");
}

void PabdCudaSolver::detect_interpolated_contacts_gpu() {
    if (!interpolated_surface_ ||
        params_.self_collision_thickness <= 0.0f ||
        params_.self_collision_stiffness <= 0.0f ||
        interpolated_contact_capacity_ <= 0) {
        last_self_broadphase_pairs_ = 0;
        last_self_broadphase_capacity_ = 0;
        last_self_broadphase_overflow_ = false;
        last_self_point_face_contacts_ = 0;
        last_self_edge_edge_contacts_ = 0;
        return;
    }

    const std::uintptr_t s = stream_handle(stream_);
    mesh_mesh_contact_detector_.detect_gpu(
        interpolated_surface_positions_dev_.gpu_data(),
        params_.self_collision_thickness, s);
    last_self_broadphase_pairs_ = mesh_mesh_contact_detector_.last_ef_count();
    last_self_broadphase_capacity_ =
        mesh_mesh_contact_detector_.broadphase().max_ef_candidates();
    last_self_broadphase_overflow_ =
        last_self_broadphase_capacity_ > 0 &&
        last_self_broadphase_pairs_ >= last_self_broadphase_capacity_;
    if (last_self_broadphase_overflow_) {
        std::fprintf(stderr,
                     "[PABD_ERROR] self EF broadphase buffer overflow: pairs=%d capacity=%d. "
                     "Some edge-face candidates were dropped before narrow phase.\n",
                     last_self_broadphase_pairs_,
                     last_self_broadphase_capacity_);
    }
}

void PabdCudaSolver::assemble_interpolated_system_gpu(float h) {
    const std::uintptr_t s = stream_handle(stream_);
    auto* stream = reinterpret_cast<cudaStream_t>(stream_);
    constexpr int block = 128;

    H_.set_zero(s);
    check_cuda(cudaMemsetAsync(rhs_.gpu_data(), 0,
                               rhs_.gpu_size() * sizeof(math::Vec3f),
                               stream),
               "interpolated rhs memset");
    check_cuda(cudaMemsetAsync(ground_contacts_.gpu_data(), 0, sizeof(int),
                               stream),
               "interpolated ground count memset");
    check_cuda(cudaMemsetAsync(interpolated_wide_contact_count_.gpu_data(), 0,
                               sizeof(int),
                               stream),
               "interpolated wide contact count memset");
    check_cuda(cudaMemsetAsync(interpolated_debug_counts_.gpu_data(), 0,
                               interpolated_debug_counts_.gpu_size() * sizeof(int),
                               stream),
               "interpolated debug count memset");
    check_cuda(cudaMemsetAsync(interpolated_normal_sum_.gpu_data(), 0,
                               sizeof(math::Vec3f),
                               stream),
               "interpolated normal sum memset");

    update_interpolated_surface_positions_gpu(x_.gpu_data());

    const float inv_h2 = 1.0f / std::max(h * h, 1.0e-8f);
    const int n_tets = static_cast<int>(tets_.gpu_size());
    if (n_tets > 0) {
        const int grid = (n_tets + block - 1) / block;
        assemble_interpolated_mass_elastic_kernel<<<grid, block, 0, stream>>>(
            H_.diag.gpu_data(),
            H_.values.gpu_data(),
            rhs_.gpu_data(),
            tets_.gpu_data(),
            tet_mass_blocks_dev_.gpu_data(),
            n_tets,
            y_.gpu_data(),
            x_.gpu_data(),
            inv_h2,
            params_.stiffness);
        check_cuda(cudaGetLastError(),
                   "assemble_interpolated_mass_elastic_kernel");
    }

    const int vertex_grid = (num_vertices_ + block - 1) / block;
    assemble_interpolated_fixed_kernel<<<vertex_grid, block, 0, stream>>>(
        H_.diag.gpu_data(),
        rhs_.gpu_data(),
        rest_.gpu_data(),
        fixed_.gpu_data(),
        num_vertices_,
        params_.fixed_weight);
    check_cuda(cudaGetLastError(), "assemble_interpolated_fixed_kernel");

    if (num_surface_vertices_ > 0) {
        const int surface_grid = (num_surface_vertices_ + block - 1) / block;
        append_interpolated_ground_contacts_kernel<<<surface_grid, block, 0, stream>>>(
            rhs_.gpu_data(),
            surface_maps_dev_.gpu_data(),
            interpolated_surface_positions_dev_.gpu_data(),
            num_surface_vertices_,
            params_.ground_y + params_.contact_gap,
            params_.ground_stiffness,
            ground_contacts_.gpu_data(),
            interpolated_wide_contacts_.gpu_data(),
            interpolated_wide_contact_count_.gpu_data(),
            static_cast<int>(interpolated_wide_contacts_.gpu_size()));
        check_cuda(cudaGetLastError(),
                   "append_interpolated_ground_contacts_kernel");
    }

    detect_interpolated_contacts_gpu();
    if (params_.self_collision_stiffness > 0.0f &&
        interpolated_contact_capacity_ > 0 &&
        mesh_mesh_contact_detector_.valid()) {
        const int contact_grid =
            (interpolated_contact_capacity_ + block - 1) / block;
        append_interpolated_self_contacts_kernel<<<contact_grid, block, 0, stream>>>(
            rhs_.gpu_data(),
            mesh_mesh_contact_detector_.contacts().gpu_data(),
            mesh_mesh_contact_detector_.count_array().gpu_data(),
            interpolated_contact_capacity_,
            surface_maps_dev_.gpu_data(),
            params_.contact_gap,
            params_.self_collision_stiffness,
            interpolated_debug_counts_.gpu_data(),
            interpolated_normal_sum_.gpu_data(),
            interpolated_wide_contacts_.gpu_data(),
            interpolated_wide_contact_count_.gpu_data(),
            static_cast<int>(interpolated_wide_contacts_.gpu_size()));
        check_cuda(cudaGetLastError(),
                   "append_interpolated_self_contacts_kernel");
    }

    collision::WideContactSpMVOp wide_op;
    wide_op.contacts = interpolated_wide_contacts_.gpu_data();
    wide_op.count_dev = interpolated_wide_contact_count_.gpu_data();
    wide_op.max_contacts =
        static_cast<int>(interpolated_wide_contacts_.gpu_size());
    wide_op.stiffness = 1.0f;
    collision::bake_wide_contact_diag(H_.diag.gpu_data(), num_vertices_,
                                      wide_op, 1.0f, s);
}

void PabdCudaSolver::step_interpolated(float dt) {
    (void)dt;
    const float frame_dt = params_.dt;
    if (frame_dt <= 0.0f) return;

    const int substeps = std::max(1, params_.substeps);
    const float sub_dt = frame_dt / static_cast<float>(substeps);
    auto* stream = reinterpret_cast<cudaStream_t>(stream_);
    constexpr int block = 128;
    const int vertex_grid = (num_vertices_ + block - 1) / block;
    for (int sub = 0; sub < substeps; ++sub) {
        const float h = sub_dt;
        const float inv_h = 1.0f / std::max(h, 1.0e-8f);

        prepare_interpolated_step_kernel<<<vertex_grid, block, 0, stream>>>(
            rest_.gpu_data(), x_.gpu_data(), old_x_.gpu_data(),
            y_.gpu_data(), velocity_.gpu_data(), fixed_.gpu_data(),
            num_vertices_, h, params_.gravity, params_.damping);
        check_cuda(cudaGetLastError(), "prepare_interpolated_step_kernel");

        solver::PCGParams pcg_params;
        pcg_params.max_iterations = params_.pcg_iterations;
        const int iterations = std::max(1, params_.iterations);
        for (int iter = 0; iter < iterations; ++iter) {
            assemble_interpolated_system_gpu(h);
            collision::WideContactSpMVOp wide_op{
                interpolated_wide_contacts_.gpu_data(),
                interpolated_wide_contact_count_.gpu_data(),
                static_cast<int>(interpolated_wide_contacts_.gpu_size()),
                1.0f};
            last_pcg_iterations_ = pcg_.solve(
                H_, DeviceSpan<math::Vec3f>::from(rhs_),
                DeviceSpan<math::Vec3f>::from(x_),
                pcg_params, stream_handle(stream_),
                collision::ContactSpMVOp{},
                wide_op);
            apply_fixed_kernel<<<vertex_grid, block, 0, stream>>>(
                x_.gpu_data(), rest_.gpu_data(), fixed_.gpu_data(), num_vertices_);
            check_cuda(cudaGetLastError(), "interpolated apply_fixed_kernel");
        }

        update_velocity_kernel<<<vertex_grid, block, 0, stream>>>(
            velocity_.gpu_data(), x_.gpu_data(), old_x_.gpu_data(),
            fixed_.gpu_data(), num_vertices_, inv_h, params_.damping);
        check_cuda(cudaGetLastError(), "interpolated update_velocity_kernel");
    }

    last_residual_ = pcg_.last_residual();
    ground_contacts_.copy_to_host(stream_handle(stream_));
    interpolated_debug_counts_.copy_to_host(stream_handle(stream_));
    interpolated_normal_sum_.copy_to_host(stream_handle(stream_));
    if (mesh_mesh_contact_detector_.valid() && interpolated_contact_capacity_ > 0) {
        mesh_mesh_contact_detector_.count_array().copy_to_host(
            stream_handle(stream_));
    }
    x_.copy_to_host(stream_handle(stream_));
    check_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)),
               "step_interpolated final sync");
    last_ground_contacts_ = ground_contacts_.cpu_data()[0];
    last_self_contacts_ = interpolated_debug_counts_.cpu_data()[0];
    last_self_vertical_contacts_ = interpolated_debug_counts_.cpu_data()[1];
    last_self_horizontal_contacts_ = interpolated_debug_counts_.cpu_data()[2];
    last_self_point_face_contacts_ = interpolated_debug_counts_.cpu_data()[3];
    last_self_edge_edge_contacts_ = interpolated_debug_counts_.cpu_data()[4];
    last_self_normal_sum_ = interpolated_normal_sum_.cpu_data()[0];
    last_self_raw_contacts_ = 0;
    last_self_contact_overflow_ = false;
    if (mesh_mesh_contact_detector_.valid() && interpolated_contact_capacity_ > 0) {
        last_self_raw_contacts_ =
            mesh_mesh_contact_detector_.count_array().cpu_data()[0];
        last_self_contact_overflow_ =
            last_self_raw_contacts_ > interpolated_contact_capacity_;
        if (last_self_contact_overflow_) {
            std::fprintf(stderr,
                         "[PABD_ERROR] self contact buffer overflow: raw=%d capacity=%d. "
                         "Some contacts were dropped before assembly.\n",
                         last_self_raw_contacts_,
                         interpolated_contact_capacity_);
        }
    }
    update_host_positions(true);
}

void PabdCudaSolver::step(float dt) {
    if (!initialized_) {
        throw std::runtime_error("PabdCudaSolver::step: call setup() first");
    }

    if (interpolated_surface_) {
        step_interpolated(dt);
        return;
    }

    const float frame_dt = dt > 0.0f ? dt : params_.dt;
    if (frame_dt <= 0.0f) {
        return;
    }

    const int substeps = std::max(1, params_.substeps);
    const float sub_dt = frame_dt / static_cast<float>(substeps);
    for (int sub = 0; sub < substeps; ++sub) {
        const float h = sub_dt;
        const float inv_h = 1.0f / h;
        const float inv_h2 = inv_h * inv_h;
        const int block = 128;
        auto* stream = reinterpret_cast<cudaStream_t>(stream_);
        const int vertex_grid = (num_vertices_ + block - 1) / block;
        const int num_tets = static_cast<int>(tets_.gpu_size());
        const int tet_grid = (num_tets + block - 1) / block;

        prepare_step_kernel<<<vertex_grid, block, 0, stream>>>(
            rest_.gpu_data(), x_.gpu_data(), old_x_.gpu_data(), y_.gpu_data(),
            velocity_.gpu_data(), fixed_.gpu_data(), num_vertices_, h,
            params_.gravity);
        check_cuda(cudaGetLastError(), "prepare_step_kernel");

        solver::PCGParams pcg_params;
        pcg_params.max_iterations = params_.pcg_iterations;

        for (int iter = 0; iter < std::max(1, params_.iterations); ++iter) {
            rotation_kernel<<<tet_grid, block, 0, stream>>>(
                x_.gpu_data(), tets_.gpu_data(), rotations_.gpu_data(), num_tets);

            const bool self_collision_active =
                params_.self_collision_thickness > 0.0f &&
                params_.self_collision_stiffness > 0.0f &&
                mesh_topology_.valid();
            if (self_collision_active) {
                self_collision_.set_stiffness(params_.self_collision_stiffness);
                self_collision_detector_.detect(
                    DeviceSpan<math::Vec3f>::from(x_),
                    params_.self_collision_thickness,
                    stream_handle(stream_));
            }
            const collision::ContactSpMVOp contact_op = self_collision_active
                ? self_collision_.make_spmv_op(self_collision_detector_)
                : collision::ContactSpMVOp{};

            H_.set_zero(stream_handle(stream_));
            check_cuda(cudaMemsetAsync(ground_contacts_.gpu_data(), 0, sizeof(int), stream),
                       "cudaMemsetAsync(ground_contacts)");
            assemble_base_matrix_kernel<<<vertex_grid, block, 0, stream>>>(
                H_.diag.gpu_data(), mass_.gpu_data(), fixed_.gpu_data(),
                num_vertices_, inv_h2, params_.fixed_weight);
            add_tet_matrix_kernel<<<(num_tets * 16 + block - 1) / block, block, 0, stream>>>(
                H_.diag.gpu_data(), H_.values.gpu_data(), tets_.gpu_data(), num_tets,
                params_.stiffness);
            add_ground_matrix_kernel<<<vertex_grid, block, 0, stream>>>(
                H_.diag.gpu_data(), x_.gpu_data(), fixed_.gpu_data(),
                ground_contacts_.gpu_data(), num_vertices_, params_.ground_y,
                params_.contact_gap, params_.ground_stiffness);
            collision::bake_contact_diag(
                H_.diag.gpu_data(), num_vertices_, contact_op, 1.0f,
                stream_handle(stream_));
            check_cuda(cudaGetLastError(), "matrix assembly kernels");

            assemble_base_rhs_kernel<<<vertex_grid, block, 0, stream>>>(
                rhs_.gpu_data(), rest_.gpu_data(), y_.gpu_data(), mass_.gpu_data(),
                fixed_.gpu_data(), num_vertices_, inv_h2, params_.fixed_weight);
            add_tet_rhs_kernel<<<(num_tets * 4 + block - 1) / block, block, 0, stream>>>(
                rhs_.gpu_data(), tets_.gpu_data(), rotations_.gpu_data(), num_tets,
                params_.stiffness);
            add_ground_rhs_kernel<<<vertex_grid, block, 0, stream>>>(
                rhs_.gpu_data(), x_.gpu_data(), fixed_.gpu_data(), num_vertices_,
                params_.ground_y, params_.contact_gap, params_.ground_stiffness);
            if (self_collision_active) {
                add_self_contact_rhs_kernel<<<
                    (contact_op.max_contacts + block - 1) / block,
                    block, 0, stream>>>(
                    contact_op.pairs, contact_op.weights, contact_op.count_dev,
                    contact_op.max_contacts, contact_op.stiffness,
                    x_.gpu_data(), rhs_.gpu_data());
            }
            check_cuda(cudaGetLastError(), "rhs assembly kernels");

            last_pcg_iterations_ = pcg_.solve(
                H_, DeviceSpan<math::Vec3f>::from(rhs_),
                DeviceSpan<math::Vec3f>::from(x_), pcg_params,
                stream_handle(stream_), contact_op);

            apply_fixed_kernel<<<vertex_grid, block, 0, stream>>>(
                x_.gpu_data(), rest_.gpu_data(), fixed_.gpu_data(), num_vertices_);
            check_cuda(cudaGetLastError(), "apply_fixed_kernel");
        }

        update_velocity_kernel<<<vertex_grid, block, 0, stream>>>(
            velocity_.gpu_data(), x_.gpu_data(), old_x_.gpu_data(), fixed_.gpu_data(),
            num_vertices_, inv_h, params_.damping);
        check_cuda(cudaGetLastError(), "update_velocity_kernel");
    }

    update_host_positions();
    ground_contacts_.copy_to_host(stream_handle(stream_));
    check_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)),
               "ground contact count sync");
    last_ground_contacts_ = ground_contacts_.cpu_data()[0];
    last_self_contacts_ =
        (params_.self_collision_thickness > 0.0f &&
         params_.self_collision_stiffness > 0.0f &&
         mesh_topology_.valid())
            ? self_collision_detector_.count(stream_handle(stream_))
            : 0;
    last_residual_ = pcg_.last_residual();
}

void PabdCudaSolver::update_host_positions(bool x_already_on_host) {
    const std::uintptr_t s = stream_handle(stream_);
    if (!x_already_on_host) {
        x_.copy_to_host(s);
        check_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)),
                   "update_host_positions sync");
    }

    if (interpolated_surface_) {
        std::vector<math::Vec3f> controls(static_cast<std::size_t>(num_vertices_));
        for (int i = 0; i < num_vertices_; ++i) {
            controls[i] = x_[i];
        }
        update_surface_positions(controls, host_surface_positions_);
        min_y_ = host_surface_positions_.empty() ? 0.0f : host_surface_positions_[0].y;
        for (int i = 0; i < num_surface_vertices_; ++i) {
            const math::Vec3f p = host_surface_positions_[i];
            min_y_ = std::min(min_y_, p.y);
            flat_positions_[3 * i + 0] = p.x;
            flat_positions_[3 * i + 1] = p.z;
            flat_positions_[3 * i + 2] = p.y;
        }
        return;
    }

    const math::Vec3f* p = x_.cpu_data();
    min_y_ = p[0].y;
    for (int i = 0; i < num_vertices_; ++i) {
        min_y_ = std::min(min_y_, p[i].y);
        flat_positions_[3 * i + 0] = p[i].x;
        flat_positions_[3 * i + 1] = p[i].z;
        flat_positions_[3 * i + 2] = p[i].y;
    }
}

PabdCudaMesh make_single_tetra_pinned_mesh() {
    PabdCudaMesh mesh;
    mesh.rest_positions = {
        math::Vec3f(0.0f, 2.0f, 0.0f),
        math::Vec3f(0.95f, 1.15f, -0.45f),
        math::Vec3f(0.85f, 1.15f, 0.45f),
        math::Vec3f(0.15f, 1.15f, 0.0f),
    };
    mesh.tets = {{{0, 1, 2, 3}}};
    mesh.fixed.assign(mesh.rest_positions.size(), 0);
    mesh.fixed[0] = 1;
    return mesh;
}

PabdCudaMesh make_hex_drop_mesh() {
    PabdCudaMesh mesh;
    const float h = 0.42f;
    const float cy = 1.4f;
    mesh.rest_positions = {
        math::Vec3f(-h, cy - h, -h),
        math::Vec3f( h, cy - h, -h),
        math::Vec3f( h, cy + h, -h),
        math::Vec3f(-h, cy + h, -h),
        math::Vec3f(-h, cy - h,  h),
        math::Vec3f( h, cy - h,  h),
        math::Vec3f( h, cy + h,  h),
        math::Vec3f(-h, cy + h,  h),
    };
    mesh.tets = {{
        {0, 1, 3, 4},
        {1, 2, 3, 6},
        {1, 3, 4, 6},
        {1, 4, 5, 6},
        {3, 4, 6, 7},
    }};
    mesh.surface_triangles = {{
        {0, 1, 2}, {0, 2, 3},
        {4, 6, 5}, {4, 7, 6},
        {0, 4, 5}, {0, 5, 1},
        {1, 5, 6}, {1, 6, 2},
        {2, 6, 7}, {2, 7, 3},
        {3, 7, 4}, {3, 4, 0},
    }};
    mesh.fixed.assign(mesh.rest_positions.size(), 0);
    return mesh;
}

PabdCudaMesh make_twenty_tets_mesh() {
    PabdCudaMesh mesh;

    // Same local tet as pabd ScenePreset::ThreeTets (usePdGlobal control points).
    const std::array<math::Vec3f, 4> local = {{
        math::Vec3f(0.0f, 0.38f, 0.0f),
        math::Vec3f(0.56f, -0.08f, 0.02f),
        math::Vec3f(-0.12f, -0.12f, 0.58f),
        math::Vec3f(0.08f, -0.56f, -0.08f),
    }};
    const std::array<math::Vec3f, 3> seed_velocities = {{
        math::Vec3f(0.62f, 0.0f, 0.08f),
        math::Vec3f(-0.55f, -0.02f, -0.02f),
        math::Vec3f(0.08f, -0.10f, 0.52f),
    }};

    constexpr int kCols = 5;
    constexpr int kRows = 4;
    constexpr float kSpacingX = 0.78f;
    constexpr float kSpacingZ = 0.72f;
    constexpr float kBaseY = 1.35f;
    constexpr float kRowLift = 0.28f;

    mesh.rest_positions.reserve(kTwentyTetsCount * 4);
    mesh.initial_velocities.reserve(kTwentyTetsCount * 4);
    mesh.tets.reserve(kTwentyTetsCount);
    mesh.surface_triangles.reserve(kTwentyTetsCount * 4);

    for (int body = 0; body < kTwentyTetsCount; ++body) {
        const int row = body / kCols;
        const int col = body % kCols;
        const math::Vec3f center(
            (static_cast<float>(col) - 0.5f * static_cast<float>(kCols - 1)) * kSpacingX,
            kBaseY + static_cast<float>(row) * kRowLift,
            (static_cast<float>(row) - 0.5f * static_cast<float>(kRows - 1)) * kSpacingZ);
        const math::Vec3f velocity = seed_velocities[body % 3];

        const int s = static_cast<int>(mesh.rest_positions.size());
        for (int k = 0; k < 4; ++k) {
            mesh.rest_positions.push_back(center + local[k]);
            mesh.initial_velocities.push_back(velocity);
        }
        mesh.tets.push_back({s + 0, s + 1, s + 2, s + 3});
        mesh.surface_triangles.push_back({s + 0, s + 2, s + 1});
        mesh.surface_triangles.push_back({s + 0, s + 1, s + 3});
        mesh.surface_triangles.push_back({s + 1, s + 2, s + 3});
        mesh.surface_triangles.push_back({s + 2, s + 0, s + 3});
    }
    mesh.fixed.assign(mesh.rest_positions.size(), 0);
    return mesh;
}

namespace {

void append_hex_block(PabdCudaMesh& mesh,
                      math::Vec3f center,
                      float half_extent) {
    const float h = half_extent;
    const int base = static_cast<int>(mesh.rest_positions.size());
    const std::array<math::Vec3f, 8> local = {{
        math::Vec3f(-h, -h, -h),
        math::Vec3f( h, -h, -h),
        math::Vec3f( h,  h, -h),
        math::Vec3f(-h,  h, -h),
        math::Vec3f(-h, -h,  h),
        math::Vec3f( h, -h,  h),
        math::Vec3f( h,  h,  h),
        math::Vec3f(-h,  h,  h),
    }};
    for (const auto& v : local) {
        mesh.rest_positions.push_back(center + v);
    }

    mesh.tets.push_back({base + 0, base + 1, base + 3, base + 4});
    mesh.tets.push_back({base + 1, base + 2, base + 3, base + 6});
    mesh.tets.push_back({base + 1, base + 3, base + 4, base + 6});
    mesh.tets.push_back({base + 1, base + 4, base + 5, base + 6});
    mesh.tets.push_back({base + 3, base + 4, base + 6, base + 7});

    const std::array<std::array<int, 3>, 12> faces = {{
        {base + 0, base + 1, base + 2}, {base + 0, base + 2, base + 3},
        {base + 4, base + 6, base + 5}, {base + 4, base + 7, base + 6},
        {base + 0, base + 4, base + 5}, {base + 0, base + 5, base + 1},
        {base + 1, base + 5, base + 6}, {base + 1, base + 6, base + 2},
        {base + 2, base + 6, base + 7}, {base + 2, base + 7, base + 3},
        {base + 3, base + 7, base + 4}, {base + 3, base + 4, base + 0},
    }};
    for (const auto& face : faces) {
        mesh.surface_triangles.push_back(face);
    }
}

}  // namespace

PabdCudaMesh make_stacked_blocks_mesh(int count,
                                      float half_extent,
                                      float ground_y,
                                      float block_gap) {
    if (count < 1) {
        throw std::invalid_argument("make_stacked_blocks_mesh: count must be >= 1");
    }

    PabdCudaMesh mesh;
    mesh.rest_positions.reserve(static_cast<std::size_t>(count) * 8);
    mesh.tets.reserve(static_cast<std::size_t>(count) * 5);
    mesh.surface_triangles.reserve(static_cast<std::size_t>(count) * kHexBlockSurfaceTris);

    const float spacing = 2.0f * half_extent + block_gap;
    const float base_y = ground_y + half_extent + block_gap;
    for (int layer = 0; layer < count; ++layer) {
        const math::Vec3f center(
            0.0f,
            base_y + spacing * static_cast<float>(layer),
            0.0f);
        append_hex_block(mesh, center, half_extent);
    }

    mesh.fixed.assign(mesh.rest_positions.size(), 0);
    return mesh;
}

PabdCudaMesh make_pd_abd_boxes_mesh(int count,
                                    float ground_y,
                                    float contact_gap,
                                    float half_extent,
                                    float initial_lift) {
    if (count < 1) {
        throw std::invalid_argument("make_pd_abd_boxes_mesh: count must be >= 1");
    }

    PabdCudaMesh mesh;
    mesh.rest_positions.reserve(static_cast<std::size_t>(count) * 4);
    mesh.tets.reserve(static_cast<std::size_t>(count));
    mesh.surface_maps.reserve(static_cast<std::size_t>(count) *
                              kPdAbdBoxSurfaceVertices);
    mesh.surface_triangles.reserve(static_cast<std::size_t>(count) *
                                   kPdAbdBoxSurfaceTris);
    mesh.surface_edges.reserve(static_cast<std::size_t>(count) * 12);
    mesh.tet_volume_overrides.reserve(static_cast<std::size_t>(count));
    mesh.tet_mass_blocks.reserve(static_cast<std::size_t>(count));

    const float h = half_extent;
    const std::string box_obj_path =
#ifdef CHYSX_SOURCE_DIR
        std::string(CHYSX_SOURCE_DIR) + "assets/meshes/pabd/box.obj";
#else
        std::string("assets/meshes/pabd/box.obj");
#endif
    io::ObjMesh obj_box;
    if (!io::load_obj(box_obj_path, obj_box)) {
        throw std::runtime_error("make_pd_abd_boxes_mesh: failed to load " +
                                 box_obj_path);
    }
    if (obj_box.positions.size() % 3 != 0 ||
        obj_box.triangles.size() % 3 != 0 ||
        obj_box.positions.empty() || obj_box.triangles.empty()) {
        throw std::runtime_error("make_pd_abd_boxes_mesh: invalid box OBJ");
    }
    std::vector<math::Vec3f> local_surface;
    local_surface.reserve(obj_box.positions.size() / 3);
    for (std::size_t i = 0; i < obj_box.positions.size(); i += 3) {
        local_surface.push_back(math::Vec3f(obj_box.positions[i + 0] * h,
                                            obj_box.positions[i + 1] * h,
                                            obj_box.positions[i + 2] * h));
    }

    std::vector<std::array<int, 3>> local_tris;
    local_tris.reserve(obj_box.triangles.size() / 3);
    std::map<std::pair<int, int>, int> edge_seen;
    auto add_obj_edge = [&](int a, int b) {
        if (a > b) std::swap(a, b);
        edge_seen[{a, b}] = 1;
    };
    for (std::size_t i = 0; i < obj_box.triangles.size(); i += 3) {
        const int a = obj_box.triangles[i + 0];
        const int b = obj_box.triangles[i + 1];
        const int c = obj_box.triangles[i + 2];
        if (a < 0 || b < 0 || c < 0 ||
            a >= static_cast<int>(local_surface.size()) ||
            b >= static_cast<int>(local_surface.size()) ||
            c >= static_cast<int>(local_surface.size())) {
            throw std::runtime_error("make_pd_abd_boxes_mesh: OBJ index out of range");
        }
        local_tris.push_back({a, b, c});
        add_obj_edge(a, b);
        add_obj_edge(b, c);
        add_obj_edge(c, a);
    }
    std::vector<std::array<int, 2>> local_edg;
    local_edg.reserve(edge_seen.size());
    for (const auto& entry : edge_seen) {
        local_edg.push_back({entry.first.first, entry.first.second});
    }

    const math::Vec3f bbox_min(-h, -h, -h);
    const math::Vec3f half_extents(h, h, h);
    const float r = 2.0f * math::length(half_extents * 2.0f);
    const std::array<math::Vec3f, 4> local_control = {{
        bbox_min,
        math::Vec3f(bbox_min.x + r, bbox_min.y, bbox_min.z),
        math::Vec3f(bbox_min.x, bbox_min.y + r, bbox_min.z),
        math::Vec3f(bbox_min.x, bbox_min.y, bbox_min.z + r),
    }};
    const float volume = 8.0f * h * h * h;
    const float base_y = ground_y + h + contact_gap + initial_lift;
    const float spacing = 2.0f * h + contact_gap;

    for (int body = 0; body < count; ++body) {
        const int dof_start = static_cast<int>(mesh.rest_positions.size());
        const int surface_start = static_cast<int>(mesh.surface_maps.size());
        const math::Vec3f center(
            0.0f, base_y + spacing * static_cast<float>(body), 0.0f);

        for (int k = 0; k < 4; ++k) {
            mesh.rest_positions.push_back(center + local_control[k]);
        }
        mesh.tets.push_back({dof_start + 0, dof_start + 1,
                             dof_start + 2, dof_start + 3});
        mesh.tet_volume_overrides.push_back(volume);

        auto append_surface_map = [&](math::Vec3f local,
                                      int self_collide,
                                      int ground_collide,
                                      math::Vec3f normal = math::Vec3f()) {
            const float w1 = (local.x - bbox_min.x) / r;
            const float w2 = (local.y - bbox_min.y) / r;
            const float w3 = (local.z - bbox_min.z) / r;
            PabdSurfaceMap map;
            map.index = {dof_start + 0, dof_start + 1,
                         dof_start + 2, dof_start + 3};
            map.weight = {1.0f - w1 - w2 - w3, w1, w2, w3};
            map.normal = normal;
            map.body = body;
            map.collide = self_collide || ground_collide;
            map.self_collide = self_collide;
            map.ground_collide = ground_collide;
            mesh.surface_maps.push_back(map);
        };

        for (const math::Vec3f& local : local_surface) {
            append_surface_map(local, 1, 1);
        }

        for (const auto& tri : local_tris) {
            mesh.surface_triangles.push_back({
                surface_start + tri[0],
                surface_start + tri[1],
                surface_start + tri[2],
            });
        }
        for (const auto& edge : local_edg) {
            mesh.surface_edges.push_back({
                surface_start + edge[0],
                surface_start + edge[1],
            });
        }

        std::array<float, 16> mass_block{};
        mass_block.fill(0.0f);
        const float inv_sqrt3 = 0.5773502691896258f;
        const float sample_mass_unit_density = volume / 8.0f;
        for (float sx : {-1.0f, 1.0f}) {
            for (float sy : {-1.0f, 1.0f}) {
                for (float sz : {-1.0f, 1.0f}) {
                    const math::Vec3f sample(sx * h * inv_sqrt3,
                                             sy * h * inv_sqrt3,
                                             sz * h * inv_sqrt3);
                    const float w1 = (sample.x - bbox_min.x) / r;
                    const float w2 = (sample.y - bbox_min.y) / r;
                    const float w3 = (sample.z - bbox_min.z) / r;
                    const std::array<float, 4> w =
                        {1.0f - w1 - w2 - w3, w1, w2, w3};
                    for (int row = 0; row < 4; ++row) {
                        for (int col = 0; col < 4; ++col) {
                            mass_block[row * 4 + col] +=
                                sample_mass_unit_density * w[row] * w[col];
                        }
                    }
                }
            }
        }
        mesh.tet_mass_blocks.push_back(mass_block);
    }

    mesh.fixed.assign(mesh.rest_positions.size(), 0);
    return mesh;
}

PabdCudaMesh make_pd_abd_tetra_edge_edge_mesh(float lower_y,
                                              float upper_y,
                                              float scale) {
    const std::string tetra_obj_path =
#ifdef CHYSX_SOURCE_DIR
        std::string(CHYSX_SOURCE_DIR) + "assets/meshes/pabd/tetra_edge.obj";
#else
        std::string("assets/meshes/pabd/tetra_edge.obj");
#endif
    io::ObjMesh obj_tetra;
    if (!io::load_obj(tetra_obj_path, obj_tetra)) {
        throw std::runtime_error("make_pd_abd_tetra_edge_edge_mesh: failed to load " +
                                 tetra_obj_path);
    }
    if (obj_tetra.positions.size() % 3 != 0 ||
        obj_tetra.triangles.size() % 3 != 0 ||
        obj_tetra.positions.empty() || obj_tetra.triangles.empty()) {
        throw std::runtime_error("make_pd_abd_tetra_edge_edge_mesh: invalid tetra OBJ");
    }

    std::vector<math::Vec3f> local_surface;
    local_surface.reserve(obj_tetra.positions.size() / 3);
    math::Vec3f bbox_min(3.402823466e38f);
    math::Vec3f bbox_max(-3.402823466e38f);
    for (std::size_t i = 0; i < obj_tetra.positions.size(); i += 3) {
        const math::Vec3f p(obj_tetra.positions[i + 0] * scale,
                            obj_tetra.positions[i + 1] * scale,
                            obj_tetra.positions[i + 2] * scale);
        local_surface.push_back(p);
        bbox_min = math::min(bbox_min, p);
        bbox_max = math::max(bbox_max, p);
    }

    std::vector<std::array<int, 3>> local_tris;
    local_tris.reserve(obj_tetra.triangles.size() / 3);
    std::map<std::pair<int, int>, int> edge_seen;
    auto add_obj_edge = [&](int a, int b) {
        if (a > b) std::swap(a, b);
        edge_seen[{a, b}] = 1;
    };
    for (std::size_t i = 0; i < obj_tetra.triangles.size(); i += 3) {
        const int a = obj_tetra.triangles[i + 0];
        const int b = obj_tetra.triangles[i + 1];
        const int c = obj_tetra.triangles[i + 2];
        if (a < 0 || b < 0 || c < 0 ||
            a >= static_cast<int>(local_surface.size()) ||
            b >= static_cast<int>(local_surface.size()) ||
            c >= static_cast<int>(local_surface.size())) {
            throw std::runtime_error("make_pd_abd_tetra_edge_edge_mesh: OBJ index out of range");
        }
        local_tris.push_back({a, b, c});
        add_obj_edge(a, b);
        add_obj_edge(b, c);
        add_obj_edge(c, a);
    }
    std::vector<std::array<int, 2>> local_edges;
    local_edges.reserve(edge_seen.size());
    for (const auto& entry : edge_seen) {
        local_edges.push_back({entry.first.first, entry.first.second});
    }

    const float volume = std::max(1.0e-6f,
        std::abs(signed_tet_volume(local_surface[0], local_surface[1],
                                   local_surface[2], local_surface[3])));
    const math::Vec3f extents = bbox_max - bbox_min;
    const float r = std::max(1.0e-6f, 2.0f * math::length(extents));
    const std::array<math::Vec3f, 4> local_control = {{
        bbox_min,
        math::Vec3f(bbox_min.x + r, bbox_min.y, bbox_min.z),
        math::Vec3f(bbox_min.x, bbox_min.y + r, bbox_min.z),
        math::Vec3f(bbox_min.x, bbox_min.y, bbox_min.z + r),
    }};

    auto rotate_y_90 = [](math::Vec3f p) {
        return math::Vec3f(p.z, p.y, -p.x);
    };

    PabdCudaMesh mesh;
    mesh.rest_positions.reserve(8);
    mesh.tets.reserve(2);
    mesh.surface_maps.reserve(local_surface.size() * 2);
    mesh.surface_triangles.reserve(local_tris.size() * 2);
    mesh.surface_edges.reserve(local_edges.size() * 2);
    mesh.tet_volume_overrides.reserve(2);
    mesh.tet_mass_blocks.reserve(2);

    auto append_body = [&](int body, math::Vec3f translation, bool rotate, bool fixed) {
        const int dof_start = static_cast<int>(mesh.rest_positions.size());
        const int surface_start = static_cast<int>(mesh.surface_maps.size());
        for (int k = 0; k < 4; ++k) {
            const math::Vec3f local = rotate ? rotate_y_90(local_control[k])
                                             : local_control[k];
            mesh.rest_positions.push_back(translation + local);
        }
        mesh.tets.push_back({dof_start + 0, dof_start + 1,
                             dof_start + 2, dof_start + 3});
        mesh.tet_volume_overrides.push_back(volume);

        for (const math::Vec3f& p0 : local_surface) {
            const float w1 = (p0.x - bbox_min.x) / r;
            const float w2 = (p0.y - bbox_min.y) / r;
            const float w3 = (p0.z - bbox_min.z) / r;
            PabdSurfaceMap map;
            map.index = {dof_start + 0, dof_start + 1,
                         dof_start + 2, dof_start + 3};
            map.weight = {1.0f - w1 - w2 - w3, w1, w2, w3};
            map.body = body;
            map.collide = 1;
            map.self_collide = 1;
            map.ground_collide = 0;
            mesh.surface_maps.push_back(map);
        }
        for (const auto& tri : local_tris) {
            mesh.surface_triangles.push_back({
                surface_start + tri[0],
                surface_start + tri[1],
                surface_start + tri[2],
            });
        }
        for (const auto& edge : local_edges) {
            mesh.surface_edges.push_back({
                surface_start + edge[0],
                surface_start + edge[1],
            });
        }

        std::array<float, 16> mass_block{};
        mass_block.fill(0.0f);
        const float lumped = volume / 4.0f;
        for (int i = 0; i < 4; ++i) {
            mass_block[i * 4 + i] = lumped;
        }
        mesh.tet_mass_blocks.push_back(mass_block);

        for (int k = 0; k < 4; ++k) {
            mesh.fixed.push_back(fixed ? 1 : 0);
        }
    };

    append_body(0, math::Vec3f(0.0f, lower_y, 0.0f), false, true);
    append_body(1, math::Vec3f(0.035f, upper_y, 0.0f), true, false);
    return mesh;
}

}  // namespace pabd_cuda
}  // namespace rigid
}  // namespace chysx
