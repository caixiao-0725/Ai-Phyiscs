// SPDX-License-Identifier: Apache-2.0

#include "pabd_cuda_solver.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "../../collision/contact_spmv.h"

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

        for (int k = 0; k < 4; ++k) {
            mass_[tv[k]] += volume * params_.density;
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
    if (!tri_vec.empty()) {
        mesh_topology_.build(tri_vec, num_vertices_);
        self_collision_detector_.bind_topology(&mesh_topology_);
        const int max_contacts = std::max(1, params_.self_collision_max_contacts);
        const int max_ef_candidates =
            std::max(max_contacts, static_cast<int>(surface_triangles_.size()) * 16);
        self_collision_detector_.reserve(max_contacts, max_ef_candidates);
    } else {
        self_collision_detector_.bind_topology(nullptr);
    }
    self_collision_.set_stiffness(params_.self_collision_stiffness);

    flat_positions_.assign(static_cast<std::size_t>(num_vertices_) * 3, 0.0f);
    flat_triangles_.resize(surface_triangles_.size() * 3);
    for (std::size_t i = 0; i < surface_triangles_.size(); ++i) {
        flat_triangles_[3 * i + 0] = surface_triangles_[i][0];
        flat_triangles_[3 * i + 1] = surface_triangles_[i][1];
        flat_triangles_[3 * i + 2] = surface_triangles_[i][2];
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
    check_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)),
               "upload_static_data sync");
}

void PabdCudaSolver::step(float dt) {
    if (!initialized_) {
        throw std::runtime_error("PabdCudaSolver::step: call setup() first");
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

void PabdCudaSolver::update_host_positions() {
    const std::uintptr_t s = stream_handle(stream_);
    x_.copy_to_host(s);
    check_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)),
               "update_host_positions sync");

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

}  // namespace pabd_cuda
}  // namespace rigid
}  // namespace chysx
