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
#include "../../collision/friction.cuh"
#include "../../io/obj_io.h"
#include "pabd_contact_measure.cuh"

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

constexpr int kCountDrivenBlocksPerSm = 4;
constexpr int kBodyContactEllMinWidth = 64;
constexpr int kBodyContactEllCapacityScale = 8;

inline int count_driven_max_blocks() {
    static const int max_blocks = [] {
        int device = 0;
        int sm_count = 0;
        check_cuda(cudaGetDevice(&device), "query current CUDA device");
        check_cuda(cudaDeviceGetAttribute(&sm_count,
                                          cudaDevAttrMultiProcessorCount,
                                          device),
                   "query CUDA SM count");
        return std::max(1, sm_count * kCountDrivenBlocksPerSm);
    }();
    return max_blocks;
}

inline int bounded_count_grid(int capacity, int block_size) {
    // The device-side count controls work; capacity only bounds the loop.
    const int launch_size = std::max(1, capacity);
    return std::min(count_driven_max_blocks(),
                    (launch_size + block_size - 1) / block_size);
}

inline int choose_body_contact_ell_width(int contact_capacity,
                                         int body_count) {
    if (contact_capacity <= 0 || body_count <= 0) return 0;

    // A contact contributes at most two body references. Reserve four times
    // that capacity-derived average degree, then round for regular addressing.
    const std::uint64_t scaled =
        static_cast<std::uint64_t>(contact_capacity) *
        kBodyContactEllCapacityScale;
    std::uint64_t desired =
        (scaled + static_cast<std::uint64_t>(body_count) - 1) /
        static_cast<std::uint64_t>(body_count);
    desired = std::max<std::uint64_t>(desired, kBodyContactEllMinWidth);
    desired = std::min<std::uint64_t>(
        desired, static_cast<std::uint64_t>(contact_capacity));

    std::uint64_t rounded = 1;
    while (rounded < desired) rounded <<= 1;
    return static_cast<int>(std::min<std::uint64_t>(
        rounded, static_cast<std::uint64_t>(contact_capacity)));
}

inline bool solve_spd4(const std::array<float, 16>& matrix,
                       const double rhs[4],
                       double solution[4]) {
    double lower[4][4]{};
    double max_diagonal = 0.0;
    for (int i = 0; i < 4; ++i) {
        max_diagonal = std::max(
            max_diagonal, std::abs(static_cast<double>(matrix[i * 4 + i])));
    }
    if (!(max_diagonal > 0.0) || !std::isfinite(max_diagonal)) return false;
    const double pivot_tolerance =
        std::max(1.0e-30, max_diagonal * 1.0e-12);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.5 *
                (static_cast<double>(matrix[i * 4 + j]) +
                 static_cast<double>(matrix[j * 4 + i]));
            for (int k = 0; k < j; ++k) {
                sum -= lower[i][k] * lower[j][k];
            }
            if (i == j) {
                if (!(sum > pivot_tolerance) || !std::isfinite(sum)) {
                    return false;
                }
                lower[i][j] = std::sqrt(sum);
            } else {
                lower[i][j] = sum / lower[j][j];
            }
        }
    }

    double intermediate[4]{};
    for (int i = 0; i < 4; ++i) {
        double sum = rhs[i];
        for (int j = 0; j < i; ++j) sum -= lower[i][j] * intermediate[j];
        intermediate[i] = sum / lower[i][i];
    }
    for (int i = 3; i >= 0; --i) {
        double sum = intermediate[i];
        for (int j = i + 1; j < 4; ++j) sum -= lower[j][i] * solution[j];
        solution[i] = sum / lower[i][i];
    }
    return true;
}

inline bool contact_inverse_mass_block(
    const std::array<float, 16>& mass,
    const std::array<unsigned char, 4>& fixed,
    std::array<float, 16>* inverse) {
    inverse->fill(0.0f);
    std::array<float, 16> free_mass = mass;
    bool has_free_dof = false;
    for (int i = 0; i < 4; ++i) {
        if (!fixed[i]) {
            has_free_dof = true;
            continue;
        }
        for (int j = 0; j < 4; ++j) {
            free_mass[i * 4 + j] = 0.0f;
            free_mass[j * 4 + i] = 0.0f;
        }
        free_mass[i * 4 + i] = 1.0f;
    }
    if (!has_free_dof) return true;

    for (int col = 0; col < 4; ++col) {
        if (fixed[col]) continue;
        double rhs[4]{};
        rhs[col] = 1.0;
        double solution[4]{};
        if (!solve_spd4(free_mass, rhs, solution)) return false;
        for (int row = 0; row < 4; ++row) {
            if (!fixed[row]) {
                (*inverse)[row * 4 + col] =
                    static_cast<float>(solution[row]);
            }
        }
    }
    return true;
}

inline bool self_contact_enabled(const PabdCudaParams& params) {
    return params.use_contact_beta
        ? params.self_collision_beta > 0.0f
        : params.self_collision_stiffness > 0.0f;
}

inline bool endpoint_hinge_effective_mass(
    const std::array<float, 16>& mass,
    const math::Vec4f& weights0,
    const math::Vec4f& weights1,
    math::Vec4f* packed_mass) {
    double rhs0[4]{};
    double rhs1[4]{};
    for (int i = 0; i < 4; ++i) {
        rhs0[i] = static_cast<double>(weights0[i]);
        rhs1[i] = static_cast<double>(weights1[i]);
    }
    double inverse_weights0[4]{};
    double inverse_weights1[4]{};
    if (!solve_spd4(mass, rhs0, inverse_weights0) ||
        !solve_spd4(mass, rhs1, inverse_weights1)) {
        return false;
    }

    double g00 = 0.0;
    double g01 = 0.0;
    double g11 = 0.0;
    for (int i = 0; i < 4; ++i) {
        g00 += rhs0[i] * inverse_weights0[i];
        g01 += rhs0[i] * inverse_weights1[i];
        g11 += rhs1[i] * inverse_weights1[i];
    }
    const double gram_scale = std::max(std::abs(g00), std::abs(g11));
    const double determinant = g00 * g11 - g01 * g01;
    const double determinant_tolerance =
        std::max(1.0e-30, gram_scale * gram_scale * 1.0e-12);
    if (!(determinant > determinant_tolerance) ||
        !std::isfinite(determinant)) {
        return false;
    }

    const double m00 = g11 / determinant;
    const double m01 = -g01 / determinant;
    const double m11 = g00 / determinant;
    if (!(m00 > 0.0) || !(m11 > 0.0) ||
        !std::isfinite(m00) || !std::isfinite(m01) ||
        !std::isfinite(m11)) {
        return false;
    }
    *packed_mass = math::Vec4f(
        static_cast<float>(m00), static_cast<float>(m01),
        static_cast<float>(m11), 0.0f);
    return true;
}

inline bool arap_effective_inertia(
    const std::array<float, 16>& mass,
    const PabdCudaSolver::TetData& tet,
    float* effective_inertia) {
    double generalized_trace = 0.0;
    for (int col = 0; col < 4; ++col) {
        double rhs[4]{};
        for (int row = 0; row < 4; ++row) {
            rhs[row] = static_cast<double>(
                math::dot(tet.grad[row], tet.grad[col]));
        }
        double solution[4]{};
        if (!solve_spd4(mass, rhs, solution)) return false;
        generalized_trace += solution[col];
    }

    // G has rank three for a non-degenerate tet. This scalar makes the mean
    // positive generalized eigenvalue of (I_eff * G, M) equal to one.
    if (!(generalized_trace > 1.0e-30) ||
        !std::isfinite(generalized_trace)) {
        return false;
    }
    const double inertia = 3.0 / generalized_trace;
    if (!(inertia > 0.0) || !std::isfinite(inertia)) return false;
    *effective_inertia = static_cast<float>(inertia);
    return std::isfinite(*effective_inertia) && *effective_inertia > 0.0f;
}

CHYSX_HDI math::Mat3f make_columns(math::Vec3f c0, math::Vec3f c1, math::Vec3f c2) {
    return math::Mat3f(c0.x, c1.x, c2.x,
                       c0.y, c1.y, c2.y,
                       c0.z, c1.z, c2.z);
}

CHYSX_HDI float tet_arap_weight(
    const PabdCudaSolver::TetData& tet,
    int body,
    const float* effective_inertias,
    float stiffness,
    float arap_beta_inv_h2) {
    return effective_inertias != nullptr
        ? arap_beta_inv_h2 * effective_inertias[body]
        : stiffness * tet.volume;
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

__device__ void atomic_add_mat3(math::Mat3f* dst, const math::Mat3f& value) {
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        atomicAdd(&dst->data[i], value.data[i]);
    }
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

__device__ void add_mat3_by_slot_device(math::Mat3f* diag,
                                        math::Mat3f* values,
                                        int slot,
                                        const math::Mat3f& value) {
    if (slot < 0) {
        atomic_add_mat3(&diag[-slot - 1], value);
    } else {
        atomic_add_mat3(&values[slot], value);
    }
}

constexpr int kBlockJacobiDofs = 12;
constexpr int kBlockJacobiBaseKSize =
    solver::kBodyBlock12PackedLowerSize;
constexpr int kBodyBlockPackedLowerSize =
    solver::kBodyBlock12PackedLowerSize;
static_assert(kBlockJacobiDofs == solver::kBodyBlock12Dofs,
              "PABD and PCG body block sizes must match");

CHYSX_HDI int sym12_index(int r, int c) {
    if (r < c) {
        const int t = r;
        r = c;
        c = t;
    }
    return r * (r + 1) / 2 + c;
}

CHYSX_HDI math::Mat3f elastic_control_block(
    const math::Vec3f& row_gradient,
    const math::Vec3f& column_gradient,
    const math::Mat3f& rotation,
    const math::Mat3f& stretch,
    PabdElasticCurvatureMode mode,
    float scale) {
    math::Mat3f block = math::Mat3f::zero();
    #pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        math::Vec3f basis(0.0f, 0.0f, 0.0f);
        basis[axis] = 1.0f;
        const math::Mat3f delta_f = math::outer(basis, column_gradient);
        const math::Mat3f delta_p = apply_world_elastic_curvature(
            delta_f, rotation, stretch, mode) * scale;
        const math::Vec3f column = delta_p * row_gradient;
        block(0, axis) = column.x;
        block(1, axis) = column.y;
        block(2, axis) = column.z;
    }
    return block;
}

__device__ __forceinline__ math::Vec3f hinge_vec3(math::Vec4f v) {
    return math::Vec3f(v.x, v.y, v.z);
}

__device__ __forceinline__ float endpoint_hinge_weight_block(
    const PabdEndpointHinge& hinge,
    int row,
    int col) {
    const float m00 = hinge.effective_mass.x;
    const float m01 = hinge.effective_mass.y;
    const float m11 = hinge.effective_mass.z;
    const float metric_w0 =
        m00 * hinge.weights0[col] + m01 * hinge.weights1[col];
    const float metric_w1 =
        m01 * hinge.weights0[col] + m11 * hinge.weights1[col];
    return hinge.weights0[row] * metric_w0 +
           hinge.weights1[row] * metric_w1;
}

__device__ __forceinline__ math::Vec3f safe_hinge_axis(math::Vec4f axis4) {
    math::Vec3f axis = hinge_vec3(axis4);
    const float len2 = math::dot(axis, axis);
    if (len2 <= 1.0e-12f) return math::Vec3f(0.0f, 1.0f, 0.0f);
    return axis * rsqrtf(len2);
}

__device__ float endpoint_hinge_axis_omega(
    const math::Vec3f* velocity,
    const math::Vec3f* x,
    const PabdCudaSolver::TetData& tet,
    const float* mass,
    const PabdEndpointHinge& hinge,
    math::Vec3f* tangent_out = nullptr,
    float* inertia_out = nullptr) {
    const math::Vec3f axis = safe_hinge_axis(hinge.axis);
    const math::Vec3f center =
        (hinge_vec3(hinge.endpoint0) + hinge_vec3(hinge.endpoint1)) * 0.5f;

    float control_mass[4];
    float total_mass = 0.0f;
    math::Vec3f average_velocity(0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        float m = 0.0f;
        #pragma unroll
        for (int col = 0; col < 4; ++col) {
            m += mass[a * 4 + col];
        }
        m = fmaxf(m, 1.0e-8f);
        control_mass[a] = m;
        total_mass += m;
        average_velocity += velocity[tet.v[a]] * m;
    }
    if (total_mass > 1.0e-8f) average_velocity /= total_mass;

    float inertia = 0.0f;
    float angular_momentum = 0.0f;
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const math::Vec3f tangent =
            math::cross(axis, x[tet.v[a]] - center);
        if (tangent_out != nullptr) tangent_out[a] = tangent;
        inertia += control_mass[a] * math::dot(tangent, tangent);
        angular_momentum += control_mass[a] *
            math::dot(tangent, velocity[tet.v[a]] - average_velocity);
    }
    if (inertia_out != nullptr) *inertia_out = inertia;
    return inertia > 1.0e-8f ? angular_momentum / inertia : 0.0f;
}

__global__ void apply_endpoint_hinge_motor_kernel(
    math::Vec3f* velocity,
    const math::Vec3f* x,
    const PabdCudaSolver::TetData* tets,
    const float* mass_blocks,
    const PabdEndpointHinge* hinges,
    int n_tets,
    float torque,
    float damping,
    float h) {
    const int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= n_tets) return;
    const PabdEndpointHinge hinge = hinges[body];
    if (hinge.body != body || hinge.motor == 0) return;

    const PabdCudaSolver::TetData tet = tets[body];
    const float* mass = mass_blocks + body * 16;
    math::Vec3f tangent[4];
    float inertia = 0.0f;
    const float omega = endpoint_hinge_axis_omega(
        velocity, x, tet, mass, hinge, tangent, &inertia);
    if (inertia <= 1.0e-8f) return;

    const float delta_omega =
        h * (torque - damping * omega) / inertia;
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        velocity[tet.v[a]] += tangent[a] * delta_omega;
    }
}

__global__ void compute_endpoint_hinge_diagnostics_kernel(
    const math::Vec3f* velocity,
    const math::Vec3f* x,
    const PabdCudaSolver::TetData* tets,
    const float* mass_blocks,
    const PabdEndpointHinge* hinges,
    int n_tets,
    math::Vec3f* diagnostics) {
    const int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= n_tets) return;
    const PabdEndpointHinge hinge = hinges[body];
    if (hinge.body != body) {
        diagnostics[body] = math::Vec3f(0.0f, 0.0f, 0.0f);
        return;
    }

    const PabdCudaSolver::TetData tet = tets[body];
    math::Vec3f y0(0.0f, 0.0f, 0.0f);
    math::Vec3f y1(0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        y0 += x[tet.v[a]] * hinge.weights0[a];
        y1 += x[tet.v[a]] * hinge.weights1[a];
    }
    const float error0 = math::length(y0 - hinge_vec3(hinge.endpoint0));
    const float error1 = math::length(y1 - hinge_vec3(hinge.endpoint1));
    const float omega = endpoint_hinge_axis_omega(
        velocity, x, tet, mass_blocks + body * 16, hinge);
    diagnostics[body] = math::Vec3f(omega, fmaxf(error0, error1),
                                    0.5f * (error0 + error1));
}

__device__ void atomic_max_nonnegative(float* destination, float value) {
    atomicMax(reinterpret_cast<unsigned int*>(destination),
              __float_as_uint(fmaxf(value, 0.0f)));
}

__global__ void compute_elastic_curvature_diagnostics_kernel(
    const math::Vec3f* x,
    const PabdCudaSolver::TetData* tets,
    int n_tets,
    float* diagnostics) {
    const int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= n_tets) return;

    const PabdCudaSolver::TetData tet = tets[body];
    const math::Mat3f ds = make_columns(
        x[tet.v[1]] - x[tet.v[0]],
        x[tet.v[2]] - x[tet.v[0]],
        x[tet.v[3]] - x[tet.v[0]]);
    const math::Mat3f f = ds * tet.inv_dm;
    const math::Mat3f rotation = polar_rotation(f);
    const math::Mat3f stretch = math::transpose(rotation) * f;
    const math::Mat3f stretch_delta = stretch - math::Mat3f::identity();
    const math::Mat3f orthogonality =
        math::transpose(f) * f - math::Mat3f::identity();
    const math::Mat3f rotational = curvature_rotational_block(stretch);

    float stretch_norm2 = 0.0f;
    float orthogonality_norm2 = 0.0f;
    float rotational_norm2 = 0.0f;
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        stretch_norm2 += stretch_delta.data[i] * stretch_delta.data[i];
        orthogonality_norm2 +=
            orthogonality.data[i] * orthogonality.data[i];
        rotational_norm2 += rotational.data[i] * rotational.data[i];
    }

    atomic_max_nonnegative(&diagnostics[0], sqrtf(stretch_norm2));
    atomic_max_nonnegative(&diagnostics[1], sqrtf(orthogonality_norm2));
    atomic_max_nonnegative(
        &diagnostics[2], fabsf(math::determinant(f) - 1.0f));
    atomic_max_nonnegative(&diagnostics[3], sqrtf(rotational_norm2));
}

__global__ void update_matrix_free_polar_gn_state_kernel(
    const math::Vec3f* __restrict__ x,
    const PabdCudaSolver::TetData* __restrict__ tets,
    const float* __restrict__ arap_effective_inertias,
    int n_tets,
    float stiffness,
    float arap_beta_inv_h2,
    math::Mat3f* __restrict__ rotations,
    math::Mat3f* __restrict__ rotational_curvatures,
    float* __restrict__ scales) {
    const int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= n_tets) return;

    const PabdCudaSolver::TetData tet = tets[body];
    const math::Mat3f ds = make_columns(
        x[tet.v[1]] - x[tet.v[0]],
        x[tet.v[2]] - x[tet.v[0]],
        x[tet.v[3]] - x[tet.v[0]]);
    const math::Mat3f f = ds * tet.inv_dm;
    const math::Mat3f rotation = polar_rotation(f);
    const math::Mat3f stretch = math::transpose(rotation) * f;
    const math::Mat3f rotational = curvature_rotational_block(stretch);
    rotations[body] = rotation;
    rotational_curvatures[body] = rotational * rotational;
    scales[body] = tet_arap_weight(
        tet, body, arap_effective_inertias, stiffness,
        arap_beta_inv_h2);
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
    math::Vec3f* surface_positions,
    float* min_y) {
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
    if (min_y != nullptr) {
        int* min_bits = reinterpret_cast<int*>(min_y);
        int old = *min_bits;
        while (p.y < __int_as_float(old)) {
            const int assumed = old;
            old = atomicCAS(min_bits, assumed, __float_as_int(p.y));
            if (old == assumed) break;
        }
    }
}

__global__ void rebuild_block_jacobi_base_k_kernel(
    const PabdCudaSolver::TetData* __restrict__ tets,
    const float* __restrict__ mass_blocks,
    const float* __restrict__ arap_effective_inertias,
    const PabdEndpointHinge* __restrict__ hinges,
    const unsigned char* __restrict__ fixed,
    const math::Vec3f* __restrict__ x,
    int n_tets,
    float inv_h2,
    float stiffness,
    float arap_beta_inv_h2,
    float fixed_weight,
    float hinge_beta_inv_h2,
    PabdElasticCurvatureMode curvature_mode,
    bool local_rest_frame,
    float* __restrict__ base_k) {
    const int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= n_tets) return;

    const PabdCudaSolver::TetData tet = tets[body];
    const float* mass = mass_blocks + body * 16;
    float* K = base_k + body * kBlockJacobiBaseKSize;
    const float w_elastic = tet_arap_weight(
        tet, body, arap_effective_inertias, stiffness,
        arap_beta_inv_h2);
    math::Mat3f rotation = math::Mat3f::identity();
    math::Mat3f stretch = math::Mat3f::identity();
    if (!local_rest_frame) {
        const math::Mat3f ds = make_columns(
            x[tet.v[1]] - x[tet.v[0]],
            x[tet.v[2]] - x[tet.v[0]],
            x[tet.v[3]] - x[tet.v[0]]);
        const math::Mat3f f = ds * tet.inv_dm;
        rotation = polar_rotation(f);
        stretch = math::transpose(rotation) * f;
    }
    const PabdElasticCurvatureMode effective_curvature = local_rest_frame
        ? PabdElasticCurvatureMode::CorotatedRest
        : curvature_mode;
    PabdEndpointHinge hinge{};
    const bool has_hinge = hinges != nullptr && hinge_beta_inv_h2 > 0.0f &&
        ((hinge = hinges[body]).body == body);

    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        #pragma unroll
        for (int b = 0; b <= a; ++b) {
            const float mass_value =
                __fmul_rn(mass[a * 4 + b], inv_h2);
            const math::Mat3f elastic = elastic_control_block(
                tet.grad[a], tet.grad[b], rotation, stretch,
                effective_curvature, w_elastic);
            const float hinge_value = has_hinge
                ? hinge_beta_inv_h2 *
                    endpoint_hinge_weight_block(hinge, a, b)
                : 0.0f;
            #pragma unroll
            for (int row_axis = 0; row_axis < 3; ++row_axis) {
                #pragma unroll
                for (int col_axis = 0; col_axis < 3; ++col_axis) {
                    const int row = 3 * a + row_axis;
                    const int col = 3 * b + col_axis;
                    if (row < col) continue;
                    float value = elastic(row_axis, col_axis);
                    if (row_axis == col_axis) {
                        value = __fadd_rn(value, mass_value + hinge_value);
                        if (a == b && fixed[tet.v[a]]) {
                            value = __fadd_rn(value, fixed_weight);
                        }
                    }
                    K[sym12_index(row, col)] = value;
                }
            }
        }
    }
}

__device__ void append_body_contact_ell_ref(
    int body,
    std::uint32_t ref,
    int n_bodies,
    int ell_width,
    int* __restrict__ ell_counts,
    std::uint32_t* __restrict__ ell_refs,
    int* __restrict__ overflow_count) {
    if (body < 0 || body >= n_bodies) return;
    const int slot = atomicAdd(&ell_counts[body], 1);
    if (slot < ell_width) {
        // Slot-major ELL: neighboring body workers read neighboring addresses.
        ell_refs[static_cast<std::size_t>(slot) * n_bodies + body] = ref;
    } else {
        atomicAdd(overflow_count, 1);
    }
}

__global__ void build_body_contact_ell_kernel(
    const collision::WideContact* __restrict__ contacts,
    const int* __restrict__ count_ptr,
    int max_contacts,
    int n_bodies,
    int ell_width,
    int* __restrict__ ell_counts,
    std::uint32_t* __restrict__ ell_refs,
    int* __restrict__ overflow_count) {
    const int n_raw = *count_ptr;
    const int n = n_raw < max_contacts ? n_raw : max_contacts;
    const int stride = blockDim.x * gridDim.x;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x;
         c < n; c += stride) {
        const collision::WideContact wc = contacts[c];
        if (wc.stiffness == 0.0f) continue;

        const int body0 = wc.base0 >= 0 ? wc.base0 / 4 : -1;
        const int body1 = wc.base1 >= 0 ? wc.base1 / 4 : -1;
        const std::uint32_t contact_ref =
            static_cast<std::uint32_t>(c) << 1;
        if (body0 >= 0) {
            append_body_contact_ell_ref(body0, contact_ref, n_bodies,
                                        ell_width, ell_counts, ell_refs,
                                        overflow_count);
        }
        if (body1 >= 0 && body1 != body0) {
            append_body_contact_ell_ref(body1, contact_ref | 1u, n_bodies,
                                        ell_width, ell_counts, ell_refs,
                                        overflow_count);
        }
    }
}

template <bool BuildRhs>
__device__ __forceinline__ void gather_body_contact_ell_to_local_system(
    int body,
    const collision::WideContact* __restrict__ contacts,
    const int* __restrict__ ell_counts,
    const std::uint32_t* __restrict__ ell_refs,
    int ell_width,
    int n_bodies,
    int n_controls,
    const math::Vec3f* __restrict__ x_lagged,
    const math::Mat3f* __restrict__ body_rotations,
    float A[kBlockJacobiDofs][kBlockJacobiDofs],
    float* b) {
    const int stored_count =
        ell_counts[body] < ell_width ? ell_counts[body] : ell_width;
    for (int slot = 0; slot < stored_count; ++slot) {
        const std::uint32_t ref =
            ell_refs[static_cast<std::size_t>(slot) * n_bodies + body];
        const int contact_id = static_cast<int>(ref >> 1);
        const int side = static_cast<int>(ref & 1u);
        const collision::WideContact wc = contacts[contact_id];

        int own_base = side == 0 ? wc.base0 : wc.base1;
        int other_base = side == 0 ? wc.base1 : wc.base0;
        math::Vec4f own_w = side == 0 ? wc.weights0 : wc.weights1;
        const math::Vec4f other_w = side == 0 ? wc.weights1 : wc.weights0;
        if (own_base < 0 || own_base / 4 != body ||
            own_base + 3 >= n_controls) {
            continue;
        }

        // Same-body contacts are represented by one ELL reference. Fold both
        // coefficient sets together so their cross terms stay in this block.
        if (other_base >= 0 && other_base / 4 == body) {
            #pragma unroll
            for (int a = 0; a < 4; ++a) own_w[a] += other_w[a];
            other_base = -1;
        }

        math::Vec3f normal_vector(wc.normal_target.x,
                                  wc.normal_target.y,
                                  wc.normal_target.z);
        if (body_rotations != nullptr) {
            normal_vector = math::transpose(body_rotations[body]) *
                            normal_vector;
        }
        const float normal[3] = {
            normal_vector.x, normal_vector.y, normal_vector.z,
        };
        const float k = wc.stiffness;
        const float tangent_k = wc.friction_target_alpha.w;
        const float normal_k = k - tangent_k;
        #pragma unroll
        for (int row = 0; row < kBlockJacobiDofs; ++row) {
            const int a = row / 3;
            const int r = row - 3 * a;
            #pragma unroll
            for (int col = 0; col <= row; ++col) {
                const int b = col / 3;
                const int q = col - 3 * b;
                const float h = normal_k * normal[r] * normal[q] +
                    (r == q ? tangent_k : 0.0f);
                A[row][col] += own_w[a] * own_w[b] * h;
            }
        }

        if constexpr (BuildRhs) {
            const math::Vec3f n(normal[0], normal[1], normal[2]);
            const math::Vec3f lagged_relative(
                wc.friction_target_alpha.x,
                wc.friction_target_alpha.y,
                wc.friction_target_alpha.z);
            math::Vec3f rhs_target = n * (k * wc.normal_target.w);
            rhs_target += (lagged_relative -
                           n * math::dot(lagged_relative, n)) * tangent_k;
            if (other_base >= 0 && other_base + 3 < n_controls) {
                math::Vec3f other_position(0.0f, 0.0f, 0.0f);
                #pragma unroll
                for (int a = 0; a < 4; ++a) {
                    other_position += x_lagged[other_base + a] * other_w[a];
                }
                const float other_normal = math::dot(n, other_position);
                rhs_target -= n * (normal_k * other_normal) +
                              other_position * tangent_k;
            }

            #pragma unroll
            for (int a = 0; a < 4; ++a) {
                const float scale = own_w[a];
                b[3 * a + 0] += rhs_target.x * scale;
                b[3 * a + 1] += rhs_target.y * scale;
                b[3 * a + 2] += rhs_target.z * scale;
            }
        }
    }
}

__global__ void build_interpolated_body_preconditioner_kernel(
    int n_bodies,
    int n_controls,
    const float* __restrict__ base_k,
    const unsigned char* __restrict__ fixed,
    const collision::WideContact* __restrict__ contacts,
    const int* __restrict__ ell_counts,
    const std::uint32_t* __restrict__ ell_refs,
    int ell_width,
    const math::Mat3f* __restrict__ body_rotations,
    const math::Vec3f* __restrict__ body_gradients,
    const math::Mat3f* __restrict__ rotational_curvatures,
    const float* __restrict__ body_scales,
    float* __restrict__ packed_lower,
    int* __restrict__ failure_count) {
    const int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= n_bodies) return;

    float A[kBlockJacobiDofs][kBlockJacobiDofs];
    const float* K = base_k + body * kBlockJacobiBaseKSize;
    #pragma unroll
    for (int row = 0; row < kBlockJacobiDofs; ++row) {
        #pragma unroll
        for (int col = 0; col <= row; ++col) {
            A[row][col] = K[sym12_index(row, col)];
        }
    }

    if (rotational_curvatures != nullptr && body_gradients != nullptr &&
        body_scales != nullptr) {
        const math::Vec3f* gradients = body_gradients + body * 4;
        const math::Mat3f rotational = rotational_curvatures[body];
        const float scale = body_scales[body];
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            #pragma unroll
            for (int b = 0; b <= a; ++b) {
                math::Mat3f correction = math::Mat3f::zero();
                #pragma unroll
                for (int axis = 0; axis < 3; ++axis) {
                    math::Vec3f basis(0.0f, 0.0f, 0.0f);
                    basis[axis] = 1.0f;
                    const math::Mat3f delta_f =
                        math::outer(basis, gradients[b]);
                    const math::Vec3f skew =
                        curvature_vex_skew(delta_f);
                    const math::Vec3f column =
                        (curvature_hat(rotational * skew) * gradients[a]) *
                        scale;
                    correction(0, axis) = column.x;
                    correction(1, axis) = column.y;
                    correction(2, axis) = column.z;
                }
                #pragma unroll
                for (int row_axis = 0; row_axis < 3; ++row_axis) {
                    #pragma unroll
                    for (int col_axis = 0; col_axis < 3; ++col_axis) {
                        const int row = 3 * a + row_axis;
                        const int col = 3 * b + col_axis;
                        if (row >= col) {
                            A[row][col] += correction(row_axis, col_axis);
                        }
                    }
                }
            }
        }
    }

    gather_body_contact_ell_to_local_system<false>(
        body, contacts, ell_counts, ell_refs, ell_width, n_bodies,
        n_controls, nullptr, body_rotations, A, nullptr);

    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        if (!fixed[body * 4 + a]) continue;
        #pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            const int row = 3 * a + axis;
            #pragma unroll
            for (int col = 0; col < row; ++col) A[row][col] = 0.0f;
            #pragma unroll
            for (int lower_row = row + 1;
                 lower_row < kBlockJacobiDofs; ++lower_row) {
                A[lower_row][row] = 0.0f;
            }
            A[row][row] = 1.0f;
        }
    }

    float diag_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < kBlockJacobiDofs; ++i) {
        diag_sum += A[i][i];
    }
    const float eps = fmaxf(1.0e-6f,
                            1.0e-6f * diag_sum / kBlockJacobiDofs);
    #pragma unroll
    for (int i = 0; i < kBlockJacobiDofs; ++i) {
        if (!fixed[body * 4 + i / 3]) A[i][i] += eps;
    }

    bool failed = false;
    #pragma unroll
    for (int i = 0; i < kBlockJacobiDofs; ++i) {
        #pragma unroll
        for (int j = 0; j <= i; ++j) {
            float sum = A[i][j];
            #pragma unroll
            for (int k = 0; k < j; ++k) {
                sum -= A[i][k] * A[j][k];
            }
            if (i == j) {
                if (sum <= eps) {
                    sum = eps;
                    failed = true;
                }
                A[i][j] = sqrtf(sum);
            } else {
                A[i][j] = sum / A[j][j];
            }
        }
    }

    float* lower = packed_lower + body * kBodyBlockPackedLowerSize;
    #pragma unroll
    for (int i = 0; i < kBlockJacobiDofs; ++i) {
        #pragma unroll
        for (int j = 0; j <= i; ++j) {
            lower[i * (i + 1) / 2 + j] = A[i][j];
        }
    }
    if (failed) atomicAdd(failure_count, 1);
}

__global__ void solve_interpolated_block_jacobi_kernel(
    const PabdCudaSolver::TetData* __restrict__ tets,
    int n_tets,
    const float* __restrict__ base_k,
    const float* __restrict__ mass_blocks,
    const float* __restrict__ arap_effective_inertias,
    const PabdEndpointHinge* __restrict__ hinges,
    const math::Vec3f* __restrict__ predicted,
    const math::Vec3f* __restrict__ rest,
    const collision::WideContact* __restrict__ contacts,
    const int* __restrict__ ell_counts,
    const std::uint32_t* __restrict__ ell_refs,
    int ell_width,
    int n_controls,
    const math::Vec3f* __restrict__ x_lagged,
    const unsigned char* __restrict__ fixed,
    float inv_h2,
    float stiffness,
    float arap_beta_inv_h2,
    float fixed_weight,
    float hinge_beta_inv_h2,
    math::Vec3f* __restrict__ x,
    float omega,
    int* __restrict__ failure_count) {
    const int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= n_tets) return;

    float A[kBlockJacobiDofs][kBlockJacobiDofs];
    float b[kBlockJacobiDofs];
    const PabdCudaSolver::TetData tet = tets[body];
    const float* K = base_k + body * kBlockJacobiBaseKSize;
    const float* mass = mass_blocks + body * 16;
    PabdEndpointHinge hinge{};
    const bool has_hinge = hinges != nullptr && hinge_beta_inv_h2 > 0.0f &&
        ((hinge = hinges[body]).body == body);

    const math::Mat3f Ds = make_columns(
        x_lagged[tet.v[1]] - x_lagged[tet.v[0]],
        x_lagged[tet.v[2]] - x_lagged[tet.v[0]],
        x_lagged[tet.v[3]] - x_lagged[tet.v[0]]);
    const math::Mat3f R = polar_rotation(Ds * tet.inv_dm);
    const float w_elastic = tet_arap_weight(
        tet, body, arap_effective_inertias, stiffness,
        arap_beta_inv_h2);

    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int id = tet.v[a];
        math::Vec3f bi(0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int col = 0; col < 4; ++col) {
            const float m = __fmul_rn(mass[a * 4 + col], inv_h2);
            const math::Vec3f p = predicted[tet.v[col]];
            bi.x = __fadd_rn(bi.x, __fmul_rn(p.x, m));
            bi.y = __fadd_rn(bi.y, __fmul_rn(p.y, m));
            bi.z = __fadd_rn(bi.z, __fmul_rn(p.z, m));
        }
        const math::Vec3f rotated = R * tet.grad[a];
        bi.x = __fadd_rn(bi.x, __fmul_rn(rotated.x, w_elastic));
        bi.y = __fadd_rn(bi.y, __fmul_rn(rotated.y, w_elastic));
        bi.z = __fadd_rn(bi.z, __fmul_rn(rotated.z, w_elastic));
        if (fixed[id]) {
            const math::Vec3f target = rest[id];
            bi.x = __fadd_rn(bi.x, __fmul_rn(target.x, fixed_weight));
            bi.y = __fadd_rn(bi.y, __fmul_rn(target.y, fixed_weight));
            bi.z = __fadd_rn(bi.z, __fmul_rn(target.z, fixed_weight));
        }
        if (has_hinge) {
            const math::Vec3f endpoint0 = hinge_vec3(hinge.endpoint0);
            const math::Vec3f endpoint1 = hinge_vec3(hinge.endpoint1);
            const math::Vec3f metric_endpoint0 =
                endpoint0 * hinge.effective_mass.x +
                endpoint1 * hinge.effective_mass.y;
            const math::Vec3f metric_endpoint1 =
                endpoint0 * hinge.effective_mass.y +
                endpoint1 * hinge.effective_mass.z;
            const math::Vec3f target =
                metric_endpoint0 * hinge.weights0[a] +
                metric_endpoint1 * hinge.weights1[a];
            bi.x = __fadd_rn(bi.x,
                             __fmul_rn(target.x, hinge_beta_inv_h2));
            bi.y = __fadd_rn(bi.y,
                             __fmul_rn(target.y, hinge_beta_inv_h2));
            bi.z = __fadd_rn(bi.z,
                             __fmul_rn(target.z, hinge_beta_inv_h2));
        }
        b[3 * a + 0] = bi.x;
        b[3 * a + 1] = bi.y;
        b[3 * a + 2] = bi.z;
    }

    #pragma unroll
    for (int r = 0; r < kBlockJacobiDofs; ++r) {
        #pragma unroll
        for (int c = 0; c < kBlockJacobiDofs; ++c) {
            A[r][c] = K[sym12_index(r, c)];
        }
    }

    gather_body_contact_ell_to_local_system<true>(
        body, contacts, ell_counts, ell_refs, ell_width, n_tets,
        n_controls, x_lagged, nullptr, A, b);

    float diag_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < kBlockJacobiDofs; ++i) {
        diag_sum += A[i][i];
    }
    const float eps = fmaxf(1.0e-6f, 1.0e-6f * diag_sum / 12.0f);
    #pragma unroll
    for (int i = 0; i < kBlockJacobiDofs; ++i) A[i][i] += eps;

    bool failed = false;
    #pragma unroll
    for (int i = 0; i < kBlockJacobiDofs; ++i) {
        #pragma unroll
        for (int j = 0; j <= i; ++j) {
            float sum = A[i][j];
            #pragma unroll
            for (int k = 0; k < j; ++k) {
                sum -= A[i][k] * A[j][k];
            }
            if (i == j) {
                if (sum <= eps) {
                    sum = eps;
                    failed = true;
                }
                A[i][j] = sqrtf(sum);
            } else {
                A[i][j] = sum / A[j][j];
            }
        }
        #pragma unroll
        for (int j = i + 1; j < kBlockJacobiDofs; ++j) {
            A[i][j] = 0.0f;
        }
    }

    float y[kBlockJacobiDofs];
    #pragma unroll
    for (int i = 0; i < kBlockJacobiDofs; ++i) {
        float sum = b[i];
        #pragma unroll
        for (int k = 0; k < i; ++k) {
            sum -= A[i][k] * y[k];
        }
        y[i] = sum / A[i][i];
    }

    float solved[kBlockJacobiDofs];
    for (int i = kBlockJacobiDofs - 1; i >= 0; --i) {
        float sum = y[i];
        for (int k = i + 1; k < kBlockJacobiDofs; ++k) {
            sum -= A[k][i] * solved[k];
        }
        solved[i] = sum / A[i][i];
    }

    const float w = fminf(fmaxf(omega, 0.0f), 1.0f);
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int id = tet.v[a];
        if (fixed[id]) continue;
        const math::Vec3f target(solved[3 * a + 0],
                                 solved[3 * a + 1],
                                 solved[3 * a + 2]);
        x[id] = x[id] * (1.0f - w) + target * w;
    }

    if (failed) {
        atomicAdd(failure_count, 1);
    }
}

__global__ void assemble_interpolated_mass_elastic_kernel(
    math::Mat3f* diag,
    math::Mat3f* values,
    math::Vec3f* rhs,
    const PabdCudaSolver::TetData* tets,
    const float* mass_blocks,
    const float* arap_effective_inertias,
    int n_tets,
    const math::Vec3f* y,
    const math::Vec3f* x,
    float inv_h2,
    float stiffness,
    float arap_beta_inv_h2,
    PabdElasticCurvatureMode curvature_mode,
    const math::Mat3f* __restrict__ precomputed_rotations,
    bool assemble_elastic_hessian) {
    const int elem_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_id >= n_tets) return;

    const PabdCudaSolver::TetData tet = tets[elem_id];
    const float* mass = mass_blocks + elem_id * 16;

    const math::Mat3f Ds = make_columns(x[tet.v[1]] - x[tet.v[0]],
                                        x[tet.v[2]] - x[tet.v[0]],
                                        x[tet.v[3]] - x[tet.v[0]]);
    const math::Mat3f F = Ds * tet.inv_dm;
    const math::Mat3f R = precomputed_rotations != nullptr
        ? precomputed_rotations[elem_id]
        : polar_rotation(F);
    const math::Mat3f stretch = assemble_elastic_hessian
        ? math::transpose(R) * F
        : math::Mat3f::identity();
    const float w_elastic = tet_arap_weight(
        tet, elem_id, arap_effective_inertias, stiffness,
        arap_beta_inv_h2);

    for (int row = 0; row < 4; ++row) {
        const int row_id = tet.v[row];
        for (int col = 0; col < 4; ++col) {
            const int col_id = tet.v[col];
            const float m = mass[row * 4 + col] * inv_h2;
            const int slot = tet.hessian_slot[row * 4 + col];
            add_scalar_identity_by_slot_device(diag, values, slot, m);
            atomic_add_vec(&rhs[row_id], (y[col_id] - x[col_id]) * m);

            if (assemble_elastic_hessian) {
                const math::Mat3f elastic = elastic_control_block(
                    tet.grad[row], tet.grad[col], R, stretch,
                    curvature_mode, w_elastic);
                add_mat3_by_slot_device(diag, values, slot, elastic);
            }
        }
        atomic_add_vec(&rhs[row_id],
                       ((R - F) * tet.grad[row]) * w_elastic);
    }
}

__global__ void assemble_interpolated_endpoint_hinge_kernel(
    math::Mat3f* diag,
    math::Mat3f* values,
    math::Vec3f* rhs,
    const PabdCudaSolver::TetData* tets,
    const PabdEndpointHinge* hinges,
    const math::Vec3f* x,
    int n_tets,
    float beta_inv_h2) {
    const int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= n_tets || beta_inv_h2 <= 0.0f) return;

    const PabdEndpointHinge hinge = hinges[body];
    if (hinge.body != body) return;
    const PabdCudaSolver::TetData tet = tets[body];
    const math::Vec3f p0 = hinge_vec3(hinge.endpoint0);
    const math::Vec3f p1 = hinge_vec3(hinge.endpoint1);
    math::Vec3f y0(0.0f, 0.0f, 0.0f);
    math::Vec3f y1(0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int col = 0; col < 4; ++col) {
        y0 += x[tet.v[col]] * hinge.weights0[col];
        y1 += x[tet.v[col]] * hinge.weights1[col];
    }
    const math::Vec3f residual0 =
        (p0 - y0) * hinge.effective_mass.x +
        (p1 - y1) * hinge.effective_mass.y;
    const math::Vec3f residual1 =
        (p0 - y0) * hinge.effective_mass.y +
        (p1 - y1) * hinge.effective_mass.z;

    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        const float w0_row = hinge.weights0[row];
        const float w1_row = hinge.weights1[row];
        atomic_add_vec(&rhs[tet.v[row]],
                       (residual0 * w0_row + residual1 * w1_row) *
                           beta_inv_h2);
        #pragma unroll
        for (int col = 0; col < 4; ++col) {
            const float value = beta_inv_h2 *
                endpoint_hinge_weight_block(hinge, row, col);
            add_scalar_identity_by_slot_device(
                diag, values, tet.hessian_slot[row * 4 + col], value);
        }
    }
}

__global__ void assemble_interpolated_fixed_kernel(
    math::Mat3f* diag,
    math::Vec3f* rhs,
    const math::Vec3f* rest,
    const math::Vec3f* x,
    const unsigned char* fixed,
    int n,
    float fixed_weight) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !fixed[i]) return;
    atomic_add_scalar_identity(&diag[i], fixed_weight);
    atomic_add_vec(&rhs[i], (rest[i] - x[i]) * fixed_weight);
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

__device__ float wide_contact_effective_mass(
    const int* dofs,
    const float* coeffs,
    int coeff_count,
    const float* inverse_mass_blocks,
    int body_count) {
    if (inverse_mass_blocks == nullptr || body_count <= 0) return 0.0f;

    int bases[2] = {-1, -1};
    float q[2][4]{};
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (i >= coeff_count) break;
        const int base = (dofs[i] / 4) * 4;
        const int lane = dofs[i] - base;
        if (lane < 0 || lane >= 4) return 0.0f;
        int group = -1;
        if (bases[0] < 0 || bases[0] == base) {
            bases[0] = base;
            group = 0;
        } else if (bases[1] < 0 || bases[1] == base) {
            bases[1] = base;
            group = 1;
        } else {
            return 0.0f;
        }
        q[group][lane] += coeffs[i];
    }

    float inverse_effective_mass = 0.0f;
    #pragma unroll
    for (int group = 0; group < 2; ++group) {
        if (bases[group] < 0) continue;
        const int body = bases[group] / 4;
        if (body < 0 || body >= body_count) return 0.0f;
        const float* inverse_mass = inverse_mass_blocks + body * 16;
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            float value = 0.0f;
            #pragma unroll
            for (int col = 0; col < 4; ++col) {
                value += inverse_mass[row * 4 + col] * q[group][col];
            }
            inverse_effective_mass += q[group][row] * value;
        }
    }
    return inverse_effective_mass > 1.0e-20f &&
                   isfinite(inverse_effective_mass)
        ? 1.0f / inverse_effective_mass
        : 0.0f;
}

__device__ float ground_contact_stiffness(
    const int* dofs,
    const float* coeffs,
    int coeff_count,
    const float* inverse_mass_blocks,
    int body_count,
    float beta_inv_h2,
    float legacy_stiffness) {
    if (inverse_mass_blocks == nullptr) return legacy_stiffness;
    const float effective_mass = wide_contact_effective_mass(
        dofs, coeffs, coeff_count, inverse_mass_blocks, body_count);
    return beta_inv_h2 * effective_mass;
}

__device__ float self_contact_stiffness(
    const collision::MeshMeshContact& contact,
    const math::Vec3f* surface_positions,
    const int* dofs,
    const float* coeffs,
    int coeff_count,
    const float* inverse_mass_blocks,
    int body_count,
    float activation_distance,
    float density,
    float beta_inv_h2,
    float legacy_stiffness,
    bool use_beta,
    PabdSelfContactMeasureMode measure_mode,
    float point_face_scale,
    float* volume_out) {
    const int ids[4] = {
        contact.vertices.x, contact.vertices.y,
        contact.vertices.z, contact.vertices.w};
    const float volume = contact_tetrahedron_volume_at_activation(
        surface_positions[ids[0]], surface_positions[ids[1]],
        surface_positions[ids[2]], surface_positions[ids[3]],
        contact.distance, activation_distance);
    if (volume_out != nullptr) *volume_out = volume;

    float stiffness = legacy_stiffness;
    if (use_beta) {
        if (measure_mode == PabdSelfContactMeasureMode::EffectiveMass) {
            stiffness = beta_inv_h2 * wide_contact_effective_mass(
                dofs, coeffs, coeff_count, inverse_mass_blocks, body_count);
        } else {
            stiffness = beta_inv_h2 * density * volume;
        }
    }
    if (contact.type ==
        static_cast<int>(collision::MeshMeshContactType::PointFace)) {
        stiffness *= fmaxf(point_face_scale, 0.0f);
    }
    return stiffness;
}

__device__ void atomic_max_nonnegative_float(float* address, float value) {
    if (!(value > 0.0f) || !isfinite(value)) return;
    auto* bits = reinterpret_cast<int*>(address);
    int old = *bits;
    const int desired = __float_as_int(value);
    while (old < desired) {
        const int assumed = old;
        old = atomicCAS(bits, assumed, desired);
        if (old == assumed) break;
    }
}

__device__ math::Vec4f compute_wide_contact_friction(
    const int* dofs,
    const float* coeffs,
    int coeff_count,
    math::Vec3f normal,
    float depth,
    float normal_stiffness,
    float friction_mu,
    float friction_epsilon,
    const math::Vec3f* current_controls,
    const math::Vec3f* old_controls) {
    math::Vec3f old_relative(0.0f, 0.0f, 0.0f);
    math::Vec3f current_relative(0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (i >= coeff_count) break;
        old_relative += old_controls[dofs[i]] * coeffs[i];
        current_relative += current_controls[dofs[i]] * coeffs[i];
    }

    float tangent_stiffness = 0.0f;
    if (friction_mu > 0.0f && depth > 0.0f) {
        const math::Vec3f delta = current_relative - old_relative;
        const math::Vec3f slip = delta - normal * math::dot(delta, normal);
        const float slip_norm = sqrtf(math::dot(slip, slip));
        const float epsilon = fmaxf(friction_epsilon, 1.0e-8f);
        const float normal_force = normal_stiffness * depth;
        tangent_stiffness = friction_mu * normal_force *
            collision::ipc_f1_sf_over_x(slip_norm, epsilon);
    }
    return math::Vec4f(old_relative.x, old_relative.y, old_relative.z,
                       tangent_stiffness);
}

__device__ math::Vec3f wide_contact_rhs_target(
    math::Vec3f normal,
    float normal_target,
    float normal_stiffness,
    math::Vec4f friction_target_alpha) {
    math::Vec3f target = normal * (normal_stiffness * normal_target);
    const float tangent_stiffness = friction_target_alpha.w;
    if (tangent_stiffness > 0.0f) {
        const math::Vec3f lagged_relative(
            friction_target_alpha.x,
            friction_target_alpha.y,
            friction_target_alpha.z);
        const math::Vec3f tangent_target = lagged_relative -
            normal * math::dot(lagged_relative, normal);
        target += tangent_target * tangent_stiffness;
    }
    return target;
}

__device__ void store_wide_contact(collision::WideContact* wide_contacts,
                                   int cid,
                                   const int* dofs,
                                   const float* coeffs,
                                   int coeff_count,
                                   math::Vec3f normal,
                                   float target,
                                   float stiffness,
                                   math::Vec4f friction_target_alpha) {
    collision::WideContact out{};
    out.base0 = -1;
    out.base1 = -1;
    out.weights0 = math::Vec4f(0.0f, 0.0f, 0.0f, 0.0f);
    out.weights1 = math::Vec4f(0.0f, 0.0f, 0.0f, 0.0f);
    out.normal_target = math::Vec4f(normal.x, normal.y, normal.z, target);
    out.friction_target_alpha = friction_target_alpha;
    out.stiffness = stiffness;
    for (int i = 0; i < coeff_count && i < 8; ++i) {
        const int base = (dofs[i] / 4) * 4;
        const int lane = dofs[i] - base;
        if (lane < 0 || lane >= 4) continue;
        if (out.base0 < 0 || out.base0 == base) {
            out.base0 = base;
            out.weights0[lane] += coeffs[i];
        } else if (out.base1 < 0 || out.base1 == base) {
            out.base1 = base;
            out.weights1[lane] += coeffs[i];
        } else {
            return;
        }
    }
    wide_contacts[cid] = out;
}

__global__ void append_interpolated_ground_contacts_kernel(
    math::Vec3f* rhs,
    const PabdSurfaceMapDevice* maps,
    const math::Vec3f* surface_positions,
    const math::Vec3f* current_controls,
    const math::Vec3f* old_controls,
    int n_surface,
    float target_y,
    float legacy_stiffness,
    const float* inverse_mass_blocks,
    int body_count,
    float beta_inv_h2,
    float friction_mu,
    float friction_epsilon,
    int* ground_count,
    collision::WideContact* wide_contacts,
    int* wide_count,
    int wide_capacity) {
    const int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= n_surface) return;

    const PabdSurfaceMapDevice map = maps[sid];
    if (!map.ground_collide || surface_positions[sid].y >= target_y) return;

    int dofs[8];
    float coeffs[8];
    int coeff_count = 0;
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int ia = map.index[a];
        const float wa = map.weight[a];
        if (ia < 0 || wa == 0.0f) continue;
        merge_wide_coeff(ia, wa, dofs, coeffs, &coeff_count);
    }
    if (coeff_count > 0) {
        const float stiffness = ground_contact_stiffness(
            dofs, coeffs, coeff_count, inverse_mass_blocks, body_count,
            beta_inv_h2, legacy_stiffness);
        if (!(stiffness > 0.0f) || !isfinite(stiffness)) return;
        atomicAdd(ground_count, 1);
        const math::Vec3f normal(0.0f, 1.0f, 0.0f);
        const math::Vec4f friction = compute_wide_contact_friction(
            dofs, coeffs, coeff_count, normal,
            target_y - surface_positions[sid].y,
            stiffness, friction_mu, friction_epsilon,
            current_controls, old_controls);
        const math::Vec3f rhs_target = wide_contact_rhs_target(
            normal, target_y, stiffness, friction);
        if (rhs != nullptr) {
            for (int i = 0; i < coeff_count; ++i) {
                atomic_add_vec(&rhs[dofs[i]], rhs_target * coeffs[i]);
            }
        }
        const int cid = atomicAdd(wide_count, 1);
        if (cid < wide_capacity) {
            store_wide_contact(wide_contacts, cid, dofs, coeffs, coeff_count,
                               normal, target_y, stiffness, friction);
        }
    }
}

__global__ void append_interpolated_self_contacts_kernel(
    math::Vec3f* rhs,
    const collision::MeshMeshContact* contacts,
    const int* contact_count,
    int max_contacts,
    const PabdSurfaceMapDevice* maps,
    const math::Vec3f* surface_positions,
    const math::Vec3f* current_controls,
    const math::Vec3f* old_controls,
    const float* inverse_mass_blocks,
    int body_count,
    float contact_gap,
    float legacy_stiffness,
    float density,
    float beta_inv_h2,
    bool use_beta,
    PabdSelfContactMeasureMode measure_mode,
    float point_face_scale,
    bool enable_point_face,
    bool enable_edge_edge,
    float normal_damping,
    float friction_mu,
    float friction_epsilon,
    int* debug_counts,
    float* contact_metrics,
    math::Vec3f* normal_sum,
    collision::WideContact* wide_contacts,
    int* wide_count,
    int wide_capacity) {
    const int n_raw = *contact_count;
    const int n = n_raw < max_contacts ? n_raw : max_contacts;
    if (contact_gap <= 0.0f) return;
    const int stride = blockDim.x * gridDim.x;
    for (int cid = blockIdx.x * blockDim.x + threadIdx.x;
         cid < n; cid += stride) {
        const collision::MeshMeshContact c = contacts[cid];
        if (c.type == static_cast<int>(collision::MeshMeshContactType::PointFace)) {
            atomicAdd(&debug_counts[3], 1);
            if (!enable_point_face) continue;
        } else if (c.type == static_cast<int>(collision::MeshMeshContactType::EdgeEdge)) {
            atomicAdd(&debug_counts[4], 1);
            if (!enable_edge_edge) continue;
        } else {
            continue;
        }
        if (!(c.separation < contact_gap)) continue;

        int dofs[8];
        float coeffs[8];
        int coeff_count = 0;
        const int ids[4] = {
            c.vertices.x, c.vertices.y, c.vertices.z, c.vertices.w};
        const float ws[4] = {
            c.weights.x, c.weights.y, c.weights.z, c.weights.w};
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

        if (coeff_count > 0) {
            float contact_volume = 0.0f;
            const float stiffness = self_contact_stiffness(
                c, surface_positions, dofs, coeffs, coeff_count,
                inverse_mass_blocks, body_count, contact_gap, density,
                beta_inv_h2, legacy_stiffness, use_beta, measure_mode,
                point_face_scale, &contact_volume);
            if (!(stiffness > 0.0f) || !isfinite(stiffness)) continue;
            atomicAdd(&debug_counts[0], 1);
            const bool point_face =
                c.type == static_cast<int>(
                    collision::MeshMeshContactType::PointFace);
            const bool edge_edge =
                c.type == static_cast<int>(
                    collision::MeshMeshContactType::EdgeEdge);
            if (point_face) atomicAdd(&debug_counts[6], 1);
            if (edge_edge) atomicAdd(&debug_counts[7], 1);
            const float depth = fmaxf(contact_gap - c.separation, 0.0f);
            atomic_max_nonnegative_float(&contact_metrics[0], depth);
            atomicAdd(&contact_metrics[1], depth * depth);
            atomicAdd(&contact_metrics[2], stiffness);
            if (point_face) {
                atomicAdd(&contact_metrics[3], stiffness);
                atomicAdd(&contact_metrics[5], contact_volume);
            }
            if (edge_edge) {
                atomicAdd(&contact_metrics[4], stiffness);
                atomicAdd(&contact_metrics[6], contact_volume);
            }
            atomic_add_vec(normal_sum, c.normal);
            if (math::abs(c.normal.y) >= 0.75f) {
                atomicAdd(&debug_counts[1], 1);
            } else {
                atomicAdd(&debug_counts[2], 1);
            }
            const math::Vec4f friction = compute_wide_contact_friction(
                dofs, coeffs, coeff_count, c.normal,
                contact_gap - c.separation,
                stiffness, friction_mu, friction_epsilon,
                current_controls, old_controls);
            // Implicit normal dashpot: beta*k*(C(x)-C(x_old))^2 / 2.
            const float damping_ratio = fmaxf(normal_damping, 0.0f);
            const float damping_stiffness = stiffness * damping_ratio;
            const float effective_stiffness =
                stiffness + damping_stiffness;
            float effective_target = contact_gap;
            if (damping_stiffness > 0.0f) {
                const math::Vec3f old_relative(
                    friction.x, friction.y, friction.z);
                const float old_separation =
                    math::dot(old_relative, c.normal);
                effective_target =
                    (stiffness * contact_gap +
                     damping_stiffness * old_separation) /
                    effective_stiffness;
            }
            if (friction.w > 0.0f) {
                atomicAdd(&debug_counts[5], 1);
            }
            const math::Vec3f rhs_target = wide_contact_rhs_target(
                c.normal, effective_target, effective_stiffness, friction);
            if (rhs != nullptr) {
                for (int i = 0; i < coeff_count; ++i) {
                    atomic_add_vec(&rhs[dofs[i]],
                                   rhs_target * coeffs[i]);
                }
            }
            const int out_cid = atomicAdd(wide_count, 1);
            if (out_cid < wide_capacity) {
                store_wide_contact(wide_contacts, out_cid, dofs, coeffs,
                                   coeff_count, c.normal, effective_target,
                                   effective_stiffness, friction);
            }
        }
    }
}

// Assemble b - Hx directly so stiff contacts do not subtract world-scale RHS
// and SpMV values after each has already been rounded to float.
__global__ void assemble_interpolated_wide_contact_residual_kernel(
    math::Vec3f* rhs,
    const collision::WideContact* contacts,
    const int* contact_count,
    int max_contacts,
    const math::Vec3f* x) {
    const int n_raw = *contact_count;
    const int n = n_raw < max_contacts ? n_raw : max_contacts;
    const int stride = blockDim.x * gridDim.x;
    for (int cid = blockIdx.x * blockDim.x + threadIdx.x;
         cid < n; cid += stride) {
        const collision::WideContact contact = contacts[cid];
        const int ids[8] = {
            contact.base0 >= 0 ? contact.base0 + 0 : -1,
            contact.base0 >= 0 ? contact.base0 + 1 : -1,
            contact.base0 >= 0 ? contact.base0 + 2 : -1,
            contact.base0 >= 0 ? contact.base0 + 3 : -1,
            contact.base1 >= 0 ? contact.base1 + 0 : -1,
            contact.base1 >= 0 ? contact.base1 + 1 : -1,
            contact.base1 >= 0 ? contact.base1 + 2 : -1,
            contact.base1 >= 0 ? contact.base1 + 3 : -1,
        };
        const float weights[8] = {
            contact.weights0.x, contact.weights0.y,
            contact.weights0.z, contact.weights0.w,
            contact.weights1.x, contact.weights1.y,
            contact.weights1.z, contact.weights1.w,
        };

        math::Vec3f relative(0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (ids[i] >= 0 && weights[i] != 0.0f) {
                relative += x[ids[i]] * weights[i];
            }
        }

        const math::Vec3f normal(contact.normal_target.x,
                                 contact.normal_target.y,
                                 contact.normal_target.z);
        const float normal_projection = math::dot(relative, normal);
        math::Vec3f residual = normal *
            (contact.stiffness *
             (contact.normal_target.w - normal_projection));

        const float tangent_stiffness = contact.friction_target_alpha.w;
        if (tangent_stiffness > 0.0f) {
            const math::Vec3f lagged_relative(
                contact.friction_target_alpha.x,
                contact.friction_target_alpha.y,
                contact.friction_target_alpha.z);
            const math::Vec3f tangent_target = lagged_relative -
                normal * math::dot(lagged_relative, normal);
            const math::Vec3f tangent_relative = relative -
                normal * normal_projection;
            residual += (tangent_target - tangent_relative) *
                        tangent_stiffness;
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (ids[i] >= 0 && weights[i] != 0.0f) {
                atomic_add_vec(&rhs[ids[i]], residual * weights[i]);
            }
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
                                      const float* arap_effective_inertias,
                                      int num_tets,
                                      float stiffness,
                                      float arap_beta_inv_h2) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_tets * 16) return;

    const int tet_id = id / 16;
    const int local = id - tet_id * 16;
    const int a = local / 4;
    const int b = local - a * 4;
    const auto tet = tets[tet_id];

    const float w = tet_arap_weight(
        tet, tet_id, arap_effective_inertias, stiffness,
        arap_beta_inv_h2);
    const float s = w * math::dot(tet.grad[a], tet.grad[b]);
    const int slot = tet.hessian_slot[local];
    if (slot < 0) {
        atomic_add_scalar_identity(&diag[-slot - 1], s);
    } else {
        atomic_add_scalar_identity(&values[slot], s);
    }
}

__global__ void add_ground_matrix_kernel(math::Mat3f* diag,
                                         const math::Vec3f* x,
                                         const float* mass,
                                         const unsigned char* fixed,
                                         int* contact_count,
                                         int n,
                                         float ground_y,
                                         float gap,
                                         float legacy_stiffness,
                                         float beta_inv_h2,
                                         bool use_beta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || fixed[i]) return;

    const float target = ground_y + gap;
    if (x[i].y >= target) return;
    const float stiffness = use_beta
        ? beta_inv_h2 * mass[i]
        : legacy_stiffness;
    if (!(stiffness > 0.0f) || !isfinite(stiffness)) return;

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
                                   const float* arap_effective_inertias,
                                   int num_tets,
                                   float stiffness,
                                   float arap_beta_inv_h2) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_tets * 4) return;

    const int tet_id = id / 4;
    const int local = id - tet_id * 4;
    const auto tet = tets[tet_id];
    const float w = tet_arap_weight(
        tet, tet_id, arap_effective_inertias, stiffness,
        arap_beta_inv_h2);
    const math::Vec3f value = rotations[tet_id] * tet.grad[local] * w;
    atomic_add_vec(&rhs[tet.v[local]], value);
}

__global__ void add_ground_rhs_kernel(math::Vec3f* rhs,
                                      const math::Vec3f* x,
                                      const float* mass,
                                      const unsigned char* fixed,
                                      int n,
                                      float ground_y,
                                      float gap,
                                      float legacy_stiffness,
                                      float beta_inv_h2,
                                      bool use_beta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || fixed[i]) return;

    const float target = ground_y + gap;
    if (x[i].y >= target) return;
    const float stiffness = use_beta
        ? beta_inv_h2 * mass[i]
        : legacy_stiffness;
    if (!(stiffness > 0.0f) || !isfinite(stiffness)) return;
    rhs[i].y += stiffness * target;
}

__global__ void compute_particle_contact_stiffness_kernel(
    const math::Vec4i* __restrict__ pairs,
    const collision::ContactWeights* __restrict__ weights,
    const int* __restrict__ count_ptr,
    int max_contacts,
    const math::Vec3f* __restrict__ positions,
    const float* __restrict__ mass,
    const unsigned char* __restrict__ fixed,
    float activation_distance,
    float density,
    float beta_inv_h2,
    PabdSelfContactMeasureMode measure_mode,
    float* __restrict__ stiffnesses) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_raw = *count_ptr;
    const int n = (n_raw < max_contacts) ? n_raw : max_contacts;
    if (c >= n) return;

    const math::Vec4i ids = pairs[c];
    const collision::ContactWeights w = weights[c];
    float stiffness = 0.0f;
    if (measure_mode == PabdSelfContactMeasureMode::EffectiveMass) {
        const int vertices[4] = {ids.x, ids.y, ids.z, ids.w};
        const float scalars[4] = {w.w0, w.w1, w.w2, w.w3};
        float inverse_effective_mass = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int vertex = vertices[i];
            if (fixed != nullptr && fixed[vertex] != 0) continue;
            const float vertex_mass = mass[vertex];
            if (vertex_mass > 1.0e-20f && isfinite(vertex_mass)) {
                inverse_effective_mass +=
                    scalars[i] * scalars[i] / vertex_mass;
            }
        }
        if (inverse_effective_mass > 1.0e-20f &&
            isfinite(inverse_effective_mass)) {
            stiffness = beta_inv_h2 / inverse_effective_mass;
        }
    } else {
        const float current_distance = activation_distance - w.depth;
        const float volume = contact_tetrahedron_volume_at_activation(
            positions[ids.x], positions[ids.y],
            positions[ids.z], positions[ids.w],
            current_distance, activation_distance);
        stiffness = beta_inv_h2 * density * volume;
    }
    stiffnesses[c] = stiffness > 0.0f && isfinite(stiffness)
        ? stiffness
        : 0.0f;
}

__global__ void add_self_contact_rhs_kernel(
    const math::Vec4i* __restrict__ pairs,
    const collision::ContactWeights* __restrict__ weights,
    const int* __restrict__ count_ptr,
    int max_contacts,
    float legacy_stiffness,
    const float* __restrict__ stiffnesses,
    const math::Vec3f* __restrict__ x,
    math::Vec3f* __restrict__ rhs) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_raw = *count_ptr;
    const int n = (n_raw < max_contacts) ? n_raw : max_contacts;
    if (c >= n) return;
    const float stiffness = stiffnesses != nullptr
        ? stiffnesses[c]
        : legacy_stiffness;
    if (!(stiffness > 0.0f) || !isfinite(stiffness)) return;

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

__global__ void apply_increment_kernel(math::Vec3f* x,
                                       const math::Vec3f* delta,
                                       const unsigned char* fixed,
                                       int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || fixed[i]) return;
    x[i] += delta[i];
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

bool PabdCudaSolver::matrix_free_polar_gn_active() const noexcept {
    return params_.global_solver != PabdGlobalSolverMode::BlockJacobi12 &&
           params_.elastic_curvature ==
               PabdElasticCurvatureMode::PolarGaussNewton &&
           params_.polar_gn_backend ==
               PabdPolarGnBackend::MatrixFreeRank3 &&
           matrix_free_body_elastic_valid_;
}

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
    if (interpolated_surface_) {
        update_interpolated_surface_positions_gpu(x_.gpu_data(), true);
    }
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
    host_endpoint_hinges_.clear();
    hinge_axis_angular_velocities_.clear();
    hinge_endpoint_errors_.clear();
    host_surface_positions_.assign(
        static_cast<std::size_t>(std::max(0, num_surface_vertices_)),
        math::Vec3f());
    rest_.resize(num_vertices_);
    x_.resize(num_vertices_);
    old_x_.resize(num_vertices_);
    y_.resize(num_vertices_);
    velocity_.resize(num_vertices_);
    rhs_.resize(num_vertices_);
    delta_.resize(num_vertices_);
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
        delta_[i] = math::Vec3f(0.0f, 0.0f, 0.0f);
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
    tet_arap_effective_inertias_.resize(host_tets.size());
    for (std::size_t t = 0; t < host_tets.size(); ++t) {
        float effective_inertia = 0.0f;
        if (!arap_effective_inertia(
                host_mass_blocks_[t], host_tets[t], &effective_inertia)) {
            throw std::invalid_argument(
                "PabdCudaSolver::setup: ARAP effective inertia is singular");
        }
        tet_arap_effective_inertias_[t] = effective_inertia;
    }
    if (interpolated_surface_) {
        tet_mass_blocks_dev_.resize(host_mass_blocks_.size() * 16);
        tet_contact_inverse_mass_blocks_.resize(
            host_mass_blocks_.size() * 16);
        for (std::size_t t = 0; t < host_mass_blocks_.size(); ++t) {
            for (int i = 0; i < 16; ++i) {
                tet_mass_blocks_dev_[t * 16 + i] = host_mass_blocks_[t][i];
            }

            const int base = static_cast<int>(t) * 4;
            std::array<float, 16> global_mass{};
            std::array<unsigned char, 4> global_fixed{};
            const TetData& tet = host_tets[t];
            for (int a = 0; a < 4; ++a) {
                const int lane_a = tet.v[a] - base;
                if (lane_a < 0 || lane_a >= 4) {
                    throw std::invalid_argument(
                        "PabdCudaSolver::setup: interpolated body controls "
                        "must be consecutive groups of four");
                }
                global_fixed[lane_a] = fixed_[tet.v[a]];
                for (int b = 0; b < 4; ++b) {
                    const int lane_b = tet.v[b] - base;
                    if (lane_b < 0 || lane_b >= 4) {
                        throw std::invalid_argument(
                            "PabdCudaSolver::setup: interpolated body controls "
                            "must be consecutive groups of four");
                    }
                    global_mass[lane_a * 4 + lane_b] =
                        host_mass_blocks_[t][a * 4 + b];
                }
            }
            std::array<float, 16> inverse_mass{};
            if (!contact_inverse_mass_block(
                    global_mass, global_fixed, &inverse_mass)) {
                throw std::invalid_argument(
                    "PabdCudaSolver::setup: contact inverse mass is singular");
            }
            for (int i = 0; i < 16; ++i) {
                tet_contact_inverse_mass_blocks_[t * 16 + i] =
                    inverse_mass[i];
            }
        }
    } else {
        tet_mass_blocks_dev_.clear();
        tet_contact_inverse_mass_blocks_.clear();
    }

    tets_.resize(host_tets.size());
    rotations_.resize(host_tets.size());
    for (std::size_t i = 0; i < host_tets.size(); ++i) {
        tets_[i] = host_tets[i];
        rotations_[i] = math::Mat3f::identity();
    }

    host_endpoint_hinges_.assign(host_tets.size(), PabdEndpointHinge{});
    if (!mesh.endpoint_hinges.empty()) {
        hinge_axis_angular_velocities_.assign(host_tets.size(), 0.0f);
        hinge_endpoint_errors_.assign(host_tets.size(), 0.0f);
    }
    for (const PabdEndpointHinge& input : mesh.endpoint_hinges) {
        if (input.body < 0 ||
            input.body >= static_cast<int>(host_tets.size())) {
            throw std::out_of_range(
                "PabdCudaSolver::setup: endpoint hinge body out of range");
        }
        if (host_endpoint_hinges_[input.body].body >= 0) {
            throw std::invalid_argument(
                "PabdCudaSolver::setup: duplicate endpoint hinge body");
        }

        PabdEndpointHinge hinge = input;
        hinge.weights0 = math::Vec4f(0.0f);
        hinge.weights1 = math::Vec4f(0.0f);
        const auto& original_tet = mesh.tets[input.body];
        const TetData& oriented_tet = host_tets[input.body];
        for (int oriented = 0; oriented < 4; ++oriented) {
            int original = -1;
            for (int candidate = 0; candidate < 4; ++candidate) {
                if (original_tet[candidate] == oriented_tet.v[oriented]) {
                    original = candidate;
                    break;
                }
            }
            if (original < 0) {
                throw std::invalid_argument(
                    "PabdCudaSolver::setup: hinge controls do not match tet");
            }
            hinge.weights0[oriented] = input.weights0[original];
            hinge.weights1[oriented] = input.weights1[original];
        }
        if (!endpoint_hinge_effective_mass(
                host_mass_blocks_[input.body], hinge.weights0,
                hinge.weights1, &hinge.effective_mass)) {
            throw std::invalid_argument(
                "PabdCudaSolver::setup: endpoint hinge effective mass is "
                "singular");
        }

        math::Vec3f axis(hinge.axis.x, hinge.axis.y, hinge.axis.z);
        const float axis_length = math::length(axis);
        if (axis_length <= 1.0e-8f) {
            throw std::invalid_argument(
                "PabdCudaSolver::setup: endpoint hinge has zero axis");
        }
        axis /= axis_length;
        hinge.axis = math::Vec4f(axis.x, axis.y, axis.z, 0.0f);
        host_endpoint_hinges_[input.body] = hinge;
    }
    if (interpolated_surface_ && !mesh.endpoint_hinges.empty()) {
        endpoint_hinges_dev_.resize(host_endpoint_hinges_.size());
        endpoint_hinge_diagnostics_dev_.resize(host_endpoint_hinges_.size());
        for (std::size_t body = 0;
             body < host_endpoint_hinges_.size(); ++body) {
            endpoint_hinges_dev_[body] = host_endpoint_hinges_[body];
            endpoint_hinge_diagnostics_dev_[body] =
                math::Vec3f(0.0f, 0.0f, 0.0f);
        }
    } else {
        endpoint_hinges_dev_.clear();
        endpoint_hinge_diagnostics_dev_.clear();
    }
    block_jacobi_base_k_.allocate_device(
        host_tets.size() * kBlockJacobiBaseKSize);
    block_jacobi_x_lagged_.allocate_device(num_vertices_);
    block_jacobi_failure_count_.resize(1);
    block_jacobi_failure_count_[0] = 0;
    elastic_curvature_diagnostics_dev_.resize(4);
    for (int i = 0; i < 4; ++i) {
        elastic_curvature_diagnostics_dev_[i] = 0.0f;
    }
    block_jacobi_base_k_valid_ = false;

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
        particle_contact_stiffnesses_.resize(max_contacts);
    } else {
        self_collision_detector_.bind_topology(nullptr);
        particle_contact_stiffnesses_.clear();
    }
    self_collision_.set_stiffness(params_.use_contact_beta
        ? (params_.self_collision_beta > 0.0f ? 1.0f : 0.0f)
        : params_.self_collision_stiffness);

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
        interpolated_surface_min_y_dev_.resize(1);
        interpolated_surface_min_y_dev_[0] = 0.0f;
        interpolated_debug_counts_.resize(8);
        for (int i = 0; i < 8; ++i) {
            interpolated_debug_counts_[i] = 0;
        }
        interpolated_contact_metrics_.resize(7);
        for (int i = 0; i < 7; ++i) {
            interpolated_contact_metrics_[i] = 0.0f;
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

        block_jacobi_contact_ell_width_ = choose_body_contact_ell_width(
            wide_capacity, static_cast<int>(host_tets.size()));
        block_jacobi_contact_ell_counts_.allocate_device(host_tets.size());
        const std::size_t ell_slots =
            host_tets.size() *
            static_cast<std::size_t>(block_jacobi_contact_ell_width_);
        block_jacobi_contact_ell_refs_.allocate_device(ell_slots);
        block_jacobi_contact_ell_overflow_.resize(1);
        block_jacobi_contact_ell_overflow_[0] = 0;

        pcg_body_preconditioner_valid_ =
            num_vertices_ == static_cast<int>(host_tets.size()) * 4;
        if (pcg_body_preconditioner_valid_) {
            pcg_body_preconditioner_rows_.resize(host_tets.size());
            for (std::size_t body = 0; body < host_tets.size(); ++body) {
                const TetData& tet = host_tets[body];
                for (int a = 0; a < 4; ++a) {
                    if (tet.v[a] != static_cast<int>(body) * 4 + a) {
                        pcg_body_preconditioner_valid_ = false;
                    }
                }
                pcg_body_preconditioner_rows_[body] = math::Vec4i(
                    tet.v[0], tet.v[1], tet.v[2], tet.v[3]);
            }
        }
        if (pcg_body_preconditioner_valid_) {
            pcg_body_preconditioner_lower_.allocate_device(
                host_tets.size() * kBodyBlockPackedLowerSize);
            pcg_body_preconditioner_failure_count_.resize(1);
            pcg_body_preconditioner_failure_count_[0] = 0;
            matrix_free_body_elastic_valid_ = true;
            matrix_free_body_gradients_.resize(host_tets.size() * 4);
            matrix_free_body_rotations_.resize(host_tets.size());
            matrix_free_body_rotational_curvatures_.resize(host_tets.size());
            matrix_free_body_scales_.resize(host_tets.size());
            const float setup_h = params_.dt /
                static_cast<float>(std::max(1, params_.substeps));
            const float setup_inv_h2 =
                1.0f / std::max(setup_h * setup_h, 1.0e-8f);
            for (std::size_t body = 0; body < host_tets.size(); ++body) {
                for (int a = 0; a < 4; ++a) {
                    matrix_free_body_gradients_[body * 4 + a] =
                        host_tets[body].grad[a];
                }
                matrix_free_body_rotations_[body] = math::Mat3f::identity();
                matrix_free_body_rotational_curvatures_[body] =
                    math::Mat3f::zero();
                matrix_free_body_scales_[body] = params_.use_arap_beta
                    ? params_.arap_beta * setup_inv_h2 *
                        tet_arap_effective_inertias_[body]
                    : params_.stiffness * host_tets[body].volume;
            }
        } else {
            pcg_body_preconditioner_lower_.clear();
            pcg_body_preconditioner_rows_.clear();
            pcg_body_preconditioner_failure_count_.clear();
            matrix_free_body_gradients_.clear();
            matrix_free_body_rotations_.clear();
            matrix_free_body_rotational_curvatures_.clear();
            matrix_free_body_scales_.clear();
            matrix_free_body_elastic_valid_ = false;
            std::printf(
                "[PABD] 12x12 PCG body preconditioner disabled: controls "
                "are not four canonical rows per body\n");
        }
        std::printf(
            "[PABD_DBG ELL] bodies=%zu width=%d slots=%zu memory=%.2f MiB\n",
            host_tets.size(), block_jacobi_contact_ell_width_, ell_slots,
            static_cast<double>(ell_slots * sizeof(std::uint32_t)) /
                (1024.0 * 1024.0));

        std::vector<int> vertex_body_ids(surface_maps_.size(), 0);
        for (std::size_t i = 0; i < surface_maps_.size(); ++i) {
            vertex_body_ids[i] = surface_maps_[i].body;
        }
        update_surface_positions(mesh.rest_positions, host_surface_positions_);
        const int max_ef_candidates = std::max(
            interpolated_contact_capacity_,
            static_cast<int>(surface_edges_.size()) * 64);
        const collision::BroadphaseBackend self_backend =
#ifdef CHYSX_HAS_OPTIX
            collision::BroadphaseBackend::OptiX;
#else
            collision::BroadphaseBackend::QuantBvh;
#endif
        mesh_mesh_contact_detector_.setup(
            tri_vec, vertex_body_ids, interpolated_contact_capacity_,
            max_ef_candidates, self_backend,
            collision::kMeshCollisionInterObject,
            &host_surface_positions_);
    } else {
        surface_maps_dev_.clear();
        surface_triangles_dev_.clear();
        surface_edges_dev_.clear();
        interpolated_surface_positions_dev_.clear();
        interpolated_surface_min_y_dev_.clear();
        interpolated_debug_counts_.clear();
        interpolated_contact_metrics_.clear();
        interpolated_normal_sum_.clear();
        interpolated_wide_contacts_.clear();
        interpolated_wide_contact_count_.clear();
        block_jacobi_contact_ell_counts_.clear();
        block_jacobi_contact_ell_refs_.clear();
        block_jacobi_contact_ell_overflow_.clear();
        pcg_body_preconditioner_lower_.clear();
        pcg_body_preconditioner_rows_.clear();
        pcg_body_preconditioner_failure_count_.clear();
        pcg_body_preconditioner_valid_ = false;
        matrix_free_body_gradients_.clear();
        matrix_free_body_rotations_.clear();
        matrix_free_body_rotational_curvatures_.clear();
        matrix_free_body_scales_.clear();
        matrix_free_body_elastic_valid_ = false;
        block_jacobi_contact_ell_width_ = 0;
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
    delta_.copy_to_device(s);
    mass_.copy_to_device(s);
    fixed_.copy_to_device(s);
    tets_.copy_to_device(s);
    tet_arap_effective_inertias_.copy_to_device(s);
    rotations_.copy_to_device(s);
    ground_contacts_.copy_to_device(s);
    if (interpolated_surface_) {
        surface_maps_dev_.copy_to_device(s);
        surface_triangles_dev_.copy_to_device(s);
        surface_edges_dev_.copy_to_device(s);
        interpolated_surface_positions_dev_.copy_to_device(s);
        interpolated_surface_min_y_dev_.copy_to_device(s);
        interpolated_wide_contacts_.copy_to_device(s);
        interpolated_wide_contact_count_.copy_to_device(s);
        block_jacobi_contact_ell_overflow_.copy_to_device(s);
        if (pcg_body_preconditioner_valid_) {
            pcg_body_preconditioner_rows_.copy_to_device(s);
            pcg_body_preconditioner_failure_count_.copy_to_device(s);
        }
        if (matrix_free_body_elastic_valid_) {
            matrix_free_body_gradients_.copy_to_device(s);
            matrix_free_body_rotations_.copy_to_device(s);
            matrix_free_body_rotational_curvatures_.copy_to_device(s);
            matrix_free_body_scales_.copy_to_device(s);
        }
        tet_mass_blocks_dev_.copy_to_device(s);
        tet_contact_inverse_mass_blocks_.copy_to_device(s);
        if (endpoint_hinges_dev_.gpu_size() > 0) {
            endpoint_hinges_dev_.copy_to_device(s);
            endpoint_hinge_diagnostics_dev_.copy_to_device(s);
        }
        interpolated_debug_counts_.copy_to_device(s);
        interpolated_contact_metrics_.copy_to_device(s);
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
    const math::Vec3f* controls_dev,
    bool update_min_y) {
    if (!interpolated_surface_ || num_surface_vertices_ <= 0 ||
        controls_dev == nullptr) {
        return;
    }

    auto* stream = reinterpret_cast<cudaStream_t>(stream_);
    constexpr int block = 128;
    const int grid = (num_surface_vertices_ + block - 1) / block;
    float* min_y_dev = nullptr;
    if (update_min_y && interpolated_surface_min_y_dev_.gpu_size() > 0) {
        interpolated_surface_min_y_dev_[0] = 3.402823466e38f;
        interpolated_surface_min_y_dev_.copy_to_device(stream_handle(stream_));
        min_y_dev = interpolated_surface_min_y_dev_.gpu_data();
    }
    update_interpolated_surface_positions_kernel<<<grid, block, 0, stream>>>(
        surface_maps_dev_.gpu_data(),
        num_surface_vertices_,
        controls_dev,
        interpolated_surface_positions_dev_.gpu_data(),
        min_y_dev);
    check_cuda(cudaGetLastError(),
               "update_interpolated_surface_positions_kernel");
}

void PabdCudaSolver::detect_interpolated_contacts_gpu() {
    if (!interpolated_surface_ ||
        params_.self_collision_thickness <= 0.0f ||
        !self_contact_enabled(params_) ||
        interpolated_contact_capacity_ <= 0) {
        last_self_broadphase_pairs_ = 0;
        last_self_broadphase_capacity_ = 0;
        last_self_broadphase_overflow_ = false;
        last_mesh_broadphase_refreshed_ = false;
        mesh_broadphase_cache_age_ = 0;
        mesh_broadphase_refresh_count_ = 0;
        last_mesh_broadphase_max_displacement_ = 0.0f;
        last_mesh_broadphase_dropped_hits_ = 0;
        last_self_point_face_contacts_ = 0;
        last_self_edge_edge_contacts_ = 0;
        last_self_friction_contacts_ = 0;
        return;
    }

    const std::uintptr_t s = stream_handle(stream_);
    mesh_mesh_contact_detector_.configure_broadphase_cache(
        params_.mesh_broadphase_interval,
        params_.mesh_broadphase_skin);
    mesh_mesh_contact_detector_.detect_gpu(
        interpolated_surface_positions_dev_.gpu_data(),
        params_.self_collision_thickness, s);
    last_self_broadphase_capacity_ =
        mesh_mesh_contact_detector_.max_ef_candidates();
}

void PabdCudaSolver::assemble_interpolated_system_gpu(float h) {
    const std::uintptr_t s = stream_handle(stream_);
    auto* stream = reinterpret_cast<cudaStream_t>(stream_);
    constexpr int block = 128;
    const bool body_owned_contacts =
        params_.global_solver == PabdGlobalSolverMode::BlockJacobi12;
    const bool matrix_free_polar_gn = matrix_free_polar_gn_active();
    const float inv_h2 = 1.0f / std::max(h * h, 1.0e-8f);
    const float* contact_inverse_mass_blocks = params_.use_contact_beta
        ? tet_contact_inverse_mass_blocks_.gpu_data()
        : nullptr;
    const int contact_body_count = static_cast<int>(tets_.gpu_size());
    math::Vec3f* contact_rhs = nullptr;

    if (!body_owned_contacts) {
        H_.set_zero(s);
        check_cuda(cudaMemsetAsync(rhs_.gpu_data(), 0,
                                   rhs_.gpu_size() * sizeof(math::Vec3f),
                                   stream),
                   "interpolated rhs memset");
    }
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
    check_cuda(cudaMemsetAsync(interpolated_contact_metrics_.gpu_data(), 0,
                               interpolated_contact_metrics_.gpu_size() *
                                   sizeof(float),
                               stream),
               "interpolated contact metrics memset");
    check_cuda(cudaMemsetAsync(interpolated_normal_sum_.gpu_data(), 0,
                               sizeof(math::Vec3f),
                               stream),
               "interpolated normal sum memset");

    update_interpolated_surface_positions_gpu(x_.gpu_data());

    if (!body_owned_contacts) {
        const float* arap_effective_inertias = params_.use_arap_beta
            ? tet_arap_effective_inertias_.gpu_data()
            : nullptr;
        const float arap_beta_inv_h2 = params_.arap_beta * inv_h2;
        const int n_tets = static_cast<int>(tets_.gpu_size());
        if (n_tets > 0) {
            const int grid = (n_tets + block - 1) / block;
            if (matrix_free_polar_gn) {
                update_matrix_free_polar_gn_state_kernel
                    <<<grid, block, 0, stream>>>(
                        x_.gpu_data(), tets_.gpu_data(),
                        arap_effective_inertias, n_tets,
                        params_.stiffness, arap_beta_inv_h2,
                        matrix_free_body_rotations_.gpu_data(),
                        matrix_free_body_rotational_curvatures_.gpu_data(),
                        matrix_free_body_scales_.gpu_data());
                check_cuda(cudaGetLastError(),
                           "update_matrix_free_polar_gn_state_kernel");
            }
            assemble_interpolated_mass_elastic_kernel<<<grid, block, 0, stream>>>(
                H_.diag.gpu_data(),
                H_.values.gpu_data(),
                rhs_.gpu_data(),
                tets_.gpu_data(),
                tet_mass_blocks_dev_.gpu_data(),
                arap_effective_inertias,
                n_tets,
                y_.gpu_data(),
                x_.gpu_data(),
                inv_h2,
                params_.stiffness,
                arap_beta_inv_h2,
                params_.elastic_curvature,
                matrix_free_polar_gn
                    ? matrix_free_body_rotations_.gpu_data()
                    : nullptr,
                !matrix_free_polar_gn);
            check_cuda(cudaGetLastError(),
                       "assemble_interpolated_mass_elastic_kernel");
            const float hinge_beta_inv_h2 = params_.hinge_beta * inv_h2;
            if (hinge_beta_inv_h2 > 0.0f &&
                endpoint_hinges_dev_.gpu_size() > 0) {
                assemble_interpolated_endpoint_hinge_kernel
                    <<<grid, block, 0, stream>>>(
                        H_.diag.gpu_data(),
                        H_.values.gpu_data(),
                        rhs_.gpu_data(),
                        tets_.gpu_data(),
                        endpoint_hinges_dev_.gpu_data(),
                        x_.gpu_data(),
                        n_tets,
                        hinge_beta_inv_h2);
                check_cuda(
                    cudaGetLastError(),
                    "assemble_interpolated_endpoint_hinge_kernel");
            }
        }

        const int vertex_grid = (num_vertices_ + block - 1) / block;
        assemble_interpolated_fixed_kernel<<<vertex_grid, block, 0, stream>>>(
            H_.diag.gpu_data(),
            rhs_.gpu_data(),
            rest_.gpu_data(),
            x_.gpu_data(),
            fixed_.gpu_data(),
            num_vertices_,
            params_.fixed_weight);
        check_cuda(cudaGetLastError(), "assemble_interpolated_fixed_kernel");
    }

    if (num_surface_vertices_ > 0) {
        const int surface_grid = (num_surface_vertices_ + block - 1) / block;
        append_interpolated_ground_contacts_kernel<<<surface_grid, block, 0, stream>>>(
            contact_rhs,
            surface_maps_dev_.gpu_data(),
            interpolated_surface_positions_dev_.gpu_data(),
            x_.gpu_data(),
            old_x_.gpu_data(),
            num_surface_vertices_,
            params_.ground_y + params_.contact_gap,
            params_.ground_stiffness,
            contact_inverse_mass_blocks,
            contact_body_count,
            params_.ground_contact_beta * inv_h2,
            params_.ground_friction,
            params_.friction_epsilon,
            ground_contacts_.gpu_data(),
            interpolated_wide_contacts_.gpu_data(),
            interpolated_wide_contact_count_.gpu_data(),
            static_cast<int>(interpolated_wide_contacts_.gpu_size()));
        check_cuda(cudaGetLastError(),
                   "append_interpolated_ground_contacts_kernel");
    }

    detect_interpolated_contacts_gpu();
    if (self_contact_enabled(params_) &&
        interpolated_contact_capacity_ > 0 &&
        mesh_mesh_contact_detector_.valid()) {
        const int contact_grid =
            bounded_count_grid(interpolated_contact_capacity_, block);
        append_interpolated_self_contacts_kernel<<<contact_grid, block, 0, stream>>>(
            contact_rhs,
            mesh_mesh_contact_detector_.contacts().gpu_data(),
            mesh_mesh_contact_detector_.count_array().gpu_data(),
            interpolated_contact_capacity_,
            surface_maps_dev_.gpu_data(),
            interpolated_surface_positions_dev_.gpu_data(),
            x_.gpu_data(),
            old_x_.gpu_data(),
            contact_inverse_mass_blocks,
            contact_body_count,
            params_.contact_gap,
            params_.self_collision_stiffness,
            params_.density,
            params_.self_collision_beta * inv_h2,
            params_.use_contact_beta,
            params_.self_contact_measure,
            params_.point_face_stiffness_scale,
            params_.enable_point_face_contacts,
            params_.enable_edge_edge_contacts,
            params_.self_collision_normal_damping,
            params_.self_collision_friction,
            params_.friction_epsilon,
            interpolated_debug_counts_.gpu_data(),
            interpolated_contact_metrics_.gpu_data(),
            interpolated_normal_sum_.gpu_data(),
            interpolated_wide_contacts_.gpu_data(),
            interpolated_wide_contact_count_.gpu_data(),
            static_cast<int>(interpolated_wide_contacts_.gpu_size()));
        check_cuda(cudaGetLastError(),
                   "append_interpolated_self_contacts_kernel");
    }

    if (!body_owned_contacts) {
        const int wide_capacity =
            static_cast<int>(interpolated_wide_contacts_.gpu_size());
        if (wide_capacity > 0) {
            const int contact_grid = bounded_count_grid(wide_capacity, block);
            assemble_interpolated_wide_contact_residual_kernel
                <<<contact_grid, block, 0, stream>>>(
                    rhs_.gpu_data(),
                    interpolated_wide_contacts_.gpu_data(),
                    interpolated_wide_contact_count_.gpu_data(),
                    wide_capacity,
                    x_.gpu_data());
            check_cuda(
                cudaGetLastError(),
                "assemble_interpolated_wide_contact_residual_kernel");
        }
        collision::WideContactSpMVOp wide_op;
        wide_op.contacts = interpolated_wide_contacts_.gpu_data();
        wide_op.count_dev = interpolated_wide_contact_count_.gpu_data();
        wide_op.max_contacts = wide_capacity;
        wide_op.stiffness = 1.0f;
        collision::bake_wide_contact_diag(H_.diag.gpu_data(), num_vertices_,
                                          wide_op, 1.0f, s);
    }
}

void PabdCudaSolver::update_block_jacobi_base_k_gpu(
    float h,
    bool local_rest_frame) {
    const int n_tets = static_cast<int>(tets_.gpu_size());
    if (n_tets <= 0) return;

    const float inv_h2 = 1.0f / std::max(h * h, 1.0e-8f);
    const PabdElasticCurvatureMode effective_curvature = local_rest_frame
        ? PabdElasticCurvatureMode::CorotatedRest
        : params_.elastic_curvature;
    const bool state_independent = local_rest_frame ||
        params_.elastic_curvature ==
        PabdElasticCurvatureMode::ProjectiveDynamics;
    if (state_independent && block_jacobi_base_k_valid_ &&
        block_jacobi_base_inv_h2_ == inv_h2 &&
        block_jacobi_base_stiffness_ == params_.stiffness &&
        block_jacobi_base_arap_beta_ == params_.arap_beta &&
        block_jacobi_base_use_arap_beta_ == params_.use_arap_beta &&
        block_jacobi_base_fixed_weight_ == params_.fixed_weight &&
        block_jacobi_base_hinge_beta_ == params_.hinge_beta &&
        block_jacobi_base_curvature_ == effective_curvature &&
        block_jacobi_base_local_rest_frame_ == local_rest_frame) {
        return;
    }

    auto* stream = reinterpret_cast<cudaStream_t>(stream_);
    constexpr int block = 128;
    const int grid = (n_tets + block - 1) / block;
    rebuild_block_jacobi_base_k_kernel<<<grid, block, 0, stream>>>(
        tets_.gpu_data(),
        tet_mass_blocks_dev_.gpu_data(),
        params_.use_arap_beta
            ? tet_arap_effective_inertias_.gpu_data()
            : nullptr,
        endpoint_hinges_dev_.gpu_data(),
        fixed_.gpu_data(),
        x_.gpu_data(),
        n_tets,
        inv_h2,
        params_.stiffness,
        params_.arap_beta * inv_h2,
        params_.fixed_weight,
        params_.hinge_beta * inv_h2,
        effective_curvature,
        local_rest_frame,
        block_jacobi_base_k_.gpu_data());
    check_cuda(cudaGetLastError(), "rebuild_block_jacobi_base_k_kernel");
    block_jacobi_base_inv_h2_ = inv_h2;
    block_jacobi_base_stiffness_ = params_.stiffness;
    block_jacobi_base_arap_beta_ = params_.arap_beta;
    block_jacobi_base_use_arap_beta_ = params_.use_arap_beta;
    block_jacobi_base_fixed_weight_ = params_.fixed_weight;
    block_jacobi_base_hinge_beta_ = params_.hinge_beta;
    block_jacobi_base_curvature_ = effective_curvature;
    block_jacobi_base_local_rest_frame_ = local_rest_frame;
    block_jacobi_base_k_valid_ = true;
}

void PabdCudaSolver::build_body_contact_ell_gpu() {
    auto* stream = reinterpret_cast<cudaStream_t>(stream_);
    const int n_bodies = static_cast<int>(tets_.gpu_size());
    if (n_bodies <= 0 || block_jacobi_contact_ell_width_ <= 0) return;

    check_cuda(cudaMemsetAsync(
                   block_jacobi_contact_ell_counts_.gpu_data(), 0,
                   block_jacobi_contact_ell_counts_.gpu_size() * sizeof(int),
                   stream),
               "body contact ELL count memset");
    check_cuda(cudaMemsetAsync(
                   block_jacobi_contact_ell_overflow_.gpu_data(), 0,
                   sizeof(int), stream),
               "body contact ELL overflow memset");

    const int wide_capacity =
        static_cast<int>(interpolated_wide_contacts_.gpu_size());
    if (wide_capacity <= 0) return;
    constexpr int block = 128;
    const int contact_grid = bounded_count_grid(wide_capacity, block);
    build_body_contact_ell_kernel<<<contact_grid, block, 0, stream>>>(
        interpolated_wide_contacts_.gpu_data(),
        interpolated_wide_contact_count_.gpu_data(),
        wide_capacity,
        n_bodies,
        block_jacobi_contact_ell_width_,
        block_jacobi_contact_ell_counts_.gpu_data(),
        block_jacobi_contact_ell_refs_.gpu_data(),
        block_jacobi_contact_ell_overflow_.gpu_data());
    check_cuda(cudaGetLastError(), "build_body_contact_ell_kernel");
}

void PabdCudaSolver::build_interpolated_pcg_body_preconditioner_gpu(float h) {
    if (!params_.pcg_body_preconditioner ||
        !pcg_body_preconditioner_valid_) return;
    const int n_bodies = static_cast<int>(tets_.gpu_size());
    if (n_bodies <= 0) return;

    const bool local_rest_frame = matrix_free_polar_gn_active();
    update_block_jacobi_base_k_gpu(h, local_rest_frame);
    build_body_contact_ell_gpu();
    auto* stream = reinterpret_cast<cudaStream_t>(stream_);
    check_cuda(cudaMemsetAsync(
                   pcg_body_preconditioner_failure_count_.gpu_data(), 0,
                   sizeof(int), stream),
               "PCG body preconditioner failure count memset");

    constexpr int block = 128;
    const int grid = (n_bodies + block - 1) / block;
    build_interpolated_body_preconditioner_kernel<<<grid, block, 0, stream>>>(
        n_bodies,
        num_vertices_,
        block_jacobi_base_k_.gpu_data(),
        fixed_.gpu_data(),
        interpolated_wide_contacts_.gpu_data(),
        block_jacobi_contact_ell_counts_.gpu_data(),
        block_jacobi_contact_ell_refs_.gpu_data(),
        block_jacobi_contact_ell_width_,
        local_rest_frame ? matrix_free_body_rotations_.gpu_data() : nullptr,
        local_rest_frame ? matrix_free_body_gradients_.gpu_data() : nullptr,
        local_rest_frame
            ? matrix_free_body_rotational_curvatures_.gpu_data()
            : nullptr,
        local_rest_frame ? matrix_free_body_scales_.gpu_data() : nullptr,
        pcg_body_preconditioner_lower_.gpu_data(),
        pcg_body_preconditioner_failure_count_.gpu_data());
    check_cuda(cudaGetLastError(),
               "build_interpolated_body_preconditioner_kernel");
}

void PabdCudaSolver::solve_interpolated_block_jacobi_gpu(float h,
                                                          float omega) {
    auto* stream = reinterpret_cast<cudaStream_t>(stream_);
    constexpr int block = 128;
    const int n_tets = static_cast<int>(tets_.gpu_size());
    if (n_tets <= 0) return;

    update_block_jacobi_base_k_gpu(h);
    check_cuda(cudaMemsetAsync(block_jacobi_failure_count_.gpu_data(), 0,
                               sizeof(int), stream),
               "block jacobi failure count memset");
    check_cuda(cudaMemcpyAsync(block_jacobi_x_lagged_.gpu_data(), x_.gpu_data(),
                               num_vertices_ * sizeof(math::Vec3f),
                               cudaMemcpyDeviceToDevice, stream),
               "block jacobi lagged x copy");

    build_body_contact_ell_gpu();

    const int body_grid = (n_tets + block - 1) / block;
    solve_interpolated_block_jacobi_kernel<<<body_grid, block, 0, stream>>>(
        tets_.gpu_data(),
        n_tets,
        block_jacobi_base_k_.gpu_data(),
        tet_mass_blocks_dev_.gpu_data(),
        params_.use_arap_beta
            ? tet_arap_effective_inertias_.gpu_data()
            : nullptr,
        endpoint_hinges_dev_.gpu_data(),
        y_.gpu_data(),
        rest_.gpu_data(),
        interpolated_wide_contacts_.gpu_data(),
        block_jacobi_contact_ell_counts_.gpu_data(),
        block_jacobi_contact_ell_refs_.gpu_data(),
        block_jacobi_contact_ell_width_,
        num_vertices_,
        block_jacobi_x_lagged_.gpu_data(),
        fixed_.gpu_data(),
        1.0f / std::max(h * h, 1.0e-8f),
        params_.stiffness,
        params_.arap_beta / std::max(h * h, 1.0e-8f),
        params_.fixed_weight,
        params_.hinge_beta / std::max(h * h, 1.0e-8f),
        x_.gpu_data(),
        omega,
        block_jacobi_failure_count_.gpu_data());
    check_cuda(cudaGetLastError(), "solve_interpolated_block_jacobi_kernel");
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
    const int n_tets = static_cast<int>(tets_.gpu_size());
    const int body_grid = (n_tets + block - 1) / block;
    for (int sub = 0; sub < substeps; ++sub) {
        const float h = sub_dt;
        const float inv_h = 1.0f / std::max(h, 1.0e-8f);

        if (n_tets > 0 && endpoint_hinges_dev_.gpu_size() > 0 &&
            (params_.motor_torque != 0.0f ||
             params_.motor_damping != 0.0f)) {
            apply_endpoint_hinge_motor_kernel
                <<<body_grid, block, 0, stream>>>(
                    velocity_.gpu_data(),
                    x_.gpu_data(),
                    tets_.gpu_data(),
                    tet_mass_blocks_dev_.gpu_data(),
                    endpoint_hinges_dev_.gpu_data(),
                    n_tets,
                    params_.motor_torque,
                    params_.motor_damping,
                    h);
            check_cuda(cudaGetLastError(),
                       "apply_endpoint_hinge_motor_kernel");
        }

        prepare_interpolated_step_kernel<<<vertex_grid, block, 0, stream>>>(
            rest_.gpu_data(), x_.gpu_data(), old_x_.gpu_data(),
            y_.gpu_data(), velocity_.gpu_data(), fixed_.gpu_data(),
            num_vertices_, h, params_.gravity, params_.damping);
        check_cuda(cudaGetLastError(), "prepare_interpolated_step_kernel");

        solver::PCGParams pcg_params;
        pcg_params.max_iterations = params_.pcg_iterations;
        pcg_params.compute_true_residual =
            params_.pcg_true_residual_diagnostics;
        const int iterations = std::max(1, params_.iterations);
        for (int iter = 0; iter < iterations; ++iter) {
            assemble_interpolated_system_gpu(h);
            if (params_.global_solver == PabdGlobalSolverMode::BlockJacobi12) {
                solve_interpolated_block_jacobi_gpu(
                    h, params_.block_jacobi_omega);
                last_pcg_iterations_ = 0;
            } else {
                check_cuda(cudaMemsetAsync(
                               delta_.gpu_data(), 0,
                               delta_.gpu_size() * sizeof(math::Vec3f), stream),
                           "interpolated delta memset");
                build_interpolated_pcg_body_preconditioner_gpu(h);
                collision::WideContactSpMVOp wide_op{
                    interpolated_wide_contacts_.gpu_data(),
                    interpolated_wide_contact_count_.gpu_data(),
                    static_cast<int>(interpolated_wide_contacts_.gpu_size()),
                    1.0f};
                solver::BodyBlock12PreconditionerOp body_preconditioner;
                const bool matrix_free_polar_gn =
                    matrix_free_polar_gn_active();
                if (params_.pcg_body_preconditioner &&
                    pcg_body_preconditioner_valid_) {
                    body_preconditioner.lower_factors =
                        pcg_body_preconditioner_lower_.gpu_data();
                    body_preconditioner.body_rows =
                        pcg_body_preconditioner_rows_.gpu_data();
                    body_preconditioner.fixed_rows = fixed_.gpu_data();
                    body_preconditioner.body_rotations =
                        matrix_free_polar_gn
                            ? matrix_free_body_rotations_.gpu_data()
                            : nullptr;
                    body_preconditioner.num_bodies =
                        static_cast<int>(tets_.gpu_size());
                }
                solver::BodyElasticSpMVOp body_elastic;
                if (matrix_free_polar_gn) {
                    body_elastic.body_rows =
                        pcg_body_preconditioner_rows_.gpu_data();
                    body_elastic.body_gradients =
                        matrix_free_body_gradients_.gpu_data();
                    body_elastic.body_rotations =
                        matrix_free_body_rotations_.gpu_data();
                    body_elastic.rotational_curvatures =
                        matrix_free_body_rotational_curvatures_.gpu_data();
                    body_elastic.body_scales =
                        matrix_free_body_scales_.gpu_data();
                    body_elastic.num_bodies = n_tets;
                }
                last_pcg_iterations_ = pcg_.solve(
                    H_, DeviceSpan<math::Vec3f>::from(rhs_),
                    DeviceSpan<math::Vec3f>::from(delta_),
                    pcg_params, stream_handle(stream_),
                    collision::ContactSpMVOp{},
                    wide_op,
                    body_preconditioner,
                    body_elastic);
                apply_increment_kernel<<<vertex_grid, block, 0, stream>>>(
                    x_.gpu_data(), delta_.gpu_data(), fixed_.gpu_data(),
                    num_vertices_);
                check_cuda(cudaGetLastError(),
                           "interpolated apply_increment_kernel");
            }
            apply_fixed_kernel<<<vertex_grid, block, 0, stream>>>(
                x_.gpu_data(), rest_.gpu_data(), fixed_.gpu_data(), num_vertices_);
            check_cuda(cudaGetLastError(), "interpolated apply_fixed_kernel");
        }

        update_velocity_kernel<<<vertex_grid, block, 0, stream>>>(
            velocity_.gpu_data(), x_.gpu_data(), old_x_.gpu_data(),
            fixed_.gpu_data(), num_vertices_, inv_h, params_.damping);
        check_cuda(cudaGetLastError(), "interpolated update_velocity_kernel");
    }

    if (n_tets > 0 && endpoint_hinges_dev_.gpu_size() > 0) {
        compute_endpoint_hinge_diagnostics_kernel
            <<<body_grid, block, 0, stream>>>(
                velocity_.gpu_data(),
                x_.gpu_data(),
                tets_.gpu_data(),
                tet_mass_blocks_dev_.gpu_data(),
                endpoint_hinges_dev_.gpu_data(),
                n_tets,
                endpoint_hinge_diagnostics_dev_.gpu_data());
        check_cuda(cudaGetLastError(),
                   "compute_endpoint_hinge_diagnostics_kernel");
    }

    if (params_.elastic_curvature_diagnostics && n_tets > 0) {
        check_cuda(cudaMemsetAsync(
                       elastic_curvature_diagnostics_dev_.gpu_data(), 0,
                       4 * sizeof(float), stream),
                   "elastic curvature diagnostics memset");
        compute_elastic_curvature_diagnostics_kernel
            <<<body_grid, block, 0, stream>>>(
                x_.gpu_data(), tets_.gpu_data(), n_tets,
                elastic_curvature_diagnostics_dev_.gpu_data());
        check_cuda(cudaGetLastError(),
                   "compute_elastic_curvature_diagnostics_kernel");
    }

    update_interpolated_surface_positions_gpu(x_.gpu_data(), true);

    if (params_.global_solver != PabdGlobalSolverMode::BlockJacobi12) {
        pcg_.copy_last_residual_to_host(stream_handle(stream_));
    }
    ground_contacts_.copy_to_host(stream_handle(stream_));
    block_jacobi_failure_count_.copy_to_host(stream_handle(stream_));
    block_jacobi_contact_ell_overflow_.copy_to_host(stream_handle(stream_));
    if (params_.global_solver != PabdGlobalSolverMode::BlockJacobi12 &&
        params_.pcg_body_preconditioner &&
        pcg_body_preconditioner_valid_) {
        pcg_body_preconditioner_failure_count_.copy_to_host(
            stream_handle(stream_));
    }
    interpolated_debug_counts_.copy_to_host(stream_handle(stream_));
    interpolated_contact_metrics_.copy_to_host(stream_handle(stream_));
    interpolated_normal_sum_.copy_to_host(stream_handle(stream_));
    interpolated_surface_min_y_dev_.copy_to_host(stream_handle(stream_));
    if (endpoint_hinge_diagnostics_dev_.gpu_size() > 0) {
        endpoint_hinge_diagnostics_dev_.copy_to_host(
            stream_handle(stream_));
    }
    if (params_.elastic_curvature_diagnostics) {
        elastic_curvature_diagnostics_dev_.copy_to_host(
            stream_handle(stream_));
    }
    if (mesh_mesh_contact_detector_.valid() && interpolated_contact_capacity_ > 0) {
        mesh_mesh_contact_detector_.count_array().copy_to_host(
            stream_handle(stream_));
    }
    if (auto_download_positions_) {
        x_.copy_to_host(stream_handle(stream_));
    }
    check_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)),
               "step_interpolated final sync");
    last_residual_ =
        params_.global_solver == PabdGlobalSolverMode::BlockJacobi12
            ? 0.0f
            : pcg_.host_last_residual();
    last_true_relative_residual_ =
        params_.global_solver == PabdGlobalSolverMode::BlockJacobi12
            ? -1.0f
            : pcg_.host_last_true_relative_residual();
    last_motor_axis_angular_velocity_ = 0.0f;
    last_hinge_endpoint_error_ = 0.0f;
    last_max_stretch_error_ = 0.0f;
    last_max_orthogonality_error_ = 0.0f;
    last_max_volume_error_ = 0.0f;
    last_max_rotational_curvature_ = 0.0f;
    std::fill(hinge_axis_angular_velocities_.begin(),
              hinge_axis_angular_velocities_.end(), 0.0f);
    std::fill(hinge_endpoint_errors_.begin(),
              hinge_endpoint_errors_.end(), 0.0f);
    for (std::size_t body = 0; body < host_endpoint_hinges_.size(); ++body) {
        if (host_endpoint_hinges_[body].body < 0 ||
            body >= endpoint_hinge_diagnostics_dev_.cpu_size()) {
            continue;
        }
        const math::Vec3f diagnostic =
            endpoint_hinge_diagnostics_dev_.cpu_data()[body];
        hinge_axis_angular_velocities_[body] = diagnostic.x;
        hinge_endpoint_errors_[body] = diagnostic.y;
        last_hinge_endpoint_error_ =
            std::max(last_hinge_endpoint_error_, diagnostic.y);
        if (host_endpoint_hinges_[body].motor != 0) {
            last_motor_axis_angular_velocity_ = diagnostic.x;
        }
    }
    if (params_.elastic_curvature_diagnostics) {
        last_max_stretch_error_ =
            elastic_curvature_diagnostics_dev_.cpu_data()[0];
        last_max_orthogonality_error_ =
            elastic_curvature_diagnostics_dev_.cpu_data()[1];
        last_max_volume_error_ =
            elastic_curvature_diagnostics_dev_.cpu_data()[2];
        last_max_rotational_curvature_ =
            elastic_curvature_diagnostics_dev_.cpu_data()[3];
    }
    min_y_ = interpolated_surface_min_y_dev_.cpu_data()[0];
    last_ground_contacts_ = ground_contacts_.cpu_data()[0];
    last_block_jacobi_failures_ =
        block_jacobi_failure_count_.cpu_data()[0];
    last_block_jacobi_contact_ell_overflow_ =
        block_jacobi_contact_ell_overflow_.cpu_data()[0];
    last_pcg_body_preconditioner_failures_ =
        params_.global_solver != PabdGlobalSolverMode::BlockJacobi12 &&
                params_.pcg_body_preconditioner &&
                pcg_body_preconditioner_valid_
            ? pcg_body_preconditioner_failure_count_.cpu_data()[0]
            : 0;
    if (last_block_jacobi_contact_ell_overflow_ > 0) {
        std::fprintf(
            stderr,
            "[PABD_ERROR] body-contact ELL overflow: "
            "dropped_refs=%d width=%d. Increase the ELL width policy.\n",
            last_block_jacobi_contact_ell_overflow_,
            block_jacobi_contact_ell_width_);
    }
    if (last_pcg_body_preconditioner_failures_ > 0) {
        std::fprintf(
            stderr,
            "[PABD_ERROR] PCG 12x12 body preconditioner Cholesky failures=%d\n",
            last_pcg_body_preconditioner_failures_);
    }
    last_self_contacts_ = interpolated_debug_counts_.cpu_data()[0];
    last_self_vertical_contacts_ = interpolated_debug_counts_.cpu_data()[1];
    last_self_horizontal_contacts_ = interpolated_debug_counts_.cpu_data()[2];
    last_self_point_face_contacts_ = interpolated_debug_counts_.cpu_data()[3];
    last_self_edge_edge_contacts_ = interpolated_debug_counts_.cpu_data()[4];
    last_self_friction_contacts_ = interpolated_debug_counts_.cpu_data()[5];
    last_self_active_point_face_contacts_ =
        interpolated_debug_counts_.cpu_data()[6];
    last_self_active_edge_edge_contacts_ =
        interpolated_debug_counts_.cpu_data()[7];
    const float* contact_metrics = interpolated_contact_metrics_.cpu_data();
    const int active_contacts = std::max(0, last_self_contacts_);
    const int active_point_face =
        std::max(0, last_self_active_point_face_contacts_);
    const int active_edge_edge =
        std::max(0, last_self_active_edge_edge_contacts_);
    last_self_max_penetration_ = contact_metrics[0];
    last_self_rms_penetration_ = active_contacts > 0
        ? std::sqrt(std::max(0.0f, contact_metrics[1] /
            static_cast<float>(active_contacts)))
        : 0.0f;
    last_self_mean_stiffness_ = active_contacts > 0
        ? contact_metrics[2] / static_cast<float>(active_contacts)
        : 0.0f;
    last_self_point_face_mean_stiffness_ = active_point_face > 0
        ? contact_metrics[3] / static_cast<float>(active_point_face)
        : 0.0f;
    last_self_edge_edge_mean_stiffness_ = active_edge_edge > 0
        ? contact_metrics[4] / static_cast<float>(active_edge_edge)
        : 0.0f;
    last_self_point_face_mean_volume_ = active_point_face > 0
        ? contact_metrics[5] / static_cast<float>(active_point_face)
        : 0.0f;
    last_self_edge_edge_mean_volume_ = active_edge_edge > 0
        ? contact_metrics[6] / static_cast<float>(active_edge_edge)
        : 0.0f;
    last_self_normal_sum_ = interpolated_normal_sum_.cpu_data()[0];
    last_self_broadphase_pairs_ =
        mesh_mesh_contact_detector_.valid()
            ? mesh_mesh_contact_detector_.last_ef_count()
            : 0;
    last_mesh_broadphase_refreshed_ =
        mesh_mesh_contact_detector_.last_broadphase_refreshed();
    mesh_broadphase_cache_age_ =
        mesh_mesh_contact_detector_.broadphase_cache_age();
    mesh_broadphase_refresh_count_ =
        mesh_mesh_contact_detector_.broadphase_refresh_count();
    last_mesh_broadphase_max_displacement_ =
        mesh_mesh_contact_detector_.last_broadphase_max_displacement();
    last_mesh_broadphase_dropped_hits_ =
        mesh_mesh_contact_detector_.last_broadphase_dropped_hits();
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
    if (last_mesh_broadphase_refreshed_ &&
        last_mesh_broadphase_dropped_hits_ > 0) {
        std::fprintf(
            stderr,
            "[PABD_ERROR] OptiX per-edge EF hit capacity overflow: "
            "dropped=%d. Increase max_hits_per_edge or reduce broadphase skin.\n",
            last_mesh_broadphase_dropped_hits_);
    }
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
    if (auto_download_positions_) {
        update_host_positions(true);
    }
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
        const float* arap_effective_inertias = params_.use_arap_beta
            ? tet_arap_effective_inertias_.gpu_data()
            : nullptr;
        const float arap_beta_inv_h2 = params_.arap_beta * inv_h2;

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
                self_contact_enabled(params_) &&
                mesh_topology_.valid();
            if (self_collision_active) {
                self_collision_.set_stiffness(params_.use_contact_beta
                    ? 1.0f
                    : params_.self_collision_stiffness);
                self_collision_detector_.detect(
                    DeviceSpan<math::Vec3f>::from(x_),
                    params_.self_collision_thickness,
                    stream_handle(stream_));
            }
            collision::ContactSpMVOp contact_op = self_collision_active
                ? self_collision_.make_spmv_op(self_collision_detector_)
                : collision::ContactSpMVOp{};
            if (self_collision_active && params_.use_contact_beta) {
                compute_particle_contact_stiffness_kernel<<<
                    (contact_op.max_contacts + block - 1) / block,
                    block, 0, stream>>>(
                    contact_op.pairs, contact_op.weights,
                    contact_op.count_dev, contact_op.max_contacts,
                    x_.gpu_data(), mass_.gpu_data(), fixed_.gpu_data(),
                    params_.self_collision_thickness,
                    params_.density,
                    params_.self_collision_beta * inv_h2,
                    params_.self_contact_measure,
                    particle_contact_stiffnesses_.gpu_data());
                contact_op.stiffnesses =
                    particle_contact_stiffnesses_.gpu_data();
            }

            H_.set_zero(stream_handle(stream_));
            check_cuda(cudaMemsetAsync(ground_contacts_.gpu_data(), 0, sizeof(int), stream),
                       "cudaMemsetAsync(ground_contacts)");
            assemble_base_matrix_kernel<<<vertex_grid, block, 0, stream>>>(
                H_.diag.gpu_data(), mass_.gpu_data(), fixed_.gpu_data(),
                num_vertices_, inv_h2, params_.fixed_weight);
            add_tet_matrix_kernel<<<(num_tets * 16 + block - 1) / block, block, 0, stream>>>(
                H_.diag.gpu_data(), H_.values.gpu_data(), tets_.gpu_data(),
                arap_effective_inertias, num_tets, params_.stiffness,
                arap_beta_inv_h2);
            add_ground_matrix_kernel<<<vertex_grid, block, 0, stream>>>(
                H_.diag.gpu_data(), x_.gpu_data(), mass_.gpu_data(),
                fixed_.gpu_data(),
                ground_contacts_.gpu_data(), num_vertices_, params_.ground_y,
                params_.contact_gap, params_.ground_stiffness,
                params_.ground_contact_beta * inv_h2,
                params_.use_contact_beta);
            collision::bake_contact_diag(
                H_.diag.gpu_data(), num_vertices_, contact_op, 1.0f,
                stream_handle(stream_));
            check_cuda(cudaGetLastError(), "matrix assembly kernels");

            assemble_base_rhs_kernel<<<vertex_grid, block, 0, stream>>>(
                rhs_.gpu_data(), rest_.gpu_data(), y_.gpu_data(), mass_.gpu_data(),
                fixed_.gpu_data(), num_vertices_, inv_h2, params_.fixed_weight);
            add_tet_rhs_kernel<<<(num_tets * 4 + block - 1) / block, block, 0, stream>>>(
                rhs_.gpu_data(), tets_.gpu_data(), rotations_.gpu_data(),
                arap_effective_inertias, num_tets, params_.stiffness,
                arap_beta_inv_h2);
            add_ground_rhs_kernel<<<vertex_grid, block, 0, stream>>>(
                rhs_.gpu_data(), x_.gpu_data(), mass_.gpu_data(),
                fixed_.gpu_data(), num_vertices_, params_.ground_y,
                params_.contact_gap, params_.ground_stiffness,
                params_.ground_contact_beta * inv_h2,
                params_.use_contact_beta);
            if (self_collision_active) {
                add_self_contact_rhs_kernel<<<
                    (contact_op.max_contacts + block - 1) / block,
                    block, 0, stream>>>(
                    contact_op.pairs, contact_op.weights, contact_op.count_dev,
                    contact_op.max_contacts, contact_op.stiffness,
                    contact_op.stiffnesses,
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
         self_contact_enabled(params_) &&
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

PabdCudaMesh make_stacked_blocks_mesh(int count,
                                      float half_extent,
                                      float ground_y,
                                      float block_gap) {
    // Keep the legacy scene name, but use exactly the same OBJ surface,
    // four-control ABD map, and inter-object PF/EE detector as PD+ABD Boxes.
    return make_pd_abd_boxes_mesh(
        count, ground_y, block_gap, half_extent, 0.0f);
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
