// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "rigid/pabd_cuda/pabd_elastic_curvature.cuh"

namespace {

using chysx::math::Mat3d;
using chysx::math::Mat3f;
using chysx::math::Vec3f;
using chysx::rigid::pabd_cuda::PabdElasticCurvatureMode;
using chysx::rigid::pabd_cuda::apply_local_elastic_curvature;
using chysx::rigid::pabd_cuda::apply_world_elastic_curvature;
using chysx::rigid::pabd_cuda::elastic_curvature_mode_name;

constexpr int kMatrixDofs = 9;

void check_cuda(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " +
                                 cudaGetErrorString(error));
    }
}

Mat3d polar_rotation(Mat3d f) {
    Mat3d r = f;
    for (int i = 0; i < 30; ++i) {
        r = (r + chysx::math::transpose(chysx::math::inverse(r))) * 0.5;
    }
    return r;
}

Mat3d matrix_basis(int column) {
    Mat3d result = Mat3d::zero();
    result(column / 3, column % 3) = 1.0;
    return result;
}

Mat3d exact_arap_hvp(const Mat3d& delta_f,
                     const Mat3d& rotation,
                     const Mat3d& stretch) {
    const Mat3d local = chysx::math::transpose(rotation) * delta_f;
    const Mat3d symmetric =
        (local + chysx::math::transpose(local)) * 0.5;
    const std::array<double, 3> skew = {
        0.5 * (local(2, 1) - local(1, 2)),
        0.5 * (local(0, 2) - local(2, 0)),
        0.5 * (local(1, 0) - local(0, 1)),
    };

    const double tr = chysx::math::trace(stretch);
    Mat3d l = Mat3d::identity() * tr - stretch;
    const Mat3d g = Mat3d::identity() - chysx::math::inverse(l) * 2.0;
    std::array<double, 3> rotated_skew{};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            rotated_skew[row] += g(row, col) * skew[col];
        }
    }
    const Mat3d skew_result(
        0.0, -rotated_skew[2], rotated_skew[1],
        rotated_skew[2], 0.0, -rotated_skew[0],
        -rotated_skew[1], rotated_skew[0], 0.0);
    return rotation * (symmetric + skew_result);
}

using Matrix9 = std::array<std::array<double, kMatrixDofs>, kMatrixDofs>;

Matrix9 zero_matrix9() {
    Matrix9 result{};
    for (auto& row : result) row.fill(0.0);
    return result;
}

Matrix9 analytic_exact_hessian(const Mat3d& f) {
    const Mat3d r = polar_rotation(f);
    const Mat3d s = chysx::math::transpose(r) * f;
    Matrix9 h = zero_matrix9();
    for (int col = 0; col < kMatrixDofs; ++col) {
        const Mat3d value = exact_arap_hvp(matrix_basis(col), r, s);
        for (int row = 0; row < kMatrixDofs; ++row) {
            h[row][col] = value.data[row];
        }
    }
    for (int row = 0; row < kMatrixDofs; ++row) {
        for (int col = row + 1; col < kMatrixDofs; ++col) {
            const double value = 0.5 * (h[row][col] + h[col][row]);
            h[row][col] = value;
            h[col][row] = value;
        }
    }
    return h;
}

Matrix9 finite_difference_hessian(const Mat3d& f) {
    constexpr double epsilon = 1.0e-5;
    Matrix9 h = zero_matrix9();
    for (int col = 0; col < kMatrixDofs; ++col) {
        const Mat3d direction = matrix_basis(col);
        const Mat3d f_plus = f + direction * epsilon;
        const Mat3d f_minus = f - direction * epsilon;
        const Mat3d g_plus = f_plus - polar_rotation(f_plus);
        const Mat3d g_minus = f_minus - polar_rotation(f_minus);
        const Mat3d derivative = (g_plus - g_minus) * (0.5 / epsilon);
        for (int row = 0; row < kMatrixDofs; ++row) {
            h[row][col] = derivative.data[row];
        }
    }
    return h;
}

double relative_matrix_error(const Matrix9& a, const Matrix9& b) {
    double diff = 0.0;
    double denom = 0.0;
    for (int row = 0; row < kMatrixDofs; ++row) {
        for (int col = 0; col < kMatrixDofs; ++col) {
            const double d = a[row][col] - b[row][col];
            diff += d * d;
            denom += b[row][col] * b[row][col];
        }
    }
    return std::sqrt(diff / std::max(denom, 1.0e-30));
}

Mat3f rotation_z(float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return Mat3f(c, -s, 0.0f,
                 s, c, 0.0f,
                 0.0f, 0.0f, 1.0f);
}

Mat3f make_stretch(float sx, float sy, float sz, float angle) {
    const Mat3f v = rotation_z(angle);
    const Mat3f diagonal(sx, 0.0f, 0.0f,
                         0.0f, sy, 0.0f,
                         0.0f, 0.0f, sz);
    return v * diagonal * chysx::math::transpose(v);
}

void validate_body_matrix_free_equivalence() {
    const Mat3f rotation = rotation_z(0.43f);
    const Mat3f stretch = make_stretch(0.72f, 1.08f, 1.31f, 0.27f);
    const std::array<Vec3f, 4> gradients = {{
        Vec3f(-1.1f, -0.7f, -0.5f),
        Vec3f(1.1f, 0.0f, 0.0f),
        Vec3f(0.0f, 0.7f, 0.0f),
        Vec3f(0.0f, 0.0f, 0.5f),
    }};
    const std::array<Vec3f, 4> values = {{
        Vec3f(0.2f, -0.3f, 0.7f),
        Vec3f(-0.6f, 0.4f, 0.1f),
        Vec3f(0.9f, -0.2f, -0.5f),
        Vec3f(-0.1f, 0.8f, -0.4f),
    }};
    constexpr float scale = 3.7f;

    Mat3f delta_f = Mat3f::zero();
    for (int b = 0; b < 4; ++b) {
        delta_f += chysx::math::outer(values[b], gradients[b]);
    }
    const Mat3f matrix_free_response = apply_world_elastic_curvature(
        delta_f, rotation, stretch,
        PabdElasticCurvatureMode::PolarGaussNewton) * scale;

    double error2 = 0.0;
    double reference2 = 0.0;
    for (int a = 0; a < 4; ++a) {
        Vec3f assembled(0.0f, 0.0f, 0.0f);
        for (int b = 0; b < 4; ++b) {
            Mat3f block = Mat3f::zero();
            for (int axis = 0; axis < 3; ++axis) {
                Vec3f basis(0.0f, 0.0f, 0.0f);
                basis[axis] = 1.0f;
                const Mat3f response = apply_world_elastic_curvature(
                    chysx::math::outer(basis, gradients[b]),
                    rotation, stretch,
                    PabdElasticCurvatureMode::PolarGaussNewton) * scale;
                const Vec3f column = response * gradients[a];
                block(0, axis) = column.x;
                block(1, axis) = column.y;
                block(2, axis) = column.z;
            }
            assembled += block * values[b];
        }
        const Vec3f matrix_free = matrix_free_response * gradients[a];
        const Vec3f difference = assembled - matrix_free;
        error2 += static_cast<double>(
            chysx::math::dot(difference, difference));
        reference2 += static_cast<double>(
            chysx::math::dot(assembled, assembled));
    }
    const double relative_error =
        std::sqrt(error2 / std::max(reference2, 1.0e-30));
    std::cout << "body_matrix_free_relative_error=" << std::scientific
              << relative_error << '\n';
    if (!(relative_error < 2.0e-6)) {
        throw std::runtime_error(
            "matrix-free body curvature does not match assembled blocks");
    }
}

double operator_error(const Mat3f& stretch,
                      PabdElasticCurvatureMode mode,
                      PabdElasticCurvatureMode reference) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (int column = 0; column < kMatrixDofs; ++column) {
        Mat3f basis = Mat3f::zero();
        basis.data[column] = 1.0f;
        const Mat3f value = apply_local_elastic_curvature(basis, stretch, mode);
        const Mat3f target =
            apply_local_elastic_curvature(basis, stretch, reference);
        for (int i = 0; i < kMatrixDofs; ++i) {
            const double delta = static_cast<double>(value.data[i]) -
                                 static_cast<double>(target.data[i]);
            numerator += delta * delta;
            denominator += static_cast<double>(target.data[i]) * target.data[i];
        }
    }
    return std::sqrt(numerator / std::max(denominator, 1.0e-30));
}

void run_spectrum_experiment() {
    std::cout << "[CURVATURE validation]\n";
    const Mat3d rotation(0.921060994, -0.389418342, 0.0,
                         0.389418342, 0.921060994, 0.0,
                         0.0, 0.0, 1.0);
    const Mat3d axis_rotation(0.955336489, 0.0, 0.295520207,
                             0.0, 1.0, 0.0,
                             -0.295520207, 0.0, 0.955336489);
    const Mat3d diagonal(0.72, 0.0, 0.0,
                         0.0, 1.08, 0.0,
                         0.0, 0.0, 1.31);
    const Mat3d stretch = axis_rotation * diagonal *
                          chysx::math::transpose(axis_rotation);
    const Mat3d f = rotation * stretch;
    const Matrix9 analytic = analytic_exact_hessian(f);
    const Matrix9 finite_difference = finite_difference_hessian(f);
    const double validation_error =
        relative_matrix_error(analytic, finite_difference);
    std::cout << "finite_difference_relative_error=" << std::scientific
              << validation_error << '\n';
    if (!(validation_error < 2.0e-5)) {
        throw std::runtime_error("analytic ARAP Hessian validation failed");
    }
    validate_body_matrix_free_equivalence();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "[CURVATURE spectrum] columns="
              << "sx,sy,sz,exact_min,exact_rot0,exact_rot1,exact_rot2,"
                 "positive_rot_rank,pd_rel_error,rest_rel_error,gn_rel_error\n";
    const std::array<std::array<float, 3>, 9> stretches = {{
        {{0.55f, 0.55f, 0.55f}},
        {{0.80f, 0.80f, 0.80f}},
        {{0.95f, 0.95f, 0.95f}},
        {{1.00f, 1.00f, 1.00f}},
        {{1.05f, 1.05f, 1.05f}},
        {{1.20f, 1.20f, 1.20f}},
        {{1.60f, 1.60f, 1.60f}},
        {{0.70f, 1.00f, 1.30f}},
        {{0.55f, 1.15f, 1.70f}},
    }};

    for (std::size_t sample = 0; sample < stretches.size(); ++sample) {
        const auto sigma = stretches[sample];
        const Mat3f s = make_stretch(sigma[0], sigma[1], sigma[2], 0.37f);
        const float rotational[3] = {
            1.0f - 2.0f / (sigma[1] + sigma[2]),
            1.0f - 2.0f / (sigma[0] + sigma[2]),
            1.0f - 2.0f / (sigma[0] + sigma[1]),
        };
        int positive_rank = 0;
        for (float value : rotational) positive_rank += value > 1.0e-6f;
        const float exact_min = std::min(
            1.0f, std::min(rotational[0],
                           std::min(rotational[1], rotational[2])));
        std::cout << "curvature_spectrum="
                  << sigma[0] << ',' << sigma[1] << ',' << sigma[2] << ','
                  << exact_min << ',' << rotational[0] << ','
                  << rotational[1] << ',' << rotational[2] << ','
                  << positive_rank << ','
                  << operator_error(s,
                       PabdElasticCurvatureMode::ProjectiveDynamics,
                       PabdElasticCurvatureMode::ProjectedNewton3) << ','
                  << operator_error(s,
                       PabdElasticCurvatureMode::CorotatedRest,
                       PabdElasticCurvatureMode::ProjectedNewton3) << ','
                  << operator_error(s,
                       PabdElasticCurvatureMode::PolarGaussNewton,
                       PabdElasticCurvatureMode::ProjectedNewton3) << '\n';
    }

    std::cout << "[CURVATURE stiffness] columns="
              << "stretch,stiffness,mode,local_condition,rotational_min,"
                 "rotational_max\n";
    const std::array<float, 3> stiffnesses = {{1.0e2f, 1.0e4f, 1.0e6f}};
    const std::array<float, 3> isotropic_stretches = {{0.8f, 1.0f, 1.2f}};
    const std::array<PabdElasticCurvatureMode, 4> modes = {{
        PabdElasticCurvatureMode::ProjectiveDynamics,
        PabdElasticCurvatureMode::CorotatedRest,
        PabdElasticCurvatureMode::PolarGaussNewton,
        PabdElasticCurvatureMode::ProjectedNewton3,
    }};
    for (float stretch_value : isotropic_stretches) {
        const float exact_rot = 1.0f - 1.0f / stretch_value;
        for (float stiffness : stiffnesses) {
            for (PabdElasticCurvatureMode mode : modes) {
                float rotational_min = 0.0f;
                float rotational_max = 0.0f;
                if (mode == PabdElasticCurvatureMode::ProjectiveDynamics) {
                    rotational_min = rotational_max = 1.0f;
                } else if (mode == PabdElasticCurvatureMode::PolarGaussNewton) {
                    rotational_min = rotational_max = exact_rot * exact_rot;
                } else if (mode == PabdElasticCurvatureMode::ProjectedNewton3) {
                    rotational_min = rotational_max = std::max(exact_rot, 0.0f);
                }
                const float smallest = std::min(1.0f, rotational_min);
                const float largest = std::max(1.0f, rotational_max);
                const double condition =
                    (1.0 + stiffness * largest) /
                    (1.0 + stiffness * smallest);
                std::cout << "curvature_stiffness="
                          << stretch_value << ',' << stiffness << ','
                          << elastic_curvature_mode_name(mode) << ','
                          << condition << ',' << rotational_min << ','
                          << rotational_max << '\n';
            }
        }
    }
}

struct CurvatureSample {
    Mat3f stretch;
    Mat3f delta_f;
};

__global__ void curvature_benchmark_kernel(
    const CurvatureSample* samples,
    Mat3f* output,
    int count,
    PabdElasticCurvatureMode mode) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const CurvatureSample sample = samples[index];
    output[index] =
        apply_local_elastic_curvature(sample.delta_f, sample.stretch, mode);
}

void run_gpu_benchmark(int sample_count, int repeats) {
    std::mt19937 random(17);
    std::uniform_real_distribution<float> stretch_distribution(0.55f, 1.70f);
    std::uniform_real_distribution<float> value_distribution(-1.0f, 1.0f);
    std::uniform_real_distribution<float> angle_distribution(-1.0f, 1.0f);

    std::vector<CurvatureSample> host_samples(sample_count);
    for (CurvatureSample& sample : host_samples) {
        sample.stretch = make_stretch(
            stretch_distribution(random),
            stretch_distribution(random),
            stretch_distribution(random),
            angle_distribution(random));
        for (float& value : sample.delta_f.data) {
            value = value_distribution(random);
        }
    }

    CurvatureSample* device_samples = nullptr;
    Mat3f* device_output = nullptr;
    check_cuda(cudaMalloc(&device_samples,
                          sizeof(CurvatureSample) * sample_count),
               "cudaMalloc samples");
    check_cuda(cudaMalloc(&device_output, sizeof(Mat3f) * sample_count),
               "cudaMalloc output");
    check_cuda(cudaMemcpy(device_samples, host_samples.data(),
                          sizeof(CurvatureSample) * sample_count,
                          cudaMemcpyHostToDevice),
               "copy samples");

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    check_cuda(cudaEventCreate(&begin), "create begin event");
    check_cuda(cudaEventCreate(&end), "create end event");

    const int block = 256;
    const int grid = (sample_count + block - 1) / block;
    const std::array<PabdElasticCurvatureMode, 4> modes = {{
        PabdElasticCurvatureMode::ProjectiveDynamics,
        PabdElasticCurvatureMode::CorotatedRest,
        PabdElasticCurvatureMode::PolarGaussNewton,
        PabdElasticCurvatureMode::ProjectedNewton3,
    }};
    std::cout << "[CURVATURE gpu] columns="
              << "mode,samples,repeats,total_ms,ns_per_body,gbodies_per_s,"
                 "checksum\n";
    for (PabdElasticCurvatureMode mode : modes) {
        for (int warmup = 0; warmup < 4; ++warmup) {
            curvature_benchmark_kernel<<<grid, block>>>(
                device_samples, device_output, sample_count, mode);
        }
        check_cuda(cudaDeviceSynchronize(), "curvature warmup");

        check_cuda(cudaEventRecord(begin), "record begin");
        for (int repeat = 0; repeat < repeats; ++repeat) {
            curvature_benchmark_kernel<<<grid, block>>>(
                device_samples, device_output, sample_count, mode);
        }
        check_cuda(cudaEventRecord(end), "record end");
        check_cuda(cudaEventSynchronize(end), "synchronize end");
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, begin, end),
                   "elapsed time");

        Mat3f checksum_matrix;
        check_cuda(cudaMemcpy(&checksum_matrix, device_output,
                              sizeof(Mat3f), cudaMemcpyDeviceToHost),
                   "copy checksum");
        double checksum = 0.0;
        for (float value : checksum_matrix.data) checksum += value;
        const double body_evaluations =
            static_cast<double>(sample_count) * repeats;
        const double ns_per_body = elapsed_ms * 1.0e6 / body_evaluations;
        const double gbodies_per_s = body_evaluations /
                                     (elapsed_ms * 1.0e6);
        std::cout << "curvature_gpu="
                  << elastic_curvature_mode_name(mode) << ','
                  << sample_count << ',' << repeats << ','
                  << std::fixed << std::setprecision(6) << elapsed_ms << ','
                  << ns_per_body << ',' << gbodies_per_s << ','
                  << checksum << '\n';
    }

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(device_output);
    cudaFree(device_samples);
}

}  // namespace

int main(int argc, char** argv) {
    int samples = 1 << 18;
    int repeats = 40;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--samples" && i + 1 < argc) {
            samples = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--repeats" && i + 1 < argc) {
            repeats = std::max(1, std::atoi(argv[++i]));
        }
    }

    try {
        run_spectrum_experiment();
        run_gpu_benchmark(samples, repeats);
    } catch (const std::exception& error) {
        std::cerr << "pabd_curvature_experiment failed: "
                  << error.what() << '\n';
        return 1;
    }
    return 0;
}
