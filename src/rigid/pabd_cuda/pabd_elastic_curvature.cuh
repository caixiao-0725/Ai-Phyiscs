// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>

#include "../../math/matrix.cuh"

namespace chysx {
namespace rigid {
namespace pabd_cuda {

enum class PabdElasticCurvatureMode : int {
    ProjectiveDynamics = 0,
    CorotatedRest = 1,
    PolarGaussNewton = 2,
    ProjectedNewton3 = 3,
};

enum class PabdPolarGnBackend : int {
    Assembled12 = 0,
    MatrixFreeRank3 = 1,
};

CHYSX_HDI math::Mat3f curvature_sym(const math::Mat3f& a) {
    return (a + math::transpose(a)) * 0.5f;
}

CHYSX_HDI math::Vec3f curvature_vex_skew(const math::Mat3f& a) {
    return math::Vec3f(0.5f * (a(2, 1) - a(1, 2)),
                       0.5f * (a(0, 2) - a(2, 0)),
                       0.5f * (a(1, 0) - a(0, 1)));
}

CHYSX_HDI math::Mat3f curvature_hat(const math::Vec3f& w) {
    return math::Mat3f(0.0f, -w.z, w.y,
                       w.z, 0.0f, -w.x,
                       -w.y, w.x, 0.0f);
}

CHYSX_HDI void curvature_jacobi_rotate(math::Mat3f& a,
                                       math::Mat3f& vectors,
                                       int p,
                                       int q) {
    const float apq = a(p, q);
    if (fabsf(apq) <= 1.0e-12f) return;

    const float app = a(p, p);
    const float aqq = a(q, q);
    const float tau = (aqq - app) / (2.0f * apq);
    const float sign = tau >= 0.0f ? 1.0f : -1.0f;
    const float t = sign / (fabsf(tau) + sqrtf(1.0f + tau * tau));
    const float c = rsqrtf(1.0f + t * t);
    const float s = t * c;

    for (int k = 0; k < 3; ++k) {
        if (k == p || k == q) continue;
        const float akp = a(k, p);
        const float akq = a(k, q);
        const float new_kp = c * akp - s * akq;
        const float new_kq = s * akp + c * akq;
        a(k, p) = new_kp;
        a(p, k) = new_kp;
        a(k, q) = new_kq;
        a(q, k) = new_kq;
    }

    a(p, p) = c * c * app - 2.0f * s * c * apq + s * s * aqq;
    a(q, q) = s * s * app + 2.0f * s * c * apq + c * c * aqq;
    a(p, q) = 0.0f;
    a(q, p) = 0.0f;

    for (int k = 0; k < 3; ++k) {
        const float vkp = vectors(k, p);
        const float vkq = vectors(k, q);
        vectors(k, p) = c * vkp - s * vkq;
        vectors(k, q) = s * vkp + c * vkq;
    }
}

CHYSX_HDI math::Mat3f curvature_project_psd3(const math::Mat3f& input) {
    math::Mat3f a = curvature_sym(input);
    math::Mat3f vectors = math::Mat3f::identity();
    for (int sweep = 0; sweep < 6; ++sweep) {
        curvature_jacobi_rotate(a, vectors, 0, 1);
        curvature_jacobi_rotate(a, vectors, 0, 2);
        curvature_jacobi_rotate(a, vectors, 1, 2);
    }

    math::Mat3f diagonal = math::Mat3f::zero();
    diagonal(0, 0) = fmaxf(a(0, 0), 0.0f);
    diagonal(1, 1) = fmaxf(a(1, 1), 0.0f);
    diagonal(2, 2) = fmaxf(a(2, 2), 0.0f);
    return vectors * diagonal * math::transpose(vectors);
}

// The exact Hessian of 0.5 * ||F - polar(F)||_F^2 separates into six
// unit symmetric-strain modes and three rotational modes. In the polar
// frame F = R S, the rotational 3x3 block is
// G = I - 2 * (tr(S) I - S)^-1.
CHYSX_HDI math::Mat3f curvature_rotational_block(const math::Mat3f& stretch) {
    const float tr = math::trace(stretch);
    math::Mat3f sylvester = math::Mat3f::identity() * tr - stretch;
    const float scale = fmaxf(fabsf(tr), 1.0f);
    sylvester(0, 0) += 1.0e-7f * scale;
    sylvester(1, 1) += 1.0e-7f * scale;
    sylvester(2, 2) += 1.0e-7f * scale;
    return math::Mat3f::identity() - math::inverse(sylvester) * 2.0f;
}

CHYSX_HDI math::Mat3f apply_local_elastic_curvature(
    const math::Mat3f& local_delta_f,
    const math::Mat3f& stretch,
    PabdElasticCurvatureMode mode) {
    if (mode == PabdElasticCurvatureMode::ProjectiveDynamics) {
        return local_delta_f;
    }

    const math::Mat3f symmetric = curvature_sym(local_delta_f);
    if (mode == PabdElasticCurvatureMode::CorotatedRest) {
        return symmetric;
    }

    const math::Vec3f skew = curvature_vex_skew(local_delta_f);
    const math::Mat3f rotational = curvature_rotational_block(stretch);
    if (mode == PabdElasticCurvatureMode::PolarGaussNewton) {
        return symmetric + curvature_hat(rotational * (rotational * skew));
    }

    const math::Mat3f projected = curvature_project_psd3(rotational);
    return symmetric + curvature_hat(projected * skew);
}

CHYSX_HDI math::Mat3f apply_world_elastic_curvature(
    const math::Mat3f& delta_f,
    const math::Mat3f& rotation,
    const math::Mat3f& stretch,
    PabdElasticCurvatureMode mode) {
    const math::Mat3f local_delta = math::transpose(rotation) * delta_f;
    return rotation * apply_local_elastic_curvature(local_delta, stretch, mode);
}

inline const char* elastic_curvature_mode_name(PabdElasticCurvatureMode mode) {
    switch (mode) {
        case PabdElasticCurvatureMode::ProjectiveDynamics:
            return "pd";
        case PabdElasticCurvatureMode::CorotatedRest:
            return "corotated_rest";
        case PabdElasticCurvatureMode::PolarGaussNewton:
            return "polar_gn";
        case PabdElasticCurvatureMode::ProjectedNewton3:
            return "projected_newton3";
    }
    return "unknown";
}

inline const char* polar_gn_backend_name(PabdPolarGnBackend backend) {
    switch (backend) {
        case PabdPolarGnBackend::Assembled12:
            return "assembled12";
        case PabdPolarGnBackend::MatrixFreeRank3:
            return "matrix_free_rank3";
    }
    return "unknown";
}

}  // namespace pabd_cuda
}  // namespace rigid
}  // namespace chysx
