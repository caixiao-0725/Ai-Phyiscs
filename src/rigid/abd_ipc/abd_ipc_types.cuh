// SPDX-License-Identifier: MIT
// ABD IPC solver types — domain-specific structs that sit on top of the
// public math (math/matrix.cuh) and geometry (geometry/tet_mesh.h) libraries.
//
// q-vector layout (12 DOF): [t(3), a1(3), a2(3), a3(3)]
// where A = [a1; a2; a3] are rows of the 3x3 affine matrix.
// World position: x = t + [x_bar . a1, x_bar . a2, x_bar . a3]

#pragma once

#include "../../math/matrix.cuh"   // Vec3f, Mat3f, Vec12f, Mat12f, ...
#include "../../math/vec.cuh"

namespace chysx {
namespace abd_ipc {

using math::Vec3f;
using math::Vec3i;
using math::Vec4i;
using math::Mat3f;
using math::Vec9f;
using math::Vec12f;
using math::Mat9f;
using math::Mat12f;

// ============================================================================
// q-vector helpers: pack/unpack the 12-DOF affine state
// ============================================================================

CHYSX_HDI Vec3f q_translation(const Vec12f& q) { return math::get_block3(q, 0); }

CHYSX_HDI void q_set_translation(Vec12f& q, Vec3f t) { math::set_block3(q, 0, t); }

CHYSX_HDI Vec3f q_affine_row(const Vec12f& q, int r) { return math::get_block3(q, 3 + r * 3); }

CHYSX_HDI void q_set_affine_row(Vec12f& q, int r, Vec3f v) {
    math::set_block3(q, 3 + r * 3, v);
}

// Build the 3x3 affine matrix A from q.
CHYSX_HDI Mat3f q_to_A(const Vec12f& q) {
    Mat3f A;
    for (int r = 0; r < 3; ++r) {
        Vec3f row = q_affine_row(q, r);
        A(r, 0) = row.x; A(r, 1) = row.y; A(r, 2) = row.z;
    }
    return A;
}

// Pack translation + affine matrix into a q vector.
CHYSX_HDI Vec12f make_q(Vec3f t, const Mat3f& A) {
    Vec12f q = Vec12f::zero();
    q_set_translation(q, t);
    for (int r = 0; r < 3; ++r)
        q_set_affine_row(q, r, Vec3f(A(r, 0), A(r, 1), A(r, 2)));
    return q;
}

// Identity q: translation=0, A=I
CHYSX_HDI Vec12f q_identity() { return make_q(Vec3f(0, 0, 0), Mat3f::identity()); }

// ============================================================================
// ABD Jacobi matrix J(x_bar)
// ============================================================================

// Compact representation: only stores the rest-frame vertex position x_bar.
// J is a 3x12 matrix (see header comment for layout).
struct ABDJacobi {
    Vec3f x_bar;

    CHYSX_HDI ABDJacobi() : x_bar(0, 0, 0) {}
    CHYSX_HDI explicit ABDJacobi(Vec3f xb) : x_bar(xb) {}

    // J * q -> world position (3-vector)
    CHYSX_HDI Vec3f mul_q(const Vec12f& q) const {
        Vec3f t = q_translation(q);
        return Vec3f(
            t.x + math::dot(x_bar, q_affine_row(q, 0)),
            t.y + math::dot(x_bar, q_affine_row(q, 1)),
            t.z + math::dot(x_bar, q_affine_row(q, 2))
        );
    }

    // J^T * g3 -> generalized force (12-vector)
    CHYSX_HDI Vec12f mul_JT(Vec3f g) const {
        Vec12f r;
        r[0] = g.x; r[1] = g.y; r[2] = g.z;
        r[3]  = x_bar.x * g.x; r[4]  = x_bar.y * g.x; r[5]  = x_bar.z * g.x;
        r[6]  = x_bar.x * g.y; r[7]  = x_bar.y * g.y; r[8]  = x_bar.z * g.y;
        r[9]  = x_bar.x * g.z; r[10] = x_bar.y * g.z; r[11] = x_bar.z * g.z;
        return r;
    }

    // Ji^T * H3x3 * Jj -> 12x12 Hessian lifting
    CHYSX_HDI static Mat12f JT_H_J(const ABDJacobi& Ji, const Mat3f& H,
                                    const ABDJacobi& Jj) {
        Mat12f R = Mat12f::zero();
        const Vec3f& xi = Ji.x_bar;
        const Vec3f& xj = Jj.x_bar;

        // (0,0) block: H
        math::set_block3x3(R, 0, 0, H);

        // (0, 3+3k) and (3+3k, 0) blocks
        for (int k = 0; k < 3; ++k) {
            Mat3f Hxj = H * xj[k];
            Mat3f xiH = xi[k] * H;
            math::set_block3x3(R, 0, 3 + k * 3, Hxj);
            math::set_block3x3(R, 3 + k * 3, 0, xiH);
        }

        // (3+3k, 3+3l) blocks
        for (int k = 0; k < 3; ++k)
            for (int l = 0; l < 3; ++l) {
                float scale = xi[k] * xj[l];
                math::set_block3x3(R, 3 + k * 3, 3 + l * 3, H * scale);
            }
        return R;
    }
};

// ============================================================================
// Dyadic Mass — integrated moments for building the 12x12 mass matrix
// ============================================================================

struct DyadicMass {
    float m;          // total mass
    Vec3f m_x_bar;    // mass-weighted centroid integral
    Mat3f m_xx;       // mass-weighted second moment integral

    CHYSX_HDI DyadicMass() : m(0), m_x_bar(0, 0, 0), m_xx(Mat3f::zero()) {}

    CHYSX_HDI Mat12f to_mat() const {
        Mat12f M = Mat12f::zero();

        // Translation block: m * I_3
        M(0,0) = m; M(1,1) = m; M(2,2) = m;

        // Cross-coupling: M(k, 3+3k+j) = m_x_bar[j]
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 3; ++j) {
                M(k, 3 + k * 3 + j) = m_x_bar[j];
                M(3 + k * 3 + j, k) = m_x_bar[j];
            }

        // Affine diagonal blocks: m_xx repeated 3 times
        for (int k = 0; k < 3; ++k)
            math::set_block3x3(M, 3 + k * 3, 3 + k * 3, m_xx);

        return M;
    }

    // Accumulate a point mass at rest position xb.
    CHYSX_HDI void add_point(float mass, Vec3f xb) {
        m += mass;
        m_x_bar += xb * mass;
        m_xx += math::outer(xb, xb) * mass;
    }
};

// ============================================================================
// Per-body state
// ============================================================================

struct ABDBody {
    Vec12f q;            // current DOFs
    Vec12f q_prev;       // previous frame DOFs
    Vec12f q_tilde;      // BDF1 prediction target
    Vec12f q_v;          // generalized velocity
    Vec12f abd_gravity;  // generalized gravity acceleration = M_inv * F_body

    Mat12f M;            // 12x12 mass matrix
    Mat12f M_inv;        // inverse mass matrix

    float kappa;         // OrthoPotential stiffness [Pa]
    float volume;        // rest volume [m^3]
    float friction;      // friction coefficient

    bool is_fixed;
    bool is_dynamic;

    int surface_vert_offset;
    int surface_vert_count;
    int surface_tri_offset;
    int surface_tri_count;
    int surface_edge_offset;
    int surface_edge_count;
};

// ============================================================================
// Contact pair types for IPC
// ============================================================================

enum class ContactType : int { PP = 0, PE = 1, PT = 2, EE = 3 };

struct ContactPair {
    ContactType type;
    int v[4];        // vertex indices
    int body[4];     // owning body per vertex
    float D;         // squared distance
};

// ============================================================================
// Solver configuration
// ============================================================================

struct ABDConfig {
    float dt = 0.01f;
    Vec3f gravity = Vec3f(0.0f, -9.8f, 0.0f);

    float default_kappa = 1.0e8f;   // OrthoPotential stiffness [Pa]

    // IPC contact
    float d_hat = 0.01f;
    float contact_kappa = 1.0e9f;
    float default_thickness = 0.0f;
    float default_friction = 0.5f;

    // Newton solver
    int newton_max_iter = 1024;
    int newton_min_iter = 1;
    float velocity_tol = 0.05f;      // m/s
    float transrate_tol = 0.1f;      // 1/s

    // PCG
    float pcg_tol = 1.0e-3f;
    int pcg_max_iter_ratio = 2;

    // Line search
    int line_search_max_iter = 8;

    // Broadphase: "none" = brute force, "quantbvh", "optix"
    bool use_bvh_broadphase = true;
    enum class BroadphaseType { BruteForce, QuantBvh, OptiX };
    BroadphaseType broadphase_type = BroadphaseType::QuantBvh;

    // Output
    int total_frames = 50;
    bool export_obj = true;
    const char* output_dir = "output/abd_ipc/";
};

}  // namespace abd_ipc
}  // namespace chysx
