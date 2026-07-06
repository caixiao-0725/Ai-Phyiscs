// SPDX-License-Identifier: MIT
// Rigid-IPC solver types — 6-DOF rigid body structs on top of the public
// math (math/matrix.cuh) and geometry libraries.
//
// DOF layout (6 DOF): [p(3), theta(3)]
// where p is the translation and theta is a rotation vector (axis-angle).
// World position: x = R(theta) * x_bar + p

#pragma once

#include "../../math/matrix.cuh"
#include "../../math/vec.cuh"

namespace chysx {
namespace rigid_ipc {

using math::Vec3f;
using math::Vec3i;
using math::Vec4i;
using math::Mat3f;
using math::Vec9f;
using math::Vec12f;
using math::Mat12f;

using Vec6f  = math::VecN<float, 6>;
using Mat6f  = math::MatN<float, 6>;
using Vec6d  = math::VecN<double, 6>;
using Mat6d  = math::MatN<double, 6>;

// ============================================================================
// Pose helpers
// ============================================================================

CHYSX_HDI Vec3f pose_position(const Vec6f& q) { return math::get_block3(q, 0); }
CHYSX_HDI Vec3f pose_rotation(const Vec6f& q) { return math::get_block3(q, 3); }

CHYSX_HDI void pose_set_position(Vec6f& q, Vec3f p) { math::set_block3(q, 0, p); }
CHYSX_HDI void pose_set_rotation(Vec6f& q, Vec3f r) { math::set_block3(q, 3, r); }

CHYSX_HDI Vec6f make_pose(Vec3f p, Vec3f theta) {
    Vec6f q = Vec6f::zero();
    pose_set_position(q, p);
    pose_set_rotation(q, theta);
    return q;
}

CHYSX_HDI Vec6f pose_identity() { return Vec6f::zero(); }

// ============================================================================
// Per-body state
// ============================================================================

struct RigidBody {
    Vec6f q;            // current DOFs [p(3), theta(3)]
    Vec6f q_prev;       // previous frame DOFs
    Vec6f q_tilde;      // BDF1 prediction target
    Vec6f q_v;          // generalized velocity [v_linear(3), omega(3)]

    float mass;
    Vec3f moment_of_inertia; // principal-axis inertia (Ix, Iy, Iz)

    // Precomputed rotation-matrix derivative state (set by solver at each step)
    Mat3f R;             // R(theta) at current pose
    Mat3f R_prev;        // R(theta_prev)
    Mat3f R0;            // initial rotation (principal axes alignment)
    Mat3f Qdot_prev;     // dR/dt at previous step

    float friction;

    bool is_fixed;
    bool is_dynamic;
    bool dof_fixed[6] = {};  // per-DOF lock: [px,py,pz,rx,ry,rz]

    int surface_vert_offset;
    int surface_vert_count;
    int surface_tri_offset;
    int surface_tri_count;
    int surface_edge_offset;
    int surface_edge_count;
};

// ============================================================================
// Contact pair types (shared with ABD-IPC)
// ============================================================================

enum class ContactType : int { PP = 0, PE = 1, PT = 2, EE = 3 };

struct ContactPair {
    ContactType type;
    int v[4];
    int body[4];
    float D;  // squared distance
};

// ============================================================================
// Solver configuration
// ============================================================================

struct RigidIPCConfig {
    float dt = 0.01f;
    Vec3f gravity = Vec3f(0.0f, -9.81f, 0.0f);

    // IPC contact barrier
    float d_hat = 1e-4f;
    float contact_kappa = 0.0f;      // 0 = auto (IPC adaptive formula)
    float default_friction = 0.0f;

    // Adaptive kappa (IPC paper)
    float kappa_min_scale = 1e11f;    // multiplier for min kappa
    float kappa_dhat_eps_scale = 1e-9f;

    // Newton solver
    int newton_max_iter = 3000;
    int newton_min_iter = 1;
    float velocity_conv_tol = 1e-2f;  // relative to bbox_diagonal
    bool velocity_conv_tol_abs = false;

    // PCG
    float pcg_tol = 1.0e-3f;
    int pcg_max_iter_ratio = 4;

    // Line search
    int line_search_max_iter = 12;

    // Broadphase
    bool use_bvh_broadphase = true;
    enum class BroadphaseType { BruteForce, QuantBvh, OptiX };
    BroadphaseType broadphase_type = BroadphaseType::QuantBvh;

    // Output
    int total_frames = 500;
    bool export_obj = true;
    const char* output_dir = "output/rigid_ipc/";
};

}  // namespace rigid_ipc
}  // namespace chysx
