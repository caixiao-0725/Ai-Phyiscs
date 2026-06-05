// SPDX-License-Identifier: MIT
// AVBD solver main loop. Adapted from avbd-demo3d (Chris Giles, 2026).

#include "avbd_solver.h"
#include "avbd_broadphase_gpu.h"
#include "avbd_narrowphase_gpu.h"
#include "avbd_graph_coloring.h"
#include "avbd_gpu_solver.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <utility>
#include <vector>

namespace chysx {
namespace avbd {

void SoaData::pack(Rigid* bodies) {
    count = 0;
    for (Rigid* b = bodies; b; b = b->next)
        count++;

    body_ptrs.resize(count);
    pos_x.resize(count); pos_y.resize(count); pos_z.resize(count);
    quat_x.resize(count); quat_y.resize(count); quat_z.resize(count); quat_w.resize(count);
    half_x.resize(count); half_y.resize(count); half_z.resize(count);
    radius.resize(count);
    mass.resize(count);
    friction.resize(count);
    moment_x.resize(count); moment_y.resize(count); moment_z.resize(count);
    vel_x.resize(count); vel_y.resize(count); vel_z.resize(count);
    velang_x.resize(count); velang_y.resize(count); velang_z.resize(count);
    prevvel_x.resize(count); prevvel_y.resize(count); prevvel_z.resize(count);

    int i = 0;
    for (Rigid* b = bodies; b; b = b->next, i++) {
        body_ptrs[i] = b;
        pos_x[i] = b->positionLin.x;
        pos_y[i] = b->positionLin.y;
        pos_z[i] = b->positionLin.z;
        quat_x[i] = b->positionAng.x;
        quat_y[i] = b->positionAng.y;
        quat_z[i] = b->positionAng.z;
        quat_w[i] = b->positionAng.w;
        half_x[i] = b->size.x * 0.5f;
        half_y[i] = b->size.y * 0.5f;
        half_z[i] = b->size.z * 0.5f;
        radius[i] = b->radius;
        mass[i] = b->mass;
        friction[i] = b->friction;
        moment_x[i] = b->moment.x;
        moment_y[i] = b->moment.y;
        moment_z[i] = b->moment.z;
        vel_x[i] = b->velocityLin.x;
        vel_y[i] = b->velocityLin.y;
        vel_z[i] = b->velocityLin.z;
        velang_x[i] = b->velocityAng.x;
        velang_y[i] = b->velocityAng.y;
        velang_z[i] = b->velocityAng.z;
        prevvel_x[i] = b->prevVelocityLin.x;
        prevvel_y[i] = b->prevVelocityLin.y;
        prevvel_z[i] = b->prevVelocityLin.z;
    }
}

void SoaData::unpack(Rigid* bodies) {
    int i = 0;
    for (Rigid* b = bodies; b; b = b->next, i++) {
        b->positionLin = float3{pos_x[i], pos_y[i], pos_z[i]};
        b->positionAng = quat{quat_x[i], quat_y[i], quat_z[i], quat_w[i]};
        b->velocityLin = float3{vel_x[i], vel_y[i], vel_z[i]};
        b->velocityAng = float3{velang_x[i], velang_y[i], velang_z[i]};
        b->prevVelocityLin = float3{prevvel_x[i], prevvel_y[i], prevvel_z[i]};
        b->syncFromQuat();
    }
}

Solver::Solver()
    : bodies(nullptr), forces(nullptr), broadphase_gpu_(nullptr) {
    defaultParams();
}

Solver::~Solver() {
    clear();
    delete broadphase_gpu_;
    delete narrowphase_gpu_;
    delete graph_coloring_gpu_;
    delete gpu_solver_;
}

Rigid* Solver::pick(float3 origin, float3 dir, float3& local) {
    const float epsilon = 1.0e-6f;
    float bestT = INFINITY;
    Rigid* bestBody = nullptr;
    float3 bestLocal = {0, 0, 0};

    for (Rigid* body = bodies; body != nullptr; body = body->next) {
        if (body->mass <= 0.0f)
            continue;

        // Always use quaternion for ray-cast (kept in sync)
        quat invRot = conjugate(body->positionAng);
        float3 o = rotate(invRot, origin - body->positionLin);
        float3 d = rotate(invRot, dir);
        float3 half = body->size * 0.5f;

        float tEnter = 0.0f;
        float tExit = INFINITY;
        bool hit = true;

        for (int i = 0; i < 3; ++i) {
            if (std::fabs(d[i]) < epsilon) {
                if (o[i] < -half[i] || o[i] > half[i]) {
                    hit = false;
                    break;
                }
                continue;
            }
            float invD = 1.0f / d[i];
            float t0 = (-half[i] - o[i]) * invD;
            float t1 = (half[i] - o[i]) * invD;
            if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
            tEnter = max(tEnter, t0);
            tExit = min(tExit, t1);
            if (tEnter > tExit) { hit = false; break; }
        }

        if (!hit) continue;
        float tHit = tEnter >= 0.0f ? tEnter : tExit;
        if (tHit < 0.0f) continue;
        if (tHit < bestT) {
            bestT = tHit;
            bestBody = body;
            bestLocal = o + d * tHit;
        }
    }

    if (!bestBody) return nullptr;
    local = bestLocal;
    return bestBody;
}

void Solver::clear() {
    while (forces) delete forces;
    while (bodies) delete bodies;
    has_ground_plane = false;
    has_sphere_collider = false;
    sphere_radius = 0.0f;
    sphere_friction = 0.5f;
    gpu_state_valid_ = false;
    gpu_state_valid_prev_ = false;
    prev_body_count_ = 0;
}

void Solver::defaultParams() {
    dt = 1.0f / 60.0f;
    gravity = -10.0f;
    iterations = 10;
    betaLin = 10000.0f;
    betaAng = 100.0f;
    alpha = 0.95f;
    gamma = 0.99f;
}

void Solver::step() {
    gpu_state_valid_ = false;

    // Affine rotation mode: CPU-only path (GPU kernels don't support it yet)
    if (rotation_mode == RotationMode::Affine) {
        stepCpuAffine();
        return;
    }

    // Detect whether any Joint/Spring forces exist
    bool has_joint_spring = false;
    for (Force* f = forces; f != nullptr; f = f->next) {
        if (dynamic_cast<Joint*>(f) || dynamic_cast<Spring*>(f)) {
            has_joint_spring = true;
            break;
        }
    }

    // Count bodies (cheap linked-list walk)
    int body_count = 0;
    for (Rigid* b = bodies; b; b = b->next) body_count++;

    // Create GPU objects if needed
    if (!gpu_solver_) gpu_solver_ = new GpuSolver();
    if (!narrowphase_gpu_) narrowphase_gpu_ = new NarrowphaseGPU();
    if (!graph_coloring_gpu_) graph_coloring_gpu_ = new GraphColoringGPU();
    if (!broadphase_gpu_) {
        broadphase_gpu_ = new BroadphaseGPU();
    }

    // Decide whether we need a full CPU→GPU upload.
    // Full upload needed when: first frame, body count changed, or joints/springs
    // mutated CPU-side body state since last frame.
    bool need_full_upload = !gpu_state_valid_prev_ ||
                            body_count != prev_body_count_ ||
                            has_joint_spring;

    if (need_full_upload) {
        soa_.pack(bodies);
        gpu_solver_->upload_bodies(
            soa_.pos_x.data(), soa_.pos_y.data(), soa_.pos_z.data(),
            soa_.quat_x.data(), soa_.quat_y.data(), soa_.quat_z.data(), soa_.quat_w.data(),
            soa_.vel_x.data(), soa_.vel_y.data(), soa_.vel_z.data(),
            soa_.velang_x.data(), soa_.velang_y.data(), soa_.velang_z.data(),
            soa_.prevvel_x.data(), soa_.prevvel_y.data(), soa_.prevvel_z.data(),
            soa_.mass.data(), soa_.moment_x.data(), soa_.moment_y.data(), soa_.moment_z.data(),
            soa_.half_x.data(), soa_.half_y.data(), soa_.half_z.data(),
            soa_.friction.data(),
            soa_.count);
    }

#ifdef AVBD_VALIDATE_BROADPHASE
    if (need_full_upload) {
        // CPU broadphase for validation (only when CPU data is fresh)
        std::vector<std::pair<int,int>> cpu_pairs;
        for (int ia = 0; ia < body_count; ia++) {
            for (int ib = ia + 1; ib < body_count; ib++) {
                Rigid* bodyA = soa_.body_ptrs[ia];
                Rigid* bodyB = soa_.body_ptrs[ib];
                float3 dp = bodyA->positionLin - bodyB->positionLin;
                float r = bodyA->radius + bodyB->radius;
                if (dot(dp, dp) <= r * r)
                    cpu_pairs.push_back({ia, ib});
            }
        }
    }
#endif

    // Sphere body sits at the end of the SoA array; exclude from broadphase.
    int broadphase_count = body_count;
    int sphere_body_idx = -1;
    if (has_sphere_collider) {
        broadphase_count = body_count - 1;
        sphere_body_idx = body_count - 1;
    }

    // Set up ground body at index body_count (virtual n+1th body)
    int ground_body_idx = body_count;
    int n_bodies_for_solver = body_count;
    if (has_ground_plane) {
        gpu_solver_->setup_ground_body(ground_z, ground_friction);
        n_bodies_for_solver = body_count + 1;
    }

    // GPU broadphase: read directly from GpuSolver GPU arrays (zero H2D)
    int pair_count = broadphase_gpu_->query_gpu(
        gpu_solver_->pos_x_dev(), gpu_solver_->pos_y_dev(), gpu_solver_->pos_z_dev(),
        gpu_solver_->quat_x_dev(), gpu_solver_->quat_y_dev(),
        gpu_solver_->quat_z_dev(), gpu_solver_->quat_w_dev(),
        gpu_solver_->half_x_dev(), gpu_solver_->half_y_dev(), gpu_solver_->half_z_dev(),
        gpu_solver_->mass_dev(),
                broadphase_count);

    {
        int n_manifolds = 0, total_contacts = 0;

        if (pair_count > 0) {
            narrowphase_gpu_->query_gpu(
                gpu_solver_->pos_x_dev(), gpu_solver_->pos_y_dev(), gpu_solver_->pos_z_dev(),
                gpu_solver_->quat_x_dev(), gpu_solver_->quat_y_dev(),
                gpu_solver_->quat_z_dev(), gpu_solver_->quat_w_dev(),
                gpu_solver_->half_x_dev(), gpu_solver_->half_y_dev(), gpu_solver_->half_z_dev(),
                gpu_solver_->friction_dev(),
                broadphase_gpu_->pair_a_dev(), broadphase_gpu_->pair_b_dev(),
                pair_count, n_bodies_for_solver,
                n_manifolds, total_contacts);
        } else {
            narrowphase_gpu_->query_gpu(
                gpu_solver_->pos_x_dev(), gpu_solver_->pos_y_dev(), gpu_solver_->pos_z_dev(),
                gpu_solver_->quat_x_dev(), gpu_solver_->quat_y_dev(),
                gpu_solver_->quat_z_dev(), gpu_solver_->quat_w_dev(),
                gpu_solver_->half_x_dev(), gpu_solver_->half_y_dev(), gpu_solver_->half_z_dev(),
                gpu_solver_->friction_dev(),
                nullptr, nullptr,
                0, n_bodies_for_solver,
                n_manifolds, total_contacts);
        }

        // Append ground-plane contacts (box-plane narrowphase)
        if (has_ground_plane) {
            narrowphase_gpu_->append_ground_plane_gpu(
                gpu_solver_->pos_x_dev(), gpu_solver_->pos_y_dev(), gpu_solver_->pos_z_dev(),
                gpu_solver_->quat_x_dev(), gpu_solver_->quat_y_dev(),
                gpu_solver_->quat_z_dev(), gpu_solver_->quat_w_dev(),
                gpu_solver_->half_x_dev(), gpu_solver_->half_y_dev(), gpu_solver_->half_z_dev(),
                gpu_solver_->friction_dev(), gpu_solver_->mass_dev(),
                broadphase_count, ground_z, ground_friction,
                ground_body_idx,
                n_manifolds, total_contacts);
        }

        // Append sphere-collider contacts (box-sphere + sphere-ground)
        if (has_sphere_collider) {
            narrowphase_gpu_->append_sphere_gpu(
                gpu_solver_->pos_x_dev(), gpu_solver_->pos_y_dev(), gpu_solver_->pos_z_dev(),
                gpu_solver_->quat_x_dev(), gpu_solver_->quat_y_dev(),
                gpu_solver_->quat_z_dev(), gpu_solver_->quat_w_dev(),
                gpu_solver_->half_x_dev(), gpu_solver_->half_y_dev(), gpu_solver_->half_z_dev(),
                gpu_solver_->friction_dev(), gpu_solver_->mass_dev(),
                broadphase_count, sphere_body_idx, sphere_radius, sphere_friction,
                has_ground_plane, ground_z, ground_friction, ground_body_idx,
                n_manifolds, total_contacts);
        }

        if (n_manifolds > 0) {
            narrowphase_gpu_->warmstart_gpu(n_manifolds, total_contacts, n_bodies_for_solver);

            const int* vtx_counts_dev = narrowphase_gpu_->vtx_counts_dev();
            const VertexEntry* vtx_table_dev = narrowphase_gpu_->vtx_table_dev();
            int vtx_stride = narrowphase_gpu_->vertex_table_stride();

            auto coloring = graph_coloring_gpu_->color_jp(
                vtx_counts_dev, vtx_table_dev, n_bodies_for_solver, vtx_stride);
            int num_colors = coloring.num_colors;
            const int* colors_dev = graph_coloring_gpu_->colors_gpu();

            gpu_solver_->solve(
                narrowphase_gpu_->manifolds_dev(),
                narrowphase_gpu_->contacts_dev(),
                n_manifolds,
                vtx_counts_dev, vtx_table_dev, vtx_stride,
                colors_dev, num_colors,
                iterations, dt, gravity,
                alpha, betaLin, gamma);

            gpu_state_valid_ = true;

            // Only download to CPU when joints/springs need it
            if (has_joint_spring) {
                if (!need_full_upload) soa_.pack(bodies);
                gpu_solver_->download_positions(
                    soa_.pos_x.data(), soa_.pos_y.data(), soa_.pos_z.data(),
                    soa_.quat_x.data(), soa_.quat_y.data(), soa_.quat_z.data(), soa_.quat_w.data(),
                    soa_.vel_x.data(), soa_.vel_y.data(), soa_.vel_z.data(),
                    soa_.velang_x.data(), soa_.velang_y.data(), soa_.velang_z.data(),
                    body_count);
                soa_.unpack(bodies);
            }

            // Snapshot contacts D2D for next frame's warm-start
            narrowphase_gpu_->snapshot_for_next_frame(n_manifolds, total_contacts, n_bodies_for_solver);
        } else {
            if (has_joint_spring) {
                // No collisions + joints/springs: CPU-only path
                if (!need_full_upload) soa_.pack(bodies);

                for (Rigid* body = bodies; body != nullptr; body = body->next) {
                    body->inertialLin = body->positionLin + body->velocityLin * dt;
                    if (body->mass > 0)
                        body->inertialLin += float3{0, 0, gravity} * (dt * dt);

                    float3 accel = (body->velocityLin - body->prevVelocityLin) / dt;
                    float accelExt = accel.z * sign(gravity);
                    float accelWeight = clamp(accelExt / std::fabs(gravity), 0.0f, 1.0f);
                    if (!std::isfinite(accelWeight))
                        accelWeight = 0.0f;

                    body->initialLin = body->positionLin;

                    if (rotation_mode == RotationMode::Affine) {
                        // Affine mode: predict via angular velocity → matrix
                        body->inertialAng = body->positionAng + body->velocityAng * dt;
                        body->inertialAff = body->affine;  // save current as inertial target
                        body->initialAng = body->positionAng;
                        body->initialAff = body->affine;
                        if (body->mass > 0) {
                            body->positionLin = body->positionLin + body->velocityLin * dt +
                                                float3{0, 0, gravity} * (accelWeight * dt * dt);
                            // Predict affine via exponential map from angular velocity
                            float3x3 dR_skew = skew(body->velocityAng * dt);
                            body->affine = (identity3x3() + dR_skew) * body->affine;
                            body->affine = polar_rotation(body->affine);
                            body->syncFromAffine();
                        }
                    } else {
                        // Axis-angle mode (original)
                        body->inertialAng = body->positionAng + body->velocityAng * dt;
                        body->initialAng = body->positionAng;
                        if (body->mass > 0) {
                            body->positionLin = body->positionLin + body->velocityLin * dt +
                                                float3{0, 0, gravity} * (accelWeight * dt * dt);
                            body->positionAng = body->positionAng + body->velocityAng * dt;
                        }
                    }
                }

                for (Force* force = forces; force != nullptr;) {
                    if (!force->initialize()) {
                        Force* next = force->next;
                        delete force;
                        force = next;
                    } else {
                        force = force->next;
                    }
                }

                if (rotation_mode == RotationMode::Affine) {
                    // Affine mode solver iterations
                    for (int it = 0; it < iterations; it++) {
                        for (Rigid* body = bodies; body != nullptr; body = body->next) {
                            if (body->mass <= 0) continue;
                            float3x3 MLin = diagonal(body->mass, body->mass, body->mass);
                            float3x3 MAng = diagonal(body->moment.x, body->moment.y, body->moment.z);
                            float3x3 lhsLin = MLin / (dt * dt);
                            float3x3 lhsAng = MAng / (dt * dt);
                            float3x3 lhsCross = float3x3{0, 0, 0, 0, 0, 0, 0, 0, 0};
                            float3 angDisp = mat_to_angular(body->affine, body->inertialAff);
                            float3 rhsLin = MLin / (dt * dt) * (body->positionLin - body->inertialLin);
                            float3 rhsAng = MAng / (dt * dt) * angDisp;
                            for (Force* force = body->forces; force != nullptr;
                                 force = (force->bodyA == body) ? force->nextA : force->nextB)
                                force->updatePrimal(body, alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
                            float3 dxLin, dxAng;
                            solve(lhsLin, lhsAng, lhsCross, -rhsLin, -rhsAng, dxLin, dxAng);
                            body->positionLin = body->positionLin + dxLin;
                            // Apply angular increment via exponential map and re-project
                            float3x3 dR = identity3x3() + skew(dxAng);
                            body->affine = dR * body->affine;
                            body->affine = polar_rotation(body->affine);
                            body->syncFromAffine();
                        }
                        for (Force* force = forces; force != nullptr; force = force->next)
                            force->updateDual(alpha);
                    }
                } else {
                    // Axis-angle mode solver iterations (original)
                    for (int it = 0; it < iterations; it++) {
                        for (Rigid* body = bodies; body != nullptr; body = body->next) {
                            if (body->mass <= 0) continue;
                            float3x3 MLin = diagonal(body->mass, body->mass, body->mass);
                            float3x3 MAng = diagonal(body->moment.x, body->moment.y, body->moment.z);
                            float3x3 lhsLin = MLin / (dt * dt);
                            float3x3 lhsAng = MAng / (dt * dt);
                            float3x3 lhsCross = float3x3{0, 0, 0, 0, 0, 0, 0, 0, 0};
                            float3 rhsLin = MLin / (dt * dt) * (body->positionLin - body->inertialLin);
                            float3 rhsAng = MAng / (dt * dt) * (body->positionAng - body->inertialAng);
                            for (Force* force = body->forces; force != nullptr;
                                 force = (force->bodyA == body) ? force->nextA : force->nextB)
                                force->updatePrimal(body, alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
                            float3 dxLin, dxAng;
                            solve(lhsLin, lhsAng, lhsCross, -rhsLin, -rhsAng, dxLin, dxAng);
                            body->positionLin = body->positionLin + dxLin;
                            body->positionAng = body->positionAng + dxAng;
                        }
                        for (Force* force = forces; force != nullptr; force = force->next)
                            force->updateDual(alpha);
                    }
                }

                // Velocity recovery
                for (Rigid* body = bodies; body != nullptr; body = body->next) {
                    body->prevVelocityLin = body->velocityLin;
                    if (body->mass > 0) {
                        body->velocityLin = (body->positionLin - body->initialLin) / dt;
                        if (rotation_mode == RotationMode::Affine) {
                            body->velocityAng = mat_to_angular(body->affine, body->initialAff) / dt;
                        } else {
                            body->velocityAng = (body->positionAng - body->initialAng) / dt;
                        }
                    }
                }

                soa_.pack(bodies);
                gpu_solver_->upload_bodies(
                    soa_.pos_x.data(), soa_.pos_y.data(), soa_.pos_z.data(),
                    soa_.quat_x.data(), soa_.quat_y.data(), soa_.quat_z.data(), soa_.quat_w.data(),
                    soa_.vel_x.data(), soa_.vel_y.data(), soa_.vel_z.data(),
                    soa_.velang_x.data(), soa_.velang_y.data(), soa_.velang_z.data(),
                    soa_.prevvel_x.data(), soa_.prevvel_y.data(), soa_.prevvel_z.data(),
                    soa_.mass.data(), soa_.moment_x.data(), soa_.moment_y.data(), soa_.moment_z.data(),
                    soa_.half_x.data(), soa_.half_y.data(), soa_.half_z.data(),
                    soa_.friction.data(),
                    soa_.count);
                gpu_state_valid_ = true;
            } else {
                // No collisions, no joints: pure free-flight.
                // Run inertial predict + velocity update without constraints.
                for (Rigid* body = bodies; body != nullptr; body = body->next) {
                    if (body->mass <= 0) continue;
                    body->prevVelocityLin = body->velocityLin;
                    body->velocityLin += float3{0, 0, gravity} * dt;
                    body->positionLin = body->positionLin + body->velocityLin * dt;
                    if (rotation_mode == RotationMode::Affine) {
                        float3x3 dR_skew = skew(body->velocityAng * dt);
                        body->affine = (identity3x3() + dR_skew) * body->affine;
                        body->affine = polar_rotation(body->affine);
                        body->syncFromAffine();
                    } else {
                        body->positionAng = body->positionAng + body->velocityAng * dt;
                    }
                }

                soa_.pack(bodies);
                gpu_solver_->upload_bodies(
                    soa_.pos_x.data(), soa_.pos_y.data(), soa_.pos_z.data(),
                    soa_.quat_x.data(), soa_.quat_y.data(), soa_.quat_z.data(), soa_.quat_w.data(),
                    soa_.vel_x.data(), soa_.vel_y.data(), soa_.vel_z.data(),
                    soa_.velang_x.data(), soa_.velang_y.data(), soa_.velang_z.data(),
                    soa_.prevvel_x.data(), soa_.prevvel_y.data(), soa_.prevvel_z.data(),
                    soa_.mass.data(), soa_.moment_x.data(), soa_.moment_y.data(), soa_.moment_z.data(),
                    soa_.half_x.data(), soa_.half_y.data(), soa_.half_z.data(),
                    soa_.friction.data(),
                    soa_.count);
                gpu_state_valid_ = true;
            }
        }
    }

    gpu_state_valid_prev_ = gpu_state_valid_;
    prev_body_count_ = body_count;
}

// Ground-plane contact point for PABD affine solver.
struct GroundContact {
    Rigid* body;
    float3 rLocal;      // contact point in body-local frame
    float3 rWorld;      // contact point in world frame (A * rLocal)
    float  penetration; // negative = penetrating
};

void Solver::stepCpuAffine() {
    static int frame = 0;
    frame++;

    int body_count = 0;
    for (Rigid* b = bodies; b; b = b->next) body_count++;

    // ---- Phase 1: Predict (symplectic Euler) ----
    for (Rigid* body = bodies; body != nullptr; body = body->next) {
        body->initialLin = body->positionLin;
        body->initialAff = body->affine;
        body->initialAng = body->positionAng;

        if (body->mass <= 0) continue;

        float3 gravDt = float3{0, 0, gravity} * dt;
        float3 newVel = body->velocityLin + gravDt;
        float3 newPos = body->positionLin + newVel * dt;
        body->inertialLin = newPos;
        body->positionLin = newPos;

        body->inertialAff = body->affine;
        float3x3 dR_skew = skew(body->velocityAng * dt);
        body->affine = (identity3x3() + dR_skew) * body->affine;
        body->affine = polar_rotation(body->affine);
        body->syncFromAffine();
    }

    // ---- Phase 2: Ground-plane collision detection ----
    // Detect box vertices penetrating the ground plane (z = ground_z, normal = +Z).
    static const float kLocalVerts[8][3] = {
        {-0.5f,-0.5f,-0.5f}, {+0.5f,-0.5f,-0.5f},
        {+0.5f,+0.5f,-0.5f}, {-0.5f,+0.5f,-0.5f},
        {-0.5f,-0.5f,+0.5f}, {+0.5f,-0.5f,+0.5f},
        {+0.5f,+0.5f,+0.5f}, {-0.5f,+0.5f,+0.5f}};

    std::vector<GroundContact> groundContacts;
    if (has_ground_plane) {
        for (Rigid* body = bodies; body != nullptr; body = body->next) {
            if (body->mass <= 0) continue;
            for (int v = 0; v < 8; v++) {
                float3 rLocal = {
                    kLocalVerts[v][0] * body->size.x,
                    kLocalVerts[v][1] * body->size.y,
                    kLocalVerts[v][2] * body->size.z};
                float3 rWorld = body->affine * rLocal;
                float3 worldPos = body->positionLin + rWorld;
                float pen = worldPos.z - ground_z;
                if (pen < 0.0f) {
                    groundContacts.push_back({body, rLocal, rWorld, pen});
                }
            }
        }
    }

    // Also process existing Force constraints (joints, springs, etc.)
    for (Force* force = forces; force != nullptr;) {
        if (!force->initialize()) {
            Force* next = force->next;
            delete force;
            force = next;
        } else {
            force = force->next;
        }
    }

    // ---- Phase 3: PABD 12×12 Schur complement solve ----
    //
    // Generalized coordinates (local frame): q_L = (c_L, A_L) ∈ R^12
    //   Global: c = c̃ + Ã·c_L,  A = Ã·A_L
    //   Inertial target in local frame: q_0 = (0, I)
    //
    // 3D point-to-point penalty:
    //   Ψ_c = ½ k ‖x_i - x_j‖²
    //   where x_i = c + A·r  (body contact point)
    //         x_j = ground projection (fixed target)
    //
    // In local frame, the contact point is:
    //   x_i^L = c_L + A_L · r
    //
    // The target point in local frame:
    //   x_j^L = Ã^T · (x_j^W - c̃)
    //
    // Hessian of ‖x_i^L‖² w.r.t. q_L = (c_L, A_L):
    //   H = k · [I₃, r^T; r, r·r^T]   (same structure as AffineSchurSolver)
    //
    // KKT:  (M + Σ H_c) q_L = M·q_0 + Σ k_c · J_c^T · x_j^L
    //
    //   src_c = m·0 + Σ k · x_j^{local}
    //   src_A = I₀·I + Σ k · (x_j^{local} ⊗ r)

    float contactStiffness = betaLin;

    for (int it = 0; it < iterations; it++) {
        if (has_ground_plane && it > 0) {
            groundContacts.clear();
            for (Rigid* body = bodies; body != nullptr; body = body->next) {
                if (body->mass <= 0) continue;
                for (int v = 0; v < 8; v++) {
                    float3 rLocal = {
                        kLocalVerts[v][0] * body->size.x,
                        kLocalVerts[v][1] * body->size.y,
                        kLocalVerts[v][2] * body->size.z};
                    float3 rWorld = body->affine * rLocal;
                    float3 worldPos = body->positionLin + rWorld;
                    if (worldPos.z - ground_z < 0.0f)
                        groundContacts.push_back({body, rLocal, rWorld,
                                                  worldPos.z - ground_z});
                }
            }
        }

        for (Rigid* body = bodies; body != nullptr; body = body->next) {
            if (body->mass <= 0) continue;

            float k = contactStiffness;
            float3 cTilde = body->inertialLin;
            float3x3 ATilde = body->inertialAff;
            float3x3 ATildeT = transpose(ATilde);

            // ---- Step A: Accumulate Hessian ----
            AffineSchurSolver schur;
            schur.reset();

            for (auto& gc : groundContacts) {
                if (gc.body != body) continue;
                float3 rWorld = body->affine * gc.rLocal;
                float3 worldPos = body->positionLin + rWorld;
                if (worldPos.z - ground_z >= 0.0f) continue;
                schur.accumulate(k, gc.rLocal);
            }

            // ---- Step B: Factor Schur complement ----
            schur.factor(body->mass, body->inertiaMatrix);

            // ---- Step C: Build RHS in local frame ----
            float3 srcC = {0, 0, 0};
            float3x3 srcA = body->inertiaMatrix;

            for (auto& gc : groundContacts) {
                if (gc.body != body) continue;
                float3 rWorld = body->affine * gc.rLocal;
                float3 worldPos = body->positionLin + rWorld;
                if (worldPos.z - ground_z >= 0.0f) continue;

                float3 xj_world = {worldPos.x, worldPos.y, ground_z};
                float3 xj_local = ATildeT * (xj_world - cTilde);

                srcC = srcC + xj_local * k;
                srcA += outer(xj_local, gc.rLocal) * k;
            }

            // ---- Step D: Solve q_L = M^{-1} · src ----
            float3 cL;
            float3x3 AL;
            schur.solve_update(srcC, srcA, cL, AL);

            // ---- Step E: Local → Global ----
            body->positionLin = cTilde + ATilde * cL;
            body->affine = ATilde * AL;
            body->affine = polar_rotation(body->affine);
            body->syncFromAffine();
        }

        for (Force* force = forces; force != nullptr; force = force->next)
            force->updateDual(alpha);
    }

    // ---- Phase 4: Velocity recovery + contact damping ----
    for (Rigid* body = bodies; body != nullptr; body = body->next) {
        if (body->mass <= 0) continue;
        body->prevVelocityLin = body->velocityLin;
        body->velocityLin = (body->positionLin - body->initialLin) / dt;
        body->velocityAng = mat_to_angular(body->affine, body->initialAff) / dt;

        // bool inContact = false;
        // for (auto& gc : groundContacts)
        //     if (gc.body == body) { inContact = true; break; }

        // if (inContact) {
        //     body->velocityLin = body->velocityLin * gamma;
        //     body->velocityAng = body->velocityAng * gamma;
        // }
    }

    if (frame <= 5 || frame % 60 == 0) {
        for (Rigid* body = bodies; body != nullptr; body = body->next) {
            if (body->mass <= 0) continue;
            int nContacts = 0;
            for (auto& gc : groundContacts)
                if (gc.body == body) nContacts++;
            std::printf("[ABD %d] pos=(%.4f,%.4f,%.4f) vel=(%.4f,%.4f,%.4f) contacts=%d\n",
                        frame, body->positionLin.x, body->positionLin.y, body->positionLin.z,
                        body->velocityLin.x, body->velocityLin.y, body->velocityLin.z,
                        nContacts);
        }
    }

    // Upload to GPU for rendering
    if (!gpu_solver_) gpu_solver_ = new GpuSolver();
    soa_.pack(bodies);
    gpu_solver_->upload_bodies(
        soa_.pos_x.data(), soa_.pos_y.data(), soa_.pos_z.data(),
        soa_.quat_x.data(), soa_.quat_y.data(), soa_.quat_z.data(), soa_.quat_w.data(),
        soa_.vel_x.data(), soa_.vel_y.data(), soa_.vel_z.data(),
        soa_.velang_x.data(), soa_.velang_y.data(), soa_.velang_z.data(),
        soa_.prevvel_x.data(), soa_.prevvel_y.data(), soa_.prevvel_z.data(),
        soa_.mass.data(), soa_.moment_x.data(), soa_.moment_y.data(), soa_.moment_z.data(),
        soa_.half_x.data(), soa_.half_y.data(), soa_.half_z.data(),
        soa_.friction.data(),
        soa_.count);
    gpu_state_valid_ = true;
    gpu_state_valid_prev_ = true;
    prev_body_count_ = body_count;
}

}  // namespace avbd
}  // namespace chysx
