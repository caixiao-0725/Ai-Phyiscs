// SPDX-License-Identifier: MIT
// AVBD solver main loop. Adapted from avbd-demo3d (Chris Giles, 2026).

#include "avbd_solver.h"

#include <cmath>

namespace chysx {
namespace avbd {

Solver::Solver()
    : bodies(nullptr), forces(nullptr) {
    defaultParams();
}

Solver::~Solver() {
    clear();
}

Rigid* Solver::pick(float3 origin, float3 dir, float3& local) {
    const float epsilon = 1.0e-6f;
    float bestT = INFINITY;
    Rigid* bestBody = nullptr;
    float3 bestLocal = {0, 0, 0};

    for (Rigid* body = bodies; body != nullptr; body = body->next) {
        if (body->mass <= 0.0f)
            continue;

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
}

void Solver::defaultParams() {
    dt = 1.0f / 60.0f;
    gravity = -10.0f;
    iterations = 10;
    betaLin = 10000.0f;
    betaAng = 100.0f;
    alpha = 0.95f;
    gamma = 0.999f;
}

void Solver::step() {
    // Broadphase collision detection (O(n^2))
    for (Rigid* bodyA = bodies; bodyA != nullptr; bodyA = bodyA->next) {
        for (Rigid* bodyB = bodyA->next; bodyB != nullptr; bodyB = bodyB->next) {
            float3 dp = bodyA->positionLin - bodyB->positionLin;
            float r = bodyA->radius + bodyB->radius;
            if (dot(dp, dp) <= r * r && !bodyA->constrainedTo(bodyB))
                new Manifold(this, bodyA, bodyB);
        }
    }

    // Initialize and warmstart forces
    for (Force* force = forces; force != nullptr;) {
        if (!force->initialize()) {
            Force* next = force->next;
            delete force;
            force = next;
        } else {
            force = force->next;
        }
    }

    // Initialize and warmstart bodies
    for (Rigid* body = bodies; body != nullptr; body = body->next) {
        body->inertialLin = body->positionLin + body->velocityLin * dt;
        if (body->mass > 0)
            body->inertialLin += float3{0, 0, gravity} * (dt * dt);
        body->inertialAng = body->positionAng + body->velocityAng * dt;

        float3 accel = (body->velocityLin - body->prevVelocityLin) / dt;
        float accelExt = accel.z * sign(gravity);
        float accelWeight = clamp(accelExt / std::fabs(gravity), 0.0f, 1.0f);
        if (!std::isfinite(accelWeight))
            accelWeight = 0.0f;

        body->initialLin = body->positionLin;
        body->initialAng = body->positionAng;
        if (body->mass > 0) {
            body->positionLin = body->positionLin + body->velocityLin * dt +
                                float3{0, 0, gravity} * (accelWeight * dt * dt);
            body->positionAng = body->positionAng + body->velocityAng * dt;
        }
    }

    // Main solver loop
    for (int it = 0; it < iterations; it++) {
        // Primal update
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
                 force = (force->bodyA == body) ? force->nextA : force->nextB) {
                force->updatePrimal(body, alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
            }

            float3 dxLin, dxAng;
            solve(lhsLin, lhsAng, lhsCross, -rhsLin, -rhsAng, dxLin, dxAng);
            body->positionLin = body->positionLin + dxLin;
            body->positionAng = body->positionAng + dxAng;
        }

        // Dual update
        for (Force* force = forces; force != nullptr; force = force->next) {
            force->updateDual(alpha);
        }
    }

    // Velocity update (BDF1)
    for (Rigid* body = bodies; body != nullptr; body = body->next) {
        body->prevVelocityLin = body->velocityLin;
        if (body->mass > 0) {
            body->velocityLin = (body->positionLin - body->initialLin) / dt;
            body->velocityAng = (body->positionAng - body->initialAng) / dt;
        }
    }
}

}  // namespace avbd
}  // namespace chysx
