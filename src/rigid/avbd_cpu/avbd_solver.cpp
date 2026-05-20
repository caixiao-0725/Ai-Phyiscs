// SPDX-License-Identifier: MIT
// AVBD solver main loop. Adapted from avbd-demo3d (Chris Giles, 2026).

#include "avbd_solver.h"
#include "avbd_broadphase_gpu.h"

#include <cmath>
#include <algorithm>
#include <cstdio>
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
    }
}

Solver::Solver()
    : bodies(nullptr), forces(nullptr), broadphase_gpu_(nullptr) {
    defaultParams();
}

Solver::~Solver() {
    clear();
    delete broadphase_gpu_;
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
    // Pack linked list into SoA for GPU broadphase
    soa_.pack(bodies);

#ifdef AVBD_VALIDATE_BROADPHASE
    // CPU broadphase for validation: collect sorted pair set
    std::vector<std::pair<int,int>> cpu_pairs;
    for (int ia = 0; ia < soa_.count; ia++) {
        for (int ib = ia + 1; ib < soa_.count; ib++) {
            Rigid* bodyA = soa_.body_ptrs[ia];
            Rigid* bodyB = soa_.body_ptrs[ib];
            float3 dp = bodyA->positionLin - bodyB->positionLin;
            float r = bodyA->radius + bodyB->radius;
            if (dot(dp, dp) <= r * r)
                cpu_pairs.push_back({ia, ib});
        }
    }
#endif

    // GPU broadphase
    if (!broadphase_gpu_) {
        broadphase_gpu_ = new BroadphaseGPU();
        int max_pairs = soa_.count * (soa_.count - 1) / 2;
        if (max_pairs < 256) max_pairs = 256;
        broadphase_gpu_->build(soa_.count, max_pairs);
    }
    int max_out = soa_.count * (soa_.count - 1) / 2;
    if (max_out < 1) max_out = 1;
    pairs_a_.resize(max_out);
    pairs_b_.resize(max_out);

    int pair_count = broadphase_gpu_->query(
        soa_.pos_x.data(), soa_.pos_y.data(), soa_.pos_z.data(),
        soa_.quat_x.data(), soa_.quat_y.data(), soa_.quat_z.data(), soa_.quat_w.data(),
        soa_.half_x.data(), soa_.half_y.data(), soa_.half_z.data(),
        soa_.mass.data(),
        soa_.count, pairs_a_.data(), pairs_b_.data());

#ifdef AVBD_VALIDATE_BROADPHASE
    {
        // Validate: GPU pairs should be a superset of CPU sphere-test pairs
        // (GPU uses AABB which is looser than sphere).
        std::vector<std::pair<int,int>> gpu_pairs;
        for (int k = 0; k < pair_count; k++) {
            int a = pairs_a_[k], b = pairs_b_[k];
            if (a > b) std::swap(a, b);
            gpu_pairs.push_back({a, b});
        }
        std::sort(gpu_pairs.begin(), gpu_pairs.end());
        std::sort(cpu_pairs.begin(), cpu_pairs.end());

        int missing = 0;
        for (auto& cp : cpu_pairs) {
            if (!std::binary_search(gpu_pairs.begin(), gpu_pairs.end(), cp)) {
                fprintf(stderr, "AVBD VALIDATE: CPU pair (%d,%d) missing from GPU!\n",
                        cp.first, cp.second);
                missing++;
            }
        }
        static int frame_count = 0;
        if (frame_count % 60 == 0) {
            fprintf(stderr, "AVBD VALIDATE [frame %d]: bodies=%d cpu_pairs=%d gpu_pairs=%d missing=%d\n",
                    frame_count, soa_.count, (int)cpu_pairs.size(), pair_count, missing);
        }
        frame_count++;
    }
#endif

    // CPU narrowphase: for each GPU broadphase pair, run SAT
    for (int k = 0; k < pair_count; k++) {
        Rigid* bodyA = soa_.body_ptrs[pairs_a_[k]];
        Rigid* bodyB = soa_.body_ptrs[pairs_b_[k]];
        if (!bodyA->constrainedTo(bodyB))
            new Manifold(this, bodyA, bodyB);
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
