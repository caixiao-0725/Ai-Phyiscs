// SPDX-License-Identifier: MIT
// Joint constraint (ball-socket + angular + fracture). Adapted from avbd-demo3d.

#include "avbd_solver.h"

#include <cmath>

namespace chysx {
namespace avbd {

namespace {
inline float3x3 geometricStiffnessBallSocket(int k, float3 v) {
    float3x3 m = diagonal(-v[k], -v[k], -v[k]);
    m[0][k] += v[0];
    m[1][k] += v[1];
    m[2][k] += v[2];
    return m;
}

inline float3 quatLogVec(quat q) {
    q = normalize(q);
    if (q.w < 0.0f)
        q = q * -1.0f;
    float sinHalf = length(q.vec());
    if (sinHalf < 1.0e-6f)
        return q.vec() * 2.0f;
    float angle = 2.0f * std::atan2(sinHalf, q.w);
    return q.vec() * (angle / sinHalf);
}

inline quat jointQuat(const Rigid* body) {
    return body ? body->positionAng : quat{0, 0, 0, 1};
}

inline float3 fixedJointAngularError(const Joint* joint) {
    quat qA = jointQuat(joint->bodyA);
    quat qB = jointQuat(joint->bodyB);
    if (joint->bodyA && joint->restInitialized) {
        quat qErr = qA * inverse(joint->restRelAng) * inverse(qB);
        return quatLogVec(qErr) * joint->torqueArm;
    }
    return (qA - qB) * joint->torqueArm;
}
}  // namespace

Joint::Joint(Solver* solver, Rigid* bodyA, Rigid* bodyB,
             float3 rA, float3 rB,
             float stiffnessLin, float stiffnessAng, float fracture)
    : Force(solver, bodyA, bodyB), rA(rA), rB(rB),
      stiffnessLin(stiffnessLin), stiffnessAng(stiffnessAng),
      fracture(fracture), torqueArm(0.0f), restRelAng{0, 0, 0, 1}, broken(false), restInitialized(false) {
    penaltyLin = penaltyAng = float3{0, 0, 0};
    lambdaLin = lambdaAng = float3{0, 0, 0};
    torqueArm = lengthSq((bodyA ? bodyA->size : float3{0, 0, 0}) + bodyB->size);
}

bool Joint::initialize() {
    RotationMode rmode = solver->rotation_mode;

    // Body-body joints keep their initial rest pose. World-anchor joints
    // (e.g. mouse drag) intentionally refresh because rA is a moving target.
    if (!restInitialized || bodyA == nullptr) {
        C0Lin = (bodyA ? bodyA->transformVec(rA, rmode) : rA) -
                bodyB->transformVec(rB, rmode);

        if (rmode == RotationMode::Affine) {
            float3x3 qA = bodyA ? bodyA->affine : identity3x3();
            float3x3 qB = bodyB->affine;
            C0Ang = mat_to_angular(qA, qB) * torqueArm;
        } else {
            quat qA = jointQuat(bodyA);
            quat qB = jointQuat(bodyB);
            restRelAng = inverse(qB) * qA;
            C0Ang = fixedJointAngularError(this);
        }
        if (bodyA != nullptr)
            restInitialized = true;
    }

    lambdaLin = lambdaLin * solver->alpha * solver->gamma;
    lambdaAng = lambdaAng * solver->alpha * solver->gamma;
    penaltyLin = clamp(penaltyLin * solver->gamma, AVBD_PENALTY_MIN, AVBD_PENALTY_MAX);
    penaltyAng = clamp(penaltyAng * solver->gamma, AVBD_PENALTY_MIN, AVBD_PENALTY_MAX);

    penaltyLin = min(penaltyLin, stiffnessLin);
    penaltyAng = min(penaltyAng, stiffnessAng);

    return !broken;
}

void Joint::updatePrimal(Rigid* body, float alpha,
                         float3x3& lhsLin, float3x3& lhsAng, float3x3& lhsCross,
                         float3& rhsLin, float3& rhsAng) {
    RotationMode rmode = solver->rotation_mode;

    if (lengthSq(penaltyLin) > 0) {
        float3x3 K = diagonal(penaltyLin.x, penaltyLin.y, penaltyLin.z);
        float3 C = (bodyA ? bodyA->transformVec(rA, rmode) : rA) -
                   bodyB->transformVec(rB, rmode);

        if (std::isinf(stiffnessLin))
            C -= C0Lin * alpha;

        float3 F = K * C + lambdaLin;

        float3x3 jLin = body == bodyA ?
            float3x3{1, 0, 0, 0, 1, 0, 0, 0, 1} : float3x3{-1, 0, 0, 0, -1, 0, 0, 0, -1};
        float3 rWorld = body == bodyA ? bodyA->rotateVec(rA, rmode) : bodyB->rotateVec(rB, rmode);
        float3x3 jAng = body == bodyA ? skew(-rWorld) : skew(rWorld);

        float3x3 jLinT = transpose(jLin);
        float3x3 jAngT = transpose(jAng);
        float3x3 jAngTk = jAngT * K;

        lhsLin += jLinT * K * jLin;
        lhsAng += jAngTk * jAng;
        lhsCross += jAngTk * jLin;

        float3 r = body == bodyA ? bodyA->rotateVec(rA, rmode) : bodyB->rotateVec(rB, rmode) * (-1.0f);
        float3x3 H =
            geometricStiffnessBallSocket(0, r) * F[0] +
            geometricStiffnessBallSocket(1, r) * F[1] +
            geometricStiffnessBallSocket(2, r) * F[2];
        lhsAng += diagonalize(H);

        rhsLin += jLinT * F;
        rhsAng += jAngT * F;
    }

    if (lengthSq(penaltyAng) > 0) {
        float3x3 K = diagonal(penaltyAng.x, penaltyAng.y, penaltyAng.z);
        float3 C;
        if (rmode == RotationMode::Affine) {
            float3x3 qA = bodyA ? bodyA->affine : identity3x3();
            float3x3 qB = bodyB->affine;
            C = mat_to_angular(qA, qB) * torqueArm;
        } else {
            C = fixedJointAngularError(this);
        }

        if (std::isinf(stiffnessAng) && !(bodyA && rmode == RotationMode::AxisAngle))
            C -= C0Ang * alpha;

        float3 F = K * C + lambdaAng;

        float3x3 jAng = (body == bodyA ?
            float3x3{1, 0, 0, 0, 1, 0, 0, 0, 1} : float3x3{-1, 0, 0, 0, -1, 0, 0, 0, -1}) * torqueArm;

        lhsAng += transpose(jAng) * K * jAng;
        rhsAng += transpose(jAng) * F;
    }
}

void Joint::updateDual(float alpha) {
    RotationMode rmode = solver->rotation_mode;

    if (lengthSq(penaltyLin) > 0) {
        float3x3 K = diagonal(penaltyLin.x, penaltyLin.y, penaltyLin.z);
        float3 C = (bodyA ? bodyA->transformVec(rA, rmode) : rA) -
                   bodyB->transformVec(rB, rmode);

        if (std::isinf(stiffnessLin)) {
            C -= C0Lin * alpha;
            float3 F = K * C + lambdaLin;
            lambdaLin = F;
        }

        penaltyLin = min(penaltyLin + abs(C) * solver->betaLin, min(stiffnessLin, AVBD_PENALTY_MAX));
    }

    if (lengthSq(penaltyAng) > 0) {
        float3x3 K = diagonal(penaltyAng.x, penaltyAng.y, penaltyAng.z);
        float3 C;
        if (rmode == RotationMode::Affine) {
            float3x3 qA = bodyA ? bodyA->affine : identity3x3();
            float3x3 qB = bodyB->affine;
            C = mat_to_angular(qA, qB) * torqueArm;
        } else {
            C = fixedJointAngularError(this);
        }

        if (std::isinf(stiffnessAng)) {
            if (bodyA && rmode == RotationMode::AxisAngle) {
                float3 F = K * C + lambdaAng;
                lambdaAng = F;
            } else {
            C -= C0Ang * alpha;
            float3 F = K * C + lambdaAng;
            lambdaAng = F;
            }
        }

        penaltyAng = min(penaltyAng + abs(C) * solver->betaAng, min(stiffnessAng, AVBD_PENALTY_MAX));
    }

    if (lengthSq(lambdaAng) > fracture * fracture) {
        penaltyLin = {0, 0, 0};
        penaltyAng = {0, 0, 0};
        lambdaLin = {0, 0, 0};
        lambdaAng = {0, 0, 0};
        broken = true;
    }
}

}  // namespace avbd
}  // namespace chysx
