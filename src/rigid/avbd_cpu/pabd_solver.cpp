// SPDX-FileCopyrightText: 2026 ChysX contributors
// SPDX-License-Identifier: MIT
// PABD solver — faithful CPU port of PeriDyno PABDConstraintSolver.
// Diagnostic output matches PeriDyno's format for side-by-side comparison.

#include "avbd_solver.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace chysx {
namespace avbd {

// ---- Parameters matching PeriDyno PABD_BoxStack example ----
static constexpr float PABD_DHAT      = 0.001f;
static constexpr float PABD_STIFFNESS = 0.01f;
static constexpr int   PABD_ITERS     = 20;
static constexpr float PABD_RELAX     = 1.0f;

// ---- Barrier functions (Contact type, N=3) ----
static float barrierPositivePart(float k, float dHat, float /*d*/) {
    return 3.0f * k / (dHat * dHat);
}
static float barrierNegativePart(float k, float dHat, float d) {
    float g = d / dHat;
    return -k * (3.0f + g * g) / dHat;
}
static float barrierEnergy(float k, float dHat, float d) {
    float g = d / dHat;
    if (g >= 1.0f) return 0.0f;
    float t = 1.0f - g;
    return k * (t + 0.5f * t * t + (1.0f / 3.0f) * t * t * t);
}

struct PConstraint {
    int id0, id1;       // id0 = surface body, id1 = penetrating body (-1 for ground)
    float3 r0, r1;      // r0 in local frame of id0, r1 in local frame of id1 (or world pos)
    float3 normal0;     // normal in local frame of id0
    float stiffness;
    float dHat;
};

static float3x3 dyadic(float3 a, float3 b) {
    return {{{a.x*b.x, a.x*b.y, a.x*b.z},
             {a.y*b.x, a.y*b.y, a.y*b.z},
             {a.z*b.x, a.z*b.y, a.z*b.z}}};
}

// ============================================================
void Solver::stepCpuPABD()
{
    static int frame = 0;
    frame++;

    int numBodies = 0;
    for (Rigid* b = bodies; b; b = b->next) numBodies++;

    std::vector<Rigid*> blist(numBodies);
    { int i = 0; for (Rigid* b = bodies; b; b = b->next) blist[i++] = b; }

    bool diag = (frame <= 3 || frame % 60 == 0);

    // DIAG: PRE-PREDICT positions
    if (diag) {
        std::printf("[PABD-CX f%d] bodies=%d dt=%.4f dHat=%.6f\n",
            frame, numBodies, dt, PABD_DHAT);
        std::printf("  PRE-PREDICT positions:\n");
        for (int i = 0; i < numBodies && i < 20; i++) {
            Rigid* b = blist[i];
            std::printf("    b%d pos=(%.6f,%.6f,%.6f) vel=(%.6f,%.6f,%.6f)\n",
                i, b->positionLin.x, b->positionLin.y, b->positionLin.z,
                b->velocityLin.x, b->velocityLin.y, b->velocityLin.z);
        }
    }

    // ---- 1. Predict: velocity solver disabled, just apply gravity and advance ----
    for (int i = 0; i < numBodies; i++) {
        Rigid* b = blist[i];
        b->initialLin = b->positionLin;
        b->initialAff = b->affine;
        b->initialAng = b->positionAng;
        if (b->mass <= 0) continue;

        b->velocityLin += float3{0, 0, gravity} * dt;
        b->positionLin += b->velocityLin * dt;

        float3x3 dR = skew(b->velocityAng * dt);
        b->affine = (identity3x3() + dR) * b->affine;
        b->affine = polar_rotation(b->affine);
        b->syncFromAffine();
    }

    // ---- 2. Collision detection ----
    // Generate constraints matching PeriDyno's NeighborElementQuery convention:
    //   r0 = R0^{-1} * (contact1 - center0)   (surface point in body0's local frame)
    //   r1 = R1^{-1} * (contact2 - center1)   (penetrating point in body1's local frame)
    //   normal0 = R0^{-1} * normal_world       (normal in body0's local frame, from body0 toward body1)

    static const float kV[8][3] = {
        {-0.5f,-0.5f,-0.5f},{+0.5f,-0.5f,-0.5f},
        {+0.5f,+0.5f,-0.5f},{-0.5f,+0.5f,-0.5f},
        {-0.5f,-0.5f,+0.5f},{+0.5f,-0.5f,+0.5f},
        {+0.5f,+0.5f,+0.5f},{-0.5f,+0.5f,+0.5f}};

    std::vector<PConstraint> cons;

    // Ground contacts: box is id0 (r0 = vertex), ground is id1=-1 (r1 = world pos on ground)
    // normal0 = R^{-1} * (0,0,-1) = toward ground, matching PeriDyno convention
    if (has_ground_plane) {
        for (int bi = 0; bi < numBodies; bi++) {
            Rigid* body = blist[bi];
            if (body->mass <= 0) continue;
            float3x3 Rinv = transpose(body->affine);
            for (int v = 0; v < 8; v++) {
                float3 rL = {kV[v][0]*body->size.x, kV[v][1]*body->size.y, kV[v][2]*body->size.z};
                float3 wp = body->positionLin + body->affine * rL;
                if (wp.z >= ground_z + 0.1f) continue;

                float3 groundPt = {wp.x, wp.y, ground_z};
                PConstraint c;
                c.id0 = bi;
                c.id1 = -1;
                c.r0 = rL;
                c.r1 = groundPt;
                c.normal0 = Rinv * float3{0, 0, -1};
                c.stiffness = PABD_STIFFNESS;
                c.dHat = PABD_DHAT;
                cons.push_back(c);
            }
        }
    }

    // Box-box: SAT-style face contact detection
    // For each pair of potentially colliding boxes, find the face axis with
    // minimum separation. Then generate vertex-vs-face constraints for vertices
    // of each box near the contact plane of the other.
    for (int ai = 0; ai < numBodies; ai++) {
        for (int bi2 = ai+1; bi2 < numBodies; bi2++) {
            Rigid* a = blist[ai]; Rigid* b = blist[bi2];
            if (a->mass <= 0 && b->mass <= 0) continue;
            if (a->constrainedTo(b)) continue;
            float3 diff = a->positionLin - b->positionLin;
            float rsum = a->radius + b->radius + 0.1f;
            if (dot(diff, diff) > rsum * rsum) continue;

            float3 halfA = a->size * 0.5f;
            float3 halfB = b->size * 0.5f;
            float3x3 RAinv = transpose(a->affine);
            float3x3 RBinv = transpose(b->affine);

            // SAT: test 6 face axes (3 from A, 3 from B)
            // Find the axis with minimum overlap (or maximum separation)
            float bestSep = -1e10f;
            int bestAxis = -1;     // 0-2 = A face, 3-5 = B face
            float3 bestNormalW = {0,0,0};

            for (int ax = 0; ax < 3; ax++) {
                // A's face axis
                float3 axDir = {a->affine[0][ax], a->affine[1][ax], a->affine[2][ax]};
                float projA = halfA[ax];
                float projB = std::fabs(dot(axDir, {b->affine[0][0], b->affine[1][0], b->affine[2][0]})) * halfB.x
                            + std::fabs(dot(axDir, {b->affine[0][1], b->affine[1][1], b->affine[2][1]})) * halfB.y
                            + std::fabs(dot(axDir, {b->affine[0][2], b->affine[1][2], b->affine[2][2]})) * halfB.z;
                float dist = std::fabs(dot(diff, axDir));
                float sep = dist - projA - projB;
                if (sep > bestSep) {
                    bestSep = sep;
                    bestAxis = ax;
                    bestNormalW = axDir;
                    if (dot(diff, axDir) < 0) bestNormalW = bestNormalW * (-1.0f);
                }
            }
            for (int ax = 0; ax < 3; ax++) {
                float3 axDir = {b->affine[0][ax], b->affine[1][ax], b->affine[2][ax]};
                float projA = std::fabs(dot(axDir, {a->affine[0][0], a->affine[1][0], a->affine[2][0]})) * halfA.x
                            + std::fabs(dot(axDir, {a->affine[0][1], a->affine[1][1], a->affine[2][1]})) * halfA.y
                            + std::fabs(dot(axDir, {a->affine[0][2], a->affine[1][2], a->affine[2][2]})) * halfA.z;
                float projB = halfB[ax];
                float dist = std::fabs(dot(diff, axDir));
                float sep = dist - projA - projB;
                if (sep > bestSep) {
                    bestSep = sep;
                    bestAxis = ax + 3;
                    bestNormalW = axDir;
                    if (dot(diff, axDir) < 0) bestNormalW = bestNormalW * (-1.0f);
                }
            }

            // Only generate contacts if separation is small enough
            if (bestSep > 0.01f) continue;

            // bestNormalW points from B to A (along diff direction)
            // Generate vertex constraints: vertices of B near A's surface, and vice versa.
            // Convention: id0 = reference body (surface owner), id1 = incident body (vertex owner)
            // normal0 = outward normal of id0 in id0's local frame

            // Vertices of B penetrating/touching A → A is id0 (surface), B is id1
            for (int v = 0; v < 8; v++) {
                float3 rB = {kV[v][0]*b->size.x, kV[v][1]*b->size.y, kV[v][2]*b->size.z};
                float3 wp = b->positionLin + b->affine * rB;
                // Signed distance along contact normal: positive = separated
                float signedDist = dot(wp - a->positionLin, bestNormalW * (-1.0f));
                // Vertex projection onto A's surface along the normal
                float3 lA = RAinv * (wp - a->positionLin);
                // Check if vertex projects within A's face (in the other two axes)
                bool withinFace = true;
                for (int ax2 = 0; ax2 < 3; ax2++) {
                    if (std::fabs(lA[ax2]) > halfA[ax2] + 0.01f) { withinFace = false; break; }
                }
                if (!withinFace) continue;

                float3 surfPt = lA;
                // Project to nearest face along the contact normal direction
                float3 nlA = RAinv * (bestNormalW * (-1.0f));
                int projAx = 0;
                float maxNL = 0;
                for (int ax2 = 0; ax2 < 3; ax2++) {
                    if (std::fabs(nlA[ax2]) > maxNL) { maxNL = std::fabs(nlA[ax2]); projAx = ax2; }
                }
                surfPt[projAx] = halfA[projAx] * (nlA[projAx] >= 0 ? 1.0f : -1.0f);
                float3 fnA = {0,0,0};
                fnA[projAx] = (nlA[projAx] >= 0) ? 1.0f : -1.0f;

                PConstraint c;
                c.id0 = ai;
                c.id1 = bi2;
                c.r0 = surfPt;
                c.r1 = rB;
                c.normal0 = fnA;
                c.stiffness = PABD_STIFFNESS;
                c.dHat = PABD_DHAT;
                cons.push_back(c);
            }

            // Vertices of A penetrating/touching B → B is id0, A is id1
            for (int v = 0; v < 8; v++) {
                float3 rA = {kV[v][0]*a->size.x, kV[v][1]*a->size.y, kV[v][2]*a->size.z};
                float3 wp = a->positionLin + a->affine * rA;
                float3 lB = RBinv * (wp - b->positionLin);
                bool withinFace = true;
                for (int ax2 = 0; ax2 < 3; ax2++) {
                    if (std::fabs(lB[ax2]) > halfB[ax2] + 0.01f) { withinFace = false; break; }
                }
                if (!withinFace) continue;

                float3 surfPt = lB;
                float3 nlB = RBinv * bestNormalW;
                int projAx = 0;
                float maxNL = 0;
                for (int ax2 = 0; ax2 < 3; ax2++) {
                    if (std::fabs(nlB[ax2]) > maxNL) { maxNL = std::fabs(nlB[ax2]); projAx = ax2; }
                }
                surfPt[projAx] = halfB[projAx] * (nlB[projAx] >= 0 ? 1.0f : -1.0f);
                float3 fnB = {0,0,0};
                fnB[projAx] = (nlB[projAx] >= 0) ? 1.0f : -1.0f;

                PConstraint c;
                c.id0 = bi2;
                c.id1 = ai;
                c.r0 = surfPt;
                c.r1 = rA;
                c.normal0 = fnB;
                c.stiffness = PABD_STIFFNESS;
                c.dHat = PABD_DHAT;
                cons.push_back(c);
            }
        }
    }

    int nCons = (int)cons.size();

    // DIAG: POST-PREDICT positions
    if (diag) {
        std::printf("  POST-PREDICT positions (contactConstraints=%d):\n", nCons);
        for (int i = 0; i < numBodies && i < 20; i++) {
            Rigid* b = blist[i];
            std::printf("    b%d pos=(%.6f,%.6f,%.6f)\n",
                i, b->positionLin.x, b->positionLin.y, b->positionLin.z);
        }
    }

    // ---- 3. PABD Position solver ----
    // Save predicted state as q* (reference frame)
    std::vector<float3>   qStarA(numBodies);
    std::vector<float3x3> qStarB(numBodies);
    std::vector<float3>   cenGlobal(numBodies);
    std::vector<float3x3> rotGlobal(numBodies);
    std::vector<float3>   cenLocal(numBodies);
    std::vector<float3x3> rotLocal(numBodies);
    for (int i = 0; i < numBodies; i++) {
        qStarA[i] = blist[i]->positionLin;
        qStarB[i] = blist[i]->affine;
        cenGlobal[i] = qStarA[i];
        rotGlobal[i] = qStarB[i];
    }

    // Transform2Local
    for (int i = 0; i < numBodies; i++) {
        float3x3 RT = transpose(qStarB[i]);
        cenLocal[i] = RT * (cenGlobal[i] - qStarA[i]);
        rotLocal[i] = RT * rotGlobal[i];
    }

    // Allocate solver arrays
    std::vector<float>    sumA(numBodies);
    std::vector<float3>   sumB(numBodies), sumC(numBodies);
    std::vector<float3x3> sumD(numBodies);
    std::vector<float3>   sum3v(numBodies), sum4v(numBodies);
    std::vector<float3x3> sum5m(numBodies), sum6m(numBodies);
    std::vector<float>    energy(numBodies);
    std::vector<float>    m1A(numBodies);
    std::vector<float3>   m1B(numBodies), m1C(numBodies);
    std::vector<float3x3> m1D(numBodies);
    std::vector<float3>   srcA(numBodies);
    std::vector<float3x3> srcBm(numBodies);

    int posIters = (iterations > 0) ? iterations : PABD_ITERS;

    for (int it = 0; it < posIters; it++) {
        // Transform2Global
        for (int i = 0; i < numBodies; i++) {
            cenGlobal[i] = qStarA[i] + qStarB[i] * cenLocal[i];
            rotGlobal[i] = qStarB[i] * rotLocal[i];
        }

        // Reset sums
        for (int i = 0; i < numBodies; i++) {
            sumA[i] = 0; sumB[i] = {0,0,0}; sumC[i] = {0,0,0};
            sumD[i] = {{{0,0,0},{0,0,0},{0,0,0}}};
            sum3v[i] = {0,0,0}; sum4v[i] = {0,0,0};
            sum5m[i] = {{{0,0,0},{0,0,0},{0,0,0}}};
            sum6m[i] = {{{0,0,0},{0,0,0},{0,0,0}}};
            energy[i] = 0;
        }

        // CalcSumsForMat1
        int nActive = 0;
        for (int ci = 0; ci < nCons; ci++) {
            auto& con = cons[ci];
            int id0 = con.id0, id1 = con.id1;
            float3 x0 = cenGlobal[id0] + rotGlobal[id0] * con.r0;
            float3 x1 = (id1 >= 0) ? (cenGlobal[id1] + rotGlobal[id1] * con.r1) : con.r1;
            float3 ng = rotGlobal[id0] * con.normal0;
            // Keep signed d: PeriDyno's contact barrier uses negative d for deep penetration.
            float d = dot(x1 - x0, ng) + 2.0f * con.dHat;

            if (d >= con.dHat) continue;
            nActive++;
            float alpha = dt * dt * barrierPositivePart(con.stiffness, con.dHat, d);
            sumA[id0] += alpha;
            sumB[id0] += con.r0 * alpha;
            sumC[id0] += con.r0 * alpha;
            sumD[id0] = sumD[id0] + dyadic(con.r0, con.r0) * alpha;
            if (id1 >= 0) {
                sumA[id1] += alpha;
                sumB[id1] += con.r1 * alpha;
                sumC[id1] += con.r1 * alpha;
                sumD[id1] = sumD[id1] + dyadic(con.r1, con.r1) * alpha;
            }
        }

        // CalcMat1Inv
        for (int i = 0; i < numBodies; i++) {
            if (blist[i]->mass <= 0 || sumA[i] < 1e-12f) {
                m1A[i] = 0; m1B[i] = {0,0,0}; m1C[i] = {0,0,0};
                m1D[i] = {{{0,0,0},{0,0,0},{0,0,0}}};
                continue;
            }
            float k = blist[i]->mass + sumA[i];
            float3 B = sumB[i] / k;
            float3 C = sumC[i] / k;
            float3x3 I = blist[i]->inertiaMatrix;
            float3x3 SP;
            for (int r = 0; r < 3; r++)
                for (int c2 = 0; c2 < 3; c2++)
                    SP[r][c2] = (I[r][c2] + sumD[i][r][c2]) / k - C[r] * B[c2];
            float3x3 SPk;
            for (int r = 0; r < 3; r++)
                for (int c2 = 0; c2 < 3; c2++)
                    SPk[r][c2] = SP[r][c2] * k;
            float3x3 SI = inverse3x3(SPk);
            for (int r = 0; r < 3; r++)
                for (int c2 = 0; c2 < 3; c2++)
                    SI[r][c2] *= k;
            m1D[i] = SI / k;
            m1C[i] = (SI * C) * (-1.0f / k);
            m1B[i] = (transpose(SI) * B) * (-1.0f / k);
            m1A[i] = (1.0f + dot(B, SI * C)) / k;
        }

        // CalcSumsForMat2
        for (int ci = 0; ci < nCons; ci++) {
            auto& con = cons[ci];
            int id0 = con.id0, id1 = con.id1;
            float3 x0 = cenGlobal[id0] + rotGlobal[id0] * con.r0;
            float3 x1 = (id1 >= 0) ? (cenGlobal[id1] + rotGlobal[id1] * con.r1) : con.r1;
            float3 ng = rotGlobal[id0] * con.normal0;
            float d = dot(x1 - x0, ng) + 2.0f * con.dHat;
            if (d >= con.dHat) continue;
            float alpha = dt * dt * barrierPositivePart(con.stiffness, con.dHat, d);
            float beta  = dt * dt * barrierNegativePart(con.stiffness, con.dHat, d);
            float Ei    = barrierEnergy(con.stiffness, con.dHat, d);

            float3x3 qSBT0 = transpose(qStarB[id0]);
            float3 n0l = qSBT0 * ng;

            if (id1 >= 0) {
                float3x3 qSBT1 = transpose(qStarB[id1]);
                float3 n1l = qSBT1 * (ng * (-1.0f));
                float3 x1_local = qSBT0 * (x1 - qStarA[id0]);
                float3 x0_local = qSBT1 * (x0 - qStarA[id1]);
                sum3v[id0] += n0l * beta;
                sum4v[id0] += x1_local * alpha;
                sum5m[id0] = sum5m[id0] + dyadic(n0l, con.r0) * beta;
                sum6m[id0] = sum6m[id0] + dyadic(x1_local, con.r0) * alpha;
                energy[id0] += Ei;
                sum3v[id1] += n1l * beta;
                sum4v[id1] += x0_local * alpha;
                sum5m[id1] = sum5m[id1] + dyadic(n1l, con.r1) * beta;
                sum6m[id1] = sum6m[id1] + dyadic(x0_local, con.r1) * alpha;
                energy[id1] += Ei;
            } else {
                float3 x1_local = qSBT0 * (x1 - qStarA[id0]);
                sum3v[id0] += n0l * beta;
                sum4v[id0] += x1_local * alpha;
                sum5m[id0] = sum5m[id0] + dyadic(n0l, con.r0) * beta;
                sum6m[id0] = sum6m[id0] + dyadic(x1_local, con.r0) * alpha;
                energy[id0] += Ei;
            }
        }

        // CalcMat2
        for (int i = 0; i < numBodies; i++) {
            float3x3 I = blist[i]->inertiaMatrix;
            srcA[i] = sum3v[i] + sum4v[i];
            srcBm[i] = I + sum5m[i] + sum6m[i];
        }

        // CalcQ
        for (int i = 0; i < numBodies; i++) {
            if (blist[i]->mass <= 0 || sumA[i] < 1e-12f) continue;
            cenLocal[i].x = m1A[i] * srcA[i].x + dot(srcBm[i][0], m1B[i]);
            cenLocal[i].y = m1A[i] * srcA[i].y + dot(srcBm[i][1], m1B[i]);
            cenLocal[i].z = m1A[i] * srcA[i].z + dot(srcBm[i][2], m1B[i]);
            for (int r = 0; r < 3; r++) {
                rotLocal[i][r].x = srcA[i][r] * m1C[i].x + dot(srcBm[i][r], m1D[i][0]);
                rotLocal[i][r].y = srcA[i][r] * m1C[i].y + dot(srcBm[i][r], m1D[i][1]);
                rotLocal[i][r].z = srcA[i][r] * m1C[i].z + dot(srcBm[i][r], m1D[i][2]);
            }
        }

        // LineSearch
        for (int i = 0; i < numBodies; i++) {
            if (blist[i]->mass <= 0 || sumA[i] < 1e-12f) continue;
            float3x3 RT = transpose(qStarB[i]);
            float3 cRef = RT * (cenGlobal[i] - qStarA[i]);
            float3x3 rRef = RT * rotGlobal[i];

            float3 partA = cRef + rRef * sumB[i];
            float a = std::fabs(dot(partA - srcA[i], cenLocal[i] - cRef));

            float3x3 sumDI;
            for (int r = 0; r < 3; r++)
                for (int c2 = 0; c2 < 3; c2++)
                    sumDI[r][c2] = sumD[i][r][c2] + blist[i]->inertiaMatrix[r][c2];
            float3x3 partB = dyadic(cRef, sumC[i]) + rRef * transpose(sumDI);
            float bv = 0;
            for (int r = 0; r < 3; r++)
                for (int c2 = 0; c2 < 3; c2++)
                    bv += std::fabs((partB[r][c2] - srcBm[i][r][c2]) * (rotLocal[i][r][c2] - rRef[r][c2]));

            float c = energy[i] * dt * dt;
            float al = c / (a + bv + 1e-10f);
            if (al > 1.0f) al = 1.0f;
            al *= PABD_RELAX;

            cenLocal[i] = cRef + (cenLocal[i] - cRef) * al;

            float3x3 matR;
            for (int r = 0; r < 3; r++)
                for (int c2 = 0; c2 < 3; c2++)
                    matR[r][c2] = rRef[r][c2] + al * (rotLocal[i][r][c2] - rRef[r][c2]);
            rotLocal[i] = polar_rotation(matR);
        }

        // DIAG: after first iteration
        if (it == 0 && diag) {
            std::printf("  IT0 after contact solve:\n");
            for (int i = 0; i < numBodies && i < 20; i++) {
                std::printf("    b%d sumA=%.6f m1A=%.6f srcA=(%.6f,%.6f,%.6f) cenLocal=(%.6f,%.6f,%.6f) energy=%.6f\n",
                    i, sumA[i], m1A[i],
                    srcA[i].x, srcA[i].y, srcA[i].z,
                    cenLocal[i].x, cenLocal[i].y, cenLocal[i].z,
                    energy[i]);
            }
        }
    }

    // Transform2Global (final)
    for (int i = 0; i < numBodies; i++) {
        cenGlobal[i] = qStarA[i] + qStarB[i] * cenLocal[i];
        rotGlobal[i] = qStarB[i] * rotLocal[i];
    }

    // ---- 4. Velocity recovery (PeriDyno Solution 1) ----
    for (int i = 0; i < numBodies; i++) {
        Rigid* b = blist[i];
        if (b->mass <= 0) continue;

        float3 p_prev = qStarA[i];  // predicted position (before solver)
        b->positionLin = cenGlobal[i];

        float3 dv = (cenGlobal[i] - p_prev) / dt;
        b->velocityLin = b->velocityLin + dv;

        float3x3 R_old = qStarB[i];
        float3x3 R_new = rotGlobal[i];
        float3x3 dRM = (R_new - R_old) * transpose(R_old);
        float3 dw = {
            0.5f * (dRM[2][1] - dRM[1][2]),
            0.5f * (dRM[0][2] - dRM[2][0]),
            0.5f * (dRM[1][0] - dRM[0][1])
        };
        dw = dw / dt;
        b->velocityAng = b->velocityAng + dw;

        quat dq = (quat{dw.x, dw.y, dw.z, 0.0f} * b->positionAng) * (0.5f * dt);
        b->positionAng = normalize(b->positionAng + dq);
        b->affine = rotation(b->positionAng);
    }

    // DIAG: FINAL positions
    if (diag) {
        std::printf("  FINAL positions:\n");
        for (int i = 0; i < numBodies && i < 20; i++) {
            Rigid* b = blist[i];
            std::printf("    b%d pos=(%.6f,%.6f,%.6f) vel=(%.6f,%.6f,%.6f)\n",
                i, b->positionLin.x, b->positionLin.y, b->positionLin.z,
                b->velocityLin.x, b->velocityLin.y, b->velocityLin.z);
        }
        std::printf("  --- end frame %d ---\n", frame);
    }

    // Visualization
    abd_contact_points.clear();
    for (auto& c : cons) {
        float3 wp = cenGlobal[c.id0] + rotGlobal[c.id0] * c.r0;
        abd_contact_points.push_back(wp);
    }
}

}  // namespace avbd
}  // namespace chysx
