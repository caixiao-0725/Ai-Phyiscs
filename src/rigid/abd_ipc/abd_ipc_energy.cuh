// SPDX-License-Identifier: MIT
// OrthoPotential energy for the ABD formulation.
//
// Penalizes deviation of the affine matrix A from SO(3):
//   Ψ(A) = Σ_{i<j} (a_i · a_j)² + Σ_i (|a_i|² − 1)²
//   E = κ * V * Ψ(A)
//
// where a_0, a_1, a_2 are the rows of A.

#pragma once

#include "abd_ipc_types.cuh"

namespace chysx {
namespace abd_ipc {

// ============================================================================
// OrthoPotential Ψ(A): energy, gradient (9D), Hessian (9x9)
// ============================================================================

// ============================================================================
// Ψ, dΨ/dq, d²Ψ/dq² — ported from libuipc sym/ortho_potential.inl
// E = κ * ‖AA^T - I‖²_F (Frobenius norm squared)
// Input: 12-DOF q.  Only q[3..11] (affine rows) contribute.
// ============================================================================

// Ψ(q) scalar value.
CHYSX_HDI float ortho_psi(const Vec12f& q, float kappa) {
    float R;
    R = kappa*(2*powf(q[10]*q[4] + q[11]*q[5] + q[3]*q[9], 2)
            + 2*powf(q[10]*q[7] + q[11]*q[8] + q[6]*q[9], 2)
            + 2*powf(q[3]*q[6] + q[4]*q[7] + q[5]*q[8], 2)
            + powf(q[10]*q[10] + q[11]*q[11] + q[9]*q[9] - 1, 2)
            + powf(q[3]*q[3] + q[4]*q[4] + q[5]*q[5] - 1, 2)
            + powf(q[6]*q[6] + q[7]*q[7] + q[8]*q[8] - 1, 2));
    return R;
}

// dΨ/dq[3..11] — 9-vector gradient.
CHYSX_HDI Vec9f ortho_grad(const Vec12f& q, float kappa) {
    float x0 = 4*q[3]*q[6] + 4*q[4]*q[7] + 4*q[5]*q[8];
    float x1 = 4*q[10]*q[4] + 4*q[11]*q[5] + 4*q[3]*q[9];
    float x2 = 4*q[3]*q[3] + 4*q[4]*q[4] + 4*q[5]*q[5] - 4;
    float x3 = 4*q[10]*q[7] + 4*q[11]*q[8] + 4*q[6]*q[9];
    float x4 = 4*q[6]*q[6] + 4*q[7]*q[7] + 4*q[8]*q[8] - 4;
    float x5 = 4*q[10]*q[10] + 4*q[11]*q[11] + 4*q[9]*q[9] - 4;
    Vec9f R;
    R[0] = kappa*(q[3]*x2 + q[6]*x0 + q[9]*x1);
    R[1] = kappa*(q[10]*x1 + q[4]*x2 + q[7]*x0);
    R[2] = kappa*(q[11]*x1 + q[5]*x2 + q[8]*x0);
    R[3] = kappa*(q[3]*x0 + q[6]*x4 + q[9]*x3);
    R[4] = kappa*(q[10]*x3 + q[4]*x0 + q[7]*x4);
    R[5] = kappa*(q[11]*x3 + q[5]*x0 + q[8]*x4);
    R[6] = kappa*(q[3]*x1 + q[6]*x3 + q[9]*x5);
    R[7] = kappa*(q[10]*x5 + q[4]*x1 + q[7]*x3);
    R[8] = kappa*(q[11]*x5 + q[5]*x1 + q[8]*x3);
    return R;
}

// d²Ψ/dq[3..11]² — 9×9 Hessian.
CHYSX_HDI Mat9f ortho_hessian(const Vec12f& q, float kappa) {
    float x0 = q[3]*q[3]; float x1 = q[4]*q[4]; float x2 = 4*x1;
    float x3 = q[6]*q[6]; float x4 = 4*x3; float x5 = x2+x4;
    float x6 = q[9]*q[9]; float x7 = 4*x6;
    float x8 = q[5]*q[5]; float x9 = 4*x8-4; float x10 = x7+x9;
    float x11 = 4*q[9]; float x12 = q[10]*x11;
    float x13 = 8*q[3]; float x14 = 4*q[6]; float x15 = q[7]*x14;
    float x16 = kappa*(q[4]*x13+x12+x15);
    float x17 = q[11]*x11; float x18 = q[8]*x14;
    float x19 = kappa*(q[5]*x13+x17+x18);
    float x20 = 4*q[4]; float x21 = q[7]*x20;
    float x22 = 4*q[5]; float x23 = q[8]*x22;
    float x24 = kappa*(q[6]*x13+x21+x23);
    float x25 = kappa*x14; float x26 = q[4]*x25; float x27 = q[5]*x25;
    float x28 = q[10]*x20; float x29 = q[11]*x22;
    float x30 = kappa*(q[9]*x13+x28+x29);
    float x31 = kappa*x11; float x32 = q[4]*x31; float x33 = q[5]*x31;
    float x34 = q[10]*q[10]; float x35 = 4*x34; float x36 = 4*x0;
    float x37 = q[7]*q[7]; float x38 = 4*x37; float x39 = x36+x38;
    float x40 = 4*q[10]; float x41 = q[11]*x40;
    float x42 = 8*q[4]; float x43 = 4*q[8]; float x44 = q[7]*x43;
    float x45 = kappa*(q[5]*x42+x41+x44);
    float x46 = kappa*q[7]; float x47 = 4*q[3]; float x48 = x46*x47;
    float x49 = q[3]*x14;
    float x50 = kappa*(q[7]*x42+x23+x49);
    float x51 = x22*x46; float x52 = kappa*q[3];
    float x53 = x40*x52; float x54 = q[3]*x11;
    float x55 = kappa*(q[10]*x42+x29+x54);
    float x56 = kappa*q[10]*x22;
    float x57 = q[8]*q[8]; float x58 = 4*x57; float x59 = x58-4;
    float x60 = q[11]*q[11]; float x61 = 4*x60; float x62 = x36+x61;
    float x63 = x43*x52; float x64 = kappa*x20;
    float x65 = q[8]*x64; float x66 = 8*q[5];
    float x67 = kappa*(q[8]*x66+x21+x49);
    float x68 = kappa*q[11]*x47; float x69 = q[11]*x64;
    float x70 = kappa*(q[11]*x66+x28+x54);
    float x71 = q[3]*x20; float x72 = 8*q[6];
    float x73 = kappa*(q[7]*x72+x12+x71);
    float x74 = q[3]*x22;
    float x75 = kappa*(q[8]*x72+x17+x74);
    float x76 = q[7]*x40; float x77 = q[11]*x43;
    float x78 = kappa*(q[9]*x72+x76+x77);
    float x79 = q[7]*x31; float x80 = q[8]*x31;
    float x81 = q[5]*x20; float x82 = 8*q[7];
    float x83 = kappa*(q[8]*x82+x41+x81);
    float x84 = q[10]*x25; float x85 = q[6]*x11;
    float x86 = kappa*(q[10]*x82+x77+x85);
    float x87 = kappa*q[8]*x40; float x88 = x38+x61;
    float x89 = q[11]*x25; float x90 = 4*q[11]*x46;
    float x91 = 8*q[11];
    float x92 = kappa*(q[8]*x91+x76+x85);
    float x93 = kappa*(8*q[10]*q[9]+x15+x71);
    float x94 = kappa*(q[9]*x91+x18+x74);
    float x95 = kappa*(q[10]*x91+x44+x81);
    Mat9f H;
    H(0,0) = kappa*(12*x0+x10+x5); H(0,1)=x16; H(0,2)=x19;
    H(0,3)=x24; H(0,4)=x26; H(0,5)=x27;
    H(0,6)=x30; H(0,7)=x32; H(0,8)=x33;
    H(1,0)=x16; H(1,1)=kappa*(12*x1+x35+x39+x9);
    H(1,2)=x45; H(1,3)=x48; H(1,4)=x50;
    H(1,5)=x51; H(1,6)=x53; H(1,7)=x55; H(1,8)=x56;
    H(2,0)=x19; H(2,1)=x45; H(2,2)=kappa*(x2+x59+x62+12*x8);
    H(2,3)=x63; H(2,4)=x65; H(2,5)=x67;
    H(2,6)=x68; H(2,7)=x69; H(2,8)=x70;
    H(3,0)=x24; H(3,1)=x48; H(3,2)=x63;
    H(3,3)=kappa*(12*x3+x39+x59+x7); H(3,4)=x73;
    H(3,5)=x75; H(3,6)=x78; H(3,7)=x79; H(3,8)=x80;
    H(4,0)=x26; H(4,1)=x50; H(4,2)=x65;
    H(4,3)=x73; H(4,4)=kappa*(x35+12*x37+x5+x59);
    H(4,5)=x83; H(4,6)=x84; H(4,7)=x86; H(4,8)=x87;
    H(5,0)=x27; H(5,1)=x51; H(5,2)=x67;
    H(5,3)=x75; H(5,4)=x83; H(5,5)=kappa*(x4+12*x57+x88+x9);
    H(5,6)=x89; H(5,7)=x90; H(5,8)=x92;
    H(6,0)=x30; H(6,1)=x53; H(6,2)=x68;
    H(6,3)=x78; H(6,4)=x84; H(6,5)=x89;
    H(6,6)=kappa*(x35+x4+12*x6+x62-4);
    H(6,7)=x93; H(6,8)=x94;
    H(7,0)=x32; H(7,1)=x55; H(7,2)=x69;
    H(7,3)=x79; H(7,4)=x86; H(7,5)=x90;
    H(7,6)=x93; H(7,7)=kappa*(x2+12*x34+x7+x88-4);
    H(7,8)=x95;
    H(8,0)=x33; H(8,1)=x56; H(8,2)=x70;
    H(8,3)=x80; H(8,4)=x87; H(8,5)=x92;
    H(8,6)=x94; H(8,7)=x95;
    H(8,8)=kappa*(x10+x35+x58+12*x60);
    return H;
}

// ============================================================================
// Full OrthoPotential: E, dE/dq (12D), d²E/dq² (12x12)
// ============================================================================

// Vdt2 = volume * dt² (matches libuipc's scaling convention)
CHYSX_HDI float ortho_energy(const Vec12f& q, float kappa, float Vdt2) {
    return ortho_psi(q, kappa) * Vdt2;
}

CHYSX_HDI Vec12f ortho_gradient(const Vec12f& q, float kappa, float Vdt2) {
    Vec9f g9 = ortho_grad(q, kappa);
    Vec12f g = Vec12f::zero();
    for (int i = 0; i < 9; ++i) g[3 + i] = g9[i] * Vdt2;
    return g;
}

CHYSX_HDI Mat12f ortho_hessian_12(const Vec12f& q, float kappa, float Vdt2,
                                   bool project_spd = true) {
    Mat9f H9 = ortho_hessian(q, kappa);
    if (project_spd) math::make_spd<float, 9>(H9);
    H9 *= Vdt2;

    Mat12f H = Mat12f::zero();
    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 9; ++j)
            H(3 + i, 3 + j) = H9(i, j);
    return H;
}

}  // namespace abd_ipc
}  // namespace chysx
