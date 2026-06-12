// SPDX-License-Identifier: MIT
// Mass properties from a closed triangle mesh (divergence theorem).
//
// Computes: volume, center of mass, inertia tensor, principal-axis
// alignment, and the diagonal mass matrix for 6-DOF rigid bodies.
//
// Reference: Chrono's ChTriangleMeshConnected::ComputeMassProperties
// (polyhedral divergence-theorem method, same as rigid-ipc/src/physics/mass.cpp)

#pragma once

#include "rigid_ipc_types.cuh"
#include "../../geometry/triangle_mesh.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace chysx {
namespace rigid_ipc {

struct MassProperties {
    float mass;
    Vec3f center;
    Mat3f inertia;             // 3x3 inertia tensor at CoM
    Vec3f principal_inertia;   // eigenvalues (Ix, Iy, Iz)
    Mat3f principal_axes;      // eigenvectors as columns (rotation to principal frame)
};

// Compute mass properties from a closed triangle mesh using the
// divergence theorem (10 volume integrals).
// `density` is in kg/m^3. Mesh must be watertight and consistently oriented.
inline bool compute_mass_properties(const Vec3f* verts, const Vec3i* faces,
                                     int n_verts, int n_faces,
                                     float density, MassProperties& out) {
    double integral[10] = {};

    for (int i = 0; i < n_faces; ++i) {
        Vec3f v0f = verts[faces[i].x];
        Vec3f v1f = verts[faces[i].y];
        Vec3f v2f = verts[faces[i].z];

        double x0 = v0f.x, y0 = v0f.y, z0 = v0f.z;
        double x1 = v1f.x, y1 = v1f.y, z1 = v1f.z;
        double x2 = v2f.x, y2 = v2f.y, z2 = v2f.z;

        double a1 = x1 - x0, b1 = y1 - y0, c1 = z1 - z0;
        double a2 = x2 - x0, b2 = y2 - y0, c2 = z2 - z0;
        double nx = b1 * c2 - c1 * b2;
        double ny = c1 * a2 - a1 * c2;
        double nz = a1 * b2 - b1 * a2;

        // Sub-expressions for integrals
        double f1x = x0 + x1 + x2;
        double f2x = x0 * x0 + x1 * x1 + x0 * x1 + x2 * f1x;
        double f3x = x0 * x0 * x0 + x1 * x1 * x1 + x0 * x0 * x1 + x0 * x1 * x1 + x2 * f2x;

        double f1y = y0 + y1 + y2;
        double f2y = y0 * y0 + y1 * y1 + y0 * y1 + y2 * f1y;
        double f3y = y0 * y0 * y0 + y1 * y1 * y1 + y0 * y0 * y1 + y0 * y1 * y1 + y2 * f2y;

        double f1z = z0 + z1 + z2;
        double f2z = z0 * z0 + z1 * z1 + z0 * z1 + z2 * f1z;
        double f3z = z0 * z0 * z0 + z1 * z1 * z1 + z0 * z0 * z1 + z0 * z1 * z1 + z2 * f2z;

        double g0x = f2x + x0 * (f1x + x0);
        double g0y = f2y + y0 * (f1y + y0);
        double g0z = f2z + z0 * (f1z + z0);
        double g1x = f2x + x1 * (f1x + x1);
        double g1y = f2y + y1 * (f1y + y1);
        double g1z = f2z + z1 * (f1z + z1);
        double g2x = f2x + x2 * (f1x + x2);
        double g2y = f2y + y2 * (f1y + y2);
        double g2z = f2z + z2 * (f1z + z2);

        integral[0] += nx * f1x;
        integral[1] += nx * f2x;
        integral[2] += ny * f2y;
        integral[3] += nz * f2z;
        integral[4] += nx * f3x;
        integral[5] += ny * f3y;
        integral[6] += nz * f3z;
        integral[7] += nx * (y0 * g0x + y1 * g1x + y2 * g2x);
        integral[8] += ny * (z0 * g0y + z1 * g1y + z2 * g2y);
        integral[9] += nz * (x0 * g0z + x1 * g1z + x2 * g2z);
    }

    integral[0] /= 6.0;
    integral[1] /= 24.0;
    integral[2] /= 24.0;
    integral[3] /= 24.0;
    integral[4] /= 60.0;
    integral[5] /= 60.0;
    integral[6] /= 60.0;
    integral[7] /= 120.0;
    integral[8] /= 120.0;
    integral[9] /= 120.0;

    double volume = integral[0];
    if (volume <= 0 || !std::isfinite(volume)) {
        printf("[rigid_ipc_mass] ERROR: non-positive volume %.6e (mesh may not be closed)\n", volume);
        return false;
    }

    double mass = density * volume;
    double cx = integral[1] / volume;
    double cy = integral[2] / volume;
    double cz = integral[3] / volume;

    // Inertia about origin
    double Ixx = density * (integral[5] + integral[6]);
    double Iyy = density * (integral[4] + integral[6]);
    double Izz = density * (integral[4] + integral[5]);
    double Ixy = -density * integral[7];
    double Iyz = -density * integral[8];
    double Ixz = -density * integral[9];

    // Parallel-axis theorem: shift to CoM
    Ixx -= mass * (cy * cy + cz * cz);
    Iyy -= mass * (cx * cx + cz * cz);
    Izz -= mass * (cx * cx + cy * cy);
    Ixy += mass * cx * cy;
    Iyz += mass * cy * cz;
    Ixz += mass * cx * cz;

    out.mass = static_cast<float>(mass);
    out.center = Vec3f(static_cast<float>(cx),
                       static_cast<float>(cy),
                       static_cast<float>(cz));
    out.inertia = Mat3f(
        static_cast<float>(Ixx), static_cast<float>(Ixy), static_cast<float>(Ixz),
        static_cast<float>(Ixy), static_cast<float>(Iyy), static_cast<float>(Iyz),
        static_cast<float>(Ixz), static_cast<float>(Iyz), static_cast<float>(Izz));

    return true;
}

// 3x3 Jacobi eigenvalue solver for symmetric matrices.
// Returns eigenvalues in `evals` and orthonormal eigenvectors as columns of `evecs`.
inline void symmetric_eigen3(const Mat3f& A, Vec3f& evals, Mat3f& evecs) {
    // Start with identity
    Mat3f V = Mat3f::identity();
    Mat3f M = A;

    for (int sweep = 0; sweep < 50; ++sweep) {
        // Find largest off-diagonal element
        float max_off = 0;
        int p = 0, qq = 1;
        for (int i = 0; i < 3; ++i)
            for (int j = i + 1; j < 3; ++j) {
                float a = fabsf(M(i, j));
                if (a > max_off) { max_off = a; p = i; qq = j; }
            }
        if (max_off < 1e-10f) break;

        float app = M(p, p), aqq = M(qq, qq), apq = M(p, qq);
        float tau = (aqq - app) / (2.0f * apq);
        float t = (tau >= 0 ? 1.0f : -1.0f) / (fabsf(tau) + sqrtf(1.0f + tau * tau));
        float c = 1.0f / sqrtf(1.0f + t * t);
        float s = t * c;

        // Givens rotation: update M
        Mat3f G = Mat3f::identity();
        G(p, p) = c;  G(p, qq) = s;
        G(qq, p) = -s; G(qq, qq) = c;

        M = transpose(G) * M * G;
        V = V * G;  // accumulate eigenvectors
    }

    float ev[3] = { M(0, 0), M(1, 1), M(2, 2) };
    int idx[3] = { 0, 1, 2 };

    // Sort eigenvalues ascending (matching Eigen's SelfAdjointEigenSolver)
    if (ev[idx[0]] > ev[idx[1]]) std::swap(idx[0], idx[1]);
    if (ev[idx[1]] > ev[idx[2]]) std::swap(idx[1], idx[2]);
    if (ev[idx[0]] > ev[idx[1]]) std::swap(idx[0], idx[1]);

    evals = Vec3f(ev[idx[0]], ev[idx[1]], ev[idx[2]]);
    for (int r = 0; r < 3; ++r) {
        evecs(r, 0) = V(r, idx[0]);
        evecs(r, 1) = V(r, idx[1]);
        evecs(r, 2) = V(r, idx[2]);
    }
}

// Compute mass properties AND align mesh vertices to the principal-axis frame.
// This modifies vertices in place: translates to CoM and rotates to principal axes.
// Returns the world-space CoM position that was subtracted.
inline bool compute_and_align_mass(geometry::TriangleMeshf& mesh, float density,
                                    MassProperties& props) {
    int nv = static_cast<int>(mesh.num_vertices());
    int nf = static_cast<int>(mesh.num_triangles());
    Vec3f* verts = mesh.vertices().cpu_data();
    const Vec3i* faces = mesh.triangles().cpu_data();

    if (!compute_mass_properties(verts, faces, nv, nf, density, props))
        return false;

    // Translate mesh to CoM
    for (int i = 0; i < nv; ++i) {
        verts[i].x -= props.center.x;
        verts[i].y -= props.center.y;
        verts[i].z -= props.center.z;
    }

    // Eigendecompose inertia to get principal axes
    Vec3f evals;
    Mat3f evecs;
    symmetric_eigen3(props.inertia, evals, evecs);

    // Ensure right-handed: flip third column if determinant < 0
    float det = math::determinant(evecs);
    if (det < 0) {
        evecs(0, 2) = -evecs(0, 2);
        evecs(1, 2) = -evecs(1, 2);
        evecs(2, 2) = -evecs(2, 2);
    }

    // Rotate vertices into principal frame: v' = R0^T * v
    Mat3f R0T = transpose(evecs);
    for (int i = 0; i < nv; ++i) {
        verts[i] = R0T * verts[i];
    }

    props.principal_inertia = evals;
    props.principal_axes = evecs;

    // Recompute inertia in principal frame (should now be diagonal)
    props.inertia = Mat3f(evals.x, 0, 0,  0, evals.y, 0,  0, 0, evals.z);

    return true;
}

// compute_J: maps principal inertia (Ix, Iy, Iz) to the diagonal of the
// body-frame tensor J used in the rotational energy:
//   E_rot = 1/2 tr(Q J Q^T)
// J = diag(J1, J2, J3) where J1 = 1/2(-Ix + Iy + Iz), etc.
CHYSX_HDI Vec3f compute_J(Vec3f I) {
    return Vec3f(
        0.5f * (-I.x + I.y + I.z),
        0.5f * (I.x - I.y + I.z),
        0.5f * (I.x + I.y - I.z));
}

}  // namespace rigid_ipc
}  // namespace chysx
