// SPDX-License-Identifier: MIT
// ABD-specific mesh utilities that build on the public geometry library.
//   - Dyadic mass integration over tetrahedral and triangle meshes
//   - Per-vertex ABDJacobi construction
//   - Body initialization from TetMesh or TriangleMesh (closed surface)

#pragma once

#include "abd_ipc_types.cuh"
#include "../../geometry/tet_mesh.h"
#include "../../geometry/triangle_mesh.h"

#include <cstddef>
#include <vector>

namespace chysx {
namespace abd_ipc {

// ============================================================================
// Dyadic mass from tetrahedral mesh
// ============================================================================

// Exact integration of {1, x, x x^T} over a tetrahedral mesh.
// Uses the analytical formulas:
//   ∫ dV        = V
//   ∫ x dV      = V/4 * (p0 + p1 + p2 + p3)
//   ∫ x x^T dV  = V/20 * (Σ_i pi pi^T + s s^T)
//   where s = p0 + p1 + p2 + p3
inline DyadicMass compute_dyadic_mass(const geometry::TetMeshf& mesh, float density) {
    DyadicMass dm;
    const Vec3f* verts = mesh.vertices().cpu_data();
    const math::Vec4i* tets = mesh.tets().cpu_data();
    std::size_t n_tet = mesh.num_tets();

    for (std::size_t i = 0; i < n_tet; ++i) {
        const auto& t = tets[i];
        Vec3f p0 = verts[t.x], p1 = verts[t.y], p2 = verts[t.z], p3 = verts[t.w];

        float V = geometry::TetMeshf::tet_volume(p0, p1, p2, p3);
        if (V < 0) V = -V;
        float m_tet = density * V;

        // ∫ x dV = V/4 * (p0+p1+p2+p3)
        Vec3f s = p0 + p1 + p2 + p3;
        dm.m += m_tet;
        dm.m_x_bar += s * (m_tet * 0.25f);

        // ∫ x x^T dV = V/20 * (Σ pi pi^T + s s^T)
        Mat3f xx_sum = math::outer(p0, p0) + math::outer(p1, p1)
                     + math::outer(p2, p2) + math::outer(p3, p3);
        Mat3f ss = math::outer(s, s);
        dm.m_xx += (xx_sum + ss) * (density * V / 20.0f);
    }
    return dm;
}

// ============================================================================
// Per-vertex Jacobi array
// ============================================================================

// Build ABDJacobi for each vertex of a TetMesh (rest positions).
inline std::vector<ABDJacobi> build_jacobi_array(const geometry::TetMeshf& mesh) {
    std::size_t nv = mesh.num_vertices();
    const Vec3f* verts = mesh.vertices().cpu_data();
    std::vector<ABDJacobi> jacobi(nv);
    for (std::size_t i = 0; i < nv; ++i)
        jacobi[i] = ABDJacobi(verts[i]);
    return jacobi;
}

// ============================================================================
// Generalized body force (matches libuipc compute_body_force)
// ============================================================================

// Compute 12-DOF generalized body force from a volume force density.
// force_density = gravity * rho [N/m³]
inline Vec12f compute_body_force(const geometry::TetMeshf& mesh,
                                  Vec3f force_density) {
    Vec12f F = Vec12f::zero();
    const Vec3f* verts = mesh.vertices().cpu_data();
    const math::Vec4i* tets = mesh.tets().cpu_data();
    std::size_t n_tet = mesh.num_tets();

    for (std::size_t i = 0; i < n_tet; ++i) {
        const auto& t = tets[i];
        Vec3f p0 = verts[t.x], p1 = verts[t.y], p2 = verts[t.z], p3 = verts[t.w];
        Vec3f e1 = p1 - p0, e2 = p2 - p0, e3 = p3 - p0;
        float D = math::dot(e1, math::cross(e2, e3));
        float V = D / 6.0f;

        // ∫ x_bar dV per component
        auto Q_p = [&](int a) {
            return D * (p0[a] + p1[a] + p2[a] + p3[a]) / 24.0f;
        };
        Vec3f Qs(Q_p(0), Q_p(1), Q_p(2));

        // F[0:3] += f * V
        math::set_block3(F, 0, math::get_block3(F, 0) + force_density * V);
        // F[3:6] += f.x * Qs, F[6:9] += f.y * Qs, F[9:12] += f.z * Qs
        math::set_block3(F, 3, math::get_block3(F, 3) + Qs * force_density.x);
        math::set_block3(F, 6, math::get_block3(F, 6) + Qs * force_density.y);
        math::set_block3(F, 9, math::get_block3(F, 9) + Qs * force_density.z);
    }
    return F;
}

// ============================================================================
// Body initialization
// ============================================================================

// Initialize an ABDBody from a TetMesh at rest (A = I, t = translation).
inline ABDBody init_body(const geometry::TetMeshf& mesh,
                         float density,
                         float kappa,
                         Vec3f gravity,
                         Vec3f translation = Vec3f(0, 0, 0),
                         bool fixed = false) {
    ABDBody body{};

    // Mass
    DyadicMass dm = compute_dyadic_mass(mesh, density);
    body.M = dm.to_mat();
    body.M_inv = math::inverse_spd<float, 12>(body.M);
    body.volume = mesh.total_volume();
    body.kappa = kappa;
    body.friction = 0.5f;
    body.is_fixed = fixed;
    body.is_dynamic = !fixed;

    // Generalized gravity: g = M_inv * F_body
    Vec3f force_density = gravity * density;
    Vec12f F_body = compute_body_force(mesh, force_density);
    body.abd_gravity = body.M_inv * F_body;

    // Initial q: identity rotation + offset
    body.q = make_q(translation, Mat3f::identity());
    body.q_prev = body.q;
    body.q_tilde = body.q;
    body.q_v = Vec12f::zero();

    return body;
}

// ============================================================================
// TriangleMesh support — closed surface mesh via divergence theorem
// (matches libuipc compute_trimesh_dyadic_mass / compute_trimesh_body_force)
// ============================================================================

// Dyadic mass from a closed triangle mesh using the divergence theorem.
// The mesh MUST be a closed, consistently-oriented manifold (outward normals).
inline DyadicMass compute_dyadic_mass(const geometry::TriangleMeshf& mesh,
                                       float density) {
    DyadicMass dm;
    const Vec3f* verts = mesh.vertices().cpu_data();
    const Vec3i* tris = mesh.triangles().cpu_data();
    std::size_t n_tri = mesh.num_triangles();

    float m_total = 0;
    Vec3f m_xbar(0, 0, 0);
    Mat3f m_xx = Mat3f::zero();

    for (std::size_t ti = 0; ti < n_tri; ++ti) {
        Vec3f p0 = verts[tris[ti].x];
        Vec3f p1 = verts[tris[ti].y];
        Vec3f p2 = verts[tris[ti].z];

        Vec3f N = math::cross(p1 - p0, p2 - p0);

        // mass: ρ * p0·N / 6
        m_total += density * math::dot(p0, N) / 6.0f;

        // first moment m_x_bar(a) += ρ/2 * N(a) * Q(a)
        for (int a = 0; a < 3; ++a) {
            float Q = (p0[a]*p0[a] + p0[a]*p1[a] + p0[a]*p2[a]
                      + p1[a]*p1[a] + p1[a]*p2[a] + p2[a]*p2[a]) / 12.0f;
            m_xbar[a] += density * 0.5f * N[a] * Q;
        }

        // second moment diagonal: m_xx(a,a) += ρ/3 * N(a) * Q_diag(a)
        for (int a = 0; a < 3; ++a) {
            float pa0 = p0[a], pa1 = p1[a], pa2 = p2[a];
            float pa0_2 = pa0*pa0, pa1_2 = pa1*pa1, pa2_2 = pa2*pa2;
            float V = pa0_2*pa0/20 + pa0_2*pa1/20 + pa0_2*pa2/20
                    + pa0*pa1_2/20 + pa0*pa1*pa2/20 + pa0*pa2_2/20
                    + pa1_2*pa1/20 + pa1_2*pa2/20 + pa1*pa2_2/20
                    + pa2_2*pa2/20;
            m_xx(a, a) += density / 3.0f * N[a] * V;
        }

        // second moment off-diagonal (only upper triangle, symmetrize later)
        auto Q_off = [&](int a, int b) {
            float pa0 = p0[a], pa1 = p1[a], pa2 = p2[a];
            float pb0 = p0[b], pb1 = p1[b], pb2 = p2[b];
            float pa0_2 = pa0*pa0, pa1_2 = pa1*pa1, pa2_2 = pa2*pa2;
            float V = 0;
            V += pa0_2*pb0/20 + pa0_2*pb1/60 + pa0_2*pb2/60;
            V += pa0*pb0*pa1/30 + pa0*pb0*pa2/30;
            V += pa0*pa1*pb1/30 + pa0*pa1*pb2/60 + pa0*pb1*pa2/60;
            V += pa0*pa2*pb2/30;
            V += pb0*pa1_2/60 + pb0*pa1*pa2/60 + pb0*pa2_2/60;
            V += pa1_2*pb1/20 + pa1_2*pb2/60;
            V += pa1*pb1*pa2/30 + pa1*pa2*pb2/30;
            V += pb1*pa2_2/60 + pa2_2*pb2/20;
            return density * 0.5f * N[a] * V;
        };
        m_xx(0, 1) += Q_off(0, 1);
        m_xx(0, 2) += Q_off(0, 2);
        m_xx(1, 2) += Q_off(1, 2);
    }

    m_xx(1, 0) = m_xx(0, 1);
    m_xx(2, 0) = m_xx(0, 2);
    m_xx(2, 1) = m_xx(1, 2);

    dm.m = m_total;
    dm.m_x_bar = m_xbar;
    dm.m_xx = m_xx;
    return dm;
}

// Enclosed volume of a closed triangle mesh (divergence theorem).
inline float trimesh_volume(const geometry::TriangleMeshf& mesh) {
    const Vec3f* verts = mesh.vertices().cpu_data();
    const Vec3i* tris = mesh.triangles().cpu_data();
    float vol = 0;
    for (std::size_t i = 0; i < mesh.num_triangles(); ++i) {
        Vec3f p0 = verts[tris[i].x], p1 = verts[tris[i].y], p2 = verts[tris[i].z];
        vol += math::dot(p0, math::cross(p1 - p0, p2 - p0)) / 6.0f;
    }
    return vol > 0 ? vol : -vol;
}

// Build ABDJacobi for each vertex of a TriangleMesh (rest positions).
inline std::vector<ABDJacobi> build_jacobi_array(const geometry::TriangleMeshf& mesh) {
    std::size_t nv = mesh.num_vertices();
    const Vec3f* verts = mesh.vertices().cpu_data();
    std::vector<ABDJacobi> jacobi(nv);
    for (std::size_t i = 0; i < nv; ++i)
        jacobi[i] = ABDJacobi(verts[i]);
    return jacobi;
}

// Generalized body force from a closed triangle mesh (divergence theorem).
inline Vec12f compute_body_force(const geometry::TriangleMeshf& mesh,
                                  Vec3f force_density) {
    Vec12f F = Vec12f::zero();
    const Vec3f* verts = mesh.vertices().cpu_data();
    const Vec3i* tris = mesh.triangles().cpu_data();
    Vec3f f = force_density;

    for (std::size_t ti = 0; ti < mesh.num_triangles(); ++ti) {
        Vec3f p0 = verts[tris[ti].x], p1 = verts[tris[ti].y], p2 = verts[tris[ti].z];
        Vec3f N = math::cross(p1 - p0, p2 - p0);
        float V = math::dot(p0, N) / 6.0f;

        auto Q_p = [&](int a) -> float {
            return 0.5f * N[a] * (p0[a]*p0[a] + p0[a]*p1[a] + p0[a]*p2[a]
                                + p1[a]*p1[a] + p1[a]*p2[a] + p2[a]*p2[a]) / 12.0f;
        };
        Vec3f Qs(Q_p(0), Q_p(1), Q_p(2));

        math::set_block3(F, 0, math::get_block3(F, 0) + f * V);
        math::set_block3(F, 3, math::get_block3(F, 3) + Qs * f.x);
        math::set_block3(F, 6, math::get_block3(F, 6) + Qs * f.y);
        math::set_block3(F, 9, math::get_block3(F, 9) + Qs * f.z);
    }
    return F;
}

// Initialize an ABDBody from a closed TriangleMesh at rest.
inline ABDBody init_body(geometry::TriangleMeshf& mesh,
                         float density,
                         float kappa,
                         Vec3f gravity,
                         Vec3f translation = Vec3f(0, 0, 0),
                         bool fixed = false) {
    ABDBody body{};

    DyadicMass dm = compute_dyadic_mass(mesh, density);
    body.M = dm.to_mat();
    body.M_inv = math::inverse_spd<float, 12>(body.M);
    body.volume = trimesh_volume(mesh);
    body.kappa = kappa;
    body.friction = 0.5f;
    body.is_fixed = fixed;
    body.is_dynamic = !fixed;

    Vec3f force_density = gravity * density;
    Vec12f F_body = compute_body_force(mesh, force_density);
    body.abd_gravity = body.M_inv * F_body;

    body.q = make_q(translation, Mat3f::identity());
    body.q_prev = body.q;
    body.q_tilde = body.q;
    body.q_v = Vec12f::zero();

    return body;
}

}  // namespace abd_ipc
}  // namespace chysx
