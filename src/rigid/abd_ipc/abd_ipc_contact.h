// SPDX-License-Identifier: MIT
// Contact detection for ABD IPC.
//
// Initial implementation: brute-force all-pairs for small body counts.
// TODO: integrate ChysX QuantBvh broadphase for scalability.

#pragma once

#include "abd_ipc_types.cuh"
#include "abd_ipc_distance.cuh"

#include <vector>

namespace chysx {
namespace abd_ipc {

// ============================================================================
// Surface representation for contact detection
// ============================================================================

struct SurfaceData {
    // Global vertex list: all surface vertices across all bodies
    std::vector<Vec3f> verts;

    // Per-body ranges into the global vertex list
    struct BodyRange {
        int vert_begin, vert_end;
        int tri_begin, tri_end;
        int edge_begin, edge_end;
    };
    std::vector<BodyRange> body_ranges;

    // Global triangle and edge lists (indices into verts)
    std::vector<Vec3i> tris;
    std::vector<math::Vec2i> edges;

    // Jacobi data: one ABDJacobi per surface vertex
    std::vector<ABDJacobi> jacobi;

    // Map: global vertex index -> body index
    std::vector<int> vert_body;
};

// ============================================================================
// Contact detection (brute-force for correctness, small body counts)
// ============================================================================

// Find all vertex-triangle pairs within d_hat distance (between different bodies).
inline void detect_pt_contacts(const SurfaceData& surf,
                                float d_hat,
                                std::vector<ContactPair>& contacts) {
    float D_hat = d_hat * d_hat;
    int nv = static_cast<int>(surf.verts.size());
    int nt = static_cast<int>(surf.tris.size());

    for (int vi = 0; vi < nv; ++vi) {
        int body_v = surf.vert_body[vi];

        for (int ti = 0; ti < nt; ++ti) {
            const Vec3i& tri = surf.tris[ti];
            int body_t0 = surf.vert_body[tri.x];
            int body_t1 = surf.vert_body[tri.y];
            int body_t2 = surf.vert_body[tri.z];

            // Skip self-body contacts
            if (body_v == body_t0 && body_v == body_t1 && body_v == body_t2)
                continue;
            // Skip if vertex is part of this triangle
            if (vi == tri.x || vi == tri.y || vi == tri.z) continue;

            float D = dist2_pt(surf.verts[vi],
                               surf.verts[tri.x],
                               surf.verts[tri.y],
                               surf.verts[tri.z]);
            if (D < D_hat) {
                ContactPair cp;
                cp.type = ContactType::PT;
                cp.v[0] = vi; cp.v[1] = tri.x; cp.v[2] = tri.y; cp.v[3] = tri.z;
                cp.body[0] = body_v;
                cp.body[1] = body_t0;
                cp.body[2] = body_t1;
                cp.body[3] = body_t2;
                cp.D = D;
                contacts.push_back(cp);
            }
        }
    }
}

// Find all edge-edge pairs within d_hat distance (between different bodies).
inline void detect_ee_contacts(const SurfaceData& surf,
                                float d_hat,
                                std::vector<ContactPair>& contacts) {
    float D_hat = d_hat * d_hat;
    int ne = static_cast<int>(surf.edges.size());

    for (int i = 0; i < ne; ++i) {
        int a0 = surf.edges[i].x, a1 = surf.edges[i].y;
        int body_a0 = surf.vert_body[a0], body_a1 = surf.vert_body[a1];

        for (int j = i + 1; j < ne; ++j) {
            int b0 = surf.edges[j].x, b1 = surf.edges[j].y;
            int body_b0 = surf.vert_body[b0], body_b1 = surf.vert_body[b1];

            // Skip same-body edges
            if (body_a0 == body_b0 && body_a0 == body_b1 &&
                body_a1 == body_b0 && body_a1 == body_b1)
                continue;
            // Skip if edges share a vertex
            if (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1) continue;

            float D = dist2_ee(surf.verts[a0], surf.verts[a1],
                               surf.verts[b0], surf.verts[b1]);
            if (D < D_hat) {
                ContactPair cp;
                cp.type = ContactType::EE;
                cp.v[0] = a0; cp.v[1] = a1; cp.v[2] = b0; cp.v[3] = b1;
                cp.body[0] = body_a0; cp.body[1] = body_a1;
                cp.body[2] = body_b0; cp.body[3] = body_b1;
                cp.D = D;
                contacts.push_back(cp);
            }
        }
    }
}

// Full detection: PT + EE
inline std::vector<ContactPair> detect_contacts(const SurfaceData& surf,
                                                  float d_hat) {
    std::vector<ContactPair> contacts;
    detect_pt_contacts(surf, d_hat, contacts);
    detect_ee_contacts(surf, d_hat, contacts);
    return contacts;
}

}  // namespace abd_ipc
}  // namespace chysx
