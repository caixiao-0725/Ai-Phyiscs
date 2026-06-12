// SPDX-License-Identifier: MIT
// Contact detection for ABD IPC.
//
// Two detection backends:
//   1. Brute-force (default for small meshes / validation)
//   2. EFContactDetector (QuantBvh EF broadphase → VF+EE narrow phase)

#pragma once

#include "abd_ipc_types.cuh"
#include "abd_ipc_distance.cuh"
#include "../../collision/ef_contact_detector.h"

#include <algorithm>
#include <vector>

namespace chysx {
namespace abd_ipc {

// ============================================================================
// Surface representation for contact detection
// ============================================================================

struct SurfaceData {
    std::vector<Vec3f> verts;

    struct BodyRange {
        int vert_begin, vert_end;
        int tri_begin, tri_end;
        int edge_begin, edge_end;
    };
    std::vector<BodyRange> body_ranges;

    std::vector<Vec3i> tris;
    std::vector<math::Vec2i> edges;
    std::vector<ABDJacobi> jacobi;
    std::vector<int> vert_body;
};

// ============================================================================
// Brute-force detection (kept for validation / small scenes)
// ============================================================================

inline void detect_pt_contacts_brute(const SurfaceData& surf,
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
            if (body_v == body_t0 && body_v == body_t1 && body_v == body_t2)
                continue;
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

inline void detect_ee_contacts_brute(const SurfaceData& surf,
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
            if (body_a0 == body_b0 && body_a0 == body_b1 &&
                body_a1 == body_b0 && body_a1 == body_b1)
                continue;
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

// Full detection: PT + EE (brute-force path)
inline std::vector<ContactPair> detect_contacts(const SurfaceData& surf,
                                                  float d_hat) {
    std::vector<ContactPair> contacts;
    detect_pt_contacts_brute(surf, d_hat, contacts);
    detect_ee_contacts_brute(surf, d_hat, contacts);
    return contacts;
}

// ============================================================================
// Convert EFContactDetector results to ABD-IPC ContactPairs
// (adds body-index annotation and cross-body filtering)
// ============================================================================

inline std::vector<ContactPair> convert_contacts(
    const std::vector<collision::ContactResult>& raw,
    const SurfaceData& surf) {

    std::vector<ContactPair> out;
    out.reserve(raw.size());

    for (auto& cr : raw) {
        ContactPair cp;
        cp.v[0] = cr.v[0]; cp.v[1] = cr.v[1];
        cp.v[2] = cr.v[2]; cp.v[3] = cr.v[3];
        cp.body[0] = surf.vert_body[cr.v[0]];
        cp.body[1] = surf.vert_body[cr.v[1]];
        cp.body[2] = surf.vert_body[cr.v[2]];
        cp.body[3] = surf.vert_body[cr.v[3]];

        if (cr.type == collision::ContactResult::VF) {
            cp.type = ContactType::PT;
            // Cross-body filter: vertex must belong to a different body
            if (cp.body[0] == cp.body[1] &&
                cp.body[0] == cp.body[2] &&
                cp.body[0] == cp.body[3])
                continue;
        } else {
            cp.type = ContactType::EE;
            // Cross-body filter: edges must span different bodies
            if (cp.body[0] == cp.body[2] &&
                cp.body[0] == cp.body[3] &&
                cp.body[1] == cp.body[2] &&
                cp.body[1] == cp.body[3])
                continue;
        }

        // Recompute squared distance for IPC barrier
        if (cp.type == ContactType::PT)
            cp.D = dist2_pt(surf.verts[cp.v[0]], surf.verts[cp.v[1]],
                            surf.verts[cp.v[2]], surf.verts[cp.v[3]]);
        else
            cp.D = dist2_ee(surf.verts[cp.v[0]], surf.verts[cp.v[1]],
                            surf.verts[cp.v[2]], surf.verts[cp.v[3]]);

        out.push_back(cp);
    }
    return out;
}

}  // namespace abd_ipc
}  // namespace chysx
