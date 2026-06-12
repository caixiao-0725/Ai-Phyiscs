// SPDX-License-Identifier: Apache-2.0
// CUDA implementation of chysx::collision::EFContactDetector.

#include "ef_contact_detector.h"

#include <cstring>
#include <cmath>
#include <algorithm>

namespace chysx {
namespace collision {

using math::Vec3f;
using math::Vec2i;
using math::Vec3i;

// ============================================================================
// Geometry helpers (CPU, matching PeriDyno's FuncNearestPointVF / EE)
// ============================================================================

namespace {

// Closest point on triangle (a,b,c) to query point p.
// Returns the closest point `out` and barycentric weights `w` (w.x for a, etc).
// Ericson, Real-Time Collision Detection, Sec. 5.1.5.
void closest_point_triangle(Vec3f p, Vec3f a, Vec3f b, Vec3f c,
                             Vec3f& out, Vec3f& w) {
    Vec3f ab = b - a, ac = c - a, ap = p - a;
    float d1 = math::dot(ab, ap), d2 = math::dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) { out = a; w = Vec3f(1,0,0); return; }

    Vec3f bp = p - b;
    float d3 = math::dot(ab, bp), d4 = math::dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) { out = b; w = Vec3f(0,1,0); return; }

    float vc = d1*d4 - d3*d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        float v = d1 / (d1 - d3);
        out = a + ab * v; w = Vec3f(1-v, v, 0); return;
    }

    Vec3f cp = p - c;
    float d5 = math::dot(ab, cp), d6 = math::dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) { out = c; w = Vec3f(0,0,1); return; }

    float vb = d5*d2 - d1*d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        float ww = d2 / (d2 - d6);
        out = a + ac * ww; w = Vec3f(1-ww, 0, ww); return;
    }

    float va = d3*d6 - d5*d4;
    if (va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0) {
        float ww = (d4-d3) / ((d4-d3)+(d5-d6));
        out = b + (c - b) * ww; w = Vec3f(0, 1-ww, ww); return;
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom, ww = vc * denom;
    out = a + ab * v + ac * ww;
    w = Vec3f(1-v-ww, v, ww);
}

// Closest points between segments (p0,p1) and (q0,q1).
// Returns parameters s,t ∈ [0,1] and closest points cp, cq.
void closest_point_segments(Vec3f p0, Vec3f p1, Vec3f q0, Vec3f q1,
                             float& s, float& t, Vec3f& cp, Vec3f& cq) {
    Vec3f d1 = p1 - p0, d2 = q1 - q0, r = p0 - q0;
    float a = math::dot(d1, d1), e = math::dot(d2, d2);
    float f = math::dot(d2, r);

    const float eps = 1e-12f;
    if (a <= eps && e <= eps) {
        s = t = 0; cp = p0; cq = q0; return;
    }
    if (a <= eps) {
        s = 0; t = std::clamp(f / e, 0.0f, 1.0f);
    } else {
        float c = math::dot(d1, r);
        if (e <= eps) {
            t = 0; s = std::clamp(-c / a, 0.0f, 1.0f);
        } else {
            float b = math::dot(d1, d2);
            float denom = a*e - b*b;
            s = (denom != 0) ? std::clamp((b*f - c*e) / denom, 0.0f, 1.0f) : 0;
            t = (b*s + f) / e;
            if (t < 0) { t = 0; s = std::clamp(-c / a, 0.0f, 1.0f); }
            else if (t > 1) { t = 1; s = std::clamp((b - c) / a, 0.0f, 1.0f); }
        }
    }
    cp = p0 + d1 * s;
    cq = q0 + d2 * t;
}

}  // namespace

// ============================================================================
// Setup
// ============================================================================

void EFContactDetector::setup(const std::vector<math::Vec3i>& tris,
                               int n_verts,
                               int max_ef_candidates,
                               BroadphaseBackend backend) {
    backend_ = backend;

    if (backend_ == BroadphaseBackend::QuantBvh) {
        qbvh_broadphase_.setup(tris, n_verts, max_ef_candidates);
    }
#ifdef CHYSX_HAS_OPTIX
    else if (backend_ == BroadphaseBackend::OptiX) {
        optix_broadphase_ = std::make_unique<OptixEFBroadphase>();
        optix_broadphase_->setup(tris, n_verts);
    }
#endif
    verts_gpu_.resize(n_verts);
}

bool EFContactDetector::valid() const noexcept {
    if (backend_ == BroadphaseBackend::QuantBvh)
        return qbvh_broadphase_.valid();
#ifdef CHYSX_HAS_OPTIX
    if (backend_ == BroadphaseBackend::OptiX)
        return optix_broadphase_ && optix_broadphase_->valid();
#endif
    return false;
}

const MeshTopology& EFContactDetector::topology() const noexcept {
#ifdef CHYSX_HAS_OPTIX
    if (backend_ == BroadphaseBackend::OptiX && optix_broadphase_)
        return optix_broadphase_->topology();
#endif
    return qbvh_broadphase_.topology();
}

// ============================================================================
// Detection (CPU positions → GPU broadphase → CPU narrow phase)
// ============================================================================

void EFContactDetector::detect(const math::Vec3f* positions_cpu,
                                int n_verts,
                                float thickness,
                                std::vector<ContactResult>& contacts) {
    if (!valid()) return;

    std::memcpy(verts_gpu_.cpu_data(), positions_cpu, n_verts * sizeof(Vec3f));
    verts_gpu_.copy_to_device();

    detect_gpu(verts_gpu_.gpu_data(), positions_cpu, n_verts,
               thickness, contacts);
}

void EFContactDetector::detect_gpu(const math::Vec3f* positions_dev,
                                    const math::Vec3f* positions_cpu,
                                    int n_verts,
                                    float thickness,
                                    std::vector<ContactResult>& contacts) {
    if (!valid()) return;

    std::vector<Vec2i> ef_pairs;

    if (backend_ == BroadphaseBackend::QuantBvh) {
        qbvh_broadphase_.query(positions_dev, thickness);
        qbvh_broadphase_.download_pairs(ef_pairs);
    }
#ifdef CHYSX_HAS_OPTIX
    else if (backend_ == BroadphaseBackend::OptiX && optix_broadphase_) {
        optix_broadphase_->query(positions_dev, thickness);
        optix_broadphase_->download_pairs(ef_pairs);
    }
#endif

    narrow_phase(ef_pairs, positions_cpu, thickness, contacts);

    // Supplement: adjacent VF/EE pairs missed by the vert_in_edge trick.
    cull_adjacent_vf(positions_cpu, thickness, contacts);
    cull_adjacent_ee(positions_cpu, thickness, contacts);
}

// ============================================================================
// Narrow phase: EF → VF + EE decomposition with distance filter
// ============================================================================

void EFContactDetector::narrow_phase(
    const std::vector<Vec2i>& ef_pairs,
    const Vec3f* pos,
    float thickness,
    std::vector<ContactResult>& contacts) {

    const auto& topo = topology();
    const Vec2i*  edges_h = topo.edges().cpu_data();
    const Vec3i*  faces_h = topo.faces().cpu_data();
    const int*    vie_h   = topo.vert_in_edge().cpu_data();
    const Vec3i*  eif_h   = topo.edge_in_face().cpu_data();

    for (auto& ef : ef_pairs) {
        int eid = ef.x;
        int fid = ef.y;

        // ---- VF: vert_in_edge[eid] vs face[fid] ----
        int vid = vie_h[eid];
        if (vid >= 0) {
            const Vec3i& f = faces_h[fid];
            if (vid != f.x && vid != f.y && vid != f.z) {
                Vec3f p  = pos[vid];
                Vec3f t0 = pos[f.x], t1 = pos[f.y], t2 = pos[f.z];
                Vec3f closest, w;
                closest_point_triangle(p, t0, t1, t2, closest, w);
                Vec3f diff = p - closest;
                float d = std::sqrt(math::dot(diff, diff));
                if (d > 1e-12f && d < thickness) {
                    ContactResult cr;
                    cr.type = ContactResult::VF;
                    cr.v[0] = vid; cr.v[1] = f.x; cr.v[2] = f.y; cr.v[3] = f.z;
                    cr.distance = d;
                    // Face normal
                    Vec3f e1 = t1 - t0, e2 = t2 - t0;
                    cr.normal = math::normalize(math::cross(e1, e2));
                    contacts.push_back(cr);
                }
            }
        }

        // ---- EE: edge[eid] vs each of face[fid]'s 3 edges ----
        const Vec3i& e3 = eif_h[fid];
        const Vec2i& ea = edges_h[eid];

        for (int j = 0; j < 3; ++j) {
            int oeid = e3.data[j];
            if (oeid < 0 || oeid == eid) continue;
            if (oeid >= eid) continue;  // dedup: only emit eid < oeid → wait, PeriDyno uses oeid < eid

            const Vec2i& eb = edges_h[oeid];
            // Shared-vertex filter
            if (ea.x == eb.x || ea.x == eb.y ||
                ea.y == eb.x || ea.y == eb.y) continue;

            Vec3f p0 = pos[ea.x], p1 = pos[ea.y];
            Vec3f q0 = pos[eb.x], q1 = pos[eb.y];
            float s, t;
            Vec3f cp, cq;
            closest_point_segments(p0, p1, q0, q1, s, t, cp, cq);
            Vec3f diff = cp - cq;
            float d = std::sqrt(math::dot(diff, diff));

            // Interior check (both parameters strictly inside) + degeneracy filter
            Vec3f d1 = p1 - p0, d2 = q1 - q0;
            float len_sq1 = math::dot(d1, d1), len_sq2 = math::dot(d2, d2);
            Vec3f cross_d = math::cross(d1, d2);
            float cross_sq = math::dot(cross_d, cross_d);
            float eps = 1e-3f * len_sq1 * len_sq2;

            if (s > 0.0f && s < 1.0f && t > 0.0f && t < 1.0f &&
                d > 1e-12f && d < thickness && cross_sq >= eps) {
                ContactResult cr;
                cr.type = ContactResult::EE;
                cr.v[0] = ea.x; cr.v[1] = ea.y; cr.v[2] = eb.x; cr.v[3] = eb.y;
                cr.distance = d;
                cr.normal = math::normalize(cross_d);
                contacts.push_back(cr);
            }
        }
    }
}

// ============================================================================
// CullVFAdjacent: supplementary VF contacts from pre-computed adjacency
// ============================================================================

void EFContactDetector::cull_adjacent_vf(
    const Vec3f* pos,
    float thickness,
    std::vector<ContactResult>& contacts) {

    const auto& topo = topology();
    int n = topo.n_adj_vf();
    if (n == 0) return;
    const auto* pairs = topo.pre_adj_vf().cpu_data();

    for (int i = 0; i < n; ++i) {
        const auto& p = pairs[i];
        int vid = p.x;
        int f0 = p.y, f1 = p.z, f2 = p.w;

        Vec3f v = pos[vid];
        Vec3f t0 = pos[f0], t1 = pos[f1], t2 = pos[f2];
        Vec3f closest, w;
        closest_point_triangle(v, t0, t1, t2, closest, w);
        Vec3f diff = v - closest;
        float d = std::sqrt(math::dot(diff, diff));

        if (d > 1e-12f && d < thickness) {
            ContactResult cr;
            cr.type = ContactResult::VF;
            cr.v[0] = vid; cr.v[1] = f0; cr.v[2] = f1; cr.v[3] = f2;
            cr.distance = d;
            Vec3f e1 = t1 - t0, e2 = t2 - t0;
            cr.normal = math::normalize(math::cross(e1, e2));
            contacts.push_back(cr);
        }
    }
}

// ============================================================================
// CullEEAdjacent: supplementary EE contacts from pre-computed adjacency
// ============================================================================

void EFContactDetector::cull_adjacent_ee(
    const Vec3f* pos,
    float thickness,
    std::vector<ContactResult>& contacts) {

    const auto& topo = topology();
    int n = topo.n_adj_ee_pre();
    if (n == 0) return;

    const auto* ee_pairs = topo.pre_adj_ee().cpu_data();
    const auto* edges_h = topo.edges().cpu_data();
    const auto* eq = topo.edge_quad().cpu_data();

    for (int i = 0; i < n; ++i) {
        const auto& pr = ee_pairs[i];
        int eid_a = pr.x, eid_b = pr.y;

        const Vec2i& ea = edges_h[eid_a];
        const Vec2i& eb = edges_h[eid_b];

        // Shared-vertex filter (shouldn't happen for pre_adj_ee, but be safe)
        if (ea.x == eb.x || ea.x == eb.y ||
            ea.y == eb.x || ea.y == eb.y) continue;

        Vec3f p0 = pos[ea.x], p1 = pos[ea.y];
        Vec3f q0 = pos[eb.x], q1 = pos[eb.y];
        float s, t;
        Vec3f cp, cq;
        closest_point_segments(p0, p1, q0, q1, s, t, cp, cq);
        Vec3f diff = cp - cq;
        float d = std::sqrt(math::dot(diff, diff));

        Vec3f d1 = p1 - p0, d2 = q1 - q0;
        float len_sq1 = math::dot(d1, d1), len_sq2 = math::dot(d2, d2);
        Vec3f cross_d = math::cross(d1, d2);
        float cross_sq = math::dot(cross_d, cross_d);
        float eps = 1e-3f * len_sq1 * len_sq2;

        if (s > 0.0f && s < 1.0f && t > 0.0f && t < 1.0f &&
            d > 1e-12f && d < thickness && cross_sq >= eps) {

            Vec3f normal = math::normalize(cross_d);

            // Orient normal using edge_quad (average face normal, matching PeriDyno)
            const auto& quad = eq[eid_a];
            if (quad.z >= 0 && quad.w >= 0) {
                Vec3f v0 = pos[quad.x], v1 = pos[quad.y];
                Vec3f v2 = pos[quad.z], v3 = pos[quad.w];
                Vec3f e0 = v1 - v0;
                Vec3f e1v = v2 - v0;
                Vec3f n0 = math::normalize(math::cross(e0, e1v));
                Vec3f e2v = v3 - v0;
                Vec3f n1 = math::normalize(math::cross(e0, e2v));
                n1 = Vec3f(-n1.x, -n1.y, -n1.z);
                Vec3f navg = Vec3f(n0.x + n1.x, n0.y + n1.y, n0.z + n1.z);
                if (math::dot(navg, normal) > 0) {
                    normal = Vec3f(-normal.x, -normal.y, -normal.z);
                }
            }

            ContactResult cr;
            cr.type = ContactResult::EE;
            cr.v[0] = ea.x; cr.v[1] = ea.y; cr.v[2] = eb.x; cr.v[3] = eb.y;
            cr.distance = d;
            cr.normal = normal;
            contacts.push_back(cr);
        }
    }
}

}  // namespace collision
}  // namespace chysx
