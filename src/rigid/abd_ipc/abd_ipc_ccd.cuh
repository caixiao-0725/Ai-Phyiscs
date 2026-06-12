// SPDX-License-Identifier: MIT
// Additive CCD (continuous collision detection) and line search for ABD IPC.
//
// Additive CCD conservatively estimates the maximum step size α ∈ (0, 1]
// such that no surface primitive pair tunnels through each other when
// moving from the current to the trial configuration.

#pragma once

#include "abd_ipc_types.cuh"
#include "abd_ipc_distance.cuh"

#include <cmath>
#include <vector>

namespace chysx {
namespace abd_ipc {

// ============================================================================
// Point–triangle CCD (additive)
// ============================================================================

// Returns the maximum safe step α ∈ (0, 1] for a point moving from p0 → p0+dp
// towards a triangle (t0,t1,t2) moving by (dt0,dt1,dt2).
// Samples along the trajectory and finds the smallest α where D first drops
// below the floor, then returns the latest safe α before that.
CHYSX_HDI float ccd_pt(Vec3f p0, Vec3f dp,
                        Vec3f t0, Vec3f dt0,
                        Vec3f t1, Vec3f dt1,
                        Vec3f t2, Vec3f dt2,
                        float d_min,
                        int max_iter = 20) {
    float d_min2 = d_min * d_min;

    // Sample along trajectory at uniform intervals to find first violation
    int n_samples = max_iter;
    for (int k = 1; k <= n_samples; ++k) {
        float t = static_cast<float>(k) / static_cast<float>(n_samples);
        Vec3f pp  = p0 + dp * t;
        Vec3f tt0 = t0 + dt0 * t;
        Vec3f tt1 = t1 + dt1 * t;
        Vec3f tt2 = t2 + dt2 * t;

        float D = dist2_pt(pp, tt0, tt1, tt2);
        if (D < d_min2) {
            // Found violation at t. Bisect to refine between (k-1)/n and k/n.
            float lo = static_cast<float>(k - 1) / static_cast<float>(n_samples);
            float hi = t;
            for (int b = 0; b < 10; ++b) {
                float mid = (lo + hi) * 0.5f;
                Vec3f mp  = p0 + dp * mid;
                Vec3f mt0 = t0 + dt0 * mid;
                Vec3f mt1 = t1 + dt1 * mid;
                Vec3f mt2 = t2 + dt2 * mid;
                float Dm = dist2_pt(mp, mt0, mt1, mt2);
                if (Dm < d_min2) hi = mid;
                else lo = mid;
            }
            return lo;
        }
    }
    return 1.0f;
}

// ============================================================================
// Edge–edge CCD
// ============================================================================

CHYSX_HDI float ccd_ee(Vec3f a0, Vec3f da0,
                        Vec3f a1, Vec3f da1,
                        Vec3f b0, Vec3f db0,
                        Vec3f b1, Vec3f db1,
                        float d_min,
                        int max_iter = 20) {
    float d_min2 = d_min * d_min;

    int n_samples = max_iter;
    for (int k = 1; k <= n_samples; ++k) {
        float t = static_cast<float>(k) / static_cast<float>(n_samples);
        Vec3f aa0 = a0 + da0 * t;
        Vec3f aa1 = a1 + da1 * t;
        Vec3f bb0 = b0 + db0 * t;
        Vec3f bb1 = b1 + db1 * t;

        float D = dist2_ee(aa0, aa1, bb0, bb1);
        if (D < d_min2) {
            float lo = static_cast<float>(k - 1) / static_cast<float>(n_samples);
            float hi = t;
            for (int b = 0; b < 10; ++b) {
                float mid = (lo + hi) * 0.5f;
                Vec3f ma0 = a0 + da0 * mid;
                Vec3f ma1 = a1 + da1 * mid;
                Vec3f mb0 = b0 + db0 * mid;
                Vec3f mb1 = b1 + db1 * mid;
                float Dm = dist2_ee(ma0, ma1, mb0, mb1);
                if (Dm < d_min2) hi = mid;
                else lo = mid;
            }
            return lo;
        }
    }
    return 1.0f;
}

// ============================================================================
// Global CCD: find max safe step across all contact pairs
// ============================================================================

// Compute the maximum safe step for a system of bodies.
// verts_cur/verts_trial are the current and trial surface vertex positions.
// tris/edges index into the vertex arrays.
// Returns the minimum alpha across all PT and EE pairs.
inline float global_ccd(
        const std::vector<Vec3f>& verts_cur,
        const std::vector<Vec3f>& verts_trial,
        const std::vector<Vec3i>& tris,
        const std::vector<math::Vec2i>& edges,
        const std::vector<ContactPair>& active_contacts,
        float d_hat,
        int max_iter = 8) {
    float alpha = 1.0f;
    float d_min = d_hat * 1e-3f;

    for (auto& cp : active_contacts) {
        if (cp.type == ContactType::PT) {
            int ip = cp.v[0], it0 = cp.v[1], it1 = cp.v[2], it2 = cp.v[3];
            Vec3f dp  = verts_trial[ip]  - verts_cur[ip];
            Vec3f dt0 = verts_trial[it0] - verts_cur[it0];
            Vec3f dt1 = verts_trial[it1] - verts_cur[it1];
            Vec3f dt2 = verts_trial[it2] - verts_cur[it2];

            float a = ccd_pt(verts_cur[ip], dp,
                              verts_cur[it0], dt0,
                              verts_cur[it1], dt1,
                              verts_cur[it2], dt2,
                              d_min, max_iter);
            if (a < alpha) alpha = a;
        } else if (cp.type == ContactType::EE) {
            int ia0 = cp.v[0], ia1 = cp.v[1], ib0 = cp.v[2], ib1 = cp.v[3];
            Vec3f da0 = verts_trial[ia0] - verts_cur[ia0];
            Vec3f da1 = verts_trial[ia1] - verts_cur[ia1];
            Vec3f db0 = verts_trial[ib0] - verts_cur[ib0];
            Vec3f db1 = verts_trial[ib1] - verts_cur[ib1];

            float a = ccd_ee(verts_cur[ia0], da0,
                              verts_cur[ia1], da1,
                              verts_cur[ib0], db0,
                              verts_cur[ib1], db1,
                              d_min, max_iter);
            if (a < alpha) alpha = a;
        }
    }
    return alpha;
}

// ============================================================================
// Energy-based line search (Armijo backtracking)
// ============================================================================

// Given current energy E0 and directional derivative (g·Δq),
// backtrack alpha until E(q + alpha*Δq) < E0 + c * alpha * (g·Δq).
// The caller provides the energy evaluation function.
// Returns the accepted alpha.
inline float line_search_energy(
        float E0, float directional_deriv,
        float alpha_ccd,
        int max_iter,
        // Evaluate E(q + alpha * dq) — caller provides this
        float (*eval_energy)(float alpha, void* ctx),
        void* ctx,
        float c = 1e-4f) {
    float alpha = alpha_ccd;

    for (int i = 0; i < max_iter; ++i) {
        float E_trial = eval_energy(alpha, ctx);
        if (E_trial <= E0 + c * alpha * directional_deriv)
            return alpha;
        alpha *= 0.5f;
    }
    return alpha;
}

}  // namespace abd_ipc
}  // namespace chysx
