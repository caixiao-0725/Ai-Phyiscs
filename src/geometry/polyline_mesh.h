// SPDX-License-Identifier: Apache-2.0
//
// chysx::geometry::PolylineMesh
//
// Polyline (multi-curve) mesh container for yarn / rod simulations.
// Stores vertices, segments (edges), and per-curve start indices.
// Each buffer is a CudaArray, matching the TriangleMesh pattern.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../memory/cuda_array.h"
#include "../math/vec.cuh"

namespace chysx {
namespace geometry {

class PolylineMesh {
public:
    using Vertex  = math::Vec3f;
    using Segment = math::Vec2i;  // (i, i+1) vertex indices

    PolylineMesh() = default;

    PolylineMesh(const PolylineMesh&) = delete;
    PolylineMesh& operator=(const PolylineMesh&) = delete;
    PolylineMesh(PolylineMesh&&) noexcept = default;
    PolylineMesh& operator=(PolylineMesh&&) noexcept = default;

    // Build from a set of curves (each curve = ordered vertex list).
    // This populates vertices_, segments_, curve_starts_, and n_curves_.
    void build_from_curves(const std::vector<std::vector<math::Vec3f>>& curves) {
        vertices_.clear();
        segments_.clear();
        curve_starts_.clear();

        std::size_t total_verts = 0;
        for (auto& c : curves) total_verts += c.size();

        vertices_.allocate_host(total_verts);
        curve_starts_.reserve(curves.size() + 1);
        n_curves_ = static_cast<int>(curves.size());

        std::size_t seg_count = 0;
        for (auto& c : curves) {
            if (c.size() >= 2) seg_count += c.size() - 1;
        }
        segments_.allocate_host(seg_count);

        Vertex*  vp = vertices_.cpu_data();
        Segment* sp = segments_.cpu_data();
        int vi = 0;
        int si = 0;

        for (auto& c : curves) {
            curve_starts_.push_back(vi);
            for (std::size_t j = 0; j < c.size(); ++j) {
                vp[vi + j] = c[j];
            }
            for (std::size_t j = 0; j + 1 < c.size(); ++j) {
                sp[si++] = Segment(vi + static_cast<int>(j),
                                   vi + static_cast<int>(j) + 1);
            }
            vi += static_cast<int>(c.size());
        }
        curve_starts_.push_back(vi);
    }

    // Arc-length resample every curve to approximately `target_len` per segment.
    // Returns a new set of curves (host-only operation).
    static std::vector<std::vector<math::Vec3f>>
    resample_curves(const std::vector<std::vector<math::Vec3f>>& curves,
                    float target_len) {
        std::vector<std::vector<math::Vec3f>> out;
        out.reserve(curves.size());
        for (auto& pts : curves) {
            out.push_back(resample_single(pts, target_len));
        }
        return out;
    }

    // Upload both vertices and segments to GPU.
    void copy_to_device() {
        vertices_.copy_to_device();
        segments_.copy_to_device();
    }

    // ---- accessors -------------------------------------------------------

    CudaArray<Vertex>&        vertices()       noexcept { return vertices_; }
    const CudaArray<Vertex>&  vertices() const noexcept { return vertices_; }
    CudaArray<Segment>&       segments()       noexcept { return segments_; }
    const CudaArray<Segment>& segments() const noexcept { return segments_; }

    std::size_t num_vertices() const noexcept { return vertices_.cpu_size(); }
    std::size_t num_segments() const noexcept { return segments_.cpu_size(); }
    int         num_curves()   const noexcept { return n_curves_; }

    // curve_starts_[i] = first vertex index of curve i.
    // curve_starts_[n_curves_] = total vertex count (sentinel).
    const std::vector<int>& curve_starts() const noexcept { return curve_starts_; }

private:
    CudaArray<Vertex>  vertices_;
    CudaArray<Segment> segments_;
    int                n_curves_ = 0;
    std::vector<int>   curve_starts_;

    // Resample a single polyline to approximately `target_len` per segment.
    static std::vector<math::Vec3f>
    resample_single(const std::vector<math::Vec3f>& pts, float target_len) {
        if (pts.size() < 2) return pts;

        // Compute cumulative arc-length.
        std::vector<float> arc(pts.size(), 0.0f);
        for (std::size_t i = 1; i < pts.size(); ++i) {
            auto d = pts[i] - pts[i - 1];
            arc[i] = arc[i - 1] + std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        }
        float total = arc.back();
        if (total < 1e-12f) return pts;

        int n_segs = std::max(1, static_cast<int>(std::round(total / target_len)));
        float step = total / n_segs;

        std::vector<math::Vec3f> out;
        out.reserve(n_segs + 1);
        out.push_back(pts.front());

        std::size_t j = 1;
        for (int i = 1; i < n_segs; ++i) {
            float s = i * step;
            while (j < pts.size() - 1 && arc[j] < s) ++j;
            float t = (s - arc[j - 1]) / (arc[j] - arc[j - 1]);
            math::Vec3f p;
            p.x = pts[j - 1].x + t * (pts[j].x - pts[j - 1].x);
            p.y = pts[j - 1].y + t * (pts[j].y - pts[j - 1].y);
            p.z = pts[j - 1].z + t * (pts[j].z - pts[j - 1].z);
            out.push_back(p);
        }
        out.push_back(pts.back());
        return out;
    }
};

}  // namespace geometry
}  // namespace chysx
