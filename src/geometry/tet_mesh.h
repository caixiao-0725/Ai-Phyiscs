// SPDX-License-Identifier: Apache-2.0
//
// chysx::geometry::TetMesh<T>
//
// Minimal tetrahedral mesh container.  Stores:
//   * vertices      : per-vertex positions (Vec3<T>)
//   * tetrahedra    : per-tet vertex indices (Vec4i)
//   * surface_tris  : boundary triangles (Vec3i), populated by extract_surface()
//   * surface_edges : boundary edges (Vec2i), populated by extract_surface()
//
// All buffers backed by CudaArray for seamless host/device transfer.
//
// extract_surface() finds boundary faces (faces shared by exactly one tet)
// and builds deduplicated edge / triangle lists on the host side.

#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "../memory/cuda_array.h"
#include "../math/vec.cuh"
#include "../math/matrix.cuh"

namespace chysx {
namespace geometry {

template <typename T = float>
class TetMesh {
public:
    using Scalar   = T;
    using Vertex   = math::Vec3<T>;
    using Tet      = math::Vec4i;     // (a, b, c, d) — indices into vertices
    using Triangle = math::Vec3i;
    using Edge     = math::Vec2i;

    TetMesh() = default;

    TetMesh(std::size_t vertex_count, std::size_t tet_count) {
        vertices_.resize(vertex_count);
        tets_.resize(tet_count);
    }

    TetMesh(const TetMesh&) = delete;
    TetMesh& operator=(const TetMesh&) = delete;
    TetMesh(TetMesh&&) noexcept = default;
    TetMesh& operator=(TetMesh&&) noexcept = default;

    // ---- sizing --------------------------------------------------------

    void resize_vertices(std::size_t n) { vertices_.resize(n); }
    void resize_tets(std::size_t n) { tets_.resize(n); }

    void clear() noexcept {
        vertices_.clear();
        tets_.clear();
        surface_tris_.clear();
        surface_edges_.clear();
    }

    // ---- volume --------------------------------------------------------

    // Signed volume of tet (a, b, c, d) using scalar triple product.
    static CHYSX_HDI T tet_volume(Vertex a, Vertex b, Vertex c, Vertex d) {
        Vertex ab = b - a, ac = c - a, ad = d - a;
        return dot(ab, cross(ac, ad)) / T{6};
    }

    // Total (unsigned) volume of the mesh, computed from host data.
    T total_volume() const {
        const Vertex* verts = vertices_.cpu_data();
        const Tet* tets     = tets_.cpu_data();
        T vol = T{0};
        for (std::size_t i = 0; i < tets_.cpu_size(); ++i) {
            const Tet& t = tets[i];
            T v = tet_volume(verts[t.x], verts[t.y], verts[t.z], verts[t.w]);
            vol += (v < T{0}) ? -v : v;
        }
        return vol;
    }

    // ---- surface extraction -------------------------------------------

    // Find boundary triangles (faces belonging to exactly one tet) and
    // build deduplicated edge / triangle lists on the host side.
    // Does NOT upload to device; caller does copy_to_device() if needed.
    void extract_surface() {
        const std::size_t n_tet = tets_.cpu_size();
        if (n_tet == 0) {
            surface_tris_.clear();
            surface_edges_.clear();
            return;
        }

        // A tet (a,b,c,d) has 4 faces. We use sorted-key lookup to find
        // faces shared by exactly one tet — those are on the boundary.
        using FaceKey = std::tuple<int, int, int>;
        // Value: oriented face (original winding) + count
        struct FaceInfo { int v[3]; int count; };
        std::map<FaceKey, FaceInfo> face_map;

        auto make_key = [](int a, int b, int c) -> FaceKey {
            int s[3] = {a, b, c};
            if (s[0] > s[1]) std::swap(s[0], s[1]);
            if (s[1] > s[2]) std::swap(s[1], s[2]);
            if (s[0] > s[1]) std::swap(s[0], s[1]);
            return {s[0], s[1], s[2]};
        };

        const Tet* tets = tets_.cpu_data();
        for (std::size_t i = 0; i < n_tet; ++i) {
            const int* v = &tets[i].x;
            // 4 faces per tet with outward-pointing normals (BCC convention)
            int faces[4][3] = {
                {v[0], v[2], v[1]},
                {v[0], v[1], v[3]},
                {v[0], v[3], v[2]},
                {v[1], v[2], v[3]},
            };
            for (int f = 0; f < 4; ++f) {
                FaceKey key = make_key(faces[f][0], faces[f][1], faces[f][2]);
                auto it = face_map.find(key);
                if (it == face_map.end()) {
                    face_map[key] = {{faces[f][0], faces[f][1], faces[f][2]}, 1};
                } else {
                    it->second.count++;
                }
            }
        }

        // Boundary faces: count == 1
        std::vector<Triangle> tris;
        for (auto& [key, info] : face_map) {
            if (info.count == 1)
                tris.push_back(Triangle(info.v[0], info.v[1], info.v[2]));
        }

        surface_tris_.allocate_host(tris.size());
        surface_tris_.allocate_device(tris.size());
        Triangle* out = surface_tris_.cpu_data();
        for (std::size_t i = 0; i < tris.size(); ++i) out[i] = tris[i];

        // Build edges from boundary triangles
        std::vector<std::pair<int, int>> edge_tmp;
        edge_tmp.reserve(tris.size() * 3);
        for (auto& tri : tris) {
            auto add = [&](int a, int b) {
                edge_tmp.emplace_back(std::min(a, b), std::max(a, b));
            };
            add(tri.x, tri.y);
            add(tri.y, tri.z);
            add(tri.z, tri.x);
        }
        std::sort(edge_tmp.begin(), edge_tmp.end());
        edge_tmp.erase(std::unique(edge_tmp.begin(), edge_tmp.end()),
                       edge_tmp.end());

        surface_edges_.allocate_host(edge_tmp.size());
        surface_edges_.allocate_device(edge_tmp.size());
        Edge* eout = surface_edges_.cpu_data();
        for (std::size_t i = 0; i < edge_tmp.size(); ++i)
            eout[i] = Edge(edge_tmp[i].first, edge_tmp[i].second);
    }

    // ---- buffer accessors ---------------------------------------------

    CudaArray<Vertex>&         vertices()       noexcept { return vertices_; }
    const CudaArray<Vertex>&   vertices() const noexcept { return vertices_; }
    CudaArray<Tet>&            tets()           noexcept { return tets_; }
    const CudaArray<Tet>&      tets() const     noexcept { return tets_; }
    CudaArray<Triangle>&       surface_tris()           noexcept { return surface_tris_; }
    const CudaArray<Triangle>& surface_tris() const     noexcept { return surface_tris_; }
    CudaArray<Edge>&           surface_edges()          noexcept { return surface_edges_; }
    const CudaArray<Edge>&     surface_edges() const    noexcept { return surface_edges_; }

    std::size_t num_vertices()      const noexcept { return vertices_.cpu_size(); }
    std::size_t num_tets()          const noexcept { return tets_.cpu_size(); }
    std::size_t num_surface_tris()  const noexcept { return surface_tris_.cpu_size(); }
    std::size_t num_surface_edges() const noexcept { return surface_edges_.cpu_size(); }

private:
    CudaArray<Vertex>   vertices_;
    CudaArray<Tet>      tets_;
    CudaArray<Triangle> surface_tris_;
    CudaArray<Edge>     surface_edges_;
};

using TetMeshf = TetMesh<float>;
using TetMeshd = TetMesh<double>;

}  // namespace geometry
}  // namespace chysx
