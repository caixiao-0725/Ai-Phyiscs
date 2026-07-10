// SPDX-License-Identifier: Apache-2.0

#include "mesh_mesh_contact.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

namespace chysx {
namespace collision {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(err));
    }
}

constexpr int kContactBlock = 256;
constexpr int kCountDrivenBlocksPerSm = 4;

inline int count_driven_max_blocks() {
    static const int max_blocks = [] {
        int device = 0;
        int sm_count = 0;
        check_cuda(cudaGetDevice(&device), "query current CUDA device");
        check_cuda(cudaDeviceGetAttribute(&sm_count,
                                          cudaDevAttrMultiProcessorCount,
                                          device),
                   "query CUDA SM count");
        return std::max(1, sm_count * kCountDrivenBlocksPerSm);
    }();
    return max_blocks;
}

inline int bounded_count_grid(int capacity) {
    const int grid = (std::max(1, capacity) + kContactBlock - 1) /
                     kContactBlock;
    return std::min(grid, count_driven_max_blocks());
}

struct DeviceFloatMax {
    __device__ float operator()(float a, float b) const {
        return fmaxf(a, b);
    }
};

__global__ void broadphase_max_displacement_sq_kernel(
    const math::Vec3f* __restrict__ positions,
    const math::Vec3f* __restrict__ reference_positions,
    int n_verts,
    std::uint32_t* __restrict__ max_displacement_sq_bits)
{
    using BlockReduce = cub::BlockReduce<float, kContactBlock>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;

    float thread_max = 0.0f;
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_verts; i += stride) {
        const math::Vec3f d = positions[i] - reference_positions[i];
        thread_max = fmaxf(thread_max, math::dot(d, d));
    }

    const float block_max = BlockReduce(reduce_storage).Reduce(
        thread_max, DeviceFloatMax{});
    if (threadIdx.x == 0) {
        atomicMax(max_displacement_sq_bits, __float_as_uint(block_max));
    }
}

__device__ void closest_point_triangle(
    math::Vec3f p,
    math::Vec3f a,
    math::Vec3f b,
    math::Vec3f c,
    math::Vec3f& closest,
    math::Vec3f& bary)
{
    math::Vec3f ab = b - a;
    math::Vec3f ac = c - a;
    math::Vec3f ap = p - a;
    float d1 = math::dot(ab, ap);
    float d2 = math::dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        closest = a;
        bary = math::Vec3f(1.0f, 0.0f, 0.0f);
        return;
    }

    math::Vec3f bp = p - b;
    float d3 = math::dot(ab, bp);
    float d4 = math::dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        closest = b;
        bary = math::Vec3f(0.0f, 1.0f, 0.0f);
        return;
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        closest = a + ab * v;
        bary = math::Vec3f(1.0f - v, v, 0.0f);
        return;
    }

    math::Vec3f cp = p - c;
    float d5 = math::dot(ab, cp);
    float d6 = math::dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        closest = c;
        bary = math::Vec3f(0.0f, 0.0f, 1.0f);
        return;
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        closest = a + ac * w;
        bary = math::Vec3f(1.0f - w, 0.0f, w);
        return;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closest = b + (c - b) * w;
        bary = math::Vec3f(0.0f, 1.0f - w, w);
        return;
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    closest = a + ab * v + ac * w;
    bary = math::Vec3f(1.0f - v - w, v, w);
}

__device__ bool closest_segment_segment(
    math::Vec3f p0,
    math::Vec3f p1,
    math::Vec3f q0,
    math::Vec3f q1,
    float& s,
    float& t,
    math::Vec3f& cp,
    math::Vec3f& cq)
{
    math::Vec3f d1 = p1 - p0;
    math::Vec3f d2 = q1 - q0;
    math::Vec3f r = p0 - q0;
    float a = math::dot(d1, d1);
    float e = math::dot(d2, d2);
    float f = math::dot(d2, r);
    const float eps = 1e-12f;

    if (a <= eps || e <= eps) return false;

    float c = math::dot(d1, r);
    float b = math::dot(d1, d2);
    float denom = a * e - b * b;
    if (fabsf(denom) <= eps) return false;

    s = fminf(fmaxf((b * f - c * e) / denom, 0.0f), 1.0f);
    t = (b * s + f) / e;
    if (t < 0.0f) {
        t = 0.0f;
        s = fminf(fmaxf(-c / a, 0.0f), 1.0f);
    } else if (t > 1.0f) {
        t = 1.0f;
        s = fminf(fmaxf((b - c) / a, 0.0f), 1.0f);
    }

    cp = p0 + d1 * s;
    cq = q0 + d2 * t;
    return true;
}

__device__ bool same_mesh_pf(
    int point,
    math::Vec3i face,
    const int* vertex_mesh_ids,
    int& mesh_point,
    int& mesh_face)
{
    mesh_point = vertex_mesh_ids[point];
    int m0 = vertex_mesh_ids[face.x];
    int m1 = vertex_mesh_ids[face.y];
    int m2 = vertex_mesh_ids[face.z];
    mesh_face = m0;
    return mesh_point == m0 && mesh_point == m1 && mesh_point == m2;
}

__device__ bool same_mesh_ee(
    math::Vec2i ea,
    math::Vec2i eb,
    const int* vertex_mesh_ids,
    int& mesh_a,
    int& mesh_b)
{
    int a0 = vertex_mesh_ids[ea.x];
    int a1 = vertex_mesh_ids[ea.y];
    int b0 = vertex_mesh_ids[eb.x];
    int b1 = vertex_mesh_ids[eb.y];
    mesh_a = a0;
    mesh_b = b0;
    return a0 == a1 && a0 == b0 && a0 == b1;
}

__device__ bool collision_category_enabled(bool same_mesh,
                                           MeshCollisionMask mask) {
    const MeshCollisionMask bit = same_mesh
        ? kMeshCollisionSelf
        : kMeshCollisionInterObject;
    return (mask & bit) != 0;
}

__device__ int collision_category(bool same_mesh) {
    return static_cast<int>(same_mesh
        ? MeshCollisionCategory::Self
        : MeshCollisionCategory::InterObject);
}

__device__ bool oriented_face_normal(
    math::Vec3i face,
    const math::Vec3f* verts,
    float orientation_sign,
    math::Vec3f& normal)
{
    if (orientation_sign == 0.0f) return false;
    const math::Vec3f raw = math::cross(
        verts[face.y] - verts[face.x],
        verts[face.z] - verts[face.x]);
    const float len_sq = math::dot(raw, raw);
    if (len_sq <= 1.0e-24f) return false;
    normal = raw * (orientation_sign * rsqrtf(len_sq));
    return true;
}

__device__ bool edge_outward_normal(
    int edge_id,
    const math::Vec3f* verts,
    const math::Vec3i* faces,
    const math::Vec2i* edge2face,
    float orientation_sign,
    math::Vec3f& normal)
{
    if (orientation_sign == 0.0f) return false;
    const math::Vec2i adjacent = edge2face[edge_id];
    math::Vec3f sum(0.0f, 0.0f, 0.0f);
    int count = 0;
    if (adjacent.x >= 0) {
        math::Vec3f face_normal;
        if (oriented_face_normal(
                faces[adjacent.x], verts, orientation_sign, face_normal)) {
            sum += face_normal;
            ++count;
        }
    }
    if (adjacent.y >= 0) {
        math::Vec3f face_normal;
        if (oriented_face_normal(
                faces[adjacent.y], verts, orientation_sign, face_normal)) {
            sum += face_normal;
            ++count;
        }
    }
    const float len_sq = math::dot(sum, sum);
    if (count == 0 || len_sq <= 1.0e-16f) return false;
    normal = sum * rsqrtf(len_sq);
    return true;
}

__device__ void emit_contact(
    MeshMeshContact* out,
    int* count,
    int max_contacts,
    const MeshMeshContact& contact)
{
    int idx = atomicAdd(count, 1);
    if (idx < max_contacts) {
        out[idx] = contact;
    }
}

__global__ void zero_count_kernel(int* count) {
    *count = 0;
}

__global__ void ef_to_mesh_mesh_contacts_kernel(
    const math::Vec2i* __restrict__ ef_pairs,
    const int* __restrict__ n_ef_ptr,
    int max_ef,
    const math::Vec3f* __restrict__ verts,
    const math::Vec3i* __restrict__ faces,
    const math::Vec2i* __restrict__ edges,
    const math::Vec2i* __restrict__ edge2face,
    const int* __restrict__ vert_in_edge,
    const math::Vec3i* __restrict__ edge_in_face,
    const int* __restrict__ vertex_mesh_ids,
    const float* __restrict__ vertex_orientation_signs,
    float thickness,
    MeshCollisionMask collision_mask,
    MeshMeshContact* contacts,
    int* contact_count,
    int max_contacts)
{
    const int n_raw = *n_ef_ptr;
    const int n_ef = n_raw < max_ef ? n_raw : max_ef;
    const float thickness_sq = thickness * thickness;
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_ef; i += stride) {
        const math::Vec2i ef = ef_pairs[i];
        const int eid = ef.x;
        const int fid = ef.y;

        const int vid = vert_in_edge[eid];
        if (vid >= 0) {
            math::Vec3i f = faces[fid];
            if (vid != f.x && vid != f.y && vid != f.z) {
                int mesh_point = -1;
                int mesh_face = -1;
                const bool same = same_mesh_pf(
                    vid, f, vertex_mesh_ids, mesh_point, mesh_face);
                if (collision_category_enabled(same, collision_mask)) {
                    math::Vec3f p = verts[vid];
                    math::Vec3f a = verts[f.x];
                    math::Vec3f b = verts[f.y];
                    math::Vec3f c = verts[f.z];
                    math::Vec3f closest;
                    math::Vec3f bary;
                    closest_point_triangle(p, a, b, c, closest, bary);
                    math::Vec3f diff = p - closest;
                    float d2 = math::dot(diff, diff);
                    if (d2 > 1e-24f && d2 < thickness_sq) {
                        float d = sqrtf(d2);
                        math::Vec3f normal = diff * (1.0f / d);
                        float separation = d;
                        bool valid_pf = true;
                        const float orientation_sign =
                            vertex_orientation_signs[f.x];
                        if (!same && orientation_sign != 0.0f) {
                            const float bary_eps = 1.0e-6f;
                            if (bary.x <= bary_eps || bary.y <= bary_eps ||
                                bary.z <= bary_eps) {
                                valid_pf = false;
                            }
                            if (valid_pf && !oriented_face_normal(
                                                f, verts, orientation_sign,
                                                normal)) {
                                valid_pf = false;
                            }
                            if (valid_pf) {
                                separation = math::dot(diff, normal);
                            }
                        }
                        if (valid_pf) {
                            MeshMeshContact contact;
                            contact.type = static_cast<int>(
                                MeshMeshContactType::PointFace);
                            contact.vertices = math::Vec4i(
                                vid, f.x, f.y, f.z);
                            contact.weights = math::Vec4f(
                                1.0f, -bary.x, -bary.y, -bary.z);
                            contact.normal = normal;
                            contact.distance = d;
                            contact.mesh_pair = math::Vec2i(
                                mesh_point, mesh_face);
                            contact.source_ef = ef;
                            contact.category = collision_category(same);
                            contact.separation = separation;
                            emit_contact(contacts, contact_count, max_contacts,
                                         contact);
                        }
                    }
                }
            }
        }

        math::Vec2i ea = edges[eid];
        math::Vec3i face_edges = edge_in_face[fid];
        for (int k = 0; k < 3; ++k) {
            int other_eid = face_edges.data[k];
            if (other_eid < 0 || other_eid == eid) continue;
            if (other_eid >= eid) continue;

            math::Vec2i eb = edges[other_eid];
            if (ea.x == eb.x || ea.x == eb.y ||
                ea.y == eb.x || ea.y == eb.y) continue;

            int mesh_a = -1;
            int mesh_b = -1;
            const bool same = same_mesh_ee(
                ea, eb, vertex_mesh_ids, mesh_a, mesh_b);
            if (!collision_category_enabled(same, collision_mask)) continue;

            math::Vec3f a0 = verts[ea.x];
            math::Vec3f a1 = verts[ea.y];
            math::Vec3f b0 = verts[eb.x];
            math::Vec3f b1 = verts[eb.y];
            float s = 0.0f;
            float t = 0.0f;
            math::Vec3f cp;
            math::Vec3f cq;
            if (!closest_segment_segment(
                    a0, a1, b0, b1, s, t, cp, cq)) continue;

            const float eps_s = 1e-6f;
            if (s <= eps_s || s >= 1.0f - eps_s ||
                t <= eps_s || t >= 1.0f - eps_s) continue;

            math::Vec3f da = a1 - a0;
            math::Vec3f db = b1 - b0;
            math::Vec3f cross_d = math::cross(da, db);
            float cross_sq = math::dot(cross_d, cross_d);
            float len_scale = math::dot(da, da) * math::dot(db, db);
            if (cross_sq < 1e-3f * len_scale) continue;

            math::Vec3f diff = cp - cq;
            float d2 = math::dot(diff, diff);
            if (d2 > 1e-24f && d2 < thickness_sq) {
                float d = sqrtf(d2);
                math::Vec3f normal = diff * (1.0f / d);
                float separation = d;
                const float orientation_sign_a =
                    vertex_orientation_signs[ea.x];
                const float orientation_sign_b =
                    vertex_orientation_signs[eb.x];
                if (!same && orientation_sign_a != 0.0f &&
                    orientation_sign_b != 0.0f) {
                    math::Vec3f outward_a;
                    math::Vec3f outward_b;
                    if (edge_outward_normal(
                            eid, verts, faces, edge2face,
                            orientation_sign_a, outward_a) &&
                        edge_outward_normal(
                            other_eid, verts, faces, edge2face,
                            orientation_sign_b, outward_b)) {
                        normal = cross_d * rsqrtf(cross_sq);
                        const math::Vec3f separating_hint =
                            outward_b - outward_a;
                        if (math::dot(normal, separating_hint) < 0.0f) {
                            normal = -normal;
                        }
                        separation = math::dot(diff, normal);
                    }
                }
                MeshMeshContact contact;
                contact.type = static_cast<int>(
                    MeshMeshContactType::EdgeEdge);
                contact.vertices = math::Vec4i(ea.x, ea.y, eb.x, eb.y);
                contact.weights = math::Vec4f(
                    1.0f - s, s, -(1.0f - t), -t);
                contact.normal = normal;
                contact.distance = d;
                contact.mesh_pair = math::Vec2i(mesh_a, mesh_b);
                contact.source_ef = ef;
                contact.category = collision_category(same);
                contact.separation = separation;
                emit_contact(contacts, contact_count, max_contacts, contact);
            }
        }
    }
}

__global__ void self_pre_vf_mesh_mesh_contacts_kernel(
    const math::Vec4i* __restrict__ adjacent_vf,
    int n_adj,
    const math::Vec3f* __restrict__ verts,
    const int* __restrict__ vertex_mesh_ids,
    float thickness,
    MeshMeshContact* contacts,
    int* contact_count,
    int max_contacts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_adj) return;

    const float thickness_sq = thickness * thickness;
    math::Vec4i pr = adjacent_vf[i];
    int vid = pr.x;
    math::Vec3i f(pr.y, pr.z, pr.w);

    int mesh_point = -1;
    int mesh_face = -1;
    if (!same_mesh_pf(vid, f, vertex_mesh_ids, mesh_point, mesh_face)) return;

    math::Vec3f closest;
    math::Vec3f bary;
    math::Vec3f p = verts[vid];
    closest_point_triangle(p, verts[f.x], verts[f.y], verts[f.z], closest, bary);
    math::Vec3f diff = p - closest;
    float d2 = math::dot(diff, diff);
    if (d2 > 1e-24f && d2 < thickness_sq) {
        float d = sqrtf(d2);
        MeshMeshContact contact;
        contact.type = static_cast<int>(MeshMeshContactType::PointFace);
        contact.vertices = math::Vec4i(vid, f.x, f.y, f.z);
        contact.weights = math::Vec4f(1.0f, -bary.x, -bary.y, -bary.z);
        contact.normal = diff * (1.0f / d);
        contact.distance = d;
        contact.mesh_pair = math::Vec2i(mesh_point, mesh_face);
        contact.source_ef = math::Vec2i(-1, -1);
        contact.category = static_cast<int>(MeshCollisionCategory::Self);
        contact.separation = d;
        emit_contact(contacts, contact_count, max_contacts, contact);
    }
}

__global__ void self_pre_ee_mesh_mesh_contacts_kernel(
    const math::Vec2i* __restrict__ adjacent_ee,
    int n_adj,
    const math::Vec3f* __restrict__ verts,
    const math::Vec2i* __restrict__ edges,
    const int* __restrict__ vertex_mesh_ids,
    float thickness,
    MeshMeshContact* contacts,
    int* contact_count,
    int max_contacts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_adj) return;

    const float thickness_sq = thickness * thickness;
    math::Vec2i edge_pair = adjacent_ee[i];
    math::Vec2i ea = edges[edge_pair.x];
    math::Vec2i eb = edges[edge_pair.y];
    if (ea.x == eb.x || ea.x == eb.y ||
        ea.y == eb.x || ea.y == eb.y) return;

    int mesh_a = -1;
    int mesh_b = -1;
    if (!same_mesh_ee(ea, eb, vertex_mesh_ids, mesh_a, mesh_b)) return;

    float s = 0.0f;
    float t = 0.0f;
    math::Vec3f cp;
    math::Vec3f cq;
    math::Vec3f a0 = verts[ea.x];
    math::Vec3f a1 = verts[ea.y];
    math::Vec3f b0 = verts[eb.x];
    math::Vec3f b1 = verts[eb.y];
    if (!closest_segment_segment(a0, a1, b0, b1, s, t, cp, cq)) return;

    const float eps_s = 1e-6f;
    if (s <= eps_s || s >= 1.0f - eps_s ||
        t <= eps_s || t >= 1.0f - eps_s) return;

    math::Vec3f da = a1 - a0;
    math::Vec3f db = b1 - b0;
    math::Vec3f cross_d = math::cross(da, db);
    float cross_sq = math::dot(cross_d, cross_d);
    float len_scale = math::dot(da, da) * math::dot(db, db);
    if (cross_sq < 1e-3f * len_scale) return;

    math::Vec3f diff = cp - cq;
    float d2 = math::dot(diff, diff);
    if (d2 > 1e-24f && d2 < thickness_sq) {
        float d = sqrtf(d2);
        MeshMeshContact contact;
        contact.type = static_cast<int>(MeshMeshContactType::EdgeEdge);
        contact.vertices = math::Vec4i(ea.x, ea.y, eb.x, eb.y);
        contact.weights = math::Vec4f(1.0f - s, s, -(1.0f - t), -t);
        contact.normal = diff * (1.0f / d);
        contact.distance = d;
        contact.mesh_pair = math::Vec2i(mesh_a, mesh_b);
        contact.source_ef = math::Vec2i(-1, -1);
        contact.category = static_cast<int>(MeshCollisionCategory::Self);
        contact.separation = d;
        emit_contact(contacts, contact_count, max_contacts, contact);
    }
}

}  // namespace

void MeshMeshContactDetector::setup(
    const std::vector<math::Vec3i>& triangles,
    const std::vector<int>& vertex_mesh_ids,
    int max_contacts,
    int max_ef_candidates,
    BroadphaseBackend backend,
    MeshCollisionMask collision_mask,
    const std::vector<math::Vec3f>* reference_positions)
{
    n_verts_ = static_cast<int>(vertex_mesh_ids.size());
    if (n_verts_ <= 0 || triangles.empty()) {
        throw std::runtime_error("MeshMeshContactDetector::setup needs vertices and triangles");
    }

    backend_ = backend;
    collision_mask_ = collision_mask & kMeshCollisionAll;
    if (backend_ == BroadphaseBackend::QuantBvh) {
        broadphase_.setup(triangles, n_verts_, max_ef_candidates);
        max_ef_candidates_ = broadphase_.max_ef_candidates();
    }
#ifdef CHYSX_HAS_OPTIX
    else if (backend_ == BroadphaseBackend::OptiX) {
        const int hits_per_edge = 64;
        optix_broadphase_ = std::make_unique<OptixEFBroadphase>();
        optix_broadphase_->setup(triangles, n_verts_, hits_per_edge);
        max_ef_candidates_ = optix_broadphase_->max_ef_candidates();
    }
#else
    else if (backend_ == BroadphaseBackend::OptiX) {
        throw std::runtime_error(
            "MeshMeshContactDetector::setup: OptiX backend requested but "
            "CHYSX_HAS_OPTIX is not enabled");
    }
#endif

    if (max_contacts <= 0) {
        max_contacts = std::max(1024, 8 * topology().n_edges());
    }
    max_contacts_ = max_contacts;

    vertex_mesh_ids_.resize(vertex_mesh_ids.size());
    std::memcpy(vertex_mesh_ids_.cpu_data(),
                vertex_mesh_ids.data(),
                vertex_mesh_ids.size() * sizeof(int));
    vertex_mesh_ids_.copy_to_device();

    vertex_orientation_signs_.resize(vertex_mesh_ids.size());
    std::fill(vertex_orientation_signs_.cpu_data(),
              vertex_orientation_signs_.cpu_data() + vertex_mesh_ids.size(),
              0.0f);
    if (reference_positions != nullptr &&
        reference_positions->size() == vertex_mesh_ids.size()) {
        int max_mesh_id = -1;
        for (int mesh_id : vertex_mesh_ids) {
            max_mesh_id = std::max(max_mesh_id, mesh_id);
        }
        const int mesh_count = max_mesh_id + 1;
        if (mesh_count > 0) {
            std::vector<double> sums(
                static_cast<std::size_t>(mesh_count) * 3, 0.0);
            std::vector<int> counts(static_cast<std::size_t>(mesh_count), 0);
            std::vector<double> mins(
                static_cast<std::size_t>(mesh_count) * 3,
                std::numeric_limits<double>::infinity());
            std::vector<double> maxs(
                static_cast<std::size_t>(mesh_count) * 3,
                -std::numeric_limits<double>::infinity());
            for (std::size_t v = 0; v < vertex_mesh_ids.size(); ++v) {
                const int mesh_id = vertex_mesh_ids[v];
                if (mesh_id < 0 || mesh_id >= mesh_count) continue;
                const math::Vec3f p = (*reference_positions)[v];
                const double values[3] = {p.x, p.y, p.z};
                for (int axis = 0; axis < 3; ++axis) {
                    const std::size_t slot =
                        static_cast<std::size_t>(mesh_id) * 3 + axis;
                    sums[slot] += values[axis];
                    mins[slot] = std::min(mins[slot], values[axis]);
                    maxs[slot] = std::max(maxs[slot], values[axis]);
                }
                ++counts[static_cast<std::size_t>(mesh_id)];
            }

            std::vector<double> volume6(
                static_cast<std::size_t>(mesh_count), 0.0);
            for (const math::Vec3i tri : triangles) {
                const int mesh_id = vertex_mesh_ids[tri.x];
                if (mesh_id < 0 || mesh_id >= mesh_count ||
                    vertex_mesh_ids[tri.y] != mesh_id ||
                    vertex_mesh_ids[tri.z] != mesh_id ||
                    counts[static_cast<std::size_t>(mesh_id)] == 0) {
                    continue;
                }
                const double inv_count = 1.0 /
                    static_cast<double>(counts[static_cast<std::size_t>(mesh_id)]);
                const std::size_t base =
                    static_cast<std::size_t>(mesh_id) * 3;
                const math::Vec3f pa = (*reference_positions)[tri.x];
                const math::Vec3f pb = (*reference_positions)[tri.y];
                const math::Vec3f pc = (*reference_positions)[tri.z];
                const double ax = pa.x - sums[base + 0] * inv_count;
                const double ay = pa.y - sums[base + 1] * inv_count;
                const double az = pa.z - sums[base + 2] * inv_count;
                const double bx = pb.x - sums[base + 0] * inv_count;
                const double by = pb.y - sums[base + 1] * inv_count;
                const double bz = pb.z - sums[base + 2] * inv_count;
                const double cx = pc.x - sums[base + 0] * inv_count;
                const double cy = pc.y - sums[base + 1] * inv_count;
                const double cz = pc.z - sums[base + 2] * inv_count;
                volume6[static_cast<std::size_t>(mesh_id)] +=
                    ax * (by * cz - bz * cy) +
                    ay * (bz * cx - bx * cz) +
                    az * (bx * cy - by * cx);
            }

            std::vector<float> orientation(
                static_cast<std::size_t>(mesh_count), 0.0f);
            for (int mesh_id = 0; mesh_id < mesh_count; ++mesh_id) {
                if (counts[static_cast<std::size_t>(mesh_id)] == 0) continue;
                const std::size_t base =
                    static_cast<std::size_t>(mesh_id) * 3;
                const double dx = maxs[base + 0] - mins[base + 0];
                const double dy = maxs[base + 1] - mins[base + 1];
                const double dz = maxs[base + 2] - mins[base + 2];
                const double diag = std::sqrt(dx * dx + dy * dy + dz * dz);
                const double threshold =
                    1.0e-10 * std::max(diag * diag * diag, 1.0e-12);
                const double signed_volume6 =
                    volume6[static_cast<std::size_t>(mesh_id)];
                if (std::abs(signed_volume6) > threshold) {
                    orientation[static_cast<std::size_t>(mesh_id)] =
                        signed_volume6 > 0.0 ? 1.0f : -1.0f;
                }
            }
            for (std::size_t v = 0; v < vertex_mesh_ids.size(); ++v) {
                const int mesh_id = vertex_mesh_ids[v];
                if (mesh_id >= 0 && mesh_id < mesh_count) {
                    vertex_orientation_signs_.cpu_data()[v] =
                        orientation[static_cast<std::size_t>(mesh_id)];
                }
            }
        }
    }
    vertex_orientation_signs_.copy_to_device();

    run_self_prepass_ = (collision_mask_ & kMeshCollisionSelf) != 0;

    contacts_.resize(static_cast<std::size_t>(max_contacts_));
    count_.resize(1);
    ef_count_snapshot_.resize(1);
    broadphase_dropped_hits_snapshot_.resize(1);
    broadphase_reference_positions_.allocate_device(
        static_cast<std::size_t>(n_verts_));
    broadphase_max_displacement_sq_bits_.resize(1);
    upload_positions_.resize(static_cast<std::size_t>(n_verts_));
    count_.cpu_data()[0] = 0;
    ef_count_snapshot_.cpu_data()[0] = 0;
    broadphase_dropped_hits_snapshot_.cpu_data()[0] = 0;
    broadphase_max_displacement_sq_bits_.cpu_data()[0] = 0;
    count_.copy_to_device();
    broadphase_max_displacement_sq_bits_.copy_to_device();
    invalidate_broadphase_cache();
}

void MeshMeshContactDetector::configure_broadphase_cache(
    int max_interval,
    float skin) noexcept
{
    max_interval = std::max(1, max_interval);
    skin = std::max(0.0f, skin);
    if (max_interval == broadphase_max_interval_ &&
        skin == broadphase_skin_) {
        return;
    }
    broadphase_max_interval_ = max_interval;
    broadphase_skin_ = skin;
    invalidate_broadphase_cache();
}

void MeshMeshContactDetector::invalidate_broadphase_cache() noexcept {
    broadphase_cache_valid_ = false;
    broadphase_cache_age_ = 0;
    broadphase_refresh_count_ = 0;
    broadphase_cached_narrow_thickness_ = -1.0f;
    last_broadphase_refreshed_ = true;
    last_broadphase_max_displacement_ = 0.0f;
}

bool MeshMeshContactDetector::valid() const noexcept {
    if (backend_ == BroadphaseBackend::QuantBvh) {
        return broadphase_.valid();
    }
#ifdef CHYSX_HAS_OPTIX
    if (backend_ == BroadphaseBackend::OptiX) {
        return optix_broadphase_ && optix_broadphase_->valid();
    }
#endif
    return false;
}

const MeshTopology& MeshMeshContactDetector::topology() const noexcept {
#ifdef CHYSX_HAS_OPTIX
    if (backend_ == BroadphaseBackend::OptiX && optix_broadphase_) {
        return optix_broadphase_->topology();
    }
#endif
    return broadphase_.topology();
}

void MeshMeshContactDetector::detect(
    const math::Vec3f* positions_cpu,
    int n_verts,
    float thickness,
    std::uintptr_t cuda_stream)
{
    if (!valid() || n_verts != n_verts_) return;
    std::memcpy(upload_positions_.cpu_data(),
                positions_cpu,
                static_cast<std::size_t>(n_verts_) * sizeof(math::Vec3f));
    upload_positions_.copy_to_device(cuda_stream);
    detect_gpu(upload_positions_.gpu_data(), thickness, cuda_stream);
}

void MeshMeshContactDetector::detect_gpu(
    const math::Vec3f* positions_dev,
    float thickness,
    std::uintptr_t cuda_stream)
{
    if (!valid() || positions_dev == nullptr || thickness <= 0.0f) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    zero_count_kernel<<<1, 1, 0, stream>>>(count_.gpu_data());
    check_cuda(cudaGetLastError(), "mesh_mesh zero_count_kernel");

    const bool cache_enabled =
        broadphase_max_interval_ > 1 && broadphase_skin_ > 0.0f;
    const bool thickness_changed =
        broadphase_cache_valid_ &&
        thickness != broadphase_cached_narrow_thickness_;
    bool refresh_broadphase =
        !cache_enabled || !broadphase_cache_valid_ || thickness_changed;
    if (cache_enabled && broadphase_cache_valid_ && !thickness_changed) {
        ++broadphase_cache_age_;
        refresh_broadphase =
            broadphase_cache_age_ >= broadphase_max_interval_;

        if (!refresh_broadphase) {
            check_cuda(cudaMemsetAsync(
                           broadphase_max_displacement_sq_bits_.gpu_data(),
                           0,
                           sizeof(std::uint32_t),
                           stream),
                       "broadphase displacement reset");
            const int grid = bounded_count_grid(n_verts_);
            broadphase_max_displacement_sq_kernel<<<
                grid, kContactBlock, 0, stream>>>(
                positions_dev,
                broadphase_reference_positions_.gpu_data(),
                n_verts_,
                broadphase_max_displacement_sq_bits_.gpu_data());
            check_cuda(cudaGetLastError(),
                       "broadphase_max_displacement_sq_kernel");
            broadphase_max_displacement_sq_bits_.copy_to_host(cuda_stream);
            check_cuda(cudaStreamSynchronize(stream),
                       "broadphase displacement decision sync");

            float max_displacement_sq = 0.0f;
            const std::uint32_t bits =
                broadphase_max_displacement_sq_bits_.cpu_data()[0];
            std::memcpy(&max_displacement_sq, &bits, sizeof(float));
            last_broadphase_max_displacement_ =
                std::sqrt(std::max(0.0f, max_displacement_sq));
            refresh_broadphase =
                last_broadphase_max_displacement_ >= 0.5f * broadphase_skin_;
        }
    }

    const math::Vec2i* ef_pairs_dev = nullptr;
    const int* ef_count_dev = nullptr;
    const int* dropped_hits_dev = nullptr;
    const float broadphase_thickness =
        thickness + (cache_enabled ? broadphase_skin_ : 0.0f);
    if (backend_ == BroadphaseBackend::QuantBvh) {
        if (refresh_broadphase) {
            broadphase_.query(
                positions_dev, broadphase_thickness, cuda_stream);
        }
        ef_pairs_dev = broadphase_.ef_pairs_dev();
        ef_count_dev = broadphase_.ef_count_dev();
    }
#ifdef CHYSX_HAS_OPTIX
    else if (backend_ == BroadphaseBackend::OptiX && optix_broadphase_) {
        if (refresh_broadphase) {
            optix_broadphase_->query(
                positions_dev,
                vertex_mesh_ids_.gpu_data(),
                collision_mask_,
                broadphase_thickness,
                cuda_stream);
        }
        ef_pairs_dev = optix_broadphase_->ef_pairs_dev();
        ef_count_dev = optix_broadphase_->ef_count_dev();
        dropped_hits_dev = optix_broadphase_->dropped_hits_dev();
    }
#endif

    last_broadphase_refreshed_ = refresh_broadphase;
    if (refresh_broadphase) {
        check_cuda(cudaMemcpyAsync(
                       broadphase_reference_positions_.gpu_data(),
                       positions_dev,
                       static_cast<std::size_t>(n_verts_) *
                           sizeof(math::Vec3f),
                       cudaMemcpyDeviceToDevice,
                       stream),
                   "broadphase reference positions snapshot");
        broadphase_cache_valid_ = true;
        broadphase_cache_age_ = 0;
        broadphase_cached_narrow_thickness_ = thickness;
        ++broadphase_refresh_count_;
    }

    const auto& topo = topology();
    if (ef_pairs_dev != nullptr && ef_count_dev != nullptr) {
        const int grid = bounded_count_grid(max_ef_candidates_);
        ef_to_mesh_mesh_contacts_kernel<<<
            grid, kContactBlock, 0, stream>>>(
            ef_pairs_dev,
            ef_count_dev,
            max_ef_candidates_,
            positions_dev,
            topo.faces().gpu_data(),
            topo.edges().gpu_data(),
            topo.edge2face().gpu_data(),
            topo.vert_in_edge().gpu_data(),
            topo.edge_in_face().gpu_data(),
            vertex_mesh_ids_.gpu_data(),
            vertex_orientation_signs_.gpu_data(),
            thickness,
            collision_mask_,
            contacts_.gpu_data(),
            count_.gpu_data(),
            max_contacts_);
        check_cuda(cudaGetLastError(), "ef_to_mesh_mesh_contacts_kernel");
    }

    if (run_self_prepass_ && topo.n_adj_vf() > 0) {
        int grid = (topo.n_adj_vf() + kContactBlock - 1) /
                   kContactBlock;
        self_pre_vf_mesh_mesh_contacts_kernel<<<
            grid, kContactBlock, 0, stream>>>(
            topo.pre_adj_vf().gpu_data(),
            topo.n_adj_vf(),
            positions_dev,
            vertex_mesh_ids_.gpu_data(),
            thickness,
            contacts_.gpu_data(),
            count_.gpu_data(),
            max_contacts_);
        check_cuda(cudaGetLastError(),
                   "self_pre_vf_mesh_mesh_contacts_kernel");
    }

    if (run_self_prepass_ && topo.n_adj_ee_pre() > 0) {
        int grid = (topo.n_adj_ee_pre() + kContactBlock - 1) /
                   kContactBlock;
        self_pre_ee_mesh_mesh_contacts_kernel<<<
            grid, kContactBlock, 0, stream>>>(
            topo.pre_adj_ee().gpu_data(),
            topo.n_adj_ee_pre(),
            positions_dev,
            topo.edges().gpu_data(),
            vertex_mesh_ids_.gpu_data(),
            thickness,
            contacts_.gpu_data(),
            count_.gpu_data(),
            max_contacts_);
        check_cuda(cudaGetLastError(),
                   "self_pre_ee_mesh_mesh_contacts_kernel");
    }

    if (ef_count_dev != nullptr) {
        check_cuda(cudaMemcpyAsync(
                       ef_count_snapshot_.cpu_data(),
                       ef_count_dev,
                       sizeof(int),
                       cudaMemcpyDeviceToHost,
                       stream),
                   "MeshMeshContactDetector::EF count snapshot");
    }
    if (dropped_hits_dev != nullptr) {
        check_cuda(cudaMemcpyAsync(
                       broadphase_dropped_hits_snapshot_.cpu_data(),
                       dropped_hits_dev,
                       sizeof(int),
                       cudaMemcpyDeviceToHost,
                       stream),
                   "MeshMeshContactDetector::dropped hit snapshot");
    } else {
        broadphase_dropped_hits_snapshot_.cpu_data()[0] = 0;
    }
}

int MeshMeshContactDetector::last_ef_count() const noexcept {
    if (ef_count_snapshot_.cpu_size() == 0) return 0;
    return std::min(ef_count_snapshot_.cpu_data()[0], max_ef_candidates_);
}

int MeshMeshContactDetector::last_broadphase_dropped_hits() const noexcept {
    if (broadphase_dropped_hits_snapshot_.cpu_size() == 0) return 0;
    return std::max(0, broadphase_dropped_hits_snapshot_.cpu_data()[0]);
}

int MeshMeshContactDetector::count(std::uintptr_t cuda_stream) {
    int host_count = 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    check_cuda(cudaMemcpyAsync(&host_count,
                               count_.gpu_data(),
                               sizeof(int),
                               cudaMemcpyDeviceToHost,
                               stream),
               "MeshMeshContactDetector::count copy");
    check_cuda(cudaStreamSynchronize(stream),
               "MeshMeshContactDetector::count sync");
    return std::min(host_count, max_contacts_);
}

void MeshMeshContactDetector::download(
    std::vector<MeshMeshContact>& out,
    std::uintptr_t cuda_stream)
{
    int n = count(cuda_stream);
    out.resize(static_cast<std::size_t>(n));
    if (n == 0) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    check_cuda(cudaMemcpyAsync(out.data(),
                               contacts_.gpu_data(),
                               static_cast<std::size_t>(n) * sizeof(MeshMeshContact),
                               cudaMemcpyDeviceToHost,
                               stream),
               "MeshMeshContactDetector::download copy");
    check_cuda(cudaStreamSynchronize(stream),
               "MeshMeshContactDetector::download sync");
}

}  // namespace collision
}  // namespace chysx
