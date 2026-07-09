// SPDX-License-Identifier: Apache-2.0

#include "mesh_mesh_contact.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

namespace chysx {
namespace collision {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(err));
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
    int n_ef,
    const math::Vec3f* __restrict__ verts,
    const math::Vec3i* __restrict__ faces,
    const math::Vec2i* __restrict__ edges,
    const int* __restrict__ vert_in_edge,
    const math::Vec3i* __restrict__ edge_in_face,
    const int* __restrict__ vertex_mesh_ids,
    float thickness,
    MeshMeshContact* contacts,
    int* contact_count,
    int max_contacts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_ef) return;

    const float thickness_sq = thickness * thickness;
    const math::Vec2i ef = ef_pairs[i];
    const int eid = ef.x;
    const int fid = ef.y;

    const int vid = vert_in_edge[eid];
    if (vid >= 0) {
        math::Vec3i f = faces[fid];
        if (vid != f.x && vid != f.y && vid != f.z) {
            int mesh_point = -1;
            int mesh_face = -1;
            if (!same_mesh_pf(vid, f, vertex_mesh_ids, mesh_point, mesh_face)) {
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
                    MeshMeshContact contact;
                    contact.type = static_cast<int>(MeshMeshContactType::PointFace);
                    contact.vertices = math::Vec4i(vid, f.x, f.y, f.z);
                    contact.weights = math::Vec4f(1.0f, -bary.x, -bary.y, -bary.z);
                    contact.normal = diff * (1.0f / d);
                    contact.distance = d;
                    contact.mesh_pair = math::Vec2i(mesh_point, mesh_face);
                    contact.source_ef = ef;
                    emit_contact(contacts, contact_count, max_contacts, contact);
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
        if (same_mesh_ee(ea, eb, vertex_mesh_ids, mesh_a, mesh_b)) continue;

        math::Vec3f a0 = verts[ea.x];
        math::Vec3f a1 = verts[ea.y];
        math::Vec3f b0 = verts[eb.x];
        math::Vec3f b1 = verts[eb.y];
        float s = 0.0f;
        float t = 0.0f;
        math::Vec3f cp;
        math::Vec3f cq;
        if (!closest_segment_segment(a0, a1, b0, b1, s, t, cp, cq)) continue;

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
            MeshMeshContact contact;
            contact.type = static_cast<int>(MeshMeshContactType::EdgeEdge);
            contact.vertices = math::Vec4i(ea.x, ea.y, eb.x, eb.y);
            contact.weights = math::Vec4f(1.0f - s, s, -(1.0f - t), -t);
            contact.normal = diff * (1.0f / d);
            contact.distance = d;
            contact.mesh_pair = math::Vec2i(mesh_a, mesh_b);
            contact.source_ef = ef;
            emit_contact(contacts, contact_count, max_contacts, contact);
        }
    }
}

__global__ void adjacent_vf_mesh_mesh_contacts_kernel(
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
    if (same_mesh_pf(vid, f, vertex_mesh_ids, mesh_point, mesh_face)) return;

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
        emit_contact(contacts, contact_count, max_contacts, contact);
    }
}

__global__ void adjacent_ee_mesh_mesh_contacts_kernel(
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
    if (same_mesh_ee(ea, eb, vertex_mesh_ids, mesh_a, mesh_b)) return;

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
        emit_contact(contacts, contact_count, max_contacts, contact);
    }
}

}  // namespace

void MeshMeshContactDetector::setup(
    const std::vector<math::Vec3i>& triangles,
    const std::vector<int>& vertex_mesh_ids,
    int max_contacts,
    int max_ef_candidates)
{
    n_verts_ = static_cast<int>(vertex_mesh_ids.size());
    if (n_verts_ <= 0 || triangles.empty()) {
        throw std::runtime_error("MeshMeshContactDetector::setup needs vertices and triangles");
    }

    broadphase_.setup(triangles, n_verts_, max_ef_candidates);

    if (max_contacts <= 0) {
        max_contacts = std::max(1024, 8 * broadphase_.topology().n_edges());
    }
    max_contacts_ = max_contacts;

    vertex_mesh_ids_.resize(vertex_mesh_ids.size());
    std::memcpy(vertex_mesh_ids_.cpu_data(),
                vertex_mesh_ids.data(),
                vertex_mesh_ids.size() * sizeof(int));
    vertex_mesh_ids_.copy_to_device();

    contacts_.resize(static_cast<std::size_t>(max_contacts_));
    count_.resize(1);
    upload_positions_.resize(static_cast<std::size_t>(n_verts_));
    count_.cpu_data()[0] = 0;
    count_.copy_to_device();
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
    last_ef_count_ = 0;
    if (!valid() || positions_dev == nullptr || thickness <= 0.0f) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    zero_count_kernel<<<1, 1, 0, stream>>>(count_.gpu_data());
    check_cuda(cudaGetLastError(), "mesh_mesh zero_count_kernel");

    broadphase_.query(positions_dev, thickness, cuda_stream);
    last_ef_count_ = broadphase_.ef_count(cuda_stream);

    const auto& topo = broadphase_.topology();
    const int block = 256;
    if (last_ef_count_ > 0) {
        int grid = (last_ef_count_ + block - 1) / block;
        ef_to_mesh_mesh_contacts_kernel<<<grid, block, 0, stream>>>(
            broadphase_.ef_pairs_dev(),
            last_ef_count_,
            positions_dev,
            topo.faces().gpu_data(),
            topo.edges().gpu_data(),
            topo.vert_in_edge().gpu_data(),
            topo.edge_in_face().gpu_data(),
            vertex_mesh_ids_.gpu_data(),
            thickness,
            contacts_.gpu_data(),
            count_.gpu_data(),
            max_contacts_);
        check_cuda(cudaGetLastError(), "ef_to_mesh_mesh_contacts_kernel");
    }

    if (topo.n_adj_vf() > 0) {
        int grid = (topo.n_adj_vf() + block - 1) / block;
        adjacent_vf_mesh_mesh_contacts_kernel<<<grid, block, 0, stream>>>(
            topo.pre_adj_vf().gpu_data(),
            topo.n_adj_vf(),
            positions_dev,
            vertex_mesh_ids_.gpu_data(),
            thickness,
            contacts_.gpu_data(),
            count_.gpu_data(),
            max_contacts_);
        check_cuda(cudaGetLastError(), "adjacent_vf_mesh_mesh_contacts_kernel");
    }

    if (topo.n_adj_ee_pre() > 0) {
        int grid = (topo.n_adj_ee_pre() + block - 1) / block;
        adjacent_ee_mesh_mesh_contacts_kernel<<<grid, block, 0, stream>>>(
            topo.pre_adj_ee().gpu_data(),
            topo.n_adj_ee_pre(),
            positions_dev,
            topo.edges().gpu_data(),
            vertex_mesh_ids_.gpu_data(),
            thickness,
            contacts_.gpu_data(),
            count_.gpu_data(),
            max_contacts_);
        check_cuda(cudaGetLastError(), "adjacent_ee_mesh_mesh_contacts_kernel");
    }
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
