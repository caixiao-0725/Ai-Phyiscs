// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "collision/contact_spmv.h"
#include "collision/mesh_mesh_contact.h"
#include "memory/cuda_array.h"
#include "rigid/pabd_cuda/pabd_contact_measure.cuh"
#include "rigid/pabd_cuda/pabd_cuda_solver.h"

namespace {

using chysx::collision::MeshMeshContact;
using chysx::collision::MeshMeshContactDetector;
using chysx::collision::MeshMeshContactType;
using chysx::collision::BroadphaseBackend;
using chysx::collision::MeshCollisionCategory;
using chysx::collision::MeshCollisionMask;
using chysx::collision::kMeshCollisionInterObject;
using chysx::collision::kMeshCollisionSelf;
using chysx::collision::ContactSpMVOp;
using chysx::collision::ContactWeights;
using chysx::collision::WideContact;
using chysx::collision::WideContactSpMVOp;
using chysx::math::Mat3f;
using chysx::math::Vec3f;
using chysx::math::Vec3i;
using chysx::math::Vec4f;
using chysx::math::Vec4i;

struct TestMesh {
    std::vector<Vec3f> positions;
    std::vector<Vec3i> triangles;
    std::vector<int> mesh_ids;
};

void append_outward_tetra(TestMesh& mesh,
                          int v0,
                          int v1,
                          int v2,
                          int v3) {
    int faces[4][4] = {
        {v1, v2, v3, v0},
        {v0, v3, v2, v1},
        {v0, v1, v3, v2},
        {v0, v2, v1, v3},
    };
    for (auto& face : faces) {
        const Vec3f a = mesh.positions[face[0]];
        const Vec3f b = mesh.positions[face[1]];
        const Vec3f c = mesh.positions[face[2]];
        const Vec3f toward_opposite = mesh.positions[face[3]] - a;
        if (chysx::math::dot(chysx::math::cross(b - a, c - a),
                             toward_opposite) > 0.0f) {
            std::swap(face[1], face[2]);
        }
        mesh.triangles.emplace_back(face[0], face[1], face[2]);
    }
}

void require(bool cond, const std::string& message) {
    if (!cond) {
        throw std::runtime_error(message);
    }
}

void require_near(float actual,
                  float expected,
                  float tolerance,
                  const std::string& message) {
    require(std::abs(actual - expected) <= tolerance,
            message + ": expected " + std::to_string(expected) +
                ", got " + std::to_string(actual));
}

int count_type(const std::vector<MeshMeshContact>& contacts,
               MeshMeshContactType type) {
    int n = 0;
    const int wanted = static_cast<int>(type);
    for (const auto& c : contacts) {
        if (c.type == wanted) ++n;
    }
    return n;
}

std::vector<MeshMeshContact> detect_contacts(const TestMesh& mesh,
                                             float thickness,
                                             int max_contacts = 256,
                                             MeshCollisionMask collision_mask =
                                                 kMeshCollisionInterObject,
                                             BroadphaseBackend backend =
                                                 BroadphaseBackend::QuantBvh) {
    MeshMeshContactDetector detector;
    detector.setup(mesh.triangles, mesh.mesh_ids, max_contacts, -1,
                   backend, collision_mask);
    detector.detect(mesh.positions.data(),
                    static_cast<int>(mesh.positions.size()),
                    thickness);

    std::vector<MeshMeshContact> contacts;
    detector.download(contacts);

    std::cout << "  ef=" << detector.last_ef_count()
              << " contacts=" << contacts.size()
              << " pf=" << count_type(contacts, MeshMeshContactType::PointFace)
              << " ee=" << count_type(contacts, MeshMeshContactType::EdgeEdge)
              << "\n";
    return contacts;
}

void require_category(const std::vector<MeshMeshContact>& contacts,
                      MeshCollisionCategory category,
                      const std::string& message) {
    const int wanted = static_cast<int>(category);
    for (const auto& c : contacts) {
        require(c.category == wanted, message);
    }
}

TestMesh point_face_case(float height, bool same_mesh) {
    TestMesh m;
    m.positions = {
        Vec3f(0.00f, 0.25f, height),
        Vec3f(0.25f, 0.25f, 0.30f),
        Vec3f(-0.25f, 0.25f, 0.30f),
        Vec3f(-1.00f, -1.00f, 0.00f),
        Vec3f(1.00f, -1.00f, 0.00f),
        Vec3f(0.00f, 1.00f, 0.00f),
    };
    m.triangles = {
        Vec3i(0, 1, 2),
        Vec3i(3, 4, 5),
    };
    m.mesh_ids = same_mesh
        ? std::vector<int>{0, 0, 0, 0, 0, 0}
        : std::vector<int>{0, 0, 0, 1, 1, 1};
    return m;
}

TestMesh edge_edge_case(bool parallel) {
    TestMesh m;
    if (!parallel) {
        m.positions = {
            Vec3f(-1.00f, 0.00f, 0.00f),
            Vec3f(1.00f, 0.00f, 0.00f),
            Vec3f(-1.00f, -0.25f, 0.40f),
            Vec3f(0.00f, -1.00f, 0.03f),
            Vec3f(0.00f, 1.00f, 0.03f),
            Vec3f(0.25f, -1.00f, 0.40f),
        };
    } else {
        m.positions = {
            Vec3f(-1.00f, 0.00f, 0.00f),
            Vec3f(1.00f, 0.00f, 0.00f),
            Vec3f(-1.00f, -0.25f, 0.40f),
            Vec3f(-1.00f, 0.05f, 0.03f),
            Vec3f(1.00f, 0.05f, 0.03f),
            Vec3f(-1.00f, 0.30f, 0.40f),
        };
    }
    m.triangles = {
        Vec3i(0, 1, 2),
        Vec3i(3, 4, 5),
    };
    m.mesh_ids = {0, 0, 0, 1, 1, 1};
    return m;
}

TestMesh self_pre_point_face_case() {
    TestMesh m;
    m.positions = {
        Vec3f(-1.00f, -1.00f, 0.00f),
        Vec3f(1.00f, -1.00f, 0.00f),
        Vec3f(0.00f, 1.00f, 0.00f),
        Vec3f(0.00f, 0.00f, 0.04f),
    };
    m.triangles = {
        Vec3i(0, 1, 2),
        Vec3i(2, 1, 3),
    };
    m.mesh_ids = {0, 0, 0, 0};
    return m;
}

TestMesh self_pre_edge_edge_case() {
    TestMesh m;
    m.positions = {
        Vec3f(-1.00f, -1.00f, 0.00f),
        Vec3f(1.00f, -1.00f, 0.00f),
        Vec3f(0.00f, 1.00f, 0.04f),
        Vec3f(0.00f, -2.00f, 0.04f),
    };
    m.triangles = {
        Vec3i(0, 1, 2),
        Vec3i(2, 1, 3),
    };
    m.mesh_ids = {0, 0, 0, 0};
    return m;
}

TestMesh oriented_point_face_case(float point_z) {
    TestMesh mesh;
    mesh.positions = {
        Vec3f(0.25f, 0.25f, point_z),
        Vec3f(0.60f, 0.25f, 0.30f),
        Vec3f(0.25f, 0.60f, 0.30f),
        Vec3f(0.00f, 0.00f, 0.00f),
        Vec3f(1.00f, 0.00f, 0.00f),
        Vec3f(0.00f, 1.00f, 0.00f),
        Vec3f(0.00f, 0.00f, -1.00f),
    };
    mesh.triangles.emplace_back(0, 1, 2);
    append_outward_tetra(mesh, 3, 4, 5, 6);
    mesh.mesh_ids = {0, 0, 0, 1, 1, 1, 1};
    return mesh;
}

TestMesh oriented_edge_edge_case(float upper_edge_z) {
    TestMesh mesh;
    mesh.positions = {
        Vec3f(-1.00f, 0.00f, 0.00f),
        Vec3f(1.00f, 0.00f, 0.00f),
        Vec3f(0.00f, -1.00f, -1.00f),
        Vec3f(0.00f, 1.00f, -1.00f),
        Vec3f(0.00f, -1.00f, upper_edge_z),
        Vec3f(0.00f, 1.00f, upper_edge_z),
        Vec3f(-1.00f, 0.00f, 1.04f),
        Vec3f(1.00f, 0.00f, 1.04f),
    };
    append_outward_tetra(mesh, 0, 1, 2, 3);
    append_outward_tetra(mesh, 4, 5, 6, 7);
    mesh.mesh_ids = {0, 0, 0, 0, 1, 1, 1, 1};
    return mesh;
}

void run_point_face_hits() {
    std::cout << "[mesh_mesh] point-face hit\n";
    auto contacts = detect_contacts(point_face_case(0.04f, false), 0.10f);
    require(count_type(contacts, MeshMeshContactType::PointFace) > 0,
            "expected at least one point-face contact");
    require_category(contacts, MeshCollisionCategory::InterObject,
                     "different-mesh contact must be InterObject");
}

void run_point_face_far() {
    std::cout << "[mesh_mesh] point-face outside thickness\n";
    auto contacts = detect_contacts(point_face_case(0.18f, false), 0.08f);
    require(count_type(contacts, MeshMeshContactType::PointFace) == 0,
            "far point-face case should not emit point-face contacts");
}

void run_same_mesh_filtered() {
    std::cout << "[mesh_mesh] same-mesh filter\n";
    auto contacts = detect_contacts(point_face_case(0.04f, true), 0.10f);
    require(contacts.empty(), "same-mesh primitives should not collide");
}

void run_same_mesh_self_collision() {
    std::cout << "[mesh_mesh] same-mesh self collision\n";
    auto contacts = detect_contacts(
        point_face_case(0.04f, true), 0.10f, 256,
        kMeshCollisionSelf);
    require(count_type(contacts, MeshMeshContactType::PointFace) > 0,
            "self-collision mode should emit same-mesh point-face contacts");
    require_category(contacts, MeshCollisionCategory::Self,
                     "same-mesh contact must be classified as Self");
}

void run_self_prepass_only_for_self() {
    std::cout << "[mesh_mesh] self-only topology pre-pass\n";
    const TestMesh mesh = self_pre_point_face_case();

    auto inter_contacts = detect_contacts(
        mesh, 0.10f, 256, kMeshCollisionInterObject);
    require(inter_contacts.empty(),
            "inter-object mode must not run the topology pre-pass");

    auto self_contacts = detect_contacts(
        mesh, 0.10f, 256, kMeshCollisionSelf);
    bool found_pre_vf = false;
    for (const auto& c : self_contacts) {
        if (c.type == static_cast<int>(MeshMeshContactType::PointFace) &&
            c.source_ef.x == -1 && c.source_ef.y == -1) {
            found_pre_vf = true;
        }
    }
    require(found_pre_vf,
            "self mode should emit the precomputed adjacent VF contact");
    require_category(self_contacts, MeshCollisionCategory::Self,
                     "self pre-pass contact must be classified as Self");

    auto self_ee_contacts = detect_contacts(
        self_pre_edge_edge_case(), 0.10f, 256, kMeshCollisionSelf);
    bool found_pre_ee = false;
    for (const auto& c : self_ee_contacts) {
        if (c.type == static_cast<int>(MeshMeshContactType::EdgeEdge) &&
            c.source_ef.x == -1 && c.source_ef.y == -1) {
            found_pre_ee = true;
        }
    }
    require(found_pre_ee,
            "self mode should emit the precomputed adjacent EE contact");
}

void run_fat_broadphase_cache(BroadphaseBackend backend) {
    std::cout << "[mesh_mesh] fat broadphase cache backend="
              << (backend == BroadphaseBackend::OptiX ? "OptiX" : "QuantBvh")
              << "\n";

    TestMesh mesh = point_face_case(0.14f, false);
    MeshMeshContactDetector detector;
    detector.setup(mesh.triangles, mesh.mesh_ids, 256, -1, backend,
                   kMeshCollisionInterObject);
    detector.configure_broadphase_cache(8, 0.24f);

    detector.detect(mesh.positions.data(),
                    static_cast<int>(mesh.positions.size()), 0.05f);
    std::vector<MeshMeshContact> contacts;
    detector.download(contacts);
    require(detector.last_broadphase_refreshed(),
            "first cached detect must refresh the broadphase");
    require(contacts.empty(),
            "reference geometry should be outside narrow thickness");

    mesh.positions[0].z = 0.04f;
    detector.detect(mesh.positions.data(),
                    static_cast<int>(mesh.positions.size()), 0.05f);
    detector.download(contacts);
    require(!detector.last_broadphase_refreshed(),
            "motion within skin/2 should reuse broadphase candidates");
    require(detector.broadphase_cache_age() == 1,
            "cached frame should have broadphase age one");
    require(count_type(contacts, MeshMeshContactType::PointFace) > 0,
            "fat cached candidates should find the new narrow contact");
    require(detector.last_broadphase_dropped_hits() == 0,
            "small cache fixture must not overflow per-edge hits");

    detector.detect(mesh.positions.data(),
                    static_cast<int>(mesh.positions.size()), 0.06f);
    detector.download(contacts);
    require(detector.last_broadphase_refreshed(),
            "changing narrow thickness must invalidate cached candidates");

    detector.configure_broadphase_cache(8, 0.20f);
    mesh = point_face_case(0.32f, false);
    detector.detect(mesh.positions.data(),
                    static_cast<int>(mesh.positions.size()), 0.05f);
    detector.download(contacts);
    mesh.positions[0].z = 0.04f;
    detector.detect(mesh.positions.data(),
                    static_cast<int>(mesh.positions.size()), 0.05f);
    detector.download(contacts);
    require(detector.last_broadphase_refreshed(),
            "motion beyond skin/2 must refresh before narrow phase");
    require(count_type(contacts, MeshMeshContactType::PointFace) > 0,
            "early refresh should recover the new narrow contact");
}

#ifdef CHYSX_HAS_OPTIX
void run_optix_collision_categories() {
    std::cout << "[mesh_mesh] OptiX collision categories\n";

    auto inter_contacts = detect_contacts(
        point_face_case(0.04f, false), 0.10f, 256,
        kMeshCollisionInterObject, BroadphaseBackend::OptiX);
    require(count_type(inter_contacts, MeshMeshContactType::PointFace) > 0,
            "OptiX should emit an inter-object point-face contact");
    require_category(inter_contacts, MeshCollisionCategory::InterObject,
                     "OptiX different-mesh contact must be InterObject");

    auto filtered_contacts = detect_contacts(
        point_face_case(0.04f, true), 0.10f, 256,
        kMeshCollisionInterObject, BroadphaseBackend::OptiX);
    require(filtered_contacts.empty(),
            "OptiX inter-object mode must reject same-mesh contacts");

    auto self_contacts = detect_contacts(
        point_face_case(0.04f, true), 0.10f, 256,
        kMeshCollisionSelf, BroadphaseBackend::OptiX);
    require(count_type(self_contacts, MeshMeshContactType::PointFace) > 0,
            "OptiX self mode should emit a same-mesh point-face contact");
    require_category(self_contacts, MeshCollisionCategory::Self,
                     "OptiX same-mesh contact must be Self");
}
#endif

void run_edge_edge_hits() {
    std::cout << "[mesh_mesh] edge-edge hit\n";
    auto contacts = detect_contacts(edge_edge_case(false), 0.08f);
    require(count_type(contacts, MeshMeshContactType::EdgeEdge) > 0,
            "expected at least one edge-edge contact");
}

void run_parallel_edge_edge_discarded() {
    std::cout << "[mesh_mesh] parallel edge-edge discarded\n";
    auto contacts = detect_contacts(edge_edge_case(true), 0.10f);
    require(count_type(contacts, MeshMeshContactType::EdgeEdge) == 0,
            "parallel edge-edge contact should be discarded");
}

bool same_unordered_pair(int a, int b, int x, int y) {
    return (a == x && b == y) || (a == y && b == x);
}

const MeshMeshContact* find_oriented_pf(
    const std::vector<MeshMeshContact>& contacts) {
    for (const auto& contact : contacts) {
        if (contact.type !=
                static_cast<int>(MeshMeshContactType::PointFace) ||
            contact.vertices.x != 0) {
            continue;
        }
        int face[3] = {
            contact.vertices.y, contact.vertices.z, contact.vertices.w};
        std::sort(face, face + 3);
        if (face[0] == 3 && face[1] == 4 && face[2] == 5) {
            return &contact;
        }
    }
    return nullptr;
}

const MeshMeshContact* find_oriented_ee(
    const std::vector<MeshMeshContact>& contacts) {
    for (const auto& contact : contacts) {
        if (contact.type !=
            static_cast<int>(MeshMeshContactType::EdgeEdge)) {
            continue;
        }
        const bool target =
            (same_unordered_pair(contact.vertices.x, contact.vertices.y,
                                 0, 1) &&
             same_unordered_pair(contact.vertices.z, contact.vertices.w,
                                 4, 5)) ||
            (same_unordered_pair(contact.vertices.x, contact.vertices.y,
                                 4, 5) &&
             same_unordered_pair(contact.vertices.z, contact.vertices.w,
                                 0, 1));
        if (target) return &contact;
    }
    return nullptr;
}

void run_oriented_point_face_crossing(BroadphaseBackend backend) {
    std::cout << "[mesh_mesh] oriented point-face crossing backend="
              << (backend == BroadphaseBackend::OptiX ? "OptiX" : "QuantBvh")
              << "\n";
    TestMesh mesh = oriented_point_face_case(0.04f);
    const std::vector<Vec3f> reference_positions = mesh.positions;
    MeshMeshContactDetector detector;
    detector.setup(mesh.triangles, mesh.mesh_ids, 256, -1, backend,
                   kMeshCollisionInterObject, &reference_positions);

    std::vector<MeshMeshContact> contacts;
    detector.detect(mesh.positions.data(),
                    static_cast<int>(mesh.positions.size()), 0.08f);
    detector.download(contacts);
    const MeshMeshContact* outside = find_oriented_pf(contacts);
    require(outside != nullptr, "expected oriented PF contact outside face");
    require(outside->normal.z > 0.9f && outside->separation > 0.0f,
            "outside PF must use outward face normal and positive separation");
    const Vec3f outside_normal = outside->normal;

    mesh.positions[0].z = -0.04f;
    detector.detect(mesh.positions.data(),
                    static_cast<int>(mesh.positions.size()), 0.08f);
    detector.download(contacts);
    const MeshMeshContact* inside = find_oriented_pf(contacts);
    require(inside != nullptr, "expected oriented PF contact inside face");
    require(chysx::math::dot(outside_normal, inside->normal) > 0.99f,
            "PF normal must not flip after crossing the face");
    require(inside->separation < 0.0f,
            "crossed PF contact must report negative signed separation");
}

void run_oriented_edge_edge_crossing(BroadphaseBackend backend) {
    std::cout << "[mesh_mesh] oriented edge-edge crossing backend="
              << (backend == BroadphaseBackend::OptiX ? "OptiX" : "QuantBvh")
              << "\n";
    TestMesh mesh = oriented_edge_edge_case(0.04f);
    const std::vector<Vec3f> reference_positions = mesh.positions;
    MeshMeshContactDetector detector;
    detector.setup(mesh.triangles, mesh.mesh_ids, 256, -1, backend,
                   kMeshCollisionInterObject, &reference_positions);

    std::vector<MeshMeshContact> contacts;
    detector.detect(mesh.positions.data(),
                    static_cast<int>(mesh.positions.size()), 0.08f);
    detector.download(contacts);
    const MeshMeshContact* outside = find_oriented_ee(contacts);
    require(outside != nullptr, "expected oriented EE contact before crossing");
    require(outside->separation > 0.0f,
            "separated EE contact must have positive separation");
    const Vec3f outside_normal = outside->normal;

    mesh.positions[4].z = -0.04f;
    mesh.positions[5].z = -0.04f;
    detector.detect(mesh.positions.data(),
                    static_cast<int>(mesh.positions.size()), 0.08f);
    detector.download(contacts);
    const MeshMeshContact* inside = find_oriented_ee(contacts);
    require(inside != nullptr, "expected oriented EE contact after crossing");
    require(chysx::math::dot(outside_normal, inside->normal) > 0.99f,
            "EE normal must not flip after the edges cross");
    require(inside->separation < 0.0f,
            "crossed EE contact must report negative signed separation");
}

void run_wide_contact_friction_spmv() {
    std::cout << "[contact_spmv] wide-contact tangent block\n";
    chysx::CudaArray<WideContact> contacts(1);
    chysx::CudaArray<int> count(1);
    chysx::CudaArray<Mat3f> diag(8);
    chysx::CudaArray<Vec3f> x(8);
    chysx::CudaArray<Vec3f> y(8);

    WideContact contact{};
    contact.weights0 = Vec4f(1.0f, 0.0f, 0.0f, 0.0f);
    contact.weights1 = Vec4f(-1.0f, 0.0f, 0.0f, 0.0f);
    contact.normal_target = Vec4f(0.0f, 1.0f, 0.0f, 0.01f);
    contact.friction_target_alpha = Vec4f(0.0f, 0.0f, 0.0f, 3.0f);
    contact.base0 = 0;
    contact.base1 = 4;
    contact.stiffness = 10.0f;
    contacts.cpu_data()[0] = contact;
    count.cpu_data()[0] = 1;
    for (int i = 0; i < 8; ++i) {
        diag.cpu_data()[i] = Mat3f{};
        x.cpu_data()[i] = Vec3f(0.0f, 0.0f, 0.0f);
        y.cpu_data()[i] = Vec3f(0.0f, 0.0f, 0.0f);
    }
    x.cpu_data()[0] = Vec3f(1.0f, 2.0f, 3.0f);
    x.cpu_data()[4] = Vec3f(4.0f, 6.0f, 8.0f);
    contacts.copy_to_device();
    count.copy_to_device();
    diag.copy_to_device();
    x.copy_to_device();
    y.copy_to_device();

    const WideContactSpMVOp op{
        contacts.gpu_data(), count.gpu_data(), 1, 1.0f};
    chysx::collision::bake_wide_contact_diag(
        diag.gpu_data(), 8, op, 1.0f, 0);
    chysx::collision::apply_wide_contact_spmv(
        op, x.gpu_data(), y.gpu_data(), 8, 1.0f, 0);
    diag.copy_to_host();
    y.copy_to_host();

    for (int id : {0, 4}) {
        const Mat3f& h = diag.cpu_data()[id];
        require_near(h.data[0], 3.0f, 1.0e-5f,
                     "wide-contact tangent xx diagonal");
        require_near(h.data[4], 10.0f, 1.0e-5f,
                     "wide-contact normal yy diagonal");
        require_near(h.data[8], 3.0f, 1.0e-5f,
                     "wide-contact tangent zz diagonal");
        require_near(h.data[1], 0.0f, 1.0e-5f,
                     "wide-contact xy coupling");
    }
    require_near(y.cpu_data()[0].x, -12.0f, 1.0e-5f,
                 "wide-contact offdiag y0.x");
    require_near(y.cpu_data()[0].y, -60.0f, 1.0e-5f,
                 "wide-contact offdiag y0.y");
    require_near(y.cpu_data()[0].z, -24.0f, 1.0e-5f,
                 "wide-contact offdiag y0.z");
    require_near(y.cpu_data()[4].x, -3.0f, 1.0e-5f,
                 "wide-contact offdiag y4.x");
    require_near(y.cpu_data()[4].y, -20.0f, 1.0e-5f,
                 "wide-contact offdiag y4.y");
    require_near(y.cpu_data()[4].z, -9.0f, 1.0e-5f,
                 "wide-contact offdiag y4.z");
}

void run_per_contact_stiffness_spmv() {
    std::cout << "[contact_spmv] per-contact stiffness\n";
    chysx::CudaArray<Vec4i> pairs(1);
    chysx::CudaArray<ContactWeights> weights(1);
    chysx::CudaArray<float> stiffnesses(1);
    chysx::CudaArray<int> count(1);
    chysx::CudaArray<Mat3f> diag(4);
    chysx::CudaArray<Vec3f> x(4);
    chysx::CudaArray<Vec3f> y(4);

    pairs.cpu_data()[0] = Vec4i(0, 1, 2, 3);
    ContactWeights contact{};
    contact.w0 = 1.0f;
    contact.w1 = -1.0f;
    contact.ny = 1.0f;
    weights.cpu_data()[0] = contact;
    stiffnesses.cpu_data()[0] = 7.0f;
    count.cpu_data()[0] = 1;
    for (int i = 0; i < 4; ++i) {
        diag.cpu_data()[i] = Mat3f{};
        x.cpu_data()[i] = Vec3f(0.0f);
        y.cpu_data()[i] = Vec3f(0.0f);
    }
    x.cpu_data()[0].y = 1.0f;
    x.cpu_data()[1].y = 3.0f;

    pairs.copy_to_device();
    weights.copy_to_device();
    stiffnesses.copy_to_device();
    count.copy_to_device();
    diag.copy_to_device();
    x.copy_to_device();
    y.copy_to_device();

    ContactSpMVOp op;
    op.pairs = pairs.gpu_data();
    op.weights = weights.gpu_data();
    op.count_dev = count.gpu_data();
    op.max_contacts = 1;
    op.stiffness = 100.0f;
    op.stiffnesses = stiffnesses.gpu_data();
    chysx::collision::bake_contact_diag(
        diag.gpu_data(), 4, op, 1.0f, 0);
    chysx::collision::apply_contact_spmv(
        op, x.gpu_data(), y.gpu_data(), 4, 1.0f, 0);
    diag.copy_to_host();
    y.copy_to_host();

    require_near(diag.cpu_data()[0].data[4], 7.0f, 1.0e-5f,
                 "per-contact diagonal stiffness");
    require_near(diag.cpu_data()[1].data[4], 7.0f, 1.0e-5f,
                 "per-contact second diagonal stiffness");
    require_near(y.cpu_data()[0].y, -21.0f, 1.0e-5f,
                 "per-contact offdiag row zero");
    require_near(y.cpu_data()[1].y, -7.0f, 1.0e-5f,
                 "per-contact offdiag row one");
}

void run_contact_tetrahedron_volume_measure() {
    using chysx::rigid::pabd_cuda::
        contact_tetrahedron_volume_at_activation;
    std::cout << "[pabd_contact_beta] PF/EE activation tetra volume\n";

    const float pf = contact_tetrahedron_volume_at_activation(
        Vec3f(0.25f, 0.25f, 0.2f),
        Vec3f(0.0f, 0.0f, 0.0f),
        Vec3f(2.0f, 0.0f, 0.0f),
        Vec3f(0.0f, 3.0f, 0.0f),
        0.2f, 0.1f);
    require_near(pf, 0.1f, 1.0e-6f,
                 "PF activation tetra volume");

    const float pf_near = contact_tetrahedron_volume_at_activation(
        Vec3f(0.25f, 0.25f, 0.02f),
        Vec3f(0.0f, 0.0f, 0.0f),
        Vec3f(2.0f, 0.0f, 0.0f),
        Vec3f(0.0f, 3.0f, 0.0f),
        0.02f, 0.1f);
    require_near(pf_near, pf, 1.0e-6f,
                 "PF measure must not vanish as contact closes");

    const float ee = contact_tetrahedron_volume_at_activation(
        Vec3f(0.0f, 0.0f, 0.0f),
        Vec3f(2.0f, 0.0f, 0.0f),
        Vec3f(0.5f, -1.0f, 0.2f),
        Vec3f(0.5f, 2.0f, 0.2f),
        0.2f, 0.1f);
    require_near(ee, 0.1f, 1.0e-6f,
                 "EE activation tetra volume");

    const float parallel_ee = contact_tetrahedron_volume_at_activation(
        Vec3f(0.0f, 0.0f, 0.0f),
        Vec3f(2.0f, 0.0f, 0.0f),
        Vec3f(0.0f, 0.2f, 0.1f),
        Vec3f(2.0f, 0.2f, 0.1f),
        0.1f, 0.1f);
    require_near(parallel_ee, 0.0f, 1.0e-7f,
                 "parallel EE tetra volume");

    const float large_pf = contact_tetrahedron_volume_at_activation(
        Vec3f(0.5f, 0.5f, 0.2f),
        Vec3f(0.0f, 0.0f, 0.0f),
        Vec3f(4.0f, 0.0f, 0.0f),
        Vec3f(0.0f, 6.0f, 0.0f),
        0.2f, 0.1f);
    require_near(large_pf, 4.0f * pf, 1.0e-6f,
                 "contact measure must scale with stencil volume");

    const float scaled_pf = contact_tetrahedron_volume_at_activation(
        Vec3f(0.5f, 0.5f, 0.4f),
        Vec3f(0.0f, 0.0f, 0.0f),
        Vec3f(4.0f, 0.0f, 0.0f),
        Vec3f(0.0f, 6.0f, 0.0f),
        0.4f, 0.2f);
    require_near(scaled_pf, 8.0f * pf, 1.0e-5f,
                 "uniformly scaled contact measure must scale cubically");
}

void run_stacked_blocks_uses_pd_abd_box_path() {
    using namespace chysx::rigid::pabd_cuda;
    std::cout << "[pabd_stack] shared OBJ + ABD contact path\n";

    const PabdCudaMesh stacked =
        make_stacked_blocks_mesh(3, 0.42f, 0.0f, 0.02f);
    const PabdCudaMesh boxes =
        make_pd_abd_boxes_mesh(3, 0.0f, 0.02f, 0.42f, 0.0f);
    require(stacked.rest_positions.size() == 12,
            "stacked blocks must have four controls per body");
    require(stacked.tets.size() == 3,
            "stacked blocks must have one ABD control tet per body");
    require(stacked.surface_maps.size() == 24,
            "stacked blocks must use the eight-vertex OBJ surface per body");
    require(stacked.surface_triangles.size() == 36,
            "stacked blocks must use the OBJ triangle topology");
    require(stacked.surface_edges.size() == boxes.surface_edges.size(),
            "stacked blocks and PD+ABD boxes must share edge topology");
    require(stacked.rest_positions.size() == boxes.rest_positions.size() &&
                stacked.surface_maps.size() == boxes.surface_maps.size() &&
                stacked.surface_triangles.size() ==
                    boxes.surface_triangles.size(),
            "stacked blocks and PD+ABD boxes must share mesh layout");
    for (std::size_t i = 0; i < stacked.rest_positions.size(); ++i) {
        require_near(stacked.rest_positions[i].x, boxes.rest_positions[i].x,
                     1.0e-7f, "shared stack control x");
        require_near(stacked.rest_positions[i].y, boxes.rest_positions[i].y,
                     1.0e-7f, "shared stack control y");
        require_near(stacked.rest_positions[i].z, boxes.rest_positions[i].z,
                     1.0e-7f, "shared stack control z");
    }
}

void run_ground_contact_beta_mass_time_normalization(
    chysx::rigid::pabd_cuda::PabdGlobalSolverMode solver_mode) {
    using namespace chysx::rigid::pabd_cuda;
    std::cout << "[pabd_ground_beta] mass/time normalization "
              << (solver_mode == PabdGlobalSolverMode::PCG
                      ? "PCG"
                      : "BlockJacobi12")
              << "\n";

    const auto normalized_ground_velocity =
        [&](float density, float body_mass_scale, float dt) {
            PabdCudaMesh mesh;
            mesh.rest_positions = {
                Vec3f(0.0f, 0.0f, 0.0f),
                Vec3f(1.0f, 0.0f, 0.0f),
                Vec3f(0.0f, 1.0f, 0.0f),
                Vec3f(0.0f, 0.0f, 1.0f),
            };
            mesh.initial_velocities = {
                Vec3f(0.0f, -1.0f, 0.0f),
                Vec3f(0.0f), Vec3f(0.0f), Vec3f(0.0f),
            };
            mesh.fixed.assign(4, 0);
            mesh.tets.push_back({0, 1, 2, 3});
            mesh.tet_volume_overrides.push_back(body_mass_scale);
            std::array<float, 16> mass{};
            for (int i = 0; i < 4; ++i) {
                mass[i * 4 + i] = body_mass_scale * 0.25f;
            }
            mesh.tet_mass_blocks.push_back(mass);
            for (int vertex = 0; vertex < 4; ++vertex) {
                PabdSurfaceMap map;
                map.index = {0, 1, 2, 3};
                map.weight = {0.0f, 0.0f, 0.0f, 0.0f};
                map.weight[vertex] = 1.0f;
                map.body = 0;
                map.self_collide = 0;
                map.ground_collide = vertex == 0 ? 1 : 0;
                mesh.surface_maps.push_back(map);
            }
            mesh.surface_triangles = {
                {0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
            mesh.surface_edges = {
                {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

            PabdCudaParams params;
            params.dt = dt;
            params.iterations = 1;
            params.gravity = 0.0f;
            params.stiffness = 0.0f;
            params.use_arap_beta = true;
            params.arap_beta = 0.0f;
            params.density = density;
            params.damping = 1.0f;
            params.ground_y = 0.0f;
            params.contact_gap = 0.0f;
            params.use_contact_beta = true;
            params.ground_contact_beta = 4.0f;
            params.ground_stiffness = 0.0f;
            params.self_collision_beta = 0.0f;
            params.self_collision_stiffness = 0.0f;
            params.self_collision_thickness = 0.0f;
            params.self_collision_max_contacts = 8;
            params.pcg_iterations = 50;
            params.global_solver = solver_mode;

            PabdCudaSolver solver;
            solver.setup(mesh, params);
            solver.step(dt);
            return solver.flat_positions()[2] / dt;
        };

    const float baseline =
        normalized_ground_velocity(1.0f, 1.0f, 0.0033f);
    const float heavy =
        normalized_ground_velocity(100.0f, 1.0f, 0.0033f);
    const float large_mass =
        normalized_ground_velocity(1.0f, 10.0f, 0.0033f);
    const float double_dt =
        normalized_ground_velocity(1.0f, 1.0f, 0.0066f);
    require_near(baseline, -0.2f, 2.0e-3f,
                 "contact beta isolated contraction");
    require_near(heavy, baseline, 2.0e-3f,
                 "contact beta response must be density invariant");
    require_near(large_mass, baseline, 2.0e-3f,
                 "ground beta response must be body-mass invariant");
    require_near(double_dt, baseline, 2.0e-3f,
                 "contact beta response must be timestep invariant");
}

void run_endpoint_hinge_motor(
    chysx::rigid::pabd_cuda::PabdGlobalSolverMode solver_mode) {
    using namespace chysx::rigid::pabd_cuda;
    std::cout << "[pabd_hinge] endpoint motor "
              << (solver_mode == PabdGlobalSolverMode::PCG
                      ? "PCG"
                      : "BlockJacobi12")
              << "\n";

    PabdCudaMesh mesh;
    mesh.rest_positions = {
        Vec3f(-1.0f, -1.0f, -1.0f),
        Vec3f(3.0f, -1.0f, -1.0f),
        Vec3f(-1.0f, 3.0f, -1.0f),
        Vec3f(-1.0f, -1.0f, 3.0f),
    };
    mesh.initial_velocities.assign(4, Vec3f(0.0f));
    mesh.fixed.assign(4, 0);
    mesh.tets.push_back({0, 1, 2, 3});
    mesh.tet_volume_overrides.push_back(1.0f);
    std::array<float, 16> mass{};
    for (int i = 0; i < 4; ++i) mass[i * 4 + i] = 1.0f;
    mesh.tet_mass_blocks.push_back(mass);

    for (int vertex = 0; vertex < 4; ++vertex) {
        PabdSurfaceMap map;
        map.index = {0, 1, 2, 3};
        map.weight = {0.0f, 0.0f, 0.0f, 0.0f};
        map.weight[vertex] = 1.0f;
        map.body = 0;
        map.ground_collide = 0;
        mesh.surface_maps.push_back(map);
    }
    mesh.surface_triangles = {
        {0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
    mesh.surface_edges = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

    PabdEndpointHinge hinge;
    hinge.body = 0;
    hinge.motor = 1;
    hinge.weights0 = Vec4f(0.375f, 0.25f, 0.25f, 0.125f);
    hinge.weights1 = Vec4f(0.125f, 0.25f, 0.25f, 0.375f);
    hinge.endpoint0 = Vec4f(0.0f, 0.0f, -0.5f, 0.0f);
    hinge.endpoint1 = Vec4f(0.0f, 0.0f, 0.5f, 0.0f);
    hinge.axis = Vec4f(0.0f, 0.0f, 1.0f, 0.0f);
    mesh.endpoint_hinges.push_back(hinge);

    PabdCudaParams params;
    params.dt = 0.0033f;
    params.iterations = 1;
    params.gravity = 0.0f;
    params.stiffness = 1.5e4f;
    params.density = 1.0f;
    params.damping = 1.0f;
    params.hinge_beta = 64.0f;
    params.motor_torque = -1.0e3f;
    params.motor_damping = 1.0f;
    params.ground_stiffness = 0.0f;
    params.self_collision_thickness = 0.0f;
    params.self_collision_stiffness = 0.0f;
    params.self_collision_max_contacts = 8;
    params.pcg_iterations = 50;
    params.global_solver = solver_mode;

    PabdCudaSolver solver;
    solver.set_auto_download_positions(false);
    solver.setup(mesh, params);
    for (int frame = 0; frame < 120; ++frame) solver.step(params.dt);

    require(solver.last_motor_axis_angular_velocity() < -0.05f,
            "endpoint hinge motor must rotate in negative axis direction");
    require(solver.last_hinge_endpoint_error() < 1.0e-3f,
            "endpoint hinge endpoints must stay on the fixed world axis");
    require(solver.last_pcg_body_preconditioner_failures() == 0,
            "endpoint hinge PCG body factorization must not fail");
    require(solver.last_block_jacobi_failures() == 0,
            "endpoint hinge Block-Jacobi factorization must not fail");

    if (solver_mode == PabdGlobalSolverMode::PCG) {
        const Vec3f offset(2.05f, 0.37f, -0.22f);
        for (Vec3f& position : mesh.rest_positions) position += offset;
        PabdEndpointHinge& shifted_hinge = mesh.endpoint_hinges.front();
        shifted_hinge.endpoint0.x += offset.x;
        shifted_hinge.endpoint0.y += offset.y;
        shifted_hinge.endpoint0.z += offset.z;
        shifted_hinge.endpoint1.x += offset.x;
        shifted_hinge.endpoint1.y += offset.y;
        shifted_hinge.endpoint1.z += offset.z;

        params.motor_torque = 0.0f;
        PabdCudaSolver rest_solver;
        rest_solver.set_auto_download_positions(false);
        rest_solver.setup(mesh, params);
        for (int frame = 0; frame < 600; ++frame) {
            rest_solver.step(params.dt);
        }
        require(
            std::abs(rest_solver.last_motor_axis_angular_velocity()) < 1.0e-4f,
            "endpoint hinge at rest must not acquire angular velocity");
        require(rest_solver.last_hinge_endpoint_error() < 1.0e-5f,
                "endpoint hinge at rest must not drift from its world axis");

        const auto one_step_positions = [&](float density) {
            PabdCudaMesh moving_mesh = mesh;
            moving_mesh.initial_velocities.assign(
                4, Vec3f(1.0f, -0.25f, 0.5f));
            PabdCudaParams moving_params = params;
            moving_params.density = density;
            moving_params.stiffness = 0.0f;

            PabdCudaSolver moving_solver;
            moving_solver.setup(moving_mesh, moving_params);
            moving_solver.step(moving_params.dt);
            return moving_solver.flat_positions();
        };
        const std::vector<float> unit_mass_positions =
            one_step_positions(1.0f);
        const std::vector<float> heavy_mass_positions =
            one_step_positions(100.0f);
        require(unit_mass_positions.size() == heavy_mass_positions.size(),
                "mass-normalized hinge position vector size");
        for (std::size_t i = 0; i < unit_mass_positions.size(); ++i) {
            require_near(
                unit_mass_positions[i], heavy_mass_positions[i], 2.0e-5f,
                "mass-normalized hinge response must be density invariant");
        }
    }
}

void run_arap_beta_mass_time_normalization(
    chysx::rigid::pabd_cuda::PabdGlobalSolverMode solver_mode) {
    using namespace chysx::rigid::pabd_cuda;
    std::cout << "[pabd_arap_beta] mass/time normalization "
              << (solver_mode == PabdGlobalSolverMode::PCG
                      ? "PCG"
                      : "BlockJacobi12")
              << "\n";

    const std::array<Vec3f, 4> rest = {
        Vec3f(-1.0f, -1.0f, -1.0f),
        Vec3f(3.0f, -1.0f, -1.0f),
        Vec3f(-1.0f, 3.0f, -1.0f),
        Vec3f(-1.0f, -1.0f, 3.0f),
    };
    const auto normalized_step = [&](float density, float dt, float beta) {
        PabdCudaMesh mesh;
        mesh.rest_positions.assign(rest.begin(), rest.end());
        mesh.initial_velocities = {
            Vec3f(0.0f, 0.0f, 0.0f),
            Vec3f(1.0f, 0.2f, -0.1f),
            Vec3f(-0.3f, 0.8f, 0.4f),
            Vec3f(0.1f, -0.2f, 0.9f),
        };
        mesh.fixed.assign(4, 0);
        mesh.tets.push_back({0, 1, 2, 3});
        mesh.tet_volume_overrides.push_back(1.0f);
        std::array<float, 16> mass{};
        for (int i = 0; i < 4; ++i) mass[i * 4 + i] = 1.0f;
        mesh.tet_mass_blocks.push_back(mass);
        for (int vertex = 0; vertex < 4; ++vertex) {
            PabdSurfaceMap map;
            map.index = {0, 1, 2, 3};
            map.weight = {0.0f, 0.0f, 0.0f, 0.0f};
            map.weight[vertex] = 1.0f;
            map.body = 0;
            map.ground_collide = 0;
            mesh.surface_maps.push_back(map);
        }
        mesh.surface_triangles = {
            {0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
        mesh.surface_edges = {
            {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

        PabdCudaParams params;
        params.dt = dt;
        params.iterations = 1;
        params.gravity = 0.0f;
        params.stiffness = 0.0f;
        params.use_arap_beta = true;
        params.arap_beta = beta;
        params.density = density;
        params.damping = 1.0f;
        params.ground_stiffness = 0.0f;
        params.self_collision_stiffness = 0.0f;
        params.self_collision_thickness = 0.0f;
        params.self_collision_max_contacts = 8;
        params.pcg_iterations = 50;
        params.global_solver = solver_mode;

        PabdCudaSolver solver;
        solver.setup(mesh, params);
        solver.step(dt);
        const std::vector<float>& positions = solver.flat_positions();
        std::vector<float> normalized(positions.size(), 0.0f);
        for (int vertex = 0; vertex < 4; ++vertex) {
            const float rest_flat[3] = {
                rest[vertex].x, rest[vertex].z, rest[vertex].y};
            for (int axis = 0; axis < 3; ++axis) {
                normalized[3 * vertex + axis] =
                    (positions[3 * vertex + axis] - rest_flat[axis]) / dt;
            }
        }
        return normalized;
    };

    const std::vector<float> baseline = normalized_step(1.0f, 0.0033f, 2.0f);
    const std::vector<float> heavy = normalized_step(100.0f, 0.0033f, 2.0f);
    const std::vector<float> double_dt = normalized_step(1.0f, 0.0066f, 2.0f);
    require(baseline.size() == heavy.size() &&
                baseline.size() == double_dt.size(),
            "ARAP beta normalized response vector size");
    for (std::size_t i = 0; i < baseline.size(); ++i) {
        require_near(baseline[i], heavy[i], 2.0e-3f,
                     "ARAP beta response must be density invariant");
        require_near(baseline[i], double_dt[i], 2.0e-3f,
                     "ARAP beta response must be timestep invariant");
    }

    const std::vector<float> zero_beta =
        normalized_step(1.0f, 0.0033f, 0.0f);
    const std::vector<float> stiff_beta =
        normalized_step(1.0f, 0.0033f, 8.0f);
    float max_beta_response_delta = 0.0f;
    for (std::size_t i = 0; i < zero_beta.size(); ++i) {
        max_beta_response_delta = std::max(
            max_beta_response_delta,
            std::abs(zero_beta[i] - stiff_beta[i]));
    }
    require(max_beta_response_delta > 1.0e-2f,
            "ARAP beta must change a non-rigid prediction");
}

}  // namespace

int main() {
    try {
        run_point_face_hits();
        run_point_face_far();
        run_same_mesh_filtered();
        run_same_mesh_self_collision();
        run_self_prepass_only_for_self();
        run_fat_broadphase_cache(BroadphaseBackend::QuantBvh);
        run_edge_edge_hits();
        run_parallel_edge_edge_discarded();
        run_oriented_point_face_crossing(BroadphaseBackend::QuantBvh);
        run_oriented_edge_edge_crossing(BroadphaseBackend::QuantBvh);
        run_wide_contact_friction_spmv();
        run_per_contact_stiffness_spmv();
        run_contact_tetrahedron_volume_measure();
        run_stacked_blocks_uses_pd_abd_box_path();
        run_ground_contact_beta_mass_time_normalization(
            chysx::rigid::pabd_cuda::PabdGlobalSolverMode::PCG);
        run_ground_contact_beta_mass_time_normalization(
            chysx::rigid::pabd_cuda::PabdGlobalSolverMode::BlockJacobi12);
        run_arap_beta_mass_time_normalization(
            chysx::rigid::pabd_cuda::PabdGlobalSolverMode::PCG);
        run_arap_beta_mass_time_normalization(
            chysx::rigid::pabd_cuda::PabdGlobalSolverMode::BlockJacobi12);
        run_endpoint_hinge_motor(
            chysx::rigid::pabd_cuda::PabdGlobalSolverMode::PCG);
        run_endpoint_hinge_motor(
            chysx::rigid::pabd_cuda::PabdGlobalSolverMode::BlockJacobi12);
#ifdef CHYSX_HAS_OPTIX
        run_optix_collision_categories();
        run_fat_broadphase_cache(BroadphaseBackend::OptiX);
        run_oriented_point_face_crossing(BroadphaseBackend::OptiX);
        run_oriented_edge_edge_crossing(BroadphaseBackend::OptiX);
#endif
    } catch (const std::exception& e) {
        std::cerr << "[mesh_mesh] FAIL: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "[mesh_mesh] all tests passed\n";
    return EXIT_SUCCESS;
}
