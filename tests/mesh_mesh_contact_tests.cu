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

namespace {

using chysx::collision::MeshMeshContact;
using chysx::collision::MeshMeshContactDetector;
using chysx::collision::MeshMeshContactType;
using chysx::collision::BroadphaseBackend;
using chysx::collision::MeshCollisionCategory;
using chysx::collision::MeshCollisionMask;
using chysx::collision::kMeshCollisionInterObject;
using chysx::collision::kMeshCollisionSelf;
using chysx::collision::WideContact;
using chysx::collision::WideContactSpMVOp;
using chysx::math::Mat3f;
using chysx::math::Vec3f;
using chysx::math::Vec3i;
using chysx::math::Vec4f;

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
