// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "collision/mesh_mesh_contact.h"

namespace {

using chysx::collision::MeshMeshContact;
using chysx::collision::MeshMeshContactDetector;
using chysx::collision::MeshMeshContactType;
using chysx::math::Vec3f;
using chysx::math::Vec3i;

struct TestMesh {
    std::vector<Vec3f> positions;
    std::vector<Vec3i> triangles;
    std::vector<int> mesh_ids;
};

void require(bool cond, const std::string& message) {
    if (!cond) {
        throw std::runtime_error(message);
    }
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
                                             int max_contacts = 256) {
    MeshMeshContactDetector detector;
    detector.setup(mesh.triangles, mesh.mesh_ids, max_contacts);
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

void run_point_face_hits() {
    std::cout << "[mesh_mesh] point-face hit\n";
    auto contacts = detect_contacts(point_face_case(0.04f, false), 0.10f);
    require(count_type(contacts, MeshMeshContactType::PointFace) > 0,
            "expected at least one point-face contact");
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

}  // namespace

int main() {
    try {
        run_point_face_hits();
        run_point_face_far();
        run_same_mesh_filtered();
        run_edge_edge_hits();
        run_parallel_edge_edge_discarded();
    } catch (const std::exception& e) {
        std::cerr << "[mesh_mesh] FAIL: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "[mesh_mesh] all tests passed\n";
    return EXIT_SUCCESS;
}
