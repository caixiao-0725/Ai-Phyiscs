// SPDX-License-Identifier: Apache-2.0

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <json/json.h>

#include "io/obj_io.h"
#include "render/scene.h"
#include "rigid/pabd_cuda/pabd_cuda_solver.h"
#include "rigid/pabd_cuda/pabd_cuda_surface_renderer.h"

namespace {

using chysx::rigid::pabd_cuda::PabdCudaMesh;
using chysx::rigid::pabd_cuda::PabdCudaParams;
using chysx::rigid::pabd_cuda::PabdCudaSolver;
using chysx::rigid::pabd_cuda::PabdCudaSurfaceRenderer;
using chysx::rigid::pabd_cuda::PabdElasticCurvatureMode;
using chysx::rigid::pabd_cuda::PabdEndpointHinge;
using chysx::rigid::pabd_cuda::PabdGlobalSolverMode;
using chysx::rigid::pabd_cuda::PabdPolarGnBackend;
using chysx::rigid::pabd_cuda::PabdSelfContactMeasureMode;
using chysx::rigid::pabd_cuda::PabdSurfaceMap;
using chysx::rigid::pabd_cuda::elastic_curvature_mode_name;
using chysx::rigid::pabd_cuda::polar_gn_backend_name;
using chysx::rigid::pabd_cuda::kDefaultStackCount;
using chysx::rigid::pabd_cuda::kPdAbdBoxSurfaceTris;
using chysx::rigid::pabd_cuda::kPdAbdDefaultBoxCount;

enum class PabdSceneKind {
    SingleTetraPinned,
    HexDropGround,
    TwentyTets,
    StackedBlocks,
    PdAbdBoxes,
    TetraEECross,
    SingleTorqueGear,
    TorqueGearLine2,
    TorqueGearLine6,
    RigidIpcChainNet4x4,
    RigidIpcChainNet8x8,
    RigidIpcChainNet16x16Ball,
    RigidIpcChainNet32x32,
    RigidIpcVerticalChain,
};

constexpr int kRigidIpcChainNet4x4Bodies = 24;
constexpr int kRigidIpcChainNet8x8Bodies = 144;
constexpr int kRigidIpcChainNet16x16BallBodies = 673;
constexpr int kRigidIpcChainNet32x32Bodies = 2880;
constexpr int kRigidIpcVerticalChainDefaultBodies = 2;

const char* self_contact_measure_name(PabdSelfContactMeasureMode mode) {
    return mode == PabdSelfContactMeasureMode::EffectiveMass
        ? "effective_mass"
        : "tetra_volume";
}

const char* self_contact_features_name(bool point_face, bool edge_edge) {
    if (point_face && edge_edge) return "PF+EE";
    if (point_face) return "PF";
    if (edge_edge) return "EE";
    return "none";
}

void body_color(int body, int count, float out_rgb[3]) {
    const float hue = static_cast<float>(body) / static_cast<float>(std::max(1, count));
    const float h = hue * 6.0f;
    const int sector = static_cast<int>(h) % 6;
    const float f = h - static_cast<float>(sector);
    switch (sector) {
        case 0: out_rgb[0] = 1.0f; out_rgb[1] = f; out_rgb[2] = 0.0f; break;
        case 1: out_rgb[0] = 1.0f - f; out_rgb[1] = 1.0f; out_rgb[2] = 0.0f; break;
        case 2: out_rgb[0] = 0.0f; out_rgb[1] = 1.0f; out_rgb[2] = f; break;
        case 3: out_rgb[0] = 0.0f; out_rgb[1] = 1.0f - f; out_rgb[2] = 1.0f; break;
        case 4: out_rgb[0] = f; out_rgb[1] = 0.0f; out_rgb[2] = 1.0f; break;
        default: out_rgb[0] = 1.0f; out_rgb[1] = 0.0f; out_rgb[2] = 1.0f - f; break;
    }
    constexpr float kMin = 0.25f;
    constexpr float kRange = 0.70f;
    out_rgb[0] = kMin + kRange * out_rgb[0];
    out_rgb[1] = kMin + kRange * out_rgb[1];
    out_rgb[2] = kMin + kRange * out_rgb[2];
}

chysx::math::Vec3f rotate_xyz_degrees(chysx::math::Vec3f p,
                                       chysx::math::Vec3f deg) {
    constexpr float kPi = 3.14159265358979323846f;
    const float rx = deg.x * kPi / 180.0f;
    const float ry = deg.y * kPi / 180.0f;
    const float rz = deg.z * kPi / 180.0f;
    const float cx = std::cos(rx);
    const float sx = std::sin(rx);
    const float cy = std::cos(ry);
    const float sy = std::sin(ry);
    const float cz = std::cos(rz);
    const float sz = std::sin(rz);

    p = chysx::math::Vec3f(p.x, cx * p.y - sx * p.z,
                           sx * p.y + cx * p.z);
    p = chysx::math::Vec3f(cy * p.x + sy * p.z, p.y,
                           -sy * p.x + cy * p.z);
    p = chysx::math::Vec3f(cz * p.x - sz * p.y,
                           sz * p.x + cz * p.y, p.z);
    return p;
}

std::string rigid_ipc_fixture_path(const char* filename) {
    return std::string("D:/github/rigid-ipc/fixtures/3D/chain/chain-net/") +
           filename;
}

std::string resolve_rigid_ipc_mesh_path(const std::string& mesh_name) {
    return std::string("D:/github/rigid-ipc/meshes/") + mesh_name;
}

std::string pabd_torque_fixture_path() {
    return "D:/github/pabd/fixtures/3D/mechanisms/gears/"
           "single-torque-driver.json";
}

std::string pabd_gear_line_fixture_path() {
    return "D:/github/pabd/fixtures/3D/mechanisms/gears/"
           "line-6-kinematic.json";
}

std::string resolve_pabd_mesh_path(const std::string& mesh_name) {
    return std::string("D:/github/pabd/meshes/") + mesh_name;
}

float bbox_volume(const chysx::math::Vec3f& min_p,
                  const chysx::math::Vec3f& max_p) {
    const chysx::math::Vec3f e = max_p - min_p;
    return std::max(1.0e-6f, e.x * e.y * e.z);
}

std::vector<std::array<int, 2>> make_obj_edges(
    const std::vector<std::array<int, 3>>& tris) {
    std::map<std::pair<int, int>, int> edge_seen;
    auto add_edge = [&](int a, int b) {
        if (a > b) std::swap(a, b);
        edge_seen[{a, b}] = 1;
    };
    for (const auto& tri : tris) {
        add_edge(tri[0], tri[1]);
        add_edge(tri[1], tri[2]);
        add_edge(tri[2], tri[0]);
    }
    std::vector<std::array<int, 2>> edges;
    edges.reserve(edge_seen.size());
    for (const auto& entry : edge_seen) {
        edges.push_back({entry.first.first, entry.first.second});
    }
    return edges;
}

PabdCudaMesh make_torque_gear_fixture_mesh(const std::string& json_path,
                                           int max_gears,
                                           float density_scale,
                                           const char* label) {
    std::ifstream file(json_path, std::ifstream::binary);
    if (!file.is_open()) {
        throw std::runtime_error(std::string(label) + ": cannot open " +
                                 json_path);
    }

    Json::CharReaderBuilder reader;
    reader["allowComments"] = true;
    reader["allowTrailingCommas"] = true;
    std::string errors;
    Json::Value root;
    if (!Json::parseFromStream(reader, file, &root, &errors)) {
        throw std::runtime_error(std::string(label) +
                                 ": JSON parse error: " + errors);
    }

    const Json::Value& bodies =
        root["rigid_body_problem"]["rigid_bodies"];
    if (!bodies.isArray() || bodies.empty()) {
        throw std::runtime_error(std::string(label) +
                                 ": fixture has no rigid bodies");
    }
    max_gears = std::max(1, max_gears);

    auto read_vec3 = [](const Json::Value& value,
                        chysx::math::Vec3f fallback) {
        if (!value.isArray() || value.size() < 3) return fallback;
        return chysx::math::Vec3f(value[0].asFloat(), value[1].asFloat(),
                                  value[2].asFloat());
    };

    PabdCudaMesh mesh;
    std::map<std::string, chysx::io::ObjMesh> obj_cache;
    int motor_count = 0;
    for (const Json::Value& gear_body : bodies) {
        if (static_cast<int>(mesh.tets.size()) >= max_gears) break;
        if (!gear_body.get("enabled", true).asBool()) continue;
        const std::string mesh_name = gear_body["mesh"].asString();
        if (mesh_name.empty()) continue;

        const std::string obj_path = resolve_pabd_mesh_path(mesh_name);
        auto obj_it = obj_cache.find(obj_path);
        if (obj_it == obj_cache.end()) {
            chysx::io::ObjMesh obj;
            if (!chysx::io::load_obj(obj_path, obj)) {
                throw std::runtime_error(std::string(label) +
                                         ": failed to load " + obj_path);
            }
            obj_it = obj_cache.emplace(obj_path, std::move(obj)).first;
        }
        const chysx::io::ObjMesh& obj = obj_it->second;
        if (obj.positions.size() % 3 != 0 ||
            obj.triangles.size() % 3 != 0 || obj.positions.empty() ||
            obj.triangles.empty()) {
            throw std::runtime_error(std::string(label) +
                                     ": invalid OBJ " + obj_path);
        }

        const chysx::math::Vec3f translation = read_vec3(
            gear_body["position"], chysx::math::Vec3f(0.0f));
        const chysx::math::Vec3f rotation = read_vec3(
            gear_body["rotation"], chysx::math::Vec3f(0.0f));
        const chysx::math::Vec3f angular_velocity = read_vec3(
            gear_body["angular_velocity"], chysx::math::Vec3f(0.0f));
        const float fixture_density = std::max(
            1.0e-6f, gear_body.get("density", 1.0f).asFloat() *
                         density_scale);
        chysx::math::Vec3f scale(1.0f);
        const Json::Value& scale_json = gear_body["scale"];
        if (scale_json.isArray() && scale_json.size() >= 3) {
            scale = read_vec3(scale_json, scale);
        } else if (!scale_json.isNull()) {
            scale = chysx::math::Vec3f(scale_json.asFloat());
        }

        chysx::math::Vec3f axis(0.0f, 1.0f, 0.0f);
        const Json::Value& fixed_flags = gear_body["is_dof_fixed"];
        if (fixed_flags.isArray() && fixed_flags.size() >= 6) {
            int free_rotation = -1;
            for (int dof = 3; dof < 6; ++dof) {
                if (!fixed_flags[dof].asBool()) {
                    if (free_rotation >= 0) {
                        throw std::runtime_error(
                            std::string(label) +
                            ": hinge needs exactly one free rotation");
                    }
                    free_rotation = dof - 3;
                }
            }
            if (free_rotation < 0) {
                throw std::runtime_error(std::string(label) +
                                         ": hinge has no free rotation");
            }
            axis = chysx::math::Vec3f(0.0f);
            axis[free_rotation] = 1.0f;
        }

        std::vector<chysx::math::Vec3f> surface;
        surface.reserve(obj.positions.size() / 3);
        chysx::math::Vec3f bbox_min(std::numeric_limits<float>::max());
        chysx::math::Vec3f bbox_max(-std::numeric_limits<float>::max());
        for (std::size_t i = 0; i < obj.positions.size(); i += 3) {
            chysx::math::Vec3f p(obj.positions[i + 0] * scale.x,
                                  obj.positions[i + 1] * scale.y,
                                  obj.positions[i + 2] * scale.z);
            p = rotate_xyz_degrees(p, rotation) + translation;
            surface.push_back(p);
            bbox_min = chysx::math::min(bbox_min, p);
            bbox_max = chysx::math::max(bbox_max, p);
        }

        std::vector<std::array<int, 3>> local_tris;
        local_tris.reserve(obj.triangles.size() / 3);
        for (std::size_t i = 0; i < obj.triangles.size(); i += 3) {
            const int a = obj.triangles[i + 0];
            const int b = obj.triangles[i + 1];
            const int c = obj.triangles[i + 2];
            if (a < 0 || b < 0 || c < 0 ||
                a >= static_cast<int>(surface.size()) ||
                b >= static_cast<int>(surface.size()) ||
                c >= static_cast<int>(surface.size())) {
                throw std::runtime_error(std::string(label) +
                                         ": OBJ index out of range");
            }
            local_tris.push_back({a, b, c});
        }
        const std::vector<std::array<int, 2>> local_edges =
            make_obj_edges(local_tris);

        const chysx::math::Vec3f extents = bbox_max - bbox_min;
        const float r = std::max(
            1.0e-6f, 2.0f * chysx::math::length(extents));
        const std::array<chysx::math::Vec3f, 4> controls = {{
            bbox_min,
            chysx::math::Vec3f(bbox_min.x + r, bbox_min.y, bbox_min.z),
            chysx::math::Vec3f(bbox_min.x, bbox_min.y + r, bbox_min.z),
            chysx::math::Vec3f(bbox_min.x, bbox_min.y, bbox_min.z + r),
        }};
        const int body = static_cast<int>(mesh.tets.size());
        const int dof_start = static_cast<int>(mesh.rest_positions.size());
        const int surface_start =
            static_cast<int>(mesh.surface_maps.size());
        mesh.rest_positions.insert(mesh.rest_positions.end(),
                                   controls.begin(), controls.end());
        mesh.initial_velocities.insert(
            mesh.initial_velocities.end(), 4,
            chysx::math::Vec3f(0.0f));
        mesh.fixed.insert(mesh.fixed.end(), 4, 0);
        mesh.tets.push_back({dof_start + 0, dof_start + 1,
                             dof_start + 2, dof_start + 3});
        const float volume = bbox_volume(bbox_min, bbox_max);
        mesh.tet_volume_overrides.push_back(volume);

        for (const chysx::math::Vec3f& p : surface) {
            const float w1 = (p.x - bbox_min.x) / r;
            const float w2 = (p.y - bbox_min.y) / r;
            const float w3 = (p.z - bbox_min.z) / r;
            PabdSurfaceMap map;
            map.index = {dof_start + 0, dof_start + 1,
                         dof_start + 2, dof_start + 3};
            map.weight = {1.0f - w1 - w2 - w3, w1, w2, w3};
            map.body = body;
            map.collide = 1;
            map.self_collide = 1;
            map.ground_collide = 0;
            mesh.surface_maps.push_back(map);
        }
        for (const auto& tri : local_tris) {
            mesh.surface_triangles.push_back({
                surface_start + tri[0], surface_start + tri[1],
                surface_start + tri[2]});
        }
        for (const auto& edge : local_edges) {
            mesh.surface_edges.push_back({surface_start + edge[0],
                                          surface_start + edge[1]});
        }

        std::array<float, 16> mass_block{};
        mass_block.fill(0.0f);
        constexpr float kInvSqrt3 = 0.5773502691896258f;
        const chysx::math::Vec3f center = (bbox_min + bbox_max) * 0.5f;
        const float sample_mass = volume * fixture_density / 8.0f;
        for (float sx : {-1.0f, 1.0f}) {
            for (float sy : {-1.0f, 1.0f}) {
                for (float sz : {-1.0f, 1.0f}) {
                    const chysx::math::Vec3f sample(
                        center.x + sx * extents.x * 0.5f * kInvSqrt3,
                        center.y + sy * extents.y * 0.5f * kInvSqrt3,
                        center.z + sz * extents.z * 0.5f * kInvSqrt3);
                    const float w1 = (sample.x - bbox_min.x) / r;
                    const float w2 = (sample.y - bbox_min.y) / r;
                    const float w3 = (sample.z - bbox_min.z) / r;
                    const float weights[4] = {
                        1.0f - w1 - w2 - w3, w1, w2, w3};
                    for (int row = 0; row < 4; ++row) {
                        for (int col = 0; col < 4; ++col) {
                            mass_block[row * 4 + col] +=
                                sample_mass * weights[row] * weights[col];
                        }
                    }
                }
            }
        }
        mesh.tet_mass_blocks.push_back(mass_block);

        float min_projection = std::numeric_limits<float>::max();
        float max_projection = -std::numeric_limits<float>::max();
        float max_radius = 0.0f;
        for (const chysx::math::Vec3f& p : surface) {
            const chysx::math::Vec3f d = p - translation;
            const float projection = chysx::math::dot(d, axis);
            min_projection = std::min(min_projection, projection);
            max_projection = std::max(max_projection, projection);
            max_radius = std::max(
                max_radius,
                chysx::math::length(d - axis * projection));
        }
        if (max_projection - min_projection < 1.0e-5f) {
            const float half_length =
                std::max(0.05f, 0.25f * max_radius);
            min_projection = -half_length;
            max_projection = half_length;
        }

        const chysx::math::Vec3f endpoint0 =
            translation + axis * min_projection;
        const chysx::math::Vec3f endpoint1 =
            translation + axis * max_projection;
        auto endpoint_weights = [&](chysx::math::Vec3f p) {
            const float w1 = (p.x - bbox_min.x) / r;
            const float w2 = (p.y - bbox_min.y) / r;
            const float w3 = (p.z - bbox_min.z) / r;
            return chysx::math::Vec4f(1.0f - w1 - w2 - w3,
                                      w1, w2, w3);
        };

        const bool motor =
            gear_body.get("type", "").asString() == "kinematic" &&
            chysx::math::length(angular_velocity) > 1.0e-6f;
        PabdEndpointHinge hinge;
        hinge.body = body;
        hinge.motor = motor ? 1 : 0;
        hinge.weights0 = endpoint_weights(endpoint0);
        hinge.weights1 = endpoint_weights(endpoint1);
        hinge.endpoint0 = chysx::math::Vec4f(
            endpoint0.x, endpoint0.y, endpoint0.z, 0.0f);
        hinge.endpoint1 = chysx::math::Vec4f(
            endpoint1.x, endpoint1.y, endpoint1.z, 0.0f);
        hinge.axis = chysx::math::Vec4f(axis.x, axis.y, axis.z, 0.0f);
        mesh.endpoint_hinges.push_back(hinge);
        motor_count += motor ? 1 : 0;

        std::printf(
            "[PABD Gear] body=%d motor=%d center=(%.3f,%.3f,%.3f) "
            "axis=(%.1f,%.1f,%.1f) endpoints=(%.4f,%.4f)\n",
            body, motor ? 1 : 0, translation.x, translation.y,
            translation.z, axis.x, axis.y, axis.z,
            min_projection, max_projection);
    }

    if (static_cast<int>(mesh.tets.size()) != max_gears) {
        throw std::runtime_error(
            std::string(label) + ": requested " +
            std::to_string(max_gears) + " gears, loaded " +
            std::to_string(mesh.tets.size()));
    }
    if (motor_count != 1) {
        throw std::runtime_error(std::string(label) +
                                 ": expected exactly one torque motor");
    }
    std::printf(
        "[PABD Gear] %s loaded %zu bodies, %zu vertices, %zu triangles "
        "from %s\n",
        label, mesh.tets.size(), mesh.surface_maps.size(),
        mesh.surface_triangles.size(), json_path.c_str());
    return mesh;
}

PabdCudaMesh make_single_torque_driver_mesh() {
    return make_torque_gear_fixture_mesh(
        pabd_torque_fixture_path(), 1, 1.0f, "Single Torque Gear");
}

PabdCudaMesh make_torque_gear_line_mesh(int gear_count) {
    return make_torque_gear_fixture_mesh(
        pabd_gear_line_fixture_path(), gear_count, 1.0f,
        gear_count == 2 ? "Torque Gear Line 2" : "Torque Gear Line 6");
}

PabdCudaMesh make_rigid_ipc_chain_net_mesh(const char* filename,
                                           const char* label,
                                           bool center_sphere_on_links = false) {
    const std::string json_path = rigid_ipc_fixture_path(filename);
    std::ifstream file(json_path, std::ifstream::binary);
    if (!file.is_open()) {
        throw std::runtime_error(std::string(label) + ": cannot open " +
                                 json_path);
    }

    Json::CharReaderBuilder reader;
    reader["allowComments"] = true;
    reader["allowTrailingCommas"] = true;
    std::string errors;
    Json::Value root;
    if (!Json::parseFromStream(reader, file, &root, &errors)) {
        throw std::runtime_error(std::string(label) + ": JSON parse error: " +
                                 errors);
    }

    const Json::Value& bodies =
        root["rigid_body_problem"]["rigid_bodies"];
    if (!bodies.isArray() || bodies.empty()) {
        throw std::runtime_error(std::string(label) + ": no rigid bodies");
    }

    chysx::math::Vec3f link_center(0.0f, 0.0f, 0.0f);
    if (center_sphere_on_links) {
        chysx::math::Vec3f link_min(std::numeric_limits<float>::max());
        chysx::math::Vec3f link_max(-std::numeric_limits<float>::max());
        bool found_link = false;
        for (const Json::Value& body : bodies) {
            if (body["mesh"].asString() == "sphere.obj") continue;
            const Json::Value& position = body["position"];
            const chysx::math::Vec3f p(position[0].asFloat(),
                                       position[1].asFloat(),
                                       position[2].asFloat());
            link_min = chysx::math::min(link_min, p);
            link_max = chysx::math::max(link_max, p);
            found_link = true;
        }
        if (!found_link) {
            throw std::runtime_error(std::string(label) +
                                     ": cannot center sphere without links");
        }
        link_center = (link_min + link_max) * 0.5f;
    }

    std::map<std::string, chysx::io::ObjMesh> obj_cache;
    PabdCudaMesh mesh;
    mesh.rest_positions.reserve(static_cast<std::size_t>(bodies.size()) * 4);
    mesh.tets.reserve(bodies.size());
    mesh.tet_volume_overrides.reserve(bodies.size());
    mesh.tet_mass_blocks.reserve(bodies.size());

    for (Json::ArrayIndex body_i = 0; body_i < bodies.size(); ++body_i) {
        const Json::Value& body = bodies[body_i];
        const std::string mesh_name = body["mesh"].asString();
        const std::string obj_path = resolve_rigid_ipc_mesh_path(mesh_name);
        auto [cache_it, inserted] = obj_cache.emplace(mesh_name,
                                                      chysx::io::ObjMesh{});
        if (inserted && !chysx::io::load_obj(obj_path, cache_it->second)) {
            throw std::runtime_error(std::string(label) +
                                     ": failed to load " +
                                     obj_path);
        }
        const chysx::io::ObjMesh& obj = cache_it->second;
        if (obj.positions.size() % 3 != 0 ||
            obj.triangles.size() % 3 != 0 ||
            obj.positions.empty() || obj.triangles.empty()) {
            throw std::runtime_error(std::string(label) + ": invalid OBJ " +
                                     obj_path);
        }

        chysx::math::Vec3f scale(1.0f, 1.0f, 1.0f);
        const Json::Value& scale_json = body["scale"];
        if (scale_json.isArray() && scale_json.size() >= 3) {
            scale = chysx::math::Vec3f(scale_json[0].asFloat(),
                                       scale_json[1].asFloat(),
                                       scale_json[2].asFloat());
        } else if (!scale_json.isNull()) {
            const float uniform_scale = scale_json.asFloat();
            scale = chysx::math::Vec3f(uniform_scale, uniform_scale,
                                       uniform_scale);
        }

        std::vector<chysx::math::Vec3f> local_surface;
        local_surface.reserve(obj.positions.size() / 3);
        chysx::math::Vec3f bbox_min(3.402823466e38f);
        chysx::math::Vec3f bbox_max(-3.402823466e38f);
        for (std::size_t i = 0; i < obj.positions.size(); i += 3) {
            const chysx::math::Vec3f p(obj.positions[i + 0] * scale.x,
                                       obj.positions[i + 1] * scale.y,
                                       obj.positions[i + 2] * scale.z);
            local_surface.push_back(p);
            bbox_min = chysx::math::min(bbox_min, p);
            bbox_max = chysx::math::max(bbox_max, p);
        }

        std::vector<std::array<int, 3>> local_tris;
        local_tris.reserve(obj.triangles.size() / 3);
        for (std::size_t i = 0; i < obj.triangles.size(); i += 3) {
            const int a = obj.triangles[i + 0];
            const int b = obj.triangles[i + 1];
            const int c = obj.triangles[i + 2];
            if (a < 0 || b < 0 || c < 0 ||
                a >= static_cast<int>(local_surface.size()) ||
                b >= static_cast<int>(local_surface.size()) ||
                c >= static_cast<int>(local_surface.size())) {
                throw std::runtime_error(
                    std::string(label) + ": OBJ index out of range");
            }
            local_tris.push_back({a, b, c});
        }
        const std::vector<std::array<int, 2>> local_edges =
            make_obj_edges(local_tris);

        const Json::Value& pos_json = body["position"];
        const Json::Value& rot_json = body["rotation"];
        chysx::math::Vec3f translation(pos_json[0].asFloat(),
                                       pos_json[1].asFloat(),
                                       pos_json[2].asFloat());
        if (center_sphere_on_links && mesh_name == "sphere.obj") {
            translation.x = link_center.x;
            translation.z = link_center.z;
            std::printf(
                "[PABD ChainNet] centered sphere at (%.3f, %.3f, %.3f)\n",
                translation.x, translation.y, translation.z);
        }
        const chysx::math::Vec3f rotation(rot_json[0].asFloat(),
                                          rot_json[1].asFloat(),
                                          rot_json[2].asFloat());
        const bool fixed = body.get("is_dof_fixed", false).asBool() ||
                           body.get("type", "").asString() == "static";

        const int dof_start = static_cast<int>(mesh.rest_positions.size());
        const int surface_start = static_cast<int>(mesh.surface_maps.size());
        const chysx::math::Vec3f extents = bbox_max - bbox_min;
        const float r = std::max(1.0e-6f,
                                 2.0f * chysx::math::length(extents));
        const std::array<chysx::math::Vec3f, 4> local_control = {{
            bbox_min,
            chysx::math::Vec3f(bbox_min.x + r, bbox_min.y, bbox_min.z),
            chysx::math::Vec3f(bbox_min.x, bbox_min.y + r, bbox_min.z),
            chysx::math::Vec3f(bbox_min.x, bbox_min.y, bbox_min.z + r),
        }};

        for (int k = 0; k < 4; ++k) {
            mesh.rest_positions.push_back(
                translation + rotate_xyz_degrees(local_control[k], rotation));
            mesh.fixed.push_back(fixed ? 1 : 0);
        }
        mesh.tets.push_back({dof_start + 0, dof_start + 1,
                             dof_start + 2, dof_start + 3});
        const float volume = bbox_volume(bbox_min, bbox_max);
        mesh.tet_volume_overrides.push_back(volume);

        for (const chysx::math::Vec3f& p0 : local_surface) {
            const float w1 = (p0.x - bbox_min.x) / r;
            const float w2 = (p0.y - bbox_min.y) / r;
            const float w3 = (p0.z - bbox_min.z) / r;
            PabdSurfaceMap map;
            map.index = {dof_start + 0, dof_start + 1,
                         dof_start + 2, dof_start + 3};
            map.weight = {1.0f - w1 - w2 - w3, w1, w2, w3};
            map.body = static_cast<int>(body_i);
            map.collide = 1;
            map.self_collide = 1;
            map.ground_collide = 0;
            mesh.surface_maps.push_back(map);
        }
        for (const auto& tri : local_tris) {
            mesh.surface_triangles.push_back({
                surface_start + tri[0],
                surface_start + tri[1],
                surface_start + tri[2],
            });
        }
        for (const auto& edge : local_edges) {
            mesh.surface_edges.push_back({
                surface_start + edge[0],
                surface_start + edge[1],
            });
        }

        std::array<float, 16> mass_block{};
        mass_block.fill(0.0f);
        const float lumped = volume / 4.0f;
        for (int i = 0; i < 4; ++i) {
            mass_block[i * 4 + i] = lumped;
        }
        mesh.tet_mass_blocks.push_back(mass_block);
    }

    std::printf("[PABD ChainNet] %s loaded %u bodies, %zu surface vertices, "
                "%zu triangles from %s\n",
                label, bodies.size(), mesh.surface_maps.size(),
                mesh.surface_triangles.size(), json_path.c_str());
    return mesh;
}

PabdCudaMesh make_rigid_ipc_chain_net_4x4_mesh() {
    return make_rigid_ipc_chain_net_mesh("chain-net-4x4.json",
                                         "Rigid-IPC Chain Net 4x4");
}

PabdCudaMesh make_rigid_ipc_chain_net_8x8_mesh() {
    return make_rigid_ipc_chain_net_mesh("chain-net-8x8.json",
                                         "Rigid-IPC Chain Net 8x8");
}

PabdCudaMesh make_rigid_ipc_chain_net_16x16_ball_mesh() {
    return make_rigid_ipc_chain_net_mesh(
        "chain-net-16x16-ball.json", "Rigid-IPC Chain Net 16x16 Ball", true);
}

PabdCudaMesh make_rigid_ipc_chain_net_32x32_mesh() {
    return make_rigid_ipc_chain_net_mesh("chain-net-32x32.json",
                                         "Rigid-IPC Chain Net 32x32");
}

PabdCudaMesh make_rigid_ipc_vertical_chain_mesh(int count) {
    const int n_links = std::max(2, count);
    const std::string obj_path =
        resolve_rigid_ipc_mesh_path("wrecking-ball/link.obj");
    chysx::io::ObjMesh obj;
    if (!chysx::io::load_obj(obj_path, obj)) {
        throw std::runtime_error("Rigid-IPC Vertical Chain: failed to load " +
                                 obj_path);
    }
    if (obj.positions.size() % 3 != 0 ||
        obj.triangles.size() % 3 != 0 ||
        obj.positions.empty() || obj.triangles.empty()) {
        throw std::runtime_error("Rigid-IPC Vertical Chain: invalid OBJ " +
                                 obj_path);
    }

    std::vector<chysx::math::Vec3f> local_surface;
    local_surface.reserve(obj.positions.size() / 3);
    chysx::math::Vec3f bbox_min(3.402823466e38f);
    chysx::math::Vec3f bbox_max(-3.402823466e38f);
    for (std::size_t i = 0; i < obj.positions.size(); i += 3) {
        const chysx::math::Vec3f p(obj.positions[i + 0],
                                   obj.positions[i + 1],
                                   obj.positions[i + 2]);
        local_surface.push_back(p);
        bbox_min = chysx::math::min(bbox_min, p);
        bbox_max = chysx::math::max(bbox_max, p);
    }

    std::vector<std::array<int, 3>> local_tris;
    local_tris.reserve(obj.triangles.size() / 3);
    for (std::size_t i = 0; i < obj.triangles.size(); i += 3) {
        const int a = obj.triangles[i + 0];
        const int b = obj.triangles[i + 1];
        const int c = obj.triangles[i + 2];
        if (a < 0 || b < 0 || c < 0 ||
            a >= static_cast<int>(local_surface.size()) ||
            b >= static_cast<int>(local_surface.size()) ||
            c >= static_cast<int>(local_surface.size())) {
            throw std::runtime_error(
                "Rigid-IPC Vertical Chain: OBJ index out of range");
        }
        local_tris.push_back({a, b, c});
    }
    const std::vector<std::array<int, 2>> local_edges =
        make_obj_edges(local_tris);

    const chysx::math::Vec3f extents = bbox_max - bbox_min;
    const float r = std::max(1.0e-6f,
                             2.0f * chysx::math::length(extents));
    const std::array<chysx::math::Vec3f, 4> local_control = {{
        bbox_min,
        chysx::math::Vec3f(bbox_min.x + r, bbox_min.y, bbox_min.z),
        chysx::math::Vec3f(bbox_min.x, bbox_min.y + r, bbox_min.z),
        chysx::math::Vec3f(bbox_min.x, bbox_min.y, bbox_min.z + r),
    }};
    const float volume = bbox_volume(bbox_min, bbox_max);

    auto to_vertical = [](chysx::math::Vec3f p) {
        return rotate_xyz_degrees(p, chysx::math::Vec3f(90.0f, 0.0f, 0.0f)) +
               chysx::math::Vec3f(0.0f, 2.0f, 0.0f);
    };

    PabdCudaMesh mesh;
    mesh.rest_positions.reserve(static_cast<std::size_t>(n_links) * 4);
    mesh.fixed.reserve(static_cast<std::size_t>(n_links) * 4);
    mesh.tets.reserve(n_links);
    mesh.surface_maps.reserve(static_cast<std::size_t>(n_links) *
                              local_surface.size());
    mesh.surface_triangles.reserve(static_cast<std::size_t>(n_links) *
                                   local_tris.size());
    mesh.surface_edges.reserve(static_cast<std::size_t>(n_links) *
                               local_edges.size());
    mesh.tet_volume_overrides.reserve(n_links);
    mesh.tet_mass_blocks.reserve(n_links);

    for (int body = 0; body < n_links; ++body) {
        const chysx::math::Vec3f chain_pos(
            0.0f, 0.0f, static_cast<float>(body) * 0.8f);
        const chysx::math::Vec3f chain_rot =
            (body % 2 == 0)
                ? chysx::math::Vec3f(90.0f, 90.0f, 0.0f)
                : chysx::math::Vec3f(90.0f, 0.0f, 90.0f);
        const bool fixed = body == 0;
        const int dof_start = static_cast<int>(mesh.rest_positions.size());
        const int surface_start = static_cast<int>(mesh.surface_maps.size());

        for (int k = 0; k < 4; ++k) {
            mesh.rest_positions.push_back(
                to_vertical(chain_pos +
                            rotate_xyz_degrees(local_control[k], chain_rot)));
            mesh.fixed.push_back(fixed ? 1 : 0);
        }
        mesh.tets.push_back({dof_start + 0, dof_start + 1,
                             dof_start + 2, dof_start + 3});
        mesh.tet_volume_overrides.push_back(volume);

        for (const chysx::math::Vec3f& p0 : local_surface) {
            const float w1 = (p0.x - bbox_min.x) / r;
            const float w2 = (p0.y - bbox_min.y) / r;
            const float w3 = (p0.z - bbox_min.z) / r;
            PabdSurfaceMap map;
            map.index = {dof_start + 0, dof_start + 1,
                         dof_start + 2, dof_start + 3};
            map.weight = {1.0f - w1 - w2 - w3, w1, w2, w3};
            map.body = body;
            map.collide = 1;
            map.self_collide = 1;
            map.ground_collide = 0;
            mesh.surface_maps.push_back(map);
        }
        for (const auto& tri : local_tris) {
            mesh.surface_triangles.push_back({
                surface_start + tri[0],
                surface_start + tri[1],
                surface_start + tri[2],
            });
        }
        for (const auto& edge : local_edges) {
            mesh.surface_edges.push_back({
                surface_start + edge[0],
                surface_start + edge[1],
            });
        }

        std::array<float, 16> mass_block{};
        mass_block.fill(0.0f);
        const float lumped = volume / 4.0f;
        for (int i = 0; i < 4; ++i) {
            mass_block[i * 4 + i] = lumped;
        }
        mesh.tet_mass_blocks.push_back(mass_block);
    }

    std::printf("[PABD Chain] Vertical Chain loaded %d bodies, %zu surface "
                "vertices, %zu triangles from %s\n",
                n_links, mesh.surface_maps.size(),
                mesh.surface_triangles.size(), obj_path.c_str());
    return mesh;
}

bool is_rigid_ipc_chain_net(PabdSceneKind kind) {
    return kind == PabdSceneKind::RigidIpcChainNet4x4 ||
           kind == PabdSceneKind::RigidIpcChainNet8x8 ||
           kind == PabdSceneKind::RigidIpcChainNet16x16Ball ||
           kind == PabdSceneKind::RigidIpcChainNet32x32;
}

bool is_torque_gear_scene(PabdSceneKind kind) {
    return kind == PabdSceneKind::SingleTorqueGear ||
           kind == PabdSceneKind::TorqueGearLine2 ||
           kind == PabdSceneKind::TorqueGearLine6;
}

int torque_gear_body_count(PabdSceneKind kind) {
    if (kind == PabdSceneKind::TorqueGearLine6) return 6;
    if (kind == PabdSceneKind::TorqueGearLine2) return 2;
    return 1;
}

bool is_rigid_ipc_link_scene(PabdSceneKind kind) {
    return is_rigid_ipc_chain_net(kind) ||
           kind == PabdSceneKind::RigidIpcVerticalChain;
}

bool is_tuned_rigid_ipc_chain_net(PabdSceneKind kind) {
    return kind == PabdSceneKind::RigidIpcChainNet4x4 ||
           kind == PabdSceneKind::RigidIpcChainNet8x8 ||
           kind == PabdSceneKind::RigidIpcChainNet16x16Ball ||
           kind == PabdSceneKind::RigidIpcChainNet32x32;
}

int rigid_ipc_chain_net_body_count(PabdSceneKind kind) {
    if (kind == PabdSceneKind::RigidIpcChainNet32x32) {
        return kRigidIpcChainNet32x32Bodies;
    }
    if (kind == PabdSceneKind::RigidIpcChainNet16x16Ball) {
        return kRigidIpcChainNet16x16BallBodies;
    }
    if (kind == PabdSceneKind::RigidIpcChainNet8x8) {
        return kRigidIpcChainNet8x8Bodies;
    }
    return kRigidIpcChainNet4x4Bodies;
}

int topology_scaled_contact_capacity(const PabdCudaMesh& mesh,
                                     int floor_capacity) {
    const int by_vertices =
        static_cast<int>(mesh.surface_maps.size()) * 8;
    const int by_triangles =
        static_cast<int>(mesh.surface_triangles.size()) * 8;
    const int by_edges =
        static_cast<int>(mesh.surface_edges.size()) * 4;
    return std::max({floor_capacity, by_vertices, by_triangles, by_edges});
}

class CudaPabdScene : public chysx::render::Scene {
public:
    CudaPabdScene(const char* scene_name, PabdSceneKind kind,
                  int initial_box_count = kPdAbdDefaultBoxCount,
                  float initial_box_lift = 0.0f)
        : name_(scene_name), kind_(kind),
          initial_box_count_(initial_box_count),
          initial_box_lift_(initial_box_lift) {}

    const char* name() const override { return name_; }

    void setup() override {
        if (!solver_) {
            solver_ = std::make_unique<PabdCudaSolver>();
            set_default_params();
        }
        rebuild_simulation();
    }

    void step(float /*dt*/) override {
        if (solver_) {
            solver_->step(params_.dt);
            ++frame_index_;
            if ((kind_ == PabdSceneKind::StackedBlocks ||
                 kind_ == PabdSceneKind::PdAbdBoxes ||
                 kind_ == PabdSceneKind::TetraEECross ||
                 is_torque_gear_scene(kind_) ||
                 is_rigid_ipc_link_scene(kind_)) &&
                (frame_index_ == 1 ||
                 frame_index_ % std::max(1, debug_interval_) == 0)) {
                print_pd_abd_stats();
            }
        }
    }

    void draw_meshes(std::vector<chysx::render::DrawMesh>& out) override {
        if (!solver_) return;
        if (gpu_surface_render_enabled()) {
            return;
        }
        if (kind_ == PabdSceneKind::PdAbdBoxes ||
            kind_ == PabdSceneKind::StackedBlocks) {
            const int num_bodies = stack_count_;
            for (int body = 0; body < num_bodies; ++body) {
                float rgb[3];
                body_color(body, num_bodies, rgb);
                out.push_back({
                    solver_->flat_positions().data(),
                    solver_->num_surface_vertices(),
                    solver_->flat_triangles().data() +
                        body * kPdAbdBoxSurfaceTris * 3,
                    kPdAbdBoxSurfaceTris,
                    rgb[0], rgb[1], rgb[2],
                    false
                });
            }
            return;
        }
        if (kind_ == PabdSceneKind::TwentyTets) {
            const int num_bodies = solver_->num_tets();
            for (int body = 0; body < num_bodies; ++body) {
                float rgb[3];
                body_color(body, num_bodies, rgb);
                out.push_back({
                    solver_->flat_positions().data(),
                    solver_->num_vertices(),
                    solver_->flat_triangles().data() + body * 12,
                    4,
                    rgb[0], rgb[1], rgb[2],
                    false
                });
            }
            return;
        }
        if (kind_ == PabdSceneKind::TetraEECross) {
            constexpr int kTetraSurfaceTris = 4;
            for (int body = 0; body < 2; ++body) {
                float rgb[3];
                body_color(body, 2, rgb);
                out.push_back({
                    solver_->flat_positions().data(),
                    solver_->num_surface_vertices(),
                    solver_->flat_triangles().data() +
                        body * kTetraSurfaceTris * 3,
                    kTetraSurfaceTris,
                    rgb[0], rgb[1], rgb[2],
                    false
                });
            }
            return;
        }
        if (is_rigid_ipc_link_scene(kind_) ||
            is_torque_gear_scene(kind_)) {
            out.push_back({
                solver_->flat_positions().data(),
                solver_->num_surface_vertices(),
                solver_->flat_triangles().data(),
                solver_->num_surface_triangles(),
                color_[0], color_[1], color_[2],
                false
            });
            return;
        }
        out.push_back({
            solver_->flat_positions().data(),
            solver_->num_vertices(),
            solver_->flat_triangles().data(),
            solver_->num_surface_triangles(),
            color_[0], color_[1], color_[2],
            false
        });
    }

    void draw_custom() override {
        if (!solver_ || !gpu_surface_render_enabled()) return;
        if (!gpu_renderer_) {
            gpu_renderer_ = std::make_unique<PabdCudaSurfaceRenderer>();
        }
        gpu_renderer_->update(
            solver_->interpolated_surface_positions_device(),
            solver_->surface_triangles_device(),
            solver_->num_surface_triangles());

        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glEnable(GL_COLOR_MATERIAL);
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
        gpu_renderer_->draw(color_[0], color_[1], color_[2]);
        glDisable(GL_LIGHTING);
    }

    bool metrics(chysx::render::SceneMetrics& out) override {
        if (!solver_) return false;
        if (kind_ == PabdSceneKind::TwentyTets) {
            out.bodies = solver_->num_tets();
        } else if (kind_ == PabdSceneKind::TetraEECross) {
            out.bodies = 2;
        } else if (is_torque_gear_scene(kind_)) {
            out.bodies = torque_gear_body_count(kind_);
        } else if (is_rigid_ipc_chain_net(kind_)) {
            out.bodies = rigid_ipc_chain_net_body_count(kind_);
        } else if (kind_ == PabdSceneKind::RigidIpcVerticalChain) {
            out.bodies = stack_count_;
        } else if (kind_ == PabdSceneKind::PdAbdBoxes) {
            out.bodies = stack_count_;
        } else if (kind_ == PabdSceneKind::StackedBlocks) {
            out.bodies = stack_count_;
        } else {
            out.bodies = 1;
        }
        out.min_z = solver_->min_y();
        return true;
    }

    void ui() override {
        if (!solver_) return;

        if (ImGui::CollapsingHeader("PABD Parameters",
                                    ImGuiTreeNodeFlags_DefaultOpen)) {
            bool changed = false;
            bool topology_changed = false;
            if (kind_ == PabdSceneKind::StackedBlocks ||
                kind_ == PabdSceneKind::PdAbdBoxes ||
                kind_ == PabdSceneKind::RigidIpcVerticalChain) {
                const int prev_stack = stack_count_;
                const char* label =
                    kind_ == PabdSceneKind::PdAbdBoxes
                        ? "box count"
                        : (kind_ == PabdSceneKind::RigidIpcVerticalChain
                               ? "link count"
                               : "stack count");
                const int max_count =
                    kind_ == PabdSceneKind::RigidIpcVerticalChain ? 64 : 30;
                changed |= ImGui::SliderInt(label, &stack_count_, 1,
                                            max_count);
                if (stack_count_ != prev_stack) {
                    rebuild_simulation();
                }
            }

            changed |= ImGui::SliderFloat("dt", &params_.dt, 0.0001f, 0.05f,
                                          "%.4f");
            changed |= ImGui::SliderInt("substeps", &params_.substeps, 1, 20);
            changed |= ImGui::SliderInt("global iterations", &params_.iterations,
                                        1, 32);
            changed |= ImGui::SliderFloat("gravity", &params_.gravity, -50.0f,
                                          50.0f, "%.2f");
            changed |= ImGui::Checkbox("mass-normalized ARAP",
                                       &params_.use_arap_beta);
            if (params_.use_arap_beta) {
                changed |= ImGui::SliderFloat(
                    "ARAP beta", &params_.arap_beta,
                    0.0f, 128.0f, "%.3f");
            } else {
                changed |= ImGui::SliderFloat(
                    "stiffness", &params_.stiffness,
                    100.0f, 500000.0f, "%.0f",
                    ImGuiSliderFlags_Logarithmic);
            }
            int curvature_mode =
                static_cast<int>(params_.elastic_curvature);
            const char* curvature_modes[] = {
                "PD",
                "CoRotated Rest",
                "Polar GN",
                "Projected Newton 3x3",
            };
            if (ImGui::Combo("elastic curvature", &curvature_mode,
                             curvature_modes,
                             4)) {
                params_.elastic_curvature =
                    static_cast<PabdElasticCurvatureMode>(curvature_mode);
                changed = true;
            }
            int polar_gn_backend =
                static_cast<int>(params_.polar_gn_backend);
            const char* polar_gn_backends[] = {
                "Assembled 12x12",
                "Matrix-free rank 3",
            };
            if (ImGui::Combo("Polar GN backend", &polar_gn_backend,
                             polar_gn_backends, 2)) {
                params_.polar_gn_backend =
                    static_cast<PabdPolarGnBackend>(polar_gn_backend);
                changed = true;
            }
            changed |= ImGui::Checkbox(
                "elastic curvature diagnostics",
                &params_.elastic_curvature_diagnostics);
            changed |= ImGui::SliderFloat("density", &params_.density, 0.01f,
                                          100000.0f, "%.3f",
                                          ImGuiSliderFlags_Logarithmic);
            changed |= ImGui::SliderFloat("damping", &params_.damping, 0.80f,
                                          1.0f, "%.3f");
            changed |= ImGui::SliderFloat("fixed weight", &params_.fixed_weight,
                                          1.0e3f, 1.0e8f, "%.1e",
                                          ImGuiSliderFlags_Logarithmic);
            if (is_torque_gear_scene(kind_)) {
                changed |= ImGui::SliderFloat(
                    "hinge beta", &params_.hinge_beta,
                    0.0f, 1024.0f, "%.1f");
                changed |= ImGui::SliderFloat(
                    "motor torque", &params_.motor_torque,
                    -1000.0f, 1000.0f, "%.2f");
                changed |= ImGui::SliderFloat(
                    "motor damping", &params_.motor_damping,
                    0.0f, 100.0f, "%.2f");
            }
            ImGui::Separator();
            const bool pcg_selected =
                params_.global_solver == PabdGlobalSolverMode::PCG;
            if (ImGui::Button(pcg_selected ? "PCG active" : "Use PCG")) {
                params_.global_solver = PabdGlobalSolverMode::PCG;
                changed = true;
            }
            ImGui::SameLine();
            const bool jacobi_selected =
                params_.global_solver == PabdGlobalSolverMode::BlockJacobi12;
            if (ImGui::Button(jacobi_selected ? "Block-Jacobi12 active"
                                              : "Use Block-Jacobi12")) {
                params_.global_solver = PabdGlobalSolverMode::BlockJacobi12;
                changed = true;
            }
            changed |= ImGui::SliderInt("PCG iterations", &params_.pcg_iterations,
                                        1, 200);
            changed |= ImGui::Checkbox("PCG Body12 preconditioner",
                                       &params_.pcg_body_preconditioner);
            changed |= ImGui::Checkbox("PCG true residual diagnostics",
                                       &params_.pcg_true_residual_diagnostics);
            changed |= ImGui::SliderFloat("Block-Jacobi omega",
                                          &params_.block_jacobi_omega,
                                          0.05f, 1.0f, "%.2f");

            ImGui::Separator();
            changed |= ImGui::SliderFloat("ground y", &params_.ground_y, -2.0f,
                                          2.0f, "%.3f");
            changed |= ImGui::SliderFloat("contact gap", &params_.contact_gap,
                                          0.0f, 0.2f, "%.3f");
            changed |= ImGui::Checkbox("adaptive contact beta",
                                       &params_.use_contact_beta);
            if (params_.use_contact_beta) {
                changed |= ImGui::SliderFloat(
                    "ground contact beta", &params_.ground_contact_beta,
                    0.0f, 128.0f, "%.2f");
            } else {
                changed |= ImGui::SliderFloat("ground stiffness",
                                              &params_.ground_stiffness,
                                              0.0f, 1.0e5f, "%.1e");
            }
            changed |= ImGui::SliderFloat("ground friction",
                                          &params_.ground_friction,
                                          0.0f, 2.0f, "%.2f");
            changed |= ImGui::SliderFloat("self collision thickness",
                                          &params_.self_collision_thickness,
                                          0.0f, 0.2f, "%.3f");
            if (params_.use_contact_beta) {
                changed |= ImGui::SliderFloat(
                    "self collision beta", &params_.self_collision_beta,
                    0.0f, 128.0f, "%.2f");
                int contact_measure = static_cast<int>(
                    params_.self_contact_measure);
                const char* contact_measure_names[] = {
                    "effective mass", "contact tetra volume"};
                if (ImGui::Combo("self contact measure", &contact_measure,
                                 contact_measure_names,
                                 IM_ARRAYSIZE(contact_measure_names))) {
                    params_.self_contact_measure =
                        static_cast<PabdSelfContactMeasureMode>(
                            contact_measure);
                    changed = true;
                }
            } else {
                changed |= ImGui::SliderFloat(
                    "self collision stiffness",
                    &params_.self_collision_stiffness,
                    0.0f, 1.0e5f, "%.1e");
            }
            changed |= ImGui::SliderFloat(
                "PF(VF) stiffness / EE",
                &params_.point_face_stiffness_scale,
                0.0f, 10.0f, "%.2f");
            changed |= ImGui::Checkbox(
                "enable PF contacts", &params_.enable_point_face_contacts);
            changed |= ImGui::Checkbox(
                "enable EE contacts", &params_.enable_edge_edge_contacts);
            changed |= ImGui::SliderFloat(
                "self collision normal damping",
                &params_.self_collision_normal_damping,
                0.0f, 50.0f, "%.2f");
            changed |= ImGui::SliderFloat("self collision friction",
                                          &params_.self_collision_friction,
                                          0.0f, 2.0f, "%.2f");
            changed |= ImGui::SliderFloat(
                "friction epsilon", &params_.friction_epsilon,
                1.0e-5f, 2.0e-2f, "%.1e", ImGuiSliderFlags_Logarithmic);
            changed |= ImGui::SliderInt("mesh broadphase interval",
                                        &params_.mesh_broadphase_interval,
                                        1, 16);
            changed |= ImGui::SliderFloat("mesh broadphase skin",
                                          &params_.mesh_broadphase_skin,
                                          0.0f, 0.2f, "%.3f");

            if (topology_changed) {
                rebuild_simulation();
            } else if (changed) {
                solver_->set_params(params_);
            }

            if (ImGui::Button("Reset Simulation")) {
                rebuild_simulation();
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(keeps current parameters)");
        }

        ImGui::Separator();
        ImGui::Text("ground contacts: %d", solver_->last_ground_contacts());
        ImGui::Text("self contacts: %d", solver_->last_self_contacts());
        ImGui::Text("friction contacts: %d",
                    solver_->last_self_friction_contacts());
        if (is_torque_gear_scene(kind_)) {
            ImGui::Text("axis angular velocity: %.6f",
                        solver_->last_motor_axis_angular_velocity());
            ImGui::Text("hinge endpoint error: %.3e",
                        solver_->last_hinge_endpoint_error());
            if (kind_ != PabdSceneKind::SingleTorqueGear) {
                const auto& axis_w =
                    solver_->hinge_axis_angular_velocities();
                for (int body = 0;
                     body < torque_gear_body_count(kind_) &&
                     body < static_cast<int>(axis_w.size()); ++body) {
                    ImGui::Text("gear %d axis w: %.6f", body,
                                axis_w[body]);
                }
            }
        }
        ImGui::Text("min y: %.5f", solver_->min_y());
        ImGui::Text("max stretch error: %.3e",
                    solver_->last_max_stretch_error());
        ImGui::Text("max orthogonality error: %.3e",
                    solver_->last_max_orthogonality_error());
        ImGui::Text("max volume error: %.3e",
                    solver_->last_max_volume_error());
        ImGui::Text("max rotational curvature: %.3e",
                    solver_->last_max_rotational_curvature());
        ImGui::Text("matrix-free Polar GN: %s",
                    solver_->matrix_free_polar_gn_active() ? "active"
                                                           : "inactive");
        ImGui::Text("PCG iters (last): %d", solver_->last_pcg_iterations());
        ImGui::Text("PCG recursive <r,M^-1 r>: %.3e",
                    solver_->last_residual());
        ImGui::Text("PCG true relative residual: %.3e",
                    solver_->last_true_relative_residual());
        ImGui::Text("PCG preconditioner: %s",
                    solver_->pcg_body_preconditioner_active()
                        ? "Body12"
                        : "Point3");
        ImGui::Text("PCG 12x12 factor failures: %d",
                    solver_->last_pcg_body_preconditioner_failures());
        ImGui::Text("broadphase: %s, age %d, max disp %.5f, dropped %d",
                    solver_->last_mesh_broadphase_refreshed()
                        ? "refresh"
                        : "cached",
                    solver_->mesh_broadphase_cache_age(),
                    solver_->last_mesh_broadphase_max_displacement(),
                    solver_->last_mesh_broadphase_dropped_hits());
        ImGui::Text("BJ failures: %d", solver_->last_block_jacobi_failures());
        ImGui::Text("BJ contact ELL: width %d, overflow %d",
                    solver_->block_jacobi_contact_ell_width(),
                    solver_->last_block_jacobi_contact_ell_overflow());
    }

private:
    const char* name_;
    PabdSceneKind kind_;
    PabdCudaParams params_{};
    std::unique_ptr<PabdCudaSolver> solver_;
    std::unique_ptr<PabdCudaSurfaceRenderer> gpu_renderer_;
    float color_[3] = {0.72f, 0.52f, 0.96f};
    int stack_count_ = kDefaultStackCount;
    int initial_box_count_ = kPdAbdDefaultBoxCount;
    float initial_box_lift_ = 0.0f;
    int frame_index_ = 0;
    int debug_interval_ = 60;
    bool headless_ = false;

    void set_headless(bool headless) override {
        headless_ = headless;
        if (solver_) {
            solver_->set_auto_download_positions(!gpu_surface_render_enabled());
        }
    }

    bool gpu_surface_render_enabled() const {
        return !headless_ &&
               (kind_ == PabdSceneKind::PdAbdBoxes ||
                kind_ == PabdSceneKind::StackedBlocks ||
                kind_ == PabdSceneKind::TetraEECross ||
                is_torque_gear_scene(kind_) ||
                is_rigid_ipc_link_scene(kind_));
    }

    void rebuild_simulation() {
        if (kind_ == PabdSceneKind::StackedBlocks ||
            kind_ == PabdSceneKind::PdAbdBoxes) {
            params_.self_collision_max_contacts =
                std::max(512, stack_count_ * 96);
        } else if (is_rigid_ipc_chain_net(kind_)) {
            params_.self_collision_max_contacts =
                kind_ == PabdSceneKind::RigidIpcChainNet32x32
                    ? 262144
                    : (kind_ == PabdSceneKind::RigidIpcChainNet8x8
                           ? 65536
                           : 16384);
        }
        PabdCudaMesh mesh = make_mesh();
        if (kind_ == PabdSceneKind::RigidIpcVerticalChain ||
            is_tuned_rigid_ipc_chain_net(kind_)) {
            params_.self_collision_max_contacts =
                topology_scaled_contact_capacity(mesh, 4096);
        } else if (kind_ == PabdSceneKind::TorqueGearLine2 ||
                   kind_ == PabdSceneKind::TorqueGearLine6) {
            params_.self_collision_max_contacts =
                topology_scaled_contact_capacity(mesh, 4096);
        }
        solver_->set_auto_download_positions(!gpu_surface_render_enabled());
        solver_->setup(mesh, params_);
        if (gpu_renderer_) {
            gpu_renderer_->reset();
        }
        frame_index_ = 0;
        if (kind_ == PabdSceneKind::PdAbdBoxes ||
            kind_ == PabdSceneKind::StackedBlocks ||
            kind_ == PabdSceneKind::TetraEECross ||
            is_torque_gear_scene(kind_) ||
            is_rigid_ipc_link_scene(kind_)) {
            int body_count = stack_count_;
            if (kind_ == PabdSceneKind::TetraEECross) {
                body_count = 2;
            } else if (is_torque_gear_scene(kind_)) {
                body_count = torque_gear_body_count(kind_);
            } else if (is_rigid_ipc_chain_net(kind_)) {
                body_count = rigid_ipc_chain_net_body_count(kind_);
            } else if (kind_ == PabdSceneKind::RigidIpcVerticalChain) {
                body_count = stack_count_;
            }
            std::printf(
                "[PABD_DBG setup] bodies=%d solver=%s dt=%.6f substeps=%d iterations=%d "
                "pcg=%d omega=%.2f damping=%.3f arap=%s stiffness=%.1f arapBeta=%.3f curvature=%s curvatureBackend=%s mfActive=%d hingeBeta=%.1f motorTau=%.2f motorDamp=%.2f contact=%s groundBeta=%.2f groundK=%.1f groundMu=%.2f selfBeta=%.2f selfMeasure=%s features=%s pfScale=%.2f selfK=%.1f selfNormalDamp=%.2f selfMu=%.2f fricEps=%.1e gap=%.4f thickness=%.4f maxContacts=%d bpN=%d bpSkin=%.4f\n",
                body_count,
                params_.global_solver == PabdGlobalSolverMode::BlockJacobi12
                    ? "BlockJacobi12"
                    : "PCG",
                params_.dt, params_.substeps,
                params_.iterations, params_.pcg_iterations,
                params_.block_jacobi_omega,
                params_.damping,
                params_.use_arap_beta ? "beta" : "physical",
                params_.stiffness,
                params_.arap_beta,
                elastic_curvature_mode_name(params_.elastic_curvature),
                polar_gn_backend_name(params_.polar_gn_backend),
                solver_->matrix_free_polar_gn_active() ? 1 : 0,
                params_.hinge_beta,
                params_.motor_torque,
                params_.motor_damping,
                params_.use_contact_beta ? "beta" : "physical",
                params_.ground_contact_beta,
                params_.ground_stiffness,
                params_.ground_friction,
                params_.self_collision_beta,
                self_contact_measure_name(params_.self_contact_measure),
                self_contact_features_name(
                    params_.enable_point_face_contacts,
                    params_.enable_edge_edge_contacts),
                params_.point_face_stiffness_scale,
                params_.self_collision_stiffness,
                params_.self_collision_normal_damping,
                params_.self_collision_friction,
                params_.friction_epsilon, params_.contact_gap,
                params_.self_collision_thickness,
                params_.self_collision_max_contacts,
                params_.mesh_broadphase_interval,
                params_.mesh_broadphase_skin);
        }
    }

    void print_pd_abd_stats() const {
        const auto ns = solver_->last_self_normal_sum();
        std::printf("[PABD_DBG frame=%d] solver=%s pcgIters=%d pcgPrec=%s ground=%d self=%d raw=%d cap=%d overflow=%d pf=%d ee=%d activePf=%d activeEe=%d maxPen=%.6g rmsPen=%.6g meanK=%.6g pfK=%.6g eeK=%.6g pfV=%.6g eeV=%.6g friction=%d bpairs=%d bcap=%d boverflow=%d bpRefresh=%d bpAge=%d bpRefreshes=%d bpDisp=%.5f bpDropped=%d vertical=%d horizontal=%d nsum=(%.3f,%.3f,%.3f) minY=%.5f stretchErr=%.3e orthoErr=%.3e volumeErr=%.3e rotCurv=%.3e pcgRho=%.3e pcgTrueRel=%.3e pcgPrecFail=%d bjFail=%d ellWidth=%d ellOverflow=%d\n",
                    frame_index_,
                    params_.global_solver == PabdGlobalSolverMode::BlockJacobi12
                        ? "BlockJacobi12"
                        : "PCG",
                    solver_->last_pcg_iterations(),
                    solver_->pcg_body_preconditioner_active()
                        ? "Body12"
                        : "Point3",
                    solver_->last_ground_contacts(),
                    solver_->last_self_contacts(),
                    solver_->last_self_raw_contacts(),
                    solver_->interpolated_contact_capacity(),
                    solver_->last_self_contact_overflow() ? 1 : 0,
                    solver_->last_self_point_face_contacts(),
                    solver_->last_self_edge_edge_contacts(),
                    solver_->last_self_active_point_face_contacts(),
                    solver_->last_self_active_edge_edge_contacts(),
                    solver_->last_self_max_penetration(),
                    solver_->last_self_rms_penetration(),
                    solver_->last_self_mean_stiffness(),
                    solver_->last_self_point_face_mean_stiffness(),
                    solver_->last_self_edge_edge_mean_stiffness(),
                    solver_->last_self_point_face_mean_volume(),
                    solver_->last_self_edge_edge_mean_volume(),
                    solver_->last_self_friction_contacts(),
                    solver_->last_self_broadphase_pairs(),
                    solver_->last_self_broadphase_capacity(),
                    solver_->last_self_broadphase_overflow() ? 1 : 0,
                    solver_->last_mesh_broadphase_refreshed() ? 1 : 0,
                    solver_->mesh_broadphase_cache_age(),
                    solver_->mesh_broadphase_refresh_count(),
                    solver_->last_mesh_broadphase_max_displacement(),
                    solver_->last_mesh_broadphase_dropped_hits(),
                    solver_->last_self_vertical_contacts(),
                    solver_->last_self_horizontal_contacts(),
                    ns.x, ns.y, ns.z, solver_->min_y(),
                    solver_->last_max_stretch_error(),
                    solver_->last_max_orthogonality_error(),
                    solver_->last_max_volume_error(),
                    solver_->last_max_rotational_curvature(),
                    solver_->last_residual(),
                    solver_->last_true_relative_residual(),
                    solver_->last_pcg_body_preconditioner_failures(),
                    solver_->last_block_jacobi_failures(),
                    solver_->block_jacobi_contact_ell_width(),
                    solver_->last_block_jacobi_contact_ell_overflow());
        if (is_torque_gear_scene(kind_)) {
            const auto& axis_w =
                solver_->hinge_axis_angular_velocities();
            const auto& endpoint_errors =
                solver_->hinge_endpoint_errors();
            std::printf("[PABD_GEARS frame=%d] axisW=[", frame_index_);
            for (int body = 0; body < torque_gear_body_count(kind_);
                 ++body) {
                if (body > 0) std::printf(",");
                const float value =
                    body < static_cast<int>(axis_w.size())
                        ? axis_w[body]
                        : 0.0f;
                std::printf("%.7f", value);
            }
            std::printf("] endpointError=[");
            for (int body = 0; body < torque_gear_body_count(kind_);
                 ++body) {
                if (body > 0) std::printf(",");
                const float value =
                    body < static_cast<int>(endpoint_errors.size())
                        ? endpoint_errors[body]
                        : 0.0f;
                std::printf("%.3e", value);
            }
            std::printf("] torque=%.3f damping=%.3f\n",
                        params_.motor_torque,
                        params_.motor_damping);
            return;
        }
        if (gpu_surface_render_enabled()) {
            std::printf("[PABD_DBG frame=%d summary] gpuSurfaceMinY=%.5f surfaceVerts=%d surfaceTris=%d\n",
                        frame_index_, solver_->min_y(),
                        solver_->num_surface_vertices(),
                        solver_->num_surface_triangles());
            return;
        }
        const float* positions = solver_->flat_positions().data();
        const int* triangles = solver_->flat_triangles().data();
        float global_min_y = std::numeric_limits<float>::max();
        float global_max_y = -std::numeric_limits<float>::max();
        if (is_rigid_ipc_link_scene(kind_)) {
            const char* label = kind_ == PabdSceneKind::RigidIpcVerticalChain
                ? "verticalChain"
                : "chainNet";
            std::printf("[PABD_DBG frame=%d summary] %sMinY=%.5f surfaceVerts=%d surfaceTris=%d\n",
                        frame_index_, label, solver_->min_y(),
                        solver_->num_surface_vertices(),
                        solver_->num_surface_triangles());
            if (kind_ == PabdSceneKind::RigidIpcVerticalChain &&
                stack_count_ <= 8 && stack_count_ > 0) {
                const int tris_per_body =
                    solver_->num_surface_triangles() / stack_count_;
                for (int body = 0; body < stack_count_; ++body) {
                    float min_x = std::numeric_limits<float>::max();
                    float max_x = -std::numeric_limits<float>::max();
                    float min_y = std::numeric_limits<float>::max();
                    float max_y = -std::numeric_limits<float>::max();
                    float sum_x = 0.0f;
                    float sum_y = 0.0f;
                    int samples = 0;
                    const int tri_start = body * tris_per_body * 3;
                    for (int t = 0; t < tris_per_body * 3; ++t) {
                        const int id = triangles[tri_start + t];
                        const float x = positions[id * 3 + 0];
                        const float y = positions[id * 3 + 2];
                        min_x = std::min(min_x, x);
                        max_x = std::max(max_x, x);
                        min_y = std::min(min_y, y);
                        max_y = std::max(max_y, y);
                        sum_x += x;
                        sum_y += y;
                        ++samples;
                    }
                    std::printf(
                        "[PABD_DBG link] body=%d cx=%.4f cy=%.4f minY=%.4f maxY=%.4f widthX=%.4f\n",
                        body, sum_x / static_cast<float>(samples),
                        sum_y / static_cast<float>(samples), min_y, max_y,
                        max_x - min_x);
                }
            }
            return;
        }
        const int body_count = kind_ == PabdSceneKind::TetraEECross
            ? 2
            : stack_count_;
        const int tris_per_body = kind_ == PabdSceneKind::TetraEECross
            ? 4
            : kPdAbdBoxSurfaceTris;
        for (int body = 0; body < body_count; ++body) {
            float min_x = std::numeric_limits<float>::max();
            float max_x = -std::numeric_limits<float>::max();
            float min_y = std::numeric_limits<float>::max();
            float max_y = -std::numeric_limits<float>::max();
            float sum_x = 0.0f;
            float sum_y = 0.0f;
            int samples = 0;
            const int tri_start = body * tris_per_body * 3;
            for (int t = 0; t < tris_per_body * 3; ++t) {
                const int id = triangles[tri_start + t];
                const float x = positions[id * 3 + 0];
                const float y = positions[id * 3 + 2];
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
                min_y = std::min(min_y, y);
                max_y = std::max(max_y, y);
                sum_x += x;
                sum_y += y;
                ++samples;
            }
            global_min_y = std::min(global_min_y, min_y);
            global_max_y = std::max(global_max_y, max_y);
            if (body_count <= 5 || body == 0 || body == body_count - 1) {
                std::printf(
                    "[PABD_DBG body] body=%d cx=%.4f cy=%.4f minY=%.4f maxY=%.4f height=%.4f widthX=%.4f\n",
                    body, sum_x / static_cast<float>(samples),
                    sum_y / static_cast<float>(samples), min_y, max_y,
                    max_y - min_y, max_x - min_x);
            }
        }
        std::printf("[PABD_DBG frame=%d summary] stackMinY=%.5f stackMaxY=%.5f\n",
                    frame_index_, global_min_y, global_max_y);
    }

    PabdCudaMesh make_mesh() const {
        if (kind_ == PabdSceneKind::PdAbdBoxes) {
            return chysx::rigid::pabd_cuda::make_pd_abd_boxes_mesh(
                stack_count_, params_.ground_y, params_.contact_gap,
                0.42f, initial_box_lift_);
        }
        if (kind_ == PabdSceneKind::TetraEECross) {
            return chysx::rigid::pabd_cuda::make_pd_abd_tetra_edge_edge_mesh();
        }
        if (kind_ == PabdSceneKind::SingleTorqueGear) {
            return make_single_torque_driver_mesh();
        }
        if (kind_ == PabdSceneKind::TorqueGearLine2) {
            return make_torque_gear_line_mesh(2);
        }
        if (kind_ == PabdSceneKind::TorqueGearLine6) {
            return make_torque_gear_line_mesh(6);
        }
        if (kind_ == PabdSceneKind::RigidIpcChainNet4x4) {
            return make_rigid_ipc_chain_net_4x4_mesh();
        }
        if (kind_ == PabdSceneKind::RigidIpcChainNet8x8) {
            return make_rigid_ipc_chain_net_8x8_mesh();
        }
        if (kind_ == PabdSceneKind::RigidIpcChainNet16x16Ball) {
            return make_rigid_ipc_chain_net_16x16_ball_mesh();
        }
        if (kind_ == PabdSceneKind::RigidIpcChainNet32x32) {
            return make_rigid_ipc_chain_net_32x32_mesh();
        }
        if (kind_ == PabdSceneKind::RigidIpcVerticalChain) {
            return make_rigid_ipc_vertical_chain_mesh(stack_count_);
        }
        if (kind_ == PabdSceneKind::TwentyTets) {
            return chysx::rigid::pabd_cuda::make_twenty_tets_mesh();
        }
        if (kind_ == PabdSceneKind::StackedBlocks) {
            return chysx::rigid::pabd_cuda::make_stacked_blocks_mesh(
                stack_count_, 0.42f, params_.ground_y,
                params_.contact_gap);
        }
        if (kind_ == PabdSceneKind::HexDropGround) {
            return chysx::rigid::pabd_cuda::make_hex_drop_mesh();
        }
        return chysx::rigid::pabd_cuda::make_single_tetra_pinned_mesh();
    }

    void set_default_params() {
        params_.dt = 0.0033f;
        params_.substeps = 1;
        params_.iterations = 1;
        params_.gravity = -9.8f;
        params_.stiffness = 10000.0f;
        params_.use_arap_beta = true;
        params_.arap_beta = 64.0f;
        params_.density = 1.0f;
        params_.damping = 1.0f;
        params_.fixed_weight = 1.0e6f;
        params_.hinge_beta = 0.0f;
        params_.motor_torque = 0.0f;
        params_.motor_damping = 0.0f;
        params_.pcg_iterations = 50;
        params_.ground_y = 0.0f;
        params_.contact_gap = 0.035f;
        params_.use_contact_beta = true;
        params_.ground_contact_beta = 0.0f;
        params_.ground_stiffness = 0.0f;
        params_.self_collision_thickness = 0.0f;
        params_.self_collision_beta = 0.0f;
        params_.self_collision_stiffness = 0.0f;
        params_.self_collision_normal_damping = 0.0f;
        params_.self_collision_max_contacts = 256;
        params_.mesh_broadphase_interval = 1;
        params_.mesh_broadphase_skin = 0.0f;
        params_.global_solver = PabdGlobalSolverMode::PCG;
        params_.block_jacobi_omega = 1.0f;
        color_[0] = 0.72f;
        color_[1] = 0.52f;
        color_[2] = 0.96f;
        stack_count_ =
            kind_ == PabdSceneKind::PdAbdBoxes
                ? std::max(1, initial_box_count_)
                : (kind_ == PabdSceneKind::RigidIpcVerticalChain
                       ? std::max(2, initial_box_count_)
                       : kDefaultStackCount);

        if (kind_ == PabdSceneKind::HexDropGround) {
            params_.ground_contact_beta = 16.0f;
            params_.ground_stiffness = 1.0e7f;
            params_.pcg_iterations = 50;
            color_[0] = 0.50f;
            color_[1] = 0.72f;
            color_[2] = 0.95f;
        } else if (kind_ == PabdSceneKind::TwentyTets) {
            params_.iterations = 1;
            params_.damping = 1.0f;
            params_.fixed_weight = 1.0e6f;
            params_.pcg_iterations = 50;
            params_.ground_y = -0.72f;
            params_.contact_gap = 0.0f;
            params_.ground_contact_beta = 16.0f;
            params_.ground_stiffness = 8.0e7f;
            params_.self_collision_thickness = 0.01f;
            params_.self_collision_beta = 16.0f;
            params_.self_collision_stiffness = 1e4f;
            color_[0] = 0.78f;
            color_[1] = 0.62f;
            color_[2] = 0.35f;
        } else if (kind_ == PabdSceneKind::StackedBlocks ||
                   kind_ == PabdSceneKind::PdAbdBoxes) {
            params_.stiffness = 10000.0f;
            params_.ground_y = 0.0f;
            params_.contact_gap = 0.02f;
            params_.ground_contact_beta = 16.0f;
            params_.ground_stiffness = 1e4f;
            params_.self_collision_thickness = 0.03f;
            params_.self_collision_beta = 16.0f;
            params_.self_collision_stiffness = 1e4f;
            params_.ground_friction = 0.4f;
            params_.self_collision_friction = 0.3f;
            params_.self_collision_max_contacts =
                std::max(512, stack_count_ * 96);
            if (kind_ == PabdSceneKind::StackedBlocks) {
                color_[0] = 0.55f;
                color_[1] = 0.68f;
                color_[2] = 0.92f;
            } else {
                color_[0] = 0.18f;
                color_[1] = 0.58f;
                color_[2] = 0.95f;
            }
        } else if (kind_ == PabdSceneKind::TetraEECross) {
            params_.dt = 0.0033f;
            params_.substeps = 1;
            params_.iterations = 1;
            params_.gravity = -9.8f;
            params_.stiffness = 10000.0f;
            params_.density = 1.0f;
            params_.damping = 1.0f;
            params_.fixed_weight = 1.0e6f;
            params_.pcg_iterations = 50;
            params_.ground_y = -10.0f;
            params_.contact_gap = 0.03f;
            params_.ground_stiffness = 0.0f;
            params_.self_collision_thickness = 0.08f;
            params_.self_collision_beta = 16.0f;
            params_.self_collision_stiffness = 2.0e4f;
            params_.self_collision_friction = 0.2f;
            params_.self_collision_max_contacts = 128;
            params_.enable_point_face_contacts = false;
            params_.enable_edge_edge_contacts = true;
            color_[0] = 0.92f;
            color_[1] = 0.52f;
            color_[2] = 0.28f;
        } else if (is_torque_gear_scene(kind_)) {
            params_.dt = 0.0033f;
            params_.substeps = 1;
            params_.iterations = 1;
            params_.gravity = 0.0f;
            params_.stiffness = 1.5e4f;
            params_.use_arap_beta = true;
            params_.arap_beta = 64.0f;
            params_.density = 10.0f;
            params_.damping = 1.0f;
            params_.fixed_weight = 1.0e6f;
            params_.hinge_beta = 64.0f;
            params_.motor_torque =
                kind_ == PabdSceneKind::SingleTorqueGear ? -100.0f
                                                         : -10.0f;
            params_.motor_damping = 1.0f;
            params_.pcg_iterations = 50;
            params_.pcg_body_preconditioner = false;
            params_.ground_y = -10.0f;
            params_.contact_gap = 0.001f;
            params_.ground_stiffness = 0.0f;
            params_.self_collision_thickness =
                kind_ == PabdSceneKind::SingleTorqueGear ? 0.0f : 0.01f;
            params_.self_collision_beta =
                kind_ == PabdSceneKind::SingleTorqueGear ? 0.0f : 16.0f;
            params_.self_collision_stiffness =
                kind_ == PabdSceneKind::SingleTorqueGear ? 0.0f : 8.0e3f;
            params_.self_collision_normal_damping =
                kind_ == PabdSceneKind::SingleTorqueGear ? 0.0f : 10.0f;
            params_.self_collision_friction = 0.0f;
            params_.self_collision_max_contacts =
                kind_ == PabdSceneKind::SingleTorqueGear ? 64 : 4096;
            params_.global_solver = PabdGlobalSolverMode::PCG;
            color_[0] = 0.88f;
            color_[1] = 0.58f;
            color_[2] = 0.16f;
        } else if (is_rigid_ipc_chain_net(kind_)) {
            params_.dt = 0.0033f;
            params_.substeps = 1;
            params_.iterations = 1;
            params_.gravity = -9.8f;
            params_.stiffness = 40000.0f;
            params_.density = 1.0f;
            params_.fixed_weight = 1.0e6f;
            params_.pcg_iterations = 30;
            params_.ground_y =
                is_tuned_rigid_ipc_chain_net(kind_) ? -50.0f : -10.0f;
            params_.contact_gap =
                is_tuned_rigid_ipc_chain_net(kind_) ? 0.01f : 0.02f;
            params_.ground_stiffness = 0.0f;
            params_.self_collision_thickness =
                is_tuned_rigid_ipc_chain_net(kind_) ? 0.015f : 0.01f;
            params_.self_collision_beta = 16.0f;
            params_.self_collision_stiffness =
                is_tuned_rigid_ipc_chain_net(kind_) ? 3.0e4f : 5.0e2f;
            // These fixtures were tuned with fixed ARAP and contact weights.
            // Keep beta scaling as an explicit UI/environment experiment.
            params_.use_arap_beta = false;
            params_.use_contact_beta = false;
            params_.self_collision_friction = 0.2f;
            params_.self_collision_max_contacts =
                kind_ == PabdSceneKind::RigidIpcChainNet32x32
                    ? 262144
                    : (kind_ == PabdSceneKind::RigidIpcChainNet16x16Ball
                           ? 131072
                           : (kind_ == PabdSceneKind::RigidIpcChainNet8x8
                                  ? 65536
                                  : 16384));
            if (kind_ == PabdSceneKind::RigidIpcChainNet32x32) {
                params_.mesh_broadphase_interval = 3;
                params_.mesh_broadphase_skin = 0.010f;
            }
            color_[0] = 0.58f;
            color_[1] = 0.74f;
            color_[2] = 0.44f;
        } else if (kind_ == PabdSceneKind::RigidIpcVerticalChain) {
            params_.dt = 0.0033f;
            params_.substeps = 1;
            params_.iterations = 1;
            params_.global_solver = PabdGlobalSolverMode::BlockJacobi12;
            params_.block_jacobi_omega = 0.8f;
            params_.gravity = -9.8f;
            params_.stiffness = 10000.0f;
            params_.density = 1.0f;
            params_.damping = 1.0f;
            params_.fixed_weight = 1.0e6f;
            params_.pcg_iterations = 50;
            params_.ground_y = -50.0f;
            params_.contact_gap = 0.01f;
            params_.ground_stiffness = 0.0f;
            params_.self_collision_thickness = 0.015f;
            params_.self_collision_beta = 16.0f;
            params_.self_collision_stiffness = 1.0e4f;
            params_.self_collision_friction = 0.2f;
            params_.self_collision_max_contacts =
                std::max(4096, stack_count_ * 512);
            color_[0] = 0.74f;
            color_[1] = 0.62f;
            color_[2] = 0.36f;
        }

        if (const char* value = std::getenv("CHYSX_PABD_BP_INTERVAL")) {
            params_.mesh_broadphase_interval =
                std::max(1, std::atoi(value));
        }
        if (const char* value = std::getenv("CHYSX_PABD_DEBUG_INTERVAL")) {
            debug_interval_ = std::max(1, std::atoi(value));
        }
        if (const char* value = std::getenv("CHYSX_PABD_BP_SKIN")) {
            params_.mesh_broadphase_skin =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("CHYSX_PABD_DAMPING")) {
            params_.damping = std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("CHYSX_PABD_STIFFNESS")) {
            params_.stiffness =
                std::max(0.0f, std::strtof(value, nullptr));
            params_.use_arap_beta = false;
        }
        if (const char* value = std::getenv("CHYSX_PABD_ARAP_BETA")) {
            params_.arap_beta =
                std::max(0.0f, std::strtof(value, nullptr));
            params_.use_arap_beta = true;
        }
        if (const char* value =
                std::getenv("CHYSX_PABD_ARAP_MASS_NORMALIZED")) {
            params_.use_arap_beta = std::atoi(value) != 0;
        }
        if (const char* value = std::getenv("CHYSX_PABD_CONTACT_GAP")) {
            params_.contact_gap =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value =
                std::getenv("CHYSX_PABD_GROUND_STIFFNESS")) {
            params_.ground_stiffness =
                std::max(0.0f, std::strtof(value, nullptr));
            params_.use_contact_beta = false;
        }
        if (const char* value =
                std::getenv("CHYSX_PABD_SELF_THICKNESS")) {
            params_.self_collision_thickness =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value =
                std::getenv("CHYSX_PABD_SELF_STIFFNESS")) {
            params_.self_collision_stiffness =
                std::max(0.0f, std::strtof(value, nullptr));
            params_.use_contact_beta = false;
        }
        if (const char* value = std::getenv("CHYSX_PABD_CONTACT_BETA")) {
            const float beta =
                std::max(0.0f, std::strtof(value, nullptr));
            params_.ground_contact_beta = beta;
            params_.self_collision_beta = beta;
            params_.use_contact_beta = true;
        }
        if (const char* value =
                std::getenv("CHYSX_PABD_GROUND_BETA")) {
            params_.ground_contact_beta =
                std::max(0.0f, std::strtof(value, nullptr));
            params_.use_contact_beta = true;
        }
        if (const char* value = std::getenv("CHYSX_PABD_SELF_BETA")) {
            params_.self_collision_beta =
                std::max(0.0f, std::strtof(value, nullptr));
            params_.use_contact_beta = true;
        }
        if (const char* value =
                std::getenv("CHYSX_PABD_CONTACT_MASS_NORMALIZED")) {
            params_.use_contact_beta = std::atoi(value) != 0;
        }
        if (const char* value =
                std::getenv("CHYSX_PABD_SELF_CONTACT_MEASURE")) {
            const std::string measure(value);
            if (measure == "effective_mass" || measure == "effective" ||
                measure == "mass") {
                params_.self_contact_measure =
                    PabdSelfContactMeasureMode::EffectiveMass;
            } else if (measure == "tetra_volume" || measure == "tetra" ||
                       measure == "volume") {
                params_.self_contact_measure =
                    PabdSelfContactMeasureMode::TetrahedronVolume;
            }
        }
        if (const char* value =
                std::getenv("CHYSX_PABD_PF_STIFFNESS_SCALE")) {
            params_.point_face_stiffness_scale =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value =
                std::getenv("CHYSX_PABD_CONTACT_FEATURES")) {
            const std::string features(value);
            if (features == "pf" || features == "vf") {
                params_.enable_point_face_contacts = true;
                params_.enable_edge_edge_contacts = false;
            } else if (features == "ee") {
                params_.enable_point_face_contacts = false;
                params_.enable_edge_edge_contacts = true;
            } else if (features == "none") {
                params_.enable_point_face_contacts = false;
                params_.enable_edge_edge_contacts = false;
            } else if (features == "both" || features == "pf+ee" ||
                       features == "vf+ee") {
                params_.enable_point_face_contacts = true;
                params_.enable_edge_edge_contacts = true;
            }
        }
        if (const char* value =
                std::getenv("CHYSX_PABD_ELASTIC_CURVATURE")) {
            const std::string curvature(value);
            if (curvature == "pd") {
                params_.elastic_curvature =
                    PabdElasticCurvatureMode::ProjectiveDynamics;
            } else if (curvature == "rest" ||
                       curvature == "corotated_rest") {
                params_.elastic_curvature =
                    PabdElasticCurvatureMode::CorotatedRest;
            } else if (curvature == "gn" || curvature == "polar_gn") {
                params_.elastic_curvature =
                    PabdElasticCurvatureMode::PolarGaussNewton;
            } else if (curvature == "projected" ||
                       curvature == "projected_newton3") {
                params_.elastic_curvature =
                    PabdElasticCurvatureMode::ProjectedNewton3;
            }
        }
        if (const char* value =
                std::getenv("CHYSX_PABD_POLAR_GN_BACKEND")) {
            const std::string backend(value);
            if (backend == "assembled" || backend == "assembled12") {
                params_.polar_gn_backend =
                    PabdPolarGnBackend::Assembled12;
            } else if (backend == "matrix_free" ||
                       backend == "matrix_free_rank3" ||
                       backend == "rank3") {
                params_.polar_gn_backend =
                    PabdPolarGnBackend::MatrixFreeRank3;
            }
        }
        if (const char* value =
                std::getenv("CHYSX_PABD_CURVATURE_DIAGNOSTICS")) {
            params_.elastic_curvature_diagnostics = std::atoi(value) != 0;
        }
        if (const char* value = std::getenv("CHYSX_PABD_HINGE_BETA")) {
            params_.hinge_beta =
                std::max(0.0f, std::strtof(value, nullptr));
        } else if (const char* value =
                       std::getenv("CHYSX_PABD_HINGE_STIFFNESS")) {
            // Legacy scripts can still select the new dimensionless control.
            params_.hinge_beta =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("CHYSX_PABD_MOTOR_TORQUE")) {
            params_.motor_torque = std::strtof(value, nullptr);
        }
        if (const char* value = std::getenv("CHYSX_PABD_MOTOR_DAMPING")) {
            params_.motor_damping =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("CHYSX_PABD_BJ_OMEGA")) {
            params_.block_jacobi_omega =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("CHYSX_PABD_PCG_ITERS")) {
            params_.pcg_iterations = std::max(1, std::atoi(value));
        }
        if (const char* value = std::getenv("CHYSX_PABD_PCG_BODY12")) {
            params_.pcg_body_preconditioner = std::atoi(value) != 0;
        }
        if (const char* value = std::getenv("CHYSX_PABD_PCG_TRUE_RESIDUAL")) {
            params_.pcg_true_residual_diagnostics = std::atoi(value) != 0;
        }
        if (const char* value = std::getenv("CHYSX_PABD_GROUND_FRICTION")) {
            params_.ground_friction =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("CHYSX_PABD_SELF_FRICTION")) {
            params_.self_collision_friction =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value =
                std::getenv("CHYSX_PABD_SELF_NORMAL_DAMPING")) {
            params_.self_collision_normal_damping =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("CHYSX_PABD_FRICTION_EPSILON")) {
            params_.friction_epsilon =
                std::max(1.0e-8f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("CHYSX_PABD_SOLVER")) {
            const std::string solver(value);
            if (solver == "pcg" || solver == "PCG") {
                params_.global_solver = PabdGlobalSolverMode::PCG;
            } else if (solver == "block_jacobi" ||
                       solver == "BlockJacobi12") {
                params_.global_solver = PabdGlobalSolverMode::BlockJacobi12;
            }
        }
    }
};

}  // namespace

extern "C" void chysx_register_pabd_cuda_scenes() {
    chysx::render::register_scene(
        "CUDA PABD: Single Tetra Pinned",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: Single Tetra Pinned",
                                     PabdSceneKind::SingleTetraPinned);
        });
    chysx::render::register_scene(
        "CUDA PABD: Hex Drop Ground",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: Hex Drop Ground",
                                     PabdSceneKind::HexDropGround);
        });
    chysx::render::register_scene(
        "CUDA PABD: Twenty Tets Collision",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: Twenty Tets Collision",
                                     PabdSceneKind::TwentyTets);
        });
    chysx::render::register_scene(
        "CUDA PABD: Stacked Blocks",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: Stacked Blocks",
                                     PabdSceneKind::StackedBlocks);
        });
    chysx::render::register_scene(
        "CUDA PABD: PD+ABD Boxes",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: PD+ABD Boxes",
                                     PabdSceneKind::PdAbdBoxes);
        });
    chysx::render::register_scene(
        "CUDA PABD: PD+ABD Box Ground",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: PD+ABD Box Ground",
                                     PabdSceneKind::PdAbdBoxes, 1, 1.0f);
        });
    chysx::render::register_scene(
        "CUDA PABD: PD+ABD Stack10",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: PD+ABD Stack10",
                                     PabdSceneKind::PdAbdBoxes, 10);
        });
    chysx::render::register_scene(
        "CUDA PABD: Tetra EE Cross",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: Tetra EE Cross",
                                     PabdSceneKind::TetraEECross);
        });
    chysx::render::register_scene(
        "CUDA PABD: Single Torque Gear",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene(
                "CUDA PABD: Single Torque Gear",
                PabdSceneKind::SingleTorqueGear);
        });
    chysx::render::register_scene(
        "CUDA PABD: Torque Gear Line 2",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene(
                "CUDA PABD: Torque Gear Line 2",
                PabdSceneKind::TorqueGearLine2);
        });
    chysx::render::register_scene(
        "CUDA PABD: Torque Gear Line 6",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene(
                "CUDA PABD: Torque Gear Line 6",
                PabdSceneKind::TorqueGearLine6);
        });
    chysx::render::register_scene(
        "CUDA PABD: Rigid-IPC Chain Net 4x4",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: Rigid-IPC Chain Net 4x4",
                                     PabdSceneKind::RigidIpcChainNet4x4);
        });
    chysx::render::register_scene(
        "CUDA PABD: Rigid-IPC Chain Net 8x8",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: Rigid-IPC Chain Net 8x8",
                                     PabdSceneKind::RigidIpcChainNet8x8);
        });
    chysx::render::register_scene(
        "CUDA PABD: Rigid-IPC Chain Net 16x16 Ball",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene(
                "CUDA PABD: Rigid-IPC Chain Net 16x16 Ball",
                PabdSceneKind::RigidIpcChainNet16x16Ball);
        });
    chysx::render::register_scene(
        "CUDA PABD: Rigid-IPC Chain Net 32x32",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: Rigid-IPC Chain Net 32x32",
                                     PabdSceneKind::RigidIpcChainNet32x32);
        });
    chysx::render::register_scene(
        "CUDA PABD: Vertical Link Chain",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: Vertical Link Chain",
                                     PabdSceneKind::RigidIpcVerticalChain,
                                     kRigidIpcVerticalChainDefaultBodies);
        });
    chysx::render::register_scene(
        "CUDA PABD: Vertical Link Chain 2",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: Vertical Link Chain 2",
                                     PabdSceneKind::RigidIpcVerticalChain, 2);
        });
    chysx::render::register_scene(
        "CUDA PABD: Vertical Link Chain 4",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: Vertical Link Chain 4",
                                     PabdSceneKind::RigidIpcVerticalChain, 4);
        });
    chysx::render::register_scene(
        "CUDA PABD: Vertical Link Chain 8",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: Vertical Link Chain 8",
                                     PabdSceneKind::RigidIpcVerticalChain, 8);
        });
    chysx::render::register_scene(
        "CUDA PABD: Vertical Link Chain 10",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: Vertical Link Chain 10",
                                     PabdSceneKind::RigidIpcVerticalChain, 10);
        });
    chysx::render::register_scene(
        "CUDA PABD: Vertical Link Chain 16",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: Vertical Link Chain 16",
                                     PabdSceneKind::RigidIpcVerticalChain, 16);
        });
    chysx::render::register_scene(
        "CUDA PABD: Vertical Link Chain 14",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: Vertical Link Chain 14",
                                     PabdSceneKind::RigidIpcVerticalChain, 14);
        });
    chysx::render::register_scene(
        "CUDA PABD: Vertical Link Chain 24",
        []() -> chysx::render::Scene* {
            return new CudaPabdScene("CUDA PABD: Vertical Link Chain 24",
                                     PabdSceneKind::RigidIpcVerticalChain, 24);
        });
}
