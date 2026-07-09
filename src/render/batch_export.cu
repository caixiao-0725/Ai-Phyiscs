// SPDX-License-Identifier: Apache-2.0
//
// Batch export entry point — runs a scene headlessly and exports
// each frame as OBJ + bgeo.

#include <cstdio>
#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "io/bgeo_writer.h"
#include "io/obj_io.h"
#include "render/scene.h"

#ifndef ASSET_PATH
#define ASSET_PATH "./"
#endif

static constexpr int DEFAULT_FRAMES = 120;
static constexpr float DT = 1.0f / 60.0f;

int main(int argc, char* argv[]) {
    chysx::render::register_all_scenes();
    const auto& reg = chysx::render::scene_registry();
    if (reg.empty()) {
        std::cerr << "No scenes registered" << std::endl;
        return 1;
    }

    std::string target_scene;
    int N_FRAMES = DEFAULT_FRAMES;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--scene" && i + 1 < argc)
            target_scene = argv[++i];
        else if (std::string(argv[i]) == "--frames" && i + 1 < argc)
            N_FRAMES = std::atoi(argv[++i]);
    }

    chysx::render::Scene* scene = nullptr;
    if (!target_scene.empty()) {
        for (auto& entry : reg) {
            if (target_scene == entry.name) { scene = entry.create(); break; }
        }
    }
    if (!scene) scene = reg[0].create();
    scene->set_headless(true);
    try {
        scene->setup();
    } catch (const std::exception& e) {
        std::cerr << "Scene setup failed: " << e.what() << std::endl;
        delete scene;
        return 1;
    }

    std::string out_dir = std::string(ASSET_PATH) + "output/";
    std::cout << "=== ChysX Batch Export ===" << std::endl;
    std::cout << "Scene: " << scene->name() << std::endl;
    std::cout << "Frames: " << N_FRAMES << "  dt=" << DT << std::endl;

    bool has_metrics = false;
    chysx::render::SceneMetrics final_metrics;
    float max_speed_all = 0.0f;
    float max_upward_all = 0.0f;
    float max_angular_all = 0.0f;
    float max_pair_pen_all = 0.0f;
    float min_z_all = 0.0f;
    bool has_nan = false;

    for (int frame = 0; frame < N_FRAMES; ++frame) {
        scene->step(DT);

        chysx::render::SceneMetrics metrics;
        if (scene->metrics(metrics)) {
            if (!has_metrics) {
                min_z_all = metrics.min_z;
                has_metrics = true;
            }
            final_metrics = metrics;
            max_speed_all = std::max(max_speed_all, metrics.max_speed);
            max_upward_all = std::max(max_upward_all, metrics.max_upward_speed);
            max_angular_all = std::max(max_angular_all, metrics.max_angular_speed);
            max_pair_pen_all = std::max(max_pair_pen_all, metrics.max_pair_penetration);
            min_z_all = std::min(min_z_all, metrics.min_z);
            has_nan = has_nan || metrics.has_nan;

            if ((frame + 1) % 50 == 0 || frame == 0) {
                std::printf("[METRICS frame=%d] bodies=%d maxSpeed=%.4f maxUp=%.4f maxAng=%.4f minZ=%.4f maxPairPen=%.4f nan=%d\n",
                            frame + 1, metrics.bodies, metrics.max_speed,
                            metrics.max_upward_speed, metrics.max_angular_speed,
                            metrics.min_z, metrics.max_pair_penetration,
                            metrics.has_nan ? 1 : 0);
            }
        }

        std::vector<chysx::render::DrawMesh> meshes;
        scene->draw_meshes(meshes);

        // Build BgeoMeshPiece from DrawMesh
        std::vector<chysx::io::BgeoMeshPiece> pieces;
        for (const auto& m : meshes) {
            pieces.push_back({m.positions, m.n_points,
                              m.triangles, m.n_tris,
                              m.color_r, m.color_g, m.color_b});
        }

        char filename[256];
        std::snprintf(filename, sizeof(filename),
                      "%sframe_%04d.bgeo", out_dir.c_str(), frame);
        chysx::io::BgeoWriter::write_multi(filename, pieces);

        // Per-piece OBJ
        for (int p = 0; p < static_cast<int>(meshes.size()); ++p) {
            const auto& m = meshes[p];
            std::snprintf(filename, sizeof(filename),
                          "%spiece%d_%04d.obj", out_dir.c_str(), p, frame);
            chysx::io::save_obj(filename, m.positions, m.n_points,
                                m.triangles, m.n_tris);
        }

        if ((frame + 1) % 10 == 0 || frame == 0) {
            std::printf("Frame %d/%d\n", frame + 1, N_FRAMES);
        }
    }

    if (has_metrics) {
        std::printf("[METRICS summary] bodies=%d maxSpeed=%.4f maxUp=%.4f maxAng=%.4f minZ=%.4f maxPairPen=%.4f finalMaxSpeed=%.4f finalMinZ=%.4f finalPairPen=%.4f nan=%d\n",
                    final_metrics.bodies, max_speed_all, max_upward_all,
                    max_angular_all, min_z_all, max_pair_pen_all,
                    final_metrics.max_speed, final_metrics.min_z,
                    final_metrics.max_pair_penetration, has_nan ? 1 : 0);
    }

    delete scene;
    std::cout << "=== Done ===" << std::endl;
    return 0;
}
