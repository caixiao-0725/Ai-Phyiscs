// SPDX-License-Identifier: MIT
// Rigid-IPC demo scenes — bolt/nut from rigid-ipc SIGGRAPH 2021.

#include "render/scene.h"
#include "rigid/rigid_ipc/rigid_ipc_solver.h"
#include "io/obj_io.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#define MKDIR(p) mkdir(p, 0755)
#endif

using namespace chysx::rigid_ipc;
using namespace chysx::math;
using chysx::render::DrawMesh;
using chysx::render::Scene;
using chysx::render::SceneMetrics;
using chysx::render::register_scene;

namespace {

// ============================================================================
// Bolt scene — screw falls onto static nut under gravity
// Matches rigid-ipc fixtures/3D/mechanisms/bolt.json
// ============================================================================

class BoltScene : public Scene {
public:
    const char* name() const override { return "Rigid-IPC: Bolt"; }

    void setup() override {
        frame_ = 0;
        solver_ = {};

        // Parameters from bolt.json
        solver_.config.dt = 0.01f;
        solver_.config.gravity = Vec3f(0.0f, -9.81f, 0.0f);
        solver_.config.d_hat = 1e-4f;
        solver_.config.contact_kappa = 0.0f;  // auto: IPC adaptive formula
        solver_.config.default_friction = 0.0f;
        solver_.config.total_frames = 500;
        solver_.config.export_obj = true;
        solver_.config.output_dir = "output/rigid_ipc/";

        float density = 8050.0f;  // steel
        float scale = 0.01f;

        // Body 0: nut (static)
        std::string nut_path = chysx::render::get_data_path("meshes/screw-and-nut/nut-big.obj");
        if (!solver_.add_body_from_obj(nut_path, density,
                                        Vec3f(0, 0, 0), Vec3f(0, 0, 0),
                                        scale, true)) {
            printf("[BoltScene] ERROR: failed to load nut mesh: %s\n", nut_path.c_str());
            return;
        }

        // Body 1: screw (dynamic) — position [0,0,0] matches bolt.json
        std::string screw_path = chysx::render::get_data_path("meshes/screw-and-nut/screw-big.obj");
        if (!solver_.add_body_from_obj(screw_path, density,
                                        Vec3f(0, 0, 0), Vec3f(0, 0, 0),
                                        scale, false)) {
            printf("[BoltScene] ERROR: failed to load screw mesh: %s\n", screw_path.c_str());
            return;
        }

        rebuild_draw_buffers();
        mkdir_recursive("output/rigid_ipc");

        printf("=== Rigid-IPC: Bolt Scene ===\n");
        printf("Bodies: %d (0=nut[static], 1=screw[dynamic])\n", solver_.num_bodies());
        printf("Config: dt=%.4f, gravity=(%.1f,%.1f,%.1f)\n",
               solver_.config.dt,
               solver_.config.gravity.x, solver_.config.gravity.y,
               solver_.config.gravity.z);
        printf("  contact_kappa=%.2e, d_hat=%.2e, density=%.0f kg/m^3, scale=%.4f\n",
               solver_.config.contact_kappa, solver_.config.d_hat, density, scale);

        for (int b = 0; b < solver_.num_bodies(); ++b) {
            auto& body = solver_.body(b);
            Vec3f pos = pose_position(body.q);
            printf("Body %d: mass=%.6f kg, I=(%.4e, %.4e, %.4e), "
                   "pos=(%.6f, %.6f, %.6f), %s\n",
                   b, body.mass,
                   body.moment_of_inertia.x, body.moment_of_inertia.y,
                   body.moment_of_inertia.z,
                   pos.x, pos.y, pos.z,
                   body.is_fixed ? "STATIC" : "DYNAMIC");
        }
        printf("==============================\n");
    }

    void step(float /*dt*/) override {
        solver_.step();
        frame_++;

        auto info = solver_.last_step_info();
        if (solver_.num_bodies() > 1) {
            auto& screw = solver_.body(1);
            Vec3f pos = pose_position(screw.q);
            Vec3f vel = pose_position(screw.q_v);
            Vec3f theta = pose_rotation(screw.q);
            Vec3f omega = pose_rotation(screw.q_v);
            printf("[Rigid-IPC] frame %d: newton=%d contacts=%d res=%.4e "
                   "min_d=%.6e\n",
                   frame_, info.newton_iters, info.contact_count,
                   info.residual, info.min_distance);
            printf("  screw: pos=(%.6f, %.6f, %.6f) vel=(%.4f, %.4f, %.4f)\n",
                   pos.x, pos.y, pos.z, vel.x, vel.y, vel.z);
            printf("  screw: theta=(%.6f, %.6f, %.6f) omega=(%.4f, %.4f, %.4f)\n",
                   theta.x, theta.y, theta.z, omega.x, omega.y, omega.z);
        }

        // Export OBJ
        if (solver_.config.export_obj)
            export_frame();

        rebuild_draw_buffers();
    }

    void draw_meshes(std::vector<DrawMesh>& out) override {
        for (int b = 0; b < solver_.num_bodies(); ++b) {
            DrawMesh dm;
            dm.positions = draw_pos_[b].data();
            dm.n_points = static_cast<int>(draw_pos_[b].size()) / 3;
            dm.triangles = draw_tri_[b].data();
            dm.n_tris = static_cast<int>(draw_tri_[b].size()) / 3;

            if (solver_.body(b).is_fixed) {
                dm.color_r = 0.6f; dm.color_g = 0.6f; dm.color_b = 0.6f;
            } else {
                dm.color_r = 0.8f; dm.color_g = 0.4f; dm.color_b = 0.1f;
            }
            dm.wireframe = false;
            out.push_back(dm);
        }
    }

    bool metrics(SceneMetrics& out) override {
        out.bodies = solver_.num_bodies();
        if (solver_.num_bodies() > 1) {
            auto& screw = solver_.body(1);
            Vec3f v = pose_position(screw.q_v);
            Vec3f omega = pose_rotation(screw.q_v);
            out.max_speed = length(v);
            out.max_upward_speed = v.y > 0 ? v.y : 0;
            out.max_angular_speed = length(omega);
        }
        return true;
    }

private:
    RigidIPCSolver solver_;
    int frame_ = 0;

    std::vector<std::vector<float>> draw_pos_;
    std::vector<std::vector<int>> draw_tri_;

    void rebuild_draw_buffers() {
        int nb = solver_.num_bodies();
        draw_pos_.resize(nb);
        draw_tri_.resize(nb);

        for (int b = 0; b < nb; ++b) {
            auto verts = solver_.body_surface_verts(b);
            auto tris  = solver_.body_surface_tris(b);

            draw_pos_[b].resize(verts.size() * 3);
            for (size_t i = 0; i < verts.size(); ++i) {
                draw_pos_[b][i * 3]     = verts[i].x;
                draw_pos_[b][i * 3 + 1] = verts[i].y;
                draw_pos_[b][i * 3 + 2] = verts[i].z;
            }

            draw_tri_[b].resize(tris.size() * 3);
            for (size_t i = 0; i < tris.size(); ++i) {
                draw_tri_[b][i * 3]     = tris[i].x;
                draw_tri_[b][i * 3 + 1] = tris[i].y;
                draw_tri_[b][i * 3 + 2] = tris[i].z;
            }
        }
    }

    void export_frame() {
        for (int b = 0; b < solver_.num_bodies(); ++b) {
            auto verts = solver_.body_surface_verts(b);
            auto tris  = solver_.body_surface_tris(b);

            std::vector<float> pos(verts.size() * 3);
            for (size_t i = 0; i < verts.size(); ++i) {
                pos[i * 3]     = verts[i].x;
                pos[i * 3 + 1] = verts[i].y;
                pos[i * 3 + 2] = verts[i].z;
            }
            std::vector<int> tri(tris.size() * 3);
            for (size_t i = 0; i < tris.size(); ++i) {
                tri[i * 3]     = tris[i].x;
                tri[i * 3 + 1] = tris[i].y;
                tri[i * 3 + 2] = tris[i].z;
            }

            char path[256];
            snprintf(path, sizeof(path), "output/rigid_ipc/body%d_frame%04d.obj",
                     b, frame_);
            chysx::io::save_obj(path, pos.data(),
                                static_cast<int>(verts.size()),
                                tri.data(),
                                static_cast<int>(tris.size()));
        }
    }

    static void mkdir_recursive(const char* dir) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s", dir);
        for (char* p = buf + 1; *p; ++p) {
            if (*p == '/' || *p == '\\') {
                char c = *p;
                *p = '\0';
                MKDIR(buf);
                *p = c;
            }
        }
        MKDIR(buf);
    }
};

}  // namespace

extern "C" void chysx_register_rigid_ipc_scenes() {
    register_scene("Rigid-IPC: Bolt",
                   []() -> Scene* { return new BoltScene(); });
}
