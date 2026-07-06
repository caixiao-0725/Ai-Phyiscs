// SPDX-License-Identifier: MIT
// ABD IPC demo scenes — faithful reproduction of libuipc hello_affine_body.

#include "render/scene.h"
#include "rigid/abd_ipc/abd_ipc_solver.h"
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

using namespace chysx::abd_ipc;
using namespace chysx::math;
using chysx::render::DrawMesh;
using chysx::render::Scene;
using chysx::render::SceneMetrics;
using chysx::render::register_scene;

namespace {

// ============================================================================
// Regular tetrahedron — matches libuipc hello_affine_body exactly
// ============================================================================

// 4 vertices, 1 tet: same as libuipc's hello_affine_body
void make_regular_tet(chysx::geometry::TetMeshf& mesh) {
    mesh.resize_vertices(4);
    mesh.resize_tets(1);

    Vec3f* v = mesh.vertices().cpu_data();
    float s32 = std::sqrt(3.0f) / 2.0f;
    v[0] = Vec3f(0.0f, 1.0f, 0.0f);
    v[1] = Vec3f(0.0f, 0.0f, 1.0f);
    v[2] = Vec3f(-s32, 0.0f, -0.5f);
    v[3] = Vec3f( s32, 0.0f, -0.5f);

    Vec4i* t = mesh.tets().cpu_data();
    t[0] = Vec4i(0, 1, 2, 3);
}

// ============================================================================
// Helpers
// ============================================================================

// Load an OBJ, uniformly scale its vertices, and add it as an ABD body.
// (ABDSolver::add_body_from_obj has no scale parameter; the screw/nut meshes
// are authored at cm-scale and need scale=0.01 like the rigid-ipc bolt scene.)
inline bool add_scaled_obj(ABDSolver& s, const std::string& path,
                           float density, float kappa, float scale,
                           Vec3f translation, bool fixed) {
    chysx::io::ObjMesh obj;
    if (!chysx::io::load_obj(path, obj)) {
        printf("[ScrewNut] ERROR: failed to load %s\n", path.c_str());
        return false;
    }
    int nv = static_cast<int>(obj.positions.size()) / 3;
    int nt = static_cast<int>(obj.triangles.size()) / 3;

    chysx::geometry::TriangleMeshf mesh(nv, nt);
    auto* v = mesh.vertices().cpu_data();
    for (int i = 0; i < nv; ++i)
        v[i] = Vec3f(obj.positions[i*3]   * scale,
                     obj.positions[i*3+1] * scale,
                     obj.positions[i*3+2] * scale);
    auto* tr = mesh.triangles().cpu_data();
    for (int i = 0; i < nt; ++i)
        tr[i] = Vec3i(obj.triangles[i*3], obj.triangles[i*3+1], obj.triangles[i*3+2]);

    s.add_body(mesh, density, kappa, translation, fixed);
    return true;
}

// Extract an approximate rotation angle [rad] from the affine matrix A of q,
// via the trace of its polar-rotation part: angle = acos((tr(A)-1)/2).
// (A stays close to SO(3) thanks to the orthogonality potential, so tr(A) is a
// good proxy for how far the body has rotated from rest.)
inline float affine_rotation_angle(const Vec12f& q) {
    Mat3f A = q_to_A(q);
    float tr = A(0,0) + A(1,1) + A(2,2);
    float c = (tr - 1.0f) * 0.5f;
    c = c < -1.0f ? -1.0f : (c > 1.0f ? 1.0f : c);
    return std::acos(c);
}

// ============================================================================
// Screw and Nut scene — gravity-driven, ABD version of the rigid-ipc bolt.
// Same configuration as the rigid-ipc bolt scene (scale 0.01, steel density,
// dt=0.01, gravity, d_hat=1e-4) but bodies are affine (very stiff elastic).
// Used to compare ABD vs rigid-ipc convergence on a rotation-heavy contact.
// ============================================================================

class ScrewNutScene : public Scene {
public:
    const char* name() const override { return "ABD-IPC: Screw and Nut"; }

    void setup() override {
        frame_ = 0;
        solver_ = {};

        // Exact libuipc 8_screw_and_nut config: zero gravity, the screw is
        // driven by a rotating motor; frictionless threaded contact converts
        // the rotation into axial screwing.
        solver_.config.dt = 0.005f;                 // main.py dt
        solver_.config.gravity = Vec3f(0.0f, 0.0f, 0.0f);  // main.py gravity 0
        solver_.config.d_hat = 0.02f;               // main.py contact.d_hat
        solver_.config.contact_kappa = 1.0e9f;      // default_model(0, 1e9)
        solver_.config.adaptive_kappa = true;       // IPC soft-start ramp
        solver_.config.use_gpu_newton = true;       // GPU full-Hessian IPC (dense contact)
        solver_.config.default_kappa = 1.0e8f;      // abd.apply_to(..., 100 MPa)
        solver_.config.default_friction = 0.0f;     // friction disabled
        solver_.config.newton_max_iter = 1000;
        solver_.config.newton_min_iter = 1;
        solver_.config.velocity_tol = 1e-3f;
        solver_.config.line_search_max_iter = 8;    // main.py line_search.max_iter
        solver_.config.total_frames = 500;

        const float density = 8050.0f;  // steel
        const float scale = 0.05f;                  // main.py Transform.scale(0.05)
        const float kappa = 1.0e8f;

        // Body 0: nut (fixed) — main.py uses nut-big-2.obj
        std::string nut = chysx::render::get_data_path("meshes/screw-and-nut/nut-big-2.obj");
        add_scaled_obj(solver_, nut, density, kappa, scale, Vec3f(0, 0, 0), true);

        // Body 1: screw (dynamic) — driven by a rotating motor about Y.
        std::string screw = chysx::render::get_data_path("meshes/screw-and-nut/screw-big-2.obj");
        add_scaled_obj(solver_, screw, density, kappa, scale, Vec3f(0, 0, 0), false);

        // Rotating motor on the screw: axis Y, -pi rad/s, strength 100 (libuipc).
        solver_.set_motor(1, Vec3f(0.0f, 1.0f, 0.0f), -3.14159265f, 100.0f);

        rebuild_draw_buffers();
        mkdir_recursive("output/abd_ipc_screw");

        printf("=== ABD-IPC: Screw and Nut ===\n");
        printf("Bodies: %d (0=nut[fixed], 1=screw[dynamic])\n", solver_.num_bodies());
        printf("Config: dt=%.4f gravity=(%.2f,%.2f,%.2f) d_hat=%.2e\n",
               solver_.config.dt, solver_.config.gravity.x,
               solver_.config.gravity.y, solver_.config.gravity.z,
               solver_.config.d_hat);
        printf("  affine_kappa=%.2e Pa contact_kappa=%.2e Pa density=%.0f scale=%.4f\n",
               solver_.config.default_kappa, solver_.config.contact_kappa,
               density, scale);
        for (int b = 0; b < solver_.num_bodies(); ++b) {
            auto& body = solver_.body(b);
            Vec3f t = q_translation(body.q);
            printf("Body %d: mass=%.6f vol=%.8f pos=(%.6f,%.6f,%.6f) %s\n",
                   b, body.M(0,0), body.volume, t.x, t.y, t.z,
                   body.is_fixed ? "FIXED" : "DYNAMIC");
        }
        printf("==============================\n");
    }

    void step(float /*dt*/) override {
        solver_.step();
        frame_++;

        auto info = solver_.last_step_info();
        auto& screw = solver_.body(1);
        Vec3f t = q_translation(screw.q);
        Vec3f v = q_translation(screw.q_v);
        float ang = affine_rotation_angle(screw.q);
        float ortho = ortho_psi(screw.q, 1.0f);
        printf("[ABD-Screw] frame %d: newton=%d contacts=%d res=%.4e\n",
               frame_, info.newton_iters, info.contact_count, info.residual);
        printf("  screw: pos=(%.6f,%.6f,%.6f) vel=(%.4f,%.4f,%.4f) rot_angle=%.6f ortho=%.3e\n",
               t.x, t.y, t.z, v.x, v.y, v.z, ang, ortho);

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
            Vec3f v = q_translation(solver_.body(1).q_v);
            out.max_speed = chysx::math::length(v);
            out.max_upward_speed = v.y > 0 ? v.y : 0;
        }
        return true;
    }

private:
    ABDSolver solver_;
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

    static void mkdir_recursive(const char* dir) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s", dir);
        for (char* p = buf + 1; *p; ++p) {
            if (*p == '/' || *p == '\\') {
                char c = *p; *p = '\0'; MKDIR(buf); *p = c;
            }
        }
        MKDIR(buf);
    }
};

// ============================================================================
// Hello Affine Body scene
// ============================================================================

class HelloAffineBody : public Scene {
public:
    const char* name() const override { return "ABD-IPC: Hello Affine Body"; }

    void setup() override {
        frame_ = 0;
        solver_ = {};

        // Match libuipc hello_affine_body exactly
        solver_.config.dt = 0.01f;
        solver_.config.gravity = Vec3f(0, -9.8f, 0);
        solver_.config.default_kappa = 1.0e8f;   // 100 MPa
        solver_.config.d_hat = 0.01f;
        solver_.config.contact_kappa = 1.0e9f;    // 1 GPa
        solver_.config.default_friction = 0.5f;
        solver_.config.newton_max_iter = 100;
        solver_.config.newton_min_iter = 1;
        solver_.config.velocity_tol = 1e-3f;  // tighter convergence
        solver_.config.total_frames = 50;

        // Broadphase backend: QuantBvh (default) or OptiX (when available)
        // solver_.config.broadphase_type = ABDConfig::BroadphaseType::OptiX;

        // Body 0 (mesh2 in libuipc): fixed regular tetrahedron at origin
        chysx::geometry::TetMeshf tet_fixed;
        make_regular_tet(tet_fixed);
        solver_.add_body(tet_fixed, 1000.0f, 1e8f, Vec3f(0, 0, 0), true);

        // Body 1 (mesh1 in libuipc): dynamic tet shifted up by 1.3
        chysx::geometry::TetMeshf tet_dyn;
        make_regular_tet(tet_dyn);
        solver_.add_body(tet_dyn, 1000.0f, 1e8f, Vec3f(0, 1.3f, 0), false);

        rebuild_draw_buffers();
        mkdir_recursive("output/abd_ipc");

        // Print initial diagnostics for comparison
        printf("=== ABD-IPC Hello Affine Body ===\n");
        printf("Bodies: %d (0=fixed, 1=dynamic)\n", solver_.num_bodies());
        printf("Config: dt=%.4f, gravity=(%.1f,%.1f,%.1f)\n",
               solver_.config.dt,
               solver_.config.gravity.x, solver_.config.gravity.y,
               solver_.config.gravity.z);
        printf("  kappa=%.2e Pa, contact_kappa=%.2e Pa, d_hat=%.4f\n",
               solver_.config.default_kappa, solver_.config.contact_kappa,
               solver_.config.d_hat);

        // Print initial body states
        for (int b = 0; b < solver_.num_bodies(); ++b) {
            auto& body = solver_.body(b);
            Vec3f pos = q_translation(body.q);
            printf("Body %d: pos=(%.4f,%.4f,%.4f) fixed=%d mass_00=%.4f vol=%.6f\n",
                   b, pos.x, pos.y, pos.z, body.is_fixed, body.M(0,0), body.volume);
        }
        printf("=================================\n");
    }

    void step(float /*dt*/) override {
        solver_.step();
        frame_++;

        auto info = solver_.last_step_info();
        printf("[ABD-IPC] frame %d: newton_iters=%d, contacts=%d, residual=%.6e\n",
               frame_, info.newton_iters, info.contact_count, info.residual);

        // Diagnostic: print body 1 translation
        auto& body1 = solver_.body(1);
        Vec3f t = q_translation(body1.q);
        Vec3f v = q_translation(body1.q_v);
        printf("  body1: pos=(%.4f, %.4f, %.4f) vel=(%.4f, %.4f, %.4f)\n",
               t.x, t.y, t.z, v.x, v.y, v.z);

        // OrthoPotential diagnostic: check affine matrix deviation from SO(3)
        float ortho_err = ortho_psi(body1.q, 1.0f);
        printf("  body1: ortho_psi=%.6e\n", ortho_err);

        // Export OBJ
        //export_frame();

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
                dm.color_r = 0.5f; dm.color_g = 0.5f; dm.color_b = 0.5f;
            } else {
                dm.color_r = 0.2f; dm.color_g = 0.6f; dm.color_b = 0.9f;
            }
            dm.wireframe = false;
            out.push_back(dm);
        }
    }

    bool metrics(SceneMetrics& out) override {
        out.bodies = solver_.num_bodies();
        if (solver_.num_bodies() > 1) {
            Vec3f v = q_translation(solver_.body(1).q_v);
            out.max_speed = chysx::math::length(v);
            out.max_upward_speed = v.y > 0 ? v.y : 0;
        }
        return true;
    }

private:
    ABDSolver solver_;
    int frame_ = 0;

    // Flat draw buffers per body
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
            snprintf(path, sizeof(path), "output/abd_ipc/body%d_frame%04d.obj",
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

extern "C" void chysx_register_abd_ipc_scenes() {
    register_scene("ABD-IPC: Hello Affine Body",
                   []() -> Scene* { return new HelloAffineBody(); });
    register_scene("ABD-IPC: Screw and Nut",
                   []() -> Scene* { return new ScrewNutScene(); });
}
