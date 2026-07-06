// SPDX-License-Identifier: Apache-2.0

#include <imgui.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "render/scene.h"
#include "rigid/pabd_cuda/pabd_cuda_solver.h"

namespace {

using chysx::rigid::pabd_cuda::PabdCudaMesh;
using chysx::rigid::pabd_cuda::PabdCudaParams;
using chysx::rigid::pabd_cuda::PabdCudaSolver;
using chysx::rigid::pabd_cuda::kDefaultStackCount;
using chysx::rigid::pabd_cuda::kHexBlockSurfaceTris;

enum class PabdSceneKind {
    SingleTetraPinned,
    HexDropGround,
    TwentyTets,
    StackedBlocks,
};

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

class CudaPabdScene : public chysx::render::Scene {
public:
    CudaPabdScene(const char* scene_name, PabdSceneKind kind)
        : name_(scene_name), kind_(kind) {}

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
        }
    }

    void draw_meshes(std::vector<chysx::render::DrawMesh>& out) override {
        if (!solver_) return;
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
        if (kind_ == PabdSceneKind::StackedBlocks) {
            const int num_bodies = stack_count_;
            for (int body = 0; body < num_bodies; ++body) {
                float rgb[3];
                body_color(body, num_bodies, rgb);
                out.push_back({
                    solver_->flat_positions().data(),
                    solver_->num_vertices(),
                    solver_->flat_triangles().data() + body * kHexBlockSurfaceTris * 3,
                    kHexBlockSurfaceTris,
                    rgb[0], rgb[1], rgb[2],
                    false
                });
            }
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

    bool metrics(chysx::render::SceneMetrics& out) override {
        if (!solver_) return false;
        if (kind_ == PabdSceneKind::TwentyTets) {
            out.bodies = solver_->num_tets();
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
            if (kind_ == PabdSceneKind::StackedBlocks) {
                const int prev_stack = stack_count_;
                changed |= ImGui::SliderInt("stack count", &stack_count_, 1, 30);
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
            changed |= ImGui::SliderFloat("stiffness", &params_.stiffness,
                                          100.0f, 500000.0f, "%.0f",
                                          ImGuiSliderFlags_Logarithmic);
            changed |= ImGui::SliderFloat("density", &params_.density, 0.01f,
                                          100000.0f, "%.3f",
                                          ImGuiSliderFlags_Logarithmic);
            changed |= ImGui::SliderFloat("damping", &params_.damping, 0.80f,
                                          1.0f, "%.3f");
            changed |= ImGui::SliderFloat("fixed weight", &params_.fixed_weight,
                                          1.0e3f, 1.0e8f, "%.1e",
                                          ImGuiSliderFlags_Logarithmic);
            changed |= ImGui::SliderInt("PCG iterations", &params_.pcg_iterations,
                                        1, 200);

            ImGui::Separator();
            changed |= ImGui::SliderFloat("ground y", &params_.ground_y, -2.0f,
                                          2.0f, "%.3f");
            changed |= ImGui::SliderFloat("contact gap", &params_.contact_gap,
                                          0.0f, 0.2f, "%.3f");
            changed |= ImGui::SliderFloat("ground stiffness",
                                          &params_.ground_stiffness,
                                          0.0f, 2.0e8f, "%.1e");
            changed |= ImGui::SliderFloat("self collision thickness",
                                          &params_.self_collision_thickness,
                                          0.0f, 0.2f, "%.3f");
            changed |= ImGui::SliderFloat("self collision stiffness",
                                          &params_.self_collision_stiffness,
                                          0.0f, 5.0e7f, "%.1e");

            if (changed) {
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
        ImGui::Text("min y: %.5f", solver_->min_y());
        ImGui::Text("PCG iters (last): %d", solver_->last_pcg_iterations());
        ImGui::Text("PCG residual: %.3e", solver_->last_residual());
    }

private:
    const char* name_;
    PabdSceneKind kind_;
    PabdCudaParams params_{};
    std::unique_ptr<PabdCudaSolver> solver_;
    float color_[3] = {0.72f, 0.52f, 0.96f};
    int stack_count_ = kDefaultStackCount;

    void rebuild_simulation() {
        if (kind_ == PabdSceneKind::StackedBlocks) {
            params_.self_collision_max_contacts =
                std::max(512, stack_count_ * 64);
        }
        solver_->setup(make_mesh(), params_);
    }

    PabdCudaMesh make_mesh() const {
        if (kind_ == PabdSceneKind::TwentyTets) {
            return chysx::rigid::pabd_cuda::make_twenty_tets_mesh();
        }
        if (kind_ == PabdSceneKind::StackedBlocks) {
            return chysx::rigid::pabd_cuda::make_stacked_blocks_mesh(
                stack_count_, 0.42f, params_.ground_y, params_.contact_gap);
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
        params_.density = 1.0f;
        params_.damping = 1.0f;
        params_.fixed_weight = 1.0e6f;
        params_.pcg_iterations = 50;
        params_.ground_y = 0.0f;
        params_.contact_gap = 0.035f;
        params_.ground_stiffness = 0.0f;
        params_.self_collision_thickness = 0.0f;
        params_.self_collision_stiffness = 0.0f;
        params_.self_collision_max_contacts = 256;
        color_[0] = 0.72f;
        color_[1] = 0.52f;
        color_[2] = 0.96f;
        stack_count_ = kDefaultStackCount;

        if (kind_ == PabdSceneKind::HexDropGround) {
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
            params_.ground_stiffness = 8.0e7f;
            params_.self_collision_thickness = 0.01f;
            params_.self_collision_stiffness = 1e4f;
            color_[0] = 0.78f;
            color_[1] = 0.62f;
            color_[2] = 0.35f;
        } else if (kind_ == PabdSceneKind::StackedBlocks) {

            params_.iterations = 1;
            params_.damping = 1.0f;
            params_.fixed_weight = 1.0e6f;
            params_.pcg_iterations = 50;
            params_.ground_y = 0.0f;
            params_.contact_gap = 0.0f;
            params_.ground_stiffness = 1.0e5f;
            params_.self_collision_thickness = 0.01f;
            params_.self_collision_stiffness = 1e4f;
            params_.self_collision_max_contacts =
                std::max(512, stack_count_ * 64);
            color_[0] = 0.55f;
            color_[1] = 0.68f;
            color_[2] = 0.92f;
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
}
