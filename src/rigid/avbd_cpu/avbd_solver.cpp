// SPDX-License-Identifier: MIT
// AVBD solver main loop. Adapted from avbd-demo3d (Chris Giles, 2026).

#include "avbd_solver.h"
#include "avbd_broadphase_gpu.h"
#include "avbd_narrowphase_gpu.h"
#include "avbd_graph_coloring.h"
#include "avbd_gpu_solver.h"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

namespace chysx {
namespace avbd {

void SoaData::pack(Rigid* bodies) {
    count = 0;
    for (Rigid* b = bodies; b; b = b->next)
        count++;

    body_ptrs.resize(count);
    pos_x.resize(count); pos_y.resize(count); pos_z.resize(count);
    quat_x.resize(count); quat_y.resize(count); quat_z.resize(count); quat_w.resize(count);
    half_x.resize(count); half_y.resize(count); half_z.resize(count);
    radius.resize(count);
    mass.resize(count);
    friction.resize(count);
    moment_x.resize(count); moment_y.resize(count); moment_z.resize(count);
    vel_x.resize(count); vel_y.resize(count); vel_z.resize(count);
    velang_x.resize(count); velang_y.resize(count); velang_z.resize(count);
    prevvel_x.resize(count); prevvel_y.resize(count); prevvel_z.resize(count);

    int i = 0;
    for (Rigid* b = bodies; b; b = b->next, i++) {
        body_ptrs[i] = b;
        pos_x[i] = b->positionLin.x;
        pos_y[i] = b->positionLin.y;
        pos_z[i] = b->positionLin.z;
        quat_x[i] = b->positionAng.x;
        quat_y[i] = b->positionAng.y;
        quat_z[i] = b->positionAng.z;
        quat_w[i] = b->positionAng.w;
        half_x[i] = b->size.x * 0.5f;
        half_y[i] = b->size.y * 0.5f;
        half_z[i] = b->size.z * 0.5f;
        radius[i] = b->radius;
        mass[i] = b->mass;
        friction[i] = b->friction;
        moment_x[i] = b->moment.x;
        moment_y[i] = b->moment.y;
        moment_z[i] = b->moment.z;
        vel_x[i] = b->velocityLin.x;
        vel_y[i] = b->velocityLin.y;
        vel_z[i] = b->velocityLin.z;
        velang_x[i] = b->velocityAng.x;
        velang_y[i] = b->velocityAng.y;
        velang_z[i] = b->velocityAng.z;
        prevvel_x[i] = b->prevVelocityLin.x;
        prevvel_y[i] = b->prevVelocityLin.y;
        prevvel_z[i] = b->prevVelocityLin.z;
    }
}

void SoaData::unpack(Rigid* bodies) {
    int i = 0;
    for (Rigid* b = bodies; b; b = b->next, i++) {
        b->positionLin = float3{pos_x[i], pos_y[i], pos_z[i]};
        b->positionAng = quat{quat_x[i], quat_y[i], quat_z[i], quat_w[i]};
        b->velocityLin = float3{vel_x[i], vel_y[i], vel_z[i]};
        b->velocityAng = float3{velang_x[i], velang_y[i], velang_z[i]};
        b->prevVelocityLin = float3{prevvel_x[i], prevvel_y[i], prevvel_z[i]};
    }
}

Solver::Solver()
    : bodies(nullptr), forces(nullptr), broadphase_gpu_(nullptr) {
    defaultParams();
}

Solver::~Solver() {
    clear();
    delete broadphase_gpu_;
    delete narrowphase_gpu_;
    delete graph_coloring_gpu_;
    delete gpu_solver_;
}

Rigid* Solver::pick(float3 origin, float3 dir, float3& local) {
    const float epsilon = 1.0e-6f;
    float bestT = INFINITY;
    Rigid* bestBody = nullptr;
    float3 bestLocal = {0, 0, 0};

    for (Rigid* body = bodies; body != nullptr; body = body->next) {
        if (body->mass <= 0.0f)
            continue;

        quat invRot = conjugate(body->positionAng);
        float3 o = rotate(invRot, origin - body->positionLin);
        float3 d = rotate(invRot, dir);
        float3 half = body->size * 0.5f;

        float tEnter = 0.0f;
        float tExit = INFINITY;
        bool hit = true;

        for (int i = 0; i < 3; ++i) {
            if (std::fabs(d[i]) < epsilon) {
                if (o[i] < -half[i] || o[i] > half[i]) {
                    hit = false;
                    break;
                }
                continue;
            }
            float invD = 1.0f / d[i];
            float t0 = (-half[i] - o[i]) * invD;
            float t1 = (half[i] - o[i]) * invD;
            if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
            tEnter = max(tEnter, t0);
            tExit = min(tExit, t1);
            if (tEnter > tExit) { hit = false; break; }
        }

        if (!hit) continue;
        float tHit = tEnter >= 0.0f ? tEnter : tExit;
        if (tHit < 0.0f) continue;
        if (tHit < bestT) {
            bestT = tHit;
            bestBody = body;
            bestLocal = o + d * tHit;
        }
    }

    if (!bestBody) return nullptr;
    local = bestLocal;
    return bestBody;
}

void Solver::clear() {
    while (forces) delete forces;
    while (bodies) delete bodies;
}

void Solver::defaultParams() {
    dt = 1.0f / 60.0f;
    gravity = -10.0f;
    iterations = 10;
    betaLin = 10000.0f;
    betaAng = 100.0f;
    alpha = 0.95f;
    gamma = 0.999f;
}

void Solver::step() {
    // Pack linked list into SoA for GPU broadphase
    soa_.pack(bodies);

#ifdef AVBD_VALIDATE_BROADPHASE
    // CPU broadphase for validation: collect sorted pair set
    std::vector<std::pair<int,int>> cpu_pairs;
    for (int ia = 0; ia < soa_.count; ia++) {
        for (int ib = ia + 1; ib < soa_.count; ib++) {
            Rigid* bodyA = soa_.body_ptrs[ia];
            Rigid* bodyB = soa_.body_ptrs[ib];
            float3 dp = bodyA->positionLin - bodyB->positionLin;
            float r = bodyA->radius + bodyB->radius;
            if (dot(dp, dp) <= r * r)
                cpu_pairs.push_back({ia, ib});
        }
    }
#endif

    // GPU broadphase
    if (!broadphase_gpu_) {
        broadphase_gpu_ = new BroadphaseGPU();
        int max_pairs = soa_.count * (soa_.count - 1) / 2;
        if (max_pairs < 256) max_pairs = 256;
        broadphase_gpu_->build(soa_.count, max_pairs);
    }
    int max_out = soa_.count * (soa_.count - 1) / 2;
    if (max_out < 1) max_out = 1;
    pairs_a_.resize(max_out);
    pairs_b_.resize(max_out);

    int pair_count = broadphase_gpu_->query(
        soa_.pos_x.data(), soa_.pos_y.data(), soa_.pos_z.data(),
        soa_.quat_x.data(), soa_.quat_y.data(), soa_.quat_z.data(), soa_.quat_w.data(),
        soa_.half_x.data(), soa_.half_y.data(), soa_.half_z.data(),
        soa_.mass.data(), soa_.friction.data(),
        soa_.count, pairs_a_.data(), pairs_b_.data());

#ifdef AVBD_VALIDATE_BROADPHASE
    {
        // Validate: GPU pairs should be a superset of CPU sphere-test pairs
        // (GPU uses AABB which is looser than sphere).
        std::vector<std::pair<int,int>> gpu_pairs;
        for (int k = 0; k < pair_count; k++) {
            int a = pairs_a_[k], b = pairs_b_[k];
            if (a > b) std::swap(a, b);
            gpu_pairs.push_back({a, b});
        }
        std::sort(gpu_pairs.begin(), gpu_pairs.end());
        std::sort(cpu_pairs.begin(), cpu_pairs.end());

        int missing = 0;
        for (auto& cp : cpu_pairs) {
            if (!std::binary_search(gpu_pairs.begin(), gpu_pairs.end(), cp)) {
                fprintf(stderr, "AVBD VALIDATE: CPU pair (%d,%d) missing from GPU!\n",
                        cp.first, cp.second);
                missing++;
            }
        }
        static int frame_count = 0;
        if (frame_count % 60 == 0) {
            fprintf(stderr, "AVBD VALIDATE [frame %d]: bodies=%d cpu_pairs=%d gpu_pairs=%d missing=%d\n",
                    frame_count, soa_.count, (int)cpu_pairs.size(), pair_count, missing);
        }
        frame_count++;
    }
#endif

    // GPU narrowphase: run SAT on GPU for all broadphase pairs
    if (!narrowphase_gpu_)
        narrowphase_gpu_ = new NarrowphaseGPU();

    if (pair_count > 0) {
        int max_manifolds = pair_count;
        int max_contacts_total = pair_count * 8;
        std::vector<GpuManifold> gpu_manifolds(max_manifolds);
        std::vector<GpuContact> gpu_contacts(max_contacts_total);
        int total_contacts = 0;

        int n_manifolds = narrowphase_gpu_->query(
            broadphase_gpu_->pos_x_dev(), broadphase_gpu_->pos_y_dev(), broadphase_gpu_->pos_z_dev(),
            broadphase_gpu_->quat_x_dev(), broadphase_gpu_->quat_y_dev(),
            broadphase_gpu_->quat_z_dev(), broadphase_gpu_->quat_w_dev(),
            broadphase_gpu_->half_x_dev(), broadphase_gpu_->half_y_dev(), broadphase_gpu_->half_z_dev(),
            broadphase_gpu_->friction_dev(),
            broadphase_gpu_->pair_a_dev(), broadphase_gpu_->pair_b_dev(),
            pair_count, soa_.count,
            gpu_manifolds.data(), gpu_contacts.data(), total_contacts);

        // GPU warm-start: match contacts against previous frame on GPU,
        // then re-download contacts with warm-start data filled in.
        narrowphase_gpu_->warmstart(n_manifolds, total_contacts, soa_.count,
                                    gpu_contacts.data());

        // // Print vertex table periodically
        // {
        //     static int vtx_frame = 0;
        //     if (vtx_frame % 300 == 0) {
        //         const int* vc = narrowphase_gpu_->vertex_counts();
        //         const VertexEntry* vt = narrowphase_gpu_->vertex_table();
        //         int stride = narrowphase_gpu_->vertex_table_stride();
        //         int overflow = 0;
        //         fprintf(stderr, "VERTEX TABLE [frame %d]: bodies=%d manifolds=%d\n",
        //                 vtx_frame, soa_.count, n_manifolds);
        //         int print_limit = soa_.count < 20 ? soa_.count : 20;
        //         for (int b = 0; b < print_limit; b++) {
        //             int cnt = vc[b];
        //             int clamped = cnt < stride ? cnt : stride;
        //             if (cnt > stride) overflow++;
        //             fprintf(stderr, "  body %2d: %d neighbors [", b, cnt);
        //             for (int s = 0; s < clamped; s++) {
        //                 const VertexEntry& e = vt[b * stride + s];
        //                 if (s > 0) fprintf(stderr, ", ");
        //                 fprintf(stderr, "(other=%d, mfld=%d)", e.other_body, e.manifold_idx);
        //             }
        //             if (cnt > stride) fprintf(stderr, ", ... +%d overflow", cnt - stride);
        //             fprintf(stderr, "]\n");
        //         }
        //         if (soa_.count > print_limit)
        //             fprintf(stderr, "  ... (%d more bodies)\n", soa_.count - print_limit);
        //         if (overflow > 0)
        //             fprintf(stderr, "  WARNING: %d bodies exceeded %d neighbor slots\n",
        //                     overflow, stride);
        //     }
        //     vtx_frame++;
        // }

        // // GPU graph coloring comparison (runs all four algorithms periodically)
        // {
        //     static int coloring_frame = 0;
        //     if (n_manifolds > 0 && coloring_frame % 300 == 0) {
        //         if (!graph_coloring_gpu_)
        //             graph_coloring_gpu_ = new GraphColoringGPU();

        //         const int* vc_dev = narrowphase_gpu_->vtx_counts_dev();
        //         const VertexEntry* vt_dev = narrowphase_gpu_->vtx_table_dev();
        //         int stride = narrowphase_gpu_->vertex_table_stride();
        //         unsigned seed = (unsigned)coloring_frame;

        //         fprintf(stderr, "\n=== GRAPH COLORING [frame %d] bodies=%d manifolds=%d ===\n",
        //                 coloring_frame, soa_.count, n_manifolds);

        //         auto vivace = graph_coloring_gpu_->color_vivace(vc_dev, vt_dev, soa_.count, stride, seed);
        //         fprintf(stderr, "  Vivace:  colors=%d  rounds=%d  time=%.3f ms\n",
        //                 vivace.num_colors, vivace.num_rounds, vivace.elapsed_ms);

        //         auto luby = graph_coloring_gpu_->color_luby(vc_dev, vt_dev, soa_.count, stride, seed);
        //         fprintf(stderr, "  Luby:    colors=%d  rounds=%d  time=%.3f ms\n",
        //                 luby.num_colors, luby.num_rounds, luby.elapsed_ms);

        //         auto jp = graph_coloring_gpu_->color_jp(vc_dev, vt_dev, soa_.count, stride, seed);
        //         fprintf(stderr, "  JP:      colors=%d  rounds=%d  time=%.3f ms\n",
        //                 jp.num_colors, jp.num_rounds, jp.elapsed_ms);

        //         auto ldf = graph_coloring_gpu_->color_ldf(vc_dev, vt_dev, soa_.count, stride);
        //         fprintf(stderr, "  LDF:     colors=%d  rounds=%d  time=%.3f ms\n",
        //                 ldf.num_colors, ldf.num_rounds, ldf.elapsed_ms);

        //         // Print coloring for first few bodies (JP result as example)
        //         const int* jp_colors = graph_coloring_gpu_->colors_cpu();
        //         int print_n = soa_.count < 20 ? soa_.count : 20;
        //         fprintf(stderr, "  JP colors: [");
        //         for (int i = 0; i < print_n; i++) {
        //             if (i > 0) fprintf(stderr, ", ");
        //             fprintf(stderr, "%d", jp_colors[i]);
        //         }
        //         if (soa_.count > print_n) fprintf(stderr, ", ...");
        //         fprintf(stderr, "]\n");

        //         // Validate: no two neighbors share the same color
        //         const int* vc = narrowphase_gpu_->vertex_counts();
        //         const VertexEntry* vt = narrowphase_gpu_->vertex_table();
        //         int violations = 0;
        //         for (int b = 0; b < soa_.count; b++) {
        //             int cnt = vc[b] < stride ? vc[b] : stride;
        //             for (int s = 0; s < cnt; s++) {
        //                 int nb = vt[b * stride + s].other_body;
        //                 if (nb >= 0 && nb < soa_.count && jp_colors[b] == jp_colors[nb])
        //                     violations++;
        //             }
        //         }
        //         fprintf(stderr, "  JP validation: %d coloring violations\n", violations / 2);
        //     }
        //     coloring_frame++;
        // }

#ifdef AVBD_VALIDATE_NARROWPHASE
        {
            int cpu_collide_count = 0;
            int gpu_collide_count = n_manifolds;

            // Build GPU result set: (min_idx, max_idx) -> num_contacts
            std::vector<std::pair<std::pair<int,int>, int>> gpu_set;
            for (int k = 0; k < n_manifolds; k++) {
                int a = gpu_manifolds[k].body_a, b = gpu_manifolds[k].body_b;
                if (a > b) std::swap(a, b);
                gpu_set.push_back({{a, b}, gpu_manifolds[k].num_contacts});
            }
            std::sort(gpu_set.begin(), gpu_set.end());

            // Run CPU narrowphase on the same pairs
            std::vector<std::pair<std::pair<int,int>, int>> cpu_set;
            for (int k = 0; k < pair_count; k++) {
                Rigid* bA = soa_.body_ptrs[pairs_a_[k]];
                Rigid* bB = soa_.body_ptrs[pairs_b_[k]];
                Manifold::Contact tmpC[8] = {};
                float3x3 tmpBasis{};
                int nc = Manifold::collide(bA, bB, tmpC, tmpBasis);
                if (nc > 0) {
                    cpu_collide_count++;
                    int a = pairs_a_[k], b = pairs_b_[k];
                    if (a > b) std::swap(a, b);
                    cpu_set.push_back({{a, b}, nc});
                }
            }
            std::sort(cpu_set.begin(), cpu_set.end());

            // Compare: find CPU collisions missing from GPU
            int missing = 0, mismatch_nc = 0;
            for (auto& cp : cpu_set) {
                auto it = std::lower_bound(gpu_set.begin(), gpu_set.end(), cp,
                    [](const auto& a, const auto& b) { return a.first < b.first; });
                if (it == gpu_set.end() || it->first != cp.first) {
                    if (missing < 10)
                        fprintf(stderr, "  NP VALIDATE: CPU pair (%d,%d) nc=%d MISSING from GPU\n",
                                cp.first.first, cp.first.second, cp.second);
                    missing++;
                } else if (it->second != cp.second) {
                    if (mismatch_nc < 10)
                        fprintf(stderr, "  NP VALIDATE: pair (%d,%d) cpu_nc=%d gpu_nc=%d\n",
                                cp.first.first, cp.first.second, cp.second, it->second);
                    mismatch_nc++;
                }
            }
            // Find GPU collisions not in CPU (extra)
            int extra = 0;
            for (auto& gp : gpu_set) {
                auto it = std::lower_bound(cpu_set.begin(), cpu_set.end(), gp,
                    [](const auto& a, const auto& b) { return a.first < b.first; });
                if (it == cpu_set.end() || it->first != gp.first)
                    extra++;
            }

            static int np_frame = 0;
            if (np_frame % 60 == 0 || missing > 0) {
                fprintf(stderr, "NP VALIDATE [frame %d]: pairs=%d cpu_collisions=%d gpu_manifolds=%d "
                        "missing=%d mismatch_nc=%d extra=%d\n",
                        np_frame, pair_count, cpu_collide_count, gpu_collide_count,
                        missing, mismatch_nc, extra);
            }
            np_frame++;
        }
#endif

        // Detect whether any Joint/Spring forces exist (excluding IgnoreCollision/Manifold)
        bool has_joint_spring = false;
        for (Force* f = forces; f != nullptr; f = f->next) {
            if (dynamic_cast<Joint*>(f) || dynamic_cast<Spring*>(f)) {
                has_joint_spring = true;
                break;
            }
        }

        // Create GPU solver if needed
        if (!gpu_solver_)
            gpu_solver_ = new GpuSolver();
        if (!graph_coloring_gpu_)
            graph_coloring_gpu_ = new GraphColoringGPU();

        // Upload body state to GPU solver
        gpu_solver_->upload_bodies(
            soa_.pos_x.data(), soa_.pos_y.data(), soa_.pos_z.data(),
            soa_.quat_x.data(), soa_.quat_y.data(), soa_.quat_z.data(), soa_.quat_w.data(),
            soa_.vel_x.data(), soa_.vel_y.data(), soa_.vel_z.data(),
            soa_.velang_x.data(), soa_.velang_y.data(), soa_.velang_z.data(),
            soa_.prevvel_x.data(), soa_.prevvel_y.data(), soa_.prevvel_z.data(),
            soa_.mass.data(), soa_.moment_x.data(), soa_.moment_y.data(), soa_.moment_z.data(),
            soa_.half_x.data(), soa_.half_y.data(), soa_.half_z.data(),
            soa_.friction.data(),
            soa_.count);

        // Graph coloring for Gauss-Seidel parallelization
        const int* vtx_counts_dev = narrowphase_gpu_->vtx_counts_dev();
        const VertexEntry* vtx_table_dev = narrowphase_gpu_->vtx_table_dev();
        int vtx_stride = narrowphase_gpu_->vertex_table_stride();

        auto coloring = graph_coloring_gpu_->color_jp(
            vtx_counts_dev, vtx_table_dev, soa_.count, vtx_stride);
        int num_colors = coloring.num_colors;
        const int* colors_dev = graph_coloring_gpu_->colors_gpu();

        // GPU solver: init bodies, colored GS iterations, velocity update
        gpu_solver_->solve(
            narrowphase_gpu_->manifolds_dev(),
            narrowphase_gpu_->contacts_dev(),
            n_manifolds,
            vtx_counts_dev, vtx_table_dev, vtx_stride,
            colors_dev, num_colors,
            iterations, dt, gravity,
            alpha, betaLin, gamma);

        // Download results back to SoA
        gpu_solver_->download_positions(
            soa_.pos_x.data(), soa_.pos_y.data(), soa_.pos_z.data(),
            soa_.quat_x.data(), soa_.quat_y.data(), soa_.quat_z.data(), soa_.quat_w.data(),
            soa_.vel_x.data(), soa_.vel_y.data(), soa_.vel_z.data(),
            soa_.velang_x.data(), soa_.velang_y.data(), soa_.velang_z.data(),
            soa_.count);

        // Write back to linked list
        soa_.unpack(bodies);

        // Snapshot contacts D2D for next frame's warm-start (no write-back needed)
        narrowphase_gpu_->snapshot_for_next_frame(n_manifolds, total_contacts, soa_.count);
    } else {
        // No collisions: still need to run body init + velocity update via CPU fallback
        for (Rigid* body = bodies; body != nullptr; body = body->next) {
            body->inertialLin = body->positionLin + body->velocityLin * dt;
            if (body->mass > 0)
                body->inertialLin += float3{0, 0, gravity} * (dt * dt);
            body->inertialAng = body->positionAng + body->velocityAng * dt;

            float3 accel = (body->velocityLin - body->prevVelocityLin) / dt;
            float accelExt = accel.z * sign(gravity);
            float accelWeight = clamp(accelExt / std::fabs(gravity), 0.0f, 1.0f);
            if (!std::isfinite(accelWeight))
                accelWeight = 0.0f;

            body->initialLin = body->positionLin;
            body->initialAng = body->positionAng;
            if (body->mass > 0) {
                body->positionLin = body->positionLin + body->velocityLin * dt +
                                    float3{0, 0, gravity} * (accelWeight * dt * dt);
                body->positionAng = body->positionAng + body->velocityAng * dt;
            }
        }

        // Initialize non-manifold forces (Joint/Spring)
        for (Force* force = forces; force != nullptr;) {
            if (!force->initialize()) {
                Force* next = force->next;
                delete force;
                force = next;
            } else {
                force = force->next;
            }
        }

        // CPU solver loop (no manifold forces, only joints/springs if any)
        for (int it = 0; it < iterations; it++) {
            for (Rigid* body = bodies; body != nullptr; body = body->next) {
                if (body->mass <= 0) continue;
                float3x3 MLin = diagonal(body->mass, body->mass, body->mass);
                float3x3 MAng = diagonal(body->moment.x, body->moment.y, body->moment.z);
                float3x3 lhsLin = MLin / (dt * dt);
                float3x3 lhsAng = MAng / (dt * dt);
                float3x3 lhsCross = float3x3{0, 0, 0, 0, 0, 0, 0, 0, 0};
                float3 rhsLin = MLin / (dt * dt) * (body->positionLin - body->inertialLin);
                float3 rhsAng = MAng / (dt * dt) * (body->positionAng - body->inertialAng);
                for (Force* force = body->forces; force != nullptr;
                     force = (force->bodyA == body) ? force->nextA : force->nextB)
                    force->updatePrimal(body, alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
                float3 dxLin, dxAng;
                solve(lhsLin, lhsAng, lhsCross, -rhsLin, -rhsAng, dxLin, dxAng);
                body->positionLin = body->positionLin + dxLin;
                body->positionAng = body->positionAng + dxAng;
            }
            for (Force* force = forces; force != nullptr; force = force->next)
                force->updateDual(alpha);
        }

        // Velocity update (BDF1)
        for (Rigid* body = bodies; body != nullptr; body = body->next) {
            body->prevVelocityLin = body->velocityLin;
            if (body->mass > 0) {
                body->velocityLin = (body->positionLin - body->initialLin) / dt;
                body->velocityAng = (body->positionAng - body->initialAng) / dt;
            }
        }
    }
}

}  // namespace avbd
}  // namespace chysx
