// SPDX-License-Identifier: MIT
// AVBD solver main loop. Adapted from avbd-demo3d (Chris Giles, 2026).

#include "avbd_solver.h"
#include "avbd_broadphase_gpu.h"
#include "avbd_narrowphase_gpu.h"
#include "avbd_graph_coloring.h"

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

        // Save manifold data for post-solver write-back
        ws_n_manifolds_ = n_manifolds;
        ws_total_contacts_ = total_contacts;
        ws_manifolds_.assign(gpu_manifolds.begin(), gpu_manifolds.begin() + n_manifolds);

        for (int k = 0; k < n_manifolds; k++) {
            const GpuManifold& gm = gpu_manifolds[k];
            Rigid* bodyA = soa_.body_ptrs[gm.body_a];
            Rigid* bodyB = soa_.body_ptrs[gm.body_b];
            if (bodyA->constrainedTo(bodyB))
                continue;

            Manifold* m = new Manifold(this, bodyA, bodyB);
            m->gpu_num_contacts_ = gm.num_contacts;
            m->ws_manifold_idx_ = k;
            m->gpu_basis_ = float3x3{
                gm.basis[0], gm.basis[1], gm.basis[2],
                gm.basis[3], gm.basis[4], gm.basis[5],
                gm.basis[6], gm.basis[7], gm.basis[8]};

            for (int c = 0; c < gm.num_contacts; c++) {
                const GpuContact& gc = gpu_contacts[gm.contact_offset + c];
                Manifold::Contact& mc = m->gpu_new_contacts_[c];
                mc.feature.key = gc.feature_key;
                mc.rA = float3{gc.rA_x, gc.rA_y, gc.rA_z};
                mc.rB = float3{gc.rB_x, gc.rB_y, gc.rB_z};
                mc.C0 = float3{gc.C0_x, gc.C0_y, gc.C0_z};
                mc.penalty = float3{gc.penalty_x, gc.penalty_y, gc.penalty_z};
                mc.lambda = float3{gc.lambda_x, gc.lambda_y, gc.lambda_z};
                mc.stick = (gc.stick != 0);
            }
        }
    }

    // Initialize and warmstart forces
    for (Force* force = forces; force != nullptr;) {
        if (!force->initialize()) {
            Force* next = force->next;
            delete force;
            force = next;
        } else {
            force = force->next;
        }
    }

    // Initialize and warmstart bodies
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

    // Main solver loop
    for (int it = 0; it < iterations; it++) {
        // Primal update
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
                 force = (force->bodyA == body) ? force->nextA : force->nextB) {
                force->updatePrimal(body, alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
            }

            float3 dxLin, dxAng;
            solve(lhsLin, lhsAng, lhsCross, -rhsLin, -rhsAng, dxLin, dxAng);
            body->positionLin = body->positionLin + dxLin;
            body->positionAng = body->positionAng + dxAng;
        }

        // Dual update
        for (Force* force = forces; force != nullptr; force = force->next) {
            force->updateDual(alpha);
        }
    }

    // Velocity update (BDF1)
    for (Rigid* body = bodies; body != nullptr; body = body->next) {
        body->prevVelocityLin = body->velocityLin;
        if (body->mass > 0) {
            body->velocityLin = (body->positionLin - body->initialLin) / dt;
            body->velocityAng = (body->positionAng - body->initialAng) / dt;
        }
    }

    // Write back post-solver contact state to GPU and snapshot for next frame's warm-start.
    // Walk all Manifold forces and pack their contacts back into the GPU contact array.
    if (narrowphase_gpu_ && ws_n_manifolds_ > 0) {
        std::vector<GpuContact> wb_contacts(ws_total_contacts_);
        // Copy current contact data back from Manifold objects
        for (Force* force = forces; force != nullptr; force = force->next) {
            Manifold* m = dynamic_cast<Manifold*>(force);
            if (!m || m->ws_manifold_idx_ < 0) continue;
            int midx = m->ws_manifold_idx_;
            if (midx >= ws_n_manifolds_) continue;
            const GpuManifold& gm = ws_manifolds_[midx];
            for (int c = 0; c < m->numContacts && c < gm.num_contacts; c++) {
                int off = gm.contact_offset + c;
                if (off >= ws_total_contacts_) continue;
                GpuContact& gc = wb_contacts[off];
                const Manifold::Contact& mc = m->contacts[c];
                gc.feature_key = mc.feature.key;
                gc.rA_x = mc.rA.x; gc.rA_y = mc.rA.y; gc.rA_z = mc.rA.z;
                gc.rB_x = mc.rB.x; gc.rB_y = mc.rB.y; gc.rB_z = mc.rB.z;
                gc.lambda_x = mc.lambda.x; gc.lambda_y = mc.lambda.y; gc.lambda_z = mc.lambda.z;
                gc.penalty_x = mc.penalty.x; gc.penalty_y = mc.penalty.y; gc.penalty_z = mc.penalty.z;
                gc.C0_x = mc.C0.x; gc.C0_y = mc.C0.y; gc.C0_z = mc.C0.z;
                gc.stick = mc.stick ? 1 : 0;
            }
        }
        narrowphase_gpu_->upload_contacts(wb_contacts.data(), ws_total_contacts_);
        narrowphase_gpu_->snapshot_for_next_frame(ws_n_manifolds_, ws_total_contacts_, soa_.count);
        ws_n_manifolds_ = 0;
        ws_total_contacts_ = 0;
    }
}

}  // namespace avbd
}  // namespace chysx
