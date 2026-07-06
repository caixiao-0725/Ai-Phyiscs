// SPDX-License-Identifier: Apache-2.0
//
// YarnSolver: main simulation driver for the Cosserat rod (yarn)
// simulation, ported from YarnBall.  Manages GPU buffers, collision
// detection via QuantBVH, and the simulation loop.

#pragma once

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../math/vec.cuh"
#include "../math/quat.cuh"
#include "../memory/cuda_array.h"
#include "../collision/bvh/lbvh.h"
#include "../geometry/polyline_mesh.h"
#include "yarn_types.cuh"

namespace chysx {
namespace yarn {

// Forward declarations for kernel launchers (in .cu files).
void launch_init_itr(YarnMetaData* d_meta, int n_verts, cudaStream_t stream);
void launch_end_itr(YarnMetaData* d_meta, int n_verts, cudaStream_t stream);
void launch_cosserat_itr(YarnMetaData* d_meta, int n_verts, bool use_step_limit, cudaStream_t stream);
void launch_quaternion_lambda_itr(YarnMetaData* d_meta, int n_verts, cudaStream_t stream);
void launch_build_aabbs(YarnMetaData* d_meta, int n_verts, cudaStream_t stream);
void launch_build_collision_list(YarnMetaData* d_meta, const math::Vec2i* pairs, int n_pairs, cudaStream_t stream);
void launch_recompute_step_limit(YarnMetaData* d_meta, int n_verts, cudaStream_t stream);
void launch_compute_aabb_centers(const collision::Aabb* bounds, math::Vec3f* centers, int n, cudaStream_t stream);

class YarnSolver {
public:
    explicit YarnSolver(int num_verts) {
        if (num_verts < 3) throw std::runtime_error("YarnSolver: too few vertices");

        meta_.numVerts = num_verts;
        meta_.gravity  = math::Vec3f(0.f, -9.8f, 0.f);
        meta_.h        = max_h_;
        meta_.lastH    = max_h_;
        meta_.drag     = 0.2f;
        meta_.damping  = 1e-6f;
        meta_.time     = 0.f;

        meta_.radius           = 1e-4f;
        meta_.barrierThickness = 8e-4f;
        meta_.accelerationRatio = 1.f;

        meta_.kCollision     = 1e-5f;
        meta_.detectionScaler = 1.2f;
        meta_.frictionCoeff  = 0.1f;
        meta_.kFriction      = 5.f;

        meta_.detectionPeriod    = 1;
        meta_.useStepSizeLimit   = 1;
        meta_.bvhRebuildPeriod   = 1.f / 10.f;
        meta_.numItr             = 8;

        h_verts_.resize(num_verts);
        h_vels_.resize(num_verts, math::Vec3f(0.f, 0.f, 0.f));
        h_qs_.resize(num_verts, math::quat_identity());
        h_qRests_.resize(num_verts, math::Vec4f(0.f, 0.f, 0.f, 5.f));
        initial_inv_masses_.resize(num_verts, 0.f);

        for (int i = 0; i < num_verts; ++i) {
            h_verts_[i].invMass = 1.f;
            h_verts_[i].lRest = 1.f;
            h_verts_[i].kStretch = 100.f;
            h_verts_[i].connectionIndex = -1;
            h_verts_[i].flags = static_cast<uint32_t>(VertexFlags::hasNext);
        }
        h_verts_[num_verts - 1].flags = 0;
    }

    ~YarnSolver() {
        if (step_graph_exec_) cudaGraphExecDestroy(step_graph_exec_);
        if (stream_) { cudaStreamSynchronize(stream_); cudaStreamDestroy(stream_); }
        if (d_meta_) cudaFree(d_meta_);
    }

    // --- Accessors (CPU-side) ---
    Vertex*             verts()    { return h_verts_.data(); }
    math::Quatf*        qs()      { return h_qs_.data(); }
    math::Vec4f*        qRests()  { return h_qRests_.data(); }
    math::Vec3f*        vels()    { return h_vels_.data(); }
    YarnMetaData&       meta()     { return meta_; }
    const YarnMetaData& meta() const { return meta_; }
    int                 num_verts() const { return meta_.numVerts; }
    float               max_h() const { return max_h_; }
    void                set_max_h(float h) { max_h_ = h; }

    // --- Configuration (called once after populating verts/qs/etc.) ---
    void configure(float density = 1e-3f) {
        const int nv = meta_.numVerts;
        meta_.maxSegLen = 0.f;
        meta_.minSegLen = FLT_MAX;

        math::Quatf lastQ = math::quat_identity();
        for (int i = 0; i < nv; ++i) {
            auto& v = h_verts_[i];

            // Fix flags
            if (i < nv - 1) {
                bool hn = has_flag(v.flags, VertexFlags::hasNext);
                h_verts_[i + 1].flags = set_flag(h_verts_[i + 1].flags, VertexFlags::hasPrev, hn);
                bool hnn = hn && has_flag(h_verts_[i + 1].flags, VertexFlags::hasNext);
                v.flags = set_flag(v.flags, VertexFlags::hasNextOrientation, hnn);
                if (!hn) v.flags |= static_cast<uint32_t>(VertexFlags::fixOrientation);
            }

            if (!has_flag(v.flags, VertexFlags::hasPrev) &&
                !has_flag(v.flags, VertexFlags::hasNext))
                throw std::runtime_error("YarnSolver: dangling vertex");

            v.lRest = 1.f / nv;

            float mass = 0.f;
            if (v.flags & static_cast<uint32_t>(VertexFlags::hasPrev))
                mass += h_verts_[i - 1].lRest;

            if (v.flags & static_cast<uint32_t>(VertexFlags::hasNext)) {
                auto& v1 = h_verts_[i + 1];
                math::Vec3f seg = v1.pos - v.pos;
                v.lRest = length(seg);
                if (v.lRest == 0.f) throw std::runtime_error("YarnSolver: zero-length segment");

                // Init orientation: minimize twist
                math::Vec3f dir = normalize(seg);
                math::Quatf qq  = quat_from_to(math::Vec3f(1.f, 0.f, 0.f), dir);
                math::Quatf rel = math::quat_multiply(math::quat_conjugate(qq), lastQ);
                // Keep only x and w components of relative rotation (remove y/z twist)
                math::Quatf t(rel.x, 0.f, 0.f, rel.w);
                float tn = length(t);
                if (tn > 1e-12f) t = t * (1.f / tn); else t = math::quat_identity();
                lastQ = math::quat_multiply(qq, t);

                mass += v.lRest;
                meta_.maxSegLen = std::max(meta_.maxSegLen, v.lRest);
                meta_.minSegLen = std::min(meta_.minSegLen, v.lRest);
            }
            h_qs_[i] = lastQ;

            mass *= 0.5f * density;
            v.invMass = (mass != 0.f) ? (v.invMass / mass) : 0.f;
            initial_inv_masses_[i] = v.invMass;
        }

        // Init rest Darboux vectors
        for (int i = 0; i < nv - 1; ++i) {
            math::Quatf rel = math::quat_multiply(math::quat_conjugate(h_qs_[i]), h_qs_[i + 1]);
            float scl = length(h_qRests_[i]);
            h_qRests_[i] = scl * rel;
        }

        // Allocate GPU memory
        alloc_gpu();
        upload_meta();
        upload();
    }

    void set_k_stretch(float k) {
        for (int i = 0; i < meta_.numVerts; ++i)
            h_verts_[i].kStretch = k * h_verts_[i].lRest;
    }

    void set_k_bend(float k) {
        k *= 4.f;
        for (int i = 0; i < meta_.numVerts; ++i) {
            math::Vec4f q = h_qRests_[i];
            float n = length(q);
            if (n > 1e-12f) q = q * (1.f / n);
            h_qRests_[i] = (k / h_verts_[i].lRest) * q;
        }
    }

    // Upload CPU → GPU.
    void upload() {
        d_verts_.copy_to_device();
        d_vels_.copy_to_device();
        d_qs_.copy_to_device();
        d_qRests_.copy_to_device();

        // Copy flags and connection indices for collision filtering
        copy_temp_data();
        cudaStreamSynchronize(stream_);
    }

    // Download GPU → CPU.
    void download() {
        d_verts_.copy_to_host();
        d_vels_.copy_to_host();
        d_qs_.copy_to_host();
        cudaStreamSynchronize(stream_);
        std::memcpy(h_verts_.data(), d_verts_.cpu_data(), sizeof(Vertex) * meta_.numVerts);
        std::memcpy(h_vels_.data(), d_vels_.cpu_data(), sizeof(math::Vec3f) * meta_.numVerts);
        std::memcpy(h_qs_.data(), d_qs_.cpu_data(), sizeof(math::Quatf) * meta_.numVerts);
    }

    // Advance simulation by dt (may take multiple sub-steps).
    float advance(float dt) {
        if (dt <= 0.f) return 0.f;

        int steps = std::max(1, static_cast<int>(std::ceil(dt / max_h_)));
        meta_.lastH = meta_.h;
        meta_.h = dt / steps;

        rebuild_cuda_graph();
        upload_meta();

        for (int s = 0; s < steps; ++s, ++step_counter_) {
            launch_init_itr(d_meta_, meta_.numVerts, stream_);

            if (meta_.detectionPeriod > 0 &&
                step_counter_ % meta_.detectionPeriod == 0)
                detect_collisions();

            cudaGraphLaunch(step_graph_exec_, stream_);
        }

        meta_.time += dt;
        return meta_.time;
    }

    // Single step at max_h.
    void step() { advance(max_h_); }

    // Export current positions as OBJ polylines.
    void export_obj(const std::string& path) {
        download();
        FILE* f = fopen(path.c_str(), "w");
        if (!f) { fprintf(stderr, "YarnSolver: cannot write %s\n", path.c_str()); return; }
        for (int i = 0; i < meta_.numVerts; ++i) {
            auto& p = h_verts_[i].pos;
            fprintf(f, "v %f %f %f\n", p.x, p.y, p.z);
        }
        // Write polylines as 'l' entries
        int start = 0;
        for (int i = 0; i < meta_.numVerts; ++i) {
            bool end = !(h_verts_[i].flags & static_cast<uint32_t>(VertexFlags::hasNext));
            if (end || i == meta_.numVerts - 1) {
                if (i - start >= 1) {
                    fprintf(f, "l");
                    for (int j = start; j <= i; ++j)
                        fprintf(f, " %d", j + 1);
                    fprintf(f, "\n");
                }
                start = i + 1;
            }
        }
        fclose(f);
    }

    // Access draw data for the viewer: flat positions array.
    const Vertex* gpu_verts() const { return d_verts_.gpu_data(); }
    int           gpu_verts_count() const { return meta_.numVerts; }

    size_t step_count() const { return step_counter_; }

private:
    float max_h_ = 1e-3f;
    size_t step_counter_ = 0;
    float last_bvh_rebuild_ = 1e30f;
    int last_graph_itr_ = -1;

    YarnMetaData meta_{};
    YarnMetaData* d_meta_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaGraphExec_t step_graph_exec_ = nullptr;

    // Host-side data
    std::vector<Vertex>      h_verts_;
    std::vector<math::Vec3f> h_vels_;
    std::vector<math::Quatf> h_qs_;
    std::vector<math::Vec4f> h_qRests_;
    std::vector<float>       initial_inv_masses_;

    // Device-side CudaArrays
    CudaArray<Vertex>      d_verts_;
    CudaArray<math::Quatf> d_qs_;
    CudaArray<math::Vec4f> d_qRests_;
    CudaArray<math::Vec3f> d_dx_;
    CudaArray<math::Vec3f> d_vels_;
    CudaArray<math::Vec3f> d_lastVels_;
    CudaArray<math::Vec3f> d_lastPos_;
    CudaArray<uint32_t>    d_lastFlags_;
    CudaArray<int>         d_lastCID_;

    CudaArray<int>   d_numCols_;
    CudaArray<float> d_maxStepSize_;
    CudaArray<float> d_paddingSize_;
    CudaArray<int>   d_collisions_;
    CudaArray<collision::Aabb> d_bounds_;
    CudaArray<math::Vec3f>     d_centers_;

    collision::LinearBvh bvh_;

    void alloc_gpu() {
        const int nv = meta_.numVerts;
        // Cap max_query_pairs to avoid OOM on large meshes
        const int max_pairs = std::min(nv * MAX_COLLISIONS_PER_SEGMENT,
                                       8 * 1024 * 1024);

        d_verts_.resize(nv);
        d_qs_.resize(nv);
        d_qRests_.resize(nv);
        d_dx_.resize(nv);
        d_vels_.resize(nv);
        d_lastVels_.resize(nv);
        d_lastPos_.resize(nv);
        d_lastFlags_.resize(nv);
        d_lastCID_.resize(nv);
        d_numCols_.resize(nv);
        d_maxStepSize_.resize(nv);
        d_paddingSize_.resize(nv);
        d_collisions_.resize(nv * MAX_COLLISIONS_PER_SEGMENT);
        d_bounds_.resize(nv);
        d_centers_.resize(nv);

        // Copy host data into CudaArrays
        std::memcpy(d_verts_.cpu_data(), h_verts_.data(), sizeof(Vertex) * nv);
        std::memcpy(d_vels_.cpu_data(), h_vels_.data(), sizeof(math::Vec3f) * nv);
        std::memcpy(d_qs_.cpu_data(), h_qs_.data(), sizeof(math::Quatf) * nv);
        std::memcpy(d_qRests_.cpu_data(), h_qRests_.data(), sizeof(math::Vec4f) * nv);
        std::memset(d_lastVels_.cpu_data(), 0, sizeof(math::Vec3f) * nv);

        // Zero device-side collision counts
        cudaMemset(d_numCols_.gpu_data(), 0, sizeof(int) * nv);

        // Build BVH
        bvh_.build(nv, max_pairs);

        cudaMalloc(&d_meta_, sizeof(YarnMetaData));
        cudaStreamCreate(&stream_);

        // Set meta pointers
        meta_.d_verts     = d_verts_.gpu_data();
        meta_.d_qs        = d_qs_.gpu_data();
        meta_.d_qRests    = d_qRests_.gpu_data();
        meta_.d_dx        = d_dx_.gpu_data();
        meta_.d_vels      = d_vels_.gpu_data();
        meta_.d_lastVels  = d_lastVels_.gpu_data();
        meta_.d_lastPos   = d_lastPos_.gpu_data();
        meta_.d_lastFlags = d_lastFlags_.gpu_data();
        meta_.d_lastCID   = d_lastCID_.gpu_data();
        meta_.d_numCols   = d_numCols_.gpu_data();
        meta_.d_maxStepSize = d_maxStepSize_.gpu_data();
        meta_.d_paddingSize = d_paddingSize_.gpu_data();
        meta_.d_collisions  = d_collisions_.gpu_data();
        meta_.d_bounds      = d_bounds_.gpu_data();
        meta_.d_boundColList = nullptr;  // we use QuantBVH pairs directly
    }

    void upload_meta() {
        meta_.detectionRadius = meta_.radius + 0.5f * meta_.barrierThickness;
        meta_.scaledDetectionRadius = meta_.detectionRadius * meta_.detectionScaler;
        cudaMemcpyAsync(d_meta_, &meta_, sizeof(YarnMetaData),
                        cudaMemcpyHostToDevice, stream_);
    }

    void copy_temp_data() {
        // Copy flags and connectionIndex from verts to flat arrays for GPU
        for (int i = 0; i < meta_.numVerts; ++i) {
            d_lastFlags_.cpu_data()[i] = h_verts_[i].flags;
            d_lastCID_.cpu_data()[i] = h_verts_[i].connectionIndex;
        }
        d_lastFlags_.copy_to_device();
        d_lastCID_.copy_to_device();
    }

    void detect_collisions() {
        const int nv = meta_.numVerts;
        auto stream_handle = reinterpret_cast<std::uintptr_t>(stream_);
        launch_build_aabbs(d_meta_, nv, stream_);

        if (last_bvh_rebuild_ >= meta_.bvhRebuildPeriod) {
            launch_compute_aabb_centers(d_bounds_.gpu_data(), d_centers_.gpu_data(),
                                         nv, stream_);
            bvh_.refit(d_bounds_.gpu_data(), d_centers_.gpu_data(), stream_handle);
            last_bvh_rebuild_ = 0.f;
        } else {
            bvh_.refit_only(d_bounds_.gpu_data(), stream_handle);
            last_bvh_rebuild_ += meta_.h * meta_.detectionPeriod;
        }

        bvh_.query_self_aabb(d_bounds_.gpu_data(), stream_handle);

        int n_pairs = 0;
        cudaMemcpyAsync(&n_pairs, bvh_.query_count_dev(), sizeof(int),
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        cudaMemsetAsync(d_numCols_.gpu_data(), 0, sizeof(int) * nv, stream_);
        launch_build_collision_list(d_meta_, bvh_.query_pairs_dev(), n_pairs, stream_);
    }

    void rebuild_cuda_graph() {
        if (meta_.numItr == last_graph_itr_) return;

        cudaStreamSynchronize(stream_);

        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

        if (meta_.useStepSizeLimit)
            launch_recompute_step_limit(d_meta_, meta_.numVerts, stream_);

        for (int i = 0; i < meta_.numItr; ++i) {
            launch_cosserat_itr(d_meta_, meta_.numVerts,
                                 meta_.useStepSizeLimit != 0, stream_);
            launch_quaternion_lambda_itr(d_meta_, meta_.numVerts, stream_);
        }

        launch_end_itr(d_meta_, meta_.numVerts, stream_);

        cudaGraph_t graph;
        cudaStreamEndCapture(stream_, &graph);

        if (step_graph_exec_) cudaGraphExecDestroy(step_graph_exec_);
        cudaGraphInstantiate(&step_graph_exec_, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph);

        last_graph_itr_ = meta_.numItr;
    }

    // Rotor::fromTo equivalent using quaternion
    static math::Quatf quat_from_to(math::Vec3f from, math::Vec3f to) {
        using namespace math;
        Vec3f h = from + to;
        float l2 = dot(h, h);
        if (l2 > 0.f) {
            h = h * (1.f / std::sqrt(l2));
        } else {
            // 180-degree rotation: pick orthogonal axis
            if (std::abs(from.z) < std::abs(from.x))
                h = normalize(Vec3f(-from.y, from.x, 0.f));
            else
                h = normalize(Vec3f(0.f, -from.z, from.y));
            return Quatf(h.x, h.y, h.z, 0.f);
        }
        Vec3f c = cross(from, h);
        float d = dot(from, h);
        return Quatf(c.x, c.y, c.z, d);
    }
};

}  // namespace yarn
}  // namespace chysx
