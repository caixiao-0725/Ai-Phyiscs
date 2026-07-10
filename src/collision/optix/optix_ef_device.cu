// SPDX-License-Identifier: Apache-2.0
// OptiX device programs for EF broadphase.
//
// Compiled to PTX offline, then embedded into host code via bin2c.
// One ray per surface edge (origin=v0, direction=v1-v0, t∈[0,1]).
// Custom AABB primitives (one per triangle, enlarged by thickness).
// Intersection shader filters covertex (edge shares vertex with face).

#include <optix.h>
#include "optix_ef_params.h"

using namespace chysx::collision::optix_ef;

extern "C" {
__constant__ Params params;
}

// ---- Miss (no-op) ----

extern "C" __global__ void __miss__ms() {}

// ---- Ray generation: one ray per edge ----

extern "C" __global__ void __raygen__ef() {
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int eid = idx.x;

    const int2 edge = params.edges[eid];
    const float3 v0 = params.vertices[edge.x];
    const float3 v1 = params.vertices[edge.y];
    const float3 d = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};

    // Reset hit count for this edge
    params.hit_counts[eid] = 0;

    optixTrace(
        params.handle,
        v0, d,
        0.0f, 1.0f,        // tmin, tmax: ray spans the edge
        0.0f,               // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        0, 0, 0             // SBT offset/stride/miss index
    );
}

// ---- Any-hit: reject intersection to continue traversal ----

extern "C" __global__ void __anyhit__ef() {
    optixIgnoreIntersection();
}

// ---- Intersection: custom AABB primitive (triangle face) ----

extern "C" __global__ void __intersection__ef() {
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int eid = idx.x;

    // Covertex filter: skip if edge shares a vertex with this face
    const int3 face = params.triangles[prim_idx];
    const int2 edge = params.edges[eid];
    if (face.x == edge.x || face.x == edge.y) return;
    if (face.y == edge.x || face.y == edge.y) return;
    if (face.z == edge.x || face.z == edge.y) return;

    if (params.vertex_mesh_ids != nullptr) {
        const int edge_mesh = params.vertex_mesh_ids[edge.x];
        const int face_mesh = params.vertex_mesh_ids[face.x];
        const bool same_mesh = edge_mesh == face_mesh;
        const unsigned int category_bit = same_mesh ? 1u : 2u;
        if ((params.collision_mask & category_bit) == 0u) return;
    }

    // Record hit
    int slot = atomicAdd(&params.hit_counts[eid], 1);
    if (slot < params.max_hits_per_edge) {
        params.hits_buffer[eid * params.max_hits_per_edge + slot] = prim_idx;
    }

    // Do not report an accepted intersection. Returning without a report
    // makes traversal continue to the next overlapping custom primitive,
    // avoiding an any-hit program transition for this broadphase-only query.
}
