// SPDX-License-Identifier: Apache-2.0
// Shared OptiX launch parameters for the EF broadphase.
// Included by both host (.cpp) and device (.cu) translation units.

#pragma once

#include <optix_types.h>
#include <cuda_runtime.h>

namespace chysx {
namespace collision {
namespace optix_ef {

struct Params {
    const float3*  vertices;     // surface vertex positions
    const int2*    edges;        // surface edges (vertex index pairs)
    const int3*    triangles;    // surface triangles (vertex index triplets)
    float          thickness;    // AABB inflation for broadphase
    OptixTraversableHandle handle;

    // Output: per-edge hit list stored as a flat buffer + per-edge counts.
    // Layout: hits_buffer[edge_id * max_hits_per_edge + local_hit_idx] = face_id
    //         hit_counts[edge_id] = number of hits for this edge
    int*           hits_buffer;
    int*           hit_counts;
    int            max_hits_per_edge;
};

struct RayGenData {};
struct MissData {};
struct HitGroupData {};

}  // namespace optix_ef
}  // namespace collision
}  // namespace chysx
