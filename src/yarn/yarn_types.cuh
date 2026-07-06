// SPDX-License-Identifier: Apache-2.0
//
// Data types for the Yarn (Cosserat Rod) simulation, ported from
// YarnBall.  All GLM types replaced with ChysX math equivalents.

#pragma once

#include <cstdint>

#include "../math/common.cuh"
#include "../math/vec.cuh"
#include "../math/matrix.cuh"
#include "../math/quat.cuh"
#include "../collision/bvh/aabb.cuh"

namespace chysx {
namespace yarn {

using math::Vec2f;
using math::Vec2i;
using math::Vec3f;
using math::Vec4f;
using math::Mat3f;
using math::Quatf;

constexpr int MAX_COLLISIONS_PER_SEGMENT = 128;

enum class VertexFlags : uint32_t {
    hasPrev            = 1,
    hasNext            = 2,
    hasNextOrientation = 4,
    fixOrientation     = 8,
    colliding          = 16,
};

CHYSX_HDI bool has_flag(uint32_t flags, VertexFlags f) {
    return (flags & static_cast<uint32_t>(f)) != 0;
}

CHYSX_HDI uint32_t set_flag(uint32_t flags, VertexFlags f, bool state) {
    uint32_t bit = static_cast<uint32_t>(f);
    return state ? (flags | bit) : (flags & ~bit);
}

// Per-vertex simulation data (matches YarnBall::Vertex layout).
struct Vertex {
    Vec3f    pos;
    float    invMass;

    float    lRest;
    float    kStretch;
    int      connectionIndex;
    uint32_t flags;
};

// 3x3 symmetric Hessian stored as 6 floats: [diag(3), upper(3)].
// Layout: [d0, d1, d2, u01, u02, u12]
struct SymHess3 {
    union {
        float dat[6];
        struct {
            Vec3f diag;
            Vec3f upper;  // (01, 02, 12)
        };
    };

    CHYSX_HD SymHess3() : dat{} {}
    CHYSX_HD explicit SymHess3(float d) : dat{d, d, d, 0, 0, 0} {}

    CHYSX_HD float& operator[](int i) { return dat[i]; }
    CHYSX_HD const float& operator[](int i) const { return dat[i]; }

    CHYSX_HD SymHess3 operator+(const SymHess3& o) const {
        SymHess3 r;
        for (int i = 0; i < 6; ++i) r.dat[i] = dat[i] + o.dat[i];
        return r;
    }
    CHYSX_HD SymHess3& operator+=(const SymHess3& o) {
        for (int i = 0; i < 6; ++i) dat[i] += o.dat[i];
        return *this;
    }
    CHYSX_HD SymHess3 operator-(const SymHess3& o) const {
        SymHess3 r;
        for (int i = 0; i < 6; ++i) r.dat[i] = dat[i] - o.dat[i];
        return r;
    }
    CHYSX_HD SymHess3& operator-=(const SymHess3& o) {
        for (int i = 0; i < 6; ++i) dat[i] -= o.dat[i];
        return *this;
    }
    CHYSX_HD SymHess3 operator*(float s) const {
        SymHess3 r;
        for (int i = 0; i < 6; ++i) r.dat[i] = dat[i] * s;
        return r;
    }
    CHYSX_HD SymHess3 operator-() const {
        SymHess3 r;
        for (int i = 0; i < 6; ++i) r.dat[i] = -dat[i];
        return r;
    }

    // Outer product v*v^T (symmetric)
    CHYSX_HD static SymHess3 outer(Vec3f v) {
        SymHess3 m;
        m.dat[0] = v.x * v.x;
        m.dat[1] = v.y * v.y;
        m.dat[2] = v.z * v.z;
        m.dat[3] = v.x * v.y;
        m.dat[4] = v.x * v.z;
        m.dat[5] = v.y * v.z;
        return m;
    }

    // Convert to full Mat3f
    CHYSX_HD explicit operator Mat3f() const {
        return Mat3f(
            dat[0], dat[3], dat[4],
            dat[3], dat[1], dat[5],
            dat[4], dat[5], dat[2]);
    }
};

// GPU-side metadata (all pointers are device pointers).
struct YarnMetaData {
    Vertex*  d_verts;
    Quatf*   d_qs;
    Vec4f*   d_qRests;

    Vec3f*   d_dx;
    Vec3f*   d_vels;
    Vec3f*   d_lastVels;

    Vec3f*    d_lastPos;
    uint32_t* d_lastFlags;
    int*      d_lastCID;

    int*   d_numCols;
    float* d_maxStepSize;
    float* d_paddingSize;
    int*   d_collisions;
    collision::Aabb* d_bounds;
    Vec2i* d_boundColList;

    Vec3f gravity;
    int   numItr;

    float h;
    float lastH;
    float time;
    int   numVerts;

    float damping;
    float drag;
    float frictionCoeff;
    float kCollision;
    float kFriction;

    float detectionRadius;
    float scaledDetectionRadius;
    float radius;
    float accelerationRatio;

    float barrierThickness;
    float detectionScaler;

    float bvhRebuildPeriod;
    int   detectionPeriod;

    float maxSegLen;
    float minSegLen;
    int   useStepSizeLimit;
};

// Segment closest-points utility (ported from Kit::segmentClosestPoints).
// Returns (u,v) such that the closest points on segment (a0,a1) and
// segment (b0,b1) are mix(a0,a1,u) and mix(b0,b1,v).
CHYSX_HDI Vec2f segment_closest_points(Vec3f a0, Vec3f a1, Vec3f b0, Vec3f b1) {
    Vec3f da = a1 - a0;
    Vec3f db = b1 - b0;

    float a = dot(da, da);
    float b = dot(da, db);
    float c = dot(db, db);

    Vec3f diff = b0 - a0;
    float denom = a * c - b * b;

    Vec2f uv;
    uv.y = b * dot(da, diff) - a * dot(db, diff);
    uv.y /= denom;

    // Handle degenerate / near-parallel case
    if (!(uv.y == uv.y)) uv.y = 0.5f;  // isnan check
    uv.y = math::clamp(uv.y, 0.f, 1.f);

    // Re-project to handle clamping
    Vec3f bp = b0 * (1.f - uv.y) + b1 * uv.y;
    uv.x = math::clamp(dot(bp - a0, da) / a, 0.f, 1.f);

    Vec3f ap = a0 * (1.f - uv.x) + a1 * uv.x;
    uv.y = math::clamp(dot(ap - b0, db) / c, 0.f, 1.f);

    return uv;
}

}  // namespace yarn
}  // namespace chysx
