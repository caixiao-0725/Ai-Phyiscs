// SPDX-License-Identifier: Apache-2.0
//
// Yarn simulation scenes for the ChysX viewer / batch exporter.
// Reproduces YarnBall's main.cpp + jsonBuilder.cpp pipeline:
//   JSON config → curve file (.poly / .bcc / .obj) → configure → glue → fix → upload.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <json/json.h>

#include "render/scene.h"
#include "yarn/yarn_solver.h"
#include "yarn/yarn_types.cuh"
#include "geometry/polyline_mesh.h"

namespace {

using namespace chysx;
using namespace chysx::yarn;
using namespace chysx::math;

// ============================================================================
// Catmull-Rom spline helpers (ported from KittenEngine Common.h / resample.h)
// ============================================================================

struct DVec3 { double x, y, z; };

static DVec3 operator+(DVec3 a, DVec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
static DVec3 operator-(DVec3 a, DVec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static DVec3 operator*(double s, DVec3 v) { return {s*v.x, s*v.y, s*v.z}; }
static double dot3(DVec3 a, DVec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static double length3(DVec3 v) { return std::sqrt(dot3(v, v)); }
static double length2_3(DVec3 v) { return dot3(v, v); }

static DVec3 to_dvec3(Vec3f v) { return {v.x, v.y, v.z}; }
static Vec3f to_vec3f(DVec3 v) { return Vec3f(static_cast<float>(v.x),
                                               static_cast<float>(v.y),
                                               static_cast<float>(v.z)); }

static DVec3 mix3(DVec3 a, DVec3 b, double t) { return (1.0 - t) * a + t * b; }

static DVec3 cmr_spline(DVec3 p0, DVec3 p1, DVec3 p2, DVec3 p3, double t) {
    DVec3 a0 = mix3(p0, p1, t + 1.0);
    DVec3 a1 = mix3(p1, p2, t);
    DVec3 a2 = mix3(p2, p3, t - 1.0);
    DVec3 b0 = mix3(a0, a1, (t + 1.0) * 0.5);
    DVec3 b1 = mix3(a1, a2, t * 0.5);
    return mix3(b0, b1, t);
}

static DVec3 sample_curve(const std::vector<Vec3f>& cmr, double t) {
    int si = std::max(1, std::min(static_cast<int>(std::floor(t)),
                                   static_cast<int>(cmr.size()) - 3));
    return cmr_spline(to_dvec3(cmr[si-1]), to_dvec3(cmr[si]),
                      to_dvec3(cmr[si+1]), to_dvec3(cmr[si+2]), t - si);
}

static std::vector<double> resample_cmr_coords(const std::vector<Vec3f>& cmr,
                                                 double start, double end,
                                                 double tar_seg_len) {
    std::vector<double> ts;
    constexpr double MIN_STEP = 0.2;
    const double step = end > start ? MIN_STEP : -MIN_STEP;
    const double r2 = tar_seg_len * tar_seg_len;

    double i = start;
    DVec3 last_pos = sample_curve(cmr, i);
    ts.push_back(i);

    while ((end > start) ? i < end : i > end) {
        i += step;
        DVec3 pos = sample_curve(cmr, i);
        double l0 = length2_3(pos - last_pos);

        if (l0 > r2) {
            l0 = std::sqrt(l0);
            double range_lo = std::min(i - step, i);
            double range_hi = std::max(i - step, i);

            double t0 = i;
            i -= 0.5 * step;
            for (int k = 0; k < 8; ++k) {
                DVec3 x = sample_curve(cmr, i);
                double d = length3(x - last_pos);
                i += (i - t0) * (tar_seg_len - d) / (d - l0);
            }
            i = std::max(range_lo, std::min(range_hi, i));

            ts.push_back(i);
            last_pos = sample_curve(cmr, i);
        }
    }

    if (ts.size() == 1) {
        ts.push_back(end);
    } else {
        double d = length3(sample_curve(cmr, ts.back()) - sample_curve(cmr, end));
        double dt = ts.back() - ts[ts.size() - 2];

        if (d > 0.5 * tar_seg_len)
            ts.push_back(end);
        else
            ts.back() = end;

        dt -= ts.back() - ts[ts.size() - 2];

        const int N = std::min(static_cast<int>(ts.size()), 8);
        for (int ii = 0; ii < N; ++ii)
            ts[ts.size() - 2 - ii] -= dt * (N - ii) / static_cast<double>(N);
    }
    return ts;
}

static std::vector<double> two_dir_resample_cmr_coords(const std::vector<Vec3f>& cmr,
                                                         double start, double end,
                                                         double tar_seg_len) {
    auto forward  = resample_cmr_coords(cmr, start, end, tar_seg_len);
    auto backward = resample_cmr_coords(cmr, end, start, tar_seg_len);

    constexpr int BORDER = 8;
    if (static_cast<int>(forward.size()) > 4 * BORDER &&
        static_cast<int>(backward.size()) > 4 * BORDER) {
        double min_error = 1e30;
        int split_f = 2 * BORDER, split_b = 2 * BORDER;
        int j = static_cast<int>(backward.size()) - 1 - BORDER;

        for (int ii = BORDER; ii < static_cast<int>(forward.size()) - 8; ++ii) {
            double t = forward[ii];
            while (j > 0 && backward[j] < t) --j;

            double t0 = backward[j + 1];
            if (min_error > std::abs(t - t0)) {
                min_error = std::abs(t - t0);
                split_f = ii; split_b = j + 1;
            }
            double t1 = backward[j];
            if (min_error > std::abs(t1 - t)) {
                min_error = std::abs(t1 - t);
                split_f = ii; split_b = j;
            }
        }

        std::vector<double> merged;
        merged.reserve(split_f + split_b + 1);
        merged.insert(merged.end(), forward.begin(), forward.begin() + split_f);
        double mid = 0.5 * (forward[split_f] + backward[split_b]);
        merged.push_back(mid);
        for (int ii = split_b - 1; ii >= 0; --ii)
            merged.push_back(backward[ii]);

        double dt_f = forward[split_f] - mid;
        for (int ii = 0; ii < BORDER && split_f - ii - 1 >= 0; ++ii)
            merged[split_f - ii - 1] -= dt_f * (BORDER - ii) / (1.0 + BORDER);
        double dt_b = backward[split_b] - mid;
        for (int ii = 0; ii < BORDER && split_f + ii + 1 < static_cast<int>(merged.size()); ++ii)
            merged[split_f + ii + 1] -= dt_b * (BORDER - ii) / (1.0 + BORDER);

        return merged;
    }
    return forward;
}

static std::vector<Vec3f> resample_cmr(const std::vector<Vec3f>& cmr,
                                         double start, double end,
                                         double tar_seg_len) {
    auto ts0 = two_dir_resample_cmr_coords(cmr, start, end, tar_seg_len);
    double total_seg_len = 0.0;
    double error0 = 0.0;
    DVec3 lp = sample_curve(cmr, ts0[0]);
    for (size_t i = 1; i < ts0.size(); ++i) {
        DVec3 pos = sample_curve(cmr, ts0[i]);
        double len = length3(pos - lp);
        error0 = std::max(error0, std::abs(len - tar_seg_len));
        total_seg_len += len;
        lp = pos;
    }

    double avg = total_seg_len / (ts0.size() - 1);
    auto ts1 = two_dir_resample_cmr_coords(cmr, start, end, avg);
    lp = sample_curve(cmr, ts1[0]);
    double error1 = 0.0;
    for (size_t i = 1; i < ts1.size(); ++i) {
        DVec3 pos = sample_curve(cmr, ts1[i]);
        double len = length3(pos - lp);
        error1 = std::max(error1, std::abs(len - avg));
        lp = pos;
    }

    auto& ts = (error0 < error1) ? ts0 : ts1;
    std::vector<Vec3f> out(ts.size());
    for (size_t i = 0; i < ts.size(); ++i)
        out[i] = to_vec3f(sample_curve(cmr, ts[i]));
    return out;
}

// ============================================================================
// Curve data container and createFromCurves (ported from reader.cpp)
// ============================================================================

struct CurveData {
    std::vector<std::vector<Vec3f>> curves;
    std::vector<bool> is_closed;
    int total_verts = 0;
};

static YarnSolver* create_from_curves(CurveData& data) {
    auto* solver = new YarnSolver(data.total_verts);
    auto* verts = solver->verts();

    int vi = 0;
    float max_len = 0.f, min_len = FLT_MAX;
    for (size_t ci = 0; ci < data.curves.size(); ++ci) {
        auto& curve = data.curves[ci];

        Vec3f last_p = curve[0];
        verts[vi].pos = last_p;
        for (size_t j = 1; j < curve.size(); ++j) {
            Vec3f p = curve[j];
            verts[vi + j].pos = p;
            float l = length(p - last_p);
            max_len = std::max(max_len, l);
            min_len = std::min(min_len, l);
            last_p = p;
        }

        verts[vi + curve.size() - 1].flags = 0;

        if (data.is_closed[ci]) {
            verts[vi].connectionIndex = vi + static_cast<int>(curve.size()) - 1;
            verts[vi + static_cast<int>(curve.size()) - 1].connectionIndex = vi;
        }
        vi += static_cast<int>(curve.size());
    }
    printf("Resampled with length max %f, min %f\n", max_len, min_len);
    return solver;
}

// ============================================================================
// BCC file reader (ported from YarnBall reader.cpp)
// ============================================================================

struct BCCHeader {
    char sign[3];
    unsigned char byteCount;
    char curveType[2];
    char dimensions;
    char upDimension;
    uint64_t curveCount;
    uint64_t totalControlPointCount;
    char fileInfo[40];
};

struct Mat4 { float m[4][4]; };

static Vec3f transform_point(const Mat4& t, Vec3f p) {
    float x = t.m[0][0]*p.x + t.m[1][0]*p.y + t.m[2][0]*p.z + t.m[3][0];
    float y = t.m[0][1]*p.x + t.m[1][1]*p.y + t.m[2][1]*p.z + t.m[3][1];
    float z = t.m[0][2]*p.x + t.m[1][2]*p.y + t.m[2][2]*p.z + t.m[3][2];
    return Vec3f(x, y, z);
}

static CurveData read_from_bcc(const std::string& path, float target_seg_len,
                                const Mat4& transform, bool break_up_closed,
                                bool allow_resample) {
    BCCHeader header;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open BCC file: " + path);
    fread(&header, sizeof(header), 1, f);

    if (header.sign[0] != 'B' || header.sign[1] != 'C' || header.sign[2] != 'C' ||
        header.byteCount != 0x44) {
        fclose(f);
        throw std::runtime_error("Unsupported BCC file: " + path);
    }

    bool is_polyline = header.curveType[0] == 'P' && header.curveType[1] == 'L';

    CurveData data;
    for (uint64_t ci = 0; ci < header.curveCount; ++ci) {
        int num_points;
        fread(&num_points, sizeof(int), 1, f);
        bool is_closed = num_points < 0;
        num_points = std::abs(num_points);

        std::vector<Vec3f> points(num_points);
        fread(points.data(), sizeof(Vec3f), num_points, f);

        for (auto& p : points)
            p = transform_point(transform, p);

        if (allow_resample && !is_polyline)
            points = resample_cmr(points, 1.0, static_cast<double>(points.size()) - 2.0,
                                  target_seg_len);

        if (num_points < 3) continue;

        data.is_closed.push_back(is_closed && !break_up_closed);
        data.curves.push_back(points);
        data.total_verts += static_cast<int>(points.size());
    }
    fclose(f);
    return data;
}

// ============================================================================
// OBJ polyline reader (ported from YarnBall reader.cpp)
// ============================================================================

static CurveData read_from_obj(const std::string& path, float target_seg_len,
                                const Mat4& transform, bool break_up_closed,
                                bool allow_resample) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open OBJ file: " + path);

    std::vector<Vec3f> vertices;
    std::vector<std::vector<int>> lines;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            Vec3f pos;
            iss >> pos.x >> pos.y >> pos.z;
            vertices.push_back(transform_point(transform, pos));
        } else if (prefix == "l") {
            std::vector<int> idx;
            int id;
            while (iss >> id) idx.push_back(id);
            lines.push_back(idx);
        }
    }

    CurveData data;
    for (auto& ln : lines) {
        std::vector<Vec3f> curve;
        curve.reserve(ln.size());
        bool closed = ln.front() == ln.back();
        for (int id : ln)
            curve.push_back(vertices[id - 1]);

        if (break_up_closed && closed) {
            curve.pop_back();
            closed = false;
        }

        if (allow_resample)
            curve = resample_cmr(curve, 1.0, static_cast<double>(curve.size()) - 2.0,
                                 target_seg_len);

        if (curve.size() < 4) continue;
        data.total_verts += static_cast<int>(curve.size());
        data.curves.push_back(curve);
        data.is_closed.push_back(closed);
    }
    return data;
}

// ============================================================================
// .poly file reader (ported from YarnBall reader.cpp)
// ============================================================================

static CurveData read_from_poly(const std::string& path, float target_seg_len,
                                 const Mat4& transform, bool break_up_closed,
                                 bool allow_resample) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open .poly file: " + path);

    std::vector<Vec3f> points;
    std::vector<Vec2i> segs;
    bool is_point = true;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        if (line == "POINTS") { is_point = true; continue; }
        else if (line == "POLYS") { is_point = false; continue; }
        else if (line == "END") break;

        if (is_point) {
            int id; float x, y, z; char colon;
            if (iss >> id >> colon >> x >> y >> z && colon == ':') {
                Vec3f p(x, y, z);
                points.push_back(transform_point(transform, p));
            }
        } else {
            int id, i, j; char colon;
            if (iss >> id >> colon >> i >> j && colon == ':')
                segs.push_back(Vec2i(i - 1, j - 1));
        }
    }

    std::unordered_multimap<int, int> seg_map;
    for (auto& s : segs) {
        seg_map.insert({s.x, s.y});
        seg_map.insert({s.y, s.x});
    }

    CurveData data;
    std::vector<bool> visited(points.size(), false);

    // Open curves first
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        if (visited[i]) continue;
        int nc = static_cast<int>(seg_map.count(i));
        if (nc > 2) throw std::runtime_error("Graphs not allowed in .poly");
        if (nc > 1) continue;

        std::vector<Vec3f> curve;
        int cur = i;
        while (!visited[cur]) {
            visited[cur] = true;
            curve.push_back(points[cur]);
            auto range = seg_map.equal_range(cur);
            bool found = false;
            for (auto it = range.first; it != range.second; ++it) {
                if (!visited[it->second]) { cur = it->second; found = true; break; }
            }
            if (!found) break;
        }
        if (curve.size() < 4) continue;
        printf("Found open curve with %zd points from %d to %d\n", curve.size(), i + 1, cur + 1);
        if (allow_resample)
            curve = resample_cmr(curve, 1.0, static_cast<double>(curve.size()) - 2.0,
                                 target_seg_len);
        if (curve.size() < 4) continue;
        data.total_verts += static_cast<int>(curve.size());
        data.curves.push_back(curve);
        data.is_closed.push_back(false);
    }

    // Closed curves
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        if (visited[i]) continue;

        std::vector<Vec3f> curve;
        int cur = i;
        while (true) {
            visited[cur] = true;
            curve.push_back(points[cur]);
            bool found = false;
            auto range = seg_map.equal_range(cur);
            for (auto it = range.first; it != range.second; ++it) {
                if (!visited[it->second]) { cur = it->second; found = true; break; }
            }
            if (!found) break;
        }
        printf("Found closed curve with %zd points from %d to %d\n", curve.size(), i + 1, cur + 1);
        if (allow_resample)
            curve = resample_cmr(curve, 1.0, static_cast<double>(curve.size()) - 2.0,
                                 target_seg_len);
        if (curve.size() < 4) continue;
        data.total_verts += static_cast<int>(curve.size());
        data.curves.push_back(curve);
        data.is_closed.push_back(!break_up_closed);
    }

    return data;
}

// ============================================================================
// glueEndpoints (ported from YarnBall jsonBuilder.cpp)
// ============================================================================

static void glue_endpoints(YarnSolver* solver, float search_radius) {
    const int nv = solver->num_verts();
    auto* verts = solver->verts();

    std::vector<int> endpoints;
    for (int i = 0; i < nv; ++i) {
        bool has_prev = i > 0 && (verts[i - 1].flags & static_cast<uint32_t>(VertexFlags::hasNext)) != 0;
        bool has_next = (verts[i].flags & static_cast<uint32_t>(VertexFlags::hasNext)) != 0;
        if (has_prev ^ has_next)
            endpoints.push_back(i);
    }

    for (int ei : endpoints) {
        Vec3f pos = verts[ei].pos;
        float min_dist = 1e30f;
        int closest = -1;
        for (int j = 0; j < nv; ++j) {
            if (std::abs(ei - j) <= 2) continue;
            float d = length(pos - verts[j].pos);
            if (d < min_dist) { min_dist = d; closest = j; }
        }
        if (min_dist <= search_radius && closest >= 0) {
            if (verts[closest].connectionIndex < 0 && verts[ei].connectionIndex < 0) {
                verts[closest].connectionIndex = ei;
                verts[ei].connectionIndex = closest;
                Vec3f avg = 0.5f * (verts[ei].pos + verts[closest].pos);
                verts[closest].pos = avg;
                verts[ei].pos = avg;
            }
        }
    }
}

// ============================================================================
// fixBorders (ported from YarnBall jsonBuilder.cpp)
// ============================================================================

struct Bound3 {
    Vec3f lo{1e30f, 1e30f, 1e30f};
    Vec3f hi{-1e30f, -1e30f, -1e30f};
    void absorb(Vec3f p) {
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
    }
    void pad(float d) { lo.x -= d; lo.y -= d; lo.z -= d; hi.x += d; hi.y += d; hi.z += d; }
    bool contains(Vec3f p) const {
        return p.x >= lo.x && p.x <= hi.x &&
               p.y >= lo.y && p.y <= hi.y &&
               p.z >= lo.z && p.z <= hi.z;
    }
};

static void fix_borders(YarnSolver* solver, float border[6]) {
    const int nv = solver->num_verts();
    auto* verts = solver->verts();

    Bound3 bounds;
    for (int i = 0; i < nv; ++i)
        bounds.absorb(verts[i].pos);

    bounds.pad(0.01f * solver->meta().radius);
    bounds.hi.x -= border[0];
    bounds.lo.x += border[1];
    bounds.hi.y -= border[2];
    bounds.lo.y += border[3];
    bounds.hi.z -= border[4];
    bounds.lo.z += border[5];

    for (int i = 0; i < nv; ++i)
        if (!bounds.contains(verts[i].pos))
            verts[i].invMass = 0.f;
}

// ============================================================================
// fixVertex (ported from YarnBall jsonBuilder.cpp)
// ============================================================================

static void fix_vertex(YarnSolver* solver, Vec3f pos, float radius = -1.f) {
    const int nv = solver->num_verts();
    auto* verts = solver->verts();

    if (radius > 0.f) {
        float r2 = radius * radius;
        for (int i = 0; i < nv; ++i) {
            Vec3f d = verts[i].pos - pos;
            if (dot(d, d) < r2)
                verts[i].invMass = 0.f;
        }
    } else {
        float min_dist = 1e30f;
        int closest = -1;
        for (int i = 0; i < nv; ++i) {
            Vec3f d = verts[i].pos - pos;
            float d2 = dot(d, d);
            if (d2 < min_dist) { min_dist = d2; closest = i; }
        }
        if (closest >= 0)
            verts[closest].invMass = 0.f;
    }
}

// ============================================================================
// Resolve a curve file path relative to the config directory
// ============================================================================

static std::string resolve_curve_path(const std::string& config_dir,
                                       const std::string& curve_file) {
    // YarnBall paths look like "configs/models/xxx.poly" relative to CWD.
    // We store data in assets/yarn_data/ and the config JSON in that directory.
    // Curve files referenced as "configs/models/xxx" → strip "configs/" prefix
    // and resolve relative to config_dir.
    std::string rel = curve_file;
    if (rel.rfind("configs/", 0) == 0)
        rel = rel.substr(8);  // strip "configs/"

    // Try: config_dir + rel
    std::string p1 = config_dir + rel;
    {
        std::ifstream test(p1);
        if (test.good()) return p1;
    }

    // Try: config_dir + filename only
    auto slash = rel.find_last_of("/\\");
    if (slash != std::string::npos) {
        std::string p2 = config_dir + rel.substr(slash + 1);
        std::ifstream test(p2);
        if (test.good()) return p2;
    }

    // Fallback: try original path as-is
    {
        std::ifstream test(curve_file);
        if (test.good()) return curve_file;
    }

    throw std::runtime_error("Cannot find curve file: " + curve_file);
}

// ============================================================================
// buildFromJSON — complete port of YarnBall's jsonBuilder.cpp
// ============================================================================

static YarnSolver* build_from_json(const std::string& json_path) {
    std::ifstream file(json_path, std::ifstream::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open JSON config: " + json_path);

    Json::CharReaderBuilder rbuilder;
    rbuilder["allowComments"] = true;
    rbuilder["allowTrailingCommas"] = true;
    std::string errs;

    Json::Value root;
    if (!Json::parseFromStream(rbuilder, file, &root, &errs))
        throw std::runtime_error("JSON parse error: " + errs);

    // Determine config directory
    std::string config_dir = json_path;
    auto last_sep = config_dir.find_last_of("/\\");
    config_dir = (last_sep != std::string::npos) ? config_dir.substr(0, last_sep + 1) : "";

    // Transform matrix (column-major, default 0.01 scale)
    Mat4 transform{};
    transform.m[0][0] = 0.01f; transform.m[1][1] = 0.01f; transform.m[2][2] = 0.01f;
    transform.m[3][3] = 1.0f;

    if (!root["transform"].isNull()) {
        auto& t = root["transform"];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                transform.m[j][i] = t[i][j].asFloat();
    }

    // Curve file
    std::string curve_file = root["curveFile"].asString();
    std::string resolved_path = resolve_curve_path(config_dir, curve_file);

    bool resample = !root["resampleLength"].isNull();
    float resample_len = resample ? root["resampleLength"].asFloat() : 3e-3f;
    bool break_up_closed = root["breakUpClosedCurves"].isNull()
                           ? false : root["breakUpClosedCurves"].asBool();

    printf("[YarnScene] Loading: %s\n", resolved_path.c_str());

    CurveData data;
    if (resolved_path.size() > 5 &&
        resolved_path.substr(resolved_path.size() - 5) == ".poly") {
        data = read_from_poly(resolved_path, resample_len, transform,
                               break_up_closed, resample);
    } else if (resolved_path.size() > 4 &&
               resolved_path.substr(resolved_path.size() - 4) == ".obj") {
        data = read_from_obj(resolved_path, resample_len, transform,
                              break_up_closed, resample);
    } else {
        data = read_from_bcc(resolved_path, resample_len, transform,
                              break_up_closed, resample);
    }

    if (data.total_verts < 3)
        throw std::runtime_error("No valid curves loaded from " + resolved_path);

    YarnSolver* solver = create_from_curves(data);

    // curveRadius → radius & barrierThickness
    if (!root["curveRadius"].isNull()) {
        float r = root["curveRadius"].asFloat();
        constexpr float ratio = 0.05f;
        solver->meta().radius = ratio * r;
        solver->meta().barrierThickness = 2.f * (1.f - ratio) * r;
    }

    // Simulation parameters
    auto& sim = root["simulation"];
    double density = 1e-3;
    double kStretch = 5e5;
    double kBend = 1e-1;

    if (!sim.isNull()) {
        if (!sim["density"].isNull())         density = sim["density"].asDouble();
        if (!sim["frictionCoeff"].isNull())   solver->meta().frictionCoeff = sim["frictionCoeff"].asFloat();
        if (!sim["kStretch"].isNull())         kStretch = sim["kStretch"].asDouble();
        if (!sim["kBend"].isNull())            kBend = sim["kBend"].asDouble();
        if (!sim["kCollision"].isNull())       solver->meta().kCollision = sim["kCollision"].asFloat();
        if (!sim["kFriction"].isNull())        solver->meta().kFriction = sim["kFriction"].asFloat();
        if (!sim["damping"].isNull())          solver->meta().damping = sim["damping"].asFloat();

        if (!sim["gravity"].isNull()) {
            auto& g = sim["gravity"];
            solver->meta().gravity = Vec3f(g[0].asFloat(), g[1].asFloat(), g[2].asFloat());
        }
        if (!sim["drag"].isNull())             solver->meta().drag = sim["drag"].asFloat();

        if (!sim["maxTimeStep"].isNull())      solver->set_max_h(sim["maxTimeStep"].asFloat());
        if (!sim["numIterations"].isNull())    solver->meta().numItr = sim["numIterations"].asInt();
        if (!sim["detectionPeriod"].isNull())  solver->meta().detectionPeriod = sim["detectionPeriod"].asInt();
        if (!sim["detectionScaler"].isNull())  solver->meta().detectionScaler = sim["detectionScaler"].asFloat();
        if (!sim["stepLimit"].isNull())        solver->meta().useStepSizeLimit = sim["stepLimit"].asBool() ? 1 : 0;
        if (!sim["bvhRebuildPeriod"].isNull()) solver->meta().bvhRebuildPeriod = sim["bvhRebuildPeriod"].asFloat();
    }

    solver->configure(static_cast<float>(density));
    solver->set_k_stretch(static_cast<float>(kStretch));
    solver->set_k_bend(static_cast<float>(kBend));

    // glueEndpoints
    if (!root["glueEndpoints"].isNull()) {
        float radius = root["glueEndpoints"].asFloat();
        if (radius >= 0.f)
            glue_endpoints(solver, radius);
    }

    // fixBorders
    if (!root["fixBorders"].isNull()) {
        auto& borders = root["fixBorders"];
        if (borders.isArray() && borders.size() == 6) {
            float b[6];
            for (int i = 0; i < 6; ++i)
                b[i] = borders[i].asFloat();
            fix_borders(solver, b);
        }
    }

    // fixVertex
    if (!root["fixVertex"].isNull()) {
        auto& verts = root["fixVertex"];
        if (verts.isArray()) {
            for (Json::ArrayIndex i = 0; i < verts.size(); ++i) {
                if (verts[i].isArray()) {
                    auto& sphere = verts[i];
                    if (sphere.size() < 3) continue;
                    Vec3f pos(sphere[0].asFloat(), sphere[1].asFloat(), sphere[2].asFloat());
                    if (sphere.size() >= 4)
                        fix_vertex(solver, pos, sphere[3].asFloat());
                    else
                        fix_vertex(solver, pos);
                } else {
                    solver->verts()[verts[i].asInt()].invMass = 0.f;
                }
            }
        }
    }

    solver->upload();
    printf("[YarnScene] Total verts: %d\n", solver->num_verts());
    return solver;
}

// ============================================================================
// Generic JSON-driven YarnBall scene
// ============================================================================

class YarnJsonScene : public render::Scene {
public:
    YarnJsonScene(const std::string& scene_name, const std::string& json_name)
        : scene_name_(scene_name), json_name_(json_name) {}

    const char* name() const override { return scene_name_.c_str(); }

    void setup() override {
        std::string data_dir = render::get_data_path("assets/yarn_data/");
        std::string json_path = data_dir + json_name_;
        printf("[YarnScene] Config: %s\n", json_path.c_str());
        solver_.reset(build_from_json(json_path));

        // Determine draw radius from curveRadius
        draw_radius_ = solver_->meta().radius;
        if (draw_radius_ < 1e-6f) draw_radius_ = 5e-4f;

        frame_ = 0;
    }

    void step(float dt) override {
        solver_->advance(dt);
        solver_->download();
        ++frame_;
    }

    void draw_meshes(std::vector<render::DrawMesh>& out) override {
        if (!solver_) return;
        build_draw_data();

        render::DrawMesh dm{};
        dm.positions = draw_pos_.data();
        dm.n_points = static_cast<int>(draw_pos_.size()) / 3;
        dm.triangles = draw_tri_.data();
        dm.n_tris = static_cast<int>(draw_tri_.size()) / 3;
        dm.color_r = color_r_; dm.color_g = color_g_; dm.color_b = color_b_;
        dm.wireframe = false;
        out.push_back(dm);
    }

private:
    std::string scene_name_;
    std::string json_name_;
    std::unique_ptr<YarnSolver> solver_;
    int frame_ = 0;

    float draw_radius_ = 5e-4f;
    float color_r_ = 0.8f, color_g_ = 0.5f, color_b_ = 0.2f;

    std::vector<float> draw_pos_;
    std::vector<int> draw_tri_;

    void build_draw_data() {
        draw_pos_.clear();
        draw_tri_.clear();

        const int nv = solver_->num_verts();
        constexpr int sides = 6;
        float r = draw_radius_;
        auto* verts = solver_->verts();

        for (int i = 0; i < nv; ++i) {
            Vec3f p = verts[i].pos;
            Vec3f tangent(0.f, 1.f, 0.f);
            if (i < nv - 1 && (verts[i].flags & static_cast<uint32_t>(VertexFlags::hasNext))) {
                tangent = verts[i + 1].pos - p;
                float tl = length(tangent);
                if (tl > 1e-12f) tangent = tangent * (1.f / tl);
            } else if (i > 0 && (verts[i].flags & static_cast<uint32_t>(VertexFlags::hasPrev))) {
                tangent = p - verts[i - 1].pos;
                float tl = length(tangent);
                if (tl > 1e-12f) tangent = tangent * (1.f / tl);
            }

            Vec3f up(0.f, 1.f, 0.f);
            if (fabsf(dot(tangent, up)) > 0.99f) up = Vec3f(1.f, 0.f, 0.f);
            Vec3f b1 = normalize(cross(tangent, up));
            Vec3f b2 = cross(tangent, b1);

            for (int j = 0; j < sides; ++j) {
                float angle = j * 2.f * 3.14159265f / sides;
                Vec3f vp = p + r * (cosf(angle) * b1 + sinf(angle) * b2);
                draw_pos_.push_back(vp.x);
                draw_pos_.push_back(vp.y);
                draw_pos_.push_back(vp.z);
            }
        }

        for (int i = 0; i < nv - 1; ++i) {
            if (!(verts[i].flags & static_cast<uint32_t>(VertexFlags::hasNext))) continue;
            int base0 = i * sides;
            int base1 = (i + 1) * sides;
            for (int j = 0; j < sides; ++j) {
                int j1 = (j + 1) % sides;
                draw_tri_.push_back(base0 + j);
                draw_tri_.push_back(base1 + j);
                draw_tri_.push_back(base1 + j1);
                draw_tri_.push_back(base0 + j);
                draw_tri_.push_back(base1 + j1);
                draw_tri_.push_back(base0 + j1);
            }
        }
    }
};

}  // anonymous namespace

// Macro to generate non-capturing factory functions for each scene.
#define YARN_SCENE(TAG, DISPLAY_NAME, JSON_FILE) \
    static chysx::render::Scene* create_yarn_##TAG() { \
        return new YarnJsonScene(DISPLAY_NAME, JSON_FILE); \
    }

YARN_SCENE(cable_work, "Yarn: Cable Work Pattern", "cable_work_pattern.json")
YARN_SCENE(letter_s,   "Yarn: Letter S",           "letterS.json")
YARN_SCENE(letter_g,   "Yarn: Letter G",           "letterG.json")
YARN_SCENE(letter_a,   "Yarn: Letter A",           "letterA.json")
YARN_SCENE(letter_h,   "Yarn: Letter H",           "letterH.json")
YARN_SCENE(letter_i,   "Yarn: Letter I",           "letterI.json")
YARN_SCENE(letter_p,   "Yarn: Letter P",           "letterP.json")
YARN_SCENE(letter_r,   "Yarn: Letter R",           "letterR.json")
YARN_SCENE(letter_g2,  "Yarn: Letter G2",          "letterG2.json")

#undef YARN_SCENE

extern "C" void chysx_register_yarn_scenes() {
    using chysx::render::register_scene;
    register_scene("Yarn: Cable Work Pattern", create_yarn_cable_work);
    register_scene("Yarn: Letter S",           create_yarn_letter_s);
    register_scene("Yarn: Letter G",           create_yarn_letter_g);
    register_scene("Yarn: Letter A",           create_yarn_letter_a);
    register_scene("Yarn: Letter H",           create_yarn_letter_h);
    register_scene("Yarn: Letter I",           create_yarn_letter_i);
    register_scene("Yarn: Letter P",           create_yarn_letter_p);
    register_scene("Yarn: Letter R",           create_yarn_letter_r);
    register_scene("Yarn: Letter G2",          create_yarn_letter_g2);
}
