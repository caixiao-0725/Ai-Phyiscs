// SPDX-License-Identifier: Apache-2.0

#include "pabd_cuda_surface_renderer.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#ifdef _WIN32
typedef void (APIENTRY *PFNGLGENBUFFERSPROC)(GLsizei, GLuint*);
typedef void (APIENTRY *PFNGLDELETEBUFFERSPROC)(GLsizei, const GLuint*);
typedef void (APIENTRY *PFNGLBINDBUFFERPROC)(GLenum, GLuint);
typedef void (APIENTRY *PFNGLBUFFERDATAPROC)(GLenum, ptrdiff_t, const void*, GLenum);

#ifndef GL_ARRAY_BUFFER
#define GL_ARRAY_BUFFER 0x8892
#endif
#ifndef GL_DYNAMIC_DRAW
#define GL_DYNAMIC_DRAW 0x88E8
#endif

static PFNGLGENBUFFERSPROC pglGenBuffers = nullptr;
static PFNGLDELETEBUFFERSPROC pglDeleteBuffers = nullptr;
static PFNGLBINDBUFFERPROC pglBindBuffer = nullptr;
static PFNGLBUFFERDATAPROC pglBufferData = nullptr;

static void load_gl_procs() {
    if (pglGenBuffers) return;
    pglGenBuffers = reinterpret_cast<PFNGLGENBUFFERSPROC>(
        wglGetProcAddress("glGenBuffers"));
    pglDeleteBuffers = reinterpret_cast<PFNGLDELETEBUFFERSPROC>(
        wglGetProcAddress("glDeleteBuffers"));
    pglBindBuffer = reinterpret_cast<PFNGLBINDBUFFERPROC>(
        wglGetProcAddress("glBindBuffer"));
    pglBufferData = reinterpret_cast<PFNGLBUFFERDATAPROC>(
        wglGetProcAddress("glBufferData"));
}

#define glGenBuffers pglGenBuffers
#define glDeleteBuffers pglDeleteBuffers
#define glBindBuffer pglBindBuffer
#define glBufferData pglBufferData
#else
static void load_gl_procs() {}
#endif

namespace chysx {
namespace rigid {
namespace pabd_cuda {

namespace {

constexpr int kBlock = 128;

inline int grid(int n) {
    return (n + kBlock - 1) / kBlock;
}

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("PabdCudaSurfaceRenderer: ") +
                                 what + " failed: " +
                                 cudaGetErrorString(err));
    }
}

__device__ math::Vec3f to_gl_coords(const math::Vec3f& p) {
    return math::Vec3f(p.x, p.z, p.y);
}

__global__ void fill_surface_vbo_kernel(
    float* vertices,
    float* normals,
    const math::Vec3f* positions,
    const math::Vec3i* triangles,
    int n_triangles) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_triangles) return;

    const math::Vec3i tri = triangles[tid];
    const math::Vec3f p0 = to_gl_coords(positions[tri.x]);
    const math::Vec3f p1 = to_gl_coords(positions[tri.y]);
    const math::Vec3f p2 = to_gl_coords(positions[tri.z]);
    math::Vec3f n = math::cross(p1 - p0, p2 - p0);
    const float len2 = math::dot(n, n);
    if (len2 > 1.0e-20f) {
        n = n * rsqrtf(len2);
    } else {
        n = math::Vec3f(0.0f, 0.0f, 1.0f);
    }

    const math::Vec3f ps[3] = {p0, p1, p2};
    const int base = tid * 9;
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        vertices[base + k * 3 + 0] = ps[k].x;
        vertices[base + k * 3 + 1] = ps[k].y;
        vertices[base + k * 3 + 2] = ps[k].z;
        normals[base + k * 3 + 0] = n.x;
        normals[base + k * 3 + 1] = n.y;
        normals[base + k * 3 + 2] = n.z;
    }
}

}  // namespace

PabdCudaSurfaceRenderer::PabdCudaSurfaceRenderer() = default;

PabdCudaSurfaceRenderer::~PabdCudaSurfaceRenderer() {
    reset();
}

void PabdCudaSurfaceRenderer::reset() {
    if (tri_res_) {
        cudaGraphicsUnregisterResource(tri_res_);
        tri_res_ = nullptr;
    }
    if (normal_res_) {
        cudaGraphicsUnregisterResource(normal_res_);
        normal_res_ = nullptr;
    }
    load_gl_procs();
    if (tri_vbo_) {
        glDeleteBuffers(1, &tri_vbo_);
        tri_vbo_ = 0;
    }
    if (normal_vbo_) {
        glDeleteBuffers(1, &normal_vbo_);
        normal_vbo_ = 0;
    }
    capacity_triangles_ = 0;
    n_triangles_ = 0;
}

void PabdCudaSurfaceRenderer::ensure_capacity(int n_triangles) {
    if (n_triangles <= capacity_triangles_) return;

    reset();
    load_gl_procs();

    capacity_triangles_ = n_triangles + n_triangles / 4 + 256;
    const std::size_t bytes =
        static_cast<std::size_t>(capacity_triangles_) * 9 * sizeof(float);

    glGenBuffers(1, &tri_vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, tri_vbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<ptrdiff_t>(bytes), nullptr,
                 GL_DYNAMIC_DRAW);

    glGenBuffers(1, &normal_vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, normal_vbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<ptrdiff_t>(bytes), nullptr,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    check_cuda(cudaGraphicsGLRegisterBuffer(
                   &tri_res_, tri_vbo_, cudaGraphicsMapFlagsWriteDiscard),
               "register tri vbo");
    check_cuda(cudaGraphicsGLRegisterBuffer(
                   &normal_res_, normal_vbo_, cudaGraphicsMapFlagsWriteDiscard),
               "register normal vbo");
}

void PabdCudaSurfaceRenderer::update(const math::Vec3f* positions_dev,
                                     const math::Vec3i* triangles_dev,
                                     int n_triangles) {
    if (positions_dev == nullptr || triangles_dev == nullptr ||
        n_triangles <= 0) {
        n_triangles_ = 0;
        return;
    }

    ensure_capacity(n_triangles);
    n_triangles_ = n_triangles;

    cudaGraphicsResource* resources[] = {tri_res_, normal_res_};
    check_cuda(cudaGraphicsMapResources(2, resources, 0), "map vbos");

    float* vertex_ptr = nullptr;
    float* normal_ptr = nullptr;
    std::size_t size = 0;
    check_cuda(cudaGraphicsResourceGetMappedPointer(
                   reinterpret_cast<void**>(&vertex_ptr), &size, tri_res_),
               "get tri vbo");
    check_cuda(cudaGraphicsResourceGetMappedPointer(
                   reinterpret_cast<void**>(&normal_ptr), &size, normal_res_),
               "get normal vbo");

    fill_surface_vbo_kernel<<<grid(n_triangles), kBlock>>>(
        vertex_ptr, normal_ptr, positions_dev, triangles_dev, n_triangles);
    check_cuda(cudaGetLastError(), "fill surface vbo kernel");

    check_cuda(cudaGraphicsUnmapResources(2, resources, 0), "unmap vbos");
}

void PabdCudaSurfaceRenderer::draw(float r, float g, float b) const {
    if (n_triangles_ <= 0) return;
    load_gl_procs();

    glColor3f(r, g, b);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, tri_vbo_);
    glVertexPointer(3, GL_FLOAT, 0, nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, normal_vbo_);
    glNormalPointer(GL_FLOAT, 0, nullptr);

    glDrawArrays(GL_TRIANGLES, 0, n_triangles_ * 3);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}

}  // namespace pabd_cuda
}  // namespace rigid
}  // namespace chysx
