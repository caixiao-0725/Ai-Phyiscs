// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#ifdef TARGET_OS_MAC
#include <OpenGL/GL.h>
#else
#include <GL/gl.h>
#endif

#include "math/vec.cuh"

namespace chysx {
namespace rigid {
namespace pabd_cuda {

class PabdCudaSurfaceRenderer {
public:
    PabdCudaSurfaceRenderer();
    ~PabdCudaSurfaceRenderer();

    PabdCudaSurfaceRenderer(const PabdCudaSurfaceRenderer&) = delete;
    PabdCudaSurfaceRenderer& operator=(const PabdCudaSurfaceRenderer&) = delete;

    void update(const math::Vec3f* positions_dev,
                const math::Vec3i* triangles_dev,
                int n_triangles);
    void draw(float r, float g, float b) const;
    void reset();

private:
    void ensure_capacity(int n_triangles);

    GLuint tri_vbo_ = 0;
    GLuint normal_vbo_ = 0;
    struct cudaGraphicsResource* tri_res_ = nullptr;
    struct cudaGraphicsResource* normal_res_ = nullptr;
    int capacity_triangles_ = 0;
    int n_triangles_ = 0;
};

}  // namespace pabd_cuda
}  // namespace rigid
}  // namespace chysx
