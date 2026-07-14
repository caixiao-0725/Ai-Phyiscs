// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>

#include "../../math/vec.cuh"

namespace chysx {
namespace rigid {
namespace pabd_cuda {

// Volume of the tetrahedron formed by a PF/EE four-point stencil after its
// normal separation is moved to the contact activation distance.  Using the
// activation distance keeps the quadrature measure finite as contact closes:
//
//     V_hat = |det(p1-p0, p2-p0, p3-p0)| / 6 * d_hat / d.
//
// For PF this is A_face * d_hat / 3.  For non-parallel EE it is
// |edge0 x edge1| * d_hat / 6.
CHYSX_HDI float contact_tetrahedron_volume_at_activation(
    const math::Vec3f& p0,
    const math::Vec3f& p1,
    const math::Vec3f& p2,
    const math::Vec3f& p3,
    float current_distance,
    float activation_distance) {
    if (!(current_distance > 1.0e-12f) ||
        !(activation_distance > 0.0f)) {
        return 0.0f;
    }
    const float six_volume = fabsf(math::dot(
        p1 - p0, math::cross(p2 - p0, p3 - p0)));
    return six_volume * (activation_distance / current_distance) / 6.0f;
}

}  // namespace pabd_cuda
}  // namespace rigid
}  // namespace chysx
