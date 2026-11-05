#pragma once

#include "tinysph/math.h"
#include "tinysph/types.h"

namespace tinysph::detail
{

/// \brief signed distance from a point to the shape surface, negative inside.
[[nodiscard]] float signedDistance(const Shape& shape, const Vec3& point);

/// \brief outward surface normal at a point (gradient of the signed distance).
[[nodiscard]] Vec3 surfaceNormal(const Shape& shape, const Vec3& point);

/// \brief acceleration a force field applies at a given point.
[[nodiscard]] Vec3 fieldAcceleration(const ForceField& field, const Vec3& point);

/// \brief bounding box of a shape.
[[nodiscard]] Aabb bounds(const Shape& shape);

template <class... Callables>
struct Overloaded : Callables...
{
   using Callables::operator()...;
};

}  // namespace tinysph::detail
