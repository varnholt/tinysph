#include "geometry.h"

#include <algorithm>
#include <cmath>

namespace tinysph::detail
{

namespace
{

float boxDistance(const Box& box, const Vec3& point)
{
   const auto offset = componentAbs(point - box._center) - box._half_extents;
   const auto outside_distance = length(componentMax(offset, {}));
   const auto inside_distance = std::min(std::max(offset.x, std::max(offset.y, offset.z)), 0.0f);
   return outside_distance + inside_distance;
}

}  // namespace

float signedDistance(const Shape& shape, const Vec3& point)
{
   return std::visit(
      Overloaded{
         [&](const Sphere& sphere) { return length(point - sphere._center) - sphere._radius; },
         [&](const Box& box) { return boxDistance(box, point); },
      },
      shape
   );
}

Vec3 surfaceNormal(const Shape& shape, const Vec3& point)
{
   // central differences work for any sdf and are cheap for these shapes
   constexpr auto epsilon = 1e-4f;
   const Vec3 dx{epsilon, 0.0f, 0.0f};
   const Vec3 dy{0.0f, epsilon, 0.0f};
   const Vec3 dz{0.0f, 0.0f, epsilon};
   const Vec3 gradient{
      signedDistance(shape, point + dx) - signedDistance(shape, point - dx),
      signedDistance(shape, point + dy) - signedDistance(shape, point - dy),
      signedDistance(shape, point + dz) - signedDistance(shape, point - dz),
   };
   return normalize(gradient);
}

Vec3 fieldAcceleration(const ForceField& field, const Vec3& point)
{
   return std::visit(
      Overloaded{
         [&](const Attractor& attractor) -> Vec3
         {
            const auto to_center = attractor._position - point;
            const auto distance = length(to_center);
            if (distance >= attractor._radius || distance < 1e-6f)
            {
               return {};
            }
            return to_center * (attractor._strength * (1.0f - distance / attractor._radius) / distance);
         },
         [&](const Vortex& vortex) -> Vec3
         {
            const auto axis = normalize(vortex._axis);
            const auto offset = point - vortex._position;
            const auto radial = offset - axis * dot(offset, axis);
            const auto distance = length(radial);
            if (distance >= vortex._radius || distance < 1e-6f)
            {
               return {};
            }
            const auto tangent = cross(axis, radial / distance);
            return tangent * (vortex._strength * (1.0f - distance / vortex._radius));
         },
         [&](const Wind& wind) -> Vec3 { return wind._region.contains(point) ? wind._acceleration : Vec3{}; },
      },
      field
   );
}

Aabb bounds(const Shape& shape)
{
   return std::visit(
      Overloaded{
         [](const Sphere& sphere) -> Aabb
         {
            const Vec3 extent{sphere._radius, sphere._radius, sphere._radius};
            return {sphere._center - extent, sphere._center + extent};
         },
         [](const Box& box) -> Aabb { return {box._center - box._half_extents, box._center + box._half_extents}; },
      },
      shape
   );
}

}  // namespace tinysph::detail
