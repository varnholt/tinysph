#include "sph/scene.h"

#include <algorithm>
#include <cmath>

namespace sph
{

namespace
{

template <class... Callables>
struct Overloaded : Callables...
{
   using Callables::operator()...;
};

float boxDistance(const BoxCollider& box, const Vec3& point)
{
   const auto offset = componentAbs(point - box._center) - box._half_extents;
   const auto outside_distance = length(componentMax(offset, {}));
   const auto inside_distance = std::min(std::max(offset.x, std::max(offset.y, offset.z)), 0.0f);
   return outside_distance + inside_distance;
}

}  // namespace

float signedDistance(const Collider& collider, const Vec3& point)
{
   return std::visit(
      Overloaded{
         [&](const SphereCollider& sphere) { return length(point - sphere._center) - sphere._radius; },
         [&](const BoxCollider& box) { return boxDistance(box, point); },
      },
      collider
   );
}

Vec3 surfaceNormal(const Collider& collider, const Vec3& point)
{
   // central differences work for any sdf and are cheap for these shapes
   constexpr auto epsilon = 1e-4f;
   const Vec3 dx{epsilon, 0.0f, 0.0f};
   const Vec3 dy{0.0f, epsilon, 0.0f};
   const Vec3 dz{0.0f, 0.0f, epsilon};
   const Vec3 gradient{
      signedDistance(collider, point + dx) - signedDistance(collider, point - dx),
      signedDistance(collider, point + dy) - signedDistance(collider, point - dy),
      signedDistance(collider, point + dz) - signedDistance(collider, point - dz),
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

Scene makeDamBreak()
{
   Scene scene;
   scene._name = "Dam break";
   scene._container = {{0.0f, 0.0f, 0.0f}, {1.6f, 1.0f, 0.6f}};
   scene._fluids.emplace_back(FluidBox{{{0.02f, 0.02f, 0.02f}, {0.5f, 0.6f, 0.58f}}, {}});
   scene._colliders.emplace_back(SphereCollider{{1.1f, 0.12f, 0.3f}, 0.12f});
   return scene;
}

Scene makeDropIntoPool()
{
   Scene scene;
   scene._name = "Drop into pool";
   scene._container = {{0.0f, 0.0f, 0.0f}, {1.0f, 1.2f, 1.0f}};
   scene._fluids.emplace_back(FluidBox{{{0.02f, 0.02f, 0.02f}, {0.98f, 0.2f, 0.98f}}, {}});
   scene._fluids.emplace_back(FluidSphere{{0.5f, 0.75f, 0.5f}, 0.15f, {0.0f, -1.0f, 0.0f}});
   return scene;
}

Scene makeTwoBlobs()
{
   Scene scene;
   scene._name = "Two blobs (zero gravity)";
   scene._container = {{0.0f, 0.0f, 0.0f}, {1.6f, 1.0f, 1.0f}};
   scene._gravity = {};
   scene._fluids.emplace_back(FluidSphere{{0.45f, 0.5f, 0.5f}, 0.15f, {1.0f, 0.0f, 0.0f}});
   scene._fluids.emplace_back(FluidSphere{{1.15f, 0.55f, 0.5f}, 0.15f, {-1.0f, 0.0f, 0.0f}});
   return scene;
}

Scene makeFountain()
{
   Scene scene;
   scene._name = "Fountain";
   scene._container = {{0.0f, 0.0f, 0.0f}, {1.2f, 1.0f, 1.2f}};
   scene._fluids.emplace_back(FluidBox{{{0.02f, 0.02f, 0.02f}, {1.18f, 0.1f, 1.18f}}, {}});
   scene._colliders.emplace_back(BoxCollider{{0.6f, 0.45f, 0.6f}, {0.15f, 0.02f, 0.15f}});
   scene._emitters.push_back(Emitter{{0.6f, 0.15f, 0.6f}, {0.0f, 3.0f, 0.0f}, 0.04f, true, 0.0f});
   scene._fields.emplace_back(Vortex{{0.6f, 0.5f, 0.6f}, {0.0f, 1.0f, 0.0f}, 4.0f, 0.5f});
   return scene;
}

}  // namespace sph
