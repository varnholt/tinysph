#pragma once

#include <string>
#include <variant>
#include <vector>

#include "sph/vec3.h"

namespace sph
{

// ----------------------------------------------------------------------------
// colliders: static solid shapes, described as signed distance functions
// ----------------------------------------------------------------------------

struct SphereCollider
{
   Vec3 _center;
   float _radius{0.1f};
};

struct BoxCollider
{
   Vec3 _center;
   Vec3 _half_extents{0.1f, 0.1f, 0.1f};
};

using Collider = std::variant<SphereCollider, BoxCollider>;

/// \brief signed distance from a point to the collider surface, negative inside.
[[nodiscard]] float signedDistance(const Collider& collider, const Vec3& point);

/// \brief outward surface normal at a point (gradient of the signed distance).
[[nodiscard]] Vec3 surfaceNormal(const Collider& collider, const Vec3& point);

// ----------------------------------------------------------------------------
// force fields: extra accelerations applied to every particle
// ----------------------------------------------------------------------------

/// \brief pulls particles towards a point, a negative strength pushes them away. falls off linearly to zero at the radius.
struct Attractor
{
   Vec3 _position;
   float _strength{20.0f};  //!< acceleration at the center in m/s^2
   float _radius{0.3f};     //!< range of influence in m
};

/// \brief swirls particles around an axis through a point.
struct Vortex
{
   Vec3 _position;
   Vec3 _axis{0.0f, 1.0f, 0.0f};
   float _strength{10.0f};  //!< tangential acceleration at the axis in m/s^2
   float _radius{0.3f};     //!< range of influence in m
};

/// \brief constant acceleration inside a box.
struct Wind
{
   Aabb _region{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};
   Vec3 _acceleration{5.0f, 0.0f, 0.0f};
};

using ForceField = std::variant<Attractor, Vortex, Wind>;

/// \brief acceleration a force field applies at a given point.
[[nodiscard]] Vec3 fieldAcceleration(const ForceField& field, const Vec3& point);

// ----------------------------------------------------------------------------
// emitters: spawn new fluid over time
// ----------------------------------------------------------------------------

/// \brief a disk shaped nozzle. it spawns one lattice layer of particles each time the previous layer has
///        travelled one particle spacing, so new particles never overlap and the flow rate follows from speed and radius.
struct Emitter
{
   Vec3 _position;
   Vec3 _velocity{0.0f, 2.0f, 0.0f};  //!< direction and speed of the jet
   float _radius{0.04f};
   bool _enabled{true};
   float _travelled{0.0f};  //!< distance the last layer has moved since it was spawned
};

// ----------------------------------------------------------------------------
// fluid volumes that are filled once when the scene is (re)loaded
// ----------------------------------------------------------------------------

struct FluidBox
{
   Aabb _box;
   Vec3 _velocity;
};

struct FluidSphere
{
   Vec3 _center;
   float _radius{0.1f};
   Vec3 _velocity;
};

using FluidVolume = std::variant<FluidBox, FluidSphere>;

struct Scene
{
   std::string _name;
   Aabb _container{{0.0f, 0.0f, 0.0f}, {1.6f, 1.0f, 0.6f}};  //!< the fluid never leaves this box
   Vec3 _gravity{0.0f, -9.81f, 0.0f};
   std::vector<FluidVolume> _fluids;
   std::vector<Collider> _colliders;
   std::vector<ForceField> _fields;
   std::vector<Emitter> _emitters;
};

// built-in starting points
[[nodiscard]] Scene makeDamBreak();
[[nodiscard]] Scene makeDropIntoPool();
[[nodiscard]] Scene makeTwoBlobs();
[[nodiscard]] Scene makeFountain();

}  // namespace sph
