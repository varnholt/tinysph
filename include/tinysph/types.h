#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>

#include "tinysph/math.h"

namespace tinysph
{

// ----------------------------------------------------------------------------------------------------
// shapes, used for fluid volumes and colliders
// ----------------------------------------------------------------------------------------------------

struct Sphere
{
   Vec3 _center;
   float _radius{0.1f};
};

struct Box
{
   Vec3 _center;
   Vec3 _half_extents{0.1f, 0.1f, 0.1f};
};

using Shape = std::variant<Sphere, Box>;

// ----------------------------------------------------------------------------------------------------
// force fields: extra accelerations applied to every particle inside their range
// ----------------------------------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------------------------------
// emitters
// ----------------------------------------------------------------------------------------------------

/// \brief a disk shaped nozzle. it spawns one lattice layer of particles each time the previous layer has
///        travelled one particle spacing, so new particles never overlap; the flow rate follows from speed and radius.
struct Emitter
{
   Vec3 _position;
   Vec3 _velocity{0.0f, 2.0f, 0.0f};  //!< direction and speed of the jet in m/s
   float _radius{0.04f};              //!< nozzle radius in m
   bool _enabled{true};
};

// ----------------------------------------------------------------------------------------------------
// handles; they stay valid until the object is removed, and are never reused within a world
// ----------------------------------------------------------------------------------------------------

enum class ColliderId : uint32_t
{
};

enum class ForceFieldId : uint32_t
{
};

enum class EmitterId : uint32_t
{
};

// ----------------------------------------------------------------------------------------------------
// configuration
// ----------------------------------------------------------------------------------------------------

enum class Backend : int32_t
{
   Cpu = 0,  //!< multithreaded through the standard parallel algorithms where available
};

/// \brief fixed for the lifetime of a world. everything is in si units (m, kg, s).
struct WorldSettings
{
   Aabb _domain{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};  //!< the fluid never leaves this box; can be changed later with setDomain()
   float _particle_spacing{0.025f};                       //!< distance between particles at rest; sets resolution and particle size
   float _rest_density{1000.0f};                          //!< rho0 in kg/m^3, water
   float _support_radius_factor{2.0f};                    //!< smoothing radius h = factor * spacing
   std::size_t _max_particles{250'000};                   //!< hard cap; adding particles beyond it fails, emitters pause
   Backend _backend{Backend::Cpu};
};

/// \brief can be changed at any time with World::setParameters().
struct FluidParameters
{
   Vec3 _gravity{0.0f, -9.81f, 0.0f};
   float _speed_of_sound{30.0f};  //!< stiffness of the equation of state p = c^2 (rho - rho0); higher = less compressible, smaller steps
   float _viscosity{0.2f};        //!< dynamic viscosity mu in Pa s
   float _artificial_viscosity{0.05f};   //!< monaghan alpha; damps approaching particle pairs; at 0 the fluid never settles
   bool _clamp_negative_pressure{true};  //!< avoids particles clumping at the free surface
   float _wall_restitution{0.0f};        //!< 0 = walls absorb the normal velocity, 1 = perfect bounce
   float _wall_friction{0.02f};          //!< fraction of the tangential velocity removed per contact
   float _max_time_step{1.0f / 400.0f};  //!< upper bound for the adaptive time step in s
   float _cfl_number{0.4f};              //!< dt <= cfl * h / (c + max speed)
};

/// \brief what happened during the last World::step() call.
struct StepStatistics
{
   float _time_step{0.0f};  //!< dt of the last substep in s
   int32_t _substeps{0};
   float _simulated_time{0.0f};         //!< sum of all substep dts in s; less than requested when max_substeps was hit
   float _max_density_ratio{0.0f};      //!< max rho / rho0
   float _average_density_ratio{0.0f};  //!< average rho / rho0
   float _max_speed{0.0f};              //!< in m/s
   double _step_ms{0.0};                //!< wall clock time of the whole call
};

}  // namespace tinysph
