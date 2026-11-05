#include "sph/solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <ranges>
#include <utility>

#include "sph/parallel.h"

namespace sph
{

namespace
{

template <class... Callables>
struct Overloaded : Callables...
{
   using Callables::operator()...;
};

/// \brief sum of w_poly6 over a cubic lattice around one particle (self included).
///        dividing rho0 by this gives a mass for which a resting lattice has exactly the rest density.
float latticeKernelSum(const Kernels& kernels, float spacing)
{
   const auto range = static_cast<int32_t>(std::ceil(kernels.supportRadius() / spacing));
   auto sum = 0.0f;
   for (auto z = -range; z <= range; ++z)
   {
      for (auto y = -range; y <= range; ++y)
      {
         for (auto x = -range; x <= range; ++x)
         {
            const Vec3 offset{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
            sum += kernels.poly6(lengthSquared(offset * spacing));
         }
      }
   }
   return sum;
}

/// \brief xorshift32; deterministic so tests and runs are reproducible.
float nextJitter(uint32_t& state)
{
   state ^= state << 13;
   state ^= state >> 17;
   state ^= state << 5;
   return static_cast<float>(state) / static_cast<float>(UINT32_MAX) - 0.5f;
}

}  // namespace

Solver::Solver(const SolverSettings& settings) : _settings(settings)
{
   applySettings();
}

void Solver::applySettings()
{
   const auto spacing = _settings._particle_spacing;
   _kernels.setSupportRadius(spacing * _settings._support_radius_factor);
   _particle_mass = _settings._rest_density / latticeKernelSum(_kernels, spacing);
   _particle_radius = spacing * 0.5f;
   setupGrid();
}

void Solver::loadScene(const Scene& scene)
{
   _scene = scene;
   _jitter_state = 1u;
   applySettings();
   clearParticles();

   for (const auto& fluid : _scene._fluids)
   {
      std::visit(
         Overloaded{
            [&](const FluidBox& box) { addFluidBox(box._box, box._velocity); },
            [&](const FluidSphere& sphere) { addFluidSphere(sphere._center, sphere._radius, sphere._velocity); },
         },
         fluid
      );
   }
}

void Solver::clearParticles()
{
   _positions.clear();
   _velocities.clear();
   _accelerations.clear();
   _densities.clear();
   _pressures.clear();
   _max_acceleration = 0.0f;
}

std::size_t Solver::addFluidBox(const Aabb& box, const Vec3& velocity)
{
   const auto spacing = _settings._particle_spacing;
   std::size_t added_count = 0;
   for (auto z = box._min.z + _particle_radius; z <= box._max.z - _particle_radius; z += spacing)
   {
      for (auto y = box._min.y + _particle_radius; y <= box._max.y - _particle_radius; y += spacing)
      {
         for (auto x = box._min.x + _particle_radius; x <= box._max.x - _particle_radius; x += spacing)
         {
            if (addParticle({x, y, z}, velocity))
            {
               ++added_count;
            }
         }
      }
   }
   return added_count;
}

std::size_t Solver::addFluidSphere(const Vec3& center, float radius, const Vec3& velocity)
{
   const auto spacing = _settings._particle_spacing;
   const auto steps = static_cast<int32_t>(std::floor(radius / spacing));
   std::size_t added_count = 0;
   for (auto z = -steps; z <= steps; ++z)
   {
      for (auto y = -steps; y <= steps; ++y)
      {
         for (auto x = -steps; x <= steps; ++x)
         {
            const auto offset = Vec3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)} * spacing;
            if (lengthSquared(offset) <= radius * radius && addParticle(center + offset, velocity))
            {
               ++added_count;
            }
         }
      }
   }
   return added_count;
}

bool Solver::addParticle(const Vec3& position, const Vec3& velocity)
{
   if (_positions.size() >= _settings._max_particles)
   {
      return false;
   }

   const auto& container = _scene._container;
   const Vec3 margin{_particle_radius, _particle_radius, _particle_radius};
   if (!Aabb{container._min + margin, container._max - margin}.contains(position))
   {
      return false;
   }

   const auto inside_collider = std::ranges::any_of(
      _scene._colliders, [&](const Collider& collider) { return signedDistance(collider, position) < _particle_radius; }
   );
   if (inside_collider)
   {
      return false;
   }

   const auto jitter_amplitude = _settings._particle_spacing * 0.01f;
   const Vec3 jitter{nextJitter(_jitter_state), nextJitter(_jitter_state), nextJitter(_jitter_state)};

   _positions.push_back(position + jitter * jitter_amplitude);
   _velocities.push_back(velocity);
   _accelerations.emplace_back();
   _densities.push_back(_settings._rest_density);
   _pressures.push_back(0.0f);
   return true;
}

StepStatistics Solver::advance(float frame_time, int32_t max_substeps)
{
   const auto start_time = std::chrono::steady_clock::now();

   StepStatistics statistics;
   auto remaining_time = frame_time;
   while (remaining_time > 1e-7f && statistics._substeps < max_substeps)
   {
      const auto time_step = std::min(computeTimeStep(), remaining_time);
      step(time_step);
      remaining_time -= time_step;
      statistics._simulated_time += time_step;
      statistics._time_step = time_step;
      ++statistics._substeps;
   }

   if (!_densities.empty())
   {
      const auto density_sum = std::reduce(_densities.begin(), _densities.end(), 0.0);
      statistics._average_density_ratio =
         static_cast<float>(density_sum / static_cast<double>(_densities.size())) / _settings._rest_density;
      statistics._max_density_ratio = std::ranges::max(_densities) / _settings._rest_density;
   }
   statistics._max_speed = maxSpeed();

   const auto elapsed = std::chrono::steady_clock::now() - start_time;
   statistics._step_ms = std::chrono::duration<double, std::milli>(elapsed).count();
   return statistics;
}

void Solver::step(float time_step)
{
   if (!_positions.empty())
   {
      rebuildGrid();
      computeDensities();
      computeAccelerations();
      integrate(time_step);
   }
   runEmitters(time_step);
}

float Solver::computeTimeStep() const
{
   const auto support_radius = _kernels.supportRadius();
   auto time_step = _settings._max_time_step;

   // cfl condition: no information may travel further than a fraction of h per step
   time_step = std::min(time_step, _settings._cfl_number * support_radius / (_settings._speed_of_sound + maxSpeed()));

   // force condition: a particle starting at rest may not move further than a fraction of h per step
   if (_max_acceleration > 0.0f)
   {
      time_step = std::min(time_step, 0.25f * std::sqrt(support_radius / _max_acceleration));
   }

   return time_step;
}

void Solver::setupGrid()
{
   const auto cell_size = _kernels.supportRadius();
   const auto& container = _scene._container;

   // one cell of padding on each side so particles touching the walls still have all 27 cells
   _grid_origin = container._min - Vec3{cell_size, cell_size, cell_size};
   const auto extent = container.size() + Vec3{cell_size, cell_size, cell_size} * 2.0f;
   _grid_size = {
      std::max(1, static_cast<int32_t>(std::ceil(extent.x / cell_size))),
      std::max(1, static_cast<int32_t>(std::ceil(extent.y / cell_size))),
      std::max(1, static_cast<int32_t>(std::ceil(extent.z / cell_size))),
   };

   const auto cell_count = static_cast<std::size_t>(_grid_size.x) * _grid_size.y * _grid_size.z;
   _cell_start.assign(cell_count + 1, 0u);
   _cell_write_offset.assign(cell_count, 0u);
}

Solver::CellCoordinate Solver::cellCoordinate(const Vec3& position) const
{
   const auto relative = (position - _grid_origin) / _kernels.supportRadius();
   return {
      std::clamp(static_cast<int32_t>(std::floor(relative.x)), 0, _grid_size.x - 1),
      std::clamp(static_cast<int32_t>(std::floor(relative.y)), 0, _grid_size.y - 1),
      std::clamp(static_cast<int32_t>(std::floor(relative.z)), 0, _grid_size.z - 1),
   };
}

std::size_t Solver::cellIndex(const CellCoordinate& coordinate) const
{
   return (static_cast<std::size_t>(coordinate.z) * _grid_size.y + coordinate.y) * _grid_size.x + coordinate.x;
}

void Solver::rebuildGrid()
{
   const auto particle_count = _positions.size();
   _cell_of_particle.resize(particle_count);
   _sorted_order.resize(particle_count);

   // 1) cell of every particle
   parallelFor(
      particle_count,
      [this](std::size_t particle_index)
      { _cell_of_particle[particle_index] = static_cast<uint32_t>(cellIndex(cellCoordinate(_positions[particle_index]))); }
   );

   // 2) count particles per cell, shifted by one so the prefix sum yields the start offsets
   std::ranges::fill(_cell_start, 0u);
   for (const auto cell : _cell_of_particle)
   {
      ++_cell_start[cell + 1];
   }
   std::partial_sum(_cell_start.begin(), _cell_start.end(), _cell_start.begin());

   // 3) scatter; a stable counting sort keeps the order deterministic
   std::copy(_cell_start.begin(), _cell_start.end() - 1, _cell_write_offset.begin());
   for (auto particle_index = 0u; particle_index < particle_count; ++particle_index)
   {
      _sorted_order[_cell_write_offset[_cell_of_particle[particle_index]]++] = particle_index;
   }

   // 4) physically reorder the particle data so neighbors are close in memory
   _scratch_positions.resize(particle_count);
   _scratch_velocities.resize(particle_count);
   parallelFor(
      particle_count,
      [this](std::size_t target_index)
      {
         const auto source_index = _sorted_order[target_index];
         _scratch_positions[target_index] = _positions[source_index];
         _scratch_velocities[target_index] = _velocities[source_index];
      }
   );
   std::swap(_positions, _scratch_positions);
   std::swap(_velocities, _scratch_velocities);
}

void Solver::computeDensities()
{
   const auto mass = _particle_mass;
   const auto rest_density = _settings._rest_density;
   const auto stiffness = _settings._speed_of_sound * _settings._speed_of_sound;
   const auto clamp_negative_pressure = _settings._clamp_negative_pressure;

   parallelFor(
      _positions.size(),
      [&](std::size_t particle_index)
      {
         auto density = 0.0f;
         forEachNeighbor(
            particle_index, [&](std::size_t, const Vec3&, float distance_squared) { density += mass * _kernels.poly6(distance_squared); }
         );

         auto pressure = stiffness * (density - rest_density);
         if (clamp_negative_pressure)
         {
            pressure = std::max(pressure, 0.0f);
         }

         _densities[particle_index] = density;
         _pressures[particle_index] = pressure;
      }
   );
}

void Solver::computeAccelerations()
{
   const auto mass = _particle_mass;
   const auto viscosity = _settings._viscosity;
   const auto gravity = _scene._gravity;
   const auto support_radius = _kernels.supportRadius();
   const auto artificial_viscosity_factor = _settings._artificial_viscosity * _settings._speed_of_sound * support_radius;
   const auto singularity_guard = 0.01f * support_radius * support_radius;

   parallelFor(
      _positions.size(),
      [&](std::size_t particle_index)
      {
         const auto density = _densities[particle_index];
         const auto pressure_term = _pressures[particle_index] / (density * density);
         const auto& velocity = _velocities[particle_index];

         Vec3 pressure_acceleration;
         Vec3 viscosity_acceleration;
         forEachNeighbor(
            particle_index,
            [&](std::size_t neighbor_index, const Vec3& offset, float distance_squared)
            {
               if (neighbor_index == particle_index)
               {
                  return;
               }
               const auto distance = std::sqrt(distance_squared);
               const auto neighbor_density = _densities[neighbor_index];
               const auto neighbor_pressure_term = _pressures[neighbor_index] / (neighbor_density * neighbor_density);

               // symmetric pressure gradient: the sum of both terms, which the original code multiplied
               auto pair_term = pressure_term + neighbor_pressure_term;

               // monaghan artificial viscosity (1992): only for pairs moving towards each other
               const auto approach = dot(velocity - _velocities[neighbor_index], offset);
               if (approach < 0.0f)
               {
                  const auto average_density = 0.5f * (density + neighbor_density);
                  pair_term -= artificial_viscosity_factor * approach / ((distance_squared + singularity_guard) * average_density);
               }

               pressure_acceleration -= _kernels.spikyGradient(offset, distance) * (mass * pair_term);

               viscosity_acceleration +=
                  (_velocities[neighbor_index] - velocity) * (mass * _kernels.viscosityLaplacian(distance) / (density * neighbor_density));
            }
         );

         auto acceleration = pressure_acceleration + viscosity_acceleration * viscosity + gravity;
         const auto& position = _positions[particle_index];
         for (const auto& field : _scene._fields)
         {
            acceleration += fieldAcceleration(field, position);
         }
         for (const auto& field : _transient_fields)
         {
            acceleration += fieldAcceleration(field, position);
         }
         _accelerations[particle_index] = acceleration;
      }
   );

   const auto max_acceleration_squared = std::ranges::fold_left(
      _accelerations | std::views::transform(lengthSquared), 0.0f, [](float lhs, float rhs) { return std::max(lhs, rhs); }
   );
   _max_acceleration = std::sqrt(max_acceleration_squared);
}

void Solver::integrate(float time_step)
{
   parallelFor(
      _positions.size(),
      [&](std::size_t particle_index)
      {
         auto& velocity = _velocities[particle_index];
         auto& position = _positions[particle_index];

         // semi-implicit euler: velocity first, then position with the new velocity
         velocity += _accelerations[particle_index] * time_step;
         position += velocity * time_step;
         resolveCollisions(position, velocity);
      }
   );
}

void Solver::resolveCollisions(Vec3& position, Vec3& velocity) const
{
   const auto restitution = _settings._wall_restitution;
   const auto friction = _settings._wall_friction;

   // push the particle out along the normal and remove the velocity component that points into the surface
   const auto apply_contact = [&](const Vec3& normal, float penetration)
   {
      position += normal * penetration;
      const auto normal_speed = dot(velocity, normal);
      if (normal_speed < 0.0f)
      {
         const auto tangential_velocity = velocity - normal * normal_speed;
         velocity = tangential_velocity * (1.0f - friction) - normal * (normal_speed * restitution);
      }
   };

   for (const auto& collider : _scene._colliders)
   {
      const auto distance = signedDistance(collider, position) - _particle_radius;
      if (distance < 0.0f)
      {
         apply_contact(surfaceNormal(collider, position), -distance);
      }
   }

   // the container comes last so it always wins
   const Vec3 margin{_particle_radius, _particle_radius, _particle_radius};
   const auto lower = _scene._container._min + margin;
   const auto upper = _scene._container._max - margin;

   if (position.x < lower.x)
   {
      apply_contact({1.0f, 0.0f, 0.0f}, lower.x - position.x);
   }
   if (position.x > upper.x)
   {
      apply_contact({-1.0f, 0.0f, 0.0f}, position.x - upper.x);
   }
   if (position.y < lower.y)
   {
      apply_contact({0.0f, 1.0f, 0.0f}, lower.y - position.y);
   }
   if (position.y > upper.y)
   {
      apply_contact({0.0f, -1.0f, 0.0f}, position.y - upper.y);
   }
   if (position.z < lower.z)
   {
      apply_contact({0.0f, 0.0f, 1.0f}, lower.z - position.z);
   }
   if (position.z > upper.z)
   {
      apply_contact({0.0f, 0.0f, -1.0f}, position.z - upper.z);
   }
}

void Solver::runEmitters(float time_step)
{
   const auto spacing = _settings._particle_spacing;
   for (auto& emitter : _scene._emitters)
   {
      const auto speed = length(emitter._velocity);
      if (!emitter._enabled || speed <= 0.0f)
      {
         continue;
      }

      emitter._travelled += speed * time_step;
      while (emitter._travelled >= spacing)
      {
         emitter._travelled -= spacing;
         emitLayer(emitter);
      }
   }
}

void Solver::emitLayer(const Emitter& emitter)
{
   const auto spacing = _settings._particle_spacing;
   const auto direction = normalize(emitter._velocity);

   // two axes spanning the nozzle disk
   const auto helper = std::abs(direction.y) < 0.9f ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{1.0f, 0.0f, 0.0f};
   const auto tangent = normalize(cross(direction, helper));
   const auto bitangent = cross(direction, tangent);

   // the layer has already moved by the leftover distance since it was due
   const auto origin = emitter._position + direction * emitter._travelled;
   const auto steps = static_cast<int32_t>(std::floor(emitter._radius / spacing));
   for (auto row = -steps; row <= steps; ++row)
   {
      for (auto column = -steps; column <= steps; ++column)
      {
         const auto offset = (tangent * static_cast<float>(column) + bitangent * static_cast<float>(row)) * spacing;
         if (lengthSquared(offset) <= emitter._radius * emitter._radius)
         {
            addParticle(origin + offset, emitter._velocity);
         }
      }
   }
}

Scene& Solver::scene()
{
   return _scene;
}

const Scene& Solver::scene() const
{
   return _scene;
}

SolverSettings& Solver::settings()
{
   return _settings;
}

const SolverSettings& Solver::settings() const
{
   return _settings;
}

void Solver::setTransientFields(std::vector<ForceField> fields)
{
   _transient_fields = std::move(fields);
}

std::size_t Solver::particleCount() const
{
   return _positions.size();
}

float Solver::particleMass() const
{
   return _particle_mass;
}

float Solver::supportRadius() const
{
   return _kernels.supportRadius();
}

std::span<const Vec3> Solver::positions() const
{
   return _positions;
}

std::span<const Vec3> Solver::velocities() const
{
   return _velocities;
}

std::span<const float> Solver::densities() const
{
   return _densities;
}

Vec3 Solver::totalMomentum() const
{
   return std::reduce(_velocities.begin(), _velocities.end(), Vec3{}) * _particle_mass;
}

float Solver::maxSpeed() const
{
   const auto max_speed_squared = std::ranges::fold_left(
      _velocities | std::views::transform(lengthSquared), 0.0f, [](float lhs, float rhs) { return std::max(lhs, rhs); }
   );
   return std::sqrt(max_speed_squared);
}

}  // namespace sph
