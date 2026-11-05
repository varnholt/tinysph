#include "tinysph/world.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "geometry.h"
#include "solver.h"

namespace tinysph
{

namespace
{

/// \brief finds the slot of an id in a list of ids kept parallel to the solver's object list.
template <class Id>
std::optional<std::size_t> slotOf(const std::vector<Id>& ids, Id id)
{
   const auto position = std::ranges::find(ids, id);
   if (position == ids.end())
   {
      return std::nullopt;
   }
   return static_cast<std::size_t>(position - ids.begin());
}

}  // namespace

struct World::Impl
{
   Impl(const WorldSettings& settings, const FluidParameters& parameters) : _solver(settings, parameters)
   {
   }

   template <class Id>
   Id nextId()
   {
      return static_cast<Id>(_next_id++);
   }

   detail::Solver _solver;
   std::vector<ColliderId> _collider_ids;       //!< parallel to _solver.colliders()
   std::vector<ForceFieldId> _force_field_ids;  //!< parallel to _solver.fields()
   std::vector<EmitterId> _emitter_ids;         //!< parallel to _solver.emitters()
   uint32_t _next_id{1};                        //!< ids are never reused, so a stale handle cannot hit a new object
};

World::World(const WorldSettings& settings, const FluidParameters& parameters) : _impl(std::make_unique<Impl>(settings, parameters))
{
}

World::~World() = default;
World::World(World&& other) noexcept = default;
World& World::operator=(World&& other) noexcept = default;

const WorldSettings& World::settings() const
{
   return _impl->_solver.settings();
}

const FluidParameters& World::parameters() const
{
   return _impl->_solver.parameters();
}

void World::setParameters(const FluidParameters& parameters)
{
   _impl->_solver.setParameters(parameters);
}

const Aabb& World::domain() const
{
   return _impl->_solver.settings()._domain;
}

void World::setDomain(const Aabb& domain)
{
   _impl->_solver.setDomain(domain);
}

std::size_t World::addFluid(const Shape& shape, const Vec3& velocity)
{
   auto& solver = _impl->_solver;
   return std::visit(
      detail::Overloaded{
         [&](const Sphere& sphere) { return solver.addFluidSphere(sphere._center, sphere._radius, velocity); },
         [&](const Box& box) { return solver.addFluidBox(detail::bounds(box), velocity); },
      },
      shape
   );
}

bool World::addParticle(const Vec3& position, const Vec3& velocity)
{
   return _impl->_solver.addParticle(position, velocity);
}

void World::clearParticles()
{
   _impl->_solver.clearParticles();
}

std::size_t World::particleCount() const
{
   return _impl->_solver.particleCount();
}

float World::particleMass() const
{
   return _impl->_solver.particleMass();
}

float World::particleRadius() const
{
   return _impl->_solver.particleRadius();
}

std::span<const Vec3> World::positions() const
{
   return _impl->_solver.positions();
}

std::span<const Vec3> World::velocities() const
{
   return _impl->_solver.velocities();
}

std::span<const float> World::densities() const
{
   return _impl->_solver.densities();
}

ColliderId World::addCollider(const Shape& shape)
{
   _impl->_solver.colliders().push_back(shape);
   return _impl->_collider_ids.emplace_back(_impl->nextId<ColliderId>());
}

bool World::setCollider(ColliderId id, const Shape& shape)
{
   const auto slot = slotOf(_impl->_collider_ids, id);
   if (!slot.has_value())
   {
      return false;
   }
   _impl->_solver.colliders()[*slot] = shape;
   return true;
}

bool World::removeCollider(ColliderId id)
{
   const auto slot = slotOf(_impl->_collider_ids, id);
   if (!slot.has_value())
   {
      return false;
   }
   auto& colliders = _impl->_solver.colliders();
   colliders.erase(colliders.begin() + static_cast<std::ptrdiff_t>(*slot));
   _impl->_collider_ids.erase(_impl->_collider_ids.begin() + static_cast<std::ptrdiff_t>(*slot));
   return true;
}

std::optional<Shape> World::collider(ColliderId id) const
{
   const auto slot = slotOf(_impl->_collider_ids, id);
   if (!slot.has_value())
   {
      return std::nullopt;
   }
   return _impl->_solver.colliders()[*slot];
}

ForceFieldId World::addForceField(const ForceField& field)
{
   _impl->_solver.fields().push_back(field);
   return _impl->_force_field_ids.emplace_back(_impl->nextId<ForceFieldId>());
}

bool World::setForceField(ForceFieldId id, const ForceField& field)
{
   const auto slot = slotOf(_impl->_force_field_ids, id);
   if (!slot.has_value())
   {
      return false;
   }
   _impl->_solver.fields()[*slot] = field;
   return true;
}

bool World::removeForceField(ForceFieldId id)
{
   const auto slot = slotOf(_impl->_force_field_ids, id);
   if (!slot.has_value())
   {
      return false;
   }
   auto& fields = _impl->_solver.fields();
   fields.erase(fields.begin() + static_cast<std::ptrdiff_t>(*slot));
   _impl->_force_field_ids.erase(_impl->_force_field_ids.begin() + static_cast<std::ptrdiff_t>(*slot));
   return true;
}

std::optional<ForceField> World::forceField(ForceFieldId id) const
{
   const auto slot = slotOf(_impl->_force_field_ids, id);
   if (!slot.has_value())
   {
      return std::nullopt;
   }
   return _impl->_solver.fields()[*slot];
}

EmitterId World::addEmitter(const Emitter& emitter)
{
   _impl->_solver.emitters().push_back({emitter, 0.0f});
   return _impl->_emitter_ids.emplace_back(_impl->nextId<EmitterId>());
}

bool World::setEmitter(EmitterId id, const Emitter& emitter)
{
   const auto slot = slotOf(_impl->_emitter_ids, id);
   if (!slot.has_value())
   {
      return false;
   }
   // keeps the travelled distance so editing a running emitter does not cause a gap or a double layer
   _impl->_solver.emitters()[*slot]._emitter = emitter;
   return true;
}

bool World::removeEmitter(EmitterId id)
{
   const auto slot = slotOf(_impl->_emitter_ids, id);
   if (!slot.has_value())
   {
      return false;
   }
   auto& emitters = _impl->_solver.emitters();
   emitters.erase(emitters.begin() + static_cast<std::ptrdiff_t>(*slot));
   _impl->_emitter_ids.erase(_impl->_emitter_ids.begin() + static_cast<std::ptrdiff_t>(*slot));
   return true;
}

std::optional<Emitter> World::emitter(EmitterId id) const
{
   const auto slot = slotOf(_impl->_emitter_ids, id);
   if (!slot.has_value())
   {
      return std::nullopt;
   }
   return _impl->_solver.emitters()[*slot]._emitter;
}

StepStatistics World::step(float time, int32_t max_substeps)
{
   return _impl->_solver.advance(time, max_substeps);
}

}  // namespace tinysph
