#include "tinysph/world.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "geometry.h"
#include "solver.h"

#if defined(TINYSPH_HAS_GPU)
#include "gpu/gpu_solver.h"
#endif

namespace tinysph
{

namespace
{

/// \brief objects of one kind stored in the solver, plus a parallel list with their handles.
template <class Id, class Stored>
class Slots
{
public:
   Slots(std::vector<Id>& ids, std::vector<Stored>& objects) : _ids(ids), _objects(objects)
   {
   }

   Id add(Id id, Stored object)
   {
      _objects.push_back(std::move(object));
      return _ids.emplace_back(id);
   }

   /// \return the stored object, or nullptr if the id is unknown
   [[nodiscard]] Stored* find(Id id)
   {
      const auto position = std::ranges::find(_ids, id);
      return position == _ids.end() ? nullptr : &_objects[static_cast<std::size_t>(position - _ids.begin())];
   }

   bool remove(Id id)
   {
      const auto position = std::ranges::find(_ids, id);
      if (position == _ids.end())
      {
         return false;
      }
      _objects.erase(_objects.begin() + (position - _ids.begin()));
      _ids.erase(position);
      return true;
   }

private:
   std::vector<Id>& _ids;
   std::vector<Stored>& _objects;
};

/// \brief rough upper bound so a tiny spacing in a huge domain fails cleanly instead of exhausting memory.
constexpr double max_grid_cells = 128.0 * 1024.0 * 1024.0;

[[noreturn]] void rejectSettings(SettingsError error)
{
#if defined(__cpp_exceptions)
   const auto message = std::string{"tinysph::World: "} + std::string{describe(error)};
   if (error == SettingsError::GpuUnavailable)
   {
      throw std::runtime_error{message};
   }
   throw std::invalid_argument{message};
#else
   std::abort();
#endif
}

}  // namespace

struct World::Impl
{
   Impl(const WorldSettings& settings, const FluidParameters& parameters) : _solver(settings, parameters)
   {
   }

   template <class Id>
   [[nodiscard]] Id nextId()
   {
      return static_cast<Id>(_next_id++);
   }

   [[nodiscard]] auto colliders()
   {
      return Slots{_collider_ids, _solver.colliders()};
   }

   [[nodiscard]] auto forceFields()
   {
      return Slots{_force_field_ids, _solver.fields()};
   }

   [[nodiscard]] auto emitters()
   {
      return Slots{_emitter_ids, _solver.emitters()};
   }

   /// \brief a world for valid settings; fails if the gpu backend finds no usable gpu.
   [[nodiscard]] static std::expected<std::unique_ptr<Impl>, SettingsError>
   create(const WorldSettings& settings, const FluidParameters& parameters)
   {
      auto impl = std::make_unique<Impl>(settings, parameters);
#if defined(TINYSPH_HAS_GPU)
      if (settings._backend == Backend::Gpu)
      {
         impl->_gpu = detail::GpuSolver::create(impl->_solver);
         if (!impl->_gpu)
         {
            return std::unexpected{SettingsError::GpuUnavailable};
         }
      }
#endif
      return impl;
   }

   [[nodiscard]] StepStatistics step(float time, int32_t max_substeps)
   {
#if defined(TINYSPH_HAS_GPU)
      if (_gpu)
      {
         return _gpu->advance(time, max_substeps);
      }
#endif
      return _solver.advance(time, max_substeps);
   }

   [[nodiscard]] std::string_view deviceDescription() const
   {
#if defined(TINYSPH_HAS_GPU)
      if (_gpu)
      {
         return _gpu->deviceDescription();
      }
#endif
      return "cpu";
   }

   detail::Solver _solver;
#if defined(TINYSPH_HAS_GPU)
   std::unique_ptr<detail::GpuSolver> _gpu;  //!< runs the substeps when the backend is Backend::Gpu; holds a reference to _solver
#endif
   std::vector<ColliderId> _collider_ids;       //!< parallel to _solver.colliders()
   std::vector<ForceFieldId> _force_field_ids;  //!< parallel to _solver.fields()
   std::vector<EmitterId> _emitter_ids;         //!< parallel to _solver.emitters()
   uint32_t _next_id{1};                        //!< ids are never reused, so a stale handle cannot hit a new object
};

std::expected<void, SettingsError> World::validate(const WorldSettings& settings)
{
   // written as !(x > 0) so nan is rejected as well
   if (!(settings._particle_spacing > 0.0f))
   {
      return std::unexpected{SettingsError::NonPositiveSpacing};
   }
   if (!(settings._rest_density > 0.0f))
   {
      return std::unexpected{SettingsError::NonPositiveRestDensity};
   }
   if (!(settings._support_radius_factor >= 1.0f))
   {
      return std::unexpected{SettingsError::SupportRadiusTooSmall};
   }
   if (settings._max_particles == 0)
   {
      return std::unexpected{SettingsError::ZeroParticleCap};
   }
   switch (settings._backend)
   {
      case Backend::Cpu:
         break;
      case Backend::Gpu:
#if !defined(TINYSPH_HAS_GPU)
         return std::unexpected{SettingsError::UnsupportedBackend};
#else
         break;
#endif
      default:
         return std::unexpected{SettingsError::UnsupportedBackend};
   }

   const auto size = settings._domain.size();
   const auto minimum_extent = settings._particle_spacing * 4.0f;
   if (!(size.x >= minimum_extent && size.y >= minimum_extent && size.z >= minimum_extent))
   {
      return std::unexpected{SettingsError::EmptyDomain};
   }

   const auto cell_size = static_cast<double>(settings._particle_spacing * settings._support_radius_factor);
   const auto cells_along = [&](float extent) { return static_cast<double>(extent) / cell_size + 3.0; };
   if (cells_along(size.x) * cells_along(size.y) * cells_along(size.z) > max_grid_cells)
   {
      return std::unexpected{SettingsError::TooManyGridCells};
   }

   return {};
}

std::expected<World, SettingsError> World::create(const WorldSettings& settings, const FluidParameters& parameters)
{
   return validate(settings).and_then([&]() { return Impl::create(settings, parameters); }
   ).transform([](std::unique_ptr<Impl> impl) { return World{std::move(impl)}; });
}

World::World(const WorldSettings& settings, const FluidParameters& parameters)
    : _impl(
         [&]()
         {
            auto impl = validate(settings).and_then([&]() { return Impl::create(settings, parameters); });
            if (!impl)
            {
               rejectSettings(impl.error());
            }
            return std::move(*impl);
         }()
      )
{
}

World::World(std::unique_ptr<Impl> impl) : _impl(std::move(impl))
{
}

World::~World() = default;
World::World(World&& other) noexcept = default;
World& World::operator=(World&& other) noexcept = default;

const WorldSettings& World::settings() const
{
   return _impl->_solver.settings();
}

std::string_view World::deviceDescription() const
{
   return _impl->deviceDescription();
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
   return _impl->colliders().add(_impl->nextId<ColliderId>(), shape);
}

bool World::setCollider(ColliderId id, const Shape& shape)
{
   auto* const stored = _impl->colliders().find(id);
   if (stored != nullptr)
   {
      *stored = shape;
   }
   return stored != nullptr;
}

bool World::removeCollider(ColliderId id)
{
   return _impl->colliders().remove(id);
}

std::optional<Shape> World::collider(ColliderId id) const
{
   const auto* const stored = _impl->colliders().find(id);
   return stored != nullptr ? std::optional{*stored} : std::nullopt;
}

ForceFieldId World::addForceField(const ForceField& field)
{
   return _impl->forceFields().add(_impl->nextId<ForceFieldId>(), field);
}

bool World::setForceField(ForceFieldId id, const ForceField& field)
{
   auto* const stored = _impl->forceFields().find(id);
   if (stored != nullptr)
   {
      *stored = field;
   }
   return stored != nullptr;
}

bool World::removeForceField(ForceFieldId id)
{
   return _impl->forceFields().remove(id);
}

std::optional<ForceField> World::forceField(ForceFieldId id) const
{
   const auto* const stored = _impl->forceFields().find(id);
   return stored != nullptr ? std::optional{*stored} : std::nullopt;
}

EmitterId World::addEmitter(const Emitter& emitter)
{
   return _impl->emitters().add(_impl->nextId<EmitterId>(), {emitter, 0.0f});
}

bool World::setEmitter(EmitterId id, const Emitter& emitter)
{
   auto* const stored = _impl->emitters().find(id);
   if (stored != nullptr)
   {
      // keeps the travelled distance so editing a running emitter does not cause a gap or a double layer
      stored->_emitter = emitter;
   }
   return stored != nullptr;
}

bool World::removeEmitter(EmitterId id)
{
   return _impl->emitters().remove(id);
}

std::optional<Emitter> World::emitter(EmitterId id) const
{
   const auto* const stored = _impl->emitters().find(id);
   return stored != nullptr ? std::optional{stored->_emitter} : std::nullopt;
}

StepStatistics World::step(Seconds time, int32_t max_substeps)
{
   return _impl->step(time.count(), max_substeps);
}

}  // namespace tinysph
