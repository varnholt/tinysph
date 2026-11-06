#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

#include "tinysph/math.h"
#include "tinysph/types.h"

namespace tinysph
{

/// \brief a fluid simulation: particles inside a box shaped domain, plus colliders, force fields and emitters.
///
/// typical use:
/// \code
///    using namespace std::chrono_literals;
///    tinysph::World world{{._domain = {{0, 0, 0}, {1, 1, 1}}}};
///    world.addFluid(tinysph::Box{{0.25f, 0.3f, 0.5f}, {0.2f, 0.25f, 0.45f}});
///    world.addCollider(tinysph::Sphere{{0.7f, 0.1f, 0.5f}, 0.1f});
///    while (running)
///    {
///       world.step(1s / 60.0);
///       draw(world.positions());
///    }
/// \endcode
///
/// the implementation is hidden behind a pointer, so internals and backends can change without breaking code
/// that uses this class. a world is movable but not copyable. it is not thread safe: call it from one thread
/// at a time; it uses multiple threads internally.
class World
{
public:
   /// \brief creates a world, or tells why the settings are invalid. use this when settings come from user input.
   [[nodiscard]] static std::expected<World, SettingsError>
   create(const WorldSettings& settings = {}, const FluidParameters& parameters = {});

   /// \brief checks settings without creating a world.
   [[nodiscard]] static std::expected<void, SettingsError> validate(const WorldSettings& settings);

   /// \brief creates a world from settings that are known to be valid.
   ///        invalid settings throw std::invalid_argument, a missing gpu throws std::runtime_error;
   ///        both terminate when exceptions are disabled.
   explicit World(const WorldSettings& settings = {}, const FluidParameters& parameters = {});
   ~World();
   World(World&& other) noexcept;
   World& operator=(World&& other) noexcept;
   World(const World&) = delete;
   World& operator=(const World&) = delete;

   // ------------------------------------------------------------------------------------------------
   // configuration
   // ------------------------------------------------------------------------------------------------

   [[nodiscard]] const WorldSettings& settings() const;

   /// \brief what the simulation runs on, e.g. "cpu" or "NVIDIA GeForce RTX 4070 (vulkan)".
   [[nodiscard]] std::string_view deviceDescription() const;

   [[nodiscard]] const FluidParameters& parameters() const;
   void setParameters(const FluidParameters& parameters);

   [[nodiscard]] const Aabb& domain() const;

   /// \brief changes the domain; particles outside are pushed back in during the next step.
   void setDomain(const Aabb& domain);

   // ------------------------------------------------------------------------------------------------
   // particles
   // ------------------------------------------------------------------------------------------------

   /// \brief fills a shape with particles at rest spacing, skipping space covered by colliders or outside the domain.
   /// \return number of particles added
   std::size_t addFluid(const Shape& shape, const Vec3& velocity = {});

   /// \brief adds one particle.
   /// \return false if it is outside the domain, inside a collider, or the particle cap is reached
   bool addParticle(const Vec3& position, const Vec3& velocity = {});

   void clearParticles();

   [[nodiscard]] std::size_t particleCount() const;
   [[nodiscard]] float particleMass() const;    //!< in kg, chosen so that particles at rest spacing have the rest density
   [[nodiscard]] float particleRadius() const;  //!< half the spacing in m, used for collisions and drawing

   /// \brief particle data, valid until the next non-const call. the order changes between steps.
   [[nodiscard]] std::span<const Vec3> positions() const;
   [[nodiscard]] std::span<const Vec3> velocities() const;
   [[nodiscard]] std::span<const float> densities() const;

   // ------------------------------------------------------------------------------------------------
   // colliders, force fields, emitters
   // ------------------------------------------------------------------------------------------------

   ColliderId addCollider(const Shape& shape);
   bool setCollider(ColliderId id, const Shape& shape);  //!< false if the id is unknown
   bool removeCollider(ColliderId id);                   //!< false if the id is unknown
   [[nodiscard]] std::optional<Shape> collider(ColliderId id) const;

   ForceFieldId addForceField(const ForceField& field);
   bool setForceField(ForceFieldId id, const ForceField& field);
   bool removeForceField(ForceFieldId id);
   [[nodiscard]] std::optional<ForceField> forceField(ForceFieldId id) const;

   EmitterId addEmitter(const Emitter& emitter);
   bool setEmitter(EmitterId id, const Emitter& emitter);
   bool removeEmitter(EmitterId id);
   [[nodiscard]] std::optional<Emitter> emitter(EmitterId id) const;

   // ------------------------------------------------------------------------------------------------
   // simulation
   // ------------------------------------------------------------------------------------------------

   /// \brief advances the simulation by the given time using adaptive substeps.
   ///        stops after max_substeps, so a slow machine runs in slow motion instead of freezing;
   ///        StepStatistics::_simulated_time tells how far it got.
   StepStatistics step(Seconds time, int32_t max_substeps = 1000);

private:
   struct Impl;
   explicit World(std::unique_ptr<Impl> impl);
   std::unique_ptr<Impl> _impl;
};

}  // namespace tinysph
