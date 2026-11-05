// tests that only use the public api, the way a library user would

#include <algorithm>
#include <cmath>
#include <print>
#include <utility>
#include <vector>

#include "test_framework.h"
#include "test_scenes.h"
#include "tinysph/tinysph.h"

namespace
{

bool isFinite(const tinysph::Vec3& value)
{
   return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

float meanSpeed(const tinysph::World& world)
{
   auto speed_sum = 0.0f;
   for (const auto& velocity : world.velocities())
   {
      speed_sum += tinysph::length(velocity);
   }
   return speed_sum / static_cast<float>(world.particleCount());
}

}  // namespace

TEST_CASE("version header matches the cmake project version")
{
   CHECK(tinysph::version.major == TINYSPH_PROJECT_VERSION_MAJOR);
   CHECK(tinysph::version.minor == TINYSPH_PROJECT_VERSION_MINOR);
   CHECK(tinysph::version.patch == TINYSPH_PROJECT_VERSION_PATCH);
}

TEST_CASE("handles are unique, stale handles are rejected")
{
   tinysph::World world;
   const auto first = world.addCollider(tinysph::Sphere{{0.5f, 0.5f, 0.5f}, 0.1f});
   const auto second = world.addCollider(tinysph::Box{{0.2f, 0.2f, 0.2f}, {0.05f, 0.05f, 0.05f}});
   CHECK(first != second);

   CHECK(world.setCollider(first, tinysph::Sphere{{0.5f, 0.5f, 0.5f}, 0.2f}));
   const auto updated = world.collider(first);
   CHECK(updated.has_value() && std::get<tinysph::Sphere>(*updated)._radius == 0.2f);

   CHECK(world.removeCollider(first));
   CHECK(!world.removeCollider(first));
   CHECK(!world.collider(first).has_value());
   CHECK(!world.setCollider(first, tinysph::Sphere{}));
   CHECK(world.collider(second).has_value());

   // ids are never reused, so the stale handle cannot reach the new object
   const auto third = world.addCollider(tinysph::Sphere{});
   CHECK(third != first);
   CHECK(!world.collider(first).has_value());

   const auto field = world.addForceField(tinysph::Attractor{});
   CHECK(world.forceField(field).has_value());
   CHECK(world.removeForceField(field));
   CHECK(!world.forceField(field).has_value());

   const auto emitter = world.addEmitter(tinysph::Emitter{});
   CHECK(world.setEmitter(emitter, tinysph::Emitter{._enabled = false}));
   CHECK(world.emitter(emitter).has_value() && !world.emitter(emitter)->_enabled);
   CHECK(world.removeEmitter(emitter));
}

TEST_CASE("addFluid fills shapes, but not colliders or space outside the domain")
{
   tinysph::World world{{._domain = {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}}}};
   const auto spacing = world.settings()._particle_spacing;

   const auto box_count = world.addFluid(tinysph::Box{{0.5f, 0.5f, 0.5f}, {0.1f, 0.1f, 0.1f}});
   CHECK_MESSAGE(box_count == 8 * 8 * 8, "{} particles in a 0.2 m box at {} m spacing", box_count, spacing);

   world.clearParticles();
   world.addCollider(tinysph::Sphere{{0.5f, 0.5f, 0.5f}, 0.1f});
   const auto around_collider = world.addFluid(tinysph::Box{{0.5f, 0.5f, 0.5f}, {0.1f, 0.1f, 0.1f}});
   CHECK(around_collider < box_count / 2);

   world.clearParticles();
   const auto half_outside = world.addFluid(tinysph::Box{{1.0f, 0.5f, 0.5f}, {0.1f, 0.1f, 0.1f}});
   CHECK_MESSAGE(half_outside > box_count / 4 && half_outside <= box_count / 2, "{} of {} particles inside", half_outside, box_count);
   CHECK(!world.addParticle({2.0f, 0.5f, 0.5f}));
}

TEST_CASE("particles follow a shrinking domain")
{
   tinysph::World world{{._domain = {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}}}};
   world.addFluid(tinysph::Box{{0.5f, 0.2f, 0.5f}, {0.45f, 0.15f, 0.45f}});
   world.setDomain({{0.0f, 0.0f, 0.0f}, {0.6f, 1.0f, 1.0f}});
   world.step(1.0f / 60.0f);

   const auto& domain = world.domain();
   const auto outside = std::ranges::count_if(world.positions(), [&](const tinysph::Vec3& position) { return !domain.contains(position); });
   CHECK_MESSAGE(outside == 0, "{} particles outside the new domain", outside);
}

TEST_CASE("a world can be moved")
{
   auto world = test::makeDamBreakWorld();
   const auto particle_count = world.particleCount();
   tinysph::World moved{std::move(world)};
   moved.step(1.0f / 60.0f);
   CHECK(moved.particleCount() == particle_count);
}

TEST_CASE("dam break stays stable and inside the domain")
{
   auto world = test::makeDamBreakWorld();
   const auto particle_count = world.particleCount();
   std::println("   dam break: {} particles", particle_count);

   const auto rest_density = world.settings()._rest_density;
   auto max_density_ratio = 0.0f;
   auto max_p99_density_ratio = 0.0f;
   auto wall_ms = 0.0;
   std::vector<float> sorted_densities;
   for (auto frame = 0; frame < 90; ++frame)
   {
      const auto statistics = world.step(1.0f / 60.0f);
      max_density_ratio = std::max(max_density_ratio, statistics._max_density_ratio);
      wall_ms += statistics._step_ms;

      // single particles spike at impacts; the 99th percentile shows how compressible the bulk is
      sorted_densities.assign(world.densities().begin(), world.densities().end());
      const auto p99_position = sorted_densities.begin() + static_cast<std::ptrdiff_t>(sorted_densities.size() * 99 / 100);
      std::ranges::nth_element(sorted_densities, p99_position);
      max_p99_density_ratio = std::max(max_p99_density_ratio, *p99_position / rest_density);
   }
   std::println(
      "   1.5 s simulated in {:.2f} s, max density ratio {:.3f}, p99 {:.3f}", wall_ms / 1000.0, max_density_ratio, max_p99_density_ratio
   );

   CHECK(world.particleCount() == particle_count);
   CHECK_MESSAGE(max_p99_density_ratio < 1.1f, "p99 density ratio {:.3f}", max_p99_density_ratio);
   CHECK_MESSAGE(max_density_ratio < 1.3f, "max density ratio {:.3f}", max_density_ratio);

   const auto& domain = world.domain();
   const tinysph::Vec3 obstacle_center{1.1f, 0.12f, 0.3f};
   auto escaped_count = 0;
   auto non_finite_count = 0;
   auto inside_collider_count = 0;
   for (std::size_t particle_index = 0; particle_index < world.particleCount(); ++particle_index)
   {
      const auto& position = world.positions()[particle_index];
      if (!isFinite(position) || !isFinite(world.velocities()[particle_index]))
      {
         ++non_finite_count;
         continue;
      }
      if (!domain.contains(position))
      {
         ++escaped_count;
      }
      if (tinysph::length(position - obstacle_center) < 0.12f)
      {
         ++inside_collider_count;
      }
   }
   CHECK_MESSAGE(non_finite_count == 0, "{} particles are nan/inf", non_finite_count);
   CHECK_MESSAGE(escaped_count == 0, "{} particles left the domain", escaped_count);
   CHECK_MESSAGE(inside_collider_count == 0, "{} particles are inside the sphere collider", inside_collider_count);
}

TEST_CASE("dam break comes to rest")
{
   // without artificial viscosity wcsph keeps 'boiling': after 6 s most particles still moved faster than 0.5 m/s
   auto world = test::makeDamBreakWorld();
   for (auto frame = 0; frame < 240; ++frame)
   {
      world.step(1.0f / 60.0f);
   }
   const auto mean_speed = meanSpeed(world);
   CHECK_MESSAGE(mean_speed < 0.05f, "mean speed after 4 s is {:.3f} m/s", mean_speed);
}

TEST_CASE("emitter spawns whole layers at the expected rate")
{
   tinysph::World world{{._domain = {{0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 1.0f}}}, {._gravity = {}}};
   const auto spacing = world.settings()._particle_spacing;
   const auto speed = 2.0f;
   world.addEmitter({._position = {0.5f, 0.1f, 0.5f}, ._velocity = {0.0f, speed, 0.0f}, ._radius = 0.04f});

   // slightly more than the time for one layer
   const auto first_time = spacing / speed * 1.01f;
   world.step(first_time);
   const auto layer_size = world.particleCount();
   CHECK(layer_size > 0);

   for (auto frame = 0; frame < 6; ++frame)
   {
      world.step(1.0f / 60.0f);
   }

   const auto expected_layers = std::floor(speed * (first_time + 6.0f / 60.0f) / spacing);
   const auto layers = static_cast<float>(world.particleCount()) / static_cast<float>(layer_size);
   CHECK_MESSAGE(std::abs(layers - expected_layers) <= 1.0f, "{} layers emitted, expected {}", layers, expected_layers);
}

TEST_CASE("an attractor pulls fluid towards it")
{
   tinysph::World world{{._domain = {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}}}, {._gravity = {}}};
   world.addFluid(tinysph::Sphere{{0.3f, 0.5f, 0.5f}, 0.1f});
   world.addForceField(tinysph::Attractor{{0.7f, 0.5f, 0.5f}, 30.0f, 0.6f});
   world.step(0.2f);

   auto center_x = 0.0f;
   for (const auto& position : world.positions())
   {
      center_x += position.x;
   }
   center_x /= static_cast<float>(world.particleCount());
   CHECK_MESSAGE(center_x > 0.35f, "fluid center moved to x = {:.3f}", center_x);
}

int main()
{
   return test::runAll();
}
