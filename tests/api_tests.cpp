// tests that only use the public api, the way a library user would

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <print>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "test_framework.h"
#include "test_scenes.h"
#include "tinysph/tinysph.h"

using namespace std::chrono_literals;

namespace
{

constexpr auto frame_time = 1s / 60.0;

bool isFinite(const tinysph::Vec3& value)
{
   return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

float meanSpeed(const tinysph::World& world)
{
   const auto velocities = world.velocities();
   const auto speed_sum = std::transform_reduce(velocities.begin(), velocities.end(), 0.0f, std::plus{}, tinysph::length);
   return speed_sum / static_cast<float>(world.particleCount());
}

}  // namespace

TEST_CASE("version header matches the cmake project version")
{
   CHECK(tinysph::version.major == TINYSPH_PROJECT_VERSION_MAJOR);
   CHECK(tinysph::version.minor == TINYSPH_PROJECT_VERSION_MINOR);
   CHECK(tinysph::version.patch == TINYSPH_PROJECT_VERSION_PATCH);
}

TEST_CASE("invalid settings are reported, not crashed on")
{
   CHECK(tinysph::World::validate({}).has_value());
   CHECK(tinysph::World::create().has_value());

   const auto check_rejected = [](const tinysph::WorldSettings& settings, tinysph::SettingsError expected_error)
   {
      const auto world = tinysph::World::create(settings);
      CHECK_MESSAGE(!world.has_value() && world.error() == expected_error, "expected '{}'", tinysph::describe(expected_error));
   };
   check_rejected({._particle_spacing = 0.0f}, tinysph::SettingsError::NonPositiveSpacing);
   check_rejected({._particle_spacing = std::nanf("")}, tinysph::SettingsError::NonPositiveSpacing);
   check_rejected({._rest_density = -1.0f}, tinysph::SettingsError::NonPositiveRestDensity);
   check_rejected({._support_radius_factor = 0.5f}, tinysph::SettingsError::SupportRadiusTooSmall);
   check_rejected({._domain = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f}}}, tinysph::SettingsError::EmptyDomain);
   check_rejected(
      {._domain = {{0.0f, 0.0f, 0.0f}, {100.0f, 100.0f, 100.0f}}, ._particle_spacing = 0.001f}, tinysph::SettingsError::TooManyGridCells
   );
   check_rejected({._max_particles = 0}, tinysph::SettingsError::ZeroParticleCap);
   check_rejected({._backend = static_cast<tinysph::Backend>(42)}, tinysph::SettingsError::UnsupportedBackend);

   auto threw = false;
   try
   {
      tinysph::World world{{._particle_spacing = -1.0f}};
   }
   catch (const std::invalid_argument&)
   {
      threw = true;
   }
   CHECK(threw);
}

TEST_CASE("step takes any std::chrono duration")
{
   tinysph::World world{test::onBackend({}), {._gravity = {}}};
   world.addFluid(tinysph::Sphere{{0.5f, 0.5f, 0.5f}, 0.1f});

   // the substeps add up in float, so compare with a tolerance
   const auto matches = [](tinysph::Seconds simulated, tinysph::Seconds requested)
   { return std::abs((simulated - requested).count()) < 1e-6f; };
   CHECK(matches(world.step(16ms)._simulated_time, 16ms));
   CHECK(matches(world.step(frame_time)._simulated_time, frame_time));

   // max_substeps caps the work; the statistics tell how far it got
   const auto capped = world.step(1s, 3);
   CHECK(capped._substeps == 3);
   CHECK(capped._simulated_time < 1s);
}

TEST_CASE("handles are unique, stale handles are rejected")
{
   tinysph::World world{test::onBackend({})};
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
   CHECK(world.emitter(emitter).transform([](const tinysph::Emitter& stored) { return !stored._enabled; }).value_or(false));
   CHECK(world.removeEmitter(emitter));
}

TEST_CASE("addFluid fills shapes, but not colliders or space outside the domain")
{
   tinysph::World world{test::onBackend({._domain = {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}}})};
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
   tinysph::World world{test::onBackend({._domain = {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}}})};
   world.addFluid(tinysph::Box{{0.5f, 0.2f, 0.5f}, {0.45f, 0.15f, 0.45f}});
   world.setDomain({{0.0f, 0.0f, 0.0f}, {0.6f, 1.0f, 1.0f}});
   world.step(frame_time);

   const auto& domain = world.domain();
   const auto outside = std::ranges::count_if(world.positions(), [&](const tinysph::Vec3& position) { return !domain.contains(position); });
   CHECK_MESSAGE(outside == 0, "{} particles outside the new domain", outside);
}

TEST_CASE("a world can be moved")
{
   auto world = test::makeDamBreakWorld();
   const auto particle_count = world.particleCount();
   tinysph::World moved{std::move(world)};
   moved.step(frame_time);
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
   tinysph::Milliseconds wall_time{};
   std::vector<float> sorted_densities;
   for (auto frame = 0; frame < 90; ++frame)
   {
      const auto statistics = world.step(frame_time);
      max_density_ratio = std::max(max_density_ratio, statistics._max_density_ratio);
      wall_time += statistics._wall_time;

      // single particles spike at impacts; the 99th percentile shows how compressible the bulk is
      sorted_densities.assign(world.densities().begin(), world.densities().end());
      const auto p99_position = sorted_densities.begin() + static_cast<std::ptrdiff_t>(sorted_densities.size() * 99 / 100);
      std::ranges::nth_element(sorted_densities, p99_position);
      max_p99_density_ratio = std::max(max_p99_density_ratio, *p99_position / rest_density);
   }
   std::println(
      "   1.5 s simulated in {:.2f} s, max density ratio {:.3f}, p99 {:.3f}",
      std::chrono::duration<double>{wall_time}.count(),
      max_density_ratio,
      max_p99_density_ratio
   );

   CHECK(world.particleCount() == particle_count);
   CHECK_MESSAGE(max_p99_density_ratio < 1.1f, "p99 density ratio {:.3f}", max_p99_density_ratio);
   CHECK_MESSAGE(max_density_ratio < 1.3f, "max density ratio {:.3f}", max_density_ratio);

   const auto& domain = world.domain();
   const tinysph::Vec3 obstacle_center{1.1f, 0.12f, 0.3f};
   auto escaped_count = 0;
   auto non_finite_count = 0;
   auto inside_collider_count = 0;
   for (const auto& [position, velocity] : std::views::zip(world.positions(), world.velocities()))
   {
      if (!isFinite(position) || !isFinite(velocity))
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
      world.step(frame_time);
   }
   const auto mean_speed = meanSpeed(world);
   CHECK_MESSAGE(mean_speed < 0.05f, "mean speed after 4 s is {:.3f} m/s", mean_speed);
}

TEST_CASE("emitter spawns whole layers at the expected rate")
{
   tinysph::World world{test::onBackend({._domain = {{0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 1.0f}}}), {._gravity = {}}};
   const auto spacing = world.settings()._particle_spacing;
   const auto speed = 2.0f;
   world.addEmitter({._position = {0.5f, 0.1f, 0.5f}, ._velocity = {0.0f, speed, 0.0f}, ._radius = 0.04f});

   // slightly more than the time for one layer
   const tinysph::Seconds first_time{spacing / speed * 1.01f};
   world.step(first_time);
   const auto layer_size = world.particleCount();
   CHECK(layer_size > 0);

   for (auto frame = 0; frame < 6; ++frame)
   {
      world.step(frame_time);
   }

   const auto total_time = first_time + 6 * tinysph::Seconds{frame_time};
   const auto expected_layers = std::floor(speed * total_time.count() / spacing);
   const auto layers = static_cast<float>(world.particleCount()) / static_cast<float>(layer_size);
   CHECK_MESSAGE(std::abs(layers - expected_layers) <= 1.0f, "{} layers emitted, expected {}", layers, expected_layers);
}

TEST_CASE("surface tension merges colliding drops")
{
   // without it the drops burst into spray: after 1 s about half of the particles are isolated
   tinysph::World world{test::onBackend({._domain = {{0.0f, 0.0f, 0.0f}, {1.6f, 1.0f, 1.0f}}}), {._gravity = {}, ._surface_tension = 1.0f}};
   world.addFluid(tinysph::Sphere{{0.45f, 0.5f, 0.5f}, 0.15f}, {1.0f, 0.0f, 0.0f});
   world.addFluid(tinysph::Sphere{{1.15f, 0.55f, 0.5f}, 0.15f}, {-1.0f, 0.0f, 0.0f});
   for (auto frame = 0; frame < 60; ++frame)
   {
      world.step(frame_time);
   }

   const auto rest_density = world.settings()._rest_density;
   const auto isolated = std::ranges::count_if(world.densities(), [&](float density) { return density < 0.35f * rest_density; });
   CHECK_MESSAGE(isolated < 20, "{} of {} particles isolated", isolated, world.particleCount());

   // one merged sphere of both volumes has a mean distance to its center of about 0.14 m
   const auto positions = world.positions();
   const auto center = std::accumulate(positions.begin(), positions.end(), tinysph::Vec3{}) / static_cast<float>(positions.size());
   const auto spread = std::transform_reduce(
                          positions.begin(),
                          positions.end(),
                          0.0f,
                          std::plus{},
                          [&](const tinysph::Vec3& position) { return tinysph::length(position - center); }
                       ) /
                       static_cast<float>(positions.size());
   CHECK_MESSAGE(spread < 0.2f, "mean distance to the center {:.3f} m", spread);
}

TEST_CASE("an attractor pulls fluid towards it")
{
   tinysph::World world{test::onBackend({._domain = {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}}}), {._gravity = {}}};
   world.addFluid(tinysph::Sphere{{0.3f, 0.5f, 0.5f}, 0.1f});
   world.addForceField(tinysph::Attractor{{0.7f, 0.5f, 0.5f}, 30.0f, 0.6f});
   world.step(200ms);

   const auto positions = world.positions();
   const auto x_sum = std::transform_reduce(
      positions.begin(), positions.end(), 0.0f, std::plus{}, [](const tinysph::Vec3& position) { return position.x; }
   );
   const auto center_x = x_sum / static_cast<float>(world.particleCount());
   CHECK_MESSAGE(center_x > 0.35f, "fluid center moved to x = {:.3f}", center_x);
}

// ----------------------------------------------------------------------------------------------------
// gpu only: the same scene on both backends
// ----------------------------------------------------------------------------------------------------

namespace
{

/// \brief dam break plus a box collider and every kind of force field, so each gpu code path runs.
tinysph::World makeMixedScene(tinysph::Backend backend)
{
   tinysph::World world{{._domain = {{0.0f, 0.0f, 0.0f}, {1.6f, 1.0f, 0.6f}}, ._backend = backend}};
   world.addCollider(tinysph::Sphere{{1.1f, 0.12f, 0.3f}, 0.12f});
   world.addCollider(tinysph::Box{{0.75f, 0.06f, 0.3f}, {0.04f, 0.06f, 0.2f}});
   world.addForceField(tinysph::Vortex{{0.3f, 0.3f, 0.3f}, {0.0f, 1.0f, 0.0f}, 5.0f, 0.2f});
   world.addForceField(tinysph::Wind{{{0.0f, 0.4f, 0.0f}, {0.5f, 1.0f, 0.6f}}, {2.0f, 0.0f, 0.0f}});
   world.addForceField(tinysph::Attractor{{1.3f, 0.3f, 0.3f}, 5.0f, 0.3f});
   world.addFluid(tinysph::Box{{0.26f, 0.31f, 0.3f}, {0.24f, 0.29f, 0.28f}});
   return world;
}

tinysph::Vec3 centerOfMass(const tinysph::World& world)
{
   const auto positions = world.positions();
   return std::accumulate(positions.begin(), positions.end(), tinysph::Vec3{}) / static_cast<float>(positions.size());
}

}  // namespace

TEST_CASE("gpu: one substep matches the cpu")
{
   if (test::backend() != tinysph::Backend::Gpu)
   {
      return;
   }
   auto cpu = makeMixedScene(tinysph::Backend::Cpu);
   auto gpu = makeMixedScene(tinysph::Backend::Gpu);
   CHECK(cpu.particleCount() == gpu.particleCount());

   // one short substep from identical particles; the order differs, so compare the sorted densities
   cpu.step(1ms);
   gpu.step(1ms);
   std::vector<float> cpu_densities(cpu.densities().begin(), cpu.densities().end());
   std::vector<float> gpu_densities(gpu.densities().begin(), gpu.densities().end());
   std::ranges::sort(cpu_densities);
   std::ranges::sort(gpu_densities);
   auto max_difference = 0.0f;
   for (const auto& [cpu_density, gpu_density] : std::views::zip(cpu_densities, gpu_densities))
   {
      max_difference = std::max(max_difference, std::abs(cpu_density - gpu_density) / cpu_density);
   }
   const auto offset = tinysph::length(centerOfMass(cpu) - centerOfMass(gpu));
   std::println(
      "   {} particles, densities differ by up to {:.2e}, centers of mass {:.2e} m apart", cpu.particleCount(), max_difference, offset
   );
   CHECK_MESSAGE(max_difference < 1e-4f, "densities differ by up to {:.2e}", max_difference);
   CHECK_MESSAGE(offset < 1e-5f, "centers of mass {:.2e} m apart", offset);
}

TEST_CASE("gpu: half a second of flow matches the cpu")
{
   if (test::backend() != tinysph::Backend::Gpu)
   {
      return;
   }
   auto cpu = makeMixedScene(tinysph::Backend::Cpu);
   auto gpu = makeMixedScene(tinysph::Backend::Gpu);

   // float round-off grows chaotically in a splash, so only the bulk motion can be compared after many substeps
   tinysph::StepStatistics cpu_statistics;
   tinysph::StepStatistics gpu_statistics;
   for (auto frame = 0; frame < 30; ++frame)
   {
      cpu_statistics = cpu.step(frame_time);
      gpu_statistics = gpu.step(frame_time);
   }
   const auto offset = tinysph::length(centerOfMass(cpu) - centerOfMass(gpu));
   CHECK_MESSAGE(offset < 0.005f, "centers of mass {:.4f} m apart", offset);

   const auto cpu_speed = meanSpeed(cpu);
   const auto gpu_speed = meanSpeed(gpu);
   CHECK_MESSAGE(std::abs(cpu_speed - gpu_speed) < 0.03f * cpu_speed, "mean speed cpu {:.3f}, gpu {:.3f} m/s", cpu_speed, gpu_speed);

   const auto density_difference = std::abs(cpu_statistics._average_density_ratio - gpu_statistics._average_density_ratio);
   CHECK_MESSAGE(density_difference < 0.005f, "average density ratio differs by {:.4f}", density_difference);
   std::println(
      "   centers of mass {:.4f} m apart, mean speed cpu {:.3f} gpu {:.3f} m/s, density ratio cpu {:.4f} gpu {:.4f}; last frame cpu "
      "{:.1f} ms, gpu {:.1f} ms",
      offset,
      cpu_speed,
      gpu_speed,
      cpu_statistics._average_density_ratio,
      gpu_statistics._average_density_ratio,
      cpu_statistics._wall_time.count(),
      gpu_statistics._wall_time.count()
   );
}

TEST_CASE("gpu: emitters outgrow the gpu buffers")
{
   if (test::backend() != tinysph::Backend::Gpu)
   {
      return;
   }
   const auto run = [](tinysph::Backend backend)
   {
      tinysph::World world{{._domain = {{0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 1.0f}}, ._backend = backend}, {._gravity = {}}};
      // 3.1 m/s emits 2.07 layers of 0.025 m per frame. a whole number of layers per frame would end every frame on a
      // layer boundary, where float round-off in the time steps decides whether the last layer is emitted
      world.addEmitter({._position = {0.5f, 0.1f, 0.5f}, ._velocity = {0.0f, 3.1f, 0.0f}, ._radius = 0.2f});
      for (auto frame = 0; frame < 20; ++frame)
      {
         world.step(frame_time);
      }
      return world;
   };
   const auto cpu = run(tinysph::Backend::Cpu);
   const auto gpu = run(tinysph::Backend::Gpu);
   CHECK_MESSAGE(cpu.particleCount() == gpu.particleCount(), "cpu {} particles, gpu {}", cpu.particleCount(), gpu.particleCount());
   CHECK_MESSAGE(gpu.particleCount() > 4096, "only {} particles, the buffers never grew", gpu.particleCount());

   const auto& domain = gpu.domain();
   const auto broken = std::ranges::count_if(
      gpu.positions(), [&](const tinysph::Vec3& position) { return !isFinite(position) || !domain.contains(position); }
   );
   CHECK_MESSAGE(broken == 0, "{} particles are nan or outside", broken);

   const auto offset = tinysph::length(centerOfMass(cpu) - centerOfMass(gpu));
   std::println("   {} particles emitted, centers of mass {:.2e} m apart", gpu.particleCount(), offset);
   CHECK_MESSAGE(offset < 0.005f, "centers of mass {:.4f} m apart", offset);
}

int main()
{
   // every test runs on each backend that is built in and finds a device; ci sets TINYSPH_REQUIRE_GPU to fail
   // instead of skipping when the gpu backend cannot run
   std::vector backends{tinysph::Backend::Cpu};
   const auto gpu_world = tinysph::World::create({._backend = tinysph::Backend::Gpu});
   if (gpu_world)
   {
      std::println("gpu backend on {}", gpu_world->deviceDescription());
      backends.push_back(tinysph::Backend::Gpu);
   }
   else
   {
      std::println("gpu backend skipped: {}", tinysph::describe(gpu_world.error()));
      // ci sets the variable on every job, empty where no gpu is expected
      const auto* const require_gpu = std::getenv("TINYSPH_REQUIRE_GPU");
      if (require_gpu != nullptr && std::string_view{require_gpu} != "" && std::string_view{require_gpu} != "0")
      {
         std::println("TINYSPH_REQUIRE_GPU is set, failing");
         return 1;
      }
   }

   auto exit_code = 0;
   for (const auto backend : backends)
   {
      std::println("===== {} backend =====", backend == tinysph::Backend::Gpu ? "gpu" : "cpu");
      test::backend() = backend;
      exit_code = test::runAll();
   }
   return exit_code;
}
