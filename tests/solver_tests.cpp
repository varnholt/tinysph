#include <algorithm>
#include <chrono>
#include <cmath>
#include <print>
#include <random>
#include <vector>

#include "sph/solver.h"
#include "test_framework.h"

namespace
{

bool isFinite(const sph::Vec3& value)
{
   return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

sph::Scene makeEmptyScene(const sph::Aabb& container, const sph::Vec3& gravity)
{
   sph::Scene scene;
   scene._name = "test";
   scene._container = container;
   scene._gravity = gravity;
   return scene;
}

}  // namespace

TEST_CASE("neighbor search matches brute force")
{
   sph::Solver solver;
   solver.loadScene(makeEmptyScene({{0.0f, 0.0f, 0.0f}, {0.5f, 0.4f, 0.3f}}, {}));

   std::mt19937 generator{1234u};
   std::uniform_real_distribution<float> distribution{0.0f, 1.0f};
   for (auto particle_number = 0; particle_number < 3000; ++particle_number)
   {
      const sph::Vec3 position{distribution(generator) * 0.5f, distribution(generator) * 0.4f, distribution(generator) * 0.3f};
      solver.addParticle(position, {});
   }
   CHECK(solver.particleCount() > 2000);

   solver.rebuildGrid();
   const auto positions = solver.positions();
   const auto support_radius_squared = solver.supportRadius() * solver.supportRadius();

   auto mismatches = 0;
   for (std::size_t particle_index = 0; particle_index < positions.size(); ++particle_index)
   {
      std::vector<std::size_t> grid_neighbors;
      solver.forEachNeighbor(
         particle_index, [&](std::size_t neighbor_index, const sph::Vec3&, float) { grid_neighbors.push_back(neighbor_index); }
      );

      std::vector<std::size_t> brute_force_neighbors;
      for (std::size_t neighbor_index = 0; neighbor_index < positions.size(); ++neighbor_index)
      {
         if (sph::lengthSquared(positions[particle_index] - positions[neighbor_index]) < support_radius_squared)
         {
            brute_force_neighbors.push_back(neighbor_index);
         }
      }

      std::ranges::sort(grid_neighbors);
      if (grid_neighbors != brute_force_neighbors)
      {
         ++mismatches;
      }
   }
   CHECK_MESSAGE(mismatches == 0, "{} particles had a different neighbor set", mismatches);
}

TEST_CASE("resting lattice has the rest density")
{
   sph::Solver solver;
   solver.loadScene(makeEmptyScene({{0.0f, 0.0f, 0.0f}, {0.4f, 0.4f, 0.4f}}, {}));
   solver.addFluidBox({{0.0f, 0.0f, 0.0f}, {0.4f, 0.4f, 0.4f}}, {});
   solver.step(1e-6f);

   const auto rest_density = solver.settings()._rest_density;
   const auto interior_margin = solver.supportRadius() * 1.5f;
   const sph::Aabb interior{
      {interior_margin, interior_margin, interior_margin}, {0.4f - interior_margin, 0.4f - interior_margin, 0.4f - interior_margin}
   };

   auto interior_count = 0;
   auto worst_error = 0.0f;
   for (std::size_t particle_index = 0; particle_index < solver.particleCount(); ++particle_index)
   {
      if (interior.contains(solver.positions()[particle_index]))
      {
         ++interior_count;
         worst_error = std::max(worst_error, std::abs(solver.densities()[particle_index] / rest_density - 1.0f));
      }
   }
   CHECK(interior_count > 100);
   CHECK_MESSAGE(worst_error < 0.01f, "worst interior density error {:.4f}", worst_error);
}

TEST_CASE("momentum is conserved without walls and gravity")
{
   sph::Solver solver;
   solver.loadScene(makeEmptyScene({{0.0f, 0.0f, 0.0f}, {4.0f, 4.0f, 4.0f}}, {}));
   solver.addFluidSphere({1.65f, 2.0f, 2.0f}, 0.15f, {1.0f, 0.0f, 0.0f});
   solver.addFluidSphere({2.35f, 2.05f, 2.0f}, 0.15f, {-1.0f, 0.0f, 0.0f});

   auto momentum_scale = 0.0f;
   for (const auto& velocity : solver.velocities())
   {
      momentum_scale += sph::length(velocity) * solver.particleMass();
   }
   const auto initial_momentum = solver.totalMomentum();

   for (auto frame = 0; frame < 24; ++frame)
   {
      solver.advance(1.0f / 60.0f, 1000);
   }

   const auto drift = sph::length(solver.totalMomentum() - initial_momentum) / momentum_scale;
   CHECK_MESSAGE(drift < 1e-3f, "relative momentum drift {:.6f}", drift);
}

TEST_CASE("dam break stays stable and inside the container")
{
   sph::Solver solver;
   solver.loadScene(sph::makeDamBreak());
   const auto particle_count = solver.particleCount();
   std::println("   dam break: {} particles", particle_count);

   const auto start_time = std::chrono::steady_clock::now();
   const auto rest_density = solver.settings()._rest_density;
   auto max_density_ratio = 0.0f;
   auto max_p99_density_ratio = 0.0f;
   auto total_substeps = 0;
   std::vector<float> sorted_densities;
   for (auto frame = 0; frame < 90; ++frame)
   {
      const auto statistics = solver.advance(1.0f / 60.0f, 1000);
      max_density_ratio = std::max(max_density_ratio, statistics._max_density_ratio);
      total_substeps += statistics._substeps;

      // single particles spike at impacts; the 99th percentile shows how compressible the bulk is
      sorted_densities.assign(solver.densities().begin(), solver.densities().end());
      const auto p99_position = sorted_densities.begin() + static_cast<std::ptrdiff_t>(sorted_densities.size() * 99 / 100);
      std::ranges::nth_element(sorted_densities, p99_position);
      max_p99_density_ratio = std::max(max_p99_density_ratio, *p99_position / rest_density);
   }
   const auto elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
   std::println(
      "   1.5 s simulated in {:.2f} s wall time, {} substeps, max density ratio {:.3f}, p99 {:.3f}",
      elapsed_s,
      total_substeps,
      max_density_ratio,
      max_p99_density_ratio
   );

   CHECK(solver.particleCount() == particle_count);
   CHECK_MESSAGE(max_p99_density_ratio < 1.1f, "p99 density ratio {:.3f}", max_p99_density_ratio);
   CHECK_MESSAGE(max_density_ratio < 1.3f, "max density ratio {:.3f}", max_density_ratio);

   const auto& scene = solver.scene();
   auto escaped_count = 0;
   auto non_finite_count = 0;
   auto inside_collider_count = 0;
   for (std::size_t particle_index = 0; particle_index < solver.particleCount(); ++particle_index)
   {
      const auto& position = solver.positions()[particle_index];
      if (!isFinite(position) || !isFinite(solver.velocities()[particle_index]))
      {
         ++non_finite_count;
         continue;
      }
      if (!scene._container.contains(position))
      {
         ++escaped_count;
      }
      if (sph::signedDistance(scene._colliders.front(), position) < 0.0f)
      {
         ++inside_collider_count;
      }
   }
   CHECK_MESSAGE(non_finite_count == 0, "{} particles are nan/inf", non_finite_count);
   CHECK_MESSAGE(escaped_count == 0, "{} particles left the container", escaped_count);
   CHECK_MESSAGE(inside_collider_count == 0, "{} particles are inside the sphere collider", inside_collider_count);
}

TEST_CASE("dam break comes to rest")
{
   // without artificial viscosity wcsph keeps 'boiling': after 6 s most particles still moved faster than 0.5 m/s
   sph::Solver solver;
   solver.loadScene(sph::makeDamBreak());
   for (auto frame = 0; frame < 240; ++frame)
   {
      solver.advance(1.0f / 60.0f, 1000);
   }

   auto speed_sum = 0.0f;
   for (const auto& velocity : solver.velocities())
   {
      speed_sum += sph::length(velocity);
   }
   const auto mean_speed = speed_sum / static_cast<float>(solver.particleCount());
   CHECK_MESSAGE(mean_speed < 0.05f, "mean speed after 4 s is {:.3f} m/s", mean_speed);
}

TEST_CASE("emitter spawns whole layers at the expected rate")
{
   sph::Solver solver;
   auto scene = makeEmptyScene({{0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 1.0f}}, {});
   scene._emitters.push_back(sph::Emitter{{0.5f, 0.1f, 0.5f}, {0.0f, 2.0f, 0.0f}, 0.04f, true, 0.0f});
   solver.loadScene(scene);

   // a single substep long enough for exactly one layer
   solver.step(solver.settings()._particle_spacing / 2.0f);
   const auto layer_size = solver.particleCount();
   CHECK(layer_size > 0);

   for (auto frame = 0; frame < 6; ++frame)
   {
      solver.advance(1.0f / 60.0f, 1000);
   }

   const auto travelled = 2.0f * (solver.settings()._particle_spacing / 2.0f + 6.0f / 60.0f);
   const auto expected_layers = std::floor(travelled / solver.settings()._particle_spacing + 1e-3f);
   const auto layers = static_cast<float>(solver.particleCount()) / static_cast<float>(layer_size);
   CHECK_MESSAGE(std::abs(layers - expected_layers) <= 1.0f, "{} layers emitted, expected {}", layers, expected_layers);
}

int main()
{
   for (const auto& test_case : test::registry())
   {
      std::println("[ RUN  ] {}", test_case._name);
      const auto failures_before = test::failureCount();
      test_case._body();
      std::println("[ {} ] {}", test::failureCount() == failures_before ? " OK " : "FAIL", test_case._name);
   }
   std::println("{} failed check(s)", test::failureCount());
   return test::failureCount() == 0 ? 0 : 1;
}
