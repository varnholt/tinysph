// tests of internals that the public api deliberately hides

#include <algorithm>
#include <random>
#include <ranges>
#include <vector>

#include "solver.h"
#include "test_framework.h"

namespace
{

tinysph::detail::Solver makeSolver(const tinysph::Aabb& domain)
{
   return tinysph::detail::Solver{{._domain = domain}, {._gravity = {}}};
}

}  // namespace

TEST_CASE("internal: neighbor search matches brute force")
{
   auto solver = makeSolver({{0.0f, 0.0f, 0.0f}, {0.5f, 0.4f, 0.3f}});

   std::mt19937 generator{1234u};
   std::uniform_real_distribution<float> distribution{0.0f, 1.0f};
   for (auto particle_number = 0; particle_number < 3000; ++particle_number)
   {
      const tinysph::Vec3 position{distribution(generator) * 0.5f, distribution(generator) * 0.4f, distribution(generator) * 0.3f};
      solver.addParticle(position, {});
   }
   CHECK(solver.particleCount() > 2000);

   solver.rebuildGrid();
   const auto positions = solver.positions();
   const auto support_radius_squared = solver.supportRadius() * solver.supportRadius();

   const auto all_indices = std::views::iota(0uz, positions.size());

   auto mismatches = 0;
   for (const auto particle_index : all_indices)
   {
      std::vector<std::size_t> grid_neighbors;
      solver.forEachNeighbor(
         particle_index, [&](std::size_t neighbor_index, const tinysph::Vec3&, float) { grid_neighbors.push_back(neighbor_index); }
      );

      const auto brute_force_neighbors =
         all_indices |
         std::views::filter(
            [&](std::size_t neighbor_index)
            { return tinysph::lengthSquared(positions[particle_index] - positions[neighbor_index]) < support_radius_squared; }
         ) |
         std::ranges::to<std::vector>();

      std::ranges::sort(grid_neighbors);
      if (grid_neighbors != brute_force_neighbors)
      {
         ++mismatches;
      }
   }
   CHECK_MESSAGE(mismatches == 0, "{} particles had a different neighbor set", mismatches);
}

TEST_CASE("internal: resting lattice has the rest density")
{
   auto solver = makeSolver({{0.0f, 0.0f, 0.0f}, {0.4f, 0.4f, 0.4f}});
   solver.addFluidBox({{0.0f, 0.0f, 0.0f}, {0.4f, 0.4f, 0.4f}}, {});
   solver.step(1e-6f);

   const auto rest_density = solver.settings()._rest_density;
   const auto interior_margin = solver.supportRadius() * 1.5f;
   const tinysph::Aabb interior{
      {interior_margin, interior_margin, interior_margin}, {0.4f - interior_margin, 0.4f - interior_margin, 0.4f - interior_margin}
   };

   auto interior_count = 0;
   auto worst_error = 0.0f;
   for (const auto& [position, density] : std::views::zip(solver.positions(), solver.densities()))
   {
      if (interior.contains(position))
      {
         ++interior_count;
         worst_error = std::max(worst_error, std::abs(density / rest_density - 1.0f));
      }
   }
   CHECK(interior_count > 100);
   CHECK_MESSAGE(worst_error < 0.01f, "worst interior density error {:.4f}", worst_error);
}

TEST_CASE("internal: momentum is conserved without walls and gravity")
{
   auto solver = makeSolver({{0.0f, 0.0f, 0.0f}, {4.0f, 4.0f, 4.0f}});
   solver.addFluidSphere({1.65f, 2.0f, 2.0f}, 0.15f, {1.0f, 0.0f, 0.0f});
   solver.addFluidSphere({2.35f, 2.05f, 2.0f}, 0.15f, {-1.0f, 0.0f, 0.0f});

   auto momentum_scale = 0.0f;
   for (const auto& velocity : solver.velocities())
   {
      momentum_scale += tinysph::length(velocity) * solver.particleMass();
   }
   const auto initial_momentum = solver.totalMomentum();

   for (auto frame = 0; frame < 24; ++frame)
   {
      solver.advance(1.0f / 60.0f, 1000);
   }

   const auto drift = tinysph::length(solver.totalMomentum() - initial_momentum) / momentum_scale;
   CHECK_MESSAGE(drift < 1e-3f, "relative momentum drift {:.6f}", drift);
}

int main()
{
   return test::runAll();
}
