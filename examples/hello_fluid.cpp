// the smallest useful tinysph program: break a dam, let it hit an obstacle, print what happens.

#include <algorithm>
#include <chrono>
#include <print>
#include <ranges>

#include <tinysph/tinysph.h>

using namespace std::chrono_literals;

int main()
{
   // a 1.6 x 1.0 x 0.6 m tank
   tinysph::World world{{._domain = {{0.0f, 0.0f, 0.0f}, {1.6f, 1.0f, 0.6f}}}};

   // a column of water on the left and a ball in its way
   world.addFluid(tinysph::Box{._center = {0.25f, 0.3f, 0.3f}, ._half_extents = {0.23f, 0.28f, 0.28f}});
   world.addCollider(tinysph::Sphere{._center = {1.1f, 0.12f, 0.3f}, ._radius = 0.12f});

   std::println(
      "tinysph {}.{}.{}: {} particles", tinysph::version.major, tinysph::version.minor, tinysph::version.patch, world.particleCount()
   );

   for (auto frame = 0; frame <= 120; ++frame)
   {
      const auto statistics = world.step(1s / 60.0);
      if (frame % 20 == 0)
      {
         // the front of the wave: the right most particle
         const auto front = std::ranges::max(world.positions() | std::views::transform(&tinysph::Vec3::x));
         std::println(
            "t = {:.2f} s   wave front at x = {:.2f} m   max speed {:.2f} m/s   {} substeps",
            frame / 60.0f,
            front,
            statistics._max_speed,
            statistics._substeps
         );
      }
   }
   return 0;
}
