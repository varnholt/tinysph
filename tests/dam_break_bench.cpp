// runs the dam break scene headless and prints density and timing statistics.
// usage: tinysph_bench [--gpu] [speed_of_sound] [viscosity] [seconds] [artificial_viscosity]

#include <algorithm>
#include <chrono>
#include <print>
#include <string>
#include <string_view>
#include <vector>

#include "test_scenes.h"
#include "tinysph/tinysph.h"

using namespace std::chrono_literals;

int main(int argument_count, char** arguments)
{
   if (argument_count > 1 && std::string_view{arguments[1]} == "--gpu")
   {
      test::backend() = tinysph::Backend::Gpu;
      --argument_count;
      ++arguments;
   }

   tinysph::FluidParameters parameters;
   if (argument_count > 1)
   {
      parameters._speed_of_sound = std::stof(arguments[1]);
   }
   if (argument_count > 2)
   {
      parameters._viscosity = std::stof(arguments[2]);
   }
   if (argument_count > 4)
   {
      parameters._artificial_viscosity = std::stof(arguments[4]);
   }
   const auto seconds = argument_count > 3 ? std::stof(arguments[3]) : 2.0f;

   auto world = test::makeDamBreakWorld(parameters);
   const auto rest_density = world.settings()._rest_density;
   std::println(
      "{}: particles {}, c {}, mu {}, alpha {}",
      world.deviceDescription(),
      world.particleCount(),
      parameters._speed_of_sound,
      parameters._viscosity,
      parameters._artificial_viscosity
   );

   const auto frame_count = static_cast<int32_t>(seconds * 60.0f);
   const auto start_time = std::chrono::steady_clock::now();
   auto worst_max_ratio = 0.0f;
   auto worst_p99_ratio = 0.0f;
   std::vector<float> sorted_densities;
   for (auto frame = 0; frame < frame_count; ++frame)
   {
      const auto statistics = world.step(1s / 60.0, 10000);

      sorted_densities.assign(world.densities().begin(), world.densities().end());
      const auto p99_position = sorted_densities.begin() + static_cast<std::ptrdiff_t>(sorted_densities.size() * 99 / 100);
      std::ranges::nth_element(sorted_densities, p99_position);
      const auto p99_ratio = *p99_position / rest_density;

      worst_max_ratio = std::max(worst_max_ratio, statistics._max_density_ratio);
      worst_p99_ratio = std::max(worst_p99_ratio, p99_ratio);
      if (frame % 10 == 0)
      {
         std::println(
            "t {:5.2f}  substeps {:3}  dt {:.5f}  max {:.3f}  p99 {:.3f}  avg {:.3f}  vmax {:.2f}  {:.1f} ms",
            (frame + 1) / 60.0f,
            statistics._substeps,
            statistics._time_step.count(),
            statistics._max_density_ratio,
            p99_ratio,
            statistics._average_density_ratio,
            statistics._max_speed,
            statistics._wall_time.count()
         );
      }
   }
   const auto elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();

   // a settled tank should be nearly at rest
   auto speed_sum = 0.0f;
   auto fast_count = 0;
   for (const auto& velocity : world.velocities())
   {
      const auto speed = tinysph::length(velocity);
      speed_sum += speed;
      if (speed > 0.5f)
      {
         ++fast_count;
      }
   }
   const auto mean_speed = speed_sum / static_cast<float>(world.particleCount());

   std::println(
      "worst max {:.3f}, worst p99 {:.3f}, final mean speed {:.3f} m/s, {} particles faster than 0.5 m/s, {:.2f} s wall for {:.2f} s "
      "simulated",
      worst_max_ratio,
      worst_p99_ratio,
      mean_speed,
      fast_count,
      elapsed_s,
      seconds
   );
   return 0;
}
