// records how the cpu backend evolves a few fixed scenes, as reference data for the rust crate in rust/. its
// tests/parity.rs runs the same scenes and compares: the aggregates within a tolerance (sph is chaotic, and compilers
// may fuse multiply-adds differently), and reports whether the particle data even matches bit for bit.
//
//   tinysph_trace > tests/data/trace.txt
//
// the format, one line per recorded frame, floats as the hex of their bits so nothing is lost to printing:
//   scene <name> particles <n>
//   frame <k> substeps <n> dt <f> time <f> max_density <f> average_density <f> max_speed <f> count <n>
//         center <f> <f> <f> kinetic <f> positions <fnv-1a 64> velocities <hash> densities <hash>
// after each frame of 1/60 s. center is the mean position and kinetic the mean squared speed, both summed in double.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <format>
#include <print>
#include <span>
#include <string>

#include "tinysph/tinysph.h"

namespace
{

template <class T>
uint64_t fnv1a(std::span<const T> values)
{
   const auto bytes = std::as_bytes(values);
   uint64_t hash = 0xcbf29ce484222325ull;
   for (const auto byte : bytes)
   {
      hash = (hash ^ std::to_integer<uint8_t>(byte)) * 0x100000001b3ull;
   }
   return hash;
}

std::string bits(float value)
{
   return std::format("{:08x}", std::bit_cast<uint32_t>(value));
}

/// \brief the dam break of the tests: a water column, a sphere obstacle, gravity.
tinysph::World damBreak()
{
   tinysph::World world{{._domain = {{0.0f, 0.0f, 0.0f}, {1.6f, 1.0f, 0.6f}}}};
   world.addCollider(tinysph::Sphere{{1.1f, 0.12f, 0.3f}, 0.12f});
   world.addFluid(tinysph::Box{{0.26f, 0.31f, 0.3f}, {0.24f, 0.29f, 0.28f}});
   return world;
}

/// \brief every feature at once: both collider shapes, all force fields, an emitter and surface tension, coarser.
tinysph::World mixed()
{
   tinysph::World world{
      {._domain = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.8f, 0.5f}}, ._particle_spacing = 0.04f},
      {._surface_tension = 0.5f, ._wall_restitution = 0.2f}
   };
   world.addCollider(tinysph::Sphere{{0.7f, 0.12f, 0.25f}, 0.1f});
   world.addCollider(tinysph::Box{{0.45f, 0.06f, 0.25f}, {0.04f, 0.06f, 0.15f}});
   world.addForceField(tinysph::Vortex{{0.2f, 0.3f, 0.25f}, {0.0f, 1.0f, 0.0f}, 5.0f, 0.2f});
   world.addForceField(tinysph::Wind{{{0.0f, 0.4f, 0.0f}, {0.4f, 0.8f, 0.5f}}, {2.0f, 0.0f, 0.0f}});
   world.addForceField(tinysph::Attractor{{0.85f, 0.3f, 0.25f}, 5.0f, 0.25f});
   world.addEmitter(tinysph::Emitter{{0.9f, 0.5f, 0.25f}, {-1.0f, 0.5f, 0.0f}, 0.05f});
   world.addFluid(tinysph::Box{{0.18f, 0.2f, 0.25f}, {0.16f, 0.18f, 0.2f}}, {0.5f, 0.0f, 0.0f});
   world.addFluid(tinysph::Sphere{{0.6f, 0.55f, 0.25f}, 0.08f});
   return world;
}

void trace(const char* name, tinysph::World world)
{
   using namespace std::chrono_literals;
   std::println("scene {} particles {}", name, world.particleCount());
   for (auto frame = 1; frame <= 30; ++frame)
   {
      const auto statistics = world.step(tinysph::Seconds{1.0f / 60.0f});
      if (frame > 3 && frame != 5 && frame % 10 != 0)
      {
         continue;
      }
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      double kinetic = 0.0;
      for (const auto& position : world.positions())
      {
         x += position.x;
         y += position.y;
         z += position.z;
      }
      for (const auto& velocity : world.velocities())
      {
         kinetic += static_cast<double>(velocity.x) * velocity.x + static_cast<double>(velocity.y) * velocity.y +
                    static_cast<double>(velocity.z) * velocity.z;
      }
      const auto count = static_cast<double>(std::max<std::size_t>(world.particleCount(), 1));
      std::println(
         "frame {} substeps {} dt {} time {} max_density {} average_density {} max_speed {} count {} center {} {} {} kinetic {} "
         "positions {:016x} velocities {:016x} densities {:016x}",
         frame,
         statistics._substeps,
         bits(statistics._time_step.count()),
         bits(statistics._simulated_time.count()),
         bits(statistics._max_density_ratio),
         bits(statistics._average_density_ratio),
         bits(statistics._max_speed),
         world.particleCount(),
         bits(static_cast<float>(x / count)),
         bits(static_cast<float>(y / count)),
         bits(static_cast<float>(z / count)),
         bits(static_cast<float>(kinetic / count)),
         fnv1a(world.positions()),
         fnv1a(world.velocities()),
         fnv1a(world.densities())
      );
   }
}

}  // namespace

int main()
{
   trace("dam_break", damBreak());
   trace("mixed", mixed());
   return 0;
}
