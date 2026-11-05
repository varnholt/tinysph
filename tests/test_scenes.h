#pragma once

#include "tinysph/tinysph.h"

namespace test
{

/// \brief 1.6 x 1.0 x 0.6 m tank, a 0.48 x 0.58 x 0.56 m water column on the left and a sphere obstacle.
inline tinysph::World makeDamBreakWorld(const tinysph::FluidParameters& parameters = {})
{
   tinysph::World world{{._domain = {{0.0f, 0.0f, 0.0f}, {1.6f, 1.0f, 0.6f}}}, parameters};
   world.addCollider(tinysph::Sphere{{1.1f, 0.12f, 0.3f}, 0.12f});
   world.addFluid(tinysph::Box{{0.26f, 0.31f, 0.3f}, {0.24f, 0.29f, 0.28f}});
   return world;
}

}  // namespace test
