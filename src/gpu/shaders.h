#pragma once

// wgsl compute shaders of the gpu backend. every pass mirrors one step of the cpu solver in solver.cpp, with the same
// kernels and the same order of terms, so both backends simulate the same fluid. each module is the common prelude
// followed by one pass; bindings are numbered per pass, the pipelines derive their layouts from the shaders.
//
// the code lives in src/gpu/shaders/*.wgsl, which the rust crate in rust/ uses too. cmake turns each file into a
// string constant named after it (common, assign_cells, scan, scatter, density, normals, forces, integrate).

namespace tinysph::detail::shaders
{

#include "tinysph_shaders.inc"

}  // namespace tinysph::detail::shaders
