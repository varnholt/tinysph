#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "tinysph/types.h"

namespace tinysph::detail
{

class Solver;

/// \brief runs the substeps of a Solver on the gpu through webgpu compute shaders (wgpu-native).
///
/// the solver keeps owning the particle data on the host, so adding particles, colliders and the rest of the World
/// api work the same for both backends. advance() uploads the particles, runs all substeps on the gpu and reads the
/// result back. per substep it only reads back the maximum speed and acceleration for the adaptive time step;
/// emitters run on the host and append their new particles to the gpu buffers.
///
/// the passes mirror the cpu solver one to one, see shaders.h. the only difference is the order of particles inside
/// a grid cell, which depends on the order of atomic operations, so results match the cpu up to float round-off.
class GpuSolver
{
public:
   /// \return nullptr if no usable gpu is found, or the gpu cannot hold the buffers these settings need.
   [[nodiscard]] static std::unique_ptr<GpuSolver> create(Solver& solver);

   ~GpuSolver();
   GpuSolver(const GpuSolver&) = delete;
   GpuSolver& operator=(const GpuSolver&) = delete;

   /// \brief like Solver::advance.
   StepStatistics advance(float frame_time, int32_t max_substeps);

   /// \brief adapter name and graphics api, e.g. "NVIDIA GeForce RTX 4070 (vulkan)".
   [[nodiscard]] const std::string& deviceDescription() const;

private:
   struct Impl;
   explicit GpuSolver(std::unique_ptr<Impl> impl);
   std::unique_ptr<Impl> _impl;
};

}  // namespace tinysph::detail
