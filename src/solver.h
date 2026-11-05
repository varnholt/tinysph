#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "kernels.h"
#include "tinysph/math.h"
#include "tinysph/types.h"

namespace tinysph::detail
{

/// \brief an emitter plus the distance its last layer has travelled.
struct EmitterState
{
   Emitter _emitter;
   float _travelled{0.0f};
};

/// \brief weakly compressible sph (wcsph) solver on the cpu.
///
/// per substep:
///   1) sort particles into a uniform grid with cell size h (counting sort, data is physically reordered)
///   2) density:      rho_i = sum_j m W_poly6(r_ij)                (self included)
///   3) pressure:     p_i   = c^2 (rho_i - rho0)                    (optionally clamped at 0)
///   4) acceleration: a_i   = -sum_j m (p_i/rho_i^2 + p_j/rho_j^2 + pi_ij) grad W_spiky
///                            + mu sum_j m (v_j - v_i) / (rho_i rho_j) lap W_visc
///                            + gravity + force fields
///      pi_ij is monaghan's artificial viscosity, only for approaching pairs.
///   5) integrate with semi-implicit euler, then resolve collisions with the domain and the colliders
///
/// all pair terms are antisymmetric in i and j, so with equal masses the internal forces cancel and momentum is conserved.
class Solver
{
public:
   Solver(const WorldSettings& settings, const FluidParameters& parameters);

   [[nodiscard]] const WorldSettings& settings() const;
   [[nodiscard]] const FluidParameters& parameters() const;
   void setParameters(const FluidParameters& parameters);
   void setDomain(const Aabb& domain);

   void clearParticles();
   std::size_t addFluidBox(const Aabb& box, const Vec3& velocity);
   std::size_t addFluidSphere(const Vec3& center, float radius, const Vec3& velocity);
   bool addParticle(const Vec3& position, const Vec3& velocity);

   [[nodiscard]] std::vector<Shape>& colliders();
   [[nodiscard]] std::vector<ForceField>& fields();
   [[nodiscard]] std::vector<EmitterState>& emitters();

   /// \brief advances by frame_time using adaptive substeps, at most max_substeps of them.
   StepStatistics advance(float frame_time, int32_t max_substeps);

   /// \brief performs exactly one substep of the given length.
   void step(float time_step);

   [[nodiscard]] float computeTimeStep() const;

   /// rief the adaptive time step for the given maximum speed and acceleration of all particles.
   [[nodiscard]] float computeTimeStep(float max_speed, float max_acceleration) const;

   /// rief fills the density and speed statistics from the current particle data.
   void fillStatistics(StepStatistics& statistics) const;

   /// \brief sorts the particles into the grid. step() does this itself; exposed for tests.
   void rebuildGrid();

   /// \brief calls callback(neighbor_index, offset, distance_squared) for every particle within h of the given particle,
   ///        including the particle itself. requires an up to date grid.
   template <class Callback>
   void forEachNeighbor(std::size_t particle_index, Callback&& callback) const;

   [[nodiscard]] std::size_t particleCount() const;
   [[nodiscard]] float particleMass() const;
   [[nodiscard]] float particleRadius() const;
   [[nodiscard]] float supportRadius() const;
   [[nodiscard]] std::span<const Vec3> positions() const;
   [[nodiscard]] std::span<const Vec3> velocities() const;
   [[nodiscard]] std::span<const float> densities() const;
   [[nodiscard]] Vec3 totalMomentum() const;
   [[nodiscard]] float maxSpeed() const;

private:
   friend class GpuSolver;  // keeps the particle data on the host in sync and runs the emitters

   struct CellCoordinate
   {
      int32_t x{0};
      int32_t y{0};
      int32_t z{0};
   };

   [[nodiscard]] CellCoordinate cellCoordinate(const Vec3& position) const;
   [[nodiscard]] std::size_t cellIndex(const CellCoordinate& coordinate) const;

   void setupGrid();
   void computeDensities();
   void computeNormals();
   void computeAccelerations();
   void integrate(float time_step);
   void resolveCollisions(Vec3& position, Vec3& velocity) const;
   void runEmitters(float time_step);
   void emitLayer(const EmitterState& state);

   WorldSettings _settings;
   FluidParameters _parameters;
   Kernels _kernels;
   float _particle_mass{0.0f};
   float _particle_radius{0.0f};  //!< half the spacing, used for collisions

   std::vector<Shape> _colliders;
   std::vector<ForceField> _fields;
   std::vector<EmitterState> _emitters;

   // particle data, structure of arrays
   std::vector<Vec3> _positions;
   std::vector<Vec3> _velocities;
   std::vector<Vec3> _accelerations;
   std::vector<float> _densities;
   std::vector<float> _pressures;
   std::vector<Vec3> _normals;  //!< surface normals for the surface tension, only filled when it is on

   // uniform grid, rebuilt every substep
   Vec3 _grid_origin;
   CellCoordinate _grid_size;
   std::vector<uint32_t> _cell_of_particle;
   std::vector<uint32_t> _cell_start;  //!< particles of cell c are [_cell_start[c], _cell_start[c + 1])
   std::vector<uint32_t> _sorted_order;
   std::vector<uint32_t> _cell_write_offset;

   // scratch buffers for reordering
   std::vector<Vec3> _scratch_positions;
   std::vector<Vec3> _scratch_velocities;

   float _max_acceleration{0.0f};
   uint32_t _jitter_state{1u};  //!< tiny deterministic jitter on spawned lattices to break perfect symmetry
};

template <class Callback>
void Solver::forEachNeighbor(std::size_t particle_index, Callback&& callback) const
{
   const auto& position = _positions[particle_index];
   const auto center = cellCoordinate(position);
   const auto support_radius_squared = _kernels.supportRadiusSquared();

   for (auto z = std::max(center.z - 1, 0); z <= std::min(center.z + 1, _grid_size.z - 1); ++z)
   {
      for (auto y = std::max(center.y - 1, 0); y <= std::min(center.y + 1, _grid_size.y - 1); ++y)
      {
         for (auto x = std::max(center.x - 1, 0); x <= std::min(center.x + 1, _grid_size.x - 1); ++x)
         {
            const auto cell = cellIndex({x, y, z});
            for (auto neighbor_index = _cell_start[cell]; neighbor_index < _cell_start[cell + 1]; ++neighbor_index)
            {
               const auto offset = position - _positions[neighbor_index];
               const auto distance_squared = lengthSquared(offset);
               if (distance_squared < support_radius_squared)
               {
                  callback(static_cast<std::size_t>(neighbor_index), offset, distance_squared);
               }
            }
         }
      }
   }
}

}  // namespace tinysph::detail
