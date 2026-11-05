#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "sph/kernels.h"
#include "sph/scene.h"
#include "sph/vec3.h"

namespace sph
{

/// \brief tunable solver parameters. everything is in si units (m, kg, s).
struct SolverSettings
{
   float _rest_density{1000.0f};         //!< rho0 in kg/m^3, water
   float _particle_spacing{0.025f};      //!< distance between particles at rest; changing it needs a scene reload
   float _support_radius_factor{2.0f};   //!< h = factor * spacing
   float _speed_of_sound{30.0f};         //!< stiffness of the equation of state p = c^2 (rho - rho0)
   float _viscosity{0.2f};               //!< dynamic viscosity mu in Pa s; far above real water, which is needed for stability
   float _artificial_viscosity{0.05f};   //!< monaghan alpha; damps only approaching pairs; without it wcsph never settles
   bool _clamp_negative_pressure{true};  //!< avoids the tensile instability (particle clumping) at free surfaces
   float _wall_restitution{0.0f};        //!< 0 = walls absorb the normal velocity, 1 = perfect bounce
   float _wall_friction{0.02f};          //!< fraction of the tangential velocity removed per contact
   float _max_time_step{1.0f / 400.0f};  //!< upper bound for the adaptive time step
   float _cfl_number{0.4f};              //!< dt <= cfl * h / (c + max speed)
   std::size_t _max_particles{250'000};  //!< hard cap, emitters stop when it is reached
};

/// \brief what happened during the last advance() call.
struct StepStatistics
{
   float _time_step{0.0f};  //!< dt of the last substep in s
   int32_t _substeps{0};
   float _simulated_time{0.0f};         //!< sum of all substep dts in s
   float _max_density_ratio{0.0f};      //!< max rho / rho0
   float _average_density_ratio{0.0f};  //!< average rho / rho0
   float _max_speed{0.0f};              //!< in m/s
   double _step_ms{0.0};                //!< wall clock time of the whole advance() call
};

/// \brief weakly compressible sph (wcsph) solver on the cpu.
///
/// per substep:
///   1) sort particles into a uniform grid with cell size h (counting sort, data is physically reordered)
///   2) density:      rho_i = sum_j m W_poly6(r_ij)                (self included)
///   3) pressure:     p_i   = c^2 (rho_i - rho0)                    (optionally clamped at 0)
///   4) acceleration: a_i   = -sum_j m (p_i/rho_i^2 + p_j/rho_j^2) grad W_spiky
///                            + mu sum_j m (v_j - v_i) / (rho_i rho_j) lap W_visc
///                            + gravity + force fields
///   5) integrate with semi-implicit euler, then resolve collisions with the container and the colliders
///
/// both pair terms are antisymmetric in i and j, so with equal masses the internal forces cancel and momentum is conserved.
class Solver
{
public:
   explicit Solver(const SolverSettings& settings = {});

   /// \brief removes all particles and fills the scene's fluid volumes.
   void loadScene(const Scene& scene);

   /// \brief re-reads the settings that depend on the particle spacing (support radius, mass).
   ///        existing particles are kept, so call loadScene() afterwards when the spacing changed.
   void applySettings();

   void clearParticles();

   /// \brief fills a box with particles on a lattice, skipping points inside colliders.
   /// \return number of particles added
   std::size_t addFluidBox(const Aabb& box, const Vec3& velocity);

   /// \brief fills a sphere with particles on a lattice, skipping points inside colliders.
   /// \return number of particles added
   std::size_t addFluidSphere(const Vec3& center, float radius, const Vec3& velocity);

   /// \brief adds one particle if it is inside the container, outside all colliders and below the particle cap.
   bool addParticle(const Vec3& position, const Vec3& velocity);

   /// \brief advances the simulation by frame_time seconds using adaptive substeps.
   ///        stops early after max_substeps so a slow machine runs in slow motion instead of freezing.
   StepStatistics advance(float frame_time, int32_t max_substeps);

   /// \brief performs exactly one substep of the given length.
   void step(float time_step);

   /// \brief time step that satisfies the cfl and the force conditions for the current state.
   [[nodiscard]] float computeTimeStep() const;

   /// \brief sorts the particles into the grid. step() does this itself; exposed for tests.
   void rebuildGrid();

   /// \brief calls callback(neighbor_index, offset, distance_squared) for every particle within h of the given particle,
   ///        including the particle itself. requires an up to date grid.
   template <class Callback>
   void forEachNeighbor(std::size_t particle_index, Callback&& callback) const;

   [[nodiscard]] Scene& scene();
   [[nodiscard]] const Scene& scene() const;

   [[nodiscard]] SolverSettings& settings();
   [[nodiscard]] const SolverSettings& settings() const;

   /// \brief force fields that only live for the next advance() call, e.g. the mouse tool.
   void setTransientFields(std::vector<ForceField> fields);

   [[nodiscard]] std::size_t particleCount() const;
   [[nodiscard]] float particleMass() const;
   [[nodiscard]] float supportRadius() const;
   [[nodiscard]] std::span<const Vec3> positions() const;
   [[nodiscard]] std::span<const Vec3> velocities() const;
   [[nodiscard]] std::span<const float> densities() const;
   [[nodiscard]] Vec3 totalMomentum() const;
   [[nodiscard]] float maxSpeed() const;

private:
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
   void computeAccelerations();
   void integrate(float time_step);
   void resolveCollisions(Vec3& position, Vec3& velocity) const;
   void runEmitters(float time_step);
   void emitLayer(const Emitter& emitter);

   SolverSettings _settings;
   Scene _scene;
   Kernels _kernels;
   float _particle_mass{0.0f};
   float _particle_radius{0.0f};  //!< half the spacing, used for collisions
   std::vector<ForceField> _transient_fields;

   // particle data, structure of arrays
   std::vector<Vec3> _positions;
   std::vector<Vec3> _velocities;
   std::vector<Vec3> _accelerations;
   std::vector<float> _densities;
   std::vector<float> _pressures;

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

}  // namespace sph
