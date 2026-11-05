# Review of the original code

Review of [smoothed_particle_hydrodynamics](https://github.com/varnholt/smoothed_particle_hydrodynamics)
at `e8ef0c6`, done before the port. Line numbers refer to that commit.
The status column says how the sandbox deals with each finding.

## Physics

| # | Where | Finding | Status |
|---|-------|---------|--------|
| 1 | `sph.cpp:1288` | Pressure term multiplies `p_i/rho_i^2 * p_j/rho_j^2` instead of adding them. Opposite signs flip repulsion into attraction; the magnitude is off by orders. The 100 m/s^2 "CFL" clamp hid the blow-ups. | Fixed, symmetric sum. |
| 2 | `sph.cpp:1099`, `evaluateNeighbor` | Self contribution `m_i W(0)` missing from the density. Isolated particles get density 0, and `1/density` falls back to 1.0. | Fixed, self included. |
| 3 | `mExamineCount = 8` | At most 8 neighbours, so density saturates and compression past that is invisible. Neighbour lists are one-sided, so momentum is not conserved. | Fixed, exhaustive 27-cell search; momentum test. |
| 4 | `sph.cpp:918/923` | `vx[3]` assigned twice, `vx[4..]`/`vy[4]`/`vz[4]` never assigned (uninitialised read). | Gone, new grid. |
| 5 | `sph.cpp:952` | Bounds check `vxi > 0` skips index 0 on every axis; the floor layer has no neighbours at all. | Gone, new grid; brute force comparison test. |
| 6 | `sph.cpp:989` | Random start, one direction scan; skewed sampling. | Gone. |
| 7 | `computePressure` | Negative pressure not clamped: tensile instability, clumping at the surface. | Clamped by default, switchable. |
| 8 | `computeAcceleration` | "CFL" check clamps acceleration; velocity is never limited. | Real CFL + force condition adaptive time step. |
| 9 | - | (found during the port) Without artificial viscosity WCSPH never settles: 90% of the particles still moved faster than 0.5 m/s six seconds after a dam break. | Monaghan artificial viscosity, alpha 0.05; settle test. |

## Boundaries

| # | Where | Finding | Status |
|---|-------|---------|--------|
| 10 | `handleBoundaryConditions` | Division by the velocity component, which can be 0; negative intersection time for particles already outside. | Replaced by projection + velocity response. |
| 11 | `applyBoundary` | Damping applied to the remaining displacement, not the velocity: perfectly elastic walls. | Restitution and friction act on the velocity. |
| 12 | `applyBoundary` | Corner handling per axis from the old position with the already reflected velocity. | Gone. |

## Threading and memory

| # | Where | Finding | Status |
|---|-------|---------|--------|
| 13 | `drawVoxels` | GUI thread iterates `QList` grid while the simulation thread clears and appends: data race, possible crash. | Gone: simulate, then render, on one thread. |
| 14 | `writeValuesToSimulation` | Parameters written from the GUI thread without locking; no input validation. | Gone, same thread; sliders with ranges. |
| 15 | `SPH::run` | Busy wait while paused, unthrottled when running. | Gone, frame driven. |
| 16 | `findNeighbors` | `qrand()` in OpenMP threads; worker threads all use seed 1. | Gone, deterministic jitter only at spawn. |
| 17 | `~SPH` | No `new[]` array is freed; unused `omp_lock_t`. | Gone, `std::vector`. |
| 18 | `Particle`, `vec3` | Uninitialised members. | Default member initialisers. |

## Build and rendering

| # | Where | Finding | Status |
|---|-------|---------|--------|
| 19 | `sph.pro:13` | `win32` and `{` on separate lines, so the block applies to every platform (Linux links `-lglu32`). `-openmp` is not a GCC flag. | Gone, CMake. |
| 20 | `visualization.cpp:343` | `glFrustum` near plane 0 raises `GL_INVALID_VALUE`; `gluLookAt` on the projection stack. | Gone, core profile renderer. |
| 21 | `sph.cpp:659/715/743` | Position compared to the cell count instead of `mMaxX`. | Gone. |
| 22 | various | Dead code: `sphui.*`, `visualization.ui`, `stepFinished`, `clearNeighbors`, `getInteractionRadius2`. | Not ported. |
