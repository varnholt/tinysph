# tinysph for Rust

[![CI](https://github.com/varnholt/tinysph/actions/workflows/ci.yml/badge.svg)](https://github.com/varnholt/tinysph/actions/workflows/ci.yml)

A weakly compressible SPH fluid simulation: particles in a box shaped domain, sphere and box colliders, attractors,
vortices and wind, emitters, surface tension, and an adaptive time step that keeps it stable.

This is the Rust port of the C++23 library [tinysph](https://github.com/varnholt/tinysph). The solver follows the C++
CPU solver operation by operation, and CI holds the two to it: the crate replays the scenes of the C++ reference trace
and compares every recorded frame.

- no dependencies and no `unsafe`
- multithreaded on the CPU with `std::thread::scope`; results do not depend on the thread count
- the same API as the C++ library, with `Result`s instead of exceptions and typed handles that are never reused
- the GPU backend of the C++ library is not ported yet; `Backend::Gpu` is rejected with `SettingsError::UnsupportedBackend`

## Usage

```toml
[dependencies]
tinysph = "0.1"
```

```rust
use std::time::Duration;
use tinysph::{Aabb, Shape, Vec3, World, WorldSettings};

// a 1.6 x 1.0 x 0.6 m tank
let settings = WorldSettings { domain: Aabb::new(Vec3::ZERO, Vec3::new(1.6, 1.0, 0.6)), ..Default::default() };
let mut world = World::new(settings, Default::default())?;

// a column of water on the left and a ball in its way
world.add_fluid(Shape::cuboid(Vec3::new(0.25, 0.3, 0.3), Vec3::new(0.23, 0.28, 0.28)), Vec3::ZERO);
world.add_collider(Shape::sphere(Vec3::new(1.1, 0.12, 0.3), 0.12));

loop {
    let statistics = world.step(Duration::from_secs(1) / 60);
    draw(world.positions(), world.particle_radius());
}
```

| | |
|---|---|
| `World::new(WorldSettings, FluidParameters)` | creates a world, or returns a `SettingsError` saying why the settings are invalid |
| `World::validate(&WorldSettings)` | checks settings without creating a world |
| `step(Duration)`, `step_with_limit(Duration, max_substeps)`, `step_seconds(f32, max_substeps)` | advance with adaptive substeps (at most 1000 by default) and return `StepStatistics` |
| `add_fluid(Shape, velocity)`, `add_particle`, `clear_particles` | fill a sphere or box at rest spacing, skipping colliders and space outside the domain |
| `positions()`, `velocities()`, `densities()` | particle data as slices; the order changes between steps |
| `particle_count()`, `particle_mass()`, `particle_radius()` | |
| `settings()`, `parameters()`, `set_parameters()`, `domain()`, `set_domain()` | the domain can change at any time, particles outside are pushed back in |
| `add_collider` / `set_collider` / `remove_collider` / `collider` | `Shape::Sphere` and `Shape::Cuboid` (the C++ `Box`), handles are `ColliderId` |
| `add_force_field` / `set_force_field` / `remove_force_field` / `force_field` | `Attractor`, `Vortex`, `Wind`, handles are `ForceFieldId` |
| `add_emitter` / `set_emitter` / `remove_emitter` / `emitter` | disk shaped nozzles that spawn non overlapping layers, handles are `EmitterId` |
| `WorldSettings` | domain, particle spacing, rest density, support radius factor, particle cap, backend; fixed per world |
| `FluidParameters` | gravity, speed of sound, viscosity, artificial viscosity, surface tension, wall restitution and friction, time step limits |
| `StepStatistics` | last `time_step`, `substeps`, `simulated_time`, max and average density ratio, `max_speed`, `wall_time` |

The defaults of `WorldSettings` and `FluidParameters` are those of the C++ library: water at 2.5 cm spacing in a
1 m cube. The solver itself is described in the [main README](https://github.com/varnholt/tinysph#solver).

A `Duration` holds whole nanoseconds, so `Duration::from_secs(1) / 60` is one ulp away from the `f32` the C++
`world.step(1s / 60.0)` uses, and the two runs drift apart after a few hundred substeps. To reproduce C++ runs exactly,
pass the frame time as `f32` seconds: `world.step_seconds(1.0 / 60.0, DEFAULT_MAX_SUBSTEPS)`.

## Examples

```sh
cargo run --release --example hello_fluid
cargo run --release --example dam_break -- [speed_of_sound] [viscosity] [seconds] [artificial_viscosity]
```

`dam_break` is the headless benchmark of the C++ library (`tinysph_bench`) and prints the same lines.

## Tests

- `tests/parity.rs` runs the two scenes of the C++ `tests/trace.cpp` (the dam break, and a scene using every collider,
  force field, an emitter and surface tension) and compares each recorded frame with `../tests/data/trace.txt`.
  Particle and substep counts must match exactly; time step, densities, speed, center of mass and kinetic energy within
  a relative tolerance of 1e-4 for the first three frames, 1e-2 up to frame 10 and 0.1 after that, because SPH is
  chaotic and a one ulp difference (a fused multiply-add, a different `sqrt`) grows with every substep. It also
  prints whether positions, velocities and densities match bit for bit; on x86-64 they do for every frame.
- `tests/api.rs` has the API tests of the C++ library, `src/tests.rs` its solver internal tests (kernels, neighbor
  search, momentum conservation, time step). Debug builds also check for integer overflow.

Minimum supported Rust version: 1.85.

## License

MIT, see `LICENSE`.
