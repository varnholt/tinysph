//! Weakly compressible sph (smoothed particle hydrodynamics) fluid simulation, a port of the c++23 library
//! [tinysph](https://github.com/varnholt/tinysph).
//!
//! Create a [`World`], fill it with fluid, add colliders, force fields and emitters, and call [`World::step`] once per
//! frame. The solver runs on the cpu on scoped threads and follows the c++ cpu solver operation by operation, so both
//! produce the same numbers. No dependencies, no unsafe code.
//!
//! ```
//! use std::time::Duration;
//! use tinysph::{Aabb, Shape, Vec3, World, WorldSettings};
//!
//! let settings = WorldSettings { domain: Aabb::new(Vec3::ZERO, Vec3::new(1.6, 1.0, 0.6)), ..Default::default() };
//! let mut world = World::new(settings, Default::default())?;
//! world.add_fluid(Shape::cuboid(Vec3::new(0.25, 0.3, 0.3), Vec3::new(0.1, 0.1, 0.1)), Vec3::ZERO);
//! let statistics = world.step(Duration::from_secs(1) / 60);
//! assert!(statistics.substeps > 0);
//! # Ok::<(), tinysph::SettingsError>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod geometry;
mod kernels;
mod math;
mod parallel;
mod solver;
mod types;
mod world;

#[cfg(test)]
mod tests;

pub use math::{Aabb, Vec3};
pub use types::{
    Attractor, Backend, ColliderId, Cuboid, Emitter, EmitterId, FluidParameters, ForceField, ForceFieldId, SettingsError, Shape, Sphere,
    StepStatistics, Vortex, Wind, WorldSettings,
};
pub use world::{DEFAULT_MAX_SUBSTEPS, World};

/// The crate version as (major, minor, patch), matching the c++ `tinysph::version`.
pub const VERSION: (u32, u32, u32) = (0, 1, 0);
