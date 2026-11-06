//! Shapes, force fields, emitters, handles, settings and statistics.

use std::fmt;
use std::time::Duration;

use crate::math::{Aabb, Vec3};

// ------------------------------------------------------------------------------------------------
// shapes, used for fluid volumes and colliders
// ------------------------------------------------------------------------------------------------

/// A sphere.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    /// center in m
    pub center: Vec3,
    /// radius in m
    pub radius: f32,
}

impl Default for Sphere {
    fn default() -> Self {
        Self { center: Vec3::ZERO, radius: 0.1 }
    }
}

/// An axis aligned box, given by its center and half extents (`Box` in the c++ library).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cuboid {
    /// center in m
    pub center: Vec3,
    /// half the size along each axis in m
    pub half_extents: Vec3,
}

impl Default for Cuboid {
    fn default() -> Self {
        Self { center: Vec3::ZERO, half_extents: Vec3::splat(0.1) }
    }
}

/// A shape, used for fluid volumes and colliders.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// a sphere
    Sphere(Sphere),
    /// an axis aligned box
    Cuboid(Cuboid),
}

impl Shape {
    /// A sphere shape.
    #[must_use]
    pub const fn sphere(center: Vec3, radius: f32) -> Self {
        Shape::Sphere(Sphere { center, radius })
    }

    /// A box shape from its center and half extents.
    #[must_use]
    pub const fn cuboid(center: Vec3, half_extents: Vec3) -> Self {
        Shape::Cuboid(Cuboid { center, half_extents })
    }
}

impl From<Sphere> for Shape {
    fn from(value: Sphere) -> Self {
        Shape::Sphere(value)
    }
}

impl From<Cuboid> for Shape {
    fn from(value: Cuboid) -> Self {
        Shape::Cuboid(value)
    }
}

// ------------------------------------------------------------------------------------------------
// force fields: extra accelerations applied to every particle inside their range
// ------------------------------------------------------------------------------------------------

/// Pulls particles towards a point, a negative strength pushes them away. Falls off linearly to zero at the radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attractor {
    /// center in m
    pub position: Vec3,
    /// acceleration at the center in m/s^2
    pub strength: f32,
    /// range of influence in m
    pub radius: f32,
}

impl Default for Attractor {
    fn default() -> Self {
        Self { position: Vec3::ZERO, strength: 20.0, radius: 0.3 }
    }
}

/// Swirls particles around an axis through a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vortex {
    /// a point on the axis in m
    pub position: Vec3,
    /// axis direction, need not be normalized
    pub axis: Vec3,
    /// tangential acceleration at the axis in m/s^2
    pub strength: f32,
    /// range of influence in m
    pub radius: f32,
}

impl Default for Vortex {
    fn default() -> Self {
        Self { position: Vec3::ZERO, axis: Vec3::new(0.0, 1.0, 0.0), strength: 10.0, radius: 0.3 }
    }
}

/// Constant acceleration inside a box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wind {
    /// where the wind blows
    pub region: Aabb,
    /// acceleration in m/s^2
    pub acceleration: Vec3,
}

impl Default for Wind {
    fn default() -> Self {
        Self { region: Aabb::new(Vec3::ZERO, Vec3::splat(1.0)), acceleration: Vec3::new(5.0, 0.0, 0.0) }
    }
}

/// An extra acceleration applied to every particle inside its range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ForceField {
    /// see [`Attractor`]
    Attractor(Attractor),
    /// see [`Vortex`]
    Vortex(Vortex),
    /// see [`Wind`]
    Wind(Wind),
}

impl From<Attractor> for ForceField {
    fn from(value: Attractor) -> Self {
        ForceField::Attractor(value)
    }
}

impl From<Vortex> for ForceField {
    fn from(value: Vortex) -> Self {
        ForceField::Vortex(value)
    }
}

impl From<Wind> for ForceField {
    fn from(value: Wind) -> Self {
        ForceField::Wind(value)
    }
}

// ------------------------------------------------------------------------------------------------
// emitters
// ------------------------------------------------------------------------------------------------

/// A disk shaped nozzle. It spawns one lattice layer of particles each time the previous layer has travelled one
/// particle spacing, so new particles never overlap; the flow rate follows from speed and radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emitter {
    /// center of the nozzle in m
    pub position: Vec3,
    /// direction and speed of the jet in m/s
    pub velocity: Vec3,
    /// nozzle radius in m
    pub radius: f32,
    /// a disabled emitter spawns nothing
    pub enabled: bool,
}

impl Default for Emitter {
    fn default() -> Self {
        Self { position: Vec3::ZERO, velocity: Vec3::new(0.0, 2.0, 0.0), radius: 0.04, enabled: true }
    }
}

// ------------------------------------------------------------------------------------------------
// handles; they stay valid until the object is removed, and are never reused within a world
// ------------------------------------------------------------------------------------------------

/// Handle of a collider.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ColliderId(pub u32);

/// Handle of a force field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ForceFieldId(pub u32);

/// Handle of an emitter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EmitterId(pub u32);

// ------------------------------------------------------------------------------------------------
// configuration
// ------------------------------------------------------------------------------------------------

/// What the simulation runs on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Backend {
    /// multithreaded on the cpu
    #[default]
    Cpu,
    /// webgpu compute shaders through wgpu (vulkan, metal or d3d12); needs the cargo feature `gpu`, without it
    /// [`World::new`](crate::World::new) returns [`SettingsError::UnsupportedBackend`], and
    /// [`SettingsError::GpuUnavailable`] if no usable gpu is found
    Gpu,
}

/// Fixed for the lifetime of a world. Everything is in si units (m, kg, s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldSettings {
    /// the fluid never leaves this box; can be changed later with [`World::set_domain`](crate::World::set_domain)
    pub domain: Aabb,
    /// distance between particles at rest; sets resolution and particle size
    pub particle_spacing: f32,
    /// rho0 in kg/m^3, water
    pub rest_density: f32,
    /// smoothing radius h = factor * spacing
    pub support_radius_factor: f32,
    /// hard cap; adding particles beyond it fails, emitters pause
    pub max_particles: usize,
    /// what the simulation runs on
    pub backend: Backend,
}

impl Default for WorldSettings {
    fn default() -> Self {
        Self {
            domain: Aabb::new(Vec3::ZERO, Vec3::splat(1.0)),
            particle_spacing: 0.025,
            rest_density: 1000.0,
            support_radius_factor: 2.0,
            max_particles: 250_000,
            backend: Backend::Cpu,
        }
    }
}

/// Can be changed at any time with [`World::set_parameters`](crate::World::set_parameters).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FluidParameters {
    /// in m/s^2
    pub gravity: Vec3,
    /// stiffness of the equation of state p = c^2 (rho - rho0); higher = less compressible, smaller steps
    pub speed_of_sound: f32,
    /// dynamic viscosity mu in Pa s
    pub viscosity: f32,
    /// monaghan alpha; damps approaching particle pairs; at 0 the fluid never settles
    pub artificial_viscosity: f32,
    /// akinci gamma; holds drops together and rounds them, 0 = off
    pub surface_tension: f32,
    /// avoids particles clumping at the free surface
    pub clamp_negative_pressure: bool,
    /// 0 = walls absorb the normal velocity, 1 = perfect bounce
    pub wall_restitution: f32,
    /// coulomb coefficient: a contact removes at most this times the normal speed change from the sliding speed
    pub wall_friction: f32,
    /// upper bound for the adaptive time step in s
    pub max_time_step: f32,
    /// dt <= cfl * h / (c + max speed)
    pub cfl_number: f32,
}

impl Default for FluidParameters {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            speed_of_sound: 30.0,
            viscosity: 0.2,
            artificial_viscosity: 0.05,
            surface_tension: 0.0,
            clamp_negative_pressure: true,
            wall_restitution: 0.0,
            wall_friction: 0.3,
            max_time_step: 1.0 / 400.0,
            cfl_number: 0.4,
        }
    }
}

/// Why [`WorldSettings`] were rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SettingsError {
    /// `particle_spacing` must be > 0
    NonPositiveSpacing,
    /// `rest_density` must be > 0
    NonPositiveRestDensity,
    /// `support_radius_factor` must be >= 1, otherwise particles have no neighbors
    SupportRadiusTooSmall,
    /// `domain` must be at least a few particles wide on every axis
    EmptyDomain,
    /// the domain is too large for the spacing
    TooManyGridCells,
    /// `max_particles` must be > 0
    ZeroParticleCap,
    /// the requested backend is not built in
    UnsupportedBackend,
    /// the gpu backend is built in, but no usable gpu was found at runtime
    GpuUnavailable,
}

impl SettingsError {
    /// A short english description, identical to the c++ `describe()`.
    #[must_use]
    pub const fn describe(self) -> &'static str {
        match self {
            SettingsError::NonPositiveSpacing => "particle spacing must be greater than zero",
            SettingsError::NonPositiveRestDensity => "rest density must be greater than zero",
            SettingsError::SupportRadiusTooSmall => "support radius factor must be at least 1",
            SettingsError::EmptyDomain => "domain must be at least four particle spacings wide on every axis",
            SettingsError::TooManyGridCells => "domain is too large for the particle spacing",
            SettingsError::ZeroParticleCap => "particle cap must be greater than zero",
            SettingsError::UnsupportedBackend => "backend is not available in this build",
            SettingsError::GpuUnavailable => "no usable gpu found",
        }
    }
}

impl fmt::Display for SettingsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.describe())
    }
}

impl std::error::Error for SettingsError {}

/// What happened during the last [`World::step`](crate::World::step) call.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepStatistics {
    /// dt of the last substep in s
    pub time_step: f32,
    /// number of substeps taken
    pub substeps: u32,
    /// sum of all substeps in s; less than requested when the substep limit was hit
    pub simulated_time: f32,
    /// max rho / rho0
    pub max_density_ratio: f32,
    /// average rho / rho0
    pub average_density_ratio: f32,
    /// in m/s
    pub max_speed: f32,
    /// time spent in the whole call
    pub wall_time: Duration,
}
