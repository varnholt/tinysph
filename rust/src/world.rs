//! The public simulation object.

use std::time::Duration;

use crate::geometry::cuboid_bounds;
use crate::math::{Aabb, Vec3};
use crate::solver::{EmitterState, Solver};
use crate::types::{
    Backend, ColliderId, Emitter, EmitterId, FluidParameters, ForceField, ForceFieldId, SettingsError, Shape, StepStatistics, WorldSettings,
};

const MAX_GRID_CELLS: f64 = 128.0 * 1024.0 * 1024.0;

/// The substep limit [`World::step`] uses.
pub const DEFAULT_MAX_SUBSTEPS: u32 = 1000;

/// A fluid simulation: particles inside a box shaped domain, plus colliders, force fields and emitters.
///
/// ```
/// use std::time::Duration;
/// use tinysph::{Aabb, Shape, Vec3, World, WorldSettings};
///
/// let settings = WorldSettings { domain: Aabb::new(Vec3::ZERO, Vec3::splat(1.0)), ..WorldSettings::default() };
/// let mut world = World::new(settings, Default::default())?;
/// world.add_fluid(Shape::cuboid(Vec3::new(0.25, 0.3, 0.5), Vec3::new(0.2, 0.25, 0.45)), Vec3::ZERO);
/// world.add_collider(Shape::sphere(Vec3::new(0.7, 0.1, 0.5), 0.1));
/// for _ in 0..3 {
///     world.step(Duration::from_secs(1) / 60);
///     let _positions = world.positions(); // draw them
/// }
/// # Ok::<(), tinysph::SettingsError>(())
/// ```
///
/// A world is not thread safe in the sense that it needs `&mut self` to step; it uses multiple threads internally.
pub struct World {
    solver: Solver,
    collider_ids: Vec<ColliderId>,      // parallel to solver.colliders
    force_field_ids: Vec<ForceFieldId>, // parallel to solver.fields
    emitter_ids: Vec<EmitterId>,        // parallel to solver.emitters
    next_id: u32,                       // ids are never reused, so a stale handle cannot hit a new object
}

fn position_of<Id: PartialEq>(ids: &[Id], id: Id) -> Option<usize> {
    ids.iter().position(|candidate| *candidate == id)
}

impl World {
    /// Creates a world, or tells why the settings are invalid.
    ///
    /// # Errors
    /// The first problem [`World::validate`] finds.
    pub fn new(settings: WorldSettings, parameters: FluidParameters) -> Result<World, SettingsError> {
        Self::validate(&settings)?;
        Ok(World {
            solver: Solver::new(settings, parameters),
            collider_ids: Vec::new(),
            force_field_ids: Vec::new(),
            emitter_ids: Vec::new(),
            next_id: 1,
        })
    }

    /// Checks settings without creating a world.
    ///
    /// # Errors
    /// The first invalid setting, in the same order as the c++ library checks them.
    pub fn validate(settings: &WorldSettings) -> Result<(), SettingsError> {
        if settings.particle_spacing.is_nan() || settings.particle_spacing <= 0.0 {
            return Err(SettingsError::NonPositiveSpacing);
        }
        if settings.rest_density.is_nan() || settings.rest_density <= 0.0 {
            return Err(SettingsError::NonPositiveRestDensity);
        }
        if settings.support_radius_factor.is_nan() || settings.support_radius_factor < 1.0 {
            return Err(SettingsError::SupportRadiusTooSmall);
        }
        if settings.max_particles == 0 {
            return Err(SettingsError::ZeroParticleCap);
        }
        match settings.backend {
            Backend::Cpu => {}
            Backend::Gpu => return Err(SettingsError::UnsupportedBackend),
        }
        let size = settings.domain.size();
        let minimum_extent = settings.particle_spacing * 4.0;
        if !(size.x >= minimum_extent && size.y >= minimum_extent && size.z >= minimum_extent) {
            return Err(SettingsError::EmptyDomain);
        }
        let cell_size = f64::from(settings.particle_spacing * settings.support_radius_factor);
        let cells_along = |extent: f32| f64::from(extent) / cell_size + 3.0;
        if cells_along(size.x) * cells_along(size.y) * cells_along(size.z) > MAX_GRID_CELLS {
            return Err(SettingsError::TooManyGridCells);
        }
        Ok(())
    }

    // --------------------------------------------------------------------------------------------
    // configuration
    // --------------------------------------------------------------------------------------------

    /// The settings the world was created with; the domain reflects [`World::set_domain`].
    #[must_use]
    pub fn settings(&self) -> &WorldSettings {
        self.solver.settings()
    }

    /// What the simulation runs on: `"cpu"`.
    #[must_use]
    pub fn device_description(&self) -> &str {
        "cpu"
    }

    /// The current fluid parameters.
    #[must_use]
    pub fn parameters(&self) -> &FluidParameters {
        self.solver.parameters()
    }

    /// Replaces the fluid parameters; takes effect with the next step.
    pub fn set_parameters(&mut self, parameters: FluidParameters) {
        self.solver.set_parameters(parameters);
    }

    /// The domain; at least four particle spacings wide on every axis.
    #[must_use]
    pub fn domain(&self) -> &Aabb {
        &self.solver.settings().domain
    }

    /// Changes the domain; particles outside are pushed back in during the next step.
    pub fn set_domain(&mut self, domain: Aabb) {
        self.solver.set_domain(domain);
    }

    // --------------------------------------------------------------------------------------------
    // particles
    // --------------------------------------------------------------------------------------------

    /// Fills a shape with particles at rest spacing, skipping space covered by colliders or outside the domain.
    /// Returns the number of particles added.
    pub fn add_fluid(&mut self, shape: Shape, velocity: Vec3) -> usize {
        match shape {
            Shape::Sphere(sphere) => self.solver.add_fluid_sphere(sphere.center, sphere.radius, velocity),
            Shape::Cuboid(cuboid) => self.solver.add_fluid_box(cuboid_bounds(&cuboid), velocity),
        }
    }

    /// Adds one particle. Returns false if it is outside the domain, inside a collider, or the particle cap is reached.
    pub fn add_particle(&mut self, position: Vec3, velocity: Vec3) -> bool {
        self.solver.add_particle(position, velocity)
    }

    /// Removes all particles.
    pub fn clear_particles(&mut self) {
        self.solver.clear_particles();
    }

    /// Number of particles.
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.solver.particle_count()
    }

    /// In kg, chosen so that particles at rest spacing have the rest density.
    #[must_use]
    pub fn particle_mass(&self) -> f32 {
        self.solver.particle_mass()
    }

    /// Half the spacing in m, used for collisions and drawing.
    #[must_use]
    pub fn particle_radius(&self) -> f32 {
        self.solver.particle_radius()
    }

    /// Particle positions. The order changes between steps.
    #[must_use]
    pub fn positions(&self) -> &[Vec3] {
        self.solver.positions()
    }

    /// Particle velocities, in the same order as the positions.
    #[must_use]
    pub fn velocities(&self) -> &[Vec3] {
        self.solver.velocities()
    }

    /// Particle densities in kg/m^3, in the same order as the positions.
    #[must_use]
    pub fn densities(&self) -> &[f32] {
        self.solver.densities()
    }

    // --------------------------------------------------------------------------------------------
    // colliders, force fields, emitters
    // --------------------------------------------------------------------------------------------

    fn next_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Adds a static collider.
    pub fn add_collider(&mut self, shape: Shape) -> ColliderId {
        let id = ColliderId(self.next_id());
        self.solver.colliders.push(shape);
        self.collider_ids.push(id);
        id
    }

    /// Moves or reshapes a collider. Returns false if the id is unknown.
    pub fn set_collider(&mut self, id: ColliderId, shape: Shape) -> bool {
        position_of(&self.collider_ids, id).map(|index| self.solver.colliders[index] = shape).is_some()
    }

    /// Removes a collider. Returns false if the id is unknown.
    pub fn remove_collider(&mut self, id: ColliderId) -> bool {
        position_of(&self.collider_ids, id)
            .map(|index| {
                self.solver.colliders.remove(index);
                self.collider_ids.remove(index);
            })
            .is_some()
    }

    /// The collider's shape, or `None` if the id is unknown.
    #[must_use]
    pub fn collider(&self, id: ColliderId) -> Option<Shape> {
        position_of(&self.collider_ids, id).map(|index| self.solver.colliders[index])
    }

    /// Adds a force field.
    pub fn add_force_field(&mut self, field: impl Into<ForceField>) -> ForceFieldId {
        let id = ForceFieldId(self.next_id());
        self.solver.fields.push(field.into());
        self.force_field_ids.push(id);
        id
    }

    /// Changes a force field. Returns false if the id is unknown.
    pub fn set_force_field(&mut self, id: ForceFieldId, field: impl Into<ForceField>) -> bool {
        let field = field.into();
        position_of(&self.force_field_ids, id).map(|index| self.solver.fields[index] = field).is_some()
    }

    /// Removes a force field. Returns false if the id is unknown.
    pub fn remove_force_field(&mut self, id: ForceFieldId) -> bool {
        position_of(&self.force_field_ids, id)
            .map(|index| {
                self.solver.fields.remove(index);
                self.force_field_ids.remove(index);
            })
            .is_some()
    }

    /// The force field, or `None` if the id is unknown.
    #[must_use]
    pub fn force_field(&self, id: ForceFieldId) -> Option<ForceField> {
        position_of(&self.force_field_ids, id).map(|index| self.solver.fields[index])
    }

    /// Adds an emitter.
    pub fn add_emitter(&mut self, emitter: Emitter) -> EmitterId {
        let id = EmitterId(self.next_id());
        self.solver.emitters.push(EmitterState { emitter, travelled: 0.0 });
        self.emitter_ids.push(id);
        id
    }

    /// Changes an emitter, keeping how far its last layer has travelled. Returns false if the id is unknown.
    pub fn set_emitter(&mut self, id: EmitterId, emitter: Emitter) -> bool {
        position_of(&self.emitter_ids, id).map(|index| self.solver.emitters[index].emitter = emitter).is_some()
    }

    /// Removes an emitter. Returns false if the id is unknown.
    pub fn remove_emitter(&mut self, id: EmitterId) -> bool {
        position_of(&self.emitter_ids, id)
            .map(|index| {
                self.solver.emitters.remove(index);
                self.emitter_ids.remove(index);
            })
            .is_some()
    }

    /// The emitter, or `None` if the id is unknown.
    #[must_use]
    pub fn emitter(&self, id: EmitterId) -> Option<Emitter> {
        position_of(&self.emitter_ids, id).map(|index| self.solver.emitters[index].emitter)
    }

    // --------------------------------------------------------------------------------------------
    // simulation
    // --------------------------------------------------------------------------------------------

    /// Advances the simulation by the given time using adaptive substeps, at most [`DEFAULT_MAX_SUBSTEPS`] of them.
    pub fn step(&mut self, time: Duration) -> StepStatistics {
        self.step_with_limit(time, DEFAULT_MAX_SUBSTEPS)
    }

    /// Like [`World::step`], but stops after `max_substeps`, so a slow machine runs in slow motion instead of
    /// freezing; [`StepStatistics::simulated_time`] tells how far it got.
    pub fn step_with_limit(&mut self, time: Duration, max_substeps: u32) -> StepStatistics {
        // a Duration holds whole nanoseconds, so 1 s / 60 is not exactly the f32 1.0 / 60.0; use step_seconds for that
        self.step_seconds(time.as_secs_f64() as f32, max_substeps)
    }

    /// Like [`World::step_with_limit`], with the time given as `f32` seconds exactly as the c++ library takes it.
    pub fn step_seconds(&mut self, seconds: f32, max_substeps: u32) -> StepStatistics {
        self.solver.advance(seconds, max_substeps)
    }
}

impl std::fmt::Debug for World {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("World")
            .field("settings", self.settings())
            .field("parameters", self.parameters())
            .field("particle_count", &self.particle_count())
            .field("colliders", &self.collider_ids.len())
            .field("force_fields", &self.force_field_ids.len())
            .field("emitters", &self.emitter_ids.len())
            .finish_non_exhaustive()
    }
}
