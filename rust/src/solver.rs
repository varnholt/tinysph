//! Weakly compressible sph (wcsph) solver on the cpu, a line by line port of the c++ `src/solver.cpp`.
//!
//! Per substep:
//!
//! ```text
//!   1) sort particles into a uniform grid with cell size h (counting sort, data is physically reordered)
//!   2) density:      rho_i = sum_j m W_poly6(r_ij)                (self included)
//!   3) pressure:     p_i   = c^2 (rho_i - rho0)                    (optionally clamped at 0)
//!   4) acceleration: a_i   = -sum_j m (p_i/rho_i^2 + p_j/rho_j^2 + pi_ij) grad W_spiky
//!                            + mu sum_j m (v_j - v_i) / (rho_i rho_j) lap W_visc
//!                            + gravity + force fields
//!      pi_ij is monaghan's artificial viscosity, only for approaching pairs.
//!   5) integrate with semi-implicit euler, then resolve collisions with the domain and the colliders
//! ```
//!
//! The order of floating point operations follows the c++ code, so both produce the same numbers (see tests/parity.rs).

use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::time::Instant;

use crate::geometry::{field_acceleration, signed_distance, surface_normal};
use crate::kernels::Kernels;
use crate::math::{Aabb, Vec3, max, min};
use crate::parallel::{ChunkBuffer, Pool};
use crate::types::{Emitter, FluidParameters, ForceField, Shape, StepStatistics, WorldSettings};

/// An emitter plus the distance its last layer has travelled.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EmitterState {
    pub(crate) emitter: Emitter,
    pub(crate) travelled: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct CellCoordinate {
    x: i32,
    y: i32,
    z: i32,
}

/// Points of a `count_x * count_y * count_z` integer lattice starting at `first`, x fastest.
fn lattice(count_x: i32, count_y: i32, count_z: i32, first: i32) -> impl Iterator<Item = Vec3> {
    (0..count_x * count_y * count_z).map(move |index| {
        let x = first + index % count_x;
        let y = first + (index / count_x) % count_y;
        let z = first + index / (count_x * count_y);
        Vec3::new(x as f32, y as f32, z as f32)
    })
}

fn centered_lattice(range: i32) -> impl Iterator<Item = Vec3> {
    let count = 2 * range + 1;
    lattice(count, count, count, -range)
}

fn lattice_kernel_sum(kernels: &Kernels, spacing: f32) -> f32 {
    let range = (kernels.support_radius() / spacing).ceil() as i32;
    let mut sum = 0.0f32;
    for offset in centered_lattice(range) {
        sum += kernels.poly6((offset * spacing).length_squared());
    }
    sum
}

fn next_jitter(state: &mut u32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state as f32 / u32::MAX as f32 - 0.5
}

fn max_length(values: &[Vec3]) -> f32 {
    values.iter().map(|value| value.length_squared()).fold(0.0, max).sqrt()
}

/// Everything the parallel phases read. Shared with the worker threads through an `Arc` while a phase runs, owned by
/// the solver alone otherwise.
pub(crate) struct Core {
    settings: WorldSettings,
    parameters: FluidParameters,
    kernels: Kernels,
    particle_mass: f32,
    particle_radius: f32,

    pub(crate) colliders: Vec<Shape>,
    pub(crate) fields: Vec<ForceField>,

    // particle data, structure of arrays
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    accelerations: Vec<Vec3>,
    densities: Vec<f32>,
    pressures: Vec<f32>,
    normals: Vec<Vec3>,

    // uniform grid, rebuilt every substep
    grid_origin: Vec3,
    grid_size: CellCoordinate,
    cell_start: Vec<u32>, // particles of cell c are [cell_start[c], cell_start[c + 1])
}

pub(crate) struct Solver {
    core: Arc<Core>,
    pool: Pool,

    pub(crate) emitters: Vec<EmitterState>,

    cell_of_particle: Vec<u32>,
    sorted_order: Vec<u32>,
    cell_write_offset: Vec<u32>,

    scratch_positions: Vec<Vec3>,
    scratch_velocities: Vec<Vec3>,

    // results of the parallel phases, per chunk
    cell_buffer: ChunkBuffer<u32>,
    density_buffer: ChunkBuffer<(f32, f32)>,
    vector_buffer: ChunkBuffer<Vec3>,
    integration_buffer: ChunkBuffer<(Vec3, Vec3)>,

    max_acceleration: f32,
    jitter_state: u32,
}

impl Deref for Solver {
    type Target = Core;

    fn deref(&self) -> &Core {
        &self.core
    }
}

impl DerefMut for Solver {
    /// Only fails while a parallel phase runs, and the solver cannot be reached then.
    fn deref_mut(&mut self) -> &mut Core {
        Arc::get_mut(&mut self.core).expect("no worker holds the solver data between phases")
    }
}

impl Solver {
    pub(crate) fn new(settings: WorldSettings, parameters: FluidParameters) -> Self {
        let spacing = settings.particle_spacing;
        let kernels = Kernels::new(spacing * settings.support_radius_factor);
        let core = Core {
            settings,
            parameters,
            kernels,
            particle_mass: settings.rest_density / lattice_kernel_sum(&kernels, spacing),
            particle_radius: spacing * 0.5,
            colliders: Vec::new(),
            fields: Vec::new(),
            positions: Vec::new(),
            velocities: Vec::new(),
            accelerations: Vec::new(),
            densities: Vec::new(),
            pressures: Vec::new(),
            normals: Vec::new(),
            grid_origin: Vec3::ZERO,
            grid_size: CellCoordinate::default(),
            cell_start: Vec::new(),
        };
        let mut solver = Self {
            core: Arc::new(core),
            pool: Pool::default(),
            emitters: Vec::new(),
            cell_of_particle: Vec::new(),
            sorted_order: Vec::new(),
            cell_write_offset: Vec::new(),
            scratch_positions: Vec::new(),
            scratch_velocities: Vec::new(),
            cell_buffer: ChunkBuffer::default(),
            density_buffer: ChunkBuffer::default(),
            vector_buffer: ChunkBuffer::default(),
            integration_buffer: ChunkBuffer::default(),
            max_acceleration: 0.0,
            jitter_state: 1,
        };
        solver.set_domain(settings.domain);
        solver
    }

    pub(crate) fn settings(&self) -> &WorldSettings {
        &self.settings
    }

    pub(crate) fn parameters(&self) -> &FluidParameters {
        &self.parameters
    }

    pub(crate) fn set_parameters(&mut self, parameters: FluidParameters) {
        self.parameters = parameters;
    }

    pub(crate) fn set_domain(&mut self, domain: Aabb) {
        let minimum_extent = self.settings.particle_spacing * 4.0;
        self.settings.domain = Aabb::new(domain.min, domain.max.max(domain.min + Vec3::splat(minimum_extent)));
        self.setup_grid();
    }

    pub(crate) fn clear_particles(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.accelerations.clear();
        self.densities.clear();
        self.pressures.clear();
        self.max_acceleration = 0.0;
    }

    pub(crate) fn add_fluid_box(&mut self, bounds: Aabb, velocity: Vec3) -> usize {
        let spacing = self.settings.particle_spacing;
        let count_along = |extent: f32| 0.max((extent / spacing + 1e-3).floor() as i32);
        let size = bounds.size();
        let (count_x, count_y, count_z) = (count_along(size.x), count_along(size.y), count_along(size.z));
        let first = bounds.min + Vec3::splat(self.particle_radius);
        let mut added_count = 0;
        for offset in lattice(count_x, count_y, count_z, 0) {
            if self.add_particle(first + offset * spacing, velocity) {
                added_count += 1;
            }
        }
        added_count
    }

    pub(crate) fn add_fluid_sphere(&mut self, center: Vec3, radius: f32, velocity: Vec3) -> usize {
        let spacing = self.settings.particle_spacing;
        let steps = (radius / spacing).floor() as i32;
        let mut added_count = 0;
        for lattice_offset in centered_lattice(steps) {
            let offset = lattice_offset * spacing;
            if offset.length_squared() <= radius * radius && self.add_particle(center + offset, velocity) {
                added_count += 1;
            }
        }
        added_count
    }

    pub(crate) fn add_particle(&mut self, position: Vec3, velocity: Vec3) -> bool {
        if self.positions.len() >= self.settings.max_particles {
            return false;
        }
        let domain = self.settings.domain;
        let margin = Vec3::splat(self.particle_radius);
        if !Aabb::new(domain.min + margin, domain.max - margin).contains(position) {
            return false;
        }
        if self.colliders.iter().any(|collider| signed_distance(collider, position) < self.particle_radius) {
            return false;
        }
        let jitter_amplitude = self.settings.particle_spacing * 0.01;
        let jitter_x = next_jitter(&mut self.jitter_state);
        let jitter_y = next_jitter(&mut self.jitter_state);
        let jitter_z = next_jitter(&mut self.jitter_state);
        self.positions.push(position + Vec3::new(jitter_x, jitter_y, jitter_z) * jitter_amplitude);
        self.velocities.push(velocity);
        self.accelerations.push(Vec3::ZERO);
        let rest_density = self.settings.rest_density;
        self.densities.push(rest_density);
        self.pressures.push(0.0);
        true
    }

    /// Advances by `frame_time` using adaptive substeps, at most `max_substeps` of them.
    pub(crate) fn advance(&mut self, frame_time: f32, max_substeps: u32) -> StepStatistics {
        let start_time = Instant::now();
        let mut statistics = StepStatistics::default();
        let mut remaining_time = frame_time;
        while remaining_time > 1e-7 && statistics.substeps < max_substeps {
            let time_step = min(self.compute_time_step(), remaining_time);
            self.step(time_step);
            remaining_time -= time_step;
            statistics.simulated_time += time_step;
            statistics.time_step = time_step;
            statistics.substeps += 1;
        }
        self.fill_statistics(&mut statistics);
        statistics.wall_time = start_time.elapsed();
        statistics
    }

    /// Performs exactly one substep of the given length.
    pub(crate) fn step(&mut self, time_step: f32) {
        if !self.positions.is_empty() {
            self.rebuild_grid();
            self.compute_densities();
            if self.parameters.surface_tension > 0.0 {
                self.compute_normals();
            }
            self.compute_accelerations();
            self.integrate(time_step);
        }
        self.run_emitters(time_step);
    }

    pub(crate) fn compute_time_step(&self) -> f32 {
        self.compute_time_step_for(self.max_speed(), self.max_acceleration)
    }

    /// The adaptive time step for the given maximum speed and acceleration of all particles.
    pub(crate) fn compute_time_step_for(&self, max_speed: f32, max_acceleration: f32) -> f32 {
        let support_radius = self.kernels.support_radius();
        let mut time_step = self.parameters.max_time_step;
        time_step = min(time_step, self.parameters.cfl_number * support_radius / (self.parameters.speed_of_sound + max_speed));
        if max_acceleration > 0.0 {
            time_step = min(time_step, 0.25 * (support_radius / max_acceleration).sqrt());
        }
        time_step
    }

    fn fill_statistics(&self, statistics: &mut StepStatistics) {
        if !self.densities.is_empty() {
            let density_sum: f64 = self.densities.iter().map(|&density| f64::from(density)).sum();
            let rest_density = self.settings.rest_density;
            statistics.average_density_ratio = (density_sum / self.densities.len() as f64) as f32 / rest_density;
            statistics.max_density_ratio = self.densities.iter().copied().fold(self.densities[0], max) / rest_density;
        }
        statistics.max_speed = self.max_speed();
    }

    fn setup_grid(&mut self) {
        let cell_size = self.kernels.support_radius();
        let domain = self.settings.domain;
        self.grid_origin = domain.min - Vec3::splat(cell_size);
        let extent = domain.size() + Vec3::splat(cell_size) * 2.0;
        let cells_along = |extent: f32| 1.max((extent / cell_size).ceil() as i32);
        self.grid_size = CellCoordinate { x: cells_along(extent.x), y: cells_along(extent.y), z: cells_along(extent.z) };
        let cell_count = self.grid_size.x as usize * self.grid_size.y as usize * self.grid_size.z as usize;
        self.cell_start = vec![0; cell_count + 1];
        self.cell_write_offset = vec![0; cell_count];
    }

    /// Sorts the particles into the grid. `step` does this itself; exposed for tests.
    pub(crate) fn rebuild_grid(&mut self) {
        let particle_count = self.positions.len();
        let core = Arc::clone(&self.core);
        self.pool
            .map(particle_count, &mut self.cell_buffer, move |index| core.cell_index(core.cell_coordinate(core.positions[index])) as u32);
        self.cell_of_particle.clear();
        self.cell_of_particle.extend(self.cell_buffer.values());

        let core = Arc::get_mut(&mut self.core).expect("no worker holds the solver data between phases");
        core.cell_start.fill(0);
        for &cell in &self.cell_of_particle {
            core.cell_start[cell as usize + 1] += 1;
        }
        for index in 1..core.cell_start.len() {
            core.cell_start[index] += core.cell_start[index - 1];
        }
        let cell_count = self.cell_write_offset.len();
        self.cell_write_offset.copy_from_slice(&core.cell_start[..cell_count]);
        self.sorted_order.resize(particle_count, 0);
        for (particle_index, &cell) in self.cell_of_particle.iter().enumerate() {
            let write_offset = &mut self.cell_write_offset[cell as usize];
            self.sorted_order[*write_offset as usize] = particle_index as u32;
            *write_offset += 1;
        }

        self.scratch_positions.clear();
        self.scratch_positions.extend(self.sorted_order.iter().map(|&source| core.positions[source as usize]));
        self.scratch_velocities.clear();
        self.scratch_velocities.extend(self.sorted_order.iter().map(|&source| core.velocities[source as usize]));
        std::mem::swap(&mut core.positions, &mut self.scratch_positions);
        std::mem::swap(&mut core.velocities, &mut self.scratch_velocities);
    }

    fn compute_densities(&mut self) {
        let core = Arc::clone(&self.core);
        self.pool.map(self.positions.len(), &mut self.density_buffer, move |particle_index| core.density_and_pressure(particle_index));
        let core = Arc::get_mut(&mut self.core).expect("no worker holds the solver data between phases");
        core.densities.clear();
        core.pressures.clear();
        for &(density, pressure) in self.density_buffer.values() {
            core.densities.push(density);
            core.pressures.push(pressure);
        }
    }

    fn compute_normals(&mut self) {
        let core = Arc::clone(&self.core);
        self.pool.map(self.positions.len(), &mut self.vector_buffer, move |particle_index| core.normal(particle_index));
        let core = Arc::get_mut(&mut self.core).expect("no worker holds the solver data between phases");
        core.normals.clear();
        core.normals.extend(self.vector_buffer.values());
    }

    fn compute_accelerations(&mut self) {
        let core = Arc::clone(&self.core);
        self.pool.map(self.positions.len(), &mut self.vector_buffer, move |particle_index| core.acceleration(particle_index));
        let core = Arc::get_mut(&mut self.core).expect("no worker holds the solver data between phases");
        core.accelerations.clear();
        core.accelerations.extend(self.vector_buffer.values());
        self.max_acceleration = max_length(&core.accelerations);
    }

    fn integrate(&mut self, time_step: f32) {
        let core = Arc::clone(&self.core);
        self.pool.map(self.positions.len(), &mut self.integration_buffer, move |particle_index| {
            let mut velocity = core.velocities[particle_index] + core.accelerations[particle_index] * time_step;
            let mut position = core.positions[particle_index] + velocity * time_step;
            core.resolve_collisions(&mut position, &mut velocity);
            (position, velocity)
        });
        let core = Arc::get_mut(&mut self.core).expect("no worker holds the solver data between phases");
        core.positions.clear();
        core.velocities.clear();
        for &(position, velocity) in self.integration_buffer.values() {
            core.positions.push(position);
            core.velocities.push(velocity);
        }
    }

    fn run_emitters(&mut self, time_step: f32) {
        let spacing = self.settings.particle_spacing;
        for emitter_index in 0..self.emitters.len() {
            let speed = self.emitters[emitter_index].emitter.velocity.length();
            if !self.emitters[emitter_index].emitter.enabled || speed <= 0.0 {
                continue;
            }
            self.emitters[emitter_index].travelled += speed * time_step;
            while self.emitters[emitter_index].travelled >= spacing {
                self.emitters[emitter_index].travelled -= spacing;
                let state = self.emitters[emitter_index];
                self.emit_layer(&state);
            }
        }
    }

    fn emit_layer(&mut self, state: &EmitterState) {
        let emitter = &state.emitter;
        let spacing = self.settings.particle_spacing;
        let direction = emitter.velocity.normalize();
        let helper = if direction.y.abs() < 0.9 { Vec3::new(0.0, 1.0, 0.0) } else { Vec3::new(1.0, 0.0, 0.0) };
        let tangent = direction.cross(helper).normalize();
        let bitangent = direction.cross(tangent);
        let origin = emitter.position + direction * state.travelled;
        let steps = (emitter.radius / spacing).floor() as i32;
        let count = 2 * steps + 1;
        for point in lattice(count, count, 1, -steps) {
            let offset = (tangent * point.x + bitangent * point.y) * spacing;
            if offset.length_squared() <= emitter.radius * emitter.radius {
                self.add_particle(origin + offset, emitter.velocity);
            }
        }
    }

    pub(crate) fn particle_count(&self) -> usize {
        self.positions.len()
    }

    pub(crate) fn particle_mass(&self) -> f32 {
        self.particle_mass
    }

    pub(crate) fn particle_radius(&self) -> f32 {
        self.particle_radius
    }

    #[cfg(test)]
    pub(crate) fn support_radius(&self) -> f32 {
        self.kernels.support_radius()
    }

    #[cfg(test)]
    pub(crate) fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub(crate) fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    pub(crate) fn velocities(&self) -> &[Vec3] {
        &self.velocities
    }

    pub(crate) fn densities(&self) -> &[f32] {
        &self.densities
    }

    #[cfg(test)]
    pub(crate) fn total_momentum(&self) -> Vec3 {
        let mut sum = Vec3::ZERO;
        for &velocity in &self.velocities {
            sum += velocity;
        }
        sum * self.particle_mass
    }

    pub(crate) fn max_speed(&self) -> f32 {
        max_length(&self.velocities)
    }
}

impl Core {
    #[inline]
    fn cell_coordinate(&self, position: Vec3) -> CellCoordinate {
        let relative = (position - self.grid_origin) / self.kernels.support_radius();
        CellCoordinate {
            x: (relative.x.floor() as i32).clamp(0, self.grid_size.x - 1),
            y: (relative.y.floor() as i32).clamp(0, self.grid_size.y - 1),
            z: (relative.z.floor() as i32).clamp(0, self.grid_size.z - 1),
        }
    }

    #[inline]
    fn cell_index(&self, coordinate: CellCoordinate) -> usize {
        (coordinate.z as usize * self.grid_size.y as usize + coordinate.y as usize) * self.grid_size.x as usize + coordinate.x as usize
    }

    /// Calls `callback(neighbor_index, offset, distance_squared)` for every particle within h of the given particle,
    /// including the particle itself. Requires an up to date grid.
    #[inline]
    pub(crate) fn for_each_neighbor(&self, particle_index: usize, mut callback: impl FnMut(usize, Vec3, f32)) {
        let position = self.positions[particle_index];
        let center = self.cell_coordinate(position);
        let support_radius_squared = self.kernels.support_radius_squared();
        let size = self.grid_size;
        for z in (center.z - 1).max(0)..=(center.z + 1).min(size.z - 1) {
            for y in (center.y - 1).max(0)..=(center.y + 1).min(size.y - 1) {
                let x_first = (center.x - 1).max(0);
                let x_last = (center.x + 1).min(size.x - 1);
                // cells along x are contiguous, so their particles form one range
                let first_cell = self.cell_index(CellCoordinate { x: x_first, y, z });
                let last_cell = self.cell_index(CellCoordinate { x: x_last, y, z });
                let begin = self.cell_start[first_cell] as usize;
                let end = self.cell_start[last_cell + 1] as usize;
                for (neighbor_index, neighbor_position) in self.positions[begin..end].iter().enumerate() {
                    let offset = position - *neighbor_position;
                    let distance_squared = offset.length_squared();
                    if distance_squared < support_radius_squared {
                        callback(begin + neighbor_index, offset, distance_squared);
                    }
                }
            }
        }
    }

    fn density_and_pressure(&self, particle_index: usize) -> (f32, f32) {
        let mass = self.particle_mass;
        let mut density = 0.0f32;
        self.for_each_neighbor(particle_index, |_, _, distance_squared| density += mass * self.kernels.poly6(distance_squared));
        let mut pressure = self.parameters.speed_of_sound * self.parameters.speed_of_sound * (density - self.settings.rest_density);
        if self.parameters.clamp_negative_pressure {
            pressure = max(pressure, 0.0);
        }
        (density, pressure)
    }

    fn normal(&self, particle_index: usize) -> Vec3 {
        let mass = self.particle_mass;
        let mut normal = Vec3::ZERO;
        self.for_each_neighbor(particle_index, |neighbor_index, offset, distance_squared| {
            if neighbor_index != particle_index {
                normal += self.kernels.spiky_gradient(offset, distance_squared.sqrt()) * (mass / self.densities[neighbor_index]);
            }
        });
        normal * self.kernels.support_radius()
    }

    fn acceleration(&self, particle_index: usize) -> Vec3 {
        let mass = self.particle_mass;
        let viscosity = self.parameters.viscosity;
        let gravity = self.parameters.gravity;
        let support_radius = self.kernels.support_radius();
        let artificial_viscosity_factor = self.parameters.artificial_viscosity * self.parameters.speed_of_sound * support_radius;
        let singularity_guard = 0.01 * support_radius * support_radius;
        let surface_tension = self.parameters.surface_tension;
        let rest_density = self.settings.rest_density;
        let density = self.densities[particle_index];
        let pressure_term = self.pressures[particle_index] / (density * density);
        let velocity = self.velocities[particle_index];
        let mut pressure_acceleration = Vec3::ZERO;
        let mut viscosity_acceleration = Vec3::ZERO;
        let mut surface_acceleration = Vec3::ZERO;
        self.for_each_neighbor(particle_index, |neighbor_index, offset, distance_squared| {
            if neighbor_index == particle_index {
                return;
            }
            let distance = distance_squared.sqrt();
            let neighbor_density = self.densities[neighbor_index];
            let neighbor_pressure_term = self.pressures[neighbor_index] / (neighbor_density * neighbor_density);
            let mut pair_term = pressure_term + neighbor_pressure_term;
            let neighbor_velocity = self.velocities[neighbor_index];
            let approach = (velocity - neighbor_velocity).dot(offset);
            if approach < 0.0 {
                let average_density = 0.5 * (density + neighbor_density);
                pair_term -= artificial_viscosity_factor * approach / ((distance_squared + singularity_guard) * average_density);
            }
            pressure_acceleration -= self.kernels.spiky_gradient(offset, distance) * (mass * pair_term);
            viscosity_acceleration +=
                (neighbor_velocity - velocity) * (mass * self.kernels.viscosity_laplacian(distance) / (density * neighbor_density));
            if surface_tension > 0.0 && distance > 1e-9 {
                let correction = 2.0 * rest_density / (density + neighbor_density);
                let cohesion = offset * (mass * self.kernels.cohesion(distance) / distance);
                let curvature = self.normals[particle_index] - self.normals[neighbor_index];
                surface_acceleration -= (cohesion + curvature) * correction;
            }
        });
        let mut acceleration =
            pressure_acceleration + viscosity_acceleration * viscosity + surface_acceleration * surface_tension + gravity;
        let position = self.positions[particle_index];
        for field in &self.fields {
            acceleration += field_acceleration(field, position);
        }
        acceleration
    }

    fn resolve_collisions(&self, position: &mut Vec3, velocity: &mut Vec3) {
        let restitution = self.parameters.wall_restitution;
        let friction = self.parameters.wall_friction;
        let apply_contact = |position: &mut Vec3, velocity: &mut Vec3, normal: Vec3, penetration: f32| {
            *position += normal * penetration;
            let normal_speed = velocity.dot(normal);
            if normal_speed < 0.0 {
                let tangential_velocity = *velocity - normal * normal_speed;
                let tangential_speed = tangential_velocity.length();
                let normal_speed_change = -normal_speed * (1.0 + restitution);
                let keep = if tangential_speed > 1e-9 { max(0.0, 1.0 - friction * normal_speed_change / tangential_speed) } else { 0.0 };
                *velocity = tangential_velocity * keep - normal * (normal_speed * restitution);
            }
        };
        for collider in &self.colliders {
            let distance = signed_distance(collider, *position) - self.particle_radius;
            if distance < 0.0 {
                apply_contact(position, velocity, surface_normal(collider, *position), -distance);
            }
        }
        let margin = Vec3::splat(self.particle_radius);
        let lower = self.settings.domain.min + margin;
        let upper = self.settings.domain.max - margin;
        if position.x < lower.x {
            apply_contact(position, velocity, Vec3::new(1.0, 0.0, 0.0), lower.x - position.x);
        }
        if position.x > upper.x {
            apply_contact(position, velocity, Vec3::new(-1.0, 0.0, 0.0), position.x - upper.x);
        }
        if position.y < lower.y {
            apply_contact(position, velocity, Vec3::new(0.0, 1.0, 0.0), lower.y - position.y);
        }
        if position.y > upper.y {
            apply_contact(position, velocity, Vec3::new(0.0, -1.0, 0.0), position.y - upper.y);
        }
        if position.z < lower.z {
            apply_contact(position, velocity, Vec3::new(0.0, 0.0, 1.0), lower.z - position.z);
        }
        if position.z > upper.z {
            apply_contact(position, velocity, Vec3::new(0.0, 0.0, -1.0), position.z - upper.z);
        }
    }
}
