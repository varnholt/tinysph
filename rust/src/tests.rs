//! Tests of internals that the public api deliberately hides; a port of the c++ `tests/solver_internal_tests.cpp`.

use crate::math::{Aabb, Vec3};
use crate::solver::Solver;
use crate::types::{FluidParameters, WorldSettings};

fn make_solver(domain: Aabb) -> Solver {
    Solver::new(WorldSettings { domain, ..Default::default() }, FluidParameters { gravity: Vec3::ZERO, ..Default::default() })
}

/// Uniform floats in [0, 1) from a 64 bit linear congruential generator.
struct Random(u64);

impl Random {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[test]
fn neighbor_search_matches_brute_force() {
    let mut solver = make_solver(Aabb::new(Vec3::ZERO, Vec3::new(0.5, 0.4, 0.3)));

    let mut random = Random(1234);
    for _ in 0..3000 {
        let position = Vec3::new(random.next() * 0.5, random.next() * 0.4, random.next() * 0.3);
        solver.add_particle(position, Vec3::ZERO);
    }
    assert!(solver.particle_count() > 2000);

    solver.rebuild_grid();
    let positions = solver.positions();
    let support_radius_squared = solver.kernels().support_radius_squared();

    let mut mismatches = 0;
    for particle_index in 0..positions.len() {
        let mut grid_neighbors = Vec::new();
        solver.for_each_neighbor(particle_index, |neighbor_index, _, _| grid_neighbors.push(neighbor_index));

        let brute_force_neighbors: Vec<usize> = (0..positions.len())
            .filter(|&neighbor_index| (positions[particle_index] - positions[neighbor_index]).length_squared() < support_radius_squared)
            .collect();

        grid_neighbors.sort_unstable();
        if grid_neighbors != brute_force_neighbors {
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "{mismatches} particles had a different neighbor set");
}

#[test]
fn resting_lattice_has_the_rest_density() {
    let mut solver = make_solver(Aabb::new(Vec3::ZERO, Vec3::splat(0.4)));
    solver.add_fluid_box(Aabb::new(Vec3::ZERO, Vec3::splat(0.4)), Vec3::ZERO);
    solver.step(1e-6);

    let rest_density = solver.settings().rest_density;
    let interior_margin = solver.support_radius() * 1.5;
    let interior = Aabb::new(Vec3::splat(interior_margin), Vec3::splat(0.4 - interior_margin));

    let mut interior_count = 0;
    let mut worst_error = 0.0f32;
    for (&position, &density) in solver.positions().iter().zip(solver.densities()) {
        if interior.contains(position) {
            interior_count += 1;
            worst_error = worst_error.max((density / rest_density - 1.0).abs());
        }
    }
    assert!(interior_count > 100);
    assert!(worst_error < 0.01, "worst interior density error {worst_error:.4}");
}

#[test]
fn momentum_is_conserved_without_walls_and_gravity() {
    let mut solver = make_solver(Aabb::new(Vec3::ZERO, Vec3::splat(4.0)));
    solver.add_fluid_sphere(Vec3::new(1.65, 2.0, 2.0), 0.15, Vec3::new(1.0, 0.0, 0.0));
    solver.add_fluid_sphere(Vec3::new(2.35, 2.05, 2.0), 0.15, Vec3::new(-1.0, 0.0, 0.0));

    let mut momentum_scale = 0.0f32;
    for velocity in solver.velocities() {
        momentum_scale += velocity.length() * solver.particle_mass();
    }
    let initial_momentum = solver.total_momentum();

    for _ in 0..24 {
        solver.advance(1.0 / 60.0, 1000);
    }

    let drift = (solver.total_momentum() - initial_momentum).length() / momentum_scale;
    assert!(drift < 1e-3, "relative momentum drift {drift:.6}");
}
