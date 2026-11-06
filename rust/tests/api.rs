//! Tests that only use the public api, the way a library user would; a port of the c++ `tests/api_tests.cpp`.

use std::time::Duration;

use tinysph::{Aabb, Attractor, Backend, DEFAULT_MAX_SUBSTEPS, Emitter, FluidParameters, SettingsError, Shape, Vec3, World, WorldSettings};

const FRAME_TIME: Duration = Duration::from_nanos(1_000_000_000 / 60);

fn frame_seconds() -> f32 {
    1.0 / 60.0
}

fn is_finite(value: Vec3) -> bool {
    value.x.is_finite() && value.y.is_finite() && value.z.is_finite()
}

fn mean_speed(world: &World) -> f32 {
    let speed_sum: f32 = world.velocities().iter().map(|velocity| velocity.length()).sum();
    speed_sum / world.particle_count() as f32
}

fn domain(max: Vec3) -> WorldSettings {
    WorldSettings { domain: Aabb::new(Vec3::ZERO, max), ..Default::default() }
}

fn no_gravity() -> FluidParameters {
    FluidParameters { gravity: Vec3::ZERO, ..Default::default() }
}

/// 1.6 x 1.0 x 0.6 m tank, a 0.48 x 0.58 x 0.56 m water column on the left and a sphere obstacle.
fn make_dam_break_world() -> World {
    let mut world = World::new(domain(Vec3::new(1.6, 1.0, 0.6)), FluidParameters::default()).unwrap();
    world.add_collider(Shape::sphere(Vec3::new(1.1, 0.12, 0.3), 0.12));
    world.add_fluid(Shape::cuboid(Vec3::new(0.26, 0.31, 0.3), Vec3::new(0.24, 0.29, 0.28)), Vec3::ZERO);
    world
}

/// Steps one frame of 1/60 s, passed as f32 seconds like the c++ `std::chrono::duration<double>` frame time.
fn step_frame(world: &mut World) -> tinysph::StepStatistics {
    world.step_seconds(frame_seconds(), DEFAULT_MAX_SUBSTEPS)
}

#[test]
fn version_matches_the_package_version() {
    let (major, minor, patch) = tinysph::VERSION;
    assert_eq!(major.to_string(), env!("CARGO_PKG_VERSION_MAJOR"));
    assert_eq!(minor.to_string(), env!("CARGO_PKG_VERSION_MINOR"));
    assert_eq!(patch.to_string(), env!("CARGO_PKG_VERSION_PATCH"));
}

#[test]
fn invalid_settings_are_reported_not_crashed_on() {
    assert!(World::validate(&WorldSettings::default()).is_ok());
    assert!(World::new(WorldSettings::default(), FluidParameters::default()).is_ok());

    let check_rejected = |settings: WorldSettings, expected_error: SettingsError| {
        let world = World::new(settings, FluidParameters::default());
        assert_eq!(world.err(), Some(expected_error), "expected '{expected_error}'");
    };
    let defaults = WorldSettings::default();
    check_rejected(WorldSettings { particle_spacing: 0.0, ..defaults }, SettingsError::NonPositiveSpacing);
    check_rejected(WorldSettings { particle_spacing: f32::NAN, ..defaults }, SettingsError::NonPositiveSpacing);
    check_rejected(WorldSettings { rest_density: -1.0, ..defaults }, SettingsError::NonPositiveRestDensity);
    check_rejected(WorldSettings { support_radius_factor: 0.5, ..defaults }, SettingsError::SupportRadiusTooSmall);
    check_rejected(domain(Vec3::new(1.0, 0.0, 1.0)), SettingsError::EmptyDomain);
    check_rejected(WorldSettings { particle_spacing: 0.001, ..domain(Vec3::splat(100.0)) }, SettingsError::TooManyGridCells);
    check_rejected(WorldSettings { max_particles: 0, ..defaults }, SettingsError::ZeroParticleCap);
    if cfg!(feature = "gpu") {
        // built in; whether a gpu is found depends on the machine, tests/gpu.rs covers it
        let world = World::new(WorldSettings { backend: Backend::Gpu, ..defaults }, FluidParameters::default());
        assert!(world.as_ref().err().is_none_or(|error| *error == SettingsError::GpuUnavailable), "unexpected {:?}", world.err());
    } else {
        check_rejected(WorldSettings { backend: Backend::Gpu, ..defaults }, SettingsError::UnsupportedBackend);
    }

    assert!(World::new(WorldSettings { particle_spacing: -1.0, ..defaults }, FluidParameters::default()).is_err());
}

#[test]
fn step_takes_any_duration() {
    let mut world = World::new(WorldSettings::default(), no_gravity()).unwrap();
    world.add_fluid(Shape::sphere(Vec3::splat(0.5), 0.1), Vec3::ZERO);

    // the substeps add up in float, so compare with a tolerance
    let matches = |simulated: f32, requested: Duration| (simulated - requested.as_secs_f32()).abs() < 1e-6;
    assert!(matches(world.step(Duration::from_millis(16)).simulated_time, Duration::from_millis(16)));
    assert!(matches(world.step(FRAME_TIME).simulated_time, FRAME_TIME));

    // max_substeps caps the work; the statistics tell how far it got
    let capped = world.step_with_limit(Duration::from_secs(1), 3);
    assert_eq!(capped.substeps, 3);
    assert!(capped.simulated_time < 1.0);
}

#[test]
fn handles_are_unique_stale_handles_are_rejected() {
    let mut world = World::new(WorldSettings::default(), FluidParameters::default()).unwrap();
    let first = world.add_collider(Shape::sphere(Vec3::splat(0.5), 0.1));
    let second = world.add_collider(Shape::cuboid(Vec3::splat(0.2), Vec3::splat(0.05)));
    assert_ne!(first, second);

    assert!(world.set_collider(first, Shape::sphere(Vec3::splat(0.5), 0.2)));
    let updated = world.collider(first);
    assert!(matches!(updated, Some(Shape::Sphere(sphere)) if sphere.radius == 0.2));

    assert!(world.remove_collider(first));
    assert!(!world.remove_collider(first));
    assert!(world.collider(first).is_none());
    assert!(!world.set_collider(first, tinysph::Sphere::default().into()));
    assert!(world.collider(second).is_some());

    // ids are never reused, so the stale handle cannot reach the new object
    let third = world.add_collider(tinysph::Sphere::default().into());
    assert_ne!(third, first);
    assert!(world.collider(first).is_none());

    let field = world.add_force_field(Attractor::default());
    assert!(world.force_field(field).is_some());
    assert!(world.remove_force_field(field));
    assert!(world.force_field(field).is_none());

    let emitter = world.add_emitter(Emitter::default());
    assert!(world.set_emitter(emitter, Emitter { enabled: false, ..Default::default() }));
    assert!(world.emitter(emitter).is_some_and(|stored| !stored.enabled));
    assert!(world.remove_emitter(emitter));
}

#[test]
fn add_fluid_fills_shapes_but_not_colliders_or_space_outside_the_domain() {
    let mut world = World::new(domain(Vec3::splat(1.0)), FluidParameters::default()).unwrap();
    let spacing = world.settings().particle_spacing;

    let box_count = world.add_fluid(Shape::cuboid(Vec3::splat(0.5), Vec3::splat(0.1)), Vec3::ZERO);
    assert_eq!(box_count, 8 * 8 * 8, "{box_count} particles in a 0.2 m box at {spacing} m spacing");

    world.clear_particles();
    world.add_collider(Shape::sphere(Vec3::splat(0.5), 0.1));
    let around_collider = world.add_fluid(Shape::cuboid(Vec3::splat(0.5), Vec3::splat(0.1)), Vec3::ZERO);
    assert!(around_collider < box_count / 2);

    world.clear_particles();
    let half_outside = world.add_fluid(Shape::cuboid(Vec3::new(1.0, 0.5, 0.5), Vec3::splat(0.1)), Vec3::ZERO);
    assert!(half_outside > box_count / 4 && half_outside <= box_count / 2, "{half_outside} of {box_count} particles inside");
    assert!(!world.add_particle(Vec3::new(2.0, 0.5, 0.5), Vec3::ZERO));
}

#[test]
fn particles_follow_a_shrinking_domain() {
    let mut world = World::new(domain(Vec3::splat(1.0)), FluidParameters::default()).unwrap();
    world.add_fluid(Shape::cuboid(Vec3::new(0.5, 0.2, 0.5), Vec3::new(0.45, 0.15, 0.45)), Vec3::ZERO);
    world.set_domain(Aabb::new(Vec3::ZERO, Vec3::new(0.6, 1.0, 1.0)));
    step_frame(&mut world);

    let domain = *world.domain();
    let outside = world.positions().iter().filter(|position| !domain.contains(**position)).count();
    assert_eq!(outside, 0, "{outside} particles outside the new domain");
}

#[test]
fn a_world_can_be_moved() {
    let world = make_dam_break_world();
    let particle_count = world.particle_count();
    let mut moved = world;
    step_frame(&mut moved);
    assert_eq!(moved.particle_count(), particle_count);
}

#[test]
fn dam_break_stays_stable_and_inside_the_domain() {
    let mut world = make_dam_break_world();
    let particle_count = world.particle_count();
    println!("   dam break: {particle_count} particles");

    let rest_density = world.settings().rest_density;
    let mut max_density_ratio = 0.0f32;
    let mut max_p99_density_ratio = 0.0f32;
    let mut wall_time = Duration::ZERO;
    let mut sorted_densities = Vec::new();
    for _ in 0..90 {
        let statistics = step_frame(&mut world);
        max_density_ratio = max_density_ratio.max(statistics.max_density_ratio);
        wall_time += statistics.wall_time;

        // single particles spike at impacts; the 99th percentile shows how compressible the bulk is
        sorted_densities.clear();
        sorted_densities.extend_from_slice(world.densities());
        let p99_position = sorted_densities.len() * 99 / 100;
        let (_, p99, _) = sorted_densities.select_nth_unstable_by(p99_position, f32::total_cmp);
        max_p99_density_ratio = max_p99_density_ratio.max(*p99 / rest_density);
    }
    println!(
        "   1.5 s simulated in {:.2} s, max density ratio {max_density_ratio:.3}, p99 {max_p99_density_ratio:.3}",
        wall_time.as_secs_f64()
    );

    assert_eq!(world.particle_count(), particle_count);
    assert!(max_p99_density_ratio < 1.1, "p99 density ratio {max_p99_density_ratio:.3}");
    assert!(max_density_ratio < 1.3, "max density ratio {max_density_ratio:.3}");

    let domain = *world.domain();
    let obstacle_center = Vec3::new(1.1, 0.12, 0.3);
    let mut escaped_count = 0;
    let mut non_finite_count = 0;
    let mut inside_collider_count = 0;
    for (&position, &velocity) in world.positions().iter().zip(world.velocities()) {
        if !is_finite(position) || !is_finite(velocity) {
            non_finite_count += 1;
            continue;
        }
        if !domain.contains(position) {
            escaped_count += 1;
        }
        if (position - obstacle_center).length() < 0.12 {
            inside_collider_count += 1;
        }
    }
    assert_eq!(non_finite_count, 0, "{non_finite_count} particles are nan/inf");
    assert_eq!(escaped_count, 0, "{escaped_count} particles left the domain");
    assert_eq!(inside_collider_count, 0, "{inside_collider_count} particles are inside the sphere collider");
}

#[test]
fn dam_break_comes_to_rest() {
    // without artificial viscosity wcsph keeps 'boiling': after 6 s most particles still moved faster than 0.5 m/s
    let mut world = make_dam_break_world();
    for _ in 0..240 {
        step_frame(&mut world);
    }
    let mean_speed = mean_speed(&world);
    assert!(mean_speed < 0.05, "mean speed after 4 s is {mean_speed:.3} m/s");
}

#[test]
fn emitter_spawns_whole_layers_at_the_expected_rate() {
    let mut world = World::new(domain(Vec3::new(1.0, 2.0, 1.0)), no_gravity()).unwrap();
    let spacing = world.settings().particle_spacing;
    let speed = 2.0f32;
    world.add_emitter(Emitter {
        position: Vec3::new(0.5, 0.1, 0.5),
        velocity: Vec3::new(0.0, speed, 0.0),
        radius: 0.04,
        ..Default::default()
    });

    // slightly more than the time for one layer
    let first_time = spacing / speed * 1.01;
    world.step_seconds(first_time, DEFAULT_MAX_SUBSTEPS);
    let layer_size = world.particle_count();
    assert!(layer_size > 0);

    for _ in 0..6 {
        step_frame(&mut world);
    }

    let total_time = first_time + 6.0 * frame_seconds();
    let expected_layers = (speed * total_time / spacing).floor();
    let layers = world.particle_count() as f32 / layer_size as f32;
    assert!((layers - expected_layers).abs() <= 1.0, "{layers} layers emitted, expected {expected_layers}");
}

#[test]
fn surface_tension_merges_colliding_drops() {
    // without it the drops burst into spray: after 1 s about half of the particles are isolated
    let parameters = FluidParameters { gravity: Vec3::ZERO, surface_tension: 1.0, ..Default::default() };
    let mut world = World::new(domain(Vec3::new(1.6, 1.0, 1.0)), parameters).unwrap();
    world.add_fluid(Shape::sphere(Vec3::new(0.45, 0.5, 0.5), 0.15), Vec3::new(1.0, 0.0, 0.0));
    world.add_fluid(Shape::sphere(Vec3::new(1.15, 0.55, 0.5), 0.15), Vec3::new(-1.0, 0.0, 0.0));
    for _ in 0..60 {
        step_frame(&mut world);
    }

    let rest_density = world.settings().rest_density;
    let isolated = world.densities().iter().filter(|&&density| density < 0.35 * rest_density).count();
    assert!(isolated < 20, "{isolated} of {} particles isolated", world.particle_count());

    // one merged sphere of both volumes has a mean distance to its center of about 0.14 m
    let positions = world.positions();
    let center = positions.iter().fold(Vec3::ZERO, |sum, &position| sum + position) / positions.len() as f32;
    let spread = positions.iter().map(|&position| (position - center).length()).sum::<f32>() / positions.len() as f32;
    assert!(spread < 0.2, "mean distance to the center {spread:.3} m");
}

#[test]
fn an_attractor_pulls_fluid_towards_it() {
    let mut world = World::new(domain(Vec3::splat(1.0)), no_gravity()).unwrap();
    world.add_fluid(Shape::sphere(Vec3::new(0.3, 0.5, 0.5), 0.1), Vec3::ZERO);
    world.add_force_field(Attractor { position: Vec3::new(0.7, 0.5, 0.5), strength: 30.0, radius: 0.6 });
    world.step(Duration::from_millis(200));

    let x_sum: f32 = world.positions().iter().map(|position| position.x).sum();
    let center_x = x_sum / world.particle_count() as f32;
    assert!(center_x > 0.35, "fluid center moved to x = {center_x:.3}");
}
