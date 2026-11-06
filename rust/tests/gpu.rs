//! The gpu tests of the c++ `tests/api_tests.cpp`: the same scene on both backends.
//!
//! They skip with a note when no gpu is found, unless the environment variable `TINYSPH_REQUIRE_GPU` is set (and not
//! empty or "0"); then they fail, as in the c++ tests. CI sets it for the lavapipe job.

#![cfg(feature = "gpu")]

use std::time::Duration;

use tinysph::{
    Aabb, Attractor, Backend, DEFAULT_MAX_SUBSTEPS, Emitter, FluidParameters, Shape, StepStatistics, Vec3, Vortex, Wind, World,
    WorldSettings,
};

fn frame_seconds() -> f32 {
    1.0 / 60.0
}

/// Whether the gpu backend can run here; panics instead of skipping when `TINYSPH_REQUIRE_GPU` is set.
fn gpu_available() -> bool {
    match World::new(WorldSettings { backend: Backend::Gpu, ..Default::default() }, FluidParameters::default()) {
        Ok(world) => {
            println!("gpu backend on {}", world.device_description());
            true
        }
        Err(error) => {
            let required = std::env::var("TINYSPH_REQUIRE_GPU").is_ok_and(|value| !value.is_empty() && value != "0");
            assert!(!required, "gpu backend unavailable ({error}), and TINYSPH_REQUIRE_GPU is set");
            println!("gpu backend skipped: {error}");
            false
        }
    }
}

fn center_of_mass(world: &World) -> Vec3 {
    let mut sum = Vec3::ZERO;
    for &position in world.positions() {
        sum += position;
    }
    sum / world.particle_count() as f32
}

fn mean_speed(world: &World) -> f32 {
    let speed_sum: f32 = world.velocities().iter().map(|velocity| velocity.length()).sum();
    speed_sum / world.particle_count() as f32
}

/// Dam break plus a box collider and every kind of force field, so each gpu code path runs.
fn make_mixed_scene(backend: Backend) -> World {
    let settings = WorldSettings { domain: Aabb::new(Vec3::ZERO, Vec3::new(1.6, 1.0, 0.6)), backend, ..Default::default() };
    let mut world = World::new(settings, FluidParameters::default()).unwrap();
    world.add_collider(Shape::sphere(Vec3::new(1.1, 0.12, 0.3), 0.12));
    world.add_collider(Shape::cuboid(Vec3::new(0.75, 0.06, 0.3), Vec3::new(0.04, 0.06, 0.2)));
    world.add_force_field(Vortex { position: Vec3::new(0.3, 0.3, 0.3), axis: Vec3::new(0.0, 1.0, 0.0), strength: 5.0, radius: 0.2 });
    world.add_force_field(Wind {
        region: Aabb::new(Vec3::new(0.0, 0.4, 0.0), Vec3::new(0.5, 1.0, 0.6)),
        acceleration: Vec3::new(2.0, 0.0, 0.0),
    });
    world.add_force_field(Attractor { position: Vec3::new(1.3, 0.3, 0.3), strength: 5.0, radius: 0.3 });
    world.add_fluid(Shape::cuboid(Vec3::new(0.26, 0.31, 0.3), Vec3::new(0.24, 0.29, 0.28)), Vec3::ZERO);
    world
}

#[test]
fn one_substep_matches_the_cpu() {
    if !gpu_available() {
        return;
    }
    let mut cpu = make_mixed_scene(Backend::Cpu);
    let mut gpu = make_mixed_scene(Backend::Gpu);
    assert_eq!(cpu.particle_count(), gpu.particle_count());

    // one short substep from identical particles; the order differs, so compare the sorted densities
    cpu.step(Duration::from_millis(1));
    gpu.step(Duration::from_millis(1));
    let mut cpu_densities = cpu.densities().to_vec();
    let mut gpu_densities = gpu.densities().to_vec();
    cpu_densities.sort_by(f32::total_cmp);
    gpu_densities.sort_by(f32::total_cmp);
    let max_difference = cpu_densities
        .iter()
        .zip(&gpu_densities)
        .map(|(cpu_density, gpu_density)| (cpu_density - gpu_density).abs() / cpu_density)
        .fold(0.0f32, f32::max);
    let offset = (center_of_mass(&cpu) - center_of_mass(&gpu)).length();
    println!("   {} particles, densities differ by up to {max_difference:.2e}, centers of mass {offset:.2e} m apart", cpu.particle_count());
    assert!(max_difference < 1e-4, "densities differ by up to {max_difference:.2e}");
    assert!(offset < 1e-5, "centers of mass {offset:.2e} m apart");
}

#[test]
fn half_a_second_of_flow_matches_the_cpu() {
    if !gpu_available() {
        return;
    }
    let mut cpu = make_mixed_scene(Backend::Cpu);
    let mut gpu = make_mixed_scene(Backend::Gpu);

    // float round-off grows chaotically in a splash, so only the bulk motion can be compared after many substeps
    let mut cpu_statistics = StepStatistics::default();
    let mut gpu_statistics = StepStatistics::default();
    for _ in 0..30 {
        cpu_statistics = cpu.step_seconds(frame_seconds(), DEFAULT_MAX_SUBSTEPS);
        gpu_statistics = gpu.step_seconds(frame_seconds(), DEFAULT_MAX_SUBSTEPS);
    }
    let offset = (center_of_mass(&cpu) - center_of_mass(&gpu)).length();
    let cpu_speed = mean_speed(&cpu);
    let gpu_speed = mean_speed(&gpu);
    let density_difference = (cpu_statistics.average_density_ratio - gpu_statistics.average_density_ratio).abs();
    println!(
        "   centers of mass {offset:.4} m apart, mean speed cpu {cpu_speed:.3} gpu {gpu_speed:.3} m/s, density ratio cpu {:.4} gpu \
         {:.4}; last frame cpu {:.1} ms, gpu {:.1} ms",
        cpu_statistics.average_density_ratio,
        gpu_statistics.average_density_ratio,
        cpu_statistics.wall_time.as_secs_f64() * 1000.0,
        gpu_statistics.wall_time.as_secs_f64() * 1000.0
    );
    assert!(offset < 0.005, "centers of mass {offset:.4} m apart");
    assert!((cpu_speed - gpu_speed).abs() < 0.03 * cpu_speed, "mean speed cpu {cpu_speed:.3}, gpu {gpu_speed:.3} m/s");
    assert!(density_difference < 0.005, "average density ratio differs by {density_difference:.4}");
}

#[test]
fn emitters_outgrow_the_gpu_buffers() {
    if !gpu_available() {
        return;
    }
    let run = |backend: Backend| {
        let settings = WorldSettings { domain: Aabb::new(Vec3::ZERO, Vec3::new(1.0, 2.0, 1.0)), backend, ..Default::default() };
        let parameters = FluidParameters { gravity: Vec3::ZERO, ..Default::default() };
        let mut world = World::new(settings, parameters).unwrap();
        world.add_emitter(Emitter { position: Vec3::new(0.5, 0.1, 0.5), velocity: Vec3::new(0.0, 3.0, 0.0), radius: 0.2, enabled: true });
        for _ in 0..20 {
            world.step_seconds(frame_seconds(), DEFAULT_MAX_SUBSTEPS);
        }
        world
    };
    let cpu = run(Backend::Cpu);
    let gpu = run(Backend::Gpu);
    assert_eq!(cpu.particle_count(), gpu.particle_count(), "cpu {} particles, gpu {}", cpu.particle_count(), gpu.particle_count());
    assert!(gpu.particle_count() > 4096, "only {} particles, the buffers never grew", gpu.particle_count());

    let domain = *gpu.domain();
    let is_finite = |position: &Vec3| position.x.is_finite() && position.y.is_finite() && position.z.is_finite();
    let broken = gpu.positions().iter().filter(|position| !is_finite(position) || !domain.contains(**position)).count();
    assert_eq!(broken, 0, "{broken} particles are nan or outside");

    let offset = (center_of_mass(&cpu) - center_of_mass(&gpu)).length();
    println!("   {} particles emitted, centers of mass {offset:.2e} m apart", gpu.particle_count());
    assert!(offset < 0.005, "centers of mass {offset:.4} m apart");
}

#[test]
fn worlds_on_two_threads_are_independent() {
    if !gpu_available() {
        return;
    }
    // all worlds share one device; two of them stepping at the same time must not disturb each other
    let mut cpu = make_mixed_scene(Backend::Cpu);
    for _ in 0..30 {
        cpu.step_seconds(frame_seconds(), DEFAULT_MAX_SUBSTEPS);
    }
    let cpu_center = center_of_mass(&cpu);
    let cpu_speed = mean_speed(&cpu);
    let run = || {
        let mut gpu = make_mixed_scene(Backend::Gpu);
        for _ in 0..30 {
            gpu.step_seconds(frame_seconds(), DEFAULT_MAX_SUBSTEPS);
        }
        ((center_of_mass(&gpu) - cpu_center).length(), mean_speed(&gpu))
    };
    let results = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..2).map(|_| scope.spawn(run)).collect();
        threads.into_iter().map(|thread| thread.join().unwrap()).collect::<Vec<_>>()
    });
    for (index, (offset, gpu_speed)) in results.into_iter().enumerate() {
        println!("   world {index}: center of mass {offset:.4} m from the cpu, mean speed cpu {cpu_speed:.3} gpu {gpu_speed:.3} m/s");
        assert!(offset < 0.005, "world {index}: centers of mass {offset:.4} m apart");
        assert!((cpu_speed - gpu_speed).abs() < 0.03 * cpu_speed, "world {index}: mean speed cpu {cpu_speed:.3}, gpu {gpu_speed:.3} m/s");
    }
}
