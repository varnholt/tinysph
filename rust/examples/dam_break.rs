//! Runs the dam break scene headless and prints density and timing statistics.
//!
//! usage: cargo run --release --example dam_break -- [speed_of_sound] [viscosity] [seconds] [artificial_viscosity]

use std::time::Instant;

use tinysph::{Aabb, FluidParameters, Shape, Vec3, World, WorldSettings};

fn argument(arguments: &[String], index: usize) -> Option<f32> {
    arguments.get(index).map(|text| text.parse().unwrap_or_else(|_| panic!("not a number: {text}")))
}

fn main() -> Result<(), tinysph::SettingsError> {
    let arguments: Vec<String> = std::env::args().collect();
    let mut parameters = FluidParameters::default();
    if let Some(value) = argument(&arguments, 1) {
        parameters.speed_of_sound = value;
    }
    if let Some(value) = argument(&arguments, 2) {
        parameters.viscosity = value;
    }
    if let Some(value) = argument(&arguments, 4) {
        parameters.artificial_viscosity = value;
    }
    let seconds = argument(&arguments, 3).unwrap_or(2.0);

    // 1.6 x 1.0 x 0.6 m tank, a 0.48 x 0.58 x 0.56 m water column on the left and a sphere obstacle
    let settings = WorldSettings { domain: Aabb::new(Vec3::ZERO, Vec3::new(1.6, 1.0, 0.6)), ..Default::default() };
    let mut world = World::new(settings, parameters)?;
    world.add_collider(Shape::sphere(Vec3::new(1.1, 0.12, 0.3), 0.12));
    world.add_fluid(Shape::cuboid(Vec3::new(0.26, 0.31, 0.3), Vec3::new(0.24, 0.29, 0.28)), Vec3::ZERO);
    let rest_density = world.settings().rest_density;

    println!(
        "{}: particles {}, c {}, mu {}, alpha {}",
        world.device_description(),
        world.particle_count(),
        parameters.speed_of_sound,
        parameters.viscosity,
        parameters.artificial_viscosity
    );

    let frame_count = (seconds * 60.0) as i32;
    let start_time = Instant::now();
    let mut worst_max_ratio = 0.0f32;
    let mut worst_p99_ratio = 0.0f32;
    let mut sorted_densities = Vec::new();
    for frame in 0..frame_count {
        let statistics = world.step_seconds(1.0 / 60.0, 10_000);
        sorted_densities.clear();
        sorted_densities.extend_from_slice(world.densities());
        let p99_index = sorted_densities.len() * 99 / 100;
        let p99_ratio = if sorted_densities.is_empty() {
            0.0
        } else {
            *sorted_densities.select_nth_unstable_by(p99_index, f32::total_cmp).1 / rest_density
        };
        worst_max_ratio = worst_max_ratio.max(statistics.max_density_ratio);
        worst_p99_ratio = worst_p99_ratio.max(p99_ratio);
        if frame % 10 == 0 {
            println!(
                "t {:5.2}  substeps {:3}  dt {:.5}  max {:.3}  p99 {:.3}  avg {:.3}  vmax {:.2}  {:.1} ms",
                (frame + 1) as f32 / 60.0,
                statistics.substeps,
                statistics.time_step,
                statistics.max_density_ratio,
                p99_ratio,
                statistics.average_density_ratio,
                statistics.max_speed,
                statistics.wall_time.as_secs_f64() * 1000.0
            );
        }
    }
    let elapsed_s = start_time.elapsed().as_secs_f64();

    // a settled tank should be nearly at rest
    let speeds: Vec<f32> = world.velocities().iter().map(|velocity| velocity.length()).collect();
    let mean_speed = speeds.iter().sum::<f32>() / world.particle_count() as f32;
    let fast_count = speeds.iter().filter(|&&speed| speed > 0.5).count();
    println!(
        "worst max {worst_max_ratio:.3}, worst p99 {worst_p99_ratio:.3}, final mean speed {mean_speed:.3} m/s, \
         {fast_count} particles faster than 0.5 m/s, {elapsed_s:.2} s wall for {seconds:.2} s simulated"
    );
    Ok(())
}
