//! The smallest useful tinysph program: break a dam, let it hit an obstacle, print what happens.

use std::time::Duration;

use tinysph::{Aabb, Shape, Vec3, World, WorldSettings};

fn main() -> Result<(), tinysph::SettingsError> {
    // a 1.6 x 1.0 x 0.6 m tank
    let settings = WorldSettings { domain: Aabb::new(Vec3::ZERO, Vec3::new(1.6, 1.0, 0.6)), ..Default::default() };
    let mut world = World::new(settings, Default::default())?;

    // a column of water on the left and a ball in its way
    world.add_fluid(Shape::cuboid(Vec3::new(0.25, 0.3, 0.3), Vec3::new(0.23, 0.28, 0.28)), Vec3::ZERO);
    world.add_collider(Shape::sphere(Vec3::new(1.1, 0.12, 0.3), 0.12));

    let (major, minor, patch) = tinysph::VERSION;
    println!("tinysph {major}.{minor}.{patch}: {} particles", world.particle_count());

    for frame in 0..=120 {
        let statistics = world.step(Duration::from_secs(1) / 60);
        if frame % 20 == 0 {
            // the front of the wave: the right most particle
            let front = world.positions().iter().map(|position| position.x).fold(f32::MIN, f32::max);
            println!(
                "t = {:.2} s   wave front at x = {front:.2} m   max speed {:.2} m/s   {} substeps",
                frame as f32 / 60.0,
                statistics.max_speed,
                statistics.substeps
            );
        }
    }
    Ok(())
}
