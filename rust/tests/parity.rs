//! Runs the scenes of the c++ `tests/trace.cpp` and compares against its output in `tests/data/trace.txt`.
//!
//! Particle count and substep counts must match exactly. The float aggregates must match within a relative tolerance
//! that is tight for the first frames and looser later, because sph is chaotic: a one ulp difference somewhere (a
//! compiler fusing a multiply-add, a different `sqrt` or float-to-int path) grows with every substep. Whether the
//! particle data matches bit for bit is printed, never asserted, since it legitimately differs between platforms.

use std::fmt::Write as _;

use tinysph::{Attractor, Emitter, FluidParameters, Shape, Vec3, Vortex, Wind, World, WorldSettings};

fn fnv1a(bytes: impl Iterator<Item = u8>) -> u64 {
    bytes.fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3))
}

fn vec3_bytes(values: &[Vec3]) -> impl Iterator<Item = u8> + '_ {
    values.iter().flat_map(|value| [value.x, value.y, value.z]).flat_map(f32::to_le_bytes)
}

fn bits(value: f32) -> String {
    format!("{:08x}", value.to_bits())
}

fn dam_break() -> World {
    let settings = WorldSettings { domain: tinysph::Aabb::new(Vec3::ZERO, Vec3::new(1.6, 1.0, 0.6)), ..Default::default() };
    let mut world = World::new(settings, FluidParameters::default()).unwrap();
    world.add_collider(Shape::sphere(Vec3::new(1.1, 0.12, 0.3), 0.12));
    world.add_fluid(Shape::cuboid(Vec3::new(0.26, 0.31, 0.3), Vec3::new(0.24, 0.29, 0.28)), Vec3::ZERO);
    world
}

fn mixed() -> World {
    let settings =
        WorldSettings { domain: tinysph::Aabb::new(Vec3::ZERO, Vec3::new(1.0, 0.8, 0.5)), particle_spacing: 0.04, ..Default::default() };
    let parameters = FluidParameters { surface_tension: 0.5, wall_restitution: 0.2, ..Default::default() };
    let mut world = World::new(settings, parameters).unwrap();
    world.add_collider(Shape::sphere(Vec3::new(0.7, 0.12, 0.25), 0.1));
    world.add_collider(Shape::cuboid(Vec3::new(0.45, 0.06, 0.25), Vec3::new(0.04, 0.06, 0.15)));
    world.add_force_field(Vortex { position: Vec3::new(0.2, 0.3, 0.25), axis: Vec3::new(0.0, 1.0, 0.0), strength: 5.0, radius: 0.2 });
    world.add_force_field(Wind {
        region: tinysph::Aabb::new(Vec3::new(0.0, 0.4, 0.0), Vec3::new(0.4, 0.8, 0.5)),
        acceleration: Vec3::new(2.0, 0.0, 0.0),
    });
    world.add_force_field(Attractor { position: Vec3::new(0.85, 0.3, 0.25), strength: 5.0, radius: 0.25 });
    world.add_emitter(Emitter { position: Vec3::new(0.9, 0.5, 0.25), velocity: Vec3::new(-1.0, 0.5, 0.0), radius: 0.05, enabled: true });
    world.add_fluid(Shape::cuboid(Vec3::new(0.18, 0.2, 0.25), Vec3::new(0.16, 0.18, 0.2)), Vec3::new(0.5, 0.0, 0.0));
    world.add_fluid(Shape::sphere(Vec3::new(0.6, 0.55, 0.25), 0.08), Vec3::ZERO);
    world
}

fn trace(output: &mut String, name: &str, mut world: World) {
    writeln!(output, "scene {name} particles {}", world.particle_count()).unwrap();
    for frame in 1..=30 {
        let statistics = world.step_seconds(1.0 / 60.0, 1000);
        if frame > 3 && frame != 5 && frame % 10 != 0 {
            continue;
        }
        let (mut x, mut y, mut z, mut kinetic) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for position in world.positions() {
            x += f64::from(position.x);
            y += f64::from(position.y);
            z += f64::from(position.z);
        }
        for velocity in world.velocities() {
            kinetic += f64::from(velocity.x) * f64::from(velocity.x)
                + f64::from(velocity.y) * f64::from(velocity.y)
                + f64::from(velocity.z) * f64::from(velocity.z);
        }
        let count = world.particle_count().max(1) as f64;
        writeln!(
            output,
            "frame {frame} substeps {} dt {} time {} max_density {} average_density {} max_speed {} count {} center {} {} {} kinetic {} \
             positions {:016x} velocities {:016x} densities {:016x}",
            statistics.substeps,
            bits(statistics.time_step),
            bits(statistics.simulated_time),
            bits(statistics.max_density_ratio),
            bits(statistics.average_density_ratio),
            bits(statistics.max_speed),
            world.particle_count(),
            bits((x / count) as f32),
            bits((y / count) as f32),
            bits((z / count) as f32),
            bits((kinetic / count) as f32),
            fnv1a(vec3_bytes(world.positions())),
            fnv1a(vec3_bytes(world.velocities())),
            fnv1a(world.densities().iter().flat_map(|value| value.to_le_bytes())),
        )
        .unwrap();
    }
}

/// Relative tolerance for the float aggregates of a frame.
fn tolerance(frame: u32) -> f64 {
    match frame {
        1..=3 => 1e-4,
        4..=10 => 1e-2,
        _ => 0.1,
    }
}

fn relative_difference(expected: f32, actual: f32) -> f64 {
    let (expected, actual) = (f64::from(expected), f64::from(actual));
    (expected - actual).abs() / expected.abs().max(1e-3)
}

#[test]
fn matches_the_cpp_trace() {
    let mut actual = String::new();
    trace(&mut actual, "dam_break", dam_break());
    trace(&mut actual, "mixed", mixed());
    print!("{actual}");

    let expected = include_str!("../../tests/data/trace.txt").replace("\r\n", "\n");
    let expected_lines: Vec<&str> = expected.lines().collect();
    let actual_lines: Vec<&str> = actual.lines().collect();
    assert_eq!(expected_lines.len(), actual_lines.len(), "line count");

    let mut failures = Vec::new();
    let mut scene = String::new();
    for (expected_line, actual_line) in expected_lines.iter().zip(&actual_lines) {
        let expected_fields: Vec<&str> = expected_line.split_whitespace().collect();
        let actual_fields: Vec<&str> = actual_line.split_whitespace().collect();
        if expected_fields[0] == "scene" {
            assert_eq!(expected_line, actual_line, "scene header");
            scene = expected_fields[1].to_owned();
            continue;
        }
        assert_eq!(expected_fields.len(), actual_fields.len(), "field count in {expected_line}");
        let frame: u32 = expected_fields[1].parse().unwrap();
        let mut worst = 0.0f64;
        let mut index = 2;
        while index < expected_fields.len() {
            let key = expected_fields[index];
            let values = if key == "center" { 3 } else { 1 };
            for value_index in index + 1..=index + values {
                let (expected_value, actual_value) = (expected_fields[value_index], actual_fields[value_index]);
                match key {
                    "substeps" | "count" => {
                        if expected_value != actual_value {
                            failures.push(format!("{scene} frame {frame}: {key} {actual_value}, expected {expected_value}"));
                        }
                    }
                    "positions" | "velocities" | "densities" => {}
                    _ => {
                        let parse = |text: &str| f32::from_bits(u32::from_str_radix(text, 16).unwrap());
                        let difference = relative_difference(parse(expected_value), parse(actual_value));
                        worst = worst.max(difference);
                        if difference > tolerance(frame) {
                            failures.push(format!(
                                "{scene} frame {frame}: {key} {} vs expected {} (relative difference {difference:.2e})",
                                parse(actual_value),
                                parse(expected_value)
                            ));
                        }
                    }
                }
            }
            index += values + 1;
        }
        let hashes_match = expected_fields[expected_fields.len() - 6..] == actual_fields[actual_fields.len() - 6..];
        println!(
            "{scene} frame {frame}: particle data {}, largest relative difference of the aggregates {worst:.2e}",
            if hashes_match { "matches bit for bit" } else { "differs" }
        );
    }
    assert!(failures.is_empty(), "differences beyond tolerance:\n{}", failures.join("\n"));
}
