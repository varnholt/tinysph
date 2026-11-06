//! The presets of the c++ sandbox (sandbox/src/sandbox_scene.cpp).

use tinysph::{Aabb, Emitter, ForceField, Shape, Vec3, Vortex};

/// What a preset puts into a world.
#[derive(Clone, Debug)]
pub struct Scene {
    pub domain: Aabb,
    pub gravity: Vec3,
    pub surface_tension: f32,
    pub fluids: Vec<(Shape, Vec3)>,
    pub colliders: Vec<Shape>,
    pub fields: Vec<ForceField>,
    pub emitters: Vec<Emitter>,
}

pub const PRESET_NAMES: [&str; 4] = ["Dam break", "Drop into pool", "Two blobs (zero gravity)", "Fountain"];

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

pub fn preset(index: usize) -> Scene {
    let base = |max: Vec3| Scene {
        domain: Aabb::new(Vec3::ZERO, max),
        gravity: v(0.0, -9.81, 0.0),
        surface_tension: 0.0,
        fluids: Vec::new(),
        colliders: Vec::new(),
        fields: Vec::new(),
        emitters: Vec::new(),
    };
    match index % PRESET_NAMES.len() {
        0 => {
            let mut scene = base(v(1.6, 1.0, 0.6));
            scene.fluids.push((Shape::cuboid(v(0.26, 0.31, 0.3), v(0.24, 0.29, 0.28)), Vec3::ZERO));
            scene.colliders.push(Shape::sphere(v(1.1, 0.12, 0.3), 0.12));
            scene
        }
        1 => {
            let mut scene = base(v(1.0, 1.2, 1.0));
            scene.surface_tension = 0.5;
            scene.fluids.push((Shape::cuboid(v(0.5, 0.11, 0.5), v(0.48, 0.09, 0.48)), Vec3::ZERO));
            scene.fluids.push((Shape::sphere(v(0.5, 0.75, 0.5), 0.15), v(0.0, -1.0, 0.0)));
            scene
        }
        2 => {
            let mut scene = base(v(1.6, 1.0, 1.0));
            scene.gravity = Vec3::ZERO;
            scene.surface_tension = 1.0;
            scene.fluids.push((Shape::sphere(v(0.45, 0.5, 0.5), 0.15), v(1.0, 0.0, 0.0)));
            scene.fluids.push((Shape::sphere(v(1.15, 0.55, 0.5), 0.15), v(-1.0, 0.0, 0.0)));
            scene
        }
        _ => {
            let mut scene = base(v(1.2, 1.0, 1.2));
            scene.surface_tension = 0.5;
            scene.fluids.push((Shape::cuboid(v(0.6, 0.06, 0.6), v(0.58, 0.04, 0.58)), Vec3::ZERO));
            scene.colliders.push(Shape::cuboid(v(0.6, 0.45, 0.6), v(0.15, 0.02, 0.15)));
            scene.emitters.push(Emitter { position: v(0.6, 0.15, 0.6), velocity: v(0.0, 3.0, 0.0), radius: 0.04, enabled: true });
            scene.fields.push(ForceField::Vortex(Vortex {
                position: v(0.6, 0.5, 0.6),
                axis: v(0.0, 1.0, 0.0),
                strength: 4.0,
                radius: 0.5,
            }));
            scene
        }
    }
}
