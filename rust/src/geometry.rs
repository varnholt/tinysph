//! Signed distances, normals and force field accelerations.

use crate::math::{Aabb, Vec3, max, min};
use crate::types::{Cuboid, ForceField, Shape};

fn box_distance(cuboid: &Cuboid, point: Vec3) -> f32 {
    let offset = (point - cuboid.center).abs() - cuboid.half_extents;
    let outside_distance = offset.max(Vec3::ZERO).length();
    let inside_distance = min(max(offset.x, max(offset.y, offset.z)), 0.0);
    outside_distance + inside_distance
}

/// Signed distance from a point to the shape surface, negative inside.
pub(crate) fn signed_distance(shape: &Shape, point: Vec3) -> f32 {
    match shape {
        Shape::Sphere(sphere) => (point - sphere.center).length() - sphere.radius,
        Shape::Cuboid(cuboid) => box_distance(cuboid, point),
    }
}

/// Outward surface normal at a point (gradient of the signed distance, central differences).
pub(crate) fn surface_normal(shape: &Shape, point: Vec3) -> Vec3 {
    let epsilon = 1e-4;
    let dx = Vec3::new(epsilon, 0.0, 0.0);
    let dy = Vec3::new(0.0, epsilon, 0.0);
    let dz = Vec3::new(0.0, 0.0, epsilon);
    let gradient = Vec3::new(
        signed_distance(shape, point + dx) - signed_distance(shape, point - dx),
        signed_distance(shape, point + dy) - signed_distance(shape, point - dy),
        signed_distance(shape, point + dz) - signed_distance(shape, point - dz),
    );
    gradient.normalize()
}

/// Acceleration a force field applies at a given point.
pub(crate) fn field_acceleration(field: &ForceField, point: Vec3) -> Vec3 {
    match field {
        ForceField::Attractor(attractor) => {
            let to_center = attractor.position - point;
            let distance = to_center.length();
            if distance >= attractor.radius || distance < 1e-6 {
                return Vec3::ZERO;
            }
            to_center * (attractor.strength * (1.0 - distance / attractor.radius) / distance)
        }
        ForceField::Vortex(vortex) => {
            let axis = vortex.axis.normalize();
            let offset = point - vortex.position;
            let radial = offset - axis * offset.dot(axis);
            let distance = radial.length();
            if distance >= vortex.radius || distance < 1e-6 {
                return Vec3::ZERO;
            }
            let tangent = axis.cross(radial / distance);
            tangent * (vortex.strength * (1.0 - distance / vortex.radius))
        }
        ForceField::Wind(wind) => {
            if wind.region.contains(point) {
                wind.acceleration
            } else {
                Vec3::ZERO
            }
        }
    }
}

/// Bounding box of a box shape.
pub(crate) fn cuboid_bounds(cuboid: &Cuboid) -> Aabb {
    Aabb::new(cuboid.center - cuboid.half_extents, cuboid.center + cuboid.half_extents)
}
