//! One closed triangle mesh around the particles (sandbox/src/surface_mesher.cpp): every particle splats a smooth bump into
//! a grid at half the particle spacing, and marching tetrahedra extract the level at half the value inside bulk fluid.

use glam::Vec3 as V3;

const CELL_SIZE_FACTOR: f32 = 0.5;
const BUMP_RADIUS_FACTOR: f32 = 2.0;

fn bump(distance_squared: f32, radius_squared: f32) -> f32 {
    if distance_squared >= radius_squared {
        return 0.0;
    }
    let falloff = 1.0 - distance_squared / radius_squared;
    falloff * falloff * falloff
}

/// The field inside fluid at rest: the sum of the bumps of a cubic lattice at the particle spacing.
fn bulk_field_value(spacing: f32, radius: f32) -> f32 {
    let range = (radius / spacing).ceil() as i32;
    let mut sum = 0.0;
    for z in -range..=range {
        for y in -range..=range {
            for x in -range..=range {
                sum += bump((V3::new(x as f32, y as f32, z as f32) * spacing).length_squared(), radius * radius);
            }
        }
    }
    sum
}

const CORNERS: [[usize; 3]; 8] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const TETRAHEDRA: [[usize; 4]; 6] = [[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]];

/// Triangles wound counter clockwise seen from outside.
fn add_triangle(output: &mut Vec<[f32; 3]>, a: V3, b: V3, c: V3, outward: V3) {
    let counter_clockwise = (b - a).cross(c - a).dot(outward) >= 0.0;
    let (second, third) = if counter_clockwise { (b, c) } else { (c, b) };
    output.extend([a.to_array(), second.to_array(), third.to_array()]);
}

fn polygonize(output: &mut Vec<[f32; 3]>, corners: [V3; 4], values: [f32; 4], iso: f32) {
    let (mut inside, mut outside) = (Vec::with_capacity(4), Vec::with_capacity(4));
    let (mut inside_center, mut outside_center) = (V3::ZERO, V3::ZERO);
    for corner in 0..4 {
        if values[corner] > iso {
            inside.push(corner);
            inside_center += corners[corner];
        } else {
            outside.push(corner);
            outside_center += corners[corner];
        }
    }
    if inside.is_empty() || outside.is_empty() {
        return;
    }
    let outward = outside_center / outside.len() as f32 - inside_center / inside.len() as f32;
    let edge = |from: usize, to: usize| {
        let t = ((iso - values[from]) / (values[to] - values[from])).clamp(0.0, 1.0);
        corners[from] + (corners[to] - corners[from]) * t
    };
    if inside.len() == 1 || outside.len() == 1 {
        let (single, others) = if inside.len() == 1 { (inside[0], &outside) } else { (outside[0], &inside) };
        add_triangle(output, edge(single, others[0]), edge(single, others[1]), edge(single, others[2]), outward);
        return;
    }
    let point_aa = edge(inside[0], outside[0]);
    let point_ab = edge(inside[0], outside[1]);
    let point_bb = edge(inside[1], outside[1]);
    let point_ba = edge(inside[1], outside[0]);
    add_triangle(output, point_aa, point_ab, point_bb, outward);
    add_triangle(output, point_aa, point_bb, point_ba, outward);
}

/// The sample grid of one build.
struct Grid {
    origin: V3,
    cell: f32,
    points: glam::IVec3,
    reach: i32,
    radius_squared: f32,
    iso: f32,
}

impl Grid {
    fn index(&self, x: i32, y: i32, z: i32) -> usize {
        ((z * self.points.y + y) * self.points.x + x) as usize
    }

    /// Adds the bumps of all particles to the planes `first_plane..` that `field` holds.
    fn splat(&self, positions: &[V3], field: &mut [f32], first_plane: i32) {
        let plane_size = (self.points.x * self.points.y) as usize;
        let last_plane = first_plane + (field.len() / plane_size) as i32 - 1;
        for position in positions {
            let center = ((*position - self.origin) / self.cell).round().as_ivec3();
            for z in (center.z - self.reach).max(first_plane)..=(center.z + self.reach).min(last_plane) {
                let dz = self.origin.z + z as f32 * self.cell - position.z;
                for y in (center.y - self.reach).max(0)..=(center.y + self.reach).min(self.points.y - 1) {
                    let dy = self.origin.y + y as f32 * self.cell - position.y;
                    let dyz = dy * dy + dz * dz;
                    if dyz >= self.radius_squared {
                        continue;
                    }
                    for x in (center.x - self.reach).max(0)..=(center.x + self.reach).min(self.points.x - 1) {
                        let dx = self.origin.x + x as f32 * self.cell - position.x;
                        field[self.index(x, y, z - first_plane)] += bump(dx * dx + dyz, self.radius_squared);
                    }
                }
            }
        }
    }

    /// Marching tetrahedra over the cells of the layers `layers`.
    fn mesh(&self, field: &[f32], layers: std::ops::Range<i32>, output: &mut Vec<[f32; 3]>) {
        let mut positions = [V3::ZERO; 8];
        let mut values = [0.0; 8];
        for z in layers {
            for y in 0..self.points.y - 1 {
                for x in 0..self.points.x - 1 {
                    let mut inside = 0;
                    for (corner, offset) in CORNERS.iter().enumerate() {
                        values[corner] = field[self.index(x + offset[0] as i32, y + offset[1] as i32, z + offset[2] as i32)];
                        inside += usize::from(values[corner] > self.iso);
                    }
                    if inside == 0 || inside == 8 {
                        continue;
                    }
                    for (corner, offset) in CORNERS.iter().enumerate() {
                        let point = glam::IVec3::new(x + offset[0] as i32, y + offset[1] as i32, z + offset[2] as i32);
                        positions[corner] = self.origin + point.as_vec3() * self.cell;
                    }
                    for tetrahedron in TETRAHEDRA {
                        polygonize(output, tetrahedron.map(|c| positions[c]), tetrahedron.map(|c| values[c]), self.iso);
                    }
                }
            }
        }
    }
}

#[derive(Default)]
pub struct Mesher {
    field: Vec<f32>,
    pub vertices: Vec<[f32; 3]>,
}

impl Mesher {
    /// Rebuilds `vertices` (three per triangle) around `positions` within the domain `min`..`max`. Natively the z planes
    /// are split over the cpu cores, as the c++ mesher does with std::execution::par.
    pub fn build(&mut self, positions: &[V3], spacing: f32, min: V3, max: V3) {
        self.vertices.clear();
        if positions.is_empty() {
            return;
        }
        let cell = spacing * CELL_SIZE_FACTOR;
        let radius = spacing * BUMP_RADIUS_FACTOR;
        let margin = V3::splat(radius + cell);
        let grid = Grid {
            origin: min - margin,
            cell,
            points: ((max - min + margin * 2.0) / cell).ceil().as_ivec3() + 1,
            reach: (radius / cell).ceil() as i32,
            radius_squared: radius * radius,
            iso: 0.5 * bulk_field_value(spacing, radius),
        };
        let plane_size = (grid.points.x * grid.points.y) as usize;
        self.field.clear();
        self.field.resize(plane_size * grid.points.z as usize, 0.0);

        #[cfg(not(target_arch = "wasm32"))]
        let threads = std::thread::available_parallelism().map_or(1, usize::from).min(grid.points.z as usize);
        #[cfg(target_arch = "wasm32")]
        let threads = 1;
        if threads > 1 {
            #[cfg(not(target_arch = "wasm32"))]
            self.build_parallel(&grid, positions, threads);
        } else {
            grid.splat(positions, &mut self.field, 0);
            grid.mesh(&self.field, 0..grid.points.z - 1, &mut self.vertices);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn build_parallel(&mut self, grid: &Grid, positions: &[V3], threads: usize) {
        let plane_size = (grid.points.x * grid.points.y) as usize;
        let planes_per_thread = (grid.points.z as usize).div_ceil(threads);
        let layer_count = grid.points.z as usize - 1;
        std::thread::scope(|scope| {
            for (thread, chunk) in self.field.chunks_mut(plane_size * planes_per_thread).enumerate() {
                scope.spawn(move || grid.splat(positions, chunk, (thread * planes_per_thread) as i32));
            }
        });
        let field = &self.field;
        let parts: Vec<Vec<[f32; 3]>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|thread| {
                    let first = (thread * planes_per_thread).min(layer_count) as i32;
                    let end = ((thread + 1) * planes_per_thread).min(layer_count) as i32;
                    scope.spawn(move || {
                        let mut output = Vec::new();
                        grid.mesh(field, first..end, &mut output);
                        output
                    })
                })
                .collect();
            handles.into_iter().map(|handle| handle.join().expect("mesher thread")).collect()
        });
        for part in parts {
            self.vertices.extend_from_slice(&part);
        }
    }
}
