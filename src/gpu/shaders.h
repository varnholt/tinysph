#pragma once

// wgsl compute shaders of the gpu backend. every pass mirrors one step of the cpu solver in solver.cpp, with the same
// kernels and the same order of terms, so both backends simulate the same fluid. each module is the common prelude
// followed by one pass; bindings are numbered per pass, the pipelines derive their layouts from the shaders.

namespace tinysph::detail::shaders
{

/// \brief shared by all particle passes: the parameters, index helpers, grid helpers and the smoothing kernels.
inline constexpr auto common = R"wgsl(
struct Params
{
   grid_origin: vec3f,
   support_radius: f32,
   grid_size: vec3i,
   particle_count: u32,
   gravity: vec3f,
   time_step: f32,
   domain_min: vec3f,
   particle_radius: f32,
   domain_max: vec3f,
   mass: f32,
   rest_density: f32,
   stiffness: f32,
   viscosity: f32,
   artificial_viscosity_factor: f32,
   singularity_guard: f32,
   surface_tension: f32,
   wall_restitution: f32,
   wall_friction: f32,
   poly6_factor: f32,
   spiky_gradient_factor: f32,
   viscosity_laplacian_factor: f32,
   cohesion_factor: f32,
   cohesion_offset: f32,
   clamp_negative_pressure: u32,
   collider_count: u32,
   field_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;

const workgroup_size = 64u;

// dispatches larger than 65535 workgroups are split into rows, see GpuSolver::dispatch
fn linearIndex(group: vec3u, groups: vec3u, local: u32) -> u32
{
   return (group.x + group.y * groups.x) * workgroup_size + local;
}

fn cellCoordinate(position: vec3f) -> vec3i
{
   let relative = (position - params.grid_origin) / params.support_radius;
   return clamp(vec3i(floor(relative)), vec3i(0), params.grid_size - vec3i(1));
}

fn cellIndex(coordinate: vec3i) -> u32
{
   return u32((coordinate.z * params.grid_size.y + coordinate.y) * params.grid_size.x + coordinate.x);
}

fn poly6(distance_squared: f32) -> f32
{
   let support_radius_squared = params.support_radius * params.support_radius;
   if (distance_squared >= support_radius_squared)
   {
      return 0.0;
   }
   let difference = support_radius_squared - distance_squared;
   return params.poly6_factor * difference * difference * difference;
}

fn spikyGradient(offset: vec3f, distance: f32) -> vec3f
{
   if (distance >= params.support_radius || distance <= 1e-9)
   {
      return vec3f(0.0);
   }
   let difference = params.support_radius - distance;
   return offset * (params.spiky_gradient_factor * difference * difference / distance);
}

fn viscosityLaplacian(distance: f32) -> f32
{
   if (distance >= params.support_radius)
   {
      return 0.0;
   }
   return params.viscosity_laplacian_factor * (params.support_radius - distance);
}

fn cohesion(distance: f32) -> f32
{
   if (distance >= params.support_radius || distance <= 0.0)
   {
      return 0.0;
   }
   let difference = params.support_radius - distance;
   let product = difference * difference * difference * distance * distance * distance;
   if (2.0 * distance > params.support_radius)
   {
      return params.cohesion_factor * product;
   }
   return params.cohesion_factor * (2.0 * product - params.cohesion_offset);
}

// the 3x3x3 block of cells around a particle, clamped to the grid. x varies fastest in the cell index, so the three
// cells of one row are contiguous in the sorted particle data and each row is a single range
struct NeighborRange
{
   lower: vec3i,
   upper: vec3i,
}

fn neighborRange(position: vec3f) -> NeighborRange
{
   let center = cellCoordinate(position);
   return NeighborRange(max(center - vec3i(1), vec3i(0)), min(center + vec3i(1), params.grid_size - vec3i(1)));
}
)wgsl";

/// \brief 1) the cell of every particle, and its rank inside the cell; the counts land shifted by one, so the
///        inclusive prefix sum of cell_start yields the start offsets like on the cpu.
inline constexpr auto assign_cells = R"wgsl(
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> cell_of_particle: array<u32>;
@group(0) @binding(3) var<storage, read_write> rank_in_cell: array<u32>;
@group(0) @binding(4) var<storage, read_write> cell_start: array<atomic<u32>>;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) local: u32)
{
   let index = linearIndex(group, groups, local);
   if (index >= params.particle_count)
   {
      return;
   }
   let cell = cellIndex(cellCoordinate(positions[index].xyz));
   cell_of_particle[index] = cell;
   rank_in_cell[index] = atomicAdd(&cell_start[cell + 1u], 1u);
}
)wgsl";

/// \brief 2) inclusive prefix sum, one workgroup of 256 per block. scan_blocks scans each block and stores its total,
///        the totals are scanned the same way one level up, then add_block_offsets adds them back down.
///        not prefixed with the common code, it has its own parameters.
inline constexpr auto scan = R"wgsl(
struct ScanParams
{
   count: u32,
}

@group(0) @binding(0) var<uniform> scan_params: ScanParams;
@group(0) @binding(1) var<storage, read_write> values: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

const block_size = 256u;
var<workgroup> partial_sums: array<u32, block_size>;

fn blockIndex(group: vec3u, groups: vec3u) -> u32
{
   return group.x + group.y * groups.x;
}

@compute @workgroup_size(block_size)
fn scan_blocks(@builtin(workgroup_id) group: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) local: u32)
{
   let block = blockIndex(group, groups);
   let index = block * block_size + local;
   var value = 0u;
   if (index < scan_params.count)
   {
      value = values[index];
   }
   partial_sums[local] = value;
   workgroupBarrier();

   // hillis-steele: log2(256) rounds, each adds the value 'offset' places to the left
   for (var offset = 1u; offset < block_size; offset = offset * 2u)
   {
      var addend = 0u;
      if (local >= offset)
      {
         addend = partial_sums[local - offset];
      }
      workgroupBarrier();
      partial_sums[local] = partial_sums[local] + addend;
      workgroupBarrier();
   }

   if (index < scan_params.count)
   {
      values[index] = partial_sums[local];
   }
   // the dispatch can round up to whole rows of workgroups; those extra blocks must not write a total
   if (local == block_size - 1u && block * block_size < scan_params.count)
   {
      block_sums[block] = partial_sums[local];
   }
}

@compute @workgroup_size(block_size)
fn add_block_offsets(@builtin(workgroup_id) group: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) local: u32)
{
   let block = blockIndex(group, groups);
   let index = block * block_size + local;
   if (block == 0u || index >= scan_params.count)
   {
      return;
   }
   values[index] = values[index] + block_sums[block - 1u];
}
)wgsl";

/// \brief 3) moves positions and velocities into cell order, so neighbors are close in memory.
inline constexpr auto scatter = R"wgsl(
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4f>;
@group(0) @binding(3) var<storage, read> cell_of_particle: array<u32>;
@group(0) @binding(4) var<storage, read> rank_in_cell: array<u32>;
@group(0) @binding(5) var<storage, read> cell_start: array<u32>;
@group(0) @binding(6) var<storage, read_write> sorted_positions: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> sorted_velocities: array<vec4f>;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) local: u32)
{
   let index = linearIndex(group, groups, local);
   if (index >= params.particle_count)
   {
      return;
   }
   let target_index = cell_start[cell_of_particle[index]] + rank_in_cell[index];
   sorted_positions[target_index] = positions[index];
   sorted_velocities[target_index] = velocities[index];
}
)wgsl";

/// \brief 4) density and pressure: rho_i = sum_j m W_poly6(r_ij), p_i = c^2 (rho_i - rho0).
inline constexpr auto density = R"wgsl(
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> cell_start: array<u32>;
@group(0) @binding(3) var<storage, read_write> density_pressure: array<vec2f>;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) local: u32)
{
   let index = linearIndex(group, groups, local);
   if (index >= params.particle_count)
   {
      return;
   }
   let position = positions[index].xyz;
   let range = neighborRange(position);
   let support_radius_squared = params.support_radius * params.support_radius;

   var density = 0.0;
   for (var z = range.lower.z; z <= range.upper.z; z++)
   {
      for (var y = range.lower.y; y <= range.upper.y; y++)
      {
         let first = cell_start[cellIndex(vec3i(range.lower.x, y, z))];
         let last = cell_start[cellIndex(vec3i(range.upper.x, y, z)) + 1u];
         for (var neighbor = first; neighbor < last; neighbor++)
         {
            let offset = position - positions[neighbor].xyz;
            let distance_squared = dot(offset, offset);
            if (distance_squared < support_radius_squared)
            {
               density += params.mass * poly6(distance_squared);
            }
         }
      }
   }

   var pressure = params.stiffness * (density - params.rest_density);
   if (params.clamp_negative_pressure != 0u)
   {
      pressure = max(pressure, 0.0);
   }
   density_pressure[index] = vec2f(density, pressure);
}
)wgsl";

/// \brief 4b) akinci surface normals n_i = h sum_j m / rho_j grad W_ij; only dispatched when surface tension is on.
inline constexpr auto normals = R"wgsl(
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> cell_start: array<u32>;
@group(0) @binding(3) var<storage, read> density_pressure: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> normals: array<vec4f>;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) local: u32)
{
   let index = linearIndex(group, groups, local);
   if (index >= params.particle_count)
   {
      return;
   }
   let position = positions[index].xyz;
   let range = neighborRange(position);
   let support_radius_squared = params.support_radius * params.support_radius;

   var normal = vec3f(0.0);
   for (var z = range.lower.z; z <= range.upper.z; z++)
   {
      for (var y = range.lower.y; y <= range.upper.y; y++)
      {
         let first = cell_start[cellIndex(vec3i(range.lower.x, y, z))];
         let last = cell_start[cellIndex(vec3i(range.upper.x, y, z)) + 1u];
         for (var neighbor = first; neighbor < last; neighbor++)
         {
            let offset = position - positions[neighbor].xyz;
            let distance_squared = dot(offset, offset);
            if (distance_squared < support_radius_squared && neighbor != index)
            {
               normal += spikyGradient(offset, sqrt(distance_squared)) * (params.mass / density_pressure[neighbor].x);
            }
         }
      }
   }
   normals[index] = vec4f(normal * params.support_radius, 0.0);
}
)wgsl";

/// \brief 5) accelerations: pressure with artificial viscosity, viscosity, surface tension, gravity, force fields.
///        also the largest acceleration, which bounds the next time step.
inline constexpr auto forces = R"wgsl(
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4f>;
@group(0) @binding(3) var<storage, read> cell_start: array<u32>;
@group(0) @binding(4) var<storage, read> density_pressure: array<vec2f>;
@group(0) @binding(5) var<storage, read> normals: array<vec4f>;
@group(0) @binding(6) var<storage, read> fields: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> accelerations: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> statistics: array<atomic<u32>>;

var<workgroup> group_max: atomic<u32>;

// three vec4 per field, the kind in the w component of the first one; see GpuSolver::uploadScene
fn fieldAcceleration(field_index: u32, point: vec3f) -> vec3f
{
   let first = fields[field_index * 3u];
   let second = fields[field_index * 3u + 1u];
   let third = fields[field_index * 3u + 2u];
   let kind = u32(first.w);

   if (kind == 0u)
   {
      // attractor: position, strength, radius
      let to_center = first.xyz - point;
      let distance = length(to_center);
      if (distance >= second.y || distance < 1e-6)
      {
         return vec3f(0.0);
      }
      return to_center * (second.x * (1.0 - distance / second.y) / distance);
   }
   if (kind == 1u)
   {
      // vortex: position, normalized axis, strength, radius
      let axis = second.xyz;
      let offset = point - first.xyz;
      let radial = offset - axis * dot(offset, axis);
      let distance = length(radial);
      if (distance >= third.x || distance < 1e-6)
      {
         return vec3f(0.0);
      }
      return cross(axis, radial / distance) * (second.w * (1.0 - distance / third.x));
   }
   // wind: region min, region max, acceleration
   if (all(point >= first.xyz) && all(point <= second.xyz))
   {
      return third.xyz;
   }
   return vec3f(0.0);
}

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) local: u32)
{
   if (local == 0u)
   {
      atomicStore(&group_max, 0u);
   }
   workgroupBarrier();

   let index = linearIndex(group, groups, local);
   if (index < params.particle_count)
   {
      let position = positions[index].xyz;
      let velocity = velocities[index].xyz;
      let density = density_pressure[index].x;
      let pressure_term = density_pressure[index].y / (density * density);
      let normal = normals[index].xyz;
      let range = neighborRange(position);
      let support_radius_squared = params.support_radius * params.support_radius;

      var pressure_acceleration = vec3f(0.0);
      var viscosity_acceleration = vec3f(0.0);
      var surface_acceleration = vec3f(0.0);
      for (var z = range.lower.z; z <= range.upper.z; z++)
      {
         for (var y = range.lower.y; y <= range.upper.y; y++)
         {
            let first = cell_start[cellIndex(vec3i(range.lower.x, y, z))];
            let last = cell_start[cellIndex(vec3i(range.upper.x, y, z)) + 1u];
            for (var neighbor = first; neighbor < last; neighbor++)
            {
               let offset = position - positions[neighbor].xyz;
               let distance_squared = dot(offset, offset);
               if (distance_squared >= support_radius_squared || neighbor == index)
               {
                  continue;
               }
               let distance = sqrt(distance_squared);
               let neighbor_density = density_pressure[neighbor].x;
               let neighbor_pressure_term = density_pressure[neighbor].y / (neighbor_density * neighbor_density);
               let neighbor_velocity = velocities[neighbor].xyz;

               var pair_term = pressure_term + neighbor_pressure_term;

               // monaghan artificial viscosity, only for pairs moving towards each other
               let approach = dot(velocity - neighbor_velocity, offset);
               if (approach < 0.0)
               {
                  let average_density = 0.5 * (density + neighbor_density);
                  pair_term -= params.artificial_viscosity_factor * approach / ((distance_squared + params.singularity_guard) * average_density);
               }

               pressure_acceleration -= spikyGradient(offset, distance) * (params.mass * pair_term);
               viscosity_acceleration += (neighbor_velocity - velocity) * (params.mass * viscosityLaplacian(distance) / (density * neighbor_density));

               if (params.surface_tension > 0.0 && distance > 1e-9)
               {
                  let correction = 2.0 * params.rest_density / (density + neighbor_density);
                  let cohesion_term = offset * (params.mass * cohesion(distance) / distance);
                  let curvature = normal - normals[neighbor].xyz;
                  surface_acceleration -= (cohesion_term + curvature) * correction;
               }
            }
         }
      }

      var acceleration = pressure_acceleration + viscosity_acceleration * params.viscosity + surface_acceleration * params.surface_tension + params.gravity;
      for (var field_index = 0u; field_index < params.field_count; field_index++)
      {
         acceleration += fieldAcceleration(field_index, position);
      }
      accelerations[index] = vec4f(acceleration, 0.0);

      // non-negative floats order like their bit patterns, so an integer max finds the float max
      atomicMax(&group_max, bitcast<u32>(dot(acceleration, acceleration)));
   }

   workgroupBarrier();
   if (local == 0u)
   {
      atomicMax(&statistics[0], atomicLoad(&group_max));
   }
}
)wgsl";

/// \brief 6) semi-implicit euler, collisions with colliders and the domain; writes back in sorted order.
///        also the largest speed, which bounds the next time step.
inline constexpr auto integrate = R"wgsl(
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4f>;
@group(0) @binding(3) var<storage, read> accelerations: array<vec4f>;
@group(0) @binding(4) var<storage, read> colliders: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> positions_out: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> velocities_out: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> statistics: array<atomic<u32>>;

var<workgroup> group_max: atomic<u32>;

// two vec4 per collider: center and kind (0 sphere, 1 box), then the radius or the half extents
fn signedDistance(collider_index: u32, point: vec3f) -> f32
{
   let first = colliders[collider_index * 2u];
   let second = colliders[collider_index * 2u + 1u];
   if (u32(first.w) == 0u)
   {
      return length(point - first.xyz) - second.x;
   }
   let offset = abs(point - first.xyz) - second.xyz;
   return length(max(offset, vec3f(0.0))) + min(max(offset.x, max(offset.y, offset.z)), 0.0);
}

fn surfaceNormal(collider_index: u32, point: vec3f) -> vec3f
{
   let epsilon = 1e-4;
   let gradient = vec3f(
      signedDistance(collider_index, point + vec3f(epsilon, 0.0, 0.0)) - signedDistance(collider_index, point - vec3f(epsilon, 0.0, 0.0)),
      signedDistance(collider_index, point + vec3f(0.0, epsilon, 0.0)) - signedDistance(collider_index, point - vec3f(0.0, epsilon, 0.0)),
      signedDistance(collider_index, point + vec3f(0.0, 0.0, epsilon)) - signedDistance(collider_index, point - vec3f(0.0, 0.0, epsilon)));
   let gradient_length = length(gradient);
   if (gradient_length > 1e-12)
   {
      return gradient / gradient_length;
   }
   return vec3f(0.0, 1.0, 0.0);
}

fn applyContact(position: ptr<function, vec3f>, velocity: ptr<function, vec3f>, normal: vec3f, penetration: f32)
{
   *position += normal * penetration;
   let normal_speed = dot(*velocity, normal);
   if (normal_speed < 0.0)
   {
      // coulomb friction, like Solver::resolveCollisions
      let tangential_velocity = *velocity - normal * normal_speed;
      let tangential_speed = length(tangential_velocity);
      let normal_speed_change = -normal_speed * (1.0 + params.wall_restitution);
      var keep = 0.0;
      if (tangential_speed > 1e-9)
      {
         keep = max(0.0, 1.0 - params.wall_friction * normal_speed_change / tangential_speed);
      }
      *velocity = tangential_velocity * keep - normal * (normal_speed * params.wall_restitution);
   }
}

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) local: u32)
{
   if (local == 0u)
   {
      atomicStore(&group_max, 0u);
   }
   workgroupBarrier();

   let index = linearIndex(group, groups, local);
   if (index < params.particle_count)
   {
      var velocity = velocities[index].xyz + accelerations[index].xyz * params.time_step;
      var position = positions[index].xyz + velocity * params.time_step;

      for (var collider_index = 0u; collider_index < params.collider_count; collider_index++)
      {
         let distance = signedDistance(collider_index, position) - params.particle_radius;
         if (distance < 0.0)
         {
            applyContact(&position, &velocity, surfaceNormal(collider_index, position), -distance);
         }
      }

      // the domain comes last so it always wins
      let lower = params.domain_min + vec3f(params.particle_radius);
      let upper = params.domain_max - vec3f(params.particle_radius);
      if (position.x < lower.x) { applyContact(&position, &velocity, vec3f(1.0, 0.0, 0.0), lower.x - position.x); }
      if (position.x > upper.x) { applyContact(&position, &velocity, vec3f(-1.0, 0.0, 0.0), position.x - upper.x); }
      if (position.y < lower.y) { applyContact(&position, &velocity, vec3f(0.0, 1.0, 0.0), lower.y - position.y); }
      if (position.y > upper.y) { applyContact(&position, &velocity, vec3f(0.0, -1.0, 0.0), position.y - upper.y); }
      if (position.z < lower.z) { applyContact(&position, &velocity, vec3f(0.0, 0.0, 1.0), lower.z - position.z); }
      if (position.z > upper.z) { applyContact(&position, &velocity, vec3f(0.0, 0.0, -1.0), position.z - upper.z); }

      positions_out[index] = vec4f(position, 0.0);
      velocities_out[index] = vec4f(velocity, 0.0);
      atomicMax(&group_max, bitcast<u32>(dot(velocity, velocity)));
   }

   workgroupBarrier();
   if (local == 0u)
   {
      atomicMax(&statistics[1], atomicLoad(&group_max));
   }
}
)wgsl";

}  // namespace tinysph::detail::shaders
