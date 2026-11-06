// shared by all particle passes: the parameters, index helpers, grid helpers and the smoothing kernels.

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
