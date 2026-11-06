// 5) accelerations: pressure with artificial viscosity, viscosity, surface tension, gravity, force fields.
// also the largest acceleration, which bounds the next time step.

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
