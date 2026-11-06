// 4) density and pressure: rho_i = sum_j m W_poly6(r_ij), p_i = c^2 (rho_i - rho0).

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
