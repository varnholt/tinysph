// 4b) akinci surface normals n_i = h sum_j m / rho_j grad W_ij; only dispatched when surface tension is on.

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
