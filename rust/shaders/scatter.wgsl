// 3) moves positions and velocities into cell order, so neighbors are close in memory.

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
