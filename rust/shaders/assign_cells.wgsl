// 1) the cell of every particle, and its rank inside the cell; the counts land shifted by one, so the
// inclusive prefix sum of cell_start yields the start offsets like on the cpu.

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
