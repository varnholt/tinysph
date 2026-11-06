// 2) inclusive prefix sum, one workgroup of 256 per block. scan_blocks scans each block and stores its total,
// the totals are scanned the same way one level up, then add_block_offsets adds them back down.
// not prefixed with the common code, it has its own parameters.

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
