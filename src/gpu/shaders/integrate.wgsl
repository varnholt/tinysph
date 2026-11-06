// 6) semi-implicit euler, collisions with colliders and the domain; writes back in sorted order.
// also the largest speed, which bounds the next time step.

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
