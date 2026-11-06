// The renderer of the c++ sandbox (sandbox/src/shader_sources.h) in wgsl.
// Fullscreen passes use gl style uv coordinates (origin bottom left); `texel` turns them into texture coordinates.

struct Frame {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    view_to_world: mat4x4<f32>,
    eye: vec4<f32>,
    light_view: vec4<f32>,
    // tan(fov_y / 2), aspect ratio, near, far
    camera: vec4<f32>,
    // sphere radius, particle radius, speed that maps to the top of the colour ramp, colour mode (0 speed, 2 flat)
    particle: vec4<f32>,
    // width, height in pixels, refraction strength
    screen: vec4<f32>,
    fluid_color: vec4<f32>,
    absorption: vec4<f32>,
};
@group(0) @binding(0) var<uniform> f: Frame;

const sun_direction = vec3<f32>(0.3904, 0.7807, 0.4880);

fn sky_gradient(direction: vec3<f32>) -> vec3<f32> {
    let horizon = vec3<f32>(0.62, 0.70, 0.78);
    let zenith = vec3<f32>(0.16, 0.32, 0.58);
    let ground = vec3<f32>(0.16, 0.15, 0.14);
    if direction.y > 0.0 {
        return mix(horizon, zenith, pow(direction.y, 0.5));
    }
    return mix(horizon, ground, pow(-direction.y, 0.35));
}

fn sky_color(direction: vec3<f32>) -> vec3<f32> {
    let sun = pow(max(dot(direction, sun_direction), 0.0), 800.0);
    return sky_gradient(direction) + vec3<f32>(1.0, 0.95, 0.85) * sun * 6.0;
}

fn to_world(direction: vec3<f32>) -> vec3<f32> {
    return (f.view_to_world * vec4<f32>(direction, 0.0)).xyz;
}

// gl uv (origin bottom left) to a texel of a texture of the given size
fn texel(uv: vec2<f32>, size: vec2<u32>) -> vec2<i32> {
    let s = vec2<i32>(size);
    return clamp(vec2<i32>(floor(vec2<f32>(uv.x, 1.0 - uv.y) * vec2<f32>(size))), vec2<i32>(0), s - 1);
}

fn clip_depth(view_position: vec3<f32>) -> f32 {
    let clip = f.projection * vec4<f32>(view_position, 1.0);
    return clip.z / clip.w;
}

// ---------------------------------------------------------------- fullscreen, sky

struct Fullscreen {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn fullscreen_vs(@builtin(vertex_index) index: u32) -> Fullscreen {
    let position = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: Fullscreen;
    out.position = vec4<f32>(position * 2.0 - 1.0, 0.0, 1.0);
    out.uv = position;
    return out;
}

@fragment
fn sky_fs(in: Fullscreen) -> @location(0) vec4<f32> {
    let ndc = in.uv * 2.0 - 1.0;
    let direction = normalize(vec3<f32>(ndc.x * f.camera.x * f.camera.y, ndc.y * f.camera.x, -1.0));
    return vec4<f32>(sky_color(normalize(to_world(direction))), 1.0);
}

// ---------------------------------------------------------------- floor and colliders

struct MeshOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) checker: f32,
};

@vertex
fn mesh_vs(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) checker: f32,
) -> MeshOut {
    var out: MeshOut;
    out.position = f.view_projection * vec4<f32>(position, 1.0);
    out.world = position;
    out.normal = normal;
    out.color = color;
    out.checker = checker;
    return out;
}

@fragment
fn mesh_fs(in: MeshOut) -> @location(0) vec4<f32> {
    let normal = normalize(in.normal);
    var color = in.color;
    if in.checker > 0.5 {
        let sum = floor(in.world.x * 10.0) + floor(in.world.z * 10.0);
        let cell = sum - 2.0 * floor(sum * 0.5);
        color *= mix(0.78, 1.0, cell);
    }
    let view_direction = normalize(f.eye.xyz - in.world);
    let diffuse = max(dot(normal, sun_direction), 0.0);
    let specular = pow(max(dot(normal, normalize(sun_direction + view_direction)), 0.0), 48.0);
    let ambient = sky_color(normal) * 0.35;
    return vec4<f32>(color * (ambient + 0.75 * diffuse) + vec3<f32>(0.15 * specular), 1.0);
}

// ---------------------------------------------------------------- particles as camera facing quads

struct SphereOut {
    @builtin(position) position: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) center: vec3<f32>,
    @location(2) value: f32,
};

fn sphere_quad(index: u32, particle: vec4<f32>, radius: f32) -> SphereOut {
    let corner = vec2<f32>(f32(index & 1u) * 2.0 - 1.0, f32((index >> 1u) & 1u) * 2.0 - 1.0);
    let center = (f.view * vec4<f32>(particle.xyz, 1.0)).xyz;
    var out: SphereOut;
    out.position = f.projection * vec4<f32>(center + vec3<f32>(corner * radius, 0.0), 1.0);
    out.corner = corner;
    out.center = center;
    out.value = particle.w / f.particle.z;
    return out;
}

@vertex
fn sphere_vs(@builtin(vertex_index) index: u32, @location(0) particle: vec4<f32>) -> SphereOut {
    return sphere_quad(index, particle, f.particle.x);
}

@vertex
fn fluid_vs(@builtin(vertex_index) index: u32, @location(0) particle: vec4<f32>) -> SphereOut {
    return sphere_quad(index, particle, f.particle.y);
}

fn water_ramp(value: f32) -> vec3<f32> {
    let deep = vec3<f32>(0.02, 0.15, 0.55);
    let mid = vec3<f32>(0.10, 0.55, 0.95);
    let foam = vec3<f32>(0.92, 0.97, 1.00);
    if value < 0.5 {
        return mix(deep, mid, value * 2.0);
    }
    return mix(mid, foam, value * 2.0 - 1.0);
}

struct ColorDepth {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn sphere_fs(in: SphereOut) -> ColorDepth {
    let radius_squared = dot(in.corner, in.corner);
    if radius_squared > 1.0 {
        discard;
    }
    let normal = vec3<f32>(in.corner, sqrt(1.0 - radius_squared));
    let light = f.light_view.xyz;
    let diffuse = max(dot(normal, light), 0.0);
    let specular = pow(max(dot(reflect(-light, normal), vec3<f32>(0.0, 0.0, 1.0)), 0.0), 32.0);
    var base = water_ramp(clamp(in.value, 0.0, 1.0));
    if f.particle.w > 1.5 {
        base = vec3<f32>(0.15, 0.45, 0.9);
    }
    var out: ColorDepth;
    out.color = vec4<f32>(base * (0.3 + 0.7 * diffuse) + vec3<f32>(specular * 0.35), 1.0);
    out.depth = clip_depth(in.center + normal * f.particle.x);
    return out;
}

struct ValueDepth {
    @location(0) value: f32,
    @builtin(frag_depth) depth: f32,
};

// linear eye depth of the front most sphere surface; 0 = empty
@fragment
fn fluid_depth_fs(in: SphereOut) -> ValueDepth {
    let radius_squared = dot(in.corner, in.corner);
    if radius_squared > 1.0 {
        discard;
    }
    let normal = vec3<f32>(in.corner, sqrt(1.0 - radius_squared));
    let view_position = in.center + normal * f.particle.y;
    var out: ValueDepth;
    out.value = -view_position.z;
    out.depth = clip_depth(view_position);
    return out;
}

// additive: all fluid along the ray
@fragment
fn fluid_thickness_fs(in: SphereOut) -> @location(0) f32 {
    let radius_squared = dot(in.corner, in.corner);
    if radius_squared > 1.0 {
        discard;
    }
    return 2.0 * sqrt(1.0 - radius_squared) * f.particle.y;
}

// ---------------------------------------------------------------- mesh surface

struct SurfaceOut {
    @builtin(position) position: vec4<f32>,
    @location(0) depth: f32,
};

@vertex
fn surface_vs(@location(0) position: vec3<f32>) -> SurfaceOut {
    let view_position = f.view * vec4<f32>(position, 1.0);
    var out: SurfaceOut;
    out.position = f.projection * view_position;
    out.depth = -view_position.z;
    return out;
}

@fragment
fn surface_depth_fs(in: SurfaceOut) -> @location(0) f32 {
    return in.depth;
}

fn linear_scene_depth(buffer_depth: f32) -> f32 {
    let near = f.camera.z;
    let far = f.camera.w;
    return near * far / (far - buffer_depth * (far - near));
}

@group(1) @binding(7) var surface_scene_depth: texture_depth_2d;

// back faces add their depth, front faces subtract theirs; the sum is the fluid length along the ray
@fragment
fn surface_thickness_fs(in: SurfaceOut, @builtin(front_facing) front: bool) -> @location(0) f32 {
    let scene_depth = linear_scene_depth(textureLoad(surface_scene_depth, vec2<i32>(in.position.xy), 0));
    let depth = min(in.depth, scene_depth);
    return select(depth, -depth, front);
}

// ---------------------------------------------------------------- bilateral smoothing

struct Smooth {
    // one texel along x or y, in uv
    direction: vec2<f32>,
    // filter radius in m
    world_radius: f32,
    // at depth 1 m
    pixels_per_meter: f32,
    // depth difference in m at which samples stop counting
    depth_falloff: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};
@group(1) @binding(0) var smooth_source: texture_2d<f32>;
@group(1) @binding(1) var<uniform> s: Smooth;

fn smooth_depth(uv: vec2<f32>) -> f32 {
    return textureLoad(smooth_source, texel(uv, textureDimensions(smooth_source)), 0).r;
}

@fragment
fn smooth_fs(in: Fullscreen) -> @location(0) f32 {
    let tap_count = 16;
    let max_radius = 256.0;
    let depth = smooth_depth(in.uv);
    if depth <= 0.0 {
        return 0.0;
    }
    let radius = clamp(s.world_radius * s.pixels_per_meter / depth, 1.0, max_radius);
    let step_size = max(1.0, radius / f32(tap_count));
    // the local slope, so that the filter follows tilted surfaces instead of flattening them
    let baseline = max(1.0, radius * 0.25);
    let forward_depth = smooth_depth(in.uv + baseline * s.direction);
    let backward_depth = smooth_depth(in.uv - baseline * s.direction);
    let forward_slope = select(1e6, (forward_depth - depth) / baseline, forward_depth > 0.0);
    let backward_slope = select(1e6, (depth - backward_depth) / baseline, backward_depth > 0.0);
    var slope = backward_slope;
    if abs(forward_slope - backward_slope) < 0.5 * s.depth_falloff / baseline {
        slope = 0.5 * (forward_slope + backward_slope);
    } else if abs(forward_slope) < abs(backward_slope) {
        slope = forward_slope;
    }
    if abs(slope) >= 1e5 {
        slope = 0.0;
    }
    let sigma = radius * 0.5;
    var sum = 0.0;
    var weight_sum = 0.0;
    for (var tap = -tap_count; tap <= tap_count; tap++) {
        let offset = f32(tap) * step_size;
        if abs(offset) > radius {
            continue;
        }
        let sample_depth = smooth_depth(in.uv + offset * s.direction);
        if sample_depth <= 0.0 {
            continue;
        }
        let spatial = exp(-offset * offset / (2.0 * sigma * sigma));
        let difference = (sample_depth - (depth + slope * offset)) / s.depth_falloff;
        let range = exp(-difference * difference);
        sum += sample_depth * spatial * range;
        weight_sum += spatial * range;
    }
    return sum / weight_sum;
}

// ---------------------------------------------------------------- composite

@group(1) @binding(2) var c_fluid_depth: texture_2d<f32>;
@group(1) @binding(3) var c_thickness: texture_2d<f32>;
@group(1) @binding(4) var c_scene: texture_2d<f32>;
@group(1) @binding(5) var c_sampler: sampler;
@group(1) @binding(6) var c_scene_depth: texture_depth_2d;

fn fluid_depth_at(uv: vec2<f32>) -> f32 {
    return textureLoad(c_fluid_depth, texel(uv, textureDimensions(c_fluid_depth)), 0).r;
}

fn view_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = uv * 2.0 - 1.0;
    return vec3<f32>(ndc.x * f.camera.x * f.camera.y * depth, ndc.y * f.camera.x * depth, -depth);
}

// the smaller of the one sided differences, so that silhouettes do not bend the normal
fn surface_derivative(uv: vec2<f32>, position: vec3<f32>, step_uv: vec2<f32>) -> vec3<f32> {
    let forward_depth = fluid_depth_at(uv + step_uv);
    let backward_depth = fluid_depth_at(uv - step_uv);
    var forward_difference = vec3<f32>(0.0);
    var backward_difference = vec3<f32>(0.0);
    if forward_depth > 0.0 {
        forward_difference = view_position(uv + step_uv, forward_depth) - position;
    }
    if backward_depth > 0.0 {
        backward_difference = position - view_position(uv - step_uv, backward_depth);
    }
    if forward_depth <= 0.0 || (backward_depth > 0.0 && abs(backward_difference.z) < abs(forward_difference.z)) {
        return backward_difference;
    }
    return forward_difference;
}

fn scene_uv(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(uv.x, 1.0 - uv.y);
}

@fragment
fn composite_fs(in: Fullscreen) -> ColorDepth {
    let size = textureDimensions(c_scene);
    let pixel = texel(in.uv, size);
    let scene_color = textureLoad(c_scene, pixel, 0).rgb;
    let scene_buffer_depth = textureLoad(c_scene_depth, pixel, 0);
    let depth = fluid_depth_at(in.uv);
    var out: ColorDepth;
    if depth <= 0.0 || depth > linear_scene_depth(scene_buffer_depth) {
        out.color = vec4<f32>(scene_color, 1.0);
        out.depth = scene_buffer_depth;
        return out;
    }
    let texel_size = 1.0 / vec2<f32>(size);
    let position = view_position(in.uv, depth);
    let derivative_x = surface_derivative(in.uv, position, vec2<f32>(texel_size.x, 0.0));
    let derivative_y = surface_derivative(in.uv, position, vec2<f32>(0.0, texel_size.y));
    var normal = cross(derivative_x, derivative_y);
    if length(normal) > 1e-12 {
        normal = normalize(normal);
    } else {
        normal = vec3<f32>(0.0, 0.0, 1.0);
    }
    let view_direction = normalize(-position);
    let thickness = textureLoad(c_thickness, pixel, 0).r;
    let radius = f.particle.y;

    // fresnel blend of the sky reflection and the refracted, absorbed scene behind the fluid
    let facing = max(dot(normal, view_direction), 0.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - facing, 5.0);
    let reflection = sky_gradient(normalize(to_world(reflect(-view_direction, normal))));
    let refracted_uv = clamp(in.uv + normal.xy * f.screen.z * min(thickness, 0.5), vec2<f32>(0.0), vec2<f32>(1.0));
    let refracted_scene = textureSampleLevel(c_scene, c_sampler, scene_uv(refracted_uv), 0.0).rgb;
    let transmittance = exp(-f.absorption.xyz * (thickness + 4.0 * radius));
    let refraction = refracted_scene * transmittance + f.fluid_color.xyz * (1.0 - transmittance);
    let specular = pow(max(dot(normal, normalize(f.light_view.xyz + view_direction)), 0.0), 120.0) * 0.6;
    let color = mix(refraction, reflection, fresnel) + vec3<f32>(specular * 1.5);
    // thin edges fade into the scene instead of ending in a hard line
    let coverage = smoothstep(0.0, radius * 0.75, thickness);
    out.color = vec4<f32>(mix(scene_color, color, coverage), 1.0);
    out.depth = clip_depth(position);
    return out;
}

// ---------------------------------------------------------------- helper lines

struct LineOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn line_vs(@location(0) position: vec3<f32>, @location(1) color: vec3<f32>) -> LineOut {
    var out: LineOut;
    out.position = f.view_projection * vec4<f32>(position, 1.0);
    out.color = color;
    return out;
}

@fragment
fn line_fs(in: LineOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
