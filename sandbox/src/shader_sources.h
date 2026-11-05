#pragma once

// all shaders stick to the glsl es 3.00 subset (webgl2): no compute, no geometry shaders, dynamic loops only.
// the header is the only platform specific part.

namespace shaders
{

#ifdef __EMSCRIPTEN__
inline constexpr const char* header = "#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\n";
#else
inline constexpr const char* header = "#version 330 core\n";
#endif

// ----------------------------------------------------------------------------------------------------
// shared snippets
// ----------------------------------------------------------------------------------------------------

/// procedural sky, used as background and for reflections. the sun doubles as the key light.
inline constexpr const char* sky_function = R"(
const vec3 sun_direction = vec3(0.3904, 0.7807, 0.4880);

// the sky without the sun; reflections use this and add the sun as a specular term, otherwise every noisy
// normal that happens to point at the tiny sun disk turns into a white sparkle
vec3 skyGradient(vec3 direction)
{
   vec3 horizon = vec3(0.62, 0.70, 0.78);
   vec3 zenith = vec3(0.16, 0.32, 0.58);
   vec3 ground = vec3(0.16, 0.15, 0.14);
   return direction.y > 0.0 ? mix(horizon, zenith, pow(direction.y, 0.5)) : mix(horizon, ground, pow(-direction.y, 0.35));
}

vec3 skyColor(vec3 direction)
{
   float sun = pow(max(dot(direction, sun_direction), 0.0), 800.0);
   return skyGradient(direction) + vec3(1.0, 0.95, 0.85) * sun * 6.0;
}
)";

// ----------------------------------------------------------------------------------------------------
// fullscreen triangle, no vertex buffer needed
// ----------------------------------------------------------------------------------------------------

inline constexpr const char* fullscreen_vertex = R"(
out vec2 v_uv;

void main()
{
   vec2 position = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
   v_uv = position;
   gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
)";

inline constexpr const char* sky_fragment = R"(
in vec2 v_uv;
uniform mat3 u_view_to_world;
uniform float u_tan_half_fov;
uniform float u_aspect;
out vec4 out_color;

void main()
{
   vec2 ndc = v_uv * 2.0 - 1.0;
   vec3 view_direction = normalize(vec3(ndc.x * u_tan_half_fov * u_aspect, ndc.y * u_tan_half_fov, -1.0));
   out_color = vec4(skyColor(normalize(u_view_to_world * view_direction)), 1.0);
}
)";

// ----------------------------------------------------------------------------------------------------
// solid geometry: floor and colliders
// ----------------------------------------------------------------------------------------------------

inline constexpr const char* mesh_vertex = R"(
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;

uniform mat4 u_view_projection;

out vec3 v_world_position;
out vec3 v_normal;
out vec3 v_color;

void main()
{
   gl_Position = u_view_projection * vec4(in_position, 1.0);
   v_world_position = in_position;
   v_normal = in_normal;
   v_color = in_color;
}
)";

inline constexpr const char* mesh_fragment = R"(
in vec3 v_world_position;
in vec3 v_normal;
in vec3 v_color;

uniform int u_checker;
uniform vec3 u_eye;

out vec4 out_color;

void main()
{
   vec3 normal = normalize(v_normal);
   vec3 color = v_color;
   if (u_checker == 1)
   {
      float cell = mod(floor(v_world_position.x * 10.0) + floor(v_world_position.z * 10.0), 2.0);
      color *= mix(0.78, 1.0, cell);
   }
   vec3 view_direction = normalize(u_eye - v_world_position);
   float diffuse = max(dot(normal, sun_direction), 0.0);
   float specular = pow(max(dot(normal, normalize(sun_direction + view_direction)), 0.0), 48.0);
   vec3 ambient = skyColor(normal) * 0.35;
   out_color = vec4(color * (ambient + 0.75 * diffuse) + vec3(0.15 * specular), 1.0);
}
)";

// ----------------------------------------------------------------------------------------------------
// particles as point sprites; shared by the sphere, depth and thickness passes
// ----------------------------------------------------------------------------------------------------

inline constexpr const char* particle_vertex = R"(
layout(location = 0) in vec3 in_position;
layout(location = 1) in float in_value;
layout(location = 2) in float in_density_ratio;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_point_scale;
uniform float u_radius;
uniform int u_particle_filter;     // 0 = all particles, 1 = bulk fluid only, 2 = spray only
uniform float u_spray_threshold;   // rho / rho0 below which a particle counts as spray

out float v_value;
out vec3 v_view_center;

void main()
{
   bool is_spray = in_density_ratio < u_spray_threshold;
   if ((u_particle_filter == 1 && is_spray) || (u_particle_filter == 2 && !is_spray))
   {
      // outside the clip volume and zero sized: dropped before rasterization
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      gl_PointSize = 0.0;
      v_value = 0.0;
      v_view_center = vec3(0.0);
      return;
   }

   vec4 view_position = u_view * vec4(in_position, 1.0);
   gl_Position = u_projection * view_position;
   gl_PointSize = u_point_scale * u_radius / -view_position.z;
   v_value = in_value;
   v_view_center = view_position.xyz;
}
)";

/// reconstructs the sphere normal (view space) from the sprite coordinate, discards outside the disc
inline constexpr const char* sphere_normal_function = R"(
vec3 sphereNormal()
{
   vec2 coordinate = gl_PointCoord * 2.0 - 1.0;
   coordinate.y = -coordinate.y;
   float radius_squared = dot(coordinate, coordinate);
   if (radius_squared > 1.0)
   {
      discard;
   }
   return vec3(coordinate, sqrt(1.0 - radius_squared));
}
)";

inline constexpr const char* sphere_fragment = R"(
in float v_value;
in vec3 v_view_center;

uniform mat4 u_projection;
uniform float u_radius;
uniform int u_color_mode;
uniform vec3 u_light_view;

out vec4 out_color;

vec3 waterRamp(float value)
{
   vec3 deep = vec3(0.02, 0.15, 0.55);
   vec3 mid = vec3(0.10, 0.55, 0.95);
   vec3 foam = vec3(0.92, 0.97, 1.00);
   return value < 0.5 ? mix(deep, mid, value * 2.0) : mix(mid, foam, value * 2.0 - 1.0);
}

void main()
{
   vec3 normal = sphereNormal();

   // per pixel depth so overlapping spheres intersect correctly
   vec4 clip_position = u_projection * vec4(v_view_center + normal * u_radius, 1.0);
   gl_FragDepth = clip_position.z / clip_position.w * 0.5 + 0.5;

   float diffuse = max(dot(normal, u_light_view), 0.0);
   float specular = pow(max(dot(reflect(-u_light_view, normal), vec3(0.0, 0.0, 1.0)), 0.0), 32.0);

   vec3 base_color = u_color_mode == 2   ? vec3(0.15, 0.45, 0.9)
                     : u_color_mode == 3 ? vec3(0.80, 0.88, 0.94)  // foam
                                         : waterRamp(clamp(v_value, 0.0, 1.0));
   out_color = vec4(base_color * (0.3 + 0.7 * diffuse) + vec3(specular * 0.35), 1.0);
}
)";

// ----------------------------------------------------------------------------------------------------
// screen space fluid: depth, thickness, bilateral smoothing, composite
// ----------------------------------------------------------------------------------------------------

/// writes the linear eye depth of the front most sphere surface
inline constexpr const char* fluid_depth_fragment = R"(
in vec3 v_view_center;

uniform mat4 u_projection;
uniform float u_radius;

out float out_depth;

void main()
{
   vec3 normal = sphereNormal();
   vec3 view_position = v_view_center + normal * u_radius;
   vec4 clip_position = u_projection * vec4(view_position, 1.0);
   gl_FragDepth = clip_position.z / clip_position.w * 0.5 + 0.5;
   out_depth = -view_position.z;
}
)";

/// additively accumulates how much fluid a view ray passes through
inline constexpr const char* fluid_thickness_fragment = R"(
uniform float u_radius;

out float out_thickness;

void main()
{
   vec3 normal = sphereNormal();
   out_thickness = 2.0 * normal.z * u_radius;
}
)";

/// separable bilateral filter: blurs depth within one surface, keeps silhouettes against farther fluid sharp
inline constexpr const char* fluid_smooth_fragment = R"(
in vec2 v_uv;

uniform sampler2D u_depth;
uniform vec2 u_direction;          // one texel along x or y
uniform float u_world_radius;      // filter radius in m
uniform float u_pixels_per_meter;  // at depth 1 m
uniform float u_depth_falloff;     // depth difference in m at which samples stop counting

out float out_depth;

// a fixed number of taps with a variable step, so the filter reaches any radius at the same cost.
// close to the camera a particle covers dozens of pixels; a filter capped at a few pixels left every particle
// visible as a bump, which turned into noisy normals and blotchy refraction
const int tap_count = 16;
const float max_radius = 256.0;

void main()
{
   float depth = texture(u_depth, v_uv).r;
   if (depth <= 0.0)
   {
      out_depth = 0.0;
      return;
   }

   float radius = clamp(u_world_radius * u_pixels_per_meter / depth, 1.0, max_radius);
   float step_size = max(1.0, radius / float(tap_count));

   // neighbors are compared against the local surface plane, not against the center depth: seen at a grazing
   // angle, depth changes a lot between pixels of the same flat surface, and a constant depth range then cut the
   // surface into one patch per particle. the slope is measured over a quarter of the radius, from the side that
   // changes less, so a silhouette next to the pixel does not tilt the plane
   float baseline = max(1.0, radius * 0.25);
   float forward_depth = texture(u_depth, v_uv + baseline * u_direction).r;
   float backward_depth = texture(u_depth, v_uv - baseline * u_direction).r;
   float forward_slope = forward_depth > 0.0 ? (forward_depth - depth) / baseline : 1e6;
   float backward_slope = backward_depth > 0.0 ? (depth - backward_depth) / baseline : 1e6;
   // both sides agree on a smooth surface: use the central slope, which does not flip from pixel to pixel;
   // only near an edge fall back to the calmer side
   float slope = abs(forward_slope - backward_slope) < 0.5 * u_depth_falloff / baseline ? 0.5 * (forward_slope + backward_slope)
               : abs(forward_slope) < abs(backward_slope)                                ? forward_slope
                                                                                          : backward_slope;
   slope = abs(slope) < 1e5 ? slope : 0.0;
   float sigma = radius * 0.5;
   float sum = 0.0;
   float weight_sum = 0.0;
   for (int tap = -tap_count; tap <= tap_count; ++tap)
   {
      float offset = float(tap) * step_size;
      if (abs(offset) > radius)
      {
         continue;
      }
      float sample_depth = texture(u_depth, v_uv + offset * u_direction).r;
      if (sample_depth <= 0.0)
      {
         continue;
      }
      float spatial = exp(-offset * offset / (2.0 * sigma * sigma));
      float difference = (sample_depth - (depth + slope * offset)) / u_depth_falloff;
      float range = exp(-difference * difference);
      sum += sample_depth * spatial * range;
      weight_sum += spatial * range;
   }
   out_depth = sum / weight_sum;
}
)";

inline constexpr const char* fluid_composite_fragment = R"(
in vec2 v_uv;

uniform sampler2D u_fluid_depth;
uniform sampler2D u_thickness;
uniform sampler2D u_scene_color;
uniform sampler2D u_scene_depth;

uniform mat4 u_projection;
uniform mat3 u_view_to_world;
uniform vec3 u_light_view;
uniform vec2 u_texel;
uniform float u_tan_half_fov;
uniform float u_aspect;
uniform float u_near;
uniform float u_far;
uniform float u_radius;

uniform vec3 u_fluid_color;
uniform vec3 u_absorption;
uniform float u_refraction_strength;

out vec4 out_color;

vec3 viewPosition(vec2 uv, float depth)
{
   vec2 ndc = uv * 2.0 - 1.0;
   return vec3(ndc.x * u_tan_half_fov * u_aspect * depth, ndc.y * u_tan_half_fov * depth, -depth);
}

float linearSceneDepth(float buffer_depth)
{
   float ndc_depth = buffer_depth * 2.0 - 1.0;
   return 2.0 * u_near * u_far / (u_far + u_near - ndc_depth * (u_far - u_near));
}

// picks the smaller of the forward and backward differences, which avoids smearing normals across silhouettes
vec3 surfaceDerivative(vec2 uv, vec3 position, vec2 step_uv)
{
   float forward_depth = texture(u_fluid_depth, uv + step_uv).r;
   float backward_depth = texture(u_fluid_depth, uv - step_uv).r;
   vec3 forward_difference = forward_depth > 0.0 ? viewPosition(uv + step_uv, forward_depth) - position : vec3(0.0);
   vec3 backward_difference = backward_depth > 0.0 ? position - viewPosition(uv - step_uv, backward_depth) : vec3(0.0);
   if (forward_depth <= 0.0 || (backward_depth > 0.0 && abs(backward_difference.z) < abs(forward_difference.z)))
   {
      return backward_difference;
   }
   return forward_difference;
}

void main()
{
   vec3 scene_color = texture(u_scene_color, v_uv).rgb;
   float scene_buffer_depth = texture(u_scene_depth, v_uv).r;
   float depth = texture(u_fluid_depth, v_uv).r;

   // no fluid here, or the fluid is hidden behind solid geometry
   if (depth <= 0.0 || depth > linearSceneDepth(scene_buffer_depth))
   {
      out_color = vec4(scene_color, 1.0);
      gl_FragDepth = scene_buffer_depth;
      return;
   }

   vec3 position = viewPosition(v_uv, depth);
   vec3 derivative_x = surfaceDerivative(v_uv, position, vec2(u_texel.x, 0.0));
   vec3 derivative_y = surfaceDerivative(v_uv, position, vec2(0.0, u_texel.y));
   vec3 normal = cross(derivative_x, derivative_y);
   normal = length(normal) > 1e-12 ? normalize(normal) : vec3(0.0, 0.0, 1.0);

   vec3 view_direction = normalize(-position);
   float thickness = texture(u_thickness, v_uv).r;

   // reflection of the sky, weighted by schlick's fresnel term (f0 of water is ~0.02)
   float facing = max(dot(normal, view_direction), 0.0);
   float fresnel = 0.02 + 0.98 * pow(1.0 - facing, 5.0);
   vec3 reflection = skyGradient(normalize(u_view_to_world * reflect(-view_direction, normal)));

   // refraction: look up the scene behind the fluid with an offset that grows with the normal and the thickness,
   // then absorb light along the path (beer-lambert), so shallow water is clear and deep water takes on its color
   vec2 refracted_uv = clamp(v_uv + normal.xy * u_refraction_strength * min(thickness, 0.5), vec2(0.0), vec2(1.0));
   vec3 refracted_scene = texture(u_scene_color, refracted_uv).rgb;
   // the minimum thickness keeps thin sheets and jets visible; physically they would be almost clear
   vec3 transmittance = exp(-u_absorption * (thickness + 4.0 * u_radius));
   vec3 refraction = refracted_scene * transmittance + u_fluid_color * (1.0 - transmittance);

   // a broad highlight: a very sharp one turns every small normal error into a bright fleck
   float specular = pow(max(dot(normal, normalize(u_light_view + view_direction)), 0.0), 120.0) * 0.6;

   vec3 color = mix(refraction, reflection, fresnel) + vec3(specular * 1.5);

   // soft edges where the fluid is only a single particle thick
   float coverage = smoothstep(0.0, u_radius * 0.75, thickness);
   out_color = vec4(mix(scene_color, color, coverage), 1.0);

   vec4 clip_position = u_projection * vec4(position, 1.0);
   gl_FragDepth = clip_position.z / clip_position.w * 0.5 + 0.5;
}
)";

// ----------------------------------------------------------------------------------------------------
// mesh surface: depth and thickness come from a closed triangle mesh instead of particle sprites
// ----------------------------------------------------------------------------------------------------

inline constexpr const char* surface_vertex = R"(
layout(location = 0) in vec3 in_position;

uniform mat4 u_view;
uniform mat4 u_projection;

out float v_depth;

void main()
{
   vec4 view_position = u_view * vec4(in_position, 1.0);
   gl_Position = u_projection * view_position;
   v_depth = -view_position.z;
}
)";

inline constexpr const char* surface_depth_fragment = R"(
in float v_depth;
out float out_depth;

void main()
{
   out_depth = v_depth;
}
)";

/// back faces add their depth, front faces subtract it: the sum is the water length along the ray, also through
/// several layers. depths are clipped to solid geometry, so water touching the floor does not go negative
inline constexpr const char* surface_thickness_fragment = R"(
in float v_depth;

uniform sampler2D u_scene_depth;
uniform vec2 u_texel;
uniform float u_near;
uniform float u_far;

out float out_thickness;

void main()
{
   float buffer_depth = texture(u_scene_depth, gl_FragCoord.xy * u_texel).r * 2.0 - 1.0;
   float scene_depth = 2.0 * u_near * u_far / (u_far + u_near - buffer_depth * (u_far - u_near));
   float depth = min(v_depth, scene_depth);
   out_thickness = gl_FrontFacing ? -depth : depth;
}
)";

// ----------------------------------------------------------------------------------------------------
// helper lines
// ----------------------------------------------------------------------------------------------------

inline constexpr const char* line_vertex = R"(
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

uniform mat4 u_view_projection;

out vec3 v_color;

void main()
{
   gl_Position = u_view_projection * vec4(in_position, 1.0);
   v_color = in_color;
}
)";

inline constexpr const char* line_fragment = R"(
in vec3 v_color;
out vec4 out_color;

void main()
{
   out_color = vec4(v_color, 1.0);
}
)";

}  // namespace shaders
