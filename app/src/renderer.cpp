#include "renderer.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <numbers>
#include <string>
#include <variant>

#include <SDL3/SDL_log.h>

#include "gl_functions.h"
#include "shader_sources.h"

namespace
{

template <class... Callables>
struct Overloaded : Callables...
{
   using Callables::operator()...;
};

uint32_t compileShader(GLenum shader_type, std::initializer_list<const char*> source_parts)
{
   std::string source{shaders::header};
   for (const auto* part : source_parts)
   {
      source += part;
   }
   const auto* source_pointer = source.c_str();

   const auto shader = gl::CreateShader(shader_type);
   gl::ShaderSource(shader, 1, &source_pointer, nullptr);
   gl::CompileShader(shader);

   GLint compiled = GL_FALSE;
   gl::GetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
   if (compiled != GL_TRUE)
   {
      std::string log(4096, '\0');
      gl::GetShaderInfoLog(shader, static_cast<GLsizei>(log.size()), nullptr, log.data());
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "shader compilation failed: %s", log.c_str());
      gl::DeleteShader(shader);
      return 0;
   }
   return shader;
}

uint32_t linkProgram(std::initializer_list<const char*> vertex_parts, std::initializer_list<const char*> fragment_parts)
{
   const auto vertex_shader = compileShader(GL_VERTEX_SHADER, vertex_parts);
   const auto fragment_shader = compileShader(GL_FRAGMENT_SHADER, fragment_parts);
   if (vertex_shader == 0 || fragment_shader == 0)
   {
      return 0;
   }

   const auto program = gl::CreateProgram();
   gl::AttachShader(program, vertex_shader);
   gl::AttachShader(program, fragment_shader);
   gl::LinkProgram(program);
   gl::DeleteShader(vertex_shader);
   gl::DeleteShader(fragment_shader);

   GLint linked = GL_FALSE;
   gl::GetProgramiv(program, GL_LINK_STATUS, &linked);
   if (linked != GL_TRUE)
   {
      std::string log(4096, '\0');
      gl::GetProgramInfoLog(program, static_cast<GLsizei>(log.size()), nullptr, log.data());
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "program link failed: %s", log.c_str());
      gl::DeleteProgram(program);
      return 0;
   }
   return program;
}

void setupVertexArray(uint32_t& vertex_array, uint32_t& buffer, std::initializer_list<int32_t> attribute_sizes)
{
   gl::GenVertexArrays(1, &vertex_array);
   gl::GenBuffers(1, &buffer);
   gl::BindVertexArray(vertex_array);
   gl::BindBuffer(GL_ARRAY_BUFFER, buffer);

   auto stride = 0;
   for (const auto attribute_size : attribute_sizes)
   {
      stride += attribute_size;
   }

   auto attribute_index = 0u;
   auto offset = 0;
   for (const auto attribute_size : attribute_sizes)
   {
      gl::EnableVertexAttribArray(attribute_index);
      gl::VertexAttribPointer(
         attribute_index,
         attribute_size,
         GL_FLOAT,
         GL_FALSE,
         stride * static_cast<GLsizei>(sizeof(float)),
         reinterpret_cast<const void*>(static_cast<std::size_t>(offset) * sizeof(float))
      );
      ++attribute_index;
      offset += attribute_size;
   }
   gl::BindVertexArray(0);
}

uint32_t createTexture(int32_t width, int32_t height, GLint internal_format, GLenum format, GLenum type, GLint filter)
{
   uint32_t texture = 0;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, type, nullptr);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glBindTexture(GL_TEXTURE_2D, 0);
   return texture;
}

void bindTexture(uint32_t program, const char* sampler_name, int32_t unit, uint32_t texture)
{
   gl::ActiveTexture(static_cast<GLenum>(GL_TEXTURE0 + unit));
   glBindTexture(GL_TEXTURE_2D, texture);
   gl::Uniform1i(gl::GetUniformLocation(program, sampler_name), unit);
}

void setUniform(uint32_t program, const char* name, float value)
{
   gl::Uniform1f(gl::GetUniformLocation(program, name), value);
}

void setUniform(uint32_t program, const char* name, const sph::Vec3& value)
{
   gl::Uniform3f(gl::GetUniformLocation(program, name), value.x, value.y, value.z);
}

void setUniform(uint32_t program, const char* name, const std::array<float, 3>& value)
{
   gl::Uniform3f(gl::GetUniformLocation(program, name), value[0], value[1], value[2]);
}

void setUniform(uint32_t program, const char* name, const Mat4& value)
{
   gl::UniformMatrix4fv(gl::GetUniformLocation(program, name), 1, GL_FALSE, value.data());
}

void setUniform(uint32_t program, const char* name, const std::array<float, 9>& value)
{
   gl::UniformMatrix3fv(gl::GetUniformLocation(program, name), 1, GL_FALSE, value.data());
}

}  // namespace

bool Renderer::initialize()
{
   _sky_program = linkProgram({shaders::fullscreen_vertex}, {shaders::sky_function, shaders::sky_fragment});
   _mesh_program = linkProgram({shaders::mesh_vertex}, {shaders::sky_function, shaders::mesh_fragment});
   _sphere_program = linkProgram({shaders::particle_vertex}, {shaders::sphere_normal_function, shaders::sphere_fragment});
   _fluid_depth_program = linkProgram({shaders::particle_vertex}, {shaders::sphere_normal_function, shaders::fluid_depth_fragment});
   _fluid_thickness_program = linkProgram({shaders::particle_vertex}, {shaders::sphere_normal_function, shaders::fluid_thickness_fragment});
   _fluid_smooth_program = linkProgram({shaders::fullscreen_vertex}, {shaders::fluid_smooth_fragment});
   _fluid_composite_program = linkProgram({shaders::fullscreen_vertex}, {shaders::sky_function, shaders::fluid_composite_fragment});
   _line_program = linkProgram({shaders::line_vertex}, {shaders::line_fragment});

   const auto programs = {
      _sky_program,
      _mesh_program,
      _sphere_program,
      _fluid_depth_program,
      _fluid_thickness_program,
      _fluid_smooth_program,
      _fluid_composite_program,
      _line_program
   };
   if (std::ranges::any_of(programs, [](uint32_t program) { return program == 0; }))
   {
      return false;
   }

   gl::GenVertexArrays(1, &_fullscreen_vertex_array);
   setupVertexArray(_particle_vertex_array, _particle_buffer, {3, 1});
   setupVertexArray(_mesh_vertex_array, _mesh_buffer, {3, 3, 3});
   setupVertexArray(_line_vertex_array, _line_buffer, {3, 3});
   return true;
}

void Renderer::shutdown()
{
   destroyTargets();
   for (auto* buffer : {&_particle_buffer, &_mesh_buffer, &_line_buffer})
   {
      gl::DeleteBuffers(1, buffer);
   }
   for (auto* vertex_array : {&_fullscreen_vertex_array, &_particle_vertex_array, &_mesh_vertex_array, &_line_vertex_array})
   {
      gl::DeleteVertexArrays(1, vertex_array);
   }
   for (const auto program :
        {_sky_program,
         _mesh_program,
         _sphere_program,
         _fluid_depth_program,
         _fluid_thickness_program,
         _fluid_smooth_program,
         _fluid_composite_program,
         _line_program})
   {
      gl::DeleteProgram(program);
   }
}

void Renderer::destroyTargets()
{
   for (auto* target : {&_scene_target, &_fluid_depth_target, &_thickness_target, &_smooth_targets[0], &_smooth_targets[1]})
   {
      if (target->_framebuffer != 0)
      {
         gl::DeleteFramebuffers(1, &target->_framebuffer);
      }
      for (auto* texture : {&target->_color_texture, &target->_depth_texture})
      {
         if (*texture != 0)
         {
            glDeleteTextures(1, texture);
         }
      }
      *target = {};
   }
   _target_width = 0;
   _target_height = 0;
}

void Renderer::resizeTargets(int32_t width, int32_t height)
{
   if (width == _target_width && height == _target_height)
   {
      return;
   }
   destroyTargets();

   const auto create_target = [&](RenderTarget& target, GLint internal_format, GLenum format, GLenum type, GLint filter, bool with_depth)
   {
      target._color_texture = createTexture(width, height, internal_format, format, type, filter);
      if (with_depth)
      {
         target._depth_texture = createTexture(width, height, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, GL_NEAREST);
      }
      gl::GenFramebuffers(1, &target._framebuffer);
      gl::BindFramebuffer(GL_FRAMEBUFFER, target._framebuffer);
      gl::FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, target._color_texture, 0);
      if (with_depth)
      {
         gl::FramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, target._depth_texture, 0);
      }
      if (gl::CheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      {
         SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "framebuffer incomplete");
      }
   };

   // float textures are sampled with nearest filtering; linear filtering of float textures is optional in gles 3.0
   create_target(_scene_target, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, GL_LINEAR, true);
   create_target(_fluid_depth_target, GL_R32F, GL_RED, GL_FLOAT, GL_NEAREST, true);
   create_target(_thickness_target, GL_R16F, GL_RED, GL_FLOAT, GL_NEAREST, false);

   // the thickness pass shares the scene depth (read only) so fluid behind solid geometry is not counted
   gl::BindFramebuffer(GL_FRAMEBUFFER, _thickness_target._framebuffer);
   gl::FramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, _scene_target._depth_texture, 0);
   if (gl::CheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
   {
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "thickness framebuffer incomplete");
   }
   create_target(_smooth_targets[0], GL_R32F, GL_RED, GL_FLOAT, GL_NEAREST, false);
   create_target(_smooth_targets[1], GL_R32F, GL_RED, GL_FLOAT, GL_NEAREST, false);
   gl::BindFramebuffer(GL_FRAMEBUFFER, 0);

   _target_width = width;
   _target_height = height;
}

void Renderer::render(
   const sph::Solver& solver,
   const OrbitCamera& camera,
   int32_t framebuffer_width,
   int32_t framebuffer_height,
   const RenderSettings& settings,
   const std::optional<sph::Attractor>& mouse_tool
)
{
   if (framebuffer_width <= 0 || framebuffer_height <= 0)
   {
      return;
   }

   const auto fluid_surface = settings._render_mode == RenderMode::FluidSurface;
   if (fluid_surface)
   {
      resizeTargets(framebuffer_width, framebuffer_height);
   }

   // per frame constants
   const auto aspect_ratio = static_cast<float>(framebuffer_width) / static_cast<float>(framebuffer_height);
   const auto view = camera.viewMatrix();
   const auto projection = camera.projectionMatrix(aspect_ratio);
   const auto view_projection = projection * view;
   const auto tan_half_fov = std::tan(camera.fieldOfViewY() * 0.5f);
   const auto particle_radius = solver.settings()._particle_spacing * 0.5f;

   // the view matrix rotation maps world to view; its transpose maps view to world
   std::array<float, 9> view_to_world{};
   for (auto column = 0; column < 3; ++column)
   {
      for (auto row = 0; row < 3; ++row)
      {
         view_to_world[static_cast<std::size_t>(column * 3 + row)] = view.at(row, column);
      }
   }
   const sph::Vec3 sun_world{0.3904f, 0.7807f, 0.4880f};
   const sph::Vec3 light_view{
      view.at(0, 0) * sun_world.x + view.at(1, 0) * sun_world.y + view.at(2, 0) * sun_world.z,
      view.at(0, 1) * sun_world.x + view.at(1, 1) * sun_world.y + view.at(2, 1) * sun_world.z,
      view.at(0, 2) * sun_world.x + view.at(1, 2) * sun_world.y + view.at(2, 2) * sun_world.z,
   };

   uploadParticles(solver, settings);
   buildMeshes(solver, settings);
   const auto particle_count = static_cast<GLsizei>(_particle_data.size() / 4);

   const auto draw_fullscreen = [this]()
   {
      gl::BindVertexArray(_fullscreen_vertex_array);
      glDrawArrays(GL_TRIANGLES, 0, 3);
   };

   const auto set_particle_uniforms = [&](uint32_t program, float radius)
   {
      gl::UseProgram(program);
      setUniform(program, "u_view", view);
      setUniform(program, "u_projection", projection);
      setUniform(program, "u_point_scale", projection.at(1, 1) * static_cast<float>(framebuffer_height));
      setUniform(program, "u_radius", radius);
      gl::BindVertexArray(_particle_vertex_array);
   };

   // 1) background: sky, floor, colliders. into the scene target when the fluid has to refract it.
   gl::BindFramebuffer(GL_FRAMEBUFFER, fluid_surface ? _scene_target._framebuffer : 0);
   glViewport(0, 0, framebuffer_width, framebuffer_height);
   glDepthMask(GL_TRUE);
   glClear(GL_DEPTH_BUFFER_BIT);

   glDisable(GL_DEPTH_TEST);
   glDepthMask(GL_FALSE);
   gl::UseProgram(_sky_program);
   setUniform(_sky_program, "u_view_to_world", view_to_world);
   setUniform(_sky_program, "u_tan_half_fov", tan_half_fov);
   setUniform(_sky_program, "u_aspect", aspect_ratio);
   draw_fullscreen();

   glEnable(GL_DEPTH_TEST);
   glDepthMask(GL_TRUE);
   if (!_mesh_data.empty())
   {
      gl::BindBuffer(GL_ARRAY_BUFFER, _mesh_buffer);
      gl::BufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(_mesh_data.size() * sizeof(float)), _mesh_data.data(), GL_STREAM_DRAW);
      gl::UseProgram(_mesh_program);
      setUniform(_mesh_program, "u_view_projection", view_projection);
      setUniform(_mesh_program, "u_eye", camera.eye());
      gl::BindVertexArray(_mesh_vertex_array);

      const auto vertex_count = static_cast<GLsizei>(_mesh_data.size() / 9);
      gl::Uniform1i(gl::GetUniformLocation(_mesh_program, "u_checker"), 1);
      glDrawArrays(GL_TRIANGLES, 0, _floor_vertex_count);
      gl::Uniform1i(gl::GetUniformLocation(_mesh_program, "u_checker"), 0);
      glDrawArrays(GL_TRIANGLES, _floor_vertex_count, vertex_count - _floor_vertex_count);
   }

   glEnable(GL_PROGRAM_POINT_SIZE);
   if (!fluid_surface)
   {
      // 2a) spheres straight into the window
      if (particle_count > 0)
      {
         set_particle_uniforms(_sphere_program, particle_radius * settings._radius_scale);
         gl::Uniform1i(gl::GetUniformLocation(_sphere_program, "u_color_mode"), static_cast<int32_t>(settings._color_mode));
         setUniform(_sphere_program, "u_light_view", light_view);
         glDrawArrays(GL_POINTS, 0, particle_count);
      }
   }
   else
   {
      // 2b) fluid depth: front most sphere surface, linear eye depth, 0 = empty
      gl::BindFramebuffer(GL_FRAMEBUFFER, _fluid_depth_target._framebuffer);
      glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      if (particle_count > 0)
      {
         set_particle_uniforms(_fluid_depth_program, particle_radius);
         glDrawArrays(GL_POINTS, 0, particle_count);
      }

      // 3) thickness: additive, so it measures all fluid along the ray. depth tested (without writing) against the
      //    scene depth, otherwise water hidden behind a collider counts as well and thin films in front of it turn dark.
      gl::BindFramebuffer(GL_FRAMEBUFFER, _thickness_target._framebuffer);
      glClear(GL_COLOR_BUFFER_BIT);
      glEnable(GL_DEPTH_TEST);
      glDepthMask(GL_FALSE);
      glEnable(GL_BLEND);
      glBlendFunc(GL_ONE, GL_ONE);
      if (particle_count > 0)
      {
         set_particle_uniforms(_fluid_thickness_program, particle_radius);
         glDrawArrays(GL_POINTS, 0, particle_count);
      }
      glDisable(GL_BLEND);

      // 4) bilateral smoothing, separable, ping pong; this turns the spheres into one hull
      gl::UseProgram(_fluid_smooth_program);
      setUniform(_fluid_smooth_program, "u_world_radius", particle_radius * settings._smoothing_radius);
      setUniform(_fluid_smooth_program, "u_pixels_per_meter", projection.at(1, 1) * static_cast<float>(framebuffer_height) * 0.5f);
      setUniform(_fluid_smooth_program, "u_depth_falloff", particle_radius * 2.0f);
      const auto texel_x = 1.0f / static_cast<float>(framebuffer_width);
      const auto texel_y = 1.0f / static_cast<float>(framebuffer_height);
      auto source_texture = _fluid_depth_target._color_texture;
      for (auto iteration = 0; iteration < settings._smoothing_iterations; ++iteration)
      {
         gl::BindFramebuffer(GL_FRAMEBUFFER, _smooth_targets[0]._framebuffer);
         bindTexture(_fluid_smooth_program, "u_depth", 0, source_texture);
         gl::Uniform2f(gl::GetUniformLocation(_fluid_smooth_program, "u_direction"), texel_x, 0.0f);
         draw_fullscreen();

         gl::BindFramebuffer(GL_FRAMEBUFFER, _smooth_targets[1]._framebuffer);
         bindTexture(_fluid_smooth_program, "u_depth", 0, _smooth_targets[0]._color_texture);
         gl::Uniform2f(gl::GetUniformLocation(_fluid_smooth_program, "u_direction"), 0.0f, texel_y);
         draw_fullscreen();

         source_texture = _smooth_targets[1]._color_texture;
      }

      // 5) composite into the window: normals from the smoothed depth, fresnel, reflection, refraction, absorption
      gl::BindFramebuffer(GL_FRAMEBUFFER, 0);
      glViewport(0, 0, framebuffer_width, framebuffer_height);
      glClear(GL_DEPTH_BUFFER_BIT);
      glEnable(GL_DEPTH_TEST);
      glDepthMask(GL_TRUE);
      glDepthFunc(GL_ALWAYS);

      const auto program = _fluid_composite_program;
      gl::UseProgram(program);
      bindTexture(program, "u_fluid_depth", 0, source_texture);
      bindTexture(program, "u_thickness", 1, _thickness_target._color_texture);
      bindTexture(program, "u_scene_color", 2, _scene_target._color_texture);
      bindTexture(program, "u_scene_depth", 3, _scene_target._depth_texture);
      setUniform(program, "u_projection", projection);
      setUniform(program, "u_view_to_world", view_to_world);
      setUniform(program, "u_light_view", light_view);
      gl::Uniform2f(gl::GetUniformLocation(program, "u_texel"), texel_x, texel_y);
      setUniform(program, "u_tan_half_fov", tan_half_fov);
      setUniform(program, "u_aspect", aspect_ratio);
      setUniform(program, "u_near", OrbitCamera::near_plane);
      setUniform(program, "u_far", OrbitCamera::far_plane);
      setUniform(program, "u_radius", particle_radius);
      setUniform(program, "u_fluid_color", settings._fluid_color);
      setUniform(program, "u_absorption", settings._absorption);
      setUniform(program, "u_refraction_strength", settings._refraction_strength);
      draw_fullscreen();

      glDepthFunc(GL_LESS);
      gl::ActiveTexture(GL_TEXTURE0);
   }

   // 6) helper lines on top, depth tested against everything above
   if (settings._show_helpers)
   {
      buildHelperLines(solver, mouse_tool);
      gl::BindBuffer(GL_ARRAY_BUFFER, _line_buffer);
      gl::BufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(_line_data.size() * sizeof(float)), _line_data.data(), GL_STREAM_DRAW);
      gl::UseProgram(_line_program);
      setUniform(_line_program, "u_view_projection", view_projection);
      gl::BindVertexArray(_line_vertex_array);
      glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(_line_data.size() / 6));
   }

   gl::BindVertexArray(0);
   gl::UseProgram(0);
   glDisable(GL_DEPTH_TEST);
}

void Renderer::uploadParticles(const sph::Solver& solver, const RenderSettings& settings)
{
   const auto positions = solver.positions();
   const auto velocities = solver.velocities();
   const auto densities = solver.densities();
   const auto rest_density = solver.settings()._rest_density;

   _particle_data.resize(positions.size() * 4);
   for (std::size_t particle_index = 0; particle_index < positions.size(); ++particle_index)
   {
      auto color_value = 0.0f;
      switch (settings._color_mode)
      {
         case ColorMode::Speed:
            color_value = sph::length(velocities[particle_index]) / settings._speed_range;
            break;
         case ColorMode::Density:
            // 0.9 rho0 -> 0, 1.1 rho0 -> 1
            color_value = (densities[particle_index] / rest_density - 0.9f) * 5.0f;
            break;
         case ColorMode::Flat:
            break;
      }

      const auto& position = positions[particle_index];
      auto* target = &_particle_data[particle_index * 4];
      target[0] = position.x;
      target[1] = position.y;
      target[2] = position.z;
      target[3] = color_value;
   }

   gl::BindBuffer(GL_ARRAY_BUFFER, _particle_buffer);
   gl::BufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(_particle_data.size() * sizeof(float)), _particle_data.data(), GL_STREAM_DRAW);
}

void Renderer::buildMeshes(const sph::Solver& solver, const RenderSettings& settings)
{
   const auto& scene = solver.scene();
   _mesh_data.clear();

   if (settings._show_floor)
   {
      const auto& container = scene._container;
      constexpr auto margin = 2.0f;
      const auto height = container._min.y - 0.002f;
      const sph::Vec3 corner_a{container._min.x - margin, height, container._min.z - margin};
      const sph::Vec3 corner_b{container._max.x + margin, height, container._min.z - margin};
      const sph::Vec3 corner_c{container._max.x + margin, height, container._max.z + margin};
      const sph::Vec3 corner_d{container._min.x - margin, height, container._max.z + margin};
      const sph::Vec3 up{0.0f, 1.0f, 0.0f};
      const sph::Vec3 floor_color{0.74f, 0.72f, 0.68f};
      addTriangle(corner_a, corner_c, corner_b, up, floor_color);
      addTriangle(corner_a, corner_d, corner_c, up, floor_color);
   }
   _floor_vertex_count = static_cast<int32_t>(_mesh_data.size() / 9);

   const sph::Vec3 collider_color{0.86f, 0.52f, 0.26f};
   for (const auto& collider : scene._colliders)
   {
      std::visit(
         Overloaded{
            [&](const sph::SphereCollider& sphere) { addSolidSphere(sphere._center, sphere._radius, collider_color); },
            [&](const sph::BoxCollider& box) { addSolidBox(box._center, box._half_extents, collider_color); },
         },
         collider
      );
   }
}

void Renderer::addTriangle(const sph::Vec3& a, const sph::Vec3& b, const sph::Vec3& c, const sph::Vec3& normal, const sph::Vec3& color)
{
   for (const auto* vertex : {&a, &b, &c})
   {
      _mesh_data.insert(_mesh_data.end(), {vertex->x, vertex->y, vertex->z, normal.x, normal.y, normal.z, color.x, color.y, color.z});
   }
}

void Renderer::addSolidBox(const sph::Vec3& center, const sph::Vec3& half_extents, const sph::Vec3& color)
{
   // one face per axis and sign: normal n, two tangents spanning the face
   const std::array<std::array<sph::Vec3, 3>, 3> axes{{
      {{{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}}},
      {{{0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}}},
      {{{0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}},
   }};
   const auto scale = [&](const sph::Vec3& direction)
   { return sph::Vec3{direction.x * half_extents.x, direction.y * half_extents.y, direction.z * half_extents.z}; };

   for (const auto& axis : axes)
   {
      for (const auto sign : {1.0f, -1.0f})
      {
         const auto normal = axis[0] * sign;
         const auto face_center = center + scale(normal);
         const auto tangent = scale(axis[1]);
         const auto bitangent = scale(axis[2]) * sign;
         const auto corner_a = face_center - tangent - bitangent;
         const auto corner_b = face_center + tangent - bitangent;
         const auto corner_c = face_center + tangent + bitangent;
         const auto corner_d = face_center - tangent + bitangent;
         addTriangle(corner_a, corner_b, corner_c, normal, color);
         addTriangle(corner_a, corner_c, corner_d, normal, color);
      }
   }
}

void Renderer::addSolidSphere(const sph::Vec3& center, float radius, const sph::Vec3& color)
{
   constexpr auto stacks = 16;
   constexpr auto slices = 32;
   constexpr auto pi = std::numbers::pi_v<float>;

   const auto point_at = [&](int32_t stack, int32_t slice)
   {
      const auto polar = pi * static_cast<float>(stack) / stacks;
      const auto azimuth = 2.0f * pi * static_cast<float>(slice) / slices;
      return sph::Vec3{std::sin(polar) * std::cos(azimuth), std::cos(polar), std::sin(polar) * std::sin(azimuth)};
   };

   for (auto stack = 0; stack < stacks; ++stack)
   {
      for (auto slice = 0; slice < slices; ++slice)
      {
         const auto normal_a = point_at(stack, slice);
         const auto normal_b = point_at(stack + 1, slice);
         const auto normal_c = point_at(stack + 1, slice + 1);
         const auto normal_d = point_at(stack, slice + 1);
         for (const auto& triangle : {std::array{normal_a, normal_b, normal_c}, std::array{normal_a, normal_c, normal_d}})
         {
            for (const auto& normal : triangle)
            {
               const auto position = center + normal * radius;
               _mesh_data.insert(
                  _mesh_data.end(), {position.x, position.y, position.z, normal.x, normal.y, normal.z, color.x, color.y, color.z}
               );
            }
         }
      }
   }
}

void Renderer::buildHelperLines(const sph::Solver& solver, const std::optional<sph::Attractor>& mouse_tool)
{
   const sph::Vec3 container_color{0.85f, 0.85f, 0.9f};
   const sph::Vec3 emitter_color{0.3f, 0.95f, 0.4f};
   const sph::Vec3 field_color{0.85f, 0.35f, 0.9f};
   const sph::Vec3 tool_color{1.0f, 1.0f, 0.3f};

   _line_data.clear();
   const auto& scene = solver.scene();
   addBox(scene._container, container_color);

   for (const auto& emitter : scene._emitters)
   {
      const auto direction = sph::normalize(emitter._velocity);
      addCircle(emitter._position, direction, emitter._radius, emitter_color);
      addLine(emitter._position, emitter._position + direction * 0.1f, emitter_color);
   }

   for (const auto& field : scene._fields)
   {
      std::visit(
         Overloaded{
            [&](const sph::Attractor& attractor) { addSphere(attractor._position, attractor._radius, field_color); },
            [&](const sph::Vortex& vortex)
            {
               const auto axis = sph::normalize(vortex._axis);
               addCircle(vortex._position, axis, vortex._radius, field_color);
               addLine(vortex._position - axis * vortex._radius, vortex._position + axis * vortex._radius, field_color);
            },
            [&](const sph::Wind& wind) { addBox(wind._region, field_color); },
         },
         field
      );
   }

   if (mouse_tool.has_value())
   {
      addSphere(mouse_tool->_position, mouse_tool->_radius, tool_color);
   }
}

void Renderer::addLine(const sph::Vec3& from, const sph::Vec3& to, const sph::Vec3& color)
{
   _line_data.insert(_line_data.end(), {from.x, from.y, from.z, color.x, color.y, color.z});
   _line_data.insert(_line_data.end(), {to.x, to.y, to.z, color.x, color.y, color.z});
}

void Renderer::addBox(const sph::Aabb& box, const sph::Vec3& color)
{
   const auto& low = box._min;
   const auto& high = box._max;
   const std::array<sph::Vec3, 8> corners{{
      {low.x, low.y, low.z},
      {high.x, low.y, low.z},
      {high.x, high.y, low.z},
      {low.x, high.y, low.z},
      {low.x, low.y, high.z},
      {high.x, low.y, high.z},
      {high.x, high.y, high.z},
      {low.x, high.y, high.z},
   }};
   constexpr std::array<std::array<int32_t, 2>, 12> edges{{
      {0, 1},
      {1, 2},
      {2, 3},
      {3, 0},
      {4, 5},
      {5, 6},
      {6, 7},
      {7, 4},
      {0, 4},
      {1, 5},
      {2, 6},
      {3, 7},
   }};
   for (const auto& edge : edges)
   {
      addLine(corners[static_cast<std::size_t>(edge[0])], corners[static_cast<std::size_t>(edge[1])], color);
   }
}

void Renderer::addCircle(const sph::Vec3& center, const sph::Vec3& normal, float radius, const sph::Vec3& color)
{
   const auto unit_normal = sph::normalize(normal);
   const auto helper = std::abs(unit_normal.y) < 0.9f ? sph::Vec3{0.0f, 1.0f, 0.0f} : sph::Vec3{1.0f, 0.0f, 0.0f};
   const auto tangent = sph::normalize(sph::cross(unit_normal, helper));
   const auto bitangent = sph::cross(unit_normal, tangent);

   constexpr auto segment_count = 32;
   for (auto segment = 0; segment < segment_count; ++segment)
   {
      const auto angle_from = 2.0f * std::numbers::pi_v<float> * static_cast<float>(segment) / segment_count;
      const auto angle_to = 2.0f * std::numbers::pi_v<float> * static_cast<float>(segment + 1) / segment_count;
      addLine(
         center + (tangent * std::cos(angle_from) + bitangent * std::sin(angle_from)) * radius,
         center + (tangent * std::cos(angle_to) + bitangent * std::sin(angle_to)) * radius,
         color
      );
   }
}

void Renderer::addSphere(const sph::Vec3& center, float radius, const sph::Vec3& color)
{
   addCircle(center, {1.0f, 0.0f, 0.0f}, radius, color);
   addCircle(center, {0.0f, 1.0f, 0.0f}, radius, color);
   addCircle(center, {0.0f, 0.0f, 1.0f}, radius, color);
}
