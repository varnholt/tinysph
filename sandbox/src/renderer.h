#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "camera.h"
#include "sandbox_scene.h"
#include "tinysph/tinysph.h"

enum class RenderMode : int32_t
{
   Spheres = 0,       //!< every particle as a shaded sphere
   FluidSurface = 1,  //!< screen space fluid: smoothed hull with reflection and refraction
};

enum class ColorMode : int32_t
{
   Speed = 0,
   Density = 1,
   Flat = 2,
};

struct RenderSettings
{
   RenderMode _render_mode{RenderMode::FluidSurface};
   bool _show_helpers{true};  //!< container, emitters and fields as lines
   bool _show_floor{true};

   // sphere mode
   ColorMode _color_mode{ColorMode::Speed};
   float _radius_scale{1.0f};  //!< 1 = particles touch at rest spacing
   float _speed_range{3.0f};   //!< speed in m/s that maps to the top of the color ramp

   // fluid surface mode
   int32_t _smoothing_iterations{3};                        //!< bilateral filter passes; more = smoother hull
   float _smoothing_radius{2.5f};                           //!< filter radius in particle radii
   float _refraction_strength{0.15f};                       //!< screen space offset per meter of thickness
   std::array<float, 3> _fluid_color{0.03f, 0.22f, 0.38f};  //!< color deep water scatters towards
   std::array<float, 3> _absorption{3.5f, 1.1f, 0.55f};     //!< beer-lambert coefficients per meter, red is absorbed first
};

/// \brief draws sky, floor, colliders, the fluid and helper lines.
///        the fluid is either drawn as sphere impostors or as a screen space surface:
///        particle depth -> thickness -> bilateral smoothing -> normals -> fresnel blend of reflection and refraction.
///        all shaders stay within glsl es 3.00 so the same code runs on webgl2.
class Renderer
{
public:
   [[nodiscard]] bool initialize();
   void shutdown();

   void render(
      const tinysph::World& world,
      const SandboxScene& scene,
      const OrbitCamera& camera,
      int32_t framebuffer_width,
      int32_t framebuffer_height,
      const RenderSettings& settings,
      const std::optional<tinysph::Attractor>& mouse_tool
   );

private:
   struct RenderTarget
   {
      uint32_t _framebuffer{0};
      uint32_t _color_texture{0};
      uint32_t _depth_texture{0};
   };

   void resizeTargets(int32_t width, int32_t height);
   void destroyTargets();

   void uploadParticles(const tinysph::World& world, const RenderSettings& settings);
   void buildMeshes(const SandboxScene& scene, const RenderSettings& settings);
   void buildHelperLines(const SandboxScene& scene, const std::optional<tinysph::Attractor>& mouse_tool);

   void addTriangle(
      const tinysph::Vec3& a,
      const tinysph::Vec3& b,
      const tinysph::Vec3& c,
      const tinysph::Vec3& normal,
      const tinysph::Vec3& color
   );
   void addSolidBox(const tinysph::Vec3& center, const tinysph::Vec3& half_extents, const tinysph::Vec3& color);
   void addSolidSphere(const tinysph::Vec3& center, float radius, const tinysph::Vec3& color);

   void addLine(const tinysph::Vec3& from, const tinysph::Vec3& to, const tinysph::Vec3& color);
   void addBox(const tinysph::Aabb& box, const tinysph::Vec3& color);
   void addCircle(const tinysph::Vec3& center, const tinysph::Vec3& normal, float radius, const tinysph::Vec3& color);
   void addSphere(const tinysph::Vec3& center, float radius, const tinysph::Vec3& color);

   // programs
   uint32_t _sky_program{0};
   uint32_t _mesh_program{0};
   uint32_t _sphere_program{0};
   uint32_t _fluid_depth_program{0};
   uint32_t _fluid_thickness_program{0};
   uint32_t _fluid_smooth_program{0};
   uint32_t _fluid_composite_program{0};
   uint32_t _line_program{0};

   // geometry
   uint32_t _fullscreen_vertex_array{0};
   uint32_t _particle_vertex_array{0};
   uint32_t _particle_buffer{0};
   uint32_t _mesh_vertex_array{0};
   uint32_t _mesh_buffer{0};
   uint32_t _line_vertex_array{0};
   uint32_t _line_buffer{0};

   // offscreen targets for the fluid surface
   RenderTarget _scene_target;                   //!< sky, floor and colliders; refraction samples it
   RenderTarget _fluid_depth_target;             //!< linear eye depth of the front most particle surface
   RenderTarget _thickness_target;               //!< accumulated fluid thickness per pixel
   std::array<RenderTarget, 2> _smooth_targets;  //!< ping pong buffers for the separable filter
   int32_t _target_width{0};
   int32_t _target_height{0};

   std::vector<float> _particle_data;  //!< x, y, z, color value per particle
   std::vector<float> _mesh_data;      //!< x, y, z, nx, ny, nz, r, g, b per vertex
   std::vector<float> _line_data;      //!< x, y, z, r, g, b per vertex
   int32_t _floor_vertex_count{0};     //!< the first vertices in _mesh_data belong to the checkered floor
};
