#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl3.h>

#include "camera.h"
#include "gl_functions.h"
#include "renderer.h"
#include "sandbox_ui.h"
#include "tinysph/tinysph.h"

namespace
{

/// \brief intersects the mouse ray with the plane through the camera target that faces the camera.
std::optional<tinysph::Vec3>
mouseOnTargetPlane(const OrbitCamera& camera, float mouse_x, float mouse_y, float window_width, float window_height)
{
   const auto ray = camera.rayThroughPoint(mouse_x, mouse_y, window_width, window_height);
   const auto plane_normal = camera.forward();
   const auto denominator = tinysph::dot(ray._direction, plane_normal);
   if (std::abs(denominator) < 1e-6f)
   {
      return std::nullopt;
   }
   const auto distance = tinysph::dot(camera.target() - ray._origin, plane_normal) / denominator;
   if (distance <= 0.0f)
   {
      return std::nullopt;
   }
   return ray._origin + ray._direction * distance;
}

}  // namespace

int main(int argument_count, char** arguments)
{
   if (!SDL_Init(SDL_INIT_VIDEO))
   {
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "SDL_Init failed: %s", SDL_GetError());
      return 1;
   }

   SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
   SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
   SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
   SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
   SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
   SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
   SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
   SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

   const auto display_scale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
   const auto window_flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
   auto* window = SDL_CreateWindow(
      "tinysph sandbox", static_cast<int32_t>(1600 * display_scale), static_cast<int32_t>(900 * display_scale), window_flags
   );
   if (window == nullptr)
   {
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "SDL_CreateWindow failed: %s", SDL_GetError());
      return 1;
   }

   auto gl_context = SDL_GL_CreateContext(window);
   if (gl_context == nullptr)
   {
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "SDL_GL_CreateContext failed: %s", SDL_GetError());
      return 1;
   }
   SDL_GL_MakeCurrent(window, gl_context);
   SDL_GL_SetSwapInterval(1);

   if (!gl::load())
   {
      return 1;
   }

   IMGUI_CHECKVERSION();
   ImGui::CreateContext();
   auto& io = ImGui::GetIO();
   io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
   ImGui::StyleColorsDark();
   auto& style = ImGui::GetStyle();
   style.ScaleAllSizes(display_scale);
   style.FontScaleDpi = display_scale;
   ImGui_ImplSDL3_InitForOpenGL(window, gl_context);
   ImGui_ImplOpenGL3_Init("#version 330 core");

   Renderer renderer;
   if (!renderer.initialize())
   {
      return 1;
   }

   SandboxState state;
   // --preset <index> starts with another scene, --spheres with sphere rendering, --zoom <wheel steps> moves the camera in, --tilt
   // <radians> changes its pitch
   const std::span<char*> argument_list{arguments, static_cast<std::size_t>(argument_count)};
   auto zoom_steps = 0.0f;
   auto tilt = 0.0f;  //!< added to the camera pitch in radians, negative looks flatter
   for (const auto [argument, value] : std::views::zip(argument_list, argument_list | std::views::drop(1)))
   {
      if (std::string_view{argument} == "--preset")
      {
         state._preset_index = std::clamp(std::atoi(value), 0, presetCount() - 1);
      }
      if (std::string_view{argument} == "--zoom")
      {
         zoom_steps = static_cast<float>(std::atof(value));
      }
      if (std::string_view{argument} == "--tilt")
      {
         tilt = static_cast<float>(std::atof(value));
      }
   }
   if (std::ranges::contains(
          argument_list | std::views::transform([](const char* argument) { return std::string_view{argument}; }), "--spheres"
       ))
   {
      state._render_settings._render_mode = RenderMode::Spheres;
   }
   loadPreset(state, state._preset_index);
   state._rebuild_requested = false;
   auto world = buildWorld(state._scene, state._particle_spacing, state._parameters);

   OrbitCamera camera;
   camera.frame(state._scene._domain);
   camera.zoom(zoom_steps);
   camera.orbit(0.0f, tilt);

   // the mouse tool is an ordinary force field that exists while the button is held
   std::optional<tinysph::ForceFieldId> mouse_field_id;

   auto running = true;
   auto previous_frame = std::chrono::steady_clock::now();
   while (running)
   {
      SDL_Event event;
      while (SDL_PollEvent(&event))
      {
         ImGui_ImplSDL3_ProcessEvent(&event);
         switch (event.type)
         {
            case SDL_EVENT_QUIT:
               running = false;
               break;
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
               if (event.window.windowID == SDL_GetWindowID(window))
               {
                  running = false;
               }
               break;
            case SDL_EVENT_KEY_DOWN:
               if (!io.WantCaptureKeyboard && !event.key.repeat)
               {
                  if (event.key.key == SDLK_SPACE)
                  {
                     state._running = !state._running;
                  }
                  else if (event.key.key == SDLK_R)
                  {
                     state._rebuild_requested = true;
                  }
               }
               break;
            case SDL_EVENT_MOUSE_MOTION:
               if (!io.WantCaptureMouse)
               {
                  if ((event.motion.state & SDL_BUTTON_RMASK) != 0)
                  {
                     camera.orbit(-event.motion.xrel * 0.005f, event.motion.yrel * 0.005f);
                  }
                  if ((event.motion.state & SDL_BUTTON_MMASK) != 0)
                  {
                     camera.pan(event.motion.xrel, event.motion.yrel);
                  }
               }
               break;
            case SDL_EVENT_MOUSE_WHEEL:
               if (!io.WantCaptureMouse)
               {
                  camera.zoom(event.wheel.y);
               }
               break;
            default:
               break;
         }
      }

      if (state._rebuild_requested)
      {
         state._rebuild_requested = false;
         world = buildWorld(state._scene, state._particle_spacing, state._parameters);
         mouse_field_id.reset();
      }
      if (state._reframe_requested)
      {
         state._reframe_requested = false;
         camera.frame(state._scene._domain);
      }
      world.setParameters(state._parameters);

      // mouse tool: an attractor that follows the cursor while the left button is held
      std::optional<tinysph::Attractor> mouse_attractor;
      auto mouse_x = 0.0f;
      auto mouse_y = 0.0f;
      const auto mouse_buttons = SDL_GetMouseState(&mouse_x, &mouse_y);
      if (state._mouse_tool._mode != MouseToolMode::Off && (mouse_buttons & SDL_BUTTON_LMASK) != 0 && !io.WantCaptureMouse)
      {
         auto window_width = 0;
         auto window_height = 0;
         SDL_GetWindowSize(window, &window_width, &window_height);
         const auto point =
            mouseOnTargetPlane(camera, mouse_x, mouse_y, static_cast<float>(window_width), static_cast<float>(window_height));
         if (point.has_value())
         {
            const auto sign = state._mouse_tool._mode == MouseToolMode::Pull ? 1.0f : -1.0f;
            mouse_attractor = tinysph::Attractor{*point, state._mouse_tool._strength * sign, state._mouse_tool._radius};
         }
      }
      if (mouse_attractor.has_value())
      {
         if (mouse_field_id.has_value())
         {
            world.setForceField(*mouse_field_id, *mouse_attractor);
         }
         else
         {
            mouse_field_id = world.addForceField(*mouse_attractor);
         }
      }
      else if (mouse_field_id.has_value())
      {
         world.removeForceField(*mouse_field_id);
         mouse_field_id.reset();
      }

      if (state._running)
      {
         state._statistics = world.step(frame_time * state._time_scale, state._max_substeps);
      }

      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplSDL3_NewFrame();
      ImGui::NewFrame();
      drawSandboxUi(world, state);
      ImGui::Render();

      auto framebuffer_width = 0;
      auto framebuffer_height = 0;
      SDL_GetWindowSizeInPixels(window, &framebuffer_width, &framebuffer_height);
      renderer.render(world, state._scene, camera, framebuffer_width, framebuffer_height, state._render_settings, mouse_attractor);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      SDL_GL_SwapWindow(window);

      const auto now = std::chrono::steady_clock::now();
      state._frame_time = std::chrono::duration_cast<tinysph::Milliseconds>(now - previous_frame);
      previous_frame = now;
   }

   renderer.shutdown();
   ImGui_ImplOpenGL3_Shutdown();
   ImGui_ImplSDL3_Shutdown();
   ImGui::DestroyContext();
   SDL_GL_DestroyContext(gl_context);
   SDL_DestroyWindow(window);
   SDL_Quit();
   return 0;
}
