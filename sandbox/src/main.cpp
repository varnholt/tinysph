#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif
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

#if defined(__EMSCRIPTEN__)
EM_JS(int32_t, browserWidth, (), { return window.innerWidth; });
EM_JS(int32_t, browserHeight, (), { return window.innerHeight; });
#endif

/// \brief quits sdl when destroyed; the first member of App, so it goes after the window and the context.
struct SdlSession
{
   SdlSession() = default;
   SdlSession(const SdlSession&) = delete;
   SdlSession& operator=(const SdlSession&) = delete;
   ~SdlSession()
   {
      SDL_Quit();
   }
};

struct GlContextDeleter
{
   void operator()(SDL_GLContext context) const
   {
      SDL_GL_DestroyContext(context);
   }
};

/// \brief everything the frame loop needs. it lives on the heap, because in the browser main() returns before the
///        first frame and the browser calls frame() from then on.
struct App
{
   App() = default;
   App(const App&) = delete;
   App& operator=(const App&) = delete;
   ~App();

   /// \brief handles events, steps the simulation and draws one frame.
   /// \return false when the window was closed
   bool frame();

   SdlSession _sdl_session;
   std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> _window{nullptr, &SDL_DestroyWindow};
   std::unique_ptr<std::remove_pointer_t<SDL_GLContext>, GlContextDeleter> _gl_context;
   bool _imgui_initialized{false};
   Renderer _renderer;
   SandboxState _state;
   std::optional<tinysph::World> _world;
   OrbitCamera _camera;
   std::optional<tinysph::ForceFieldId> _mouse_field_id;  //!< the mouse tool is a force field that exists while the button is held
   std::chrono::steady_clock::time_point _previous_frame{std::chrono::steady_clock::now()};

   // --capture: every frame is written to _capture_directory without the ui, the app quits after _capture_frames
   std::filesystem::path _capture_directory;
   int32_t _capture_frames{0};
   int32_t _captured_frames{0};
   std::vector<uint8_t> _capture_pixels;

   [[nodiscard]] bool capturing() const
   {
      return !_capture_directory.empty();
   }

   /// \brief writes the current back buffer as a binary ppm, top row first.
   void captureFrame(int32_t width, int32_t height);
};

App::~App()
{
   if (_imgui_initialized)
   {
      _renderer.shutdown();
      ImGui_ImplOpenGL3_Shutdown();
      ImGui_ImplSDL3_Shutdown();
      ImGui::DestroyContext();
   }
}

}  // namespace

int main(int argument_count, char** arguments)
{
   if (!SDL_Init(SDL_INIT_VIDEO))
   {
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "SDL_Init failed: %s", SDL_GetError());
      return 1;
   }
   auto app = std::make_unique<App>();

   SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
#if defined(__EMSCRIPTEN__)
   // webgl 2
   SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
   SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
   SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#else
   SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
   SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
   SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
#endif
   SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
   SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
   SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
   SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

   const auto display_scale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
   const auto window_flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
#if defined(__EMSCRIPTEN__)
   // the canvas fills the browser window, frame() follows when it is resized
   const auto window_width = browserWidth();
   const auto window_height = browserHeight();
#else
   const auto window_width = static_cast<int32_t>(1600 * display_scale);
   const auto window_height = static_cast<int32_t>(900 * display_scale);
#endif
   // sdl objects are owned by unique_ptr, so every early return cleans up
   app->_window.reset(SDL_CreateWindow("tinysph sandbox", window_width, window_height, window_flags));
   if (app->_window == nullptr)
   {
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "SDL_CreateWindow failed: %s", SDL_GetError());
      return 1;
   }
   auto* const window = app->_window.get();

   app->_gl_context.reset(SDL_GL_CreateContext(window));
   if (app->_gl_context == nullptr)
   {
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "SDL_GL_CreateContext failed: %s", SDL_GetError());
      return 1;
   }
   SDL_GL_MakeCurrent(window, app->_gl_context.get());
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
   ImGui_ImplSDL3_InitForOpenGL(window, app->_gl_context.get());
#if defined(__EMSCRIPTEN__)
   ImGui_ImplOpenGL3_Init("#version 300 es");
   io.IniFilename = nullptr;  // no file system to keep window positions in
#else
   ImGui_ImplOpenGL3_Init("#version 330 core");
#endif
   app->_imgui_initialized = true;

   if (!app->_renderer.initialize())
   {
      return 1;
   }

   auto& state = app->_state;
   // --preset <index> starts with another scene, --spheres with sphere rendering, --zoom <wheel steps> moves the camera in, --tilt
   // <radians> changes its pitch, --gpu simulates on the gpu, --capture <directory> [--frames <count>] writes the frames
   // without the ui as ppm images and quits
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
      if (std::string_view{argument} == "--capture")
      {
         app->_capture_directory = value;
      }
      if (std::string_view{argument} == "--frames")
      {
         app->_capture_frames = std::max(1, std::atoi(value));
      }
   }
   if (app->capturing())
   {
      std::filesystem::create_directories(app->_capture_directory);
      if (app->_capture_frames == 0)
      {
         app->_capture_frames = 180;
      }
      // every frame advances exactly one frame time, never slow motion, so the capture is the same on any machine
      state._max_substeps = 100'000;
   }
   if (std::ranges::contains(
          argument_list | std::views::transform([](const char* argument) { return std::string_view{argument}; }), "--spheres"
       ))
   {
      state._render_settings._render_mode = RenderMode::Spheres;
   }
   if (std::ranges::contains(
          argument_list | std::views::transform([](const char* argument) { return std::string_view{argument}; }), "--mesh"
       ))
   {
      state._render_settings._render_mode = RenderMode::MeshSurface;
   }
   if (std::ranges::contains(
          argument_list | std::views::transform([](const char* argument) { return std::string_view{argument}; }), "--gpu"
       ))
   {
      state._backend = tinysph::Backend::Gpu;
   }
   loadPreset(state, state._preset_index);
   state._rebuild_requested = false;
   app->_world.emplace(buildWorld(state._scene, state._particle_spacing, state._backend, state._parameters));

   app->_camera.frame(state._scene._domain);
   app->_camera.zoom(zoom_steps);
   app->_camera.orbit(0.0f, tilt);

#if defined(__EMSCRIPTEN__)
   // the browser owns the loop: frame() runs once per animation frame, main() returns right away and the app lives on
   emscripten_set_main_loop_arg([](void* user_data) { static_cast<App*>(user_data)->frame(); }, app.release(), 0, false);
#else
   while (app->frame())
   {
   }
#endif
   return 0;
}

namespace
{

bool App::frame()
{
   auto& state = _state;
   auto& world = *_world;
   auto& camera = _camera;
   auto& mouse_field_id = _mouse_field_id;
   auto* const window = _window.get();
   const auto& io = ImGui::GetIO();

#if defined(__EMSCRIPTEN__)
   auto window_width = 0;
   auto window_height = 0;
   SDL_GetWindowSize(window, &window_width, &window_height);
   if (window_width != browserWidth() || window_height != browserHeight())
   {
      SDL_SetWindowSize(window, browserWidth(), browserHeight());
   }
#endif

   auto running = true;
   {
      SDL_Event event;
      while (SDL_PollEvent(&event))
      {
         // a capture only reacts to closing the window, so clicks into it cannot end up in the recording
         if (capturing() && event.type != SDL_EVENT_QUIT && event.type != SDL_EVENT_WINDOW_CLOSE_REQUESTED)
         {
            continue;
         }
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
         world = buildWorld(state._scene, state._particle_spacing, state._backend, state._parameters);
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
      if (!capturing() && state._mouse_tool._mode != MouseToolMode::Off && (mouse_buttons & SDL_BUTTON_LMASK) != 0 && !io.WantCaptureMouse)
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

      if (!capturing())
      {
         ImGui_ImplOpenGL3_NewFrame();
         ImGui_ImplSDL3_NewFrame();
         ImGui::NewFrame();
         drawSandboxUi(world, state);
         ImGui::Render();
      }

      auto framebuffer_width = 0;
      auto framebuffer_height = 0;
      SDL_GetWindowSizeInPixels(window, &framebuffer_width, &framebuffer_height);
      _renderer.render(world, state._scene, camera, framebuffer_width, framebuffer_height, state._render_settings, mouse_attractor);
      state._mesh_time = _renderer.meshBuildTime();
      if (capturing())
      {
         captureFrame(framebuffer_width, framebuffer_height);
         running = running && ++_captured_frames < _capture_frames;
      }
      else
      {
         ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      }
      SDL_GL_SwapWindow(window);

      const auto now = std::chrono::steady_clock::now();
      state._frame_time = std::chrono::duration_cast<tinysph::Milliseconds>(now - _previous_frame);
      _previous_frame = now;
   }
   return running;
}

void App::captureFrame(int32_t width, int32_t height)
{
   const auto row_size = static_cast<std::size_t>(width) * 3;
   _capture_pixels.resize(row_size * static_cast<std::size_t>(height));
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, _capture_pixels.data());

   const auto path = _capture_directory / std::format("frame_{:04}.ppm", _captured_frames);
   std::ofstream file{path, std::ios::binary};
   file << std::format("P6\n{} {}\n255\n", width, height);
   // opengl rows start at the bottom, ppm rows at the top
   for (auto row = height - 1; row >= 0; --row)
   {
      file.write(
         reinterpret_cast<const char*>(_capture_pixels.data() + row_size * static_cast<std::size_t>(row)),
         static_cast<std::streamsize>(row_size)
      );
   }
}

}  // namespace
