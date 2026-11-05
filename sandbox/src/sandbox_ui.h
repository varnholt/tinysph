#pragma once

#include <cstdint>

#include "renderer.h"
#include "sandbox_scene.h"
#include "tinysph/tinysph.h"

/// \brief simulated time per rendered frame at time scale 1.
inline constexpr tinysph::Seconds frame_time{1.0f / 60.0f};

enum class MouseToolMode : int32_t
{
   Off = 0,
   Pull = 1,
   Push = 2,
};

/// \brief while the left mouse button is held in the viewport, an attractor follows the cursor.
struct MouseTool
{
   MouseToolMode _mode{MouseToolMode::Pull};
   float _strength{60.0f};  //!< acceleration at the center in m/s^2
   float _radius{0.15f};    //!< in m
};

/// \brief editable values for the "add fluid" panel.
struct FluidDraft
{
   tinysph::Box _box{{0.7f, 0.6f, 0.3f}, {0.1f, 0.1f, 0.1f}};
   tinysph::Sphere _sphere{{0.8f, 0.7f, 0.3f}, 0.1f};
   tinysph::Vec3 _velocity{0.0f, 0.0f, 0.0f};
   bool _keep_in_scene{true};  //!< also add it to the scene so a reset recreates it
};

struct SandboxState
{
   SandboxScene _scene;
   tinysph::FluidParameters _parameters;              //!< pushed to the world every frame
   float _particle_spacing{0.025f};                   //!< applied when the world is rebuilt
   tinysph::Backend _backend{tinysph::Backend::Cpu};  //!< applied when the world is rebuilt

   bool _running{true};
   bool _rebuild_requested{false};  //!< rebuild the world from _scene (reset, preset, spacing)
   bool _reframe_requested{false};
   float _time_scale{1.0f};    //!< simulated seconds per real second
   int32_t _max_substeps{40};  //!< per frame; when reached the simulation runs in slow motion
   int32_t _preset_index{0};
   tinysph::StepStatistics _statistics;
   tinysph::Milliseconds _frame_time{};
   tinysph::Milliseconds _mesh_time{};
   RenderSettings _render_settings;
   MouseTool _mouse_tool;
   FluidDraft _fluid_draft;
};

/// \brief loads a preset into the state (scene and gravity) and requests a rebuild.
void loadPreset(SandboxState& state, int32_t preset_index);

/// \brief draws the sandbox control window; edits go to the scene and, through handles, to the world.
void drawSandboxUi(tinysph::World& world, SandboxState& state);
