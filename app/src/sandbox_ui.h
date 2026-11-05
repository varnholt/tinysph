#pragma once

#include <cstdint>

#include "renderer.h"
#include "sph/scene.h"
#include "sph/solver.h"

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
   sph::Aabb _box{{0.6f, 0.5f, 0.2f}, {0.8f, 0.7f, 0.4f}};
   sph::Vec3 _sphere_center{0.8f, 0.7f, 0.3f};
   float _sphere_radius{0.1f};
   sph::Vec3 _velocity{0.0f, 0.0f, 0.0f};
   bool _keep_in_scene{true};  //!< also add it to the scene so a reset recreates it
};

struct SandboxState
{
   bool _running{true};
   bool _reset_requested{false};
   bool _reframe_requested{false};
   float _time_scale{1.0f};    //!< simulated seconds per real second
   int32_t _max_substeps{40};  //!< per frame; when reached the simulation runs in slow motion
   int32_t _preset_index{0};
   sph::StepStatistics _statistics;
   float _frame_ms{0.0f};
   RenderSettings _render_settings;
   MouseTool _mouse_tool;
   FluidDraft _fluid_draft;
};

/// \brief preset scenes the ui offers.
[[nodiscard]] sph::Scene makePresetScene(int32_t preset_index);

/// \brief draws the sandbox control window and applies the changes to the solver.
void drawSandboxUi(sph::Solver& solver, SandboxState& state);
