#include "sandbox_ui.h"

#include <array>
#include <optional>
#include <string>
#include <variant>

#include <imgui.h>

namespace
{

template <class... Callables>
struct Overloaded : Callables...
{
   using Callables::operator()...;
};

struct Preset
{
   const char* _name;
   sph::Scene (*_factory)();
};

constexpr std::array<Preset, 4> presets{{
   {"Dam break", &sph::makeDamBreak},
   {"Drop into pool", &sph::makeDropIntoPool},
   {"Two blobs (zero gravity)", &sph::makeTwoBlobs},
   {"Fountain", &sph::makeFountain},
}};

float* asFloats(sph::Vec3& value)
{
   return &value.x;
}

/// \brief draws one list entry and returns true when its remove button was pressed.
template <class Function>
bool listEntry(const std::string& label, std::size_t index, Function&& draw_body)
{
   ImGui::PushID(static_cast<int32_t>(index));
   auto remove_requested = false;
   if (ImGui::TreeNodeEx(label.c_str(), ImGuiTreeNodeFlags_DefaultOpen))
   {
      draw_body();
      remove_requested = ImGui::SmallButton("Remove");
      ImGui::TreePop();
   }
   ImGui::PopID();
   return remove_requested;
}

void drawSimulationSection(sph::Solver& solver, SandboxState& state)
{
   if (!ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
   {
      return;
   }

   if (ImGui::Button(state._running ? "Pause (space)" : "Play (space)"))
   {
      state._running = !state._running;
   }
   ImGui::SameLine();
   ImGui::BeginDisabled(state._running);
   if (ImGui::Button("Step"))
   {
      solver.advance(1.0f / 60.0f * state._time_scale, state._max_substeps);
   }
   ImGui::EndDisabled();
   ImGui::SameLine();
   if (ImGui::Button("Reset (r)"))
   {
      state._reset_requested = true;
   }

   ImGui::SliderFloat("time scale", &state._time_scale, 0.05f, 2.0f, "%.2fx");
   ImGui::SliderInt("max substeps / frame", &state._max_substeps, 1, 200);

   const auto& statistics = state._statistics;
   const auto realtime_ratio = statistics._simulated_time / (1.0f / 60.0f);
   ImGui::Text("particles      %zu", solver.particleCount());
   ImGui::Text("substeps       %d  (dt %.2f ms)", statistics._substeps, statistics._time_step * 1000.0f);
   ImGui::Text("solver         %.1f ms / frame", statistics._step_ms);
   ImGui::Text("frame          %.1f ms", state._frame_ms);
   ImGui::Text("sim speed      %.2fx real time", realtime_ratio);
   ImGui::Text("density        avg %.3f  max %.3f  (rho/rho0)", statistics._average_density_ratio, statistics._max_density_ratio);
   ImGui::Text("max speed      %.2f m/s", statistics._max_speed);
}

void drawSceneSection(sph::Solver& solver, SandboxState& state)
{
   if (!ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
   {
      return;
   }

   if (ImGui::BeginCombo("preset", presets[static_cast<std::size_t>(state._preset_index)]._name))
   {
      for (auto preset_index = 0; preset_index < static_cast<int32_t>(presets.size()); ++preset_index)
      {
         if (ImGui::Selectable(presets[static_cast<std::size_t>(preset_index)]._name, preset_index == state._preset_index))
         {
            state._preset_index = preset_index;
            solver.loadScene(makePresetScene(preset_index));
            state._reframe_requested = true;
         }
      }
      ImGui::EndCombo();
   }

   auto& scene = solver.scene();
   ImGui::DragFloat3("gravity", asFloats(scene._gravity), 0.05f);
   if (ImGui::Button("Clear particles"))
   {
      solver.clearParticles();
      scene._fluids.clear();
   }

   ImGui::SeparatorText("container");
   auto container_changed = ImGui::DragFloat3("min", asFloats(scene._container._min), 0.01f);
   container_changed |= ImGui::DragFloat3("max", asFloats(scene._container._max), 0.01f);
   if (container_changed)
   {
      // the grid depends on the container; particles outside are pushed back in by the next step
      scene._container._max = sph::componentMax(scene._container._max, scene._container._min + sph::Vec3{0.1f, 0.1f, 0.1f});
      solver.applySettings();
   }
}

void drawFluidSection(sph::Solver& solver, SandboxState& state)
{
   if (!ImGui::CollapsingHeader("Fluid"))
   {
      return;
   }

   auto& settings = solver.settings();
   ImGui::SliderFloat("speed of sound", &settings._speed_of_sound, 5.0f, 100.0f, "%.1f m/s");
   ImGui::SetItemTooltip("stiffness: higher = less compressible, but smaller time steps");
   ImGui::SliderFloat("viscosity", &settings._viscosity, 0.0f, 5.0f, "%.3f Pa s", ImGuiSliderFlags_Logarithmic);
   ImGui::SliderFloat("artificial viscosity", &settings._artificial_viscosity, 0.0f, 0.5f, "%.3f", ImGuiSliderFlags_Logarithmic);
   ImGui::SetItemTooltip("monaghan alpha: damps colliding particles only; at 0 the fluid keeps boiling");
   ImGui::Checkbox("clamp negative pressure", &settings._clamp_negative_pressure);
   ImGui::SetItemTooltip("prevents particles from clumping at the free surface");
   ImGui::SliderFloat("wall restitution", &settings._wall_restitution, 0.0f, 1.0f);
   ImGui::SliderFloat("wall friction", &settings._wall_friction, 0.0f, 1.0f);
   ImGui::SliderFloat("cfl number", &settings._cfl_number, 0.05f, 1.0f);

   auto spacing = settings._particle_spacing;
   if (ImGui::SliderFloat("particle spacing", &spacing, 0.01f, 0.05f, "%.4f m"))
   {
      settings._particle_spacing = spacing;
   }
   ImGui::SetItemTooltip("resolution; applied on reset (particle count grows with 1/spacing^3)");

   ImGui::SeparatorText("add fluid");
   auto& draft = state._fluid_draft;
   ImGui::DragFloat3("initial velocity", asFloats(draft._velocity), 0.05f);
   ImGui::Checkbox("keep in scene (survives reset)", &draft._keep_in_scene);

   ImGui::DragFloat3("box min", asFloats(draft._box._min), 0.01f);
   ImGui::DragFloat3("box max", asFloats(draft._box._max), 0.01f);
   if (ImGui::Button("Add box"))
   {
      solver.addFluidBox(draft._box, draft._velocity);
      if (draft._keep_in_scene)
      {
         solver.scene()._fluids.emplace_back(sph::FluidBox{draft._box, draft._velocity});
      }
   }

   ImGui::DragFloat3("sphere center", asFloats(draft._sphere_center), 0.01f);
   ImGui::DragFloat("sphere radius", &draft._sphere_radius, 0.005f, 0.02f, 1.0f);
   if (ImGui::Button("Add sphere"))
   {
      solver.addFluidSphere(draft._sphere_center, draft._sphere_radius, draft._velocity);
      if (draft._keep_in_scene)
      {
         solver.scene()._fluids.emplace_back(sph::FluidSphere{draft._sphere_center, draft._sphere_radius, draft._velocity});
      }
   }
}

void drawCollidersSection(sph::Solver& solver)
{
   if (!ImGui::CollapsingHeader("Colliders"))
   {
      return;
   }

   auto& scene = solver.scene();
   const auto center = scene._container.center();
   if (ImGui::Button("Add sphere##collider"))
   {
      scene._colliders.emplace_back(sph::SphereCollider{center, 0.1f});
   }
   ImGui::SameLine();
   if (ImGui::Button("Add box##collider"))
   {
      scene._colliders.emplace_back(sph::BoxCollider{center, {0.1f, 0.1f, 0.1f}});
   }

   std::optional<std::size_t> remove_index;
   for (std::size_t collider_index = 0; collider_index < scene._colliders.size(); ++collider_index)
   {
      auto& collider = scene._colliders[collider_index];
      const auto is_sphere = std::holds_alternative<sph::SphereCollider>(collider);
      const auto label = std::string{is_sphere ? "sphere " : "box "} + std::to_string(collider_index);
      const auto remove_requested = listEntry(
         label,
         collider_index,
         [&]()
         {
            std::visit(
               Overloaded{
                  [](sph::SphereCollider& sphere)
                  {
                     ImGui::DragFloat3("center", asFloats(sphere._center), 0.01f);
                     ImGui::DragFloat("radius", &sphere._radius, 0.005f, 0.01f, 2.0f);
                  },
                  [](sph::BoxCollider& box)
                  {
                     ImGui::DragFloat3("center", asFloats(box._center), 0.01f);
                     ImGui::DragFloat3("half extents", asFloats(box._half_extents), 0.005f, 0.005f, 2.0f);
                  },
               },
               collider
            );
         }
      );
      if (remove_requested)
      {
         remove_index = collider_index;
      }
   }
   if (remove_index.has_value())
   {
      scene._colliders.erase(scene._colliders.begin() + static_cast<std::ptrdiff_t>(*remove_index));
   }
}

void drawForceFieldsSection(sph::Solver& solver)
{
   if (!ImGui::CollapsingHeader("Force fields"))
   {
      return;
   }

   auto& scene = solver.scene();
   const auto center = scene._container.center();
   if (ImGui::Button("Add attractor"))
   {
      scene._fields.emplace_back(sph::Attractor{center, 20.0f, 0.3f});
   }
   ImGui::SameLine();
   if (ImGui::Button("Add vortex"))
   {
      scene._fields.emplace_back(sph::Vortex{center, {0.0f, 1.0f, 0.0f}, 10.0f, 0.3f});
   }
   ImGui::SameLine();
   if (ImGui::Button("Add wind"))
   {
      scene._fields.emplace_back(sph::Wind{scene._container, {3.0f, 0.0f, 0.0f}});
   }

   std::optional<std::size_t> remove_index;
   for (std::size_t field_index = 0; field_index < scene._fields.size(); ++field_index)
   {
      auto& field = scene._fields[field_index];
      const auto type_name = std::visit(
         Overloaded{
            [](const sph::Attractor&) { return "attractor "; },
            [](const sph::Vortex&) { return "vortex "; },
            [](const sph::Wind&) { return "wind "; },
         },
         field
      );
      const auto remove_requested = listEntry(
         std::string{type_name} + std::to_string(field_index),
         field_index,
         [&]()
         {
            std::visit(
               Overloaded{
                  [](sph::Attractor& attractor)
                  {
                     ImGui::DragFloat3("position", asFloats(attractor._position), 0.01f);
                     ImGui::DragFloat("strength", &attractor._strength, 0.5f, -200.0f, 200.0f);
                     ImGui::DragFloat("radius", &attractor._radius, 0.005f, 0.01f, 3.0f);
                  },
                  [](sph::Vortex& vortex)
                  {
                     ImGui::DragFloat3("position", asFloats(vortex._position), 0.01f);
                     ImGui::DragFloat3("axis", asFloats(vortex._axis), 0.01f);
                     ImGui::DragFloat("strength", &vortex._strength, 0.5f, -200.0f, 200.0f);
                     ImGui::DragFloat("radius", &vortex._radius, 0.005f, 0.01f, 3.0f);
                  },
                  [](sph::Wind& wind)
                  {
                     ImGui::DragFloat3("region min", asFloats(wind._region._min), 0.01f);
                     ImGui::DragFloat3("region max", asFloats(wind._region._max), 0.01f);
                     ImGui::DragFloat3("acceleration", asFloats(wind._acceleration), 0.05f);
                  },
               },
               field
            );
         }
      );
      if (remove_requested)
      {
         remove_index = field_index;
      }
   }
   if (remove_index.has_value())
   {
      scene._fields.erase(scene._fields.begin() + static_cast<std::ptrdiff_t>(*remove_index));
   }
}

void drawEmittersSection(sph::Solver& solver)
{
   if (!ImGui::CollapsingHeader("Emitters"))
   {
      return;
   }

   auto& scene = solver.scene();
   if (ImGui::Button("Add emitter"))
   {
      const auto center = scene._container.center();
      scene._emitters.push_back(sph::Emitter{{center.x, scene._container._max.y - 0.1f, center.z}, {0.0f, -1.0f, 0.0f}, 0.04f, true, 0.0f});
   }

   std::optional<std::size_t> remove_index;
   for (std::size_t emitter_index = 0; emitter_index < scene._emitters.size(); ++emitter_index)
   {
      auto& emitter = scene._emitters[emitter_index];
      const auto remove_requested = listEntry(
         "emitter " + std::to_string(emitter_index),
         emitter_index,
         [&]()
         {
            ImGui::Checkbox("enabled", &emitter._enabled);
            ImGui::DragFloat3("position", asFloats(emitter._position), 0.01f);
            ImGui::DragFloat3("velocity", asFloats(emitter._velocity), 0.05f);
            ImGui::DragFloat("radius", &emitter._radius, 0.002f, 0.01f, 0.3f);
         }
      );
      if (remove_requested)
      {
         remove_index = emitter_index;
      }
   }
   if (remove_index.has_value())
   {
      scene._emitters.erase(scene._emitters.begin() + static_cast<std::ptrdiff_t>(*remove_index));
   }
}

void drawInteractionSection(SandboxState& state)
{
   if (!ImGui::CollapsingHeader("Mouse tool", ImGuiTreeNodeFlags_DefaultOpen))
   {
      return;
   }

   auto& tool = state._mouse_tool;
   auto mode = static_cast<int32_t>(tool._mode);
   ImGui::RadioButton("off", &mode, static_cast<int32_t>(MouseToolMode::Off));
   ImGui::SameLine();
   ImGui::RadioButton("pull", &mode, static_cast<int32_t>(MouseToolMode::Pull));
   ImGui::SameLine();
   ImGui::RadioButton("push", &mode, static_cast<int32_t>(MouseToolMode::Push));
   tool._mode = static_cast<MouseToolMode>(mode);
   ImGui::SliderFloat("strength##tool", &tool._strength, 1.0f, 300.0f, "%.0f m/s^2", ImGuiSliderFlags_Logarithmic);
   ImGui::SliderFloat("radius##tool", &tool._radius, 0.03f, 0.5f, "%.2f m");
   ImGui::TextDisabled("left drag: tool, right drag: orbit, middle drag: pan, wheel: zoom");
}

void drawRenderSection(SandboxState& state)
{
   if (!ImGui::CollapsingHeader("Render"))
   {
      return;
   }

   auto& render_settings = state._render_settings;
   auto render_mode = static_cast<int32_t>(render_settings._render_mode);
   ImGui::RadioButton("spheres", &render_mode, static_cast<int32_t>(RenderMode::Spheres));
   ImGui::SameLine();
   ImGui::RadioButton("fluid surface", &render_mode, static_cast<int32_t>(RenderMode::FluidSurface));
   ImGui::SetItemTooltip("smoothed hull around the particles with reflection and refraction (screen space fluid)");
   render_settings._render_mode = static_cast<RenderMode>(render_mode);

   if (render_settings._render_mode == RenderMode::Spheres)
   {
      auto color_mode = static_cast<int32_t>(render_settings._color_mode);
      ImGui::Combo("color", &color_mode, "speed\0density\0flat\0");
      render_settings._color_mode = static_cast<ColorMode>(color_mode);
      ImGui::SliderFloat("speed range", &render_settings._speed_range, 0.5f, 10.0f, "%.1f m/s");
      ImGui::SliderFloat("particle size", &render_settings._radius_scale, 0.3f, 2.0f);
   }
   else
   {
      ImGui::SliderInt("smoothing passes", &render_settings._smoothing_iterations, 0, 8);
      ImGui::SliderFloat("smoothing radius", &render_settings._smoothing_radius, 0.5f, 6.0f, "%.1f particle radii");
      ImGui::SliderFloat("refraction", &render_settings._refraction_strength, 0.0f, 0.5f);
      ImGui::ColorEdit3("deep water color", render_settings._fluid_color.data());
      ImGui::DragFloat3("absorption (r g b, 1/m)", render_settings._absorption.data(), 0.05f, 0.0f, 20.0f);
   }

   ImGui::Checkbox("show helpers", &render_settings._show_helpers);
   ImGui::SameLine();
   ImGui::Checkbox("show floor", &render_settings._show_floor);
   if (ImGui::Button("Frame container"))
   {
      state._reframe_requested = true;
   }
}

}  // namespace

sph::Scene makePresetScene(int32_t preset_index)
{
   return presets[static_cast<std::size_t>(preset_index) % presets.size()]._factory();
}

void drawSandboxUi(sph::Solver& solver, SandboxState& state)
{
   ImGui::SetNextWindowPos({10.0f, 10.0f}, ImGuiCond_FirstUseEver);
   ImGui::SetNextWindowSize({420.0f, 720.0f}, ImGuiCond_FirstUseEver);
   if (ImGui::Begin("SPH Sandbox"))
   {
      drawSimulationSection(solver, state);
      drawSceneSection(solver, state);
      drawInteractionSection(state);
      drawFluidSection(solver, state);
      drawCollidersSection(solver);
      drawForceFieldsSection(solver);
      drawEmittersSection(solver);
      drawRenderSection(state);
   }
   ImGui::End();
}
