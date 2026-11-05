#include "sandbox_ui.h"

#include <chrono>
#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <variant>

#include <imgui.h>

namespace
{

template <class... Callables>
struct Overloaded : Callables...
{
   using Callables::operator()...;
};

float* asFloats(tinysph::Vec3& value)
{
   return &value.x;
}

/// \brief draws one list entry; returns true when its remove button was pressed.
///        draw_body returns true when it changed something.
template <class Function>
bool listEntry(const std::string& label, std::ptrdiff_t index, bool& changed, Function&& draw_body)
{
   ImGui::PushID(static_cast<int32_t>(index));
   auto remove_requested = false;
   if (ImGui::TreeNodeEx(label.c_str(), ImGuiTreeNodeFlags_DefaultOpen))
   {
      changed = draw_body();
      remove_requested = ImGui::SmallButton("Remove");
      ImGui::TreePop();
   }
   ImGui::PopID();
   return remove_requested;
}

/// \brief edits a shape in place; returns true when it changed.
bool editShape(tinysph::Shape& shape)
{
   return std::visit(
      Overloaded{
         [](tinysph::Sphere& sphere)
         {
            auto changed = ImGui::DragFloat3("center", asFloats(sphere._center), 0.01f);
            changed |= ImGui::DragFloat("radius", &sphere._radius, 0.005f, 0.01f, 2.0f);
            return changed;
         },
         [](tinysph::Box& box)
         {
            auto changed = ImGui::DragFloat3("center", asFloats(box._center), 0.01f);
            changed |= ImGui::DragFloat3("half extents", asFloats(box._half_extents), 0.005f, 0.005f, 2.0f);
            return changed;
         },
      },
      shape
   );
}

void drawSimulationSection(tinysph::World& world, SandboxState& state)
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
      state._statistics = world.step(frame_time * state._time_scale, state._max_substeps);
   }
   ImGui::EndDisabled();
   ImGui::SameLine();
   if (ImGui::Button("Reset (r)"))
   {
      state._rebuild_requested = true;
   }

   // switching rebuilds the world; without a usable gpu buildWorld() falls back to the cpu
   auto use_gpu = state._backend == tinysph::Backend::Gpu;
#if defined(__EMSCRIPTEN__)
   // the gpu backend is wgpu-native, which does not run in the browser (yet)
   ImGui::BeginDisabled();
   ImGui::Checkbox("gpu", &use_gpu);
   ImGui::EndDisabled();
   ImGui::SetItemTooltip("not available in the browser yet");
#else
   if (ImGui::Checkbox("gpu", &use_gpu))
   {
      state._backend = use_gpu ? tinysph::Backend::Gpu : tinysph::Backend::Cpu;
      state._rebuild_requested = true;
   }
#endif
   ImGui::SameLine();
   ImGui::TextDisabled("%.*s", static_cast<int>(world.deviceDescription().size()), world.deviceDescription().data());

   ImGui::SliderFloat("time scale", &state._time_scale, 0.05f, 2.0f, "%.2fx");
   ImGui::SliderInt("max substeps / frame", &state._max_substeps, 1, 200);

   const auto& statistics = state._statistics;
   const auto realtime_ratio = statistics._simulated_time / frame_time;
   ImGui::Text("particles      %zu", world.particleCount());
   ImGui::Text(
      "substeps       %d  (dt %.2f ms)", statistics._substeps, std::chrono::duration<float, std::milli>{statistics._time_step}.count()
   );
   ImGui::Text("solver         %.1f ms / frame", statistics._wall_time.count());
   ImGui::Text("frame          %.1f ms", state._frame_time.count());
   ImGui::Text("surface mesh   %.1f ms", state._mesh_time.count());
   ImGui::Text("sim speed      %.2fx real time", realtime_ratio);
   ImGui::Text("density        avg %.3f  max %.3f  (rho/rho0)", statistics._average_density_ratio, statistics._max_density_ratio);
   ImGui::Text("max speed      %.2f m/s", statistics._max_speed);
}

void drawSceneSection(tinysph::World& world, SandboxState& state)
{
   if (!ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
   {
      return;
   }

   if (ImGui::BeginCombo("preset", presetName(state._preset_index)))
   {
      for (auto preset_index = 0; preset_index < presetCount(); ++preset_index)
      {
         if (ImGui::Selectable(presetName(preset_index), preset_index == state._preset_index))
         {
            loadPreset(state, preset_index);
            state._reframe_requested = true;
         }
      }
      ImGui::EndCombo();
   }

   ImGui::DragFloat3("gravity", asFloats(state._parameters._gravity), 0.05f);
   if (ImGui::Button("Clear particles"))
   {
      world.clearParticles();
      state._scene._fluids.clear();
   }

   ImGui::SeparatorText("domain");
   auto& domain = state._scene._domain;
   auto domain_changed = ImGui::DragFloat3("min", asFloats(domain._min), 0.01f);
   domain_changed |= ImGui::DragFloat3("max", asFloats(domain._max), 0.01f);
   if (domain_changed)
   {
      world.setDomain(domain);
      domain = world.domain();
   }
}

void drawFluidSection(tinysph::World& world, SandboxState& state)
{
   if (!ImGui::CollapsingHeader("Fluid"))
   {
      return;
   }

   auto& parameters = state._parameters;
   ImGui::SliderFloat("speed of sound", &parameters._speed_of_sound, 5.0f, 100.0f, "%.1f m/s");
   ImGui::SetItemTooltip("stiffness: higher = less compressible, but smaller time steps");
   ImGui::SliderFloat("viscosity", &parameters._viscosity, 0.0f, 5.0f, "%.3f Pa s", ImGuiSliderFlags_Logarithmic);
   ImGui::SliderFloat("artificial viscosity", &parameters._artificial_viscosity, 0.0f, 0.5f, "%.3f", ImGuiSliderFlags_Logarithmic);
   ImGui::SetItemTooltip("monaghan alpha: damps colliding particles only; at 0 the fluid keeps boiling");
   ImGui::SliderFloat("surface tension", &parameters._surface_tension, 0.0f, 5.0f, "%.2f");
   ImGui::SetItemTooltip("holds drops together and rounds them; costs about half again the solver time");
   ImGui::Checkbox("clamp negative pressure", &parameters._clamp_negative_pressure);
   ImGui::SetItemTooltip("prevents particles from clumping at the free surface");
   ImGui::SliderFloat("wall restitution", &parameters._wall_restitution, 0.0f, 1.0f);
   ImGui::SliderFloat("wall friction", &parameters._wall_friction, 0.0f, 1.0f);
   ImGui::SliderFloat("cfl number", &parameters._cfl_number, 0.05f, 1.0f);

   ImGui::SliderFloat("particle spacing", &state._particle_spacing, 0.01f, 0.05f, "%.4f m");
   ImGui::SetItemTooltip("resolution; applied on reset (particle count grows with 1/spacing^3)");

   ImGui::SeparatorText("add fluid");
   auto& draft = state._fluid_draft;
   ImGui::DragFloat3("initial velocity", asFloats(draft._velocity), 0.05f);
   ImGui::Checkbox("keep in scene (survives reset)", &draft._keep_in_scene);

   ImGui::DragFloat3("box center", asFloats(draft._box._center), 0.01f);
   ImGui::DragFloat3("box half extents", asFloats(draft._box._half_extents), 0.005f, 0.01f, 2.0f);
   if (ImGui::Button("Add box"))
   {
      world.addFluid(draft._box, draft._velocity);
      if (draft._keep_in_scene)
      {
         state._scene._fluids.push_back({draft._box, draft._velocity});
      }
   }

   ImGui::DragFloat3("sphere center", asFloats(draft._sphere._center), 0.01f);
   ImGui::DragFloat("sphere radius", &draft._sphere._radius, 0.005f, 0.02f, 1.0f);
   if (ImGui::Button("Add sphere"))
   {
      world.addFluid(draft._sphere, draft._velocity);
      if (draft._keep_in_scene)
      {
         state._scene._fluids.push_back({draft._sphere, draft._velocity});
      }
   }
}

void drawCollidersSection(tinysph::World& world, SandboxState& state)
{
   if (!ImGui::CollapsingHeader("Colliders"))
   {
      return;
   }

   auto& colliders = state._scene._colliders;
   const auto center = state._scene._domain.center();
   const auto add_collider = [&](const tinysph::Shape& shape) { colliders.push_back({shape, world.addCollider(shape)}); };
   if (ImGui::Button("Add sphere##collider"))
   {
      add_collider(tinysph::Sphere{center, 0.1f});
   }
   ImGui::SameLine();
   if (ImGui::Button("Add box##collider"))
   {
      add_collider(tinysph::Box{center, {0.1f, 0.1f, 0.1f}});
   }

   std::optional<std::ptrdiff_t> remove_index;
   for (auto collider_index = std::ptrdiff_t{0}; collider_index < std::ssize(colliders); ++collider_index)
   {
      auto& collider = colliders[static_cast<std::size_t>(collider_index)];
      const auto is_sphere = std::holds_alternative<tinysph::Sphere>(collider._shape);
      const auto label = std::string{is_sphere ? "sphere " : "box "} + std::to_string(collider_index);
      auto changed = false;
      if (listEntry(label, collider_index, changed, [&]() { return editShape(collider._shape); }))
      {
         remove_index = collider_index;
      }
      if (changed)
      {
         world.setCollider(collider._id, collider._shape);
      }
   }
   if (remove_index.has_value())
   {
      const auto removed = colliders.begin() + *remove_index;
      world.removeCollider(removed->_id);
      colliders.erase(removed);
   }
}

void drawForceFieldsSection(tinysph::World& world, SandboxState& state)
{
   if (!ImGui::CollapsingHeader("Force fields"))
   {
      return;
   }

   auto& fields = state._scene._fields;
   const auto center = state._scene._domain.center();
   const auto add_field = [&](const tinysph::ForceField& field) { fields.push_back({field, world.addForceField(field)}); };
   if (ImGui::Button("Add attractor"))
   {
      add_field(tinysph::Attractor{center, 20.0f, 0.3f});
   }
   ImGui::SameLine();
   if (ImGui::Button("Add vortex"))
   {
      add_field(tinysph::Vortex{center, {0.0f, 1.0f, 0.0f}, 10.0f, 0.3f});
   }
   ImGui::SameLine();
   if (ImGui::Button("Add wind"))
   {
      add_field(tinysph::Wind{state._scene._domain, {3.0f, 0.0f, 0.0f}});
   }

   std::optional<std::ptrdiff_t> remove_index;
   for (auto field_index = std::ptrdiff_t{0}; field_index < std::ssize(fields); ++field_index)
   {
      auto& field = fields[static_cast<std::size_t>(field_index)];
      const auto type_name = std::visit(
         Overloaded{
            [](const tinysph::Attractor&) { return "attractor "; },
            [](const tinysph::Vortex&) { return "vortex "; },
            [](const tinysph::Wind&) { return "wind "; },
         },
         field._field
      );
      auto changed = false;
      const auto remove_requested = listEntry(
         std::string{type_name} + std::to_string(field_index),
         field_index,
         changed,
         [&]()
         {
            return std::visit(
               Overloaded{
                  [](tinysph::Attractor& attractor)
                  {
                     auto edited = ImGui::DragFloat3("position", asFloats(attractor._position), 0.01f);
                     edited |= ImGui::DragFloat("strength", &attractor._strength, 0.5f, -200.0f, 200.0f);
                     edited |= ImGui::DragFloat("radius", &attractor._radius, 0.005f, 0.01f, 3.0f);
                     return edited;
                  },
                  [](tinysph::Vortex& vortex)
                  {
                     auto edited = ImGui::DragFloat3("position", asFloats(vortex._position), 0.01f);
                     edited |= ImGui::DragFloat3("axis", asFloats(vortex._axis), 0.01f);
                     edited |= ImGui::DragFloat("strength", &vortex._strength, 0.5f, -200.0f, 200.0f);
                     edited |= ImGui::DragFloat("radius", &vortex._radius, 0.005f, 0.01f, 3.0f);
                     return edited;
                  },
                  [](tinysph::Wind& wind)
                  {
                     auto edited = ImGui::DragFloat3("region min", asFloats(wind._region._min), 0.01f);
                     edited |= ImGui::DragFloat3("region max", asFloats(wind._region._max), 0.01f);
                     edited |= ImGui::DragFloat3("acceleration", asFloats(wind._acceleration), 0.05f);
                     return edited;
                  },
               },
               field._field
            );
         }
      );
      if (remove_requested)
      {
         remove_index = field_index;
      }
      if (changed)
      {
         world.setForceField(field._id, field._field);
      }
   }
   if (remove_index.has_value())
   {
      const auto removed = fields.begin() + *remove_index;
      world.removeForceField(removed->_id);
      fields.erase(removed);
   }
}

void drawEmittersSection(tinysph::World& world, SandboxState& state)
{
   if (!ImGui::CollapsingHeader("Emitters"))
   {
      return;
   }

   auto& emitters = state._scene._emitters;
   if (ImGui::Button("Add emitter"))
   {
      const auto center = state._scene._domain.center();
      const tinysph::Emitter emitter{{center.x, state._scene._domain._max.y - 0.1f, center.z}, {0.0f, -1.0f, 0.0f}, 0.04f, true};
      emitters.push_back({emitter, world.addEmitter(emitter)});
   }

   std::optional<std::ptrdiff_t> remove_index;
   for (auto emitter_index = std::ptrdiff_t{0}; emitter_index < std::ssize(emitters); ++emitter_index)
   {
      auto& emitter = emitters[static_cast<std::size_t>(emitter_index)];
      auto changed = false;
      const auto remove_requested = listEntry(
         "emitter " + std::to_string(emitter_index),
         emitter_index,
         changed,
         [&]()
         {
            auto edited = ImGui::Checkbox("enabled", &emitter._emitter._enabled);
            edited |= ImGui::DragFloat3("position", asFloats(emitter._emitter._position), 0.01f);
            edited |= ImGui::DragFloat3("velocity", asFloats(emitter._emitter._velocity), 0.05f);
            edited |= ImGui::DragFloat("radius", &emitter._emitter._radius, 0.002f, 0.01f, 0.3f);
            return edited;
         }
      );
      if (remove_requested)
      {
         remove_index = emitter_index;
      }
      if (changed)
      {
         world.setEmitter(emitter._id, emitter._emitter);
      }
   }
   if (remove_index.has_value())
   {
      const auto removed = emitters.begin() + *remove_index;
      world.removeEmitter(removed->_id);
      emitters.erase(removed);
   }
}

void drawInteractionSection(SandboxState& state)
{
   if (!ImGui::CollapsingHeader("Mouse tool", ImGuiTreeNodeFlags_DefaultOpen))
   {
      return;
   }

   auto& tool = state._mouse_tool;
   auto mode = std::to_underlying(tool._mode);
   ImGui::RadioButton("off", &mode, std::to_underlying(MouseToolMode::Off));
   ImGui::SameLine();
   ImGui::RadioButton("pull", &mode, std::to_underlying(MouseToolMode::Pull));
   ImGui::SameLine();
   ImGui::RadioButton("push", &mode, std::to_underlying(MouseToolMode::Push));
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
   auto render_mode = std::to_underlying(render_settings._render_mode);
   ImGui::RadioButton("spheres", &render_mode, std::to_underlying(RenderMode::Spheres));
   ImGui::SameLine();
   ImGui::RadioButton("fluid surface", &render_mode, std::to_underlying(RenderMode::FluidSurface));
   ImGui::SameLine();
   ImGui::RadioButton("mesh surface", &render_mode, std::to_underlying(RenderMode::MeshSurface));
   ImGui::SetItemTooltip("one closed triangle mesh around the particles (marching tetrahedra), built on the cpu every frame");
   ImGui::SetItemTooltip("smoothed hull around the particles with reflection and refraction (screen space fluid)");
   render_settings._render_mode = static_cast<RenderMode>(render_mode);

   if (render_settings._render_mode == RenderMode::Spheres)
   {
      auto color_mode = std::to_underlying(render_settings._color_mode);
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
      ImGui::Checkbox("spray as foam", &render_settings._separate_spray);
      ImGui::SetItemTooltip("particles with low density are drawn as droplets instead of being pulled into the surface");
      ImGui::BeginDisabled(!render_settings._separate_spray);
      ImGui::SliderFloat("spray below", &render_settings._spray_threshold, 0.05f, 0.9f, "%.2f rest density");
      ImGui::EndDisabled();
   }

   ImGui::Checkbox("show helpers", &render_settings._show_helpers);
   ImGui::SameLine();
   ImGui::Checkbox("show floor", &render_settings._show_floor);
   if (ImGui::Button("Frame domain"))
   {
      state._reframe_requested = true;
   }
}

}  // namespace

void loadPreset(SandboxState& state, int32_t preset_index)
{
   state._preset_index = preset_index;
   state._scene = makePresetScene(preset_index);
   state._parameters._gravity = state._scene._gravity;
   state._parameters._surface_tension = state._scene._surface_tension;
   state._rebuild_requested = true;
}

void drawSandboxUi(tinysph::World& world, SandboxState& state)
{
   ImGui::SetNextWindowPos({10.0f, 10.0f}, ImGuiCond_FirstUseEver);
   ImGui::SetNextWindowSize({420.0f, 720.0f}, ImGuiCond_FirstUseEver);
   if (ImGui::Begin("tinysph sandbox"))
   {
      drawSimulationSection(world, state);
      drawSceneSection(world, state);
      drawInteractionSection(state);
      drawFluidSection(world, state);
      drawCollidersSection(world, state);
      drawForceFieldsSection(world, state);
      drawEmittersSection(world, state);
      drawRenderSection(state);
   }
   ImGui::End();
}
