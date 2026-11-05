#pragma once

#include <string>
#include <vector>

#include "tinysph/tinysph.h"

// the sandbox keeps its own description of the scene: what to fill on reset, and the colliders, fields and emitters
// together with their handles in the current world. the library itself only simulates.

struct SceneFluid
{
   tinysph::Shape _shape;
   tinysph::Vec3 _velocity;
};

struct SceneCollider
{
   tinysph::Shape _shape;
   tinysph::ColliderId _id{};
};

struct SceneForceField
{
   tinysph::ForceField _field;
   tinysph::ForceFieldId _id{};
};

struct SceneEmitter
{
   tinysph::Emitter _emitter;
   tinysph::EmitterId _id{};
};

struct SandboxScene
{
   std::string _name;
   tinysph::Aabb _domain{{0.0f, 0.0f, 0.0f}, {1.6f, 1.0f, 0.6f}};
   tinysph::Vec3 _gravity{0.0f, -9.81f, 0.0f};
   float _surface_tension{0.0f};  //!< off by default, it costs about half again the solver time
   std::vector<SceneFluid> _fluids;
   std::vector<SceneCollider> _colliders;
   std::vector<SceneForceField> _fields;
   std::vector<SceneEmitter> _emitters;
};

/// \brief creates a world for the scene, fills the fluids and stores the new handles in the scene.
[[nodiscard]] tinysph::World buildWorld(SandboxScene& scene, float particle_spacing, const tinysph::FluidParameters& parameters);

[[nodiscard]] int32_t presetCount();
[[nodiscard]] const char* presetName(int32_t preset_index);
[[nodiscard]] SandboxScene makePresetScene(int32_t preset_index);
