#include "sandbox_scene.h"

#include <array>

namespace
{

SandboxScene makeDamBreak()
{
   SandboxScene scene;
   scene._name = "Dam break";
   scene._domain = {{0.0f, 0.0f, 0.0f}, {1.6f, 1.0f, 0.6f}};
   scene._fluids.push_back({tinysph::Box{{0.26f, 0.31f, 0.3f}, {0.24f, 0.29f, 0.28f}}, {}});
   scene._colliders.push_back({tinysph::Sphere{{1.1f, 0.12f, 0.3f}, 0.12f}});
   return scene;
}

SandboxScene makeDropIntoPool()
{
   SandboxScene scene;
   scene._name = "Drop into pool";
   scene._domain = {{0.0f, 0.0f, 0.0f}, {1.0f, 1.2f, 1.0f}};
   scene._fluids.push_back({tinysph::Box{{0.5f, 0.11f, 0.5f}, {0.48f, 0.09f, 0.48f}}, {}});
   scene._fluids.push_back({tinysph::Sphere{{0.5f, 0.75f, 0.5f}, 0.15f}, {0.0f, -1.0f, 0.0f}});
   return scene;
}

SandboxScene makeTwoBlobs()
{
   SandboxScene scene;
   scene._name = "Two blobs (zero gravity)";
   scene._domain = {{0.0f, 0.0f, 0.0f}, {1.6f, 1.0f, 1.0f}};
   scene._gravity = {};
   scene._fluids.push_back({tinysph::Sphere{{0.45f, 0.5f, 0.5f}, 0.15f}, {1.0f, 0.0f, 0.0f}});
   scene._fluids.push_back({tinysph::Sphere{{1.15f, 0.55f, 0.5f}, 0.15f}, {-1.0f, 0.0f, 0.0f}});
   return scene;
}

SandboxScene makeFountain()
{
   SandboxScene scene;
   scene._name = "Fountain";
   scene._domain = {{0.0f, 0.0f, 0.0f}, {1.2f, 1.0f, 1.2f}};
   scene._fluids.push_back({tinysph::Box{{0.6f, 0.06f, 0.6f}, {0.58f, 0.04f, 0.58f}}, {}});
   scene._colliders.push_back({tinysph::Box{{0.6f, 0.45f, 0.6f}, {0.15f, 0.02f, 0.15f}}});
   scene._emitters.push_back({tinysph::Emitter{{0.6f, 0.15f, 0.6f}, {0.0f, 3.0f, 0.0f}, 0.04f, true}});
   scene._fields.push_back({tinysph::Vortex{{0.6f, 0.5f, 0.6f}, {0.0f, 1.0f, 0.0f}, 4.0f, 0.5f}});
   return scene;
}

struct Preset
{
   const char* _name;
   SandboxScene (*_factory)();
};

constexpr std::array<Preset, 4> presets{{
   {"Dam break", &makeDamBreak},
   {"Drop into pool", &makeDropIntoPool},
   {"Two blobs (zero gravity)", &makeTwoBlobs},
   {"Fountain", &makeFountain},
}};

}  // namespace

tinysph::World buildWorld(SandboxScene& scene, float particle_spacing, const tinysph::FluidParameters& parameters)
{
   tinysph::World world{{._domain = scene._domain, ._particle_spacing = particle_spacing}, parameters};

   // colliders first, so fluid is not spawned inside them
   for (auto& collider : scene._colliders)
   {
      collider._id = world.addCollider(collider._shape);
   }
   for (auto& field : scene._fields)
   {
      field._id = world.addForceField(field._field);
   }
   for (auto& emitter : scene._emitters)
   {
      emitter._id = world.addEmitter(emitter._emitter);
   }
   for (const auto& fluid : scene._fluids)
   {
      world.addFluid(fluid._shape, fluid._velocity);
   }
   return world;
}

int32_t presetCount()
{
   return static_cast<int32_t>(presets.size());
}

const char* presetName(int32_t preset_index)
{
   return presets[static_cast<std::size_t>(preset_index) % presets.size()]._name;
}

SandboxScene makePresetScene(int32_t preset_index)
{
   return presets[static_cast<std::size_t>(preset_index) % presets.size()]._factory();
}
