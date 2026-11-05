#include "gpu/gpu_solver.h"

#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <initializer_list>
#include <mutex>
#include <print>
#include <span>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "geometry.h"
#include "gpu/shaders.h"
#include "solver.h"

namespace tinysph::detail
{

namespace
{

// ----------------------------------------------------------------------------------------------------
// ownership of webgpu handles
// ----------------------------------------------------------------------------------------------------

struct Releaser
{
   void operator()(WGPUInstance handle) const
   {
      wgpuInstanceRelease(handle);
   }
   void operator()(WGPUAdapter handle) const
   {
      wgpuAdapterRelease(handle);
   }
   void operator()(WGPUDevice handle) const
   {
      wgpuDeviceRelease(handle);
   }
   void operator()(WGPUQueue handle) const
   {
      wgpuQueueRelease(handle);
   }
   void operator()(WGPUBuffer handle) const
   {
      wgpuBufferRelease(handle);
   }
   void operator()(WGPUShaderModule handle) const
   {
      wgpuShaderModuleRelease(handle);
   }
   void operator()(WGPUComputePipeline handle) const
   {
      wgpuComputePipelineRelease(handle);
   }
   void operator()(WGPUBindGroupLayout handle) const
   {
      wgpuBindGroupLayoutRelease(handle);
   }
   void operator()(WGPUBindGroup handle) const
   {
      wgpuBindGroupRelease(handle);
   }
   void operator()(WGPUCommandEncoder handle) const
   {
      wgpuCommandEncoderRelease(handle);
   }
   void operator()(WGPUCommandBuffer handle) const
   {
      wgpuCommandBufferRelease(handle);
   }
   void operator()(WGPUComputePassEncoder handle) const
   {
      wgpuComputePassEncoderRelease(handle);
   }
};

/// \brief a webgpu handle that is released when it goes out of scope.
template <class Handle>
using Owned = std::unique_ptr<std::remove_pointer_t<Handle>, Releaser>;

WGPUStringView view(std::string_view text)
{
   return {text.data(), text.size()};
}

std::string toString(WGPUStringView text)
{
   if (text.data == nullptr)
   {
      return {};
   }
   return text.length == WGPU_STRLEN ? std::string{text.data} : std::string{text.data, text.length};
}

std::string_view backendName(WGPUBackendType backend)
{
   switch (backend)
   {
      case WGPUBackendType_D3D11:
         return "d3d11";
      case WGPUBackendType_D3D12:
         return "d3d12";
      case WGPUBackendType_Metal:
         return "metal";
      case WGPUBackendType_Vulkan:
         return "vulkan";
      case WGPUBackendType_OpenGL:
         return "opengl";
      case WGPUBackendType_OpenGLES:
         return "opengles";
      default:
         return "unknown api";
   }
}

/// \brief webgpu requests complete through callbacks; wgpu-native fires them while events are processed.
///        gives up after a few seconds so a broken driver cannot hang World::create().
template <class Done>
bool processEventsUntil(WGPUInstance instance, Done&& done)
{
   const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
   while (!done())
   {
      if (std::chrono::steady_clock::now() > deadline)
      {
         return false;
      }
      wgpuInstanceProcessEvents(instance);
      if (!done())
      {
         std::this_thread::sleep_for(std::chrono::microseconds{100});
      }
   }
   return true;
}

constexpr uint32_t particle_workgroup_size = 64;  // matches workgroup_size in shaders.h
constexpr uint32_t scan_block_size = 256;         // matches block_size in shaders.h
constexpr uint64_t vec4_size = 16;
constexpr uint64_t vec2_size = 8;
constexpr uint64_t u32_size = 4;

// ----------------------------------------------------------------------------------------------------
// what the shaders see; the layout must match struct Params in shaders.h (wgsl uniform layout rules)
// ----------------------------------------------------------------------------------------------------

struct GpuParams
{
   Vec3 _grid_origin;
   float _support_radius;
   std::array<int32_t, 3> _grid_size;
   uint32_t _particle_count;
   Vec3 _gravity;
   float _time_step;
   Vec3 _domain_min;
   float _particle_radius;
   Vec3 _domain_max;
   float _mass;
   float _rest_density;
   float _stiffness;
   float _viscosity;
   float _artificial_viscosity_factor;
   float _singularity_guard;
   float _surface_tension;
   float _wall_restitution;
   float _wall_friction;
   float _poly6_factor;
   float _spiky_gradient_factor;
   float _viscosity_laplacian_factor;
   float _cohesion_factor;
   float _cohesion_offset;
   uint32_t _clamp_negative_pressure;
   uint32_t _collider_count;
   uint32_t _field_count;
};
static_assert(sizeof(GpuParams) == 144, "GpuParams must match the wgsl Params struct");
static_assert(sizeof(Vec3) == 12, "Vec3 is packed into vec3f + one scalar");

// ----------------------------------------------------------------------------------------------------
// device and pipelines, shared by all worlds in the process
// ----------------------------------------------------------------------------------------------------

struct Pipeline
{
   Owned<WGPUComputePipeline> _pipeline;
   Owned<WGPUBindGroupLayout> _layout;
};

class Context
{
   // declared first, so they are released after the pipelines below
   Owned<WGPUInstance> _instance;
   Owned<WGPUAdapter> _adapter;
   Owned<WGPUDevice> _device;
   Owned<WGPUQueue> _queue;

public:
   /// \return the shared context, or nullptr if there is no usable gpu.
   static std::shared_ptr<Context> acquire()
   {
      static std::mutex mutex;
      static std::weak_ptr<Context> shared;

      const std::scoped_lock lock{mutex};
      if (auto context = shared.lock())
      {
         return context;
      }
      auto context = std::make_shared<Context>();
      if (!context->initialize())
      {
         return nullptr;
      }
      shared = context;
      return context;
   }

   [[nodiscard]] WGPUDevice device() const
   {
      return _device.get();
   }

   [[nodiscard]] WGPUQueue queue() const
   {
      return _queue.get();
   }

   [[nodiscard]] const WGPULimits& limits() const
   {
      return _limits;
   }

   [[nodiscard]] const std::string& description() const
   {
      return _description;
   }

   Pipeline _assign_cells;
   Pipeline _scan_blocks;
   Pipeline _add_block_offsets;
   Pipeline _scatter;
   Pipeline _density;
   Pipeline _normals;
   Pipeline _forces;
   Pipeline _integrate;

   [[nodiscard]] Owned<WGPUBuffer> createBuffer(std::string_view label, uint64_t size, WGPUBufferUsage usage) const
   {
      WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
      descriptor.label = view(label);
      // webgpu wants sizes in multiples of 4 and rejects empty bindings
      descriptor.size = std::max<uint64_t>((size + 3) / 4 * 4, 16);
      descriptor.usage = usage;
      return Owned<WGPUBuffer>{wgpuDeviceCreateBuffer(_device.get(), &descriptor)};
   }

   [[nodiscard]] Owned<WGPUBindGroup> createBindGroup(const Pipeline& pipeline, std::initializer_list<WGPUBuffer> buffers) const
   {
      std::vector<WGPUBindGroupEntry> entries;
      entries.reserve(buffers.size());
      auto binding = 0u;
      for (auto* const buffer : buffers)
      {
         WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
         entry.binding = binding++;
         entry.buffer = buffer;
         entry.offset = 0;
         entry.size = WGPU_WHOLE_SIZE;
         entries.push_back(entry);
      }
      WGPUBindGroupDescriptor descriptor = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
      descriptor.layout = pipeline._layout.get();
      descriptor.entryCount = entries.size();
      descriptor.entries = entries.data();
      return Owned<WGPUBindGroup>{wgpuDeviceCreateBindGroup(_device.get(), &descriptor)};
   }

   /// \brief maps a readback buffer and blocks until the gpu has written it.
   void mapForReading(WGPUBuffer buffer, uint64_t size) const
   {
      auto status = WGPUMapAsyncStatus_Force32;
      WGPUBufferMapCallbackInfo callback = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
      callback.mode = WGPUCallbackMode_AllowProcessEvents;
      callback.callback = [](WGPUMapAsyncStatus map_status, WGPUStringView, void* userdata, void*)
      { *static_cast<WGPUMapAsyncStatus*>(userdata) = map_status; };
      callback.userdata1 = &status;
      wgpuBufferMapAsync(buffer, WGPUMapMode_Read, 0, static_cast<size_t>(size), callback);
      while (status == WGPUMapAsyncStatus_Force32)
      {
         wgpuDevicePoll(_device.get(), true, nullptr);
         wgpuInstanceProcessEvents(_instance.get());
      }
      if (status != WGPUMapAsyncStatus_Success)
      {
         reportError("mapping a readback buffer failed");
      }
   }

   void reportError(std::string_view message) const
   {
      _error = true;
      std::println(stderr, "tinysph gpu error: {}", message);
   }

private:
   bool initialize()
   {
      _instance.reset(wgpuCreateInstance(nullptr));
      if (!_instance)
      {
         return false;
      }

      // a real gpu first; the fallback is a software rasterizer (warp, lavapipe) on machines without one
      for (const auto fallback : {false, true})
      {
         _adapter = requestAdapter(fallback);
         if (_adapter)
         {
            break;
         }
      }
      if (!_adapter)
      {
         return false;
      }

      // ask for everything the adapter can do, the defaults are tuned for the web and too small for big worlds
      WGPULimits adapter_limits = WGPU_LIMITS_INIT;
      if (wgpuAdapterGetLimits(_adapter.get(), &adapter_limits) != WGPUStatus_Success)
      {
         return false;
      }
      adapter_limits.nextInChain = nullptr;

      // the forces pass binds 8 storage buffers, the minimum webgpu guarantees
      if (adapter_limits.maxStorageBuffersPerShaderStage < 8)
      {
         return false;
      }

      _device = requestDevice(adapter_limits);
      if (!_device)
      {
         return false;
      }
      _queue.reset(wgpuDeviceGetQueue(_device.get()));
      _limits = WGPU_LIMITS_INIT;
      wgpuDeviceGetLimits(_device.get(), &_limits);

      WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
      if (wgpuAdapterGetInfo(_adapter.get(), &info) == WGPUStatus_Success)
      {
         _description = std::format("{} ({})", toString(info.device), backendName(info.backendType));
         wgpuAdapterInfoFreeMembers(info);
      }

      const std::string common{shaders::common};
      _assign_cells = createPipeline("assign cells", common + shaders::assign_cells, "main");
      _scan_blocks = createPipeline("scan blocks", shaders::scan, "scan_blocks");
      _add_block_offsets = createPipeline("add block offsets", shaders::scan, "add_block_offsets");
      _scatter = createPipeline("scatter", common + shaders::scatter, "main");
      _density = createPipeline("density", common + shaders::density, "main");
      _normals = createPipeline("normals", common + shaders::normals, "main");
      _forces = createPipeline("forces", common + shaders::forces, "main");
      _integrate = createPipeline("integrate", common + shaders::integrate, "main");

      // shader compile errors arrive through the error callback
      wgpuDevicePoll(_device.get(), true, nullptr);
      return !_error;
   }

   Owned<WGPUAdapter> requestAdapter(bool fallback)
   {
      WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
      options.powerPreference = WGPUPowerPreference_HighPerformance;
      options.forceFallbackAdapter = fallback ? 1u : 0u;

      struct Result
      {
         bool _done{false};
         WGPUAdapter _adapter{nullptr};
      } result;

      WGPURequestAdapterCallbackInfo callback = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
      callback.mode = WGPUCallbackMode_AllowProcessEvents;
      callback.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView, void* userdata, void*)
      {
         auto* const request = static_cast<Result*>(userdata);
         request->_adapter = status == WGPURequestAdapterStatus_Success ? adapter : nullptr;
         request->_done = true;
      };
      callback.userdata1 = &result;
      wgpuInstanceRequestAdapter(_instance.get(), &options, callback);
      if (!processEventsUntil(_instance.get(), [&]() { return result._done; }))
      {
         return {};
      }
      return Owned<WGPUAdapter>{result._adapter};
   }

   Owned<WGPUDevice> requestDevice(const WGPULimits& limits)
   {
      WGPUDeviceDescriptor descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
      descriptor.label = view("tinysph");
      descriptor.requiredLimits = &limits;
      descriptor.uncapturedErrorCallbackInfo.callback = [](WGPUDevice const*, WGPUErrorType, WGPUStringView message, void* userdata, void*)
      { static_cast<const Context*>(userdata)->reportError(toString(message)); };
      descriptor.uncapturedErrorCallbackInfo.userdata1 = this;

      struct Result
      {
         bool _done{false};
         WGPUDevice _device{nullptr};
         std::string _message;
      } result;

      WGPURequestDeviceCallbackInfo callback = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
      callback.mode = WGPUCallbackMode_AllowProcessEvents;
      callback.callback = [](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata, void*)
      {
         auto* const request = static_cast<Result*>(userdata);
         request->_device = status == WGPURequestDeviceStatus_Success ? device : nullptr;
         request->_message = toString(message);
         request->_done = true;
      };
      callback.userdata1 = &result;
      wgpuAdapterRequestDevice(_adapter.get(), &descriptor, callback);
      if (!processEventsUntil(_instance.get(), [&]() { return result._done; }))
      {
         return {};
      }
      if (result._device == nullptr && !result._message.empty())
      {
         std::println(stderr, "tinysph gpu: no device: {}", result._message);
      }
      return Owned<WGPUDevice>{result._device};
   }

   Pipeline createPipeline(std::string_view label, const std::string& code, std::string_view entry_point) const
   {
      WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
      source.code = view(code);
      WGPUShaderModuleDescriptor module_descriptor = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
      module_descriptor.nextInChain = &source.chain;
      module_descriptor.label = view(label);
      const Owned<WGPUShaderModule> module{wgpuDeviceCreateShaderModule(_device.get(), &module_descriptor)};

      // no explicit layout: webgpu derives it from the bindings the entry point uses
      WGPUComputePipelineDescriptor pipeline_descriptor = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
      pipeline_descriptor.label = view(label);
      pipeline_descriptor.compute.module = module.get();
      pipeline_descriptor.compute.entryPoint = view(entry_point);

      Pipeline pipeline;
      pipeline._pipeline.reset(wgpuDeviceCreateComputePipeline(_device.get(), &pipeline_descriptor));
      pipeline._layout.reset(wgpuComputePipelineGetBindGroupLayout(pipeline._pipeline.get(), 0));
      return pipeline;
   }

   WGPULimits _limits = WGPU_LIMITS_INIT;
   std::string _description{"unknown gpu"};
   mutable std::atomic<bool> _error{false};
};

/// \brief one level of the multi level prefix sum: scans 'count' values in blocks of 256 and writes the block totals.
struct ScanLevel
{
   uint32_t _count{0};
   Owned<WGPUBuffer> _parameters;
   Owned<WGPUBuffer> _block_sums;
   Owned<WGPUBindGroup> _scan_group;
   Owned<WGPUBindGroup> _add_group;
};

std::vector<std::array<float, 4>> toVec4(std::span<const Vec3> values)
{
   std::vector<std::array<float, 4>> result(values.size());
   std::ranges::transform(values, result.begin(), [](const Vec3& value) { return std::array{value.x, value.y, value.z, 0.0f}; });
   return result;
}

}  // namespace

// ----------------------------------------------------------------------------------------------------
// per world state
// ----------------------------------------------------------------------------------------------------

struct GpuSolver::Impl
{
   Impl(Solver& solver, std::shared_ptr<Context> context) : _solver(solver), _context(std::move(context))
   {
      const auto& c = *_context;
      constexpr auto uniform = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
      _parameters = c.createBuffer("parameters", sizeof(GpuParams), uniform);
      _statistics = c.createBuffer("statistics", 4 * u32_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);
      _statistics_readback = c.createBuffer("statistics readback", 4 * u32_size, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
   }

   StepStatistics advance(float frame_time, int32_t max_substeps);

   void syncGrid();
   void uploadScene();
   void ensureCapacity(std::size_t required, std::size_t preserved_count);
   void uploadParticles(std::size_t first, std::size_t last);
   void createBindGroups();
   void runSubstep(float time_step, float& max_speed);
   void download(std::size_t density_count);
   [[nodiscard]] GpuParams makeParams(float time_step) const;
   void dispatch(WGPUComputePassEncoder pass, const Pipeline& pipeline, WGPUBindGroup group, uint32_t invocations, uint32_t group_size)
      const;

   Solver& _solver;
   std::shared_ptr<Context> _context;

   // particle buffers; 'positions' and 'velocities' hold the state between substeps, the rest is per substep
   std::size_t _capacity{0};
   std::size_t _device_count{0};
   Owned<WGPUBuffer> _positions;
   Owned<WGPUBuffer> _velocities;
   Owned<WGPUBuffer> _sorted_positions;
   Owned<WGPUBuffer> _sorted_velocities;
   Owned<WGPUBuffer> _accelerations;
   Owned<WGPUBuffer> _normals;
   Owned<WGPUBuffer> _density_pressure;
   Owned<WGPUBuffer> _cell_of_particle;
   Owned<WGPUBuffer> _rank_in_cell;
   Owned<WGPUBuffer> _readback;

   // uniform grid
   std::array<int32_t, 3> _grid_size{};
   uint32_t _cell_start_count{0};  //!< cells + 1
   Owned<WGPUBuffer> _cell_start;
   std::vector<ScanLevel> _scan_levels;

   // scene
   std::size_t _collider_capacity{0};
   std::size_t _field_capacity{0};
   uint32_t _collider_count{0};
   uint32_t _field_count{0};
   Owned<WGPUBuffer> _colliders;
   Owned<WGPUBuffer> _fields;

   Owned<WGPUBuffer> _parameters;
   Owned<WGPUBuffer> _statistics;  //!< [0] max acceleration^2, [1] max speed^2, as float bits
   Owned<WGPUBuffer> _statistics_readback;

   Owned<WGPUBindGroup> _assign_group;
   Owned<WGPUBindGroup> _scatter_group;
   Owned<WGPUBindGroup> _density_group;
   Owned<WGPUBindGroup> _normals_group;
   Owned<WGPUBindGroup> _forces_group;
   Owned<WGPUBindGroup> _integrate_group;
   bool _bind_groups_dirty{true};

   float _max_acceleration{0.0f};
};

StepStatistics GpuSolver::Impl::advance(float frame_time, int32_t max_substeps)
{
   const auto start_time = std::chrono::steady_clock::now();
   auto& solver = _solver;

   syncGrid();
   uploadScene();

   const auto particle_count = solver._positions.size();
   ensureCapacity(particle_count, 0);
   uploadParticles(0, particle_count);
   _device_count = particle_count;
   if (particle_count == 0)
   {
      _max_acceleration = 0.0f;
   }

   auto max_speed = solver.maxSpeed();
   auto density_count = std::size_t{0};  // particles whose density the gpu has computed
   StepStatistics statistics;
   auto remaining_time = frame_time;
   while (remaining_time > 1e-7f && statistics._substeps < max_substeps)
   {
      const auto time_step = std::min(solver.computeTimeStep(max_speed, _max_acceleration), remaining_time);
      if (_device_count > 0)
      {
         runSubstep(time_step, max_speed);
         density_count = _device_count;
      }

      // emitters run on the host and append to the gpu buffers
      solver.runEmitters(time_step);
      const auto new_count = solver._positions.size();
      if (new_count > _device_count)
      {
         ensureCapacity(new_count, _device_count);
         uploadParticles(_device_count, new_count);
         for (auto index = _device_count; index < new_count; ++index)
         {
            max_speed = std::max(max_speed, length(solver._velocities[index]));
         }
         _device_count = new_count;
      }

      remaining_time -= time_step;
      statistics._simulated_time += Seconds{time_step};
      statistics._time_step = Seconds{time_step};
      ++statistics._substeps;
   }

   download(density_count);
   solver.fillStatistics(statistics);
   statistics._wall_time = std::chrono::duration_cast<Milliseconds>(std::chrono::steady_clock::now() - start_time);
   return statistics;
}

void GpuSolver::Impl::syncGrid()
{
   const auto& size = _solver._grid_size;
   const std::array grid_size{size.x, size.y, size.z};
   if (grid_size == _grid_size && _cell_start)
   {
      return;
   }
   _grid_size = grid_size;
   _cell_start_count = static_cast<uint32_t>(size.x * size.y * size.z + 1);

   const auto& c = *_context;
   constexpr auto storage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
   _cell_start = c.createBuffer("cell start", _cell_start_count * u32_size, storage);

   // levels until one block covers everything: 256 cells per block, 65536 per two levels, 16.7 million per three
   _scan_levels.clear();
   auto count = _cell_start_count;
   while (true)
   {
      const auto block_count = (count + scan_block_size - 1) / scan_block_size;
      ScanLevel level;
      level._count = count;
      level._parameters = c.createBuffer("scan parameters", 16, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
      const std::array<uint32_t, 4> parameters{count, 0, 0, 0};
      wgpuQueueWriteBuffer(c.queue(), level._parameters.get(), 0, parameters.data(), sizeof(parameters));
      level._block_sums = c.createBuffer("scan block sums", block_count * u32_size, storage);
      _scan_levels.push_back(std::move(level));
      if (block_count <= 1)
      {
         break;
      }
      count = block_count;
   }

   // level n scans the block sums of level n - 1
   for (auto index = 0uz; index < _scan_levels.size(); ++index)
   {
      auto& level = _scan_levels[index];
      auto* const values = index == 0 ? _cell_start.get() : _scan_levels[index - 1]._block_sums.get();
      level._scan_group = c.createBindGroup(c._scan_blocks, {level._parameters.get(), values, level._block_sums.get()});
      level._add_group = c.createBindGroup(c._add_block_offsets, {level._parameters.get(), values, level._block_sums.get()});
   }

   _bind_groups_dirty = true;
}

void GpuSolver::Impl::uploadScene()
{
   const auto& c = *_context;
   constexpr auto storage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;

   // two vec4 per collider: center + kind, then radius or half extents
   std::vector<std::array<float, 4>> colliders;
   for (const auto& collider : _solver._colliders)
   {
      std::visit(
         Overloaded{
            [&](const Sphere& sphere)
            {
               colliders.push_back({sphere._center.x, sphere._center.y, sphere._center.z, 0.0f});
               colliders.push_back({sphere._radius, 0.0f, 0.0f, 0.0f});
            },
            [&](const Box& box)
            {
               colliders.push_back({box._center.x, box._center.y, box._center.z, 1.0f});
               colliders.push_back({box._half_extents.x, box._half_extents.y, box._half_extents.z, 0.0f});
            },
         },
         collider
      );
   }

   // three vec4 per field, the kind in the first w
   std::vector<std::array<float, 4>> fields;
   for (const auto& field : _solver._fields)
   {
      std::visit(
         Overloaded{
            [&](const Attractor& attractor)
            {
               fields.push_back({attractor._position.x, attractor._position.y, attractor._position.z, 0.0f});
               fields.push_back({attractor._strength, attractor._radius, 0.0f, 0.0f});
               fields.push_back({});
            },
            [&](const Vortex& vortex)
            {
               const auto axis = normalize(vortex._axis);
               fields.push_back({vortex._position.x, vortex._position.y, vortex._position.z, 1.0f});
               fields.push_back({axis.x, axis.y, axis.z, vortex._strength});
               fields.push_back({vortex._radius, 0.0f, 0.0f, 0.0f});
            },
            [&](const Wind& wind)
            {
               const auto& region = wind._region;
               fields.push_back({region._min.x, region._min.y, region._min.z, 2.0f});
               fields.push_back({region._max.x, region._max.y, region._max.z, 0.0f});
               fields.push_back({wind._acceleration.x, wind._acceleration.y, wind._acceleration.z, 0.0f});
            },
         },
         field
      );
   }

   const auto upload =
      [&](std::vector<std::array<float, 4>>& values, std::size_t& capacity, Owned<WGPUBuffer>& buffer, std::string_view label)
   {
      if (!buffer || values.size() > capacity)
      {
         capacity = std::max<std::size_t>({values.size(), capacity * 2, 8});
         buffer = c.createBuffer(label, capacity * vec4_size, storage);
         _bind_groups_dirty = true;
      }
      if (!values.empty())
      {
         wgpuQueueWriteBuffer(c.queue(), buffer.get(), 0, values.data(), values.size() * vec4_size);
      }
   };
   upload(colliders, _collider_capacity, _colliders, "colliders");
   upload(fields, _field_capacity, _fields, "force fields");
   _collider_count = static_cast<uint32_t>(_solver._colliders.size());
   _field_count = static_cast<uint32_t>(_solver._fields.size());
}

void GpuSolver::Impl::ensureCapacity(std::size_t required, std::size_t preserved_count)
{
   if (required <= _capacity && _positions)
   {
      return;
   }

   const auto& c = *_context;
   const auto capacity = std::min(std::max<std::size_t>({required, _capacity * 2, 1024}), _solver._settings._max_particles);
   constexpr auto storage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

   auto positions = c.createBuffer("positions", capacity * vec4_size, storage);
   auto velocities = c.createBuffer("velocities", capacity * vec4_size, storage);

   // emitters can outgrow the buffers in the middle of advance(); then the state on the gpu has to move over
   if (preserved_count > 0 && _positions)
   {
      const Owned<WGPUCommandEncoder> encoder{wgpuDeviceCreateCommandEncoder(c.device(), nullptr)};
      wgpuCommandEncoderCopyBufferToBuffer(encoder.get(), _positions.get(), 0, positions.get(), 0, preserved_count * vec4_size);
      wgpuCommandEncoderCopyBufferToBuffer(encoder.get(), _velocities.get(), 0, velocities.get(), 0, preserved_count * vec4_size);
      const Owned<WGPUCommandBuffer> commands{wgpuCommandEncoderFinish(encoder.get(), nullptr)};
      auto* const command_buffer = commands.get();
      wgpuQueueSubmit(c.queue(), 1, &command_buffer);
   }

   _positions = std::move(positions);
   _velocities = std::move(velocities);
   _sorted_positions = c.createBuffer("sorted positions", capacity * vec4_size, storage);
   _sorted_velocities = c.createBuffer("sorted velocities", capacity * vec4_size, storage);
   _accelerations = c.createBuffer("accelerations", capacity * vec4_size, storage);
   _normals = c.createBuffer("normals", capacity * vec4_size, storage);
   _density_pressure = c.createBuffer("density and pressure", capacity * vec2_size, storage);
   _cell_of_particle = c.createBuffer("cell of particle", capacity * u32_size, storage);
   _rank_in_cell = c.createBuffer("rank in cell", capacity * u32_size, storage);
   _readback = c.createBuffer("readback", capacity * (2 * vec4_size + vec2_size), WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
   _capacity = capacity;
   _bind_groups_dirty = true;
}

void GpuSolver::Impl::uploadParticles(std::size_t first, std::size_t last)
{
   if (first >= last)
   {
      return;
   }
   const auto& c = *_context;
   const auto count = last - first;
   const auto positions = toVec4(std::span{_solver._positions}.subspan(first, count));
   const auto velocities = toVec4(std::span{_solver._velocities}.subspan(first, count));
   wgpuQueueWriteBuffer(c.queue(), _positions.get(), first * vec4_size, positions.data(), count * vec4_size);
   wgpuQueueWriteBuffer(c.queue(), _velocities.get(), first * vec4_size, velocities.data(), count * vec4_size);
}

void GpuSolver::Impl::createBindGroups()
{
   const auto& c = *_context;
   auto* const parameters = _parameters.get();
   _assign_group =
      c.createBindGroup(c._assign_cells, {parameters, _positions.get(), _cell_of_particle.get(), _rank_in_cell.get(), _cell_start.get()});
   _scatter_group = c.createBindGroup(
      c._scatter,
      {parameters,
       _positions.get(),
       _velocities.get(),
       _cell_of_particle.get(),
       _rank_in_cell.get(),
       _cell_start.get(),
       _sorted_positions.get(),
       _sorted_velocities.get()}
   );
   _density_group = c.createBindGroup(c._density, {parameters, _sorted_positions.get(), _cell_start.get(), _density_pressure.get()});
   _normals_group =
      c.createBindGroup(c._normals, {parameters, _sorted_positions.get(), _cell_start.get(), _density_pressure.get(), _normals.get()});
   _forces_group = c.createBindGroup(
      c._forces,
      {parameters,
       _sorted_positions.get(),
       _sorted_velocities.get(),
       _cell_start.get(),
       _density_pressure.get(),
       _normals.get(),
       _fields.get(),
       _accelerations.get(),
       _statistics.get()}
   );
   _integrate_group = c.createBindGroup(
      c._integrate,
      {parameters,
       _sorted_positions.get(),
       _sorted_velocities.get(),
       _accelerations.get(),
       _colliders.get(),
       _positions.get(),
       _velocities.get(),
       _statistics.get()}
   );
   _bind_groups_dirty = false;
}

GpuParams GpuSolver::Impl::makeParams(float time_step) const
{
   const auto& solver = _solver;
   const auto& settings = solver._settings;
   const auto& parameters = solver._parameters;
   const auto& kernels = solver._kernels;
   const auto support_radius = kernels.supportRadius();

   return {
      ._grid_origin = solver._grid_origin,
      ._support_radius = support_radius,
      ._grid_size = _grid_size,
      ._particle_count = static_cast<uint32_t>(_device_count),
      ._gravity = parameters._gravity,
      ._time_step = time_step,
      ._domain_min = settings._domain._min,
      ._particle_radius = solver._particle_radius,
      ._domain_max = settings._domain._max,
      ._mass = solver._particle_mass,
      ._rest_density = settings._rest_density,
      ._stiffness = parameters._speed_of_sound * parameters._speed_of_sound,
      ._viscosity = parameters._viscosity,
      ._artificial_viscosity_factor = parameters._artificial_viscosity * parameters._speed_of_sound * support_radius,
      ._singularity_guard = 0.01f * support_radius * support_radius,
      ._surface_tension = parameters._surface_tension,
      ._wall_restitution = parameters._wall_restitution,
      ._wall_friction = parameters._wall_friction,
      ._poly6_factor = kernels.poly6Factor(),
      ._spiky_gradient_factor = kernels.spikyGradientFactor(),
      ._viscosity_laplacian_factor = kernels.viscosityLaplacianFactor(),
      ._cohesion_factor = kernels.cohesionFactor(),
      ._cohesion_offset = kernels.cohesionOffset(),
      ._clamp_negative_pressure = parameters._clamp_negative_pressure ? 1u : 0u,
      ._collider_count = _collider_count,
      ._field_count = _field_count,
   };
}

void GpuSolver::Impl::dispatch(
   WGPUComputePassEncoder pass,
   const Pipeline& pipeline,
   WGPUBindGroup group,
   uint32_t invocations,
   uint32_t group_size
) const
{
   const auto group_count = (invocations + group_size - 1) / group_size;
   if (group_count == 0)
   {
      return;
   }
   // one dimension holds at most 65535 workgroups; larger dispatches become rows, the shaders flatten them again
   const auto x = std::min(group_count, _context->limits().maxComputeWorkgroupsPerDimension);
   const auto y = (group_count + x - 1) / x;
   wgpuComputePassEncoderSetPipeline(pass, pipeline._pipeline.get());
   wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, nullptr);
   wgpuComputePassEncoderDispatchWorkgroups(pass, x, y, 1);
}

void GpuSolver::Impl::runSubstep(float time_step, float& max_speed)
{
   const auto& c = *_context;
   if (_bind_groups_dirty)
   {
      createBindGroups();
   }

   const auto params = makeParams(time_step);
   wgpuQueueWriteBuffer(c.queue(), _parameters.get(), 0, &params, sizeof(params));

   const Owned<WGPUCommandEncoder> encoder{wgpuDeviceCreateCommandEncoder(c.device(), nullptr)};
   wgpuCommandEncoderClearBuffer(encoder.get(), _cell_start.get(), 0, _cell_start_count * u32_size);
   wgpuCommandEncoderClearBuffer(encoder.get(), _statistics.get(), 0, 4 * u32_size);

   {
      const Owned<WGPUComputePassEncoder> pass{wgpuCommandEncoderBeginComputePass(encoder.get(), nullptr)};
      const auto count = static_cast<uint32_t>(_device_count);

      // 1) + 2) counting sort into the grid
      dispatch(pass.get(), c._assign_cells, _assign_group.get(), count, particle_workgroup_size);
      for (const auto& level : _scan_levels)
      {
         dispatch(pass.get(), c._scan_blocks, level._scan_group.get(), level._count, scan_block_size);
      }
      for (auto index = _scan_levels.size(); index-- > 1;)
      {
         // the block sums of level index - 1 are scanned now, add them to its values
         const auto& level = _scan_levels[index - 1];
         dispatch(pass.get(), c._add_block_offsets, level._add_group.get(), level._count, scan_block_size);
      }
      dispatch(pass.get(), c._scatter, _scatter_group.get(), count, particle_workgroup_size);

      // 3) - 5) the sph terms on the sorted data
      dispatch(pass.get(), c._density, _density_group.get(), count, particle_workgroup_size);
      if (params._surface_tension > 0.0f)
      {
         dispatch(pass.get(), c._normals, _normals_group.get(), count, particle_workgroup_size);
      }
      dispatch(pass.get(), c._forces, _forces_group.get(), count, particle_workgroup_size);
      dispatch(pass.get(), c._integrate, _integrate_group.get(), count, particle_workgroup_size);
      wgpuComputePassEncoderEnd(pass.get());
   }

   wgpuCommandEncoderCopyBufferToBuffer(encoder.get(), _statistics.get(), 0, _statistics_readback.get(), 0, 4 * u32_size);
   const Owned<WGPUCommandBuffer> commands{wgpuCommandEncoderFinish(encoder.get(), nullptr)};
   auto* const command_buffer = commands.get();
   wgpuQueueSubmit(c.queue(), 1, &command_buffer);

   // the next time step depends on these two maxima, so this is the one sync point per substep
   c.mapForReading(_statistics_readback.get(), 4 * u32_size);
   std::array<uint32_t, 4> maxima{};
   std::memcpy(maxima.data(), wgpuBufferGetConstMappedRange(_statistics_readback.get(), 0, sizeof(maxima)), sizeof(maxima));
   wgpuBufferUnmap(_statistics_readback.get());
   _max_acceleration = std::sqrt(std::bit_cast<float>(maxima[0]));
   max_speed = std::sqrt(std::bit_cast<float>(maxima[1]));
}

void GpuSolver::Impl::download(std::size_t density_count)
{
   if (_device_count == 0)
   {
      return;
   }
   const auto& c = *_context;
   const auto velocity_offset = _capacity * vec4_size;
   const auto density_offset = 2 * _capacity * vec4_size;

   const Owned<WGPUCommandEncoder> encoder{wgpuDeviceCreateCommandEncoder(c.device(), nullptr)};
   wgpuCommandEncoderCopyBufferToBuffer(encoder.get(), _positions.get(), 0, _readback.get(), 0, _device_count * vec4_size);
   wgpuCommandEncoderCopyBufferToBuffer(encoder.get(), _velocities.get(), 0, _readback.get(), velocity_offset, _device_count * vec4_size);
   if (density_count > 0)
   {
      wgpuCommandEncoderCopyBufferToBuffer(
         encoder.get(), _density_pressure.get(), 0, _readback.get(), density_offset, density_count * vec2_size
      );
   }
   const Owned<WGPUCommandBuffer> commands{wgpuCommandEncoderFinish(encoder.get(), nullptr)};
   auto* const command_buffer = commands.get();
   wgpuQueueSubmit(c.queue(), 1, &command_buffer);

   const auto readback_size = _capacity * (2 * vec4_size + vec2_size);
   c.mapForReading(_readback.get(), readback_size);
   const auto* const data = static_cast<const float*>(wgpuBufferGetConstMappedRange(_readback.get(), 0, readback_size));
   if (data != nullptr)
   {
      const auto* const velocities = data + velocity_offset / sizeof(float);
      const auto* const densities = data + density_offset / sizeof(float);
      for (auto index = 0uz; index < _device_count; ++index)
      {
         _solver._positions[index] = {data[4 * index], data[4 * index + 1], data[4 * index + 2]};
         _solver._velocities[index] = {velocities[4 * index], velocities[4 * index + 1], velocities[4 * index + 2]};
      }
      // particles emitted during the last substep keep the rest density they were created with, like on the cpu
      for (auto index = 0uz; index < density_count; ++index)
      {
         _solver._densities[index] = densities[2 * index];
      }
   }
   wgpuBufferUnmap(_readback.get());
}

// ----------------------------------------------------------------------------------------------------
// GpuSolver
// ----------------------------------------------------------------------------------------------------

std::unique_ptr<GpuSolver> GpuSolver::create(Solver& solver)
{
   auto context = Context::acquire();
   if (!context)
   {
      return nullptr;
   }

   // the largest buffers must fit into one binding
   const auto& limits = context->limits();
   const auto largest_binding = std::min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
   const auto& grid = solver._grid_size;
   const auto cell_count = static_cast<uint64_t>(grid.x) * grid.y * grid.z + 1;
   if (solver._settings._max_particles * vec4_size > largest_binding || cell_count * u32_size > largest_binding)
   {
      std::println(stderr, "tinysph gpu: {} cannot hold buffers for these settings", context->description());
      return nullptr;
   }

   return std::unique_ptr<GpuSolver>{new GpuSolver{std::make_unique<Impl>(solver, std::move(context))}};
}

GpuSolver::GpuSolver(std::unique_ptr<Impl> impl) : _impl(std::move(impl))
{
}

GpuSolver::~GpuSolver() = default;

StepStatistics GpuSolver::advance(float frame_time, int32_t max_substeps)
{
   return _impl->advance(frame_time, max_substeps);
}

const std::string& GpuSolver::deviceDescription() const
{
   return _impl->_context->description();
}

}  // namespace tinysph::detail
