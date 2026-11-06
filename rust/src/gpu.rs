//! Runs the substeps of a [`Solver`] through wgpu compute shaders, a port of the c++ `src/gpu/gpu_solver.cpp`.
//!
//! The solver keeps owning the particle data on the host, so adding particles, colliders and the rest of the world api
//! work the same for both backends. [`GpuSolver::advance`] uploads the particles, runs all substeps on the gpu and reads
//! the result back. Per substep it only reads back the maximum speed and acceleration for the adaptive time step;
//! emitters run on the host and append their new particles to the gpu buffers.
//!
//! The passes are the wgsl files the c++ library uses (see `build.rs`) and mirror the cpu solver one to one. The only
//! difference is the order of particles inside a grid cell, which depends on the order of atomic operations, so results
//! match the cpu up to float round-off.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::Instant;

use crate::math::{Vec3, max, min};
use crate::solver::Solver;
use crate::types::{ForceField, Shape, StepStatistics};

/// The wgsl source of one shader file; `build.rs` sets `TINYSPH_SHADER_DIR`.
macro_rules! shader {
    ($name:literal) => {
        include_str!(concat!(env!("TINYSPH_SHADER_DIR"), "/", $name, ".wgsl"))
    };
}

const PARTICLE_WORKGROUP_SIZE: u32 = 64; // workgroup_size in common.wgsl
const SCAN_BLOCK_SIZE: u32 = 256; // block_size in scan.wgsl
const VEC4_SIZE: u64 = 16;
const VEC2_SIZE: u64 = 8;
const U32_SIZE: u64 = 4;
const PARAMS_SIZE: u64 = 144; // struct Params in common.wgsl, wgsl uniform layout rules
const STATISTICS_SIZE: u64 = 4 * U32_SIZE;

/// The one place that waits for the gpu. Natively it blocks the calling thread; a web build would await instead.
mod blocking {
    /// Runs a future to completion on the calling thread.
    pub(super) fn wait<F: Future>(future: F) -> F::Output {
        pollster::block_on(future)
    }

    /// Blocks until the submitted work is done.
    pub(super) fn wait_for_queue(device: &wgpu::Device) -> bool {
        device.poll(wgpu::PollType::wait_indefinitely()).is_ok()
    }

    /// Blocks until a buffer mapping requested after `submission` has completed, and returns whether it succeeded.
    ///
    /// Worlds on other threads share the device and poll it too. Any of those polls may pick up this mapping and run
    /// its callback, possibly while this thread's own poll has already returned, so the result comes through a channel
    /// that is waited on, not checked once.
    pub(super) fn wait_for_mapping(
        device: &wgpu::Device,
        submission: wgpu::SubmissionIndex,
        mapping: &std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    ) -> bool {
        if device.poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: None }).is_err() {
            return false;
        }
        // keep polling while waiting, so the callback runs even if no other thread polls again
        loop {
            match mapping.recv_timeout(std::time::Duration::from_millis(1)) {
                Ok(result) => return result.is_ok(),
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return false,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if device.poll(wgpu::PollType::Poll).is_err() {
                        return false;
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// little endian packing of what the shaders read; webgpu buffers are little endian on every platform
// ------------------------------------------------------------------------------------------------

#[derive(Default)]
struct Bytes(Vec<u8>);

impl Bytes {
    fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    fn f32(&mut self, value: f32) -> &mut Self {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }

    fn u32(&mut self, value: u32) -> &mut Self {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }

    fn i32(&mut self, value: i32) -> &mut Self {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }

    fn vec3(&mut self, value: Vec3) -> &mut Self {
        self.f32(value.x).f32(value.y).f32(value.z)
    }

    fn vec4(&mut self, value: Vec3, w: f32) -> &mut Self {
        self.vec3(value).f32(w)
    }

    /// Number of vec4 written so far.
    fn vec4_count(&self) -> usize {
        self.0.len() / VEC4_SIZE as usize
    }
}

fn read_f32(data: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
}

fn read_vec3(data: &[u8], offset: usize) -> Vec3 {
    Vec3::new(read_f32(data, offset), read_f32(data, offset + 4), read_f32(data, offset + 8))
}

fn backend_name(backend: wgpu::Backend) -> &'static str {
    // the names the c++ library prints
    match backend {
        wgpu::Backend::Dx12 => "d3d12",
        wgpu::Backend::Metal => "metal",
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Gl => "opengl",
        _ => "unknown api",
    }
}

// ------------------------------------------------------------------------------------------------
// device and pipelines, shared by all worlds in the process
// ------------------------------------------------------------------------------------------------

struct Pipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: wgpu::Limits,
    description: String,
    error: Arc<AtomicBool>,

    assign_cells: Pipeline,
    scan_blocks: Pipeline,
    add_block_offsets: Pipeline,
    scatter: Pipeline,
    density: Pipeline,
    normals: Pipeline,
    forces: Pipeline,
    integrate: Pipeline,
}

static SHARED_CONTEXT: Mutex<Weak<Context>> = Mutex::new(Weak::new());

impl Context {
    /// The shared context, or `None` if there is no usable gpu.
    fn acquire() -> Option<Arc<Context>> {
        let mut shared = SHARED_CONTEXT.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(context) = shared.upgrade() {
            return Some(context);
        }
        let context = Arc::new(Self::create()?);
        *shared = Arc::downgrade(&context);
        Some(context)
    }

    fn create() -> Option<Context> {
        // vulkan, metal and d3d12 only. the opengl backend would make its own wgl/egl context current on the calling
        // thread, which silently breaks an application that renders with opengl on that thread
        let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_descriptor.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(instance_descriptor);

        // a real gpu first; the fallback is a software rasterizer (warp, lavapipe) on machines without one
        let adapter = [false, true].into_iter().find_map(|fallback| {
            let options = wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: fallback,
                ..wgpu::RequestAdapterOptions::default()
            };
            blocking::wait(instance.request_adapter(&options)).ok()
        })?;

        // ask for everything the adapter can do, the defaults are tuned for the web and too small for big worlds
        let adapter_limits = adapter.limits();

        // the forces pass binds 8 storage buffers, the minimum webgpu guarantees
        if adapter_limits.max_storage_buffers_per_shader_stage < 8 {
            return None;
        }

        let device_descriptor =
            wgpu::DeviceDescriptor { label: Some("tinysph"), required_limits: adapter_limits, ..wgpu::DeviceDescriptor::default() };
        let (device, queue) = match blocking::wait(adapter.request_device(&device_descriptor)) {
            Ok(device_and_queue) => device_and_queue,
            Err(error) => {
                eprintln!("tinysph gpu: no device: {error}");
                return None;
            }
        };
        let error = Arc::new(AtomicBool::new(false));
        let error_flag = Arc::clone(&error);
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            error_flag.store(true, Ordering::Relaxed);
            eprintln!("tinysph gpu error: {error}");
        }));

        let info = adapter.get_info();
        let description = format!("{} ({})", info.name, backend_name(info.backend));

        let common = shader!("common");
        let with_common = |pass: &str| format!("{common}{pass}");
        let create_pipeline = |label: &str, code: &str, entry_point: &str| {
            let module = device
                .create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()) });
            // no explicit layout: webgpu derives it from the bindings the entry point uses
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            let layout = pipeline.get_bind_group_layout(0);
            Pipeline { pipeline, layout }
        };
        let context = Context {
            assign_cells: create_pipeline("assign cells", &with_common(shader!("assign_cells")), "main"),
            scan_blocks: create_pipeline("scan blocks", shader!("scan"), "scan_blocks"),
            add_block_offsets: create_pipeline("add block offsets", shader!("scan"), "add_block_offsets"),
            scatter: create_pipeline("scatter", &with_common(shader!("scatter")), "main"),
            density: create_pipeline("density", &with_common(shader!("density")), "main"),
            normals: create_pipeline("normals", &with_common(shader!("normals")), "main"),
            forces: create_pipeline("forces", &with_common(shader!("forces")), "main"),
            integrate: create_pipeline("integrate", &with_common(shader!("integrate")), "main"),
            limits: device.limits(),
            device,
            queue,
            description,
            error,
        };

        // shader compile errors arrive through the error callback
        blocking::wait_for_queue(&context.device);
        if context.error.load(Ordering::Relaxed) { None } else { Some(context) }
    }

    fn create_buffer(&self, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            // webgpu wants sizes in multiples of 4 and rejects empty bindings
            size: size.div_ceil(4).saturating_mul(4).max(16),
            usage,
            mapped_at_creation: false,
        })
    }

    fn create_bind_group(&self, pipeline: &Pipeline, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
            .iter()
            .zip(0..)
            .map(|(buffer, binding)| wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.layout, entries: &entries })
    }

    fn submit(&self, encoder: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
        self.queue.submit([encoder.finish()])
    }

    /// Maps the first `size` bytes of a readback buffer that `submission` writes, blocks until the gpu has written them
    /// and hands them to `read`.
    fn read_back(&self, submission: wgpu::SubmissionIndex, buffer: &wgpu::Buffer, size: u64, read: impl FnOnce(&[u8])) {
        let (sender, receiver) = mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, ..size, move |result| {
            // the receiver only goes away if waiting failed, then nobody needs the result
            let _ = sender.send(result);
        });
        let mapped = blocking::wait_for_mapping(&self.device, submission, &receiver);
        if !mapped {
            self.report_error("mapping a readback buffer failed");
            return;
        }
        match buffer.get_mapped_range(..size) {
            Ok(view) => read(&view),
            Err(error) => self.report_error(&format!("reading a readback buffer failed: {error}")),
        }
        buffer.unmap();
    }

    fn report_error(&self, message: &str) {
        self.error.store(true, Ordering::Relaxed);
        eprintln!("tinysph gpu error: {message}");
    }
}

// ------------------------------------------------------------------------------------------------
// per world state
// ------------------------------------------------------------------------------------------------

const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST).union(wgpu::BufferUsages::COPY_SRC);
const READBACK: wgpu::BufferUsages = wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST);
const UNIFORM: wgpu::BufferUsages = wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST);

/// Particle buffers; `positions` and `velocities` hold the state between substeps, the rest is per substep.
struct ParticleBuffers {
    positions: wgpu::Buffer,
    velocities: wgpu::Buffer,
    sorted_positions: wgpu::Buffer,
    sorted_velocities: wgpu::Buffer,
    accelerations: wgpu::Buffer,
    normals: wgpu::Buffer,
    density_pressure: wgpu::Buffer,
    cell_of_particle: wgpu::Buffer,
    rank_in_cell: wgpu::Buffer,
    readback: wgpu::Buffer,
}

/// One level of the multi level prefix sum: scans `count` values in blocks of 256 and writes the block totals. The bind
/// groups keep its buffers alive.
struct ScanLevel {
    count: u32,
    scan_group: wgpu::BindGroup,
    add_group: wgpu::BindGroup,
}

/// The uniform grid: where each cell's particles start, and the prefix sum that computes it.
struct Grid {
    size: [i32; 3],
    cell_start_count: u32, // cells + 1
    cell_start: wgpu::Buffer,
    scan_levels: Vec<ScanLevel>,
}

struct BindGroups {
    assign: wgpu::BindGroup,
    scatter: wgpu::BindGroup,
    density: wgpu::BindGroup,
    normals: wgpu::BindGroup,
    forces: wgpu::BindGroup,
    integrate: wgpu::BindGroup,
}

/// Runs the substeps of a [`Solver`] on the gpu, see the module documentation.
pub(crate) struct GpuSolver {
    context: Arc<Context>,

    capacity: usize,
    device_count: usize,
    particles: ParticleBuffers,
    grid: Grid,

    // scene, in vec4
    collider_capacity: usize,
    field_capacity: usize,
    collider_count: u32,
    field_count: u32,
    colliders: wgpu::Buffer,
    fields: wgpu::Buffer,

    parameters: wgpu::Buffer,
    statistics: wgpu::Buffer, // [0] max acceleration^2, [1] max speed^2, as float bits
    statistics_readback: wgpu::Buffer,

    bind_groups: BindGroups,
    bind_groups_dirty: bool,

    max_acceleration: f32,
}

impl GpuSolver {
    /// `None` if no usable gpu is found, or the gpu cannot hold the buffers these settings need.
    pub(crate) fn new(solver: &Solver) -> Option<Self> {
        let context = Context::acquire()?;

        // the largest buffers must fit into one binding
        let limits = &context.limits;
        let largest_binding = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
        let cell_count = solver.grid_size().iter().map(|&cells| cells as u64).product::<u64>() + 1;
        if (solver.settings().max_particles as u64).saturating_mul(VEC4_SIZE) > largest_binding || cell_count * U32_SIZE > largest_binding {
            eprintln!("tinysph gpu: {} cannot hold buffers for these settings", context.description);
            return None;
        }

        let capacity = Self::grown_capacity(solver, 0, 0);
        let particles = Self::create_particle_buffers(&context, capacity);
        let grid = Self::create_grid(&context, solver.grid_size());
        let colliders = context.create_buffer("colliders", 8 * VEC4_SIZE, STORAGE);
        let fields = context.create_buffer("force fields", 8 * VEC4_SIZE, STORAGE);
        let parameters = context.create_buffer("parameters", PARAMS_SIZE, UNIFORM);
        let statistics = context.create_buffer("statistics", STATISTICS_SIZE, STORAGE);
        let statistics_readback = context.create_buffer("statistics readback", STATISTICS_SIZE, READBACK);
        let bind_groups = Self::create_bind_groups(&context, &particles, &grid, &colliders, &fields, &parameters, &statistics);
        Some(Self {
            context,
            capacity,
            device_count: 0,
            particles,
            grid,
            collider_capacity: 8,
            field_capacity: 8,
            collider_count: 0,
            field_count: 0,
            colliders,
            fields,
            parameters,
            statistics,
            statistics_readback,
            bind_groups,
            bind_groups_dirty: false,
            max_acceleration: 0.0,
        })
    }

    /// Adapter name and graphics api, e.g. "NVIDIA GeForce RTX 4070 (vulkan)".
    pub(crate) fn device_description(&self) -> &str {
        &self.context.description
    }

    /// Like [`Solver::advance`].
    pub(crate) fn advance(&mut self, solver: &mut Solver, frame_time: f32, max_substeps: u32) -> StepStatistics {
        let start_time = Instant::now();

        self.sync_grid(solver);
        self.upload_scene(solver);

        let particle_count = solver.particle_count();
        self.ensure_capacity(solver, particle_count, 0);
        self.upload_particles(solver, 0, particle_count);
        self.device_count = particle_count;
        if particle_count == 0 {
            self.max_acceleration = 0.0;
        }

        let mut max_speed = solver.max_speed();
        let mut density_count = 0; // particles whose density the gpu has computed
        let mut statistics = StepStatistics::default();
        let mut remaining_time = frame_time;
        while remaining_time > 1e-7 && statistics.substeps < max_substeps {
            let time_step = min(solver.compute_time_step_for(max_speed, self.max_acceleration), remaining_time);
            if self.device_count > 0 {
                max_speed = self.run_substep(solver, time_step);
                density_count = self.device_count;
            }

            // emitters run on the host and append to the gpu buffers
            solver.run_emitters(time_step);
            let new_count = solver.particle_count();
            if new_count > self.device_count {
                self.ensure_capacity(solver, new_count, self.device_count);
                self.upload_particles(solver, self.device_count, new_count);
                for velocity in &solver.velocities()[self.device_count..new_count] {
                    max_speed = max(max_speed, velocity.length());
                }
                self.device_count = new_count;
            }

            remaining_time -= time_step;
            statistics.simulated_time += time_step;
            statistics.time_step = time_step;
            statistics.substeps += 1;
        }

        self.download(solver, density_count);
        solver.fill_statistics(&mut statistics);
        statistics.wall_time = start_time.elapsed();
        statistics
    }

    fn create_grid(context: &Context, size: [i32; 3]) -> Grid {
        let cell_start_count = (size[0] * size[1] * size[2] + 1) as u32;
        let cell_start = context.create_buffer("cell start", u64::from(cell_start_count) * U32_SIZE, STORAGE);

        // levels until one block covers everything: 256 cells per block, 65536 per two levels, 16.7 million per three
        let mut levels: Vec<(u32, wgpu::Buffer, wgpu::Buffer)> = Vec::new();
        let mut count = cell_start_count;
        loop {
            let block_count = count.div_ceil(SCAN_BLOCK_SIZE);
            let parameters = context.create_buffer("scan parameters", 16, UNIFORM);
            let mut values = Bytes::with_capacity(16);
            values.u32(count).u32(0).u32(0).u32(0);
            context.queue.write_buffer(&parameters, 0, &values.0);
            let block_sums = context.create_buffer("scan block sums", u64::from(block_count) * U32_SIZE, STORAGE);
            levels.push((count, parameters, block_sums));
            if block_count <= 1 {
                break;
            }
            count = block_count;
        }

        // level n scans the block sums of level n - 1
        let mut scan_levels = Vec::with_capacity(levels.len());
        for index in 0..levels.len() {
            let values = if index == 0 { &cell_start } else { &levels[index - 1].2 };
            let (count, parameters, block_sums) = &levels[index];
            scan_levels.push(ScanLevel {
                count: *count,
                scan_group: context.create_bind_group(&context.scan_blocks, &[parameters, values, block_sums]),
                add_group: context.create_bind_group(&context.add_block_offsets, &[parameters, values, block_sums]),
            });
        }
        Grid { size, cell_start_count, cell_start, scan_levels }
    }

    fn sync_grid(&mut self, solver: &Solver) {
        let size = solver.grid_size();
        if size != self.grid.size {
            self.grid = Self::create_grid(&self.context, size);
            self.bind_groups_dirty = true;
        }
    }

    fn upload_scene(&mut self, solver: &Solver) {
        // two vec4 per collider: center + kind, then radius or half extents
        let mut colliders = Bytes::default();
        for collider in &solver.colliders {
            match collider {
                Shape::Sphere(sphere) => colliders.vec4(sphere.center, 0.0).vec4(Vec3::new(sphere.radius, 0.0, 0.0), 0.0),
                Shape::Cuboid(cuboid) => colliders.vec4(cuboid.center, 1.0).vec4(cuboid.half_extents, 0.0),
            };
        }

        // three vec4 per field, the kind in the first w
        let mut fields = Bytes::default();
        for field in &solver.fields {
            match field {
                ForceField::Attractor(attractor) => fields
                    .vec4(attractor.position, 0.0)
                    .vec4(Vec3::new(attractor.strength, attractor.radius, 0.0), 0.0)
                    .vec4(Vec3::ZERO, 0.0),
                ForceField::Vortex(vortex) => fields
                    .vec4(vortex.position, 1.0)
                    .vec4(vortex.axis.normalize(), vortex.strength)
                    .vec4(Vec3::new(vortex.radius, 0.0, 0.0), 0.0),
                ForceField::Wind(wind) => fields.vec4(wind.region.min, 2.0).vec4(wind.region.max, 0.0).vec4(wind.acceleration, 0.0),
            };
        }

        let context = &self.context;
        let mut upload = |values: &Bytes, capacity: &mut usize, buffer: &mut wgpu::Buffer, label: &str| {
            if values.vec4_count() > *capacity {
                *capacity = values.vec4_count().max(*capacity * 2).max(8);
                *buffer = context.create_buffer(label, *capacity as u64 * VEC4_SIZE, STORAGE);
                self.bind_groups_dirty = true;
            }
            if !values.0.is_empty() {
                context.queue.write_buffer(buffer, 0, &values.0);
            }
        };
        upload(&colliders, &mut self.collider_capacity, &mut self.colliders, "colliders");
        upload(&fields, &mut self.field_capacity, &mut self.fields, "force fields");
        self.collider_count = solver.colliders.len() as u32;
        self.field_count = solver.fields.len() as u32;
    }

    fn grown_capacity(solver: &Solver, required: usize, capacity: usize) -> usize {
        required.max(capacity * 2).max(1024).min(solver.settings().max_particles)
    }

    fn create_particle_buffers(context: &Context, capacity: usize) -> ParticleBuffers {
        let capacity = capacity as u64;
        ParticleBuffers {
            positions: context.create_buffer("positions", capacity * VEC4_SIZE, STORAGE),
            velocities: context.create_buffer("velocities", capacity * VEC4_SIZE, STORAGE),
            sorted_positions: context.create_buffer("sorted positions", capacity * VEC4_SIZE, STORAGE),
            sorted_velocities: context.create_buffer("sorted velocities", capacity * VEC4_SIZE, STORAGE),
            accelerations: context.create_buffer("accelerations", capacity * VEC4_SIZE, STORAGE),
            normals: context.create_buffer("normals", capacity * VEC4_SIZE, STORAGE),
            density_pressure: context.create_buffer("density and pressure", capacity * VEC2_SIZE, STORAGE),
            cell_of_particle: context.create_buffer("cell of particle", capacity * U32_SIZE, STORAGE),
            rank_in_cell: context.create_buffer("rank in cell", capacity * U32_SIZE, STORAGE),
            readback: context.create_buffer("readback", capacity * (2 * VEC4_SIZE + VEC2_SIZE), READBACK),
        }
    }

    fn ensure_capacity(&mut self, solver: &Solver, required: usize, preserved_count: usize) {
        if required <= self.capacity {
            return;
        }
        let capacity = Self::grown_capacity(solver, required, self.capacity);
        let particles = Self::create_particle_buffers(&self.context, capacity);

        // emitters can outgrow the buffers in the middle of advance(); then the state on the gpu has to move over
        if preserved_count > 0 {
            let size = preserved_count as u64 * VEC4_SIZE;
            let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_buffer(&self.particles.positions, 0, &particles.positions, 0, size);
            encoder.copy_buffer_to_buffer(&self.particles.velocities, 0, &particles.velocities, 0, size);
            let _ = self.context.submit(encoder);
        }

        self.particles = particles;
        self.capacity = capacity;
        self.bind_groups_dirty = true;
    }

    fn upload_particles(&self, solver: &Solver, first: usize, last: usize) {
        if first >= last {
            return;
        }
        let mut positions = Bytes::with_capacity((last - first) * VEC4_SIZE as usize);
        for &position in &solver.positions()[first..last] {
            positions.vec4(position, 0.0);
        }
        let mut velocities = Bytes::with_capacity((last - first) * VEC4_SIZE as usize);
        for &velocity in &solver.velocities()[first..last] {
            velocities.vec4(velocity, 0.0);
        }
        let offset = first as u64 * VEC4_SIZE;
        self.context.queue.write_buffer(&self.particles.positions, offset, &positions.0);
        self.context.queue.write_buffer(&self.particles.velocities, offset, &velocities.0);
    }

    fn create_bind_groups(
        context: &Context,
        particles: &ParticleBuffers,
        grid: &Grid,
        colliders: &wgpu::Buffer,
        fields: &wgpu::Buffer,
        parameters: &wgpu::Buffer,
        statistics: &wgpu::Buffer,
    ) -> BindGroups {
        let p = particles;
        let cell_start = &grid.cell_start;
        BindGroups {
            assign: context
                .create_bind_group(&context.assign_cells, &[parameters, &p.positions, &p.cell_of_particle, &p.rank_in_cell, cell_start]),
            scatter: context.create_bind_group(
                &context.scatter,
                &[
                    parameters,
                    &p.positions,
                    &p.velocities,
                    &p.cell_of_particle,
                    &p.rank_in_cell,
                    cell_start,
                    &p.sorted_positions,
                    &p.sorted_velocities,
                ],
            ),
            density: context.create_bind_group(&context.density, &[parameters, &p.sorted_positions, cell_start, &p.density_pressure]),
            normals: context
                .create_bind_group(&context.normals, &[parameters, &p.sorted_positions, cell_start, &p.density_pressure, &p.normals]),
            forces: context.create_bind_group(
                &context.forces,
                &[
                    parameters,
                    &p.sorted_positions,
                    &p.sorted_velocities,
                    cell_start,
                    &p.density_pressure,
                    &p.normals,
                    fields,
                    &p.accelerations,
                    statistics,
                ],
            ),
            integrate: context.create_bind_group(
                &context.integrate,
                &[
                    parameters,
                    &p.sorted_positions,
                    &p.sorted_velocities,
                    &p.accelerations,
                    colliders,
                    &p.positions,
                    &p.velocities,
                    statistics,
                ],
            ),
        }
    }

    /// The `Params` uniform of common.wgsl, 144 bytes.
    fn make_params(&self, solver: &Solver, time_step: f32) -> Bytes {
        let settings = solver.settings();
        let parameters = solver.parameters();
        let kernels = solver.kernels();
        let support_radius = kernels.support_radius();
        let [grid_x, grid_y, grid_z] = self.grid.size;

        let mut params = Bytes::with_capacity(PARAMS_SIZE as usize);
        params
            .vec3(solver.grid_origin())
            .f32(support_radius)
            .i32(grid_x)
            .i32(grid_y)
            .i32(grid_z)
            .u32(self.device_count as u32)
            .vec3(parameters.gravity)
            .f32(time_step)
            .vec3(settings.domain.min)
            .f32(solver.particle_radius())
            .vec3(settings.domain.max)
            .f32(solver.particle_mass())
            .f32(settings.rest_density)
            .f32(parameters.speed_of_sound * parameters.speed_of_sound)
            .f32(parameters.viscosity)
            .f32(parameters.artificial_viscosity * parameters.speed_of_sound * support_radius)
            .f32(0.01 * support_radius * support_radius)
            .f32(parameters.surface_tension)
            .f32(parameters.wall_restitution)
            .f32(parameters.wall_friction)
            .f32(kernels.poly6_factor)
            .f32(kernels.spiky_gradient_factor)
            .f32(kernels.viscosity_laplacian_factor)
            .f32(kernels.cohesion_factor)
            .f32(kernels.cohesion_offset)
            .u32(u32::from(parameters.clamp_negative_pressure))
            .u32(self.collider_count)
            .u32(self.field_count);
        debug_assert_eq!(params.0.len() as u64, PARAMS_SIZE, "the layout must match struct Params in common.wgsl");
        params
    }

    fn dispatch(&self, pass: &mut wgpu::ComputePass<'_>, pipeline: &Pipeline, group: &wgpu::BindGroup, invocations: u32, group_size: u32) {
        let group_count = invocations.div_ceil(group_size);
        if group_count == 0 {
            return;
        }
        // one dimension holds at most 65535 workgroups; larger dispatches become rows, the shaders flatten them again
        let x = group_count.min(self.context.limits.max_compute_workgroups_per_dimension);
        let y = group_count.div_ceil(x);
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }

    /// Runs one substep on the gpu and returns the new maximum speed.
    fn run_substep(&mut self, solver: &Solver, time_step: f32) -> f32 {
        if self.bind_groups_dirty {
            self.bind_groups = Self::create_bind_groups(
                &self.context,
                &self.particles,
                &self.grid,
                &self.colliders,
                &self.fields,
                &self.parameters,
                &self.statistics,
            );
            self.bind_groups_dirty = false;
        }

        let context = &self.context;
        let params = self.make_params(solver, time_step);
        context.queue.write_buffer(&self.parameters, 0, &params.0);

        let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.clear_buffer(&self.grid.cell_start, 0, Some(u64::from(self.grid.cell_start_count) * U32_SIZE));
        encoder.clear_buffer(&self.statistics, 0, Some(STATISTICS_SIZE));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let count = self.device_count as u32;
            let groups = &self.bind_groups;

            // 1) + 2) counting sort into the grid
            self.dispatch(&mut pass, &context.assign_cells, &groups.assign, count, PARTICLE_WORKGROUP_SIZE);
            for level in &self.grid.scan_levels {
                self.dispatch(&mut pass, &context.scan_blocks, &level.scan_group, level.count, SCAN_BLOCK_SIZE);
            }
            for index in (1..self.grid.scan_levels.len()).rev() {
                // the block sums of level index - 1 are scanned now, add them to its values
                let level = &self.grid.scan_levels[index - 1];
                self.dispatch(&mut pass, &context.add_block_offsets, &level.add_group, level.count, SCAN_BLOCK_SIZE);
            }
            self.dispatch(&mut pass, &context.scatter, &groups.scatter, count, PARTICLE_WORKGROUP_SIZE);

            // 3) - 5) the sph terms on the sorted data
            self.dispatch(&mut pass, &context.density, &groups.density, count, PARTICLE_WORKGROUP_SIZE);
            if solver.parameters().surface_tension > 0.0 {
                self.dispatch(&mut pass, &context.normals, &groups.normals, count, PARTICLE_WORKGROUP_SIZE);
            }
            self.dispatch(&mut pass, &context.forces, &groups.forces, count, PARTICLE_WORKGROUP_SIZE);
            self.dispatch(&mut pass, &context.integrate, &groups.integrate, count, PARTICLE_WORKGROUP_SIZE);
        }
        encoder.copy_buffer_to_buffer(&self.statistics, 0, &self.statistics_readback, 0, STATISTICS_SIZE);
        let submission = context.submit(encoder);

        // the next time step depends on these two maxima, so this is the one sync point per substep
        let mut maxima = [0.0f32; 2];
        context.read_back(submission, &self.statistics_readback, STATISTICS_SIZE, |data| {
            maxima = [read_f32(data, 0), read_f32(data, 4)];
        });
        self.max_acceleration = maxima[0].sqrt();
        maxima[1].sqrt()
    }

    /// Copies positions, velocities and the densities of the first `density_count` particles back to the solver.
    fn download(&self, solver: &mut Solver, density_count: usize) {
        if self.device_count == 0 {
            return;
        }
        let context = &self.context;
        let particles = &self.particles;
        let device_count = self.device_count;
        let velocity_offset = self.capacity as u64 * VEC4_SIZE;
        let density_offset = 2 * self.capacity as u64 * VEC4_SIZE;

        let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let vec4_bytes = device_count as u64 * VEC4_SIZE;
        encoder.copy_buffer_to_buffer(&particles.positions, 0, &particles.readback, 0, vec4_bytes);
        encoder.copy_buffer_to_buffer(&particles.velocities, 0, &particles.readback, velocity_offset, vec4_bytes);
        if density_count > 0 {
            let density_bytes = density_count as u64 * VEC2_SIZE;
            encoder.copy_buffer_to_buffer(&particles.density_pressure, 0, &particles.readback, density_offset, density_bytes);
        }
        let submission = context.submit(encoder);

        let readback_size = self.capacity as u64 * (2 * VEC4_SIZE + VEC2_SIZE);
        let (positions, velocities, densities) = solver.particles_mut();
        context.read_back(submission, &particles.readback, readback_size, |data| {
            let (velocity_offset, density_offset) = (velocity_offset as usize, density_offset as usize);
            for index in 0..device_count {
                positions[index] = read_vec3(data, 16 * index);
                velocities[index] = read_vec3(data, velocity_offset + 16 * index);
            }
            // particles emitted during the last substep keep the rest density they were created with, like on the cpu
            for (index, density) in densities[..density_count].iter_mut().enumerate() {
                *density = read_f32(data, density_offset + 8 * index);
            }
        });
    }
}
