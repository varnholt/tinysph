//! The renderer of the c++ sandbox (sandbox/src/renderer.cpp) on wgpu: sky, checkered floor, lit colliders, helper lines and
//! the fluid as sphere impostors or as a screen space surface (depth, thickness, bilateral smoothing, normals, fresnel blend
//! of reflection and refraction), from particle sprites or from a marching tetrahedra mesh.
//!
//! All passes run offscreen in `prepare`, on the encoder egui hands out; `paint` only copies the result into egui's pass.
//! Formats stay within plain WebGPU: R32Float is only loaded, never filtered or blended; thickness is blended in R16Float.

use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::{self, wgpu};
use glam::{Mat4, Vec3 as V3};
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("shaders.wgsl");
const BLIT_SHADER: &str = r"
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
struct Out { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) index: u32) -> Out {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return Out(vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0), vec2<f32>(p.x, 1.0 - p.y));
}
fn color(in: Out) -> vec3<f32> { return textureSampleLevel(source, source_sampler, in.uv, 0.0).rgb; }
@fragment fn fs(in: Out) -> @location(0) vec4<f32> { return vec4<f32>(color(in), 1.0); }
// an srgb target encodes again what the scene already wrote as display values
@fragment fn fs_srgb(in: Out) -> @location(0) vec4<f32> {
    let c = color(in);
    return vec4<f32>(select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045)), 1.0);
}
";

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const VALUE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const THICKNESS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
pub const NEAR: f32 = 0.01;
pub const FAR: f32 = 100.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderMode {
    Spheres,
    FluidSurface,
    MeshSurface,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    Speed,
    Flat,
}

/// The c++ `RenderSettings`.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub mode: RenderMode,
    pub show_helpers: bool,
    pub show_floor: bool,
    pub color_mode: ColorMode,
    /// 1 = particles touch at rest spacing
    pub radius_scale: f32,
    /// speed in m/s that maps to the top of the colour ramp
    pub speed_range: f32,
    /// bilateral filter passes; more = smoother hull
    pub smoothing_iterations: u32,
    /// filter radius in particle radii
    pub smoothing_radius: f32,
    /// screen space offset per meter of thickness
    pub refraction_strength: f32,
    /// colour deep water scatters towards
    pub fluid_color: [f32; 3],
    /// beer-lambert coefficients per meter, red is absorbed first
    pub absorption: [f32; 3],
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            mode: RenderMode::MeshSurface,
            show_helpers: true,
            show_floor: true,
            color_mode: ColorMode::Speed,
            radius_scale: 1.0,
            speed_range: 3.0,
            smoothing_iterations: 3,
            smoothing_radius: 4.0,
            refraction_strength: 0.15,
            fluid_color: [0.03, 0.22, 0.38],
            absorption: [3.5, 1.1, 0.55],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
    view_projection: [[f32; 4]; 4],
    view_to_world: [[f32; 4]; 4],
    eye: [f32; 4],
    light_view: [f32; 4],
    camera: [f32; 4],
    particle: [f32; 4],
    screen: [f32; 4],
    fluid_color: [f32; 4],
    absorption: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SmoothUniforms {
    direction: [f32; 2],
    world_radius: f32,
    pixels_per_meter: f32,
    depth_falloff: f32,
    pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

/// Floor and collider triangles; `checker` 1 for the floor.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub checker: f32,
}

/// One frame's worth of drawing.
pub struct Frame {
    pub view: Mat4,
    pub fov_y: f32,
    pub eye: V3,
    /// size of the viewport in pixels
    pub width: u32,
    pub height: u32,
    pub particle_radius: f32,
    pub settings: Settings,
    /// xyz position, w speed
    pub particles: Vec<[f32; 4]>,
    pub lines: Vec<LineVertex>,
    pub meshes: Vec<MeshVertex>,
    /// the mesh surface, three positions per triangle
    pub surface: Vec<[f32; 3]>,
}

impl Frame {
    pub fn projection(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.width as f32 / self.height.max(1) as f32, NEAR, FAR)
    }
}

struct Buffer {
    buffer: wgpu::Buffer,
    capacity: u64,
    count: u32,
}

impl Buffer {
    fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Buffer { buffer, capacity: 64, count: 0 }
    }

    fn upload<T: Pod>(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[T]) {
        self.count = data.len() as u32;
        let bytes: &[u8] = bytemuck::cast_slice(data);
        if bytes.is_empty() {
            return;
        }
        if bytes.len() as u64 > self.capacity {
            self.buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            self.capacity = bytes.len() as u64;
        } else {
            queue.write_buffer(&self.buffer, 0, bytes);
        }
    }
}

/// The render targets at the size of the viewport and the bind groups that read them.
struct Targets {
    width: u32,
    height: u32,
    scene_color: wgpu::TextureView,
    scene_depth: wgpu::TextureView,
    fluid_depth: wgpu::TextureView,
    fluid_depth_test: wgpu::TextureView,
    thickness: wgpu::TextureView,
    smooth: [wgpu::TextureView; 2],
    output_color: wgpu::TextureView,
    output_depth: wgpu::TextureView,
    /// fluid depth -> smooth 0 (x), smooth 0 -> smooth 1 (y), smooth 1 -> smooth 0 (x)
    smooth_groups: [wgpu::BindGroup; 3],
    /// smoothed or raw depth
    composite_groups: [wgpu::BindGroup; 2],
    surface_thickness_group: wgpu::BindGroup,
    /// scene, output
    blit_groups: [wgpu::BindGroup; 2],
}

struct Pipelines {
    sky: wgpu::RenderPipeline,
    mesh: wgpu::RenderPipeline,
    sphere: wgpu::RenderPipeline,
    line: wgpu::RenderPipeline,
    fluid_depth: wgpu::RenderPipeline,
    fluid_thickness: wgpu::RenderPipeline,
    surface_depth: wgpu::RenderPipeline,
    surface_thickness: wgpu::RenderPipeline,
    smooth: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    blit: wgpu::RenderPipeline,
}

/// The gpu objects, kept in egui_wgpu's callback resources.
struct Resources {
    pipelines: Pipelines,
    uniforms: wgpu::Buffer,
    frame_group: wgpu::BindGroup,
    /// x and y direction
    smooth_uniforms: [wgpu::Buffer; 2],
    smooth_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    surface_thickness_layout: wgpu::BindGroupLayout,
    blit_layout: wgpu::BindGroupLayout,
    linear_sampler: wgpu::Sampler,
    particles: Buffer,
    lines: Buffer,
    meshes: Buffer,
    surface: Buffer,
    targets: Option<Targets>,
    /// what paint copies: false = scene target, true = output target
    show_output: bool,
}

fn texture_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

const UNFILTERED: wgpu::TextureSampleType = wgpu::TextureSampleType::Float { filterable: false };
const FILTERED: wgpu::TextureSampleType = wgpu::TextureSampleType::Float { filterable: true };

fn depth_state(write: bool, compare: wgpu::CompareFunction) -> Option<wgpu::DepthStencilState> {
    Some(wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(write),
        depth_compare: Some(compare),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    })
}

pub fn init(render_state: &egui_wgpu::RenderState) {
    let device = &render_state.device;
    let module = device
        .create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("sandbox"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
    let blit_module = device
        .create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("blit"), source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()) });
    let layout_of =
        |entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries });
    let frame_layout = layout_of(&[uniform_entry(0)]);
    let smooth_layout = layout_of(&[texture_entry(0, UNFILTERED), uniform_entry(1)]);
    let composite_layout = layout_of(&[
        texture_entry(2, UNFILTERED),
        texture_entry(3, UNFILTERED),
        texture_entry(4, FILTERED),
        sampler_entry(5),
        texture_entry(6, wgpu::TextureSampleType::Depth),
    ]);
    let surface_thickness_layout = layout_of(&[texture_entry(7, wgpu::TextureSampleType::Depth)]);
    let blit_layout = layout_of(&[texture_entry(0, FILTERED), sampler_entry(1)]);

    let pipeline_layout = |groups: &[&wgpu::BindGroupLayout]| {
        let groups: Vec<_> = groups.iter().map(|group| Some(*group)).collect();
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &groups, immediate_size: 0 })
    };
    let frame_only = pipeline_layout(&[&frame_layout]);
    let with_smooth = pipeline_layout(&[&frame_layout, &smooth_layout]);
    let with_composite = pipeline_layout(&[&frame_layout, &composite_layout]);
    let with_scene_depth = pipeline_layout(&[&frame_layout, &surface_thickness_layout]);
    let blit_only = pipeline_layout(&[&blit_layout]);

    struct Description<'a> {
        layout: &'a wgpu::PipelineLayout,
        vs: &'a str,
        fs: &'a str,
        buffers: &'a [Option<wgpu::VertexBufferLayout<'a>>],
        topology: wgpu::PrimitiveTopology,
        format: wgpu::TextureFormat,
        blend: Option<wgpu::BlendState>,
        depth: Option<wgpu::DepthStencilState>,
    }
    let create = |module: &wgpu::ShaderModule, d: Description<'_>| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(d.fs),
            layout: Some(d.layout),
            vertex: wgpu::VertexState { module, entry_point: Some(d.vs), buffers: d.buffers, compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some(d.fs),
                targets: &[Some(wgpu::ColorTargetState { format: d.format, blend: d.blend, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { topology: d.topology, ..Default::default() },
            depth_stencil: d.depth,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    };
    use wgpu::CompareFunction::{Always, Less};
    use wgpu::PrimitiveTopology::{LineList, TriangleList, TriangleStrip};
    let particle_buffer = [Some(wgpu::VertexBufferLayout {
        array_stride: 16,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![0 => Float32x4],
    })];
    let mesh_buffer = [Some(wgpu::VertexBufferLayout {
        array_stride: 40,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3, 3 => Float32],
    })];
    let line_buffer = [Some(wgpu::VertexBufferLayout {
        array_stride: 24,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
    })];
    let surface_buffer = [Some(wgpu::VertexBufferLayout {
        array_stride: 12,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3],
    })];
    let additive = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
    };
    let base = |vs, fs, buffers, topology, format, depth| Description {
        layout: &frame_only,
        vs,
        fs,
        buffers,
        topology,
        format,
        blend: None,
        depth,
    };
    let pipelines = Pipelines {
        sky: create(&module, base("fullscreen_vs", "sky_fs", &[], TriangleList, COLOR_FORMAT, depth_state(false, Always))),
        mesh: create(&module, base("mesh_vs", "mesh_fs", &mesh_buffer, TriangleList, COLOR_FORMAT, depth_state(true, Less))),
        sphere: create(&module, base("sphere_vs", "sphere_fs", &particle_buffer, TriangleStrip, COLOR_FORMAT, depth_state(true, Less))),
        line: create(&module, base("line_vs", "line_fs", &line_buffer, LineList, COLOR_FORMAT, depth_state(true, Less))),
        fluid_depth: create(
            &module,
            base("fluid_vs", "fluid_depth_fs", &particle_buffer, TriangleStrip, VALUE_FORMAT, depth_state(true, Less)),
        ),
        // depth tested against the scene without writing, so that fluid behind a collider does not count
        fluid_thickness: create(
            &module,
            Description {
                blend: Some(additive),
                ..base("fluid_vs", "fluid_thickness_fs", &particle_buffer, TriangleStrip, THICKNESS_FORMAT, depth_state(false, Less))
            },
        ),
        surface_depth: create(
            &module,
            base("surface_vs", "surface_depth_fs", &surface_buffer, TriangleList, VALUE_FORMAT, depth_state(true, Less)),
        ),
        surface_thickness: create(
            &module,
            Description {
                layout: &with_scene_depth,
                blend: Some(additive),
                ..base("surface_vs", "surface_thickness_fs", &surface_buffer, TriangleList, THICKNESS_FORMAT, None)
            },
        ),
        smooth: create(
            &module,
            Description { layout: &with_smooth, ..base("fullscreen_vs", "smooth_fs", &[], TriangleList, VALUE_FORMAT, None) },
        ),
        composite: create(
            &module,
            Description {
                layout: &with_composite,
                ..base("fullscreen_vs", "composite_fs", &[], TriangleList, COLOR_FORMAT, depth_state(true, Always))
            },
        ),
        blit: create(
            &blit_module,
            Description {
                layout: &blit_only,
                ..base(
                    "vs",
                    if render_state.target_format.is_srgb() { "fs_srgb" } else { "fs" },
                    &[],
                    TriangleList,
                    render_state.target_format,
                    None,
                )
            },
        ),
    };

    let uniform_buffer = |size: usize| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let uniforms = uniform_buffer(size_of::<Uniforms>());
    let frame_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &frame_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }],
    });
    let resources = Resources {
        pipelines,
        uniforms,
        frame_group,
        smooth_uniforms: [uniform_buffer(size_of::<SmoothUniforms>()), uniform_buffer(size_of::<SmoothUniforms>())],
        smooth_layout,
        composite_layout,
        surface_thickness_layout,
        blit_layout,
        linear_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }),
        particles: Buffer::new(device),
        lines: Buffer::new(device),
        meshes: Buffer::new(device),
        surface: Buffer::new(device),
        targets: None,
        show_output: false,
    };
    render_state.renderer.write().callback_resources.insert(resources);
}

impl Resources {
    fn targets(&mut self, device: &wgpu::Device, width: u32, height: u32) -> &Targets {
        if self.targets.as_ref().is_none_or(|targets| targets.width != width || targets.height != height) {
            self.targets = Some(self.create_targets(device, width, height));
        }
        self.targets.as_ref().unwrap()
    }

    fn create_targets(&self, device: &wgpu::Device, width: u32, height: u32) -> Targets {
        let texture = |format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let scene_color = texture(COLOR_FORMAT);
        let scene_depth = texture(DEPTH_FORMAT);
        let fluid_depth = texture(VALUE_FORMAT);
        let thickness = texture(THICKNESS_FORMAT);
        let smooth = [texture(VALUE_FORMAT), texture(VALUE_FORMAT)];
        let output_color = texture(COLOR_FORMAT);
        let group = |layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry<'_>]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries })
        };
        let view = |binding, view| wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(view) };
        let smooth_group = |source, direction: usize| {
            group(
                &self.smooth_layout,
                &[view(0, source), wgpu::BindGroupEntry { binding: 1, resource: self.smooth_uniforms[direction].as_entire_binding() }],
            )
        };
        let sampler = |binding| wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::Sampler(&self.linear_sampler) };
        let composite_group = |depth| {
            group(&self.composite_layout, &[view(2, depth), view(3, &thickness), view(4, &scene_color), sampler(5), view(6, &scene_depth)])
        };
        Targets {
            width,
            height,
            smooth_groups: [smooth_group(&fluid_depth, 0), smooth_group(&smooth[0], 1), smooth_group(&smooth[1], 0)],
            composite_groups: [composite_group(&smooth[1]), composite_group(&fluid_depth)],
            surface_thickness_group: group(&self.surface_thickness_layout, &[view(7, &scene_depth)]),
            blit_groups: [
                group(&self.blit_layout, &[view(0, &scene_color), sampler(1)]),
                group(&self.blit_layout, &[view(0, &output_color), sampler(1)]),
            ],
            fluid_depth_test: texture(DEPTH_FORMAT),
            output_depth: texture(DEPTH_FORMAT),
            scene_color,
            scene_depth,
            fluid_depth,
            thickness,
            smooth,
            output_color,
        }
    }
}

fn color_attachment(view: &wgpu::TextureView, clear: Option<wgpu::Color>) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations { load: clear.map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear), store: wgpu::StoreOp::Store },
    })
}

fn depth_attachment(view: &wgpu::TextureView, clear: bool) -> Option<wgpu::RenderPassDepthStencilAttachment<'_>> {
    Some(wgpu::RenderPassDepthStencilAttachment {
        view,
        depth_ops: Some(wgpu::Operations {
            load: if clear { wgpu::LoadOp::Clear(1.0) } else { wgpu::LoadOp::Load },
            store: wgpu::StoreOp::Store,
        }),
        stencil_ops: None,
    })
}

fn begin_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    label: &str,
    color: Option<wgpu::RenderPassColorAttachment<'a>>,
    depth: Option<wgpu::RenderPassDepthStencilAttachment<'a>>,
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[color],
        depth_stencil_attachment: depth,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

const CLEAR: Option<wgpu::Color> = Some(wgpu::Color::TRANSPARENT);

impl egui_wgpu::CallbackTrait for Frame {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(r) = resources.get_mut::<Resources>() else { return Vec::new() };
        if self.width == 0 || self.height == 0 {
            return Vec::new();
        }
        let settings = &self.settings;
        let aspect = self.width as f32 / self.height as f32;
        let projection = self.projection();
        let view_to_world = Mat4::from_mat3(glam::Mat3::from_mat4(self.view).transpose());
        let light_view = glam::Mat3::from_mat4(self.view) * V3::new(0.3904, 0.7807, 0.4880);
        let tan_half_fov = (self.fov_y * 0.5).tan();
        let uniforms = Uniforms {
            view: self.view.to_cols_array_2d(),
            projection: projection.to_cols_array_2d(),
            view_projection: (projection * self.view).to_cols_array_2d(),
            view_to_world: view_to_world.to_cols_array_2d(),
            eye: self.eye.extend(1.0).into(),
            light_view: light_view.extend(0.0).into(),
            camera: [tan_half_fov, aspect, NEAR, FAR],
            particle: [
                self.particle_radius * settings.radius_scale,
                self.particle_radius,
                settings.speed_range,
                if settings.color_mode == ColorMode::Flat { 2.0 } else { 0.0 },
            ],
            screen: [self.width as f32, self.height as f32, settings.refraction_strength, 0.0],
            fluid_color: [settings.fluid_color[0], settings.fluid_color[1], settings.fluid_color[2], 0.0],
            absorption: [settings.absorption[0], settings.absorption[1], settings.absorption[2], 0.0],
        };
        queue.write_buffer(&r.uniforms, 0, bytemuck::bytes_of(&uniforms));
        let mesh_surface = settings.mode == RenderMode::MeshSurface;
        let fluid_surface = settings.mode != RenderMode::Spheres;
        // the mesh is already one smooth surface; it only gets a light pass against the faceting of the triangles
        let world_radius = self.particle_radius * if mesh_surface { 1.0 } else { settings.smoothing_radius };
        let iterations = if mesh_surface { 1 } else { settings.smoothing_iterations };
        for (direction, buffer) in [[1.0 / self.width as f32, 0.0], [0.0, 1.0 / self.height as f32]].iter().zip(&r.smooth_uniforms) {
            let smooth = SmoothUniforms {
                direction: *direction,
                world_radius,
                pixels_per_meter: projection.y_axis.y * self.height as f32 * 0.5,
                depth_falloff: self.particle_radius * 2.0,
                pad: [0.0; 3],
            };
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&smooth));
        }
        r.particles.upload(device, queue, &self.particles);
        r.lines.upload(device, queue, if settings.show_helpers { &self.lines[..] } else { &[] });
        r.meshes.upload(device, queue, &self.meshes);
        r.surface.upload(device, queue, if mesh_surface { &self.surface[..] } else { &[] });
        r.show_output = fluid_surface;
        r.targets(device, self.width, self.height);
        let (Some(t), p) = (r.targets.as_ref(), &r.pipelines) else { return Vec::new() };
        let draw_lines = |pass: &mut wgpu::RenderPass<'_>| {
            if r.lines.count > 0 {
                pass.set_pipeline(&p.line);
                pass.set_vertex_buffer(0, r.lines.buffer.slice(..));
                pass.draw(0..r.lines.count, 0..1);
            }
        };
        let draw_particles = |pass: &mut wgpu::RenderPass<'_>, pipeline| {
            if r.particles.count > 0 {
                pass.set_pipeline(pipeline);
                pass.set_vertex_buffer(0, r.particles.buffer.slice(..));
                pass.draw(0..4, 0..r.particles.count);
            }
        };
        let draw_surface = |pass: &mut wgpu::RenderPass<'_>, pipeline| {
            if r.surface.count > 0 {
                pass.set_pipeline(pipeline);
                pass.set_vertex_buffer(0, r.surface.buffer.slice(..));
                pass.draw(0..r.surface.count, 0..1);
            }
        };

        // 1) background: sky, floor, colliders. spheres and lines go here as well when there is no fluid surface.
        {
            let mut pass = begin_pass(encoder, "scene", color_attachment(&t.scene_color, CLEAR), depth_attachment(&t.scene_depth, true));
            pass.set_bind_group(0, &r.frame_group, &[]);
            pass.set_pipeline(&p.sky);
            pass.draw(0..3, 0..1);
            if r.meshes.count > 0 {
                pass.set_pipeline(&p.mesh);
                pass.set_vertex_buffer(0, r.meshes.buffer.slice(..));
                pass.draw(0..r.meshes.count, 0..1);
            }
            if !fluid_surface {
                draw_particles(&mut pass, &p.sphere);
                draw_lines(&mut pass);
            }
        }
        if !fluid_surface {
            return Vec::new();
        }
        // 2) fluid depth: front most surface, linear eye depth, 0 = empty
        {
            let mut pass =
                begin_pass(encoder, "fluid depth", color_attachment(&t.fluid_depth, CLEAR), depth_attachment(&t.fluid_depth_test, true));
            pass.set_bind_group(0, &r.frame_group, &[]);
            if mesh_surface {
                draw_surface(&mut pass, &p.surface_depth);
            } else {
                draw_particles(&mut pass, &p.fluid_depth);
            }
        }
        // 3) thickness, additive
        {
            let depth = if mesh_surface { None } else { depth_attachment(&t.scene_depth, false) };
            let mut pass = begin_pass(encoder, "thickness", color_attachment(&t.thickness, CLEAR), depth);
            pass.set_bind_group(0, &r.frame_group, &[]);
            if mesh_surface {
                pass.set_bind_group(1, &t.surface_thickness_group, &[]);
                draw_surface(&mut pass, &p.surface_thickness);
            } else {
                draw_particles(&mut pass, &p.fluid_thickness);
            }
        }
        // 4) bilateral smoothing, separable, ping pong; this turns the spheres into one hull
        for iteration in 0..iterations {
            for (target, group) in [(&t.smooth[0], if iteration == 0 { 0 } else { 2 }), (&t.smooth[1], 1)] {
                let mut pass = begin_pass(encoder, "smooth", color_attachment(target, None), None);
                pass.set_bind_group(0, &r.frame_group, &[]);
                pass.set_bind_group(1, &t.smooth_groups[group], &[]);
                pass.set_pipeline(&p.smooth);
                pass.draw(0..3, 0..1);
            }
        }
        // 5) composite: normals from the smoothed depth, fresnel, reflection, refraction, absorption; then the lines
        {
            let mut pass =
                begin_pass(encoder, "composite", color_attachment(&t.output_color, CLEAR), depth_attachment(&t.output_depth, true));
            pass.set_bind_group(0, &r.frame_group, &[]);
            pass.set_bind_group(1, &t.composite_groups[if iterations > 0 { 0 } else { 1 }], &[]);
            pass.set_pipeline(&p.composite);
            pass.draw(0..3, 0..1);
            draw_lines(&mut pass);
        }
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, resources: &egui_wgpu::CallbackResources) {
        let Some(r) = resources.get::<Resources>() else { return };
        let Some(t) = &r.targets else { return };
        pass.set_pipeline(&r.pipelines.blit);
        pass.set_bind_group(0, &t.blit_groups[usize::from(r.show_output)], &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Line segments for an axis aligned box.
pub fn box_lines(lines: &mut Vec<LineVertex>, min: V3, max: V3, color: [f32; 3]) {
    let corner = |i: usize| {
        V3::new(if i & 1 == 0 { min.x } else { max.x }, if i & 2 == 0 { min.y } else { max.y }, if i & 4 == 0 { min.z } else { max.z })
    };
    for a in 0..8 {
        for bit in [1, 2, 4] {
            if a & bit == 0 {
                lines.push(LineVertex { position: corner(a).into(), color });
                lines.push(LineVertex { position: corner(a | bit).into(), color });
            }
        }
    }
}

pub fn line(lines: &mut Vec<LineVertex>, from: V3, to: V3, color: [f32; 3]) {
    lines.push(LineVertex { position: from.into(), color });
    lines.push(LineVertex { position: to.into(), color });
}

/// A circle around `normal`.
pub fn circle_lines(lines: &mut Vec<LineVertex>, center: V3, normal: V3, radius: f32, color: [f32; 3]) {
    const SEGMENTS: usize = 48;
    let (tangent, bitangent) = normal.normalize().any_orthonormal_pair();
    let point = |i: usize| {
        let (s, c) = (i as f32 / SEGMENTS as f32 * std::f32::consts::TAU).sin_cos();
        center + (tangent * c + bitangent * s) * radius
    };
    for i in 0..SEGMENTS {
        line(lines, point(i), point(i + 1), color);
    }
}

/// Three great circles of a sphere.
pub fn sphere_lines(lines: &mut Vec<LineVertex>, center: V3, radius: f32, color: [f32; 3]) {
    for normal in [V3::X, V3::Y, V3::Z] {
        circle_lines(lines, center, normal, radius, color);
    }
}

fn triangle(meshes: &mut Vec<MeshVertex>, corners: [V3; 3], normals: [V3; 3], color: [f32; 3], checker: f32) {
    for (position, normal) in corners.into_iter().zip(normals) {
        meshes.push(MeshVertex { position: position.into(), normal: normal.into(), color, checker });
    }
}

/// The checkered floor 2 m around the domain, just below it.
pub fn floor_mesh(meshes: &mut Vec<MeshVertex>, min: V3, max: V3) {
    const MARGIN: f32 = 2.0;
    let height = min.y - 0.002;
    let a = V3::new(min.x - MARGIN, height, min.z - MARGIN);
    let b = V3::new(max.x + MARGIN, height, min.z - MARGIN);
    let c = V3::new(max.x + MARGIN, height, max.z + MARGIN);
    let d = V3::new(min.x - MARGIN, height, max.z + MARGIN);
    let color = [0.74, 0.72, 0.68];
    triangle(meshes, [a, c, b], [V3::Y; 3], color, 1.0);
    triangle(meshes, [a, d, c], [V3::Y; 3], color, 1.0);
}

pub const COLLIDER_COLOR: [f32; 3] = [0.86, 0.52, 0.26];

pub fn solid_box(meshes: &mut Vec<MeshVertex>, center: V3, half: V3) {
    for axis in [[V3::X, V3::Y, V3::Z], [V3::Y, V3::Z, V3::X], [V3::Z, V3::X, V3::Y]] {
        for sign in [1.0, -1.0] {
            let normal = axis[0] * sign;
            let face = center + normal * half;
            let tangent = axis[1] * half;
            let bitangent = axis[2] * half * sign;
            let (a, b, c, d) =
                (face - tangent - bitangent, face + tangent - bitangent, face + tangent + bitangent, face - tangent + bitangent);
            triangle(meshes, [a, b, c], [normal; 3], COLLIDER_COLOR, 0.0);
            triangle(meshes, [a, c, d], [normal; 3], COLLIDER_COLOR, 0.0);
        }
    }
}

pub fn solid_sphere(meshes: &mut Vec<MeshVertex>, center: V3, radius: f32) {
    const STACKS: u32 = 16;
    const SLICES: u32 = 32;
    let point = |stack: u32, slice: u32| {
        let polar = std::f32::consts::PI * stack as f32 / STACKS as f32;
        let azimuth = std::f32::consts::TAU * slice as f32 / SLICES as f32;
        V3::new(polar.sin() * azimuth.cos(), polar.cos(), polar.sin() * azimuth.sin())
    };
    for stack in 0..STACKS {
        for slice in 0..SLICES {
            let (a, b, c, d) = (point(stack, slice), point(stack + 1, slice), point(stack + 1, slice + 1), point(stack, slice + 1));
            for normals in [[a, b, c], [a, c, d]] {
                triangle(meshes, normals.map(|normal| center + normal * radius), normals, COLLIDER_COLOR, 0.0);
            }
        }
    }
}
