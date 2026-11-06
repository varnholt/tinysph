//! Draws particles as shaded sphere impostors coloured by speed, and the domain, colliders and mouse tool as lines.
//! It paints inside egui's render pass through an `egui_wgpu` callback, so it works the same natively and on the web.

use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::{self, wgpu};
use glam::{Mat4, Vec3 as V3};
use wgpu::util::DeviceExt;

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const SHADER: &str = r"
struct Uniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    // x: particle radius, y: speed that maps to the top of the colour ramp
    params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct SphereOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) center_view: vec3<f32>,
    @location(2) color: vec3<f32>,
};

fn ramp(t: f32) -> vec3<f32> {
    let deep = vec3<f32>(0.05, 0.22, 0.55);
    let mid = vec3<f32>(0.12, 0.55, 0.9);
    let fast = vec3<f32>(0.85, 0.95, 1.0);
    if (t < 0.5) { return mix(deep, mid, t * 2.0); }
    return mix(mid, fast, t * 2.0 - 1.0);
}

@vertex
fn sphere_vs(@builtin(vertex_index) vertex: u32, @location(0) particle: vec4<f32>) -> SphereOut {
    let corner = vec2<f32>(f32(vertex & 1u) * 2.0 - 1.0, f32((vertex >> 1u) & 1u) * 2.0 - 1.0);
    let center_view = (u.view * vec4<f32>(particle.xyz, 1.0)).xyz;
    // the quad is a bit larger than the sphere and sits in front of it, so the silhouette is never clipped
    let r = u.params.x;
    let position = center_view + vec3<f32>(corner * r * 1.15, r);
    var out: SphereOut;
    out.clip = u.projection * vec4<f32>(position, 1.0);
    out.corner = corner * 1.15;
    out.center_view = center_view;
    out.color = ramp(clamp(particle.w / u.params.y, 0.0, 1.0));
    return out;
}

struct SphereFragment {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn sphere_fs(in: SphereOut) -> SphereFragment {
    let d2 = dot(in.corner, in.corner);
    if (d2 > 1.0) { discard; }
    let normal = vec3<f32>(in.corner, sqrt(1.0 - d2));
    let surface = in.center_view + normal * u.params.x;
    let clip = u.projection * vec4<f32>(surface, 1.0);
    let light = normalize(vec3<f32>(0.4, 0.8, 0.6));
    let diffuse = max(dot(normal, light), 0.0);
    let half_vector = normalize(light + vec3<f32>(0.0, 0.0, 1.0));
    let specular = pow(max(dot(normal, half_vector), 0.0), 48.0);
    var out: SphereFragment;
    out.color = vec4<f32>(in.color * (0.3 + 0.7 * diffuse) + vec3<f32>(0.5 * specular), 1.0);
    out.depth = clip.z / clip.w;
    return out;
}

struct LineOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn line_vs(@location(0) position: vec3<f32>, @location(1) color: vec3<f32>) -> LineOut {
    var out: LineOut;
    out.clip = u.projection * u.view * vec4<f32>(position, 1.0);
    out.color = color;
    return out;
}

@fragment
fn line_fs(in: LineOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
";

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

/// The gpu objects, kept in egui_wgpu's callback resources.
struct Resources {
    sphere_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    particles: wgpu::Buffer,
    particle_capacity: u64,
    particle_count: u32,
    lines: wgpu::Buffer,
    line_capacity: u64,
    line_count: u32,
}

/// Creates the pipelines; call once with eframe's render state.
pub fn init(render_state: &egui_wgpu::RenderState) {
    let device = &render_state.device;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sandbox"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniforms"),
        size: size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let depth = || {
        Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        })
    };
    let target = [Some(wgpu::ColorTargetState::from(render_state.target_format))];
    let pipeline = |label: &str, vs: &str, fs: &str, buffers: &[Option<wgpu::VertexBufferLayout<'_>>], topology| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some(vs), buffers, compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(fs),
                targets: &target,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { topology, ..Default::default() },
            depth_stencil: depth(),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    };
    let sphere_pipeline = pipeline(
        "spheres",
        "sphere_vs",
        "sphere_fs",
        &[Some(wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0 => Float32x4],
        })],
        wgpu::PrimitiveTopology::TriangleStrip,
    );
    let line_pipeline = pipeline(
        "lines",
        "line_vs",
        "line_fs",
        &[Some(wgpu::VertexBufferLayout {
            array_stride: 24,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        })],
        wgpu::PrimitiveTopology::LineList,
    );
    let buffer = |label, size, usage| {
        device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
    };
    let resources = Resources {
        sphere_pipeline,
        line_pipeline,
        uniforms,
        bind_group,
        particles: buffer("particles", 16, wgpu::BufferUsages::VERTEX),
        particle_capacity: 16,
        particle_count: 0,
        lines: buffer("lines", 24, wgpu::BufferUsages::VERTEX),
        line_capacity: 24,
        line_count: 0,
    };
    render_state.renderer.write().callback_resources.insert(resources);
}

/// One frame's worth of drawing.
pub struct Frame {
    pub view: Mat4,
    pub projection: Mat4,
    pub radius: f32,
    pub max_speed: f32,
    /// xyz position, w speed
    pub particles: Vec<[f32; 4]>,
    pub lines: Vec<LineVertex>,
}

fn upload(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &mut wgpu::Buffer, capacity: &mut u64, data: &[u8]) {
    if data.is_empty() {
        return;
    }
    if (data.len() as u64) > *capacity {
        *buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: data,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        *capacity = data.len() as u64;
    } else {
        queue.write_buffer(buffer, 0, data);
    }
}

impl egui_wgpu::CallbackTrait for Frame {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(r) = resources.get_mut::<Resources>() else { return Vec::new() };
        let uniforms = Uniforms {
            view: self.view.to_cols_array_2d(),
            projection: self.projection.to_cols_array_2d(),
            params: [self.radius, self.max_speed, 0.0, 0.0],
        };
        queue.write_buffer(&r.uniforms, 0, bytemuck::bytes_of(&uniforms));
        upload(device, queue, &mut r.particles, &mut r.particle_capacity, bytemuck::cast_slice(&self.particles));
        r.particle_count = self.particles.len() as u32;
        upload(device, queue, &mut r.lines, &mut r.line_capacity, bytemuck::cast_slice(&self.lines));
        r.line_count = self.lines.len() as u32;
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, resources: &egui_wgpu::CallbackResources) {
        let Some(r) = resources.get::<Resources>() else { return };
        pass.set_bind_group(0, &r.bind_group, &[]);
        if r.line_count > 0 {
            pass.set_pipeline(&r.line_pipeline);
            pass.set_vertex_buffer(0, r.lines.slice(..));
            pass.draw(0..r.line_count, 0..1);
        }
        if r.particle_count > 0 {
            pass.set_pipeline(&r.sphere_pipeline);
            pass.set_vertex_buffer(0, r.particles.slice(..));
            pass.draw(0..4, 0..r.particle_count);
        }
    }
}

/// Line segments for an axis aligned box.
pub fn box_lines(lines: &mut Vec<LineVertex>, min: V3, max: V3, color: [f32; 3]) {
    let corner = |i: usize| V3::new(if i & 1 == 0 { min.x } else { max.x }, if i & 2 == 0 { min.y } else { max.y }, if i & 4 == 0 { min.z } else { max.z });
    for a in 0..8 {
        for bit in [1, 2, 4] {
            if a & bit == 0 {
                lines.push(LineVertex { position: corner(a).into(), color });
                lines.push(LineVertex { position: corner(a | bit).into(), color });
            }
        }
    }
}

/// Three great circles of a sphere.
pub fn sphere_lines(lines: &mut Vec<LineVertex>, center: V3, radius: f32, color: [f32; 3]) {
    const SEGMENTS: usize = 48;
    for axis in 0..3 {
        let point = |i: usize| {
            let angle = i as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
            let (s, c) = angle.sin_cos();
            let offset = match axis {
                0 => V3::new(c, s, 0.0),
                1 => V3::new(c, 0.0, s),
                _ => V3::new(0.0, c, s),
            };
            (center + offset * radius).into()
        };
        for i in 0..SEGMENTS {
            lines.push(LineVertex { position: point(i), color });
            lines.push(LineVertex { position: point(i + 1), color });
        }
    }
}
