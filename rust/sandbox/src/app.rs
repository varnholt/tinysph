//! The sandbox: ui panel, camera, mouse tool and the frame loop.

use eframe::egui_wgpu;
use egui::{Color32, Sense};
use glam::{Mat4, Vec3 as V3};
use tinysph::{Attractor, Backend, FluidParameters, ForceField, Shape, Vec3};

use crate::mesher::Mesher;
pub use crate::render::RenderMode;
use crate::render::{self, ColorMode, Frame};
use crate::scene::{self, PRESET_NAMES, Scene};
use crate::sim::{Build, Sim};

/// Command line options, the same as the c++ sandbox's.
#[derive(Clone, Default)]
pub struct Options {
    pub preset: usize,
    pub cpu: bool,
    pub capture: Option<std::path::PathBuf>,
    pub frames: u32,
    /// draw the ui panel in captured frames too
    pub ui: bool,
    /// wheel steps towards the target, as `--zoom` in the c++ sandbox
    pub zoom: f32,
    /// added to the camera pitch in radians
    pub tilt: f32,
    pub render_mode: Option<render::RenderMode>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Off,
    Pull,
    Push,
}

/// field of view in y, radians, as in the c++ sandbox
const FOV_Y: f32 = 0.8;

struct Camera {
    yaw: f32,
    pitch: f32,
    distance: f32,
    target: V3,
}

fn to_glam(v: Vec3) -> V3 {
    V3::new(v.x, v.y, v.z)
}

impl Camera {
    fn framing(scene: &Scene) -> Self {
        // as the c++ OrbitCamera::frame: back off until the bounding sphere of the domain fits
        let size = to_glam(scene.domain.max - scene.domain.min);
        let distance = size.length() * 0.5 / (FOV_Y * 0.5).sin() * 0.9;
        Camera { yaw: -0.5, pitch: 0.35, distance, target: to_glam(scene.domain.min) + size * 0.5 }
    }

    fn eye(&self) -> V3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + V3::new(sy * cp, sp, cy * cp) * self.distance
    }

    fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, V3::Y)
    }

    fn projection(aspect: f32) -> Mat4 {
        Mat4::perspective_rh(FOV_Y, aspect, render::NEAR, render::FAR)
    }

    /// The c++ wheel: every step moves 10 % closer.
    fn zoom(&mut self, steps: f32) {
        self.distance = (self.distance * 0.9f32.powf(steps)).clamp(0.1, 50.0);
    }
}

pub struct App {
    options: Options,
    sim: Sim,
    preset: usize,
    scene: Scene,
    parameters: FluidParameters,
    spacing: f32,
    fallback_spacing: f32,
    backend: Backend,
    running: bool,
    time_scale: f32,
    max_substeps: u32,
    tool: Tool,
    tool_strength: f32,
    tool_radius: f32,
    attractor: Option<Attractor>,
    camera: Camera,
    render: render::Settings,
    mesher: Mesher,
    frame_ms: f32,
    last_frame: web_time::Instant,
    frame: u32,
    show_ui: bool,
}

impl App {
    pub fn new(context: &eframe::CreationContext<'_>, options: Options) -> Self {
        if let Some(render_state) = &context.wgpu_render_state {
            render::init(render_state);
        }
        let scene = scene::preset(options.preset);
        let web = cfg!(target_arch = "wasm32");
        let mut app = App {
            preset: options.preset % PRESET_NAMES.len(),
            camera: {
                let mut camera = Camera::framing(&scene);
                camera.zoom(options.zoom);
                camera.pitch = (camera.pitch + options.tilt).clamp(-1.5, 1.5);
                camera
            },
            // the mesh surface is the c++ default; its mesher runs on one cpu thread, so the browser starts with the
            // particle based surface
            render: render::Settings {
                mode: options.render_mode.unwrap_or(if web { RenderMode::FluidSurface } else { RenderMode::MeshSurface }),
                ..Default::default()
            },
            mesher: Mesher::default(),
            parameters: FluidParameters::default(),
            spacing: 0.025,
            // one cpu thread in a browser
            fallback_spacing: if web { 0.045 } else { 0.025 },
            backend: if options.cpu { Backend::Cpu } else { Backend::Gpu },
            running: true,
            time_scale: 1.0,
            max_substeps: if options.capture.is_some() { 100_000 } else { 40 },
            tool: Tool::Pull,
            tool_strength: 60.0,
            tool_radius: if web { 0.27 } else { 0.15 },
            attractor: None,
            frame_ms: 0.0,
            last_frame: web_time::Instant::now(),
            frame: 0,
            show_ui: options.capture.is_none() || options.ui,
            sim: Sim::default(),
            scene,
            options,
        };
        app.reset();
        app
    }

    fn reset(&mut self) {
        self.scene = scene::preset(self.preset);
        self.parameters.gravity = self.scene.gravity;
        self.parameters.surface_tension = self.scene.surface_tension;
        self.sim.rebuild(Build {
            scene: self.scene.clone(),
            spacing: self.spacing,
            backend: self.backend,
            parameters: self.parameters,
            fallback_spacing: self.fallback_spacing,
        });
    }

    fn panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("tinysph");
        let mut preset = self.preset;
        egui::ComboBox::from_label("preset").selected_text(PRESET_NAMES[preset]).show_ui(ui, |ui| {
            for (index, name) in PRESET_NAMES.iter().enumerate() {
                ui.selectable_value(&mut preset, index, *name);
            }
        });
        let mut reset = false;
        if preset != self.preset {
            self.preset = preset;
            self.camera = Camera::framing(&scene::preset(preset));
            reset = true;
        }
        ui.horizontal(|ui| {
            if ui.button(if self.running { "Pause" } else { "Play" }).clicked() {
                self.running = !self.running;
            }
            reset |= ui.button("Reset").clicked();
        });

        ui.separator();
        ui.label("Simulation (applied on reset)");
        ui.horizontal(|ui| {
            reset |= ui.radio_value(&mut self.backend, Backend::Gpu, "GPU").changed();
            reset |= ui.radio_value(&mut self.backend, Backend::Cpu, "CPU").changed();
        });
        ui.add(egui::Slider::new(&mut self.spacing, 0.012..=0.06).text("particle spacing (m)"));

        ui.separator();
        ui.label("Fluid");
        let p = &mut self.parameters;
        ui.add(egui::Slider::new(&mut p.gravity.y, -20.0..=0.0).text("gravity y"));
        ui.add(egui::Slider::new(&mut p.speed_of_sound, 5.0..=80.0).text("speed of sound"));
        ui.add(egui::Slider::new(&mut p.viscosity, 0.0..=2.0).text("viscosity"));
        ui.add(egui::Slider::new(&mut p.artificial_viscosity, 0.0..=0.5).text("artificial viscosity"));
        ui.add(egui::Slider::new(&mut p.surface_tension, 0.0..=2.0).text("surface tension"));
        ui.add(egui::Slider::new(&mut p.wall_restitution, 0.0..=1.0).text("wall restitution"));
        ui.add(egui::Slider::new(&mut p.wall_friction, 0.0..=1.0).text("wall friction"));
        ui.add(egui::Slider::new(&mut self.time_scale, 0.1..=2.0).text("time scale"));
        ui.add(egui::Slider::new(&mut self.max_substeps, 1..=200).text("max substeps"));

        ui.separator();
        ui.label("Rendering");
        let r = &mut self.render;
        ui.horizontal(|ui| {
            ui.radio_value(&mut r.mode, RenderMode::Spheres, "spheres");
            ui.radio_value(&mut r.mode, RenderMode::FluidSurface, "fluid surface");
            ui.radio_value(&mut r.mode, RenderMode::MeshSurface, "mesh surface");
        });
        if r.mode == RenderMode::Spheres {
            ui.horizontal(|ui| {
                ui.radio_value(&mut r.color_mode, ColorMode::Speed, "speed");
                ui.radio_value(&mut r.color_mode, ColorMode::Flat, "flat");
            });
            ui.add(egui::Slider::new(&mut r.radius_scale, 0.3..=2.0).text("radius scale"));
            ui.add(egui::Slider::new(&mut r.speed_range, 0.5..=10.0).text("speed range (m/s)"));
        } else {
            if r.mode == RenderMode::FluidSurface {
                ui.add(egui::Slider::new(&mut r.smoothing_iterations, 1..=8).text("smoothing passes"));
                ui.add(egui::Slider::new(&mut r.smoothing_radius, 1.0..=10.0).text("smoothing radius (particle radii)"));
            }
            ui.add(egui::Slider::new(&mut r.refraction_strength, 0.0..=0.5).text("refraction"));
            ui.horizontal(|ui| {
                ui.label("fluid colour");
                ui.color_edit_button_rgb(&mut r.fluid_color);
            });
            ui.add(egui::Slider::new(&mut r.absorption[0], 0.0..=10.0).text("absorption red"));
            ui.add(egui::Slider::new(&mut r.absorption[1], 0.0..=10.0).text("absorption green"));
            ui.add(egui::Slider::new(&mut r.absorption[2], 0.0..=10.0).text("absorption blue"));
        }
        ui.checkbox(&mut r.show_helpers, "helpers (container, emitters, fields)");
        ui.checkbox(&mut r.show_floor, "floor");
        ui.separator();
        ui.label("Mouse tool (left drag; right drag orbits, middle pans, wheel zooms)");
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.tool, Tool::Off, "off");
            ui.radio_value(&mut self.tool, Tool::Pull, "pull");
            ui.radio_value(&mut self.tool, Tool::Push, "push");
        });
        ui.add(egui::Slider::new(&mut self.tool_strength, 5.0..=200.0).text("strength"));
        ui.add(egui::Slider::new(&mut self.tool_radius, 0.05..=0.5).text("radius"));

        ui.separator();
        let note = self.sim.note();
        if !note.is_empty() {
            ui.colored_label(Color32::from_rgb(255, 190, 90), note);
        }
        let backend = match self.sim.backend() {
            Some(Backend::Gpu) => "GPU",
            Some(Backend::Cpu) => "CPU",
            None => "starting",
        };
        ui.label(format!("backend: {backend} ({})", self.sim.device()));
        self.sim.with_snapshot(|snapshot| {
            let statistics = &snapshot.statistics;
            ui.label(format!("particles: {}", snapshot.positions.len()));
            ui.label(format!("substeps: {}, time step {:.2} ms", statistics.substeps, statistics.time_step * 1000.0));
            ui.label(format!("max density ratio: {:.3}", statistics.max_density_ratio));
            ui.label(format!(
                "step: {:.1} ms, frame: {:.1} ms ({:.0} fps)",
                snapshot.step_ms,
                self.frame_ms,
                1000.0 / self.frame_ms.max(0.01)
            ));
        });
        if reset {
            self.reset();
        }
    }

    /// The point under the cursor on the plane through the camera target that faces the camera.
    fn mouse_point(&self, rect: egui::Rect, pointer: egui::Pos2) -> Option<V3> {
        let ndc =
            glam::Vec2::new((pointer.x - rect.min.x) / rect.width() * 2.0 - 1.0, 1.0 - (pointer.y - rect.min.y) / rect.height() * 2.0);
        let inverse = (Camera::projection(rect.aspect_ratio()) * self.camera.view()).inverse();
        let near = inverse.project_point3(ndc.extend(0.0));
        let far = inverse.project_point3(ndc.extend(1.0));
        let direction = (far - near).normalize();
        let normal = (self.camera.eye() - self.camera.target).normalize();
        let denominator = direction.dot(normal);
        if denominator.abs() < 1e-5 {
            return None;
        }
        let t = (self.camera.target - near).dot(normal) / denominator;
        (t > 0.0).then(|| near + direction * t)
    }

    fn viewport(&mut self, ui: &mut egui::Ui) {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());

        // camera
        let delta = response.drag_delta();
        let tool_active = self.tool != Tool::Off && response.dragged_by(egui::PointerButton::Primary);
        if response.dragged_by(egui::PointerButton::Secondary)
            || (self.tool == Tool::Off && response.dragged_by(egui::PointerButton::Primary))
        {
            self.camera.yaw -= delta.x * 0.008;
            self.camera.pitch = (self.camera.pitch + delta.y * 0.008).clamp(-1.5, 1.5);
        }
        if response.dragged_by(egui::PointerButton::Middle) {
            let view = self.camera.view().inverse();
            let scale = self.camera.distance * 0.0015;
            self.camera.target += (view.x_axis.truncate() * -delta.x + view.y_axis.truncate() * delta.y) * scale;
        }
        if response.hovered() {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            self.camera.zoom(scroll / 50.0);
        }

        // mouse tool
        self.attractor = None;
        if tool_active
            && self.options.capture.is_none()
            && let Some(point) = response.interact_pointer_pos().and_then(|pointer| self.mouse_point(rect, pointer))
        {
            let sign = if self.tool == Tool::Pull { 1.0 } else { -1.0 };
            self.attractor = Some(Attractor {
                position: Vec3::new(point.x, point.y, point.z),
                strength: self.tool_strength * sign,
                radius: self.tool_radius,
            });
        }

        // helper lines as in the c++ sandbox: container, emitters, fields, mouse tool
        let mut lines = Vec::new();
        let (domain_min, domain_max) = (to_glam(self.scene.domain.min), to_glam(self.scene.domain.max));
        render::box_lines(&mut lines, domain_min, domain_max, [0.85, 0.85, 0.9]);
        let emitter_color = [0.3, 0.95, 0.4];
        for emitter in &self.scene.emitters {
            let direction = to_glam(emitter.velocity).normalize_or(V3::Y);
            let position = to_glam(emitter.position);
            render::circle_lines(&mut lines, position, direction, emitter.radius, emitter_color);
            render::line(&mut lines, position, position + direction * 0.1, emitter_color);
        }
        let field_color = [0.85, 0.35, 0.9];
        for field in &self.scene.fields {
            match field {
                ForceField::Attractor(attractor) => {
                    render::sphere_lines(&mut lines, to_glam(attractor.position), attractor.radius, field_color);
                }
                ForceField::Vortex(vortex) => {
                    let (position, axis) = (to_glam(vortex.position), to_glam(vortex.axis).normalize_or(V3::Y));
                    render::circle_lines(&mut lines, position, axis, vortex.radius, field_color);
                    render::line(&mut lines, position - axis * vortex.radius, position + axis * vortex.radius, field_color);
                }
                ForceField::Wind(wind) => render::box_lines(&mut lines, to_glam(wind.region.min), to_glam(wind.region.max), field_color),
            }
        }
        if let Some(attractor) = &self.attractor {
            render::sphere_lines(&mut lines, to_glam(attractor.position), attractor.radius, [1.0, 1.0, 0.3]);
        }

        // floor and solid colliders
        let mut meshes = Vec::new();
        if self.render.show_floor {
            render::floor_mesh(&mut meshes, domain_min, domain_max);
        }
        for collider in &self.scene.colliders {
            match collider {
                Shape::Sphere(sphere) => render::solid_sphere(&mut meshes, to_glam(sphere.center), sphere.radius),
                Shape::Cuboid(cuboid) => render::solid_box(&mut meshes, to_glam(cuboid.center), to_glam(cuboid.half_extents)),
            }
        }

        let mesh_surface = self.render.mode == RenderMode::MeshSurface;
        let mesher = &mut self.mesher;
        let (particles, radius) = self.sim.with_snapshot(|snapshot| {
            let particles =
                snapshot.positions.iter().zip(&snapshot.velocities).map(|(p, v)| [p.x, p.y, p.z, v.length()]).collect::<Vec<_>>();
            if mesh_surface {
                let positions: Vec<V3> = snapshot.positions.iter().map(|p| to_glam(*p)).collect();
                mesher.build(&positions, snapshot.radius * 2.0, domain_min, domain_max);
            }
            (particles, snapshot.radius)
        });
        let pixels_per_point = ui.ctx().pixels_per_point();
        let frame = Frame {
            view: self.camera.view(),
            fov_y: FOV_Y,
            eye: self.camera.eye(),
            width: (rect.width() * pixels_per_point).round() as u32,
            height: (rect.height() * pixels_per_point).round() as u32,
            particle_radius: radius,
            settings: self.render,
            particles,
            lines,
            meshes,
            surface: if mesh_surface { std::mem::take(&mut self.mesher.vertices) } else { Vec::new() },
        };
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, frame));
    }

    fn capture(&mut self, context: &egui::Context) {
        let Some(directory) = self.options.capture.clone() else { return };
        // screenshots arrive a frame or two after they were requested
        let screenshots: Vec<_> = context.input(|input| {
            input
                .raw
                .events
                .iter()
                .filter_map(|event| match event {
                    egui::Event::Screenshot { image, user_data, .. } => Some((image.clone(), user_data.data.clone())),
                    _ => None,
                })
                .collect()
        });
        for (image, data) in screenshots {
            let index = data.and_then(|data| data.downcast_ref::<u32>().copied()).unwrap_or(0);
            #[cfg(not(target_arch = "wasm32"))]
            {
                let path = directory.join(format!("frame_{index:04}.png"));
                if let Err(error) = crate::save_png(&path, &image) {
                    eprintln!("capture: {}: {error}", path.display());
                }
                if index + 1 >= self.options.frames {
                    let (particles, step_ms) = self.sim.with_snapshot(|snapshot| (snapshot.positions.len(), snapshot.step_ms));
                    println!(
                        "captured {} frames to {}: {particles} particles on {}, last step {step_ms:.1} ms",
                        self.options.frames,
                        directory.display(),
                        self.sim.device()
                    );
                    context.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
            #[cfg(target_arch = "wasm32")]
            let _ = (image, index, &directory);
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let context = ui.ctx().clone();
        let now = web_time::Instant::now();
        self.frame_ms = self.frame_ms * 0.9 + now.duration_since(self.last_frame).as_secs_f32() * 1000.0 * 0.1;
        self.last_frame = now;

        self.capture(&context);
        let capturing = self.options.capture.is_some();
        // capture mode waits for the world, steps at a fixed 60 Hz and writes every frame
        let ready = !self.sim.busy() && self.sim.backend().is_some();
        if self.running && ready && (!capturing || self.frame < self.options.frames) {
            let seconds = if capturing { 1.0 / 60.0 } else { (1.0 / 60.0) * self.time_scale };
            self.sim.step(seconds, self.max_substeps, self.parameters, self.attractor);
            if capturing {
                context.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new(self.frame)));
            }
            self.frame += 1;
        }

        if self.show_ui {
            egui::Panel::left("panel").resizable(false).show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.panel(ui));
            });
        }
        egui::CentralPanel::default().frame(egui::Frame::NONE.fill(Color32::from_rgb(22, 24, 30))).show(ui, |ui| self.viewport(ui));
        context.request_repaint();
    }
}
