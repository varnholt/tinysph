//! The sandbox: ui panel, camera, mouse tool and the frame loop.

use eframe::egui_wgpu;
use egui::{Color32, Sense};
use glam::{Mat4, Vec3 as V3};
use tinysph::{Attractor, Backend, FluidParameters, Shape, Vec3};

use crate::render::{self, Frame};
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
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Off,
    Pull,
    Push,
}

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
        let size = to_glam(scene.domain.max - scene.domain.min);
        Camera { yaw: -0.5, pitch: 0.35, distance: size.max_element() * 2.1, target: to_glam(scene.domain.min) + size * 0.5 }
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
        Mat4::perspective_rh(45f32.to_radians(), aspect, 0.02, 50.0)
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
            camera: Camera::framing(&scene),
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
            ui.label(format!("step: {:.1} ms, frame: {:.1} ms ({:.0} fps)", snapshot.step_ms, self.frame_ms, 1000.0 / self.frame_ms.max(0.01)));
        });
        if reset {
            self.reset();
        }
    }

    /// The point under the cursor on the plane through the camera target that faces the camera.
    fn mouse_point(&self, rect: egui::Rect, pointer: egui::Pos2) -> Option<V3> {
        let ndc = glam::Vec2::new((pointer.x - rect.min.x) / rect.width() * 2.0 - 1.0, 1.0 - (pointer.y - rect.min.y) / rect.height() * 2.0);
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
        if response.dragged_by(egui::PointerButton::Secondary) || (self.tool == Tool::Off && response.dragged_by(egui::PointerButton::Primary)) {
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
            self.camera.distance = (self.camera.distance * (-scroll * 0.002).exp()).clamp(0.2, 20.0);
        }

        // mouse tool
        self.attractor = None;
        if tool_active && self.options.capture.is_none() {
            if let Some(point) = response.interact_pointer_pos().and_then(|pointer| self.mouse_point(rect, pointer)) {
                let sign = if self.tool == Tool::Pull { 1.0 } else { -1.0 };
                self.attractor =
                    Some(Attractor { position: Vec3::new(point.x, point.y, point.z), strength: self.tool_strength * sign, radius: self.tool_radius });
            }
        }

        // lines: domain, colliders, mouse tool
        let mut lines = Vec::new();
        render::box_lines(&mut lines, to_glam(self.scene.domain.min), to_glam(self.scene.domain.max), [0.55, 0.6, 0.7]);
        for collider in &self.scene.colliders {
            match collider {
                Shape::Sphere(sphere) => render::sphere_lines(&mut lines, to_glam(sphere.center), sphere.radius, [0.95, 0.75, 0.3]),
                Shape::Cuboid(cuboid) => {
                    let (center, half) = (to_glam(cuboid.center), to_glam(cuboid.half_extents));
                    render::box_lines(&mut lines, center - half, center + half, [0.95, 0.75, 0.3]);
                }
            }
        }
        if let Some(attractor) = &self.attractor {
            let color = if attractor.strength > 0.0 { [0.4, 1.0, 0.5] } else { [1.0, 0.4, 0.4] };
            render::sphere_lines(&mut lines, to_glam(attractor.position), attractor.radius, color);
        }

        let (particles, radius) = self.sim.with_snapshot(|snapshot| {
            let particles = snapshot.positions.iter().zip(&snapshot.velocities).map(|(p, v)| [p.x, p.y, p.z, v.length()]).collect::<Vec<_>>();
            (particles, snapshot.radius)
        });
        let frame = Frame {
            view: self.camera.view(),
            projection: Camera::projection(rect.aspect_ratio()),
            radius,
            max_speed: 3.0,
            particles,
            lines,
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
