//! Owns the world. Natively a step blocks; in a browser the gpu backend must await its readbacks, so creating and
//! stepping a world runs as a task and the app draws the last finished frame meanwhile.

use std::cell::RefCell;
use std::rc::Rc;

use tinysph::{Attractor, Backend, FluidParameters, ForceFieldId, SettingsError, StepStatistics, Vec3, World, WorldSettings};

use crate::scene::Scene;

/// The particles of the last finished step, for drawing.
#[derive(Default)]
pub struct Snapshot {
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub radius: f32,
    pub statistics: StepStatistics,
    /// wall time of the last step in ms, measured here (the library reports zero in a browser)
    pub step_ms: f32,
}

#[derive(Default)]
struct Shared {
    world: Option<World>,
    busy: bool,
    snapshot: Snapshot,
    backend: Option<Backend>,
    device: String,
    note: String,
    mouse_field: Option<ForceFieldId>,
    generation: u64,
}

/// What a world is built from.
#[derive(Clone)]
pub struct Build {
    pub scene: Scene,
    pub spacing: f32,
    pub backend: Backend,
    pub parameters: FluidParameters,
    /// spacing to use when the gpu is unavailable and the cpu takes over
    pub fallback_spacing: f32,
}

#[derive(Clone, Default)]
pub struct Sim {
    shared: Rc<RefCell<Shared>>,
}

fn settings(build: &Build, backend: Backend, spacing: f32) -> WorldSettings {
    WorldSettings { domain: build.scene.domain, particle_spacing: spacing, backend, ..WorldSettings::default() }
}

fn populate(world: &mut World, scene: &Scene) {
    // colliders first, so fluid is not spawned inside them
    for collider in &scene.colliders {
        world.add_collider(*collider);
    }
    for field in &scene.fields {
        world.add_force_field(*field);
    }
    for emitter in &scene.emitters {
        world.add_emitter(*emitter);
    }
    for (shape, velocity) in &scene.fluids {
        world.add_fluid(*shape, *velocity);
    }
}

impl Sim {
    pub fn busy(&self) -> bool {
        self.shared.borrow().busy
    }

    /// The backend the world runs on, `None` while it is being created.
    pub fn backend(&self) -> Option<Backend> {
        self.shared.borrow().backend
    }

    pub fn device(&self) -> String {
        self.shared.borrow().device.clone()
    }

    /// Why the gpu is not used, if it was asked for and is not.
    pub fn note(&self) -> String {
        self.shared.borrow().note.clone()
    }

    pub fn with_snapshot<R>(&self, read: impl FnOnce(&Snapshot) -> R) -> R {
        read(&self.shared.borrow().snapshot)
    }

    fn install(shared: &Rc<RefCell<Shared>>, generation: u64, result: Result<World, SettingsError>, note: String) {
        let mut shared = shared.borrow_mut();
        if shared.generation != generation {
            return; // a newer rebuild was requested meanwhile
        }
        shared.busy = false;
        shared.note = note;
        match result {
            Ok(world) => {
                shared.backend = Some(world.settings().backend);
                shared.device = world.device_description().to_owned();
                shared.snapshot = Snapshot {
                    positions: world.positions().to_vec(),
                    velocities: world.velocities().to_vec(),
                    radius: world.particle_radius(),
                    ..Snapshot::default()
                };
                shared.world = Some(world);
            }
            Err(error) => shared.note = format!("{error}"),
        }
    }

    /// Replaces the world. With the gpu backend unavailable it falls back to the cpu at `fallback_spacing`.
    pub fn rebuild(&self, build: Build) {
        let generation = {
            let mut shared = self.shared.borrow_mut();
            shared.generation += 1;
            shared.world = None;
            shared.backend = None;
            shared.mouse_field = None;
            shared.busy = true;
            shared.generation
        };
        let create_cpu = |build: &Build, note: String| {
            let mut world = World::new(settings(build, Backend::Cpu, build.fallback_spacing), build.parameters)?;
            populate(&mut world, &build.scene);
            Ok::<_, SettingsError>((world, note))
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            let result = match World::new(settings(&build, build.backend, build.spacing), build.parameters) {
                Ok(mut world) => {
                    populate(&mut world, &build.scene);
                    Ok((world, String::new()))
                }
                Err(error) if build.backend == Backend::Gpu => create_cpu(&build, format!("gpu: {error}, using the cpu")),
                Err(error) => Err(error),
            };
            let (result, note) = match result {
                Ok((world, note)) => (Ok(world), note),
                Err(error) => (Err(error), String::new()),
            };
            Self::install(&self.shared, generation, result, note);
        }
        #[cfg(target_arch = "wasm32")]
        {
            let shared = Rc::clone(&self.shared);
            wasm_bindgen_futures::spawn_local(async move {
                let created = if build.backend == Backend::Gpu && crate::web::has_webgpu() {
                    World::new_async(settings(&build, Backend::Gpu, build.spacing), build.parameters).await
                } else {
                    Err(SettingsError::GpuUnavailable)
                };
                let result = match created {
                    Ok(mut world) => {
                        populate(&mut world, &build.scene);
                        Ok((world, String::new()))
                    }
                    Err(_) if build.backend == Backend::Gpu => {
                        create_cpu(&build, "WebGPU is not available in this browser, simulating on the cpu with fewer particles".into())
                    }
                    Err(_) => create_cpu(&build, String::new()),
                };
                let (result, note) = match result {
                    Ok((world, note)) => (Ok(world), note),
                    Err(error) => (Err(error), String::new()),
                };
                Self::install(&shared, generation, result, note);
            });
        }
    }

    /// Advances the world by one frame, with the mouse attractor if the tool is active. In a browser this starts a task
    /// and returns at once; frames requested while a step is running are dropped.
    pub fn step(&self, seconds: f32, max_substeps: u32, parameters: FluidParameters, attractor: Option<Attractor>) {
        let (mut world, generation) = {
            let mut shared = self.shared.borrow_mut();
            if shared.busy {
                return;
            }
            let Some(mut world) = shared.world.take() else { return };
            world.set_parameters(parameters);
            match (attractor, shared.mouse_field) {
                (Some(attractor), Some(id)) => {
                    world.set_force_field(id, attractor);
                }
                (Some(attractor), None) => shared.mouse_field = Some(world.add_force_field(attractor)),
                (None, Some(id)) => {
                    world.remove_force_field(id);
                    shared.mouse_field = None;
                }
                (None, None) => {}
            }
            shared.busy = true;
            (world, shared.generation)
        };
        let shared = Rc::clone(&self.shared);
        let finish = move |world: World, statistics: StepStatistics, step_ms: f32| {
            let mut shared = shared.borrow_mut();
            if shared.generation != generation {
                return;
            }
            shared.busy = false;
            let snapshot = &mut shared.snapshot;
            snapshot.positions.clear();
            snapshot.positions.extend_from_slice(world.positions());
            snapshot.velocities.clear();
            snapshot.velocities.extend_from_slice(world.velocities());
            snapshot.radius = world.particle_radius();
            snapshot.statistics = statistics;
            snapshot.step_ms = step_ms;
            shared.world = Some(world);
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            let start = web_time::Instant::now();
            let statistics = world.step_seconds(seconds, max_substeps);
            finish(world, statistics, start.elapsed().as_secs_f32() * 1000.0);
        }
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move {
            let start = web_time::Instant::now();
            let statistics = world.step_seconds_async(seconds, max_substeps).await;
            finish(world, statistics, start.elapsed().as_secs_f32() * 1000.0);
        });
    }
}
