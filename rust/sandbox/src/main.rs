//! Interactive sandbox for tinysph, natively (wgpu on vulkan, metal or d3d12) and in a browser on WebGPU.
//!
//! `tinysph-sandbox [--preset N] [--cpu] [--zoom STEPS] [--tilt RADIANS] [--spheres|--surface|--mesh] [--capture <dir> [--frames N]]`

mod app;
mod mesher;
mod render;
mod scene;
mod sim;

#[cfg(not(target_arch = "wasm32"))]
fn save_png(path: &std::path::Path, image: &egui::ColorImage) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut encoder = png::Encoder::new(file, image.size[0] as u32, image.size[1] as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let pixels: Vec<u8> = image.pixels.iter().flat_map(|pixel| pixel.to_array()).collect();
    encoder.write_header()?.write_image_data(&pixels)?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    let mut options = app::Options { frames: 1, ..app::Options::default() };
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = || arguments.next().unwrap_or_default();
        match argument.as_str() {
            "--preset" => options.preset = value().parse().unwrap_or(0),
            "--cpu" => options.cpu = true,
            "--ui" => options.ui = true,
            "--gpu" => options.cpu = false,
            "--capture" => options.capture = Some(value().into()),
            "--zoom" => options.zoom = value().parse().unwrap_or(0.0),
            "--tilt" => options.tilt = value().parse().unwrap_or(0.0),
            "--spheres" => options.render_mode = Some(app::RenderMode::Spheres),
            "--surface" => options.render_mode = Some(app::RenderMode::FluidSurface),
            "--mesh" => options.render_mode = Some(app::RenderMode::MeshSurface),
            "--frames" => options.frames = value().parse::<u32>().unwrap_or(1).max(1),
            _ => {
                eprintln!(
                    "usage: tinysph-sandbox [--preset N] [--cpu] [--zoom STEPS] [--tilt RADIANS] [--spheres|--surface|--mesh] [--capture <dir> [--frames N] [--ui]]"
                );
                std::process::exit(2);
            }
        }
    }
    if let Some(directory) = &options.capture {
        std::fs::create_dir_all(directory).expect("cannot create the capture directory");
    }
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1600.0, 900.0]).with_title("tinysph sandbox"),
        ..Default::default()
    };
    eframe::run_native("tinysph sandbox", native_options, Box::new(|context| Ok(Box::new(app::App::new(context, options)))))
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
mod web {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::*;

    /// Whether the browser exposes `navigator.gpu`.
    pub fn has_webgpu() -> bool {
        web_sys::window().is_some_and(|window| js_sys_has_gpu(&window.navigator()))
    }

    fn js_sys_has_gpu(navigator: &web_sys::Navigator) -> bool {
        let value: &JsValue = navigator.as_ref();
        js_get(value, "gpu").is_some_and(|gpu| !gpu.is_undefined() && !gpu.is_null())
    }

    #[wasm_bindgen(inline_js = "export function get(object, key) { return object[key]; }")]
    extern "C" {
        #[wasm_bindgen(js_name = get)]
        fn js_get_raw(object: &JsValue, key: &str) -> JsValue;
    }

    fn js_get(object: &JsValue, key: &str) -> Option<JsValue> {
        Some(js_get_raw(object, key))
    }

    /// Starts the sandbox on the canvas with id `canvas`. `?preset=N` picks a preset, `?cpu` the cpu backend.
    #[wasm_bindgen(start)]
    pub fn start() {
        let window = web_sys::window().expect("no window");
        let search = window.location().search().unwrap_or_default();
        let mut options = crate::app::Options { frames: 1, ..Default::default() };
        for pair in search.trim_start_matches('?').split('&') {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            match key {
                "preset" => options.preset = value.parse().unwrap_or(0),
                "cpu" => options.cpu = true,
                _ => {}
            }
        }
        let canvas = window
            .document()
            .and_then(|document| document.get_element_by_id("canvas"))
            .and_then(|element| element.dyn_into::<web_sys::HtmlCanvasElement>().ok())
            .expect("no canvas with id 'canvas'");
        let web_options = eframe::WebOptions::default();
        wasm_bindgen_futures::spawn_local(async move {
            let result = eframe::WebRunner::new()
                .start(canvas, web_options, Box::new(|context| Ok(Box::new(crate::app::App::new(context, options)))))
                .await;
            if let Err(error) = result {
                web_sys::console::error_1(&error);
            }
        });
    }
}
