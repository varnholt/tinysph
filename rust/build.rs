//! Tells the gpu backend where its wgsl shaders are, through the `TINYSPH_SHADER_DIR` environment variable.
//!
//! Inside the repository the crate builds from `../src/gpu/shaders`, the files the c++ library compiles in, so both
//! backends always run the same shaders. A packaged crate cannot reach files outside its directory, so it falls back to
//! the copy in `shaders/`, which `cargo package` ships; `tests/shaders.rs` fails when that copy differs from the shared
//! files. Without the `gpu` feature this does nothing.

use std::env;
use std::path::{Path, PathBuf};

const SHADERS: [&str; 8] = ["common", "assign_cells", "scan", "scatter", "density", "normals", "forces", "integrate"];

fn has_all_shaders(directory: &Path) -> bool {
    SHADERS.iter().all(|name| directory.join(format!("{name}.wgsl")).is_file())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if env::var_os("CARGO_FEATURE_GPU").is_none() {
        return;
    }

    let manifest_directory = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let shared = manifest_directory.join("..").join("src").join("gpu").join("shaders");
    let vendored = manifest_directory.join("shaders");
    let directory = if has_all_shaders(&shared) { shared } else { vendored };
    assert!(has_all_shaders(&directory), "the wgsl shaders are missing from {}", directory.display());

    // include_str! tracks the files themselves; this reruns the script when one appears or disappears
    println!("cargo:rerun-if-changed={}", directory.display());
    println!("cargo:rustc-env=TINYSPH_SHADER_DIR={}", directory.display());
}
