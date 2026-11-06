//! The crate ships a copy of the shared wgsl shaders in `shaders/` for builds outside the repository (see build.rs).
//! This keeps the copy identical to `../src/gpu/shaders`, the files the c++ library and repository builds use.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// File name to contents of every .wgsl file in a directory, line endings normalized.
fn shaders_in(directory: &Path) -> BTreeMap<String, String> {
    fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", directory.display()))
        .map(|entry| entry.expect("directory entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "wgsl"))
        .map(|path| {
            let name = path.file_name().expect("file name").to_string_lossy().into_owned();
            let contents = fs::read_to_string(&path).unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
            (name, contents.replace("\r\n", "\n"))
        })
        .collect()
}

#[test]
fn vendored_shaders_match_the_shared_ones() {
    let manifest_directory = Path::new(env!("CARGO_MANIFEST_DIR"));
    let shared_directory = manifest_directory.join("../src/gpu/shaders");
    if !shared_directory.is_dir() {
        println!("not inside the tinysph repository, nothing to compare");
        return;
    }
    let shared = shaders_in(&shared_directory);
    let vendored = shaders_in(&manifest_directory.join("shaders"));
    assert!(!shared.is_empty(), "no shaders in {}", shared_directory.display());
    assert_eq!(
        shared.keys().collect::<Vec<_>>(),
        vendored.keys().collect::<Vec<_>>(),
        "rust/shaders must hold the same files as src/gpu/shaders; copy them over"
    );
    for (name, contents) in &shared {
        assert!(vendored[name] == *contents, "rust/shaders/{name} differs from src/gpu/shaders/{name}; copy it over");
    }
}
