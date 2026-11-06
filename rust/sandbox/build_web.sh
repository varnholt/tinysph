#!/bin/sh
# builds the browser version into sandbox/dist: cargo for wasm32, then wasm-bindgen (the cli version must match the
# wasm-bindgen crate in Cargo.lock). serve the directory over http, e.g. python -m http.server -d sandbox/dist
set -e
cd "$(dirname "$0")/.."
cargo build --release -p tinysph-sandbox --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir sandbox/dist target/wasm32-unknown-unknown/release/tinysph-sandbox.wasm
cp sandbox/web/index.html sandbox/dist/
