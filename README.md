# SPH Sandbox

An interactive sandbox for smoothed particle hydrodynamics: set up a scene with fluid volumes,
colliders, emitters and force fields, then poke the water with the mouse.

It started as a port of [smoothed_particle_hydrodynamics](https://github.com/varnholt/smoothed_particle_hydrodynamics)
(Qt, 2015/2016) to C++23, CMake and Dear ImGui. The port fixed the solver along the way; see
[docs/original-code-review.md](docs/original-code-review.md). The theory notes from the original
project live in [docs/sph-notes.md](docs/sph-notes.md).

## Build

Requirements: CMake 3.25+, a C++23 compiler (tested with MSVC 19.44 / Visual Studio 2022).
SDL3 and Dear ImGui are fetched by CMake.

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
build\app\Release\sph_sandbox.exe
```

Tests and benchmark:

```
ctest --test-dir build -C Release --output-on-failure
build\tests\Release\sph_bench.exe [speed_of_sound] [viscosity] [seconds] [artificial_viscosity]
```

`-DSPH_BUILD_APP=OFF` builds only the solver and the tests.

## Controls

| Input | Action |
|-------|--------|
| left drag | mouse tool (pull or push the fluid) |
| right drag | orbit |
| middle drag | pan |
| wheel | zoom |
| space | play / pause |
| r | reset the scene |

## Layout

| Path | Contents |
|------|----------|
| `core/` | `sph_core`, the solver. No dependencies apart from the standard library. |
| `app/` | The sandbox: SDL3 window, OpenGL 3.3 renderer, Dear ImGui panels. |
| `tests/` | Solver tests (`sph_tests`) and a headless dam break benchmark (`sph_bench`). |
| `docs/` | Notes and the review of the original code. |

## Solver

Weakly compressible SPH (WCSPH) with the Mueller et al. 2003 kernels:

- uniform grid with cell size h, rebuilt every substep with a counting sort that also reorders the particle data
- density with the poly6 kernel, self contribution included
- pressure `p = c^2 (rho - rho0)`, optionally clamped at 0
- symmetric pressure force (spiky gradient), viscosity (viscosity kernel laplacian), Monaghan artificial viscosity
- semi-implicit Euler with an adaptive time step (CFL and force condition)
- colliders as signed distance functions, the container as six planes

The particle data is stored as separate arrays per attribute and the grid is a counting sort, so the
same passes can move to GPU compute shaders later.

## Code style

Same as deceptus_engine: `_clang-format` and `.cmake-format.yaml` are copied from there.

```
clang-format -i core/include/sph/*.h core/src/*.cpp app/src/*.h app/src/*.cpp tests/*.h tests/*.cpp
uv sync
uv run cmake-format -i CMakeLists.txt core/CMakeLists.txt app/CMakeLists.txt tests/CMakeLists.txt
```

## Roadmap

1. ~~Port to CMake + C++23 + Dear ImGui, fix the solver, add tests~~
2. WebGPU renderer (Dawn / wgpu-native natively, browser WebGPU under Emscripten) with screen space fluid rendering
3. Solver passes as WGSL compute shaders, checked against the CPU solver
4. Scene save / load, moving colliders, more shapes (capsule, mesh SDF)
5. Emscripten build
6. Optional: position based fluids (PBF) as a second solver

## License

MIT, see [LICENSE](LICENSE). The original Qt project remains under LGPL-3.0; this sandbox is a rewrite by the same author and is released under MIT.
