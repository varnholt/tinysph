# tinysph

[![CI](https://github.com/varnholt/tinysph/actions/workflows/ci.yml/badge.svg)](https://github.com/varnholt/tinysph/actions/workflows/ci.yml)

A tiny C++23 library for smoothed particle hydrodynamics (SPH): fluid volumes, colliders, force fields and emitters
behind one small API, with no dependencies beyond the standard library. Comes with an interactive sandbox that renders
the fluid as a surface with reflection and refraction, and lets you poke the water with the mouse.

```cpp
#include <tinysph/tinysph.h>
using namespace std::chrono_literals;

tinysph::World world{{._domain = {{0, 0, 0}, {1.6f, 1, 0.6f}}}};
world.addFluid(tinysph::Box{._center = {0.25f, 0.3f, 0.3f}, ._half_extents = {0.23f, 0.28f, 0.28f}});
world.addCollider(tinysph::Sphere{._center = {1.1f, 0.12f, 0.3f}, ._radius = 0.12f});

while (running)
{
   world.step(16ms);           // any std::chrono duration
   draw(world.positions());    // std::span<const tinysph::Vec3>
}
```

Settings that come from user input can be checked instead of trusted:

```cpp
if (auto world = tinysph::World::create(settings))
{
   run(*world);
}
else
{
   std::println("invalid settings: {}", tinysph::describe(world.error()));
}
```

See [examples/hello_fluid.cpp](examples/hello_fluid.cpp) for a complete program.

## Use it in your project

FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(tinysph GIT_REPOSITORY https://github.com/varnholt/tinysph.git GIT_TAG v0.1.0)
FetchContent_MakeAvailable(tinysph)
target_link_libraries(your_target PRIVATE tinysph::tinysph)
```

When pulled in this way only the library is built; the sandbox, tests and examples are skipped.

Or install it and use `find_package`:

```
cmake -S . -B build -DTINYSPH_BUILD_SANDBOX=OFF
cmake --build build --config Release
cmake --install build --config Release --prefix <prefix>
```

```cmake
find_package(tinysph 0.1 REQUIRED)
target_link_libraries(your_target PRIVATE tinysph::tinysph)
```

## API

Everything public lives in `include/tinysph/`:

| Header | Contents |
|--------|----------|
| `tinysph.h` | includes everything below |
| `world.h` | `World`: the simulation. Particles, colliders, force fields, emitters, `step()` |
| `types.h` | shapes (`Sphere`, `Box`), force fields (`Attractor`, `Vortex`, `Wind`), `Emitter`, handles, `WorldSettings`, `FluidParameters`, `SettingsError`, `StepStatistics`, `Seconds` |
| `math.h` | `Vec3`, `Aabb` |
| `version.h` | `tinysph::version` |

Design rules that keep it stable:

- `World` hides its implementation behind a pointer, so the solver, its data layout and future backends (GPU) can change without breaking your code.
- Colliders, force fields and emitters are referred to by typed handles (`ColliderId`, ...). A handle never points to a different object after removal; ids are not reused.
- `WorldSettings` is fixed per world (resolution, rest density, backend); `FluidParameters` can be changed at any time.
- Time is a `std::chrono` duration, so `step(16)` (seconds or milliseconds?) does not compile; `step(16ms)` does.
- Invalid settings are reported through `std::expected` by `World::create()`; the constructor throws `std::invalid_argument` (or terminates without exceptions).
- Semantic versioning. Until 1.0 the API may still change between minor versions; from 1.0 on, only with a new major version.

## Build

Requirements: CMake 3.25+ and a C++23 compiler. CI builds and tests with MSVC (Visual Studio 2022), GCC 14, Clang 20 and AppleClang (macOS 15).
The sandbox fetches SDL3 and Dear ImGui.

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
build\sandbox\Release\tinysph_sandbox.exe
```

Tests and benchmark:

```
ctest --test-dir build -C Release --output-on-failure
build\tests\Release\tinysph_bench.exe [speed_of_sound] [viscosity] [seconds] [artificial_viscosity]
```

| Option | Default | |
|--------|---------|-|
| `TINYSPH_BUILD_SANDBOX` | top level only | interactive sandbox |
| `TINYSPH_BUILD_TESTS` | top level only | `tinysph_api_tests`, `tinysph_internal_tests`, `tinysph_bench` |
| `TINYSPH_BUILD_EXAMPLES` | top level only | `hello_fluid` |
| `TINYSPH_INSTALL` | top level only | install rules and the CMake package |

## Sandbox controls

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
| `include/tinysph/` | public API |
| `src/` | implementation (`tinysph::detail`), not installed |
| `sandbox/` | the interactive sandbox: SDL3 window, OpenGL 3.3 / GLSL ES 3.00 renderer, Dear ImGui panels |
| `examples/` | small programs using the library |
| `tests/` | API tests, internal tests, headless dam break benchmark |
| `docs/` | notes and the review of the original code |

## Solver

Weakly compressible SPH (WCSPH) with the Mueller et al. 2003 kernels:

- uniform grid with cell size h, rebuilt every substep with a counting sort that also reorders the particle data
- density with the poly6 kernel, self contribution included
- pressure `p = c^2 (rho - rho0)`, optionally clamped at 0
- symmetric pressure force (spiky gradient), viscosity (viscosity kernel laplacian), Monaghan artificial viscosity
- semi-implicit Euler with an adaptive time step (CFL and force condition)
- colliders as signed distance functions, the domain as six planes

The particle data is stored as separate arrays per attribute and the grid is a counting sort, so the
same passes can move to GPU compute shaders.

## Background

tinysph started as a port of [smoothed_particle_hydrodynamics](https://github.com/varnholt/smoothed_particle_hydrodynamics)
(Qt, 2015/2016). The port fixed the solver along the way; see [docs/original-code-review.md](docs/original-code-review.md).
The theory notes from the original project live in [docs/sph-notes.md](docs/sph-notes.md).

## Code style

Same as deceptus_engine: `_clang-format` and `.cmake-format.yaml` are copied from there.

```
clang-format -i include/tinysph/*.h src/*.h src/*.cpp sandbox/src/*.h sandbox/src/*.cpp tests/*.h tests/*.cpp examples/*.cpp
uv sync
uv run cmake-format -i CMakeLists.txt sandbox/CMakeLists.txt tests/CMakeLists.txt examples/CMakeLists.txt
```

## Roadmap

1. ~~Port to CMake + C++23 + Dear ImGui, fix the solver, add tests~~
2. ~~Library with a stable API and CMake package~~
3. GPU backend: solver passes as WGSL compute shaders behind the same `World` API, checked against the CPU solver
4. WebGPU renderer (Dawn / wgpu-native natively, browser WebGPU under Emscripten)
5. Emscripten build and an online demo
6. Scene save / load, moving colliders, more shapes (capsule, mesh SDF)
7. Optional: position based fluids (PBF) as a second solver

## License

MIT, see [LICENSE](LICENSE). The original Qt project remains under LGPL-3.0; tinysph is a rewrite by the same author and is released under MIT.
