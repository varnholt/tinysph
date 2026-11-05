cmake_minimum_required(VERSION 3.25)
project(sph_sandbox LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SPH_BUILD_APP "Build the interactive sandbox (SDL3 + OpenGL + Dear ImGui)" ON)
option(SPH_BUILD_TESTS "Build the solver tests" ON)

if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(core)

if(SPH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SPH_BUILD_APP)
    add_subdirectory(app)
endif()
