enable_language(C)

include(FetchContent)

# ----------------------------------------------------------------------------------------------------
# sdl3
# ----------------------------------------------------------------------------------------------------

set(SDL_SHARED
    OFF
    CACHE BOOL "" FORCE)
set(SDL_STATIC
    ON
    CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY
    OFF
    CACHE BOOL "" FORCE)

FetchContent_Declare(
    sdl3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG release-3.2.28
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(sdl3)

# ----------------------------------------------------------------------------------------------------
# dear imgui
# ----------------------------------------------------------------------------------------------------

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.9
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(imgui)

add_library(
    imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC SDL3::SDL3-static)

# ----------------------------------------------------------------------------------------------------
# sandbox
# ----------------------------------------------------------------------------------------------------

find_package(OpenGL REQUIRED)

add_executable(
    tinysph_sandbox
    src/camera.cpp
    src/camera.h
    src/gl_functions.cpp
    src/gl_functions.h
    src/main.cpp
    src/matrix.h
    src/renderer.cpp
    src/renderer.h
    src/sandbox_ui.cpp
    src/sandbox_scene.cpp
    src/sandbox_scene.h
    src/sandbox_ui.h
    src/shader_sources.h)
target_link_libraries(tinysph_sandbox PRIVATE tinysph::tinysph tinysph_warnings imgui SDL3::SDL3-static OpenGL::GL)

if(WIN32)
    set_target_properties(tinysph_sandbox PROPERTIES WIN32_EXECUTABLE ON)
endif()
