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

if(NOT EMSCRIPTEN)
    find_package(OpenGL REQUIRED)
endif()

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
    src/sandbox_scene.cpp
    src/sandbox_scene.h
    src/sandbox_ui.cpp
    src/sandbox_ui.h
    src/shader_sources.h
    src/surface_mesher.cpp
    src/surface_mesher.h)
target_link_libraries(tinysph_sandbox PRIVATE tinysph::tinysph tinysph_warnings imgui
                                              SDL3::SDL3-static)

if(EMSCRIPTEN)
    # webgl 2 in a page that fills the browser window; the page is web/shell.html
    set_target_properties(tinysph_sandbox PROPERTIES SUFFIX ".html" OUTPUT_NAME "index")
    target_link_options(
        tinysph_sandbox
        PRIVATE
        -sMIN_WEBGL_VERSION=2
        -sMAX_WEBGL_VERSION=2
        -sGL_ENABLE_GET_PROC_ADDRESS=1
        -sALLOW_MEMORY_GROWTH=1
        -sENVIRONMENT=web
        "--shell-file=${CMAKE_CURRENT_SOURCE_DIR}/web/shell.html")
    set_property(
        TARGET tinysph_sandbox
        APPEND
        PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/web/shell.html)
else()
    target_link_libraries(tinysph_sandbox PRIVATE OpenGL::GL)
endif()

if(WIN32)
    set_target_properties(tinysph_sandbox PROPERTIES WIN32_EXECUTABLE ON)
endif()

tinysph_copy_runtime_dlls(tinysph_sandbox)
