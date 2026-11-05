# downloads a prebuilt wgpu-native release (https://github.com/gfx-rs/wgpu-native) and provides the
# imported target tinysph::wgpu_native. it is a shared library, so its runtime file has to sit next
# to executables on windows; use tinysph_copy_runtime_dlls(<target>) for executables built in this
# project.

include(FetchContent)

set(TINYSPH_WGPU_VERSION v29.0.1.1)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$" OR CMAKE_OSX_ARCHITECTURES STREQUAL
                                                               "arm64")
    set(_tinysph_wgpu_arch aarch64)
else()
    set(_tinysph_wgpu_arch x86_64)
endif()

if(WIN32)
    set(_tinysph_wgpu_platform windows-${_tinysph_wgpu_arch}-msvc)
elseif(APPLE)
    set(_tinysph_wgpu_platform macos-${_tinysph_wgpu_arch})
else()
    set(_tinysph_wgpu_platform linux-${_tinysph_wgpu_arch})
endif()

# sha256 digests as published on the github release
set(_tinysph_wgpu_hash_windows-x86_64-msvc
    7e67d7445c42aeb85e30f88930fd8d7d83ee769e3390aeb1ada75ebf3cf78132)
set(_tinysph_wgpu_hash_windows-aarch64-msvc
    4a876421a8c1e5fe72f849b3722214280fe485cb1c56f77f8b0c82414be5b29f)
set(_tinysph_wgpu_hash_linux-x86_64
    95a4d90c071005a98d03eab348beaa6b07e16eb00d1dcdb9f8348f75eb97ec5a)
set(_tinysph_wgpu_hash_linux-aarch64
    015fcdf1dbae82e614a783cc38017e5399ae0927a889fe9b69c9b664bc61b47a)
set(_tinysph_wgpu_hash_macos-x86_64
    8e2f7378548ddd0e2cf21e7d864dda46e953f0af724855a33778b85ead206d41)
set(_tinysph_wgpu_hash_macos-aarch64
    a5797a37b1adf720bcd5dcffb291edbbd5b7b14be0a3874c28e6393a655a7a3e)

FetchContent_Declare(
    wgpu_native
    URL https://github.com/gfx-rs/wgpu-native/releases/download/${TINYSPH_WGPU_VERSION}/wgpu-${_tinysph_wgpu_platform}-release.zip
    URL_HASH SHA256=${_tinysph_wgpu_hash_${_tinysph_wgpu_platform}}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(wgpu_native)

set(_tinysph_wgpu_lib_dir ${wgpu_native_SOURCE_DIR}/lib)
if(WIN32)
    set(TINYSPH_WGPU_RUNTIME_NAME wgpu_native.dll)
    set(TINYSPH_WGPU_IMPLIB_NAME wgpu_native.dll.lib)
elseif(APPLE)
    set(TINYSPH_WGPU_RUNTIME_NAME libwgpu_native.dylib)
else()
    set(TINYSPH_WGPU_RUNTIME_NAME libwgpu_native.so)
endif()

add_library(tinysph::wgpu_native SHARED IMPORTED GLOBAL)
set_target_properties(
    tinysph::wgpu_native
    PROPERTIES IMPORTED_LOCATION ${_tinysph_wgpu_lib_dir}/${TINYSPH_WGPU_RUNTIME_NAME}
               INTERFACE_INCLUDE_DIRECTORIES ${wgpu_native_SOURCE_DIR}/include
               INTERFACE_COMPILE_DEFINITIONS WGPU_SHARED_LIBRARY)
if(WIN32)
    set_target_properties(
        tinysph::wgpu_native PROPERTIES IMPORTED_IMPLIB
                                        ${_tinysph_wgpu_lib_dir}/${TINYSPH_WGPU_IMPLIB_NAME})
elseif(NOT APPLE)
    # the release .so has no soname, so the linker would record its absolute build path
    set_target_properties(tinysph::wgpu_native PROPERTIES IMPORTED_NO_SONAME TRUE)
endif()

# copies the runtime dlls an executable depends on next to it; a no-op outside windows, where the
# rpath finds them
function(tinysph_copy_runtime_dlls target)
    if(WIN32)
        add_custom_command(
            TARGET ${target}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target}>
                    $<TARGET_FILE_DIR:${target}>
            COMMAND_EXPAND_LISTS)
    endif()
endfunction()
