cmake_minimum_required(VERSION 3.25)
project(
    tinysph
    VERSION 0.1.0
    DESCRIPTION "Tiny C++23 smoothed particle hydrodynamics library"
    LANGUAGES CXX)

include(GNUInstallDirs)

# the sandbox, tests and examples are only built by default when tinysph is the top level project,
# so pulling it in with FetchContent or add_subdirectory only builds the library
option(TINYSPH_BUILD_SANDBOX "Build the interactive sandbox (SDL3 + OpenGL + Dear ImGui)"
       ${PROJECT_IS_TOP_LEVEL})
option(TINYSPH_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
option(TINYSPH_BUILD_EXAMPLES "Build the examples" ${PROJECT_IS_TOP_LEVEL})
option(TINYSPH_INSTALL "Generate install and package config rules" ${PROJECT_IS_TOP_LEVEL})
set(TINYSPH_USES_TBB OFF)

# warnings for our own targets only, never forced onto consumers
add_library(tinysph_warnings INTERFACE)
if(MSVC)
    target_compile_options(tinysph_warnings INTERFACE /W4 /permissive- /utf-8)
else()
    target_compile_options(tinysph_warnings INTERFACE -Wall -Wextra -Wpedantic)
    # partially designated initializers like World{{._domain = ...}} are the intended way to use the
    # settings structs
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL
                                                 19)
        target_compile_options(tinysph_warnings
                               INTERFACE -Wno-missing-designated-field-initializers)
    endif()
endif()

# ----------------------------------------------------------------------------------------------------
# library
# ----------------------------------------------------------------------------------------------------

add_library(
    tinysph
    include/tinysph/math.h
    include/tinysph/tinysph.h
    include/tinysph/types.h
    include/tinysph/version.h
    include/tinysph/world.h
    src/geometry.cpp
    src/geometry.h
    src/kernels.h
    src/parallel.h
    src/solver.cpp
    src/solver.h
    src/world.cpp)
add_library(tinysph::tinysph ALIAS tinysph)

target_compile_features(tinysph PUBLIC cxx_std_23)
target_include_directories(tinysph PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(tinysph PRIVATE $<BUILD_INTERFACE:tinysph_warnings>)
set_target_properties(
    tinysph
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}
               WINDOWS_EXPORT_ALL_SYMBOLS ON)

# libstdc++ implements the parallel algorithms on top of tbb, for gcc and for clang alike
if(NOT MSVC)
    find_package(TBB QUIET)
    if(TBB_FOUND)
        target_link_libraries(tinysph PRIVATE TBB::tbb)
        set(TINYSPH_USES_TBB ON)
    endif()
endif()

# ----------------------------------------------------------------------------------------------------
# install, so consumers can use find_package(tinysph)
# ----------------------------------------------------------------------------------------------------

if(TINYSPH_INSTALL)
    include(CMakePackageConfigHelpers)

    install(
        TARGETS tinysph
        EXPORT tinysphTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY include/tinysph DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(
        EXPORT tinysphTargets
        NAMESPACE tinysph::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tinysph)

    configure_package_config_file(
        cmake/tinysphConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/tinysphConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tinysph)
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/tinysphConfigVersion.cmake
                                     COMPATIBILITY SameMinorVersion)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tinysphConfig.cmake
                  ${CMAKE_CURRENT_BINARY_DIR}/tinysphConfigVersion.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tinysph)
endif()

# ----------------------------------------------------------------------------------------------------
# extras
# ----------------------------------------------------------------------------------------------------

if(TINYSPH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TINYSPH_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(TINYSPH_BUILD_SANDBOX)
    add_subdirectory(sandbox)
endif()
