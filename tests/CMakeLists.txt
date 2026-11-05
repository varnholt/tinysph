add_executable(tinysph_api_tests api_tests.cpp test_framework.h test_scenes.h)
target_link_libraries(tinysph_api_tests PRIVATE tinysph::tinysph tinysph_warnings)
target_compile_definitions(
    tinysph_api_tests
    PRIVATE TINYSPH_PROJECT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
            TINYSPH_PROJECT_VERSION_MINOR=${PROJECT_VERSION_MINOR}
            TINYSPH_PROJECT_VERSION_PATCH=${PROJECT_VERSION_PATCH})
add_test(NAME tinysph_api_tests COMMAND tinysph_api_tests)

# internals are not part of the installed api, so these tests reach into src/
add_executable(tinysph_internal_tests solver_internal_tests.cpp test_framework.h)
target_link_libraries(tinysph_internal_tests PRIVATE tinysph::tinysph tinysph_warnings)
target_include_directories(tinysph_internal_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME tinysph_internal_tests COMMAND tinysph_internal_tests)

add_executable(tinysph_bench dam_break_bench.cpp test_scenes.h)
target_link_libraries(tinysph_bench PRIVATE tinysph::tinysph tinysph_warnings)

foreach(target tinysph_api_tests tinysph_internal_tests tinysph_bench)
    tinysph_copy_runtime_dlls(${target})
endforeach()

# std::getenv reads TINYSPH_REQUIRE_GPU; msvc deprecates it in favor of _dupenv_s
target_compile_definitions(tinysph_api_tests
                           PRIVATE $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>)
