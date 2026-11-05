add_executable(sph_tests solver_tests.cpp test_framework.h)
target_link_libraries(sph_tests PRIVATE sph_core)
add_test(NAME sph_tests COMMAND sph_tests)

add_executable(sph_bench dam_break_bench.cpp)
target_link_libraries(sph_bench PRIVATE sph_core)
