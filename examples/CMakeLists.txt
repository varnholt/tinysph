add_executable(hello_fluid hello_fluid.cpp)
target_link_libraries(hello_fluid PRIVATE tinysph::tinysph tinysph_warnings)
