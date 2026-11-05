add_library(sph_core src/scene.cpp src/solver.cpp)
target_include_directories(sph_core PUBLIC include)
target_compile_features(sph_core PUBLIC cxx_std_23)

# libstdc++ implements the parallel algorithms on top of TBB
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    find_package(TBB QUIET)
    if(TBB_FOUND)
        target_link_libraries(sph_core PUBLIC TBB::tbb)
    endif()
endif()
