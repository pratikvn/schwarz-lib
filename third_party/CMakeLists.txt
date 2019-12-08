# For each external package, check if they are already available, otherwise use ours
if(BUILD_TESTS)
    ginkgo_add_subdirectory(GTest gtest)
endif()

if(SCHWARZ_DEVEL_TOOLS)
  set(GCF_IGNORE_LIST "third_party" CACHE STRING "Ignore directories for GCF")
  add_subdirectory(git-cmake-format)
endif()


ginkgo_add_subdirectory(gflags gflags)
if(BUILD_BENCHMARKS)
    ginkgo_add_subdirectory(RapidJSON rapidjson)

    # RapidJSON does not provide a modern CMake target so far. Let's fix that!
    if(RapidJSON_FOUND)
        add_library(rapidjson INTERFACE)
        target_include_directories(rapidjson INTERFACE ${RapidJSON_INCLUDE_DIRS})
    endif()
endif()

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
