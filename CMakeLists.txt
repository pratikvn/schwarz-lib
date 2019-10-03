SET(TARGET "schwarz")
CMAKE_MINIMUM_REQUIRED(VERSION 3.8)

PROJECT(${TARGET} LANGUAGES C CXX)

option(SCHWARZ_BUILD_BENCHMARKING "Build an example to benchmark the library" ON)
option(SCHWARZ_BUILD_METIS "Use the METIS partition library" OFF)
option(SCHWARZ_BUILD_CHOLMOD "Use the CHOLMOD direct solver" OFF)
option(SCHWARZ_BUILD_CLANG_TIDY "Use clang-tidy for static checks of code." OFF)
option(SCHWARZ_BUILD_CUDA "Build the CUDA kernels" OFF)
option(SCHWARZ_BUILD_DEALII "Build with deal.ii support" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(SCHWARZ_BUILD_CHOLMOD)
  if(NOT DEFINED ENV{CHOLMOD_DIR})
    message(SEND_ERROR "Please specify where CHOLMOD has been installed.")
  endif()
  set(CHOLMOD_INCLUDE_DIR $ENV{CHOLMOD_DIR}/include)
  set(CHOLMOD_LIBRARY_DIR $ENV{CHOLMOD_DIR}/lib)
  include_directories(${CHOLMOD_INCLUDE_DIR})
  link_directories(${CHOLMOD_LIBRARY_DIR})
endif()

if(SCHWARZ_BUILD_METIS)
  if(NOT DEFINED ENV{METIS_DIR})
    message(SEND_ERROR "Please specify where METIS has been installed.")
  endif()
  set(METIS_INCLUDE_DIR $ENV{METIS_DIR}/include)
  set(METIS_LIBRARY_DIR $ENV{METIS_DIR}/lib)
  include_directories(${METIS_INCLUDE_DIR})
  link_directories(${METIS_LIBRARY_DIR})
endif()


find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

set(BOOST_ROOT $ENV{BOOST_DIR})
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/include)

ADD_LIBRARY(schwarz SHARED)

if(SCHWARZ_BUILD_CUDA)
  enable_language(CUDA)
  set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" PARENT_SCOPE)
  set(CMAKE_CUDA_COMPILER_VERSION ${CMAKE_CUDA_COMPILER_VERSION} PARENT_SCOPE)
  set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} PARENT_SCOPE)

  find_library(CUDA_RUNTIME_LIBS cudart
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  find_library(CUBLAS cublas
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  find_library(CUSPARSE cusparse
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  
endif()

TARGET_SOURCES(schwarz PRIVATE 
  source/initialization.cpp
  source/communicate.cpp
  source/utils.cpp
  source/solve.cpp
  source/schwarz_solver.cpp
  source/exception.cpp
  )

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/schwarz/config.hpp @ONLY)

if(SCHWARZ_BUILD_CUDA)
  target_sources(schwarz PRIVATE
    source/cuda_gather.cu)
endif()

if(USE_CLANG_TIDY)
  find_program(
    CLANG_TIDY_EXE
    NAMES "clang-tidy"
    DOC "Path to clang-tidy executable"
    )
endif()

if(NOT CLANG_TIDY_EXE)
  message(STATUS "clang-tidy not found.")
else()
  message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
  set(DO_CLANG_TIDY "${CLANG_TIDY_EXE}" "-checks=*,-fuchsia-default-arguments,-clang-analyzer-alpha.*")
endif()

include(cmake/package_helpers.cmake)
ginkgo_find_package(gflags gflags FALSE 2.2.2)
add_subdirectory(third_party)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

find_package(easy_profiler)

find_package(HWLOC)

# target_compile_options(schwarz PRIVATE "-fext-numeric-literals")

set_target_properties(
  schwarz PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  COMPILE_FLAGS "${WARNING_FLAGS}"
  )

if(CLANG_TIDY_EXE)
  set_target_properties(
    schwarz PROPERTIES
    CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
  )
endif()

find_package(Ginkgo REQUIRED)

if(SCHWARZ_BUILD_DEALII)
  TARGET_LINK_LIBRARIES(schwarz dealii)
endif()


if(SCHWARZ_BUILD_METIS)
TARGET_LINK_LIBRARIES(schwarz metis)
endif()
if(SCHWARZ_BUILD_CHOLMOD)
  TARGET_LINK_LIBRARIES(schwarz cholmod)
endif()

TARGET_LINK_LIBRARIES(schwarz Ginkgo::ginkgo gflags )

TARGET_LINK_LIBRARIES(schwarz ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
if(SCHWARZ_BUILD_CUDA)
  target_include_directories(schwarz
    SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(schwarz ${CUDA_RUNTIME_LIBS} ${CUBLAS} ${CUSPARSE})
endif()
if(HWLOC_FOUND)
  TARGET_INCLUDE_DIRECTORIES(schwarz PRIVATE ${HWLOC_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(schwarz ${HWLOC_LIBRARIES})
endif()

if(SCHWARZ_BUILD_BENCHMARKING)
  add_subdirectory(benchmarking)
endif()

if(easy_profiler_FOUND)
  TARGET_LINK_LIBRARIES(schwarz easy_profiler)
endif()
