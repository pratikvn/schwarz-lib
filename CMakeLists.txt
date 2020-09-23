SET(TARGET "schwz")
CMAKE_MINIMUM_REQUIRED(VERSION 3.8)

PROJECT(${TARGET} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SCHWARZ_BUILD_BENCHMARKING "Build an example to benchmark the library" ON)
option(SCHWARZ_BUILD_METIS "Use the METIS partition library" ON)
option(SCHWARZ_BUILD_CHOLMOD "Use the CHOLMOD direct solver" OFF)
option(SCHWARZ_BUILD_UMFPACK "Use the UMFPACK direct solver" OFF)
option(SCHWARZ_BUILD_CUDA "Build the CUDA kernels" ON)
option(SCHWARZ_WITH_HWLOC "Build with hwloc support" ON)
option(SCHWARZ_BUILD_DEALII "Build with deal.ii support" OFF)
option(SCHWARZ_BUILD_CLANG_TIDY "Use clang-tidy for static checks of code." OFF)
option(SCHWARZ_DEVEL_TOOLS "Switch on some developer tools" ON)
option(SCHWARZ_BUILD_DOC "Build the doxygen documentation" OFF)
set(SCHWARZ_CUDA_COMPILER_FLAGS "" CACHE STRING
  "Set the required NVCC compiler flags, mainly used for warnings. Current default is an empty string")

# Optional packages
include(cmake/package_helpers.cmake)
ginkgo_find_package(gflags gflags FALSE 2.2.2)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

set(Schwarz_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
set(Schwarz_BINARY_DIR "${CMAKE_BINARY_DIR}")

set(SCHWARZ_HAVE_CHOLMOD 0)
if(SCHWARZ_BUILD_CHOLMOD)
  if(NOT DEFINED ENV{CHOLMOD_DIR})
    message(FATAL_ERROR "Please specify where CHOLMOD has been installed in CHOLMOD_DIR.")
  endif()
  set(CHOLMOD_INCLUDE_DIR $ENV{CHOLMOD_DIR}/include)
  set(CHOLMOD_LIBRARY_DIR $ENV{CHOLMOD_DIR}/lib)
  include_directories(${CHOLMOD_INCLUDE_DIR})
  link_directories(${CHOLMOD_LIBRARY_DIR})
  set(SCHWARZ_HAVE_CHOLMOD 1)
endif()

set(SCHWARZ_HAVE_UMFPACK 0)
if(SCHWARZ_BUILD_UMFPACK)
  if(NOT DEFINED ENV{UMFPACK_DIR})
    message(FATAL_ERROR "Please specify where UMFPACK has been installed in UMFPACK_DIR.")
  endif()
  set(UMFPACK_INCLUDE_DIR $ENV{UMFPACK_DIR}/include)
  set(UMFPACK_LIBRARY_DIR $ENV{UMFPACK_DIR}/lib)
  include_directories(${UMFPACK_INCLUDE_DIR})
  link_directories(${UMFPACK_LIBRARY_DIR})
  set(SCHWARZ_HAVE_UMFPACK 1)
endif()

set(SCHWARZ_HAVE_METIS 0)
if(SCHWARZ_BUILD_METIS)
  if(NOT DEFINED ENV{METIS_DIR})
    message(FATAL_ERROR "Please specify where METIS has been installed in METIS_DIR.")
  endif()
  set(METIS_INCLUDE_DIR $ENV{METIS_DIR}/include)
  set(METIS_LIBRARY_DIR $ENV{METIS_DIR}/lib)
  message(STATUS "metis inc ${METIS_INCLUDE_DIR}")
  message(STATUS "metis lib ${METIS_LIBRARY_DIR}")
  include_directories(${METIS_INCLUDE_DIR})
  link_directories(${METIS_LIBRARY_DIR})
  set(SCHWARZ_HAVE_METIS 1)
endif()

set(SCHWARZ_HAVE_DEALII 0)
if(SCHWARZ_BUILD_DEALII)
  find_package(deal.II REQUIRED
    HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
  set(SCHWARZ_HAVE_DEALII 1)
  DEAL_II_INITIALIZE_CACHED_VARIABLES()
endif()

if(SCHWARZ_HAVE_DEALII)
  set(TARGET "${TARGET}_dealii")
endif()


set(SCHWARZ_HAVE_HWLOC 0)
if(SCHWARZ_WITH_HWLOC)
  find_package(HWLOC)
  set(SCHWARZ_HAVE_HWLOC 1)
endif()

# Add some matrices for configuration.
add_subdirectory(matrices)


set(SCHWARZ_HAVE_CUDA 0)
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
  set(SCHWARZ_HAVE_CUDA 1)
endif()

# Required packages

find_package(Ginkgo REQUIRED
  HINTS ${Ginkgo_DIR} $ENV{Ginkgo_DIR})

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/include)


ADD_LIBRARY(${TARGET} SHARED )


TARGET_SOURCES(${TARGET} PRIVATE
  source/initialization.cpp
  source/communicate.cpp
  source/exception.cpp
  source/utils.cpp
  source/solve.cpp
  source/schwarz_base.cpp
  source/restricted_schwarz.cpp
  )


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/schwarz/config.hpp @ONLY)

if(SCHWARZ_BUILD_CUDA)
  target_sources(${TARGET} PRIVATE
    source/gather_kernel.cu
    source/scatter_kernel.cu)
  target_compile_options(${TARGET}
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
  target_compile_options(${TARGET}
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${SCHWARZ_CUDA_COMPILER_FLAGS}>)
else()
  target_sources(${TARGET} PRIVATE
    source/cuda_dummy_gather.cpp)
endif()

include(cmake/information_helpers.cmake)
schwarz_git_information()

if(SCHWARZ_BUILD_DOC)
  add_subdirectory(doc)
endif()


if(SCHWARZ_BUILD_CLANG_TIDY)
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

ADD_SUBDIRECTORY(third_party)

SET_TARGET_PROPERTIES(
  ${TARGET} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  COMPILE_FLAGS "${WARNING_FLAGS}"
  )

if(CLANG_TIDY_EXE)
  SET_TARGET_PROPERTIES(
    ${TARGET} PROPERTIES
    CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
    )
endif()

# Link optional packages
if(SCHWARZ_BUILD_METIS)
  TARGET_LINK_LIBRARIES(${TARGET} metis)
endif()

if(SCHWARZ_BUILD_CHOLMOD)
  TARGET_LINK_LIBRARIES(${TARGET} cholmod)
endif()

if(SCHWARZ_BUILD_UMFPACK)
  TARGET_LINK_LIBRARIES(${TARGET} umfpack)
endif()

if(SCHWARZ_WITH_HWLOC)
  if(HWLOC_FOUND)
    TARGET_INCLUDE_DIRECTORIES(${TARGET} PRIVATE ${HWLOC_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(${TARGET} ${HWLOC_LIBRARIES})
  endif()
endif()

if(SCHWARZ_BUILD_DEALII)
  DEAL_II_SETUP_TARGET(${TARGET})
endif()

if(SCHWARZ_BUILD_CUDA)
  TARGET_INCLUDE_DIRECTORIES(${TARGET}
    SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(${TARGET} ${CUDA_RUNTIME_LIBS} ${CUBLAS} ${CUSPARSE})
endif()


# Link the required packages
TARGET_LINK_LIBRARIES(${TARGET} Ginkgo::ginkgo)

TARGET_LINK_LIBRARIES(${TARGET} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})

find_package(OpenMP REQUIRED)
TARGET_LINK_LIBRARIES(${TARGET} ${OpenMP_CXX_LIBRARIES})


if(SCHWARZ_BUILD_BENCHMARKING)
  ADD_SUBDIRECTORY(benchmarking)
  TARGET_LINK_LIBRARIES(${TARGET} gflags)
endif()

# Set this up properly. TODO
# find_package(easy_profiler)
# if(easy_profiler_FOUND)
#   TARGET_LINK_LIBRARIES(${TARGET} easy_profiler)
# endif()
