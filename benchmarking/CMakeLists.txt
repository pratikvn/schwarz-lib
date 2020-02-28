set(BENCH_TARGET "bench_ras")

include_directories(
  include
  ${CMAKE_BINARY_DIR}/schwarz-lib/include
  ${CMAKE_SOURCE_DIR}/schwarz-lib/include)

add_executable(${BENCH_TARGET} bench_ras.cpp)

target_link_libraries(${BENCH_TARGET} PUBLIC schwarz)
if(SCHWARZ_BUILD_DEALII)
  DEAL_II_SETUP_TARGET(${BENCH_TARGET})
endif()
