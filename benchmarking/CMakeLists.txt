include_directories(
  include
  ${CMAKE_BINARY_DIR}/schwarz-lib/include
  ${CMAKE_SOURCE_DIR}/schwarz-lib/include)

add_executable(bench_ras bench_ras.cpp)

target_link_libraries(bench_ras schwarz)
