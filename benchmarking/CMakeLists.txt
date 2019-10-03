include_directories(
  include
  ${CMAKE_BINARY_DIR}/schwarz-lib/include
  ${CMAKE_SOURCE_DIR}/schwarz-lib/include)

add_executable(poisson_ras poisson-ras.cpp)

target_link_libraries(poisson_ras schwarz)
