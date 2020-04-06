include_directories(
  include
  ${CMAKE_BINARY_DIR}/schwarz-lib/include
  ${CMAKE_SOURCE_DIR}/schwarz-lib/include)


if(SCHWARZ_BUILD_DEALII)
  set(BENCH_TARGET "dealii_ex_6;dealii_ex_9")
  FOREACH(_file ${BENCH_TARGET})
    MESSAGE(STATUS "Adding executable for ${_file}")
    add_executable(${_file} ${_file}.cpp)
    DEAL_II_SETUP_TARGET(${_file})
    target_link_libraries(${_file} schwz)
  ENDFOREACH()
else()
  set(BENCH_TARGET "bench_ras")
  add_executable(${BENCH_TARGET} ${BENCH_TARGET}.cpp)
  target_link_libraries(${BENCH_TARGET} schwz)
endif()

