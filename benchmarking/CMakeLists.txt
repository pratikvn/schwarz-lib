set(BENCH_TARGET
  bench_ras
  bench_oras
  )

include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR})

foreach( name ${BENCH_TARGET} )
  add_executable( ${name} ${name}.cpp)
  target_link_libraries(${name} PUBLIC schwarz)
  if(SCHWARZ_BUILD_DEALII)
    DEAL_II_SETUP_TARGET(${name})
  endif()
endforeach()


