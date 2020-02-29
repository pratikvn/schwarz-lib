set(TEST_EXECS
  smat_gen
  smat_gen2
  )

foreach( name ${TEST_EXECS} )
  add_executable( ${name} ${name}.cpp $<TARGET_OBJECTS:catch_main> )
endforeach()


# target_link_libraries(${EXEC} schwarz)
