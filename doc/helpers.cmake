# configures the file <in> into the variable <variable>
function(schwarz_configure_to_string in variable)
    set(fin "${in}")
    file(READ "${fin}" str)
    string(CONFIGURE "${str}" str_conf)
    set(${variable} "${str_conf}" PARENT_SCOPE)
endfunction()

macro(schwarz_to_string variable)
  set(${variable} "")
  foreach(var  ${ARGN})
    set(${variable} "${${variable}} ${var}")
  endforeach()
  string(STRIP "${${variable}}" ${variable})
endmacro()

# writes the concatenated configured files <in1,2>
# in <base_in> into <out>
function(schwarz_file_concat base_path in1 in2 out)
    schwarz_configure_to_string("${base_path}/${in1}" s1)
    schwarz_configure_to_string("${base_path}/${in2}" s2)
    string(CONCAT so "${s1}" "\n" "${s2}")
    file(WRITE "${out}" "${so}")
endfunction()

# adds a pdflatex build step
function(schwarz_doc_pdf name path)
    add_custom_command(TARGET "${name}" POST_BUILD
        COMMAND make
        COMMAND "${CMAKE_COMMAND}" -E copy refman.pdf
        "${CMAKE_CURRENT_BINARY_DIR}/${name}.pdf"
        WORKING_DIRECTORY "${path}"
        COMMENT "Generating ${name} PDF from LaTeX"
        VERBATIM
        )
endfunction()


# generates the documentation named <name> with the additional
# config file <in> in <pdf/html> format
function(schwarz_doc_gen name in pdf mainpage-in)
    set(DIR_BASE "${CMAKE_SOURCE_DIR}")
    set(DOC_BASE "${CMAKE_CURRENT_SOURCE_DIR}")
    set(DIR_SCRIPT "${DOC_BASE}/scripts")
    set(DIR_OUT "${CMAKE_CURRENT_BINARY_DIR}/${name}")
    set(MAINPAGE "${DIR_OUT}/MAINPAGE-${name}.md")
    set(doxyfile "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile-${name}")
    set(layout "${DOC_BASE}/DoxygenLayout.xml")
    schwarz_file_concat("${DOC_BASE}/pages"
      "${mainpage-in}" BASE_DOC.md "${MAINPAGE}"
      )
    set(doxygen_base_input
      "${DOC_BASE}/headers/"
      )
    list(APPEND doxygen_base_input
      ${CMAKE_BINARY_DIR}/include/schwarz/config.hpp
      ${DIR_BASE}/include
      ${DIR_BASE}/source
      ${MAINPAGE}
      )
    set(doxygen_dev_input
      "${DIR_BASE}/include"
      )
    set(doxygen_image_path "${CMAKE_SOURCE_DIR}/doc/images/")
    file(GLOB doxygen_depend
      ${DOC_BASE}/headers/*.hpp
      ${DIR_BASE}/include/schwarz/*.hpp
      )
    list(APPEND doxygen_depend
      ${CMAKE_BINARY_DIR}/include/schwarz/config.hpp
      )
    list(APPEND doxygen_dev_input
      ${doxygen_base_input}
      )
    # pick some markdown files we want as pages
    set(doxygen_markdown_files "../../INSTALL.md ../../TESTING.md ../../benchmarking/BENCHMARKING.md")
    schwarz_to_string(doxygen_base_input_str ${doxygen_base_input} )
    schwarz_to_string(doxygen_dev_input_str ${doxygen_dev_input} )
    schwarz_to_string(doxygen_image_path_str ${doxygen_image_path} )
    add_custom_target("${name}" ALL
      #DEPEND "${doxyfile}.stamp" Doxyfile.in ${in} ${in2}
        COMMAND "${DOXYGEN_EXECUTABLE}" ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS
        ${doxyfile}
        ${layout}
        ${doxygen_depend}
        #COMMAND "${CMAKE_COMMAND}" cmake -E touch "${doxyfile}.stamp"
        COMMENT "Generating ${name} documentation with Doxygen"
        VERBATIM
        )
    if(pdf)
        schwarz_doc_pdf("${name}" "${DIR_OUT}")
    endif()
    schwarz_file_concat("${DOC_BASE}/conf"
        Doxyfile.in "${in}" "${doxyfile}"
        )
endfunction()
