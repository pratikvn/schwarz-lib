macro(schwarz_git_information)
    if(EXISTS "${Schwarz_SOURCE_DIR}/.git")
        find_package(Git QUIET)
        if(GIT_FOUND)
            execute_process(
                COMMAND ${GIT_EXECUTABLE} describe --contains --all HEAD
                WORKING_DIRECTORY ${Schwarz_SOURCE_DIR}
                OUTPUT_VARIABLE SCHWARZ_GIT_BRANCH
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            execute_process(
                COMMAND ${GIT_EXECUTABLE} log -1 --format=%H ${Gingko_SOURCE_DIR}
                WORKING_DIRECTORY ${Schwarz_SOURCE_DIR}
                OUTPUT_VARIABLE SCHWARZ_GIT_REVISION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            execute_process(
                COMMAND ${GIT_EXECUTABLE} log -1 --format=%h ${Gingko_SOURCE_DIR}
                WORKING_DIRECTORY ${Schwarz_SOURCE_DIR}
                OUTPUT_VARIABLE SCHWARZ_GIT_SHORTREV
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        endif()
    endif()
endmacro(schwarz_git_information)
