# Provide functions for adding unittests.

# add_unittest():
# Adds a unittest for given sources and libs.
#
# Parameters:
# ------------
# NAME: name of the test (required)
# SOURCES: source files (required)
# LIBS: libraries for linking (optional)
function(add_unittest)
    # define keyword arguments
    set(prefix AU)
    set(noValues)
    set(singleValues NAME)
    set(multiValues SOURCES LIBS)

    # process the arguments passed in
    include(CMakeParseArguments)
    cmake_parse_arguments(${prefix}
        "${noValues}"
        "${singleValues}"
        "${multiValues}"
        ${ARGN})

    # check the arguments
    if(NOT AU_NAME)
        message(FATAL_ERROR "Missing NAME in the function add_unittest.")
    endif()
    if(NOT AU_SOURCES)
        message(FATAL_ERROR "Missing SOURCES in the function add_unittest.")
    endif()

    # create executable
    add_executable(${AU_NAME} ${AU_SOURCES})
    target_link_libraries(${AU_NAME} ${AU_LIBS})
    
    # add test
    add_test(NAME ${AU_NAME} COMMAND ${AU_NAME})
endfunction()
