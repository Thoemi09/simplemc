#########################
# Doxygen
#########################
find_package(Doxygen REQUIRED)

# Fetch doxygen-awesome-css
if(SIMPLEMC_USE_DOXYGEN_AWESOME)
    message(STATUS "-----------------------------------------------")
    message(STATUS "Fetching doxygen-awsome ...")
    include(FetchContent)
    FetchContent_Declare(doxyawesome
      GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
      GIT_TAG main)
    FetchContent_MakeAvailable(doxyawesome)
    message(STATUS "doxyawesome_SOURCE_DIR: ${doxyawesome_SOURCE_DIR}")
    message(STATUS "-----------------------------------------------")
    file(COPY ${doxyawesome_SOURCE_DIR}/doxygen-awesome.css DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY ${doxyawesome_SOURCE_DIR}/doxygen-awesome-sidebar-only.css DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    set(DOXYGEN_CSS_FILE ${CMAKE_CURRENT_BINARY_DIR}/doxygen-awesome.css)
    set(DOXYGEN_CSS_SIDEBAR_ONLY_FILE ${CMAKE_CURRENT_BINARY_DIR}/doxygen-awesome-sidebar-only.css)
endif()

# configure the Doxyfile
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

# configure markdown files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/overview.md.in ${CMAKE_CURRENT_BINARY_DIR}/overview.md @ONLY)

# add custom target for Doxygen
add_custom_target(doxygen ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating Doxygen documentation"
    VERBATIM
)
