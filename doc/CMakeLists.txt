#########################
# Doxygen
#########################
find_package(Doxygen REQUIRED)

# configure the Doxyfile
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

# configure markdown files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/overview.md.in ${CMAKE_CURRENT_BINARY_DIR}/overview.md @ONLY)

# Fetch doxygen-awesome-css
include(FetchContent)
FetchContent_Declare(doxyawesome
  GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
  GIT_TAG main)
FetchContent_MakeAvailable(doxyawesome)
file(COPY ${doxyawesome_SOURCE_DIR}/doxygen-awesome.css DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${doxyawesome_SOURCE_DIR}/doxygen-awesome-sidebar-only.css DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# add custom target for Doxygen
add_custom_target(doxygen ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating Doxygen documentation"
    VERBATIM
)
