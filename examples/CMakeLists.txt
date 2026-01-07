# Build examples that use simplemc.
#
# All examples codes from the docs and tutorials are built here to ensure they are up to date.

# Function to create example executables.
# Usage: simplemc_add_examples(target1 target2 ...)
function(simplemc_add_examples)
    foreach(target ${ARGN})
        add_executable(${target} ${target}.cpp)
        target_link_libraries(${target} PRIVATE simplemc::simplemc)
    endforeach()
endfunction()

add_subdirectory(utils)
add_subdirectory(mpi)
add_subdirectory(random)
add_subdirectory(grids)
add_subdirectory(numeric)
add_subdirectory(tutorials)
