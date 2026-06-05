# Build serialize sublibrary for simplemc.
#
# The following targets are defined:
# ---------------------------------
# simplemc-serialize-core (simplemc::simplemc-serialize-core): Provides the serializer concepts, ADL 
# detection helpers, and library-shipped simplemc_save / simplemc_load overloads for simplemc-owned 
# types.
# simplemc-serialize (simplemc::simplemc-serialize): Interface library aggregating core + all
# available backends (json, hdf5).

# add serialize core interface library
add_library(simplemc-serialize-core INTERFACE)
add_library(simplemc::simplemc-serialize-core ALIAS simplemc-serialize-core)
target_link_libraries(simplemc-serialize-core
    INTERFACE
        simplemc::simplemc-utils)

# build backend sublibraries
add_subdirectory(json)
if(SIMPLEMC_USE_HDF5)
    add_subdirectory(hdf5)
endif()

# add aggregate interface library — links core + every shipped backend
add_library(simplemc-serialize INTERFACE)
add_library(simplemc::simplemc-serialize ALIAS simplemc-serialize)
target_link_libraries(simplemc-serialize
    INTERFACE
        simplemc::simplemc-serialize-core
        simplemc::simplemc-serialize-json
        $<$<BOOL:${SIMPLEMC_USE_HDF5}>:simplemc::simplemc-serialize-hdf5>)
