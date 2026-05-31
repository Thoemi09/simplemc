# Build serialize sublibrary for simplemc.
#
# The following targets are defined:
# ---------------------------------
# simplemc-serialize-core (simplemc::simplemc-serialize-core): Header-only interface library carrying
# the serializer/deserializer concepts, ADL detection helpers, and library-shipped simplemc_save /
# simplemc_load overloads for simplemc-owned types.
# simplemc-serialize (simplemc::simplemc-serialize): Interface library aggregating core + all
# available backends (currently json; hdf5 in the future).

# add serialize core interface library
add_library(simplemc-serialize-core INTERFACE)
add_library(simplemc::simplemc-serialize-core ALIAS simplemc-serialize-core)
target_link_libraries(simplemc-serialize-core
    INTERFACE
        simplemc::simplemc-utils
        simplemc::simplemc-numeric
        simplemc::simplemc-accs
        simplemc::simplemc-grids
        simplemc::simplemc-random)

# build backend sublibraries
add_subdirectory(json)

# add aggregate interface library — links core + every shipped backend
add_library(simplemc-serialize INTERFACE)
add_library(simplemc::simplemc-serialize ALIAS simplemc-serialize)
target_link_libraries(simplemc-serialize
    INTERFACE
        simplemc::simplemc-serialize-core
        simplemc::simplemc-serialize-json)
