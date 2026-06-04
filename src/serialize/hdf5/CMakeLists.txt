# Build serialize-hdf5 sublibrary for simplemc.
#
# The following targets are defined:
# ---------------------------------
# simplemc-serialize-hdf5 (simplemc::simplemc-serialize-hdf5): Header-only interface library
# providing the HDF5 backend (hdf5_serializer) on top of HighFive.

add_library(simplemc-serialize-hdf5 INTERFACE)
add_library(simplemc::simplemc-serialize-hdf5 ALIAS simplemc-serialize-hdf5)
target_link_libraries(simplemc-serialize-hdf5
    INTERFACE
        simplemc::simplemc-serialize-core
        simplemc::simplemc-numeric
        HighFive::HighFive)
