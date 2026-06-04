# Build tests for simplemc-serialize.
#
# Backend specific tests are built in the respective subdirectories.

add_subdirectory(json)
if(SIMPLEMC_USE_HDF5)
    add_subdirectory(hdf5)
endif()
