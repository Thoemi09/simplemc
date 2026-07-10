# Build example codes that use simplemc::simplemc-serialize.

add_subdirectory(json)

if(SIMPLEMC_USE_HDF5)
    add_subdirectory(hdf5)
    add_subdirectory(variant)
endif()
