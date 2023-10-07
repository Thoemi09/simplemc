# Define install directories from GNUInstallDirs module, rpaths and default install prefix.
#
# The following variables are set:
# ---------------------------------
# CMAKE_INSTALL_PREFIX: Default install directory if not specified by user.
# CMAKE_INSTALL_RPATH: Installation rpath according to GNUInstallDirs.
# CMAKE_INSTALL_RPATH_USE_LINK_PATH: ON.

include(GNUInstallDirs)

# install directory
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install CACHE PATH "Install directory" FORCE)
endif()

# rpaths
file(RELATIVE_PATH relDir
    ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
    ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH $ORIGIN $ORIGIN/${relDir})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
