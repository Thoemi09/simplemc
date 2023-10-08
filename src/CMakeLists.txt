# generate config.h header file
configure_file(config.h.in ${PROJECT_BINARY_DIR}/include/simplemc/config.h)

# add interface library for warnings
add_library(warnings INTERFACE)
add_library(simplemc::warnings ALIAS warnings)
target_compile_options(warnings 
    INTERFACE
        -Wall
        -Wextra
        -Wfloat-conversion
        -Wpedantic)