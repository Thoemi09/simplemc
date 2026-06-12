# Build serialize-variant sublibrary for simplemc.
#
# The following targets are defined:
# ---------------------------------
# simplemc-serialize-variant (simplemc::simplemc-serialize-variant): A backend-erasing serializer that 
# dispatches to one of several serializer backends at runtime.

add_library(simplemc-serialize-variant INTERFACE)
add_library(simplemc::simplemc-serialize-variant ALIAS simplemc-serialize-variant)
target_link_libraries(simplemc-serialize-variant
    INTERFACE
        simplemc::simplemc-serialize-core)
