# Build tutorial codes that use simplemc::simplemc-utils.
#
# The following targets are defined:
# ---------------------------------
# tut_utils_3: Tutorial using fmt to print formatted output.

add_executable(tut_utils_3 tut_utils_3.cpp)
target_link_libraries(tut_utils_3 PRIVATE simplemc::simplemc)