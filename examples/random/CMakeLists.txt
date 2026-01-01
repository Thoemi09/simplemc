# Build example codes that use simplemc::simplemc-random.
#
# The following targets are defined:
# ---------------------------------
# doc_rngs: Doc example using xoshiro256pp together with std::normal_distribution.

add_executable(doc_rngs doc_rngs.cpp)
target_link_libraries(doc_rngs PRIVATE simplemc::simplemc)
