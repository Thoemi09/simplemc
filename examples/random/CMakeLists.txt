# Build example codes that use simplemc::simplemc-random.
#
# The following targets are defined:
# ---------------------------------
# doc_rngs: Doc example using xoshiro256pp together with std::normal_distribution.
# doc_splitmix64: Doc example using simplemc::splitmix64.

add_executable(doc_rngs doc_rngs.cpp)
target_link_libraries(doc_rngs PRIVATE simplemc::simplemc)

add_executable(doc_splitmix64 doc_splitmix64.cpp)
target_link_libraries(doc_splitmix64 PRIVATE simplemc::simplemc)
