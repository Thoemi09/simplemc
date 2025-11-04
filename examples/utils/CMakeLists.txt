# Build examples codes that use simplemc::simplemc-utils.
#
# The following targets are defined:
# ---------------------------------
# doc_file_io: Doc example using file_handle and open_file.

add_executable(doc_file_io doc_file_io.cpp)
target_link_libraries(doc_file_io PRIVATE simplemc::simplemc)