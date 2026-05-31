# Build serialize-json sublibrary for simplemc.
#
# The following targets are defined:
# ---------------------------------
# simplemc-serialize-json (simplemc::simplemc-serialize-json): Compiled library providing the JSON
# backend (json_serializer, json_deserializer) plus the relocated JSON file IO helpers.

add_library(simplemc-serialize-json)
add_library(simplemc::simplemc-serialize-json ALIAS simplemc-serialize-json)
target_sources(simplemc-serialize-json PRIVATE file_io.cpp)
target_link_libraries(simplemc-serialize-json
    PUBLIC
        simplemc::simplemc-serialize-core
        simplemc::simplemc-numeric
        nlohmann_json::nlohmann_json)
