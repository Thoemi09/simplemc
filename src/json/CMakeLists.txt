# Build json library for simplemc.
# 
# The following targets are defined:
# ---------------------------------
# simplemc-json (simplemc::simplemc-json)

add_library(simplemc-json)
add_library(simplemc::simplemc-json ALIAS simplemc-json)
target_sources(simplemc-json PRIVATE file_io.cpp)
target_link_libraries(simplemc-json
    PUBLIC
        simplemc::simplemc-utils
        nlohmann_json::nlohmann_json)
