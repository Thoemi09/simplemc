# Build serialization library for simplemc.
#
# The following targets are defined:
# ---------------------------------
# simplemc-serialization (simplemc::simplemc-serialization)

add_library(simplemc-serialization INTERFACE)
add_library(simplemc::simplemc-serialization ALIAS simplemc-serialization)
target_link_libraries(simplemc-serialization
    INTERFACE
        simplemc::simplemc-utils
        $<$<BOOL:${SIMPLEMC_WITH_TOMLPLUSPLUS}>:tomlplusplus::tomlplusplus>
        $<$<BOOL:${SIMPLEMC_WITH_NLOHMANN_JSON}>:nlohmann_json::nlohmann_json>
        $<$<BOOL:${SIMPLEMC_WITH_HIGHFIVE}>:HighFive::HighFive>)
