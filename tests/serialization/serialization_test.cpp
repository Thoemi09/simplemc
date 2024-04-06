#include <gtest/gtest.h>

#include <simplemc/config.hpp>

#include <fmt/format.h>

// Test if the config header is generated correctly.
TEST(SimplemcSerialization, ConfigHeader) {
#ifdef SIMPLEMC_WITH_TOMLPLUSPLUS
    fmt::print("Built with toml++ support\n");
#endif
#ifdef SIMPLEMC_WITH_NLOHMANN_JSON
    fmt::print("Built with nlohmann_json support\n");
#endif
#ifdef SIMPLEMC_WITH_HIGHFIVE
    fmt::print("Built with HighFive support\n");
#endif
}
