include(UnitTests)

# library specific tests
add_subdirectory(utils)

# temporary test
add_unittest(NAME tmp-test SOURCES tmp_test.cpp LIBS fmt::fmt range-v3::range-v3 nlohmann_json::nlohmann_json)