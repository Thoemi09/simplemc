include(UnitTests)

# library specific tests
add_subdirectory(utils)

# temporary test
add_unittest(NAME tmp-test SOURCES tmp_test.cpp LIBS fmt::fmt)