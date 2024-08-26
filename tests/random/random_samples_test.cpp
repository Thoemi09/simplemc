#include "../test_utils.hpp"

#include <simplemc/random/samples.hpp>

#include <random>

// Test exclusive uniform int distribution.
TEST(SimplemcRandom, ExclusiveUniformIntDistributionSamples) {
    std::mt19937_64 eng;
    int min = 0;
    int max = 3;
    int exclude = 1;
    for (int i = 0; i < 100; ++i) {
        auto res = simplemc::exclusive_uniform_int_sample(eng, min, max, exclude);
        ASSERT_TRUE(res >= min && res <= max && res != exclude);
    }
    min = -2;
    max = 2;
    exclude = 1;
    for (int i = 0; i < 100; ++i) {
        auto res = simplemc::exclusive_uniform_int_sample(eng, min, max, exclude);
        ASSERT_TRUE(res >= min && res <= max && res != exclude);
    }
    min = -4;
    max = -1;
    exclude = -2;
    for (int i = 0; i < 100; ++i) {
        auto res = simplemc::exclusive_uniform_int_sample(eng, min, max, exclude);
        ASSERT_TRUE(res >= min && res <= max && res != exclude);
    }
}
