#include "../test_utils.hpp"

#include <fmt/ranges.h>
#include <simplemc/random/discrete_alias_distribution.hpp>
#include <simplemc/random/discrete_distribution.hpp>
#include <simplemc/random/uniform_real_distribution.hpp>

#include <random>
#include <sstream>
#include <vector>

// Test uniform real distribution.
TEST(SimplemcRandom, UniformRealDistribution) {
    std::mt19937_64 mt_std, mt_smc;
    std::uniform_real_distribution<double> std_urd;
    simplemc::uniform_real_distribution smc_urd;
    for (int i = 0; i < 1000; ++i) {
        ASSERT_NEAR(std_urd(mt_std), smc_urd(mt_smc), 1e-14);
    }
}

// Test restoring a uniform real distribution.
TEST(SimplemcRandom, RestoreUniformRealDistribution) {
    simplemc::uniform_real_distribution dist(-2.0, 5.0), dist2;
    ASSERT_NE(dist, dist2);
    std::stringstream ss;
    ss << dist;
    ss >> dist2;
    ASSERT_EQ(dist, dist2);
}

// Test discrete distribution.
TEST(SimplemcRandom, DiscreteDistribution) {
    std::mt19937_64 mt_std, mt_smc;
    simplemc::discrete_distribution<int> smc_dist { 1.0, 5.0, 4.0 };
    std::discrete_distribution<int> std_dist { 1.0, 5.0, 4.0 };
    for (int i = 0; i < 100; i++) {
        ASSERT_EQ(smc_dist(mt_smc), std_dist(mt_std));
    }
}

// Test restoring a discrete distribution.
TEST(SimplemcRandom, RestoreDiscreteDistribution) {
    simplemc::discrete_distribution<std::uint64_t> dist1 { 1.23, 8.2912, 4.23 }, dist2;
    ASSERT_NE(dist1, dist2);
    std::stringstream ss;
    ss << dist1;
    ss >> dist2;
    check_range_near(dist1.probabilities(), dist2.probabilities(), 1e-14);
}

// Test discrete alias distribution.
TEST(SimplemcRandom, DiscreteAliasDistribution) {
    std::mt19937_64 mt_smc;
    simplemc::discrete_alias_distribution<int> smc_dist { 1.0, 5.0, 4.0 };
    std::vector<int> counts(3, 0);
    for (int i = 0; i < 10000; i++) {
        counts[smc_dist(mt_smc)] += 1;
    }
    fmt::print("Counts: {}\n", counts);
}

// Test restoring a discrete alias distribution.
TEST(SimplemcRandom, RestoreDiscreteAliasDistribution) {
    simplemc::discrete_alias_distribution<long> dist1 { 1.23, 8.2912, 4.23 }, dist2;
    ASSERT_NE(dist1, dist2);
    std::stringstream ss;
    ss << dist1;
    ss >> dist2;
    check_range_near(dist1.probabilities(), dist2.probabilities(), 1e-14);
}
