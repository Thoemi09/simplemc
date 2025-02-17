#include "../test_utils.hpp"

#include <fmt/ranges.h>
#include <simplemc/random/discrete_alias_distribution.hpp>
#include <simplemc/random/discrete_distribution.hpp>
#include <simplemc/random/uniform_real_distribution.hpp>

#include <random>
#include <sstream>
#include <vector>

// Test distributions.
template <typename T>
void test_dist(T& smc_dist, auto& ref_dist, const typename T::param_type& smc_params, auto const& ref_params) {
    // constructors
    auto def_params = typename T::param_type{};
    T dist_def {};
    T dist_01 { def_params };
    T dist_p { smc_params };
    ASSERT_EQ(dist_def, dist_01);
    ASSERT_NE(dist_01, dist_p);
    ASSERT_EQ(dist_01.param(), def_params);
    ASSERT_EQ(dist_p.param(), smc_params);

    // compare with reference distribution
    std::mt19937_64 std_rng, smc_rng;
    for (int i = 0; i < 1000; ++i) {
        ASSERT_NEAR(ref_dist(std_rng), smc_dist(smc_rng), 1e-14);
        ASSERT_NEAR(ref_dist(std_rng, ref_params), smc_dist(smc_rng, smc_params), 1e-14);
    }

    // stream operator
    std::stringstream ss;
    ss << dist_p;
    ss >> dist_01;
    ASSERT_EQ(dist_p, dist_01);
}

// Test uniform real distribution.
TEST(SimplemcRandom, UniformRealDistribution) {
    using smc_type = simplemc::uniform_real_distribution;
    using std_type = std::uniform_real_distribution<double>;
    auto smc_dist = smc_type {};
    auto smc_params = smc_type::param_type { -2, 1 };
    auto std_dist = std_type {};
    auto std_params = std_type::param_type { -2, 1 };
    test_dist(smc_dist, std_dist, smc_params, std_params);
}

// Test discrete distribution.
TEST(SimplemcRandom, DiscreteDistribution) {
    using smc_type = simplemc::discrete_distribution<int>;
    using std_type = std::discrete_distribution<int>;
    auto smc_dist = smc_type {};
    auto smc_params = smc_type::param_type { 1.0, 5.0, 4.0 };
    auto std_dist = std_type {};
    auto std_params = std_type::param_type { 1.0, 5.0, 4.0 };
    test_dist(smc_dist, std_dist, smc_params, std_params);
}

// Test discrete alias distribution.
TEST(SimplemcRandom, DiscreteAliasDistribution) {
    using smc_type = simplemc::discrete_alias_distribution<long>;
    auto smc_dist = smc_type {};
    auto smc_params = smc_type::param_type { 1.0, 5.0, 4.0 };
    test_dist(smc_dist, smc_dist, smc_params, smc_params);

    // check distribution
    std::mt19937_64 rng;
    smc_dist.param(smc_params);
    std::vector<int> hist(3, 0);
    for (int i = 0; i < 100000; i++) {
        hist[smc_dist(rng)] += 1;
    }
    fmt::print("Histogram: {}\n", hist);
}
