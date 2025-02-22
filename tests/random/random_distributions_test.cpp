#include "../test_utils.hpp"

#include <fmt/ranges.h>
#include <simplemc/random/discrete_alias_distribution.hpp>
#include <simplemc/random/discrete_distribution.hpp>
#include <simplemc/random/uniform_real_distribution.hpp>

#include <random>
#include <sstream>

// Test distributions.
template <typename T>
void test_dist(T& smc_dist, const typename T::param_type& smc_params, auto hist, auto ref_fun) {
    // constructors
    auto def_params = typename T::param_type {};
    T dist_def {};
    T dist_01 { def_params };
    T dist_p { smc_params };
    ASSERT_EQ(dist_def, dist_01);
    ASSERT_NE(dist_01, dist_p);
    ASSERT_EQ(dist_01.param(), def_params);
    ASSERT_EQ(dist_p.param(), smc_params);

    // check distribution
    std::mt19937_64 rng;
    smc_dist.param(smc_params);
    for (int i = 0; i < 1000000; i++) {
        hist.add(smc_dist(rng));
    }
    hist.print(ref_fun);
    ASSERT_TRUE(hist.check(ref_fun, 1e-2));

    // stream operator
    std::stringstream ss;
    ss << dist_p;
    ss >> dist_01;
    ASSERT_EQ(dist_p, dist_01);
}

// Test uniform real distribution.
TEST(SimplemcRandom, UniformRealDistribution) {
    using smc_type = simplemc::uniform_real_distribution;
    auto smc_dist = smc_type {};
    auto smc_params = smc_type::param_type { -2, 1 };
    test_dist(smc_dist, smc_params, histogram { -2, 1, 15 }, [](auto) { return 1.0 / 3; });
}

// Test discrete distribution.
TEST(SimplemcRandom, DiscreteDistribution) {
    using smc_type = simplemc::discrete_distribution<int>;
    auto smc_dist = smc_type {};
    auto smc_params = smc_type::param_type { 1.0, 5.0, 4.0 };
    test_dist(smc_dist, smc_params, histogram { -0.5, 2.5, 3 },
        [&smc_params](auto i) { return smc_params.probabilities()[static_cast<std::size_t>(i)]; });
}

// Test discrete alias distribution.
TEST(SimplemcRandom, DiscreteAliasDistribution) {
    using smc_type = simplemc::discrete_alias_distribution<long>;
    auto smc_dist = smc_type {};
    auto smc_params = smc_type::param_type { 1.0, 5.0, 4.0 };
    test_dist(smc_dist, smc_params, histogram { -0.5, 2.5, 3 }, [&smc_params](auto i) {
        return smc_params.probabilities()[static_cast<std::size_t>(i)];
        ;
    });
}
