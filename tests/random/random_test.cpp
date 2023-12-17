/**
 * @file random_test.cpp
 * @brief Unit tests for simplemc-random.
 */

#include "../test_utils.hpp"

#include <simplemc/random.hpp>
#include <simplemc/utils/timer.hpp>

#include <fmt/ranges.h>

#include <cstdint>
#include <random>
#include <sstream>
#include <vector>

// Simple histogram class for testing.
class histogram01 {
public:
    explicit histogram01(int nbins) : nbins_(nbins), step_(1.0 / nbins), hist_(nbins, 0.0) {}

    void add(double value) {
        int idx = static_cast<int>(value / step_);
        hist_[idx] += 1.0;
        nsamples_ += 1;
    }

    void print() const { fmt::print("Histogram:\n{}\n", hist_); }

    [[nodiscard]] bool check_bins(double tol) const {
        double exact = static_cast<double>(nsamples_) / nbins_;
        double err = tol * exact;
        double lower = exact - err;
        double upper = exact + err;
        bool res = true;
        for (int i = 0; i < nbins_; i++) {
            if (hist_[i] < lower || hist_[i] > upper) {
                res = false;
                break;
            }
        }
        return res;
    }

private:
    long nsamples_ { 0 };
    int nbins_;
    double step_;
    std::vector<double> hist_;
};

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

// Test splitmix64 RNG.
TEST(SimplemcRandom, Splitmix64) {
    simplemc::splitmix64 eng;
    simplemc::uniform_real_distribution dist;
    histogram01 hist(10);
    for (int i = 0; i < 1000000; ++i) {
        hist.add(dist(eng));
    }
    ASSERT_TRUE(hist.check_bins(1e-2));
}

// Test restoring a splitmix64 RNG.
TEST(SimplemcRandom, RestoreSplitmix64) {
    simplemc::splitmix64 eng;
    auto eng2 = eng;
    ASSERT_EQ(eng, eng2);
    eng.discard(100);
    ASSERT_NE(eng, eng2);
    std::stringstream ss;
    ss << eng;
    ss >> eng2;
    ASSERT_EQ(eng, eng2);
}

// Test xoshiro256 RNGs.
TEST(SimplemcRandom, Xoshiro256) {
    using namespace simplemc;
    xoshiro256p xop;
    xoshiro256pp xopp;
    xoshiro256ss xoss;
    uniform_real_distribution urd;
    histogram01 hist_p(10), hist_pp(10), hist_ss(10);
    for (int i = 0; i < 1000000; ++i) {
        hist_p.add(urd(xop));
        hist_pp.add(urd(xopp));
        hist_ss.add(urd(xoss));
    }
    ASSERT_TRUE(hist_p.check_bins(1e-2));
    ASSERT_TRUE(hist_pp.check_bins(1e-2));
    ASSERT_TRUE(hist_ss.check_bins(1e-2));
}

// Test restoring a xoshiro256 RNG.
TEST(SimplemcRandom, RestoreXoshiro256) {
    simplemc::xoshiro256p eng;
    auto eng2 = eng;
    ASSERT_EQ(eng, eng2);
    eng.jump();
    ASSERT_NE(eng, eng2);
    std::stringstream ss;
    ss << eng;
    ss >> eng2;
    ASSERT_EQ(eng, eng2);
}

// Test discrete distribution.
TEST(SimplemcRandom, DiscreteDistribution) {
    simplemc::discrete_distribution<int> smc_dist { 1.0, 5.0, 4.0 };
    std::discrete_distribution<int> std_dist { 1.0, 5.0, 4.0 };
    simplemc::xoshiro256ss xop1, xop2;
    for (int i = 0; i < 100; i++) {
        ASSERT_EQ(smc_dist(xop1), std_dist(xop2));
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

// Test restoring a discrete alias distribution.
TEST(SimplemcRandom, RestoreDiscreteAliasDistribution) {
    simplemc::discrete_alias_distribution<long> dist1 { 1.23, 8.2912, 4.23 }, dist2;
    ASSERT_NE(dist1, dist2);
    std::stringstream ss;
    ss << dist1;
    ss >> dist2;
    check_range_near(dist1.probabilities(), dist2.probabilities(), 1e-14);
}

// Test speeds of different distributions.
TEST(SimplemcRandom, SpeedDiscreteDistribution) {
    using namespace simplemc;
    // std::vector<double> weights { 1, 3, 5, 0, 2, 6, 18, 2, 11, 8 };
    std::vector<double> weights { 1, 6, 3 };
    std::vector<int> hist(weights.size(), 0);
    xoshiro256p xop;
    int n = 1000000;
    discrete_distribution<int> smc_dist(weights.begin(), weights.end());
    timer t;
    t.start();
    for (int i = 0; i < n; ++i) {
        hist[smc_dist(xop)] += 1;
    }
    t.stop();
    fmt::print("Runtime simplemc::discrete_distribution = {}\n", time_passed(t.start_time(), t.stop_time()));
    fmt::print("{}\n", hist);
    std::vector<int> hist2(weights.size(), 0);
    std::discrete_distribution<int> std_dist(weights.begin(), weights.end());
    t.start();
    for (int i = 0; i < n; ++i) {
        hist2[std_dist(xop)] += 1;
    }
    t.stop();
    fmt::print("Runtime std::discrete_distribution = {}\n", time_passed(t.start_time(), t.stop_time()));
    fmt::print("{}\n", hist2);
    std::vector<int> hist3(weights.size(), 0);
    discrete_alias_distribution<int> smc_al_dist(weights.begin(), weights.end());
    t.start();
    for (int i = 0; i < n; ++i) {
        hist3[smc_al_dist(xop)] += 1;
    }
    t.stop();
    fmt::print("Runtime simplemc::discrete_alias_distribution = {}\n", time_passed(t.start_time(), t.stop_time()));
    fmt::print("{}\n", hist3);
}

// Test exclusive uniform int distribution.
TEST(SimplemcRandom, ExclusiveUniformIntDistributionSamples) {
    simplemc::xoshiro256ss eng;
    int min = 0;
    int max = 3;
    int exclude = 1;
    for (int i = 0; i < 100; ++i) {
        auto res = exclusive_uniform_int_sample(eng, min, max, exclude);
        ASSERT_TRUE(res >= min && res <= max && res != exclude);
    }
    min = -2;
    max = 2;
    exclude = 1;
    for (int i = 0; i < 100; ++i) {
        auto res = exclusive_uniform_int_sample(eng, min, max, exclude);
        ASSERT_TRUE(res >= min && res <= max && res != exclude);
    }
    min = -4;
    max = -1;
    exclude = -2;
    for (int i = 0; i < 100; ++i) {
        auto res = exclusive_uniform_int_sample(eng, min, max, exclude);
        ASSERT_TRUE(res >= min && res <= max && res != exclude);
    }
}

// Test seeding RNGs.
TEST(SimplemcRandom, SeedRng) {
    std::mt19937_64 mt1, mt2, mt3;
    simplemc::seed_rng(mt1, 0);
    simplemc::seed_rng(mt2, 0);
    simplemc::seed_rng(mt3, 2);
    for (int i = 0; i < 1000000; ++i) {
        auto res1 = mt1();
        auto res2 = mt2();
        auto res3 = mt3();
        ASSERT_EQ(res1, res2);
        ASSERT_NE(res1, res3);
    }
    simplemc::xoshiro256pp xop1, xop2;
    simplemc::seed_rng(xop1, 0);
    simplemc::seed_rng(xop2, 0);
    ASSERT_EQ(xop1, xop2);
    simplemc::seed_rng(xop2, 1);
    ASSERT_NE(xop1, xop2);
}
