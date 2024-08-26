#include "../test_utils.hpp"

#include <simplemc/random/seed_rng.hpp>
#include <simplemc/random/splitmix64.hpp>
#include <simplemc/random/xoshiro256.hpp>

#include <random>
#include <sstream>

// Test the splitmix64 RNG.
TEST(SimplemcRandom, Splitmix64) {
    simplemc::splitmix64 eng;
    std::uniform_real_distribution dist;
    histogram01 hist(10);
    for (int i = 0; i < 1000000; ++i) {
        hist.add(dist(eng));
    }
    ASSERT_TRUE(hist.check_uniform(1e-2));
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

// Test the xoshiro256 RNGs.
TEST(SimplemcRandom, Xoshiro256) {
    using namespace simplemc;
    xoshiro256p xop;
    xoshiro256pp xopp;
    xoshiro256ss xoss;
    std::uniform_real_distribution urd;
    histogram01 hist_p(10), hist_pp(10), hist_ss(10);
    for (int i = 0; i < 1000000; ++i) {
        hist_p.add(urd(xop));
        hist_pp.add(urd(xopp));
        hist_ss.add(urd(xoss));
    }
    ASSERT_TRUE(hist_p.check_uniform(1e-2));
    ASSERT_TRUE(hist_pp.check_uniform(1e-2));
    ASSERT_TRUE(hist_ss.check_uniform(1e-2));
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

// Test seeding RNGs.
TEST(SimplemcRandom, SeedRng) {
    std::mt19937_64 mt1, mt2, mt3, mt4;
    simplemc::seed_rng(mt1, 0);
    simplemc::seed_rng(mt2, 0);
    simplemc::seed_rng(mt3, 0, 10);
    simplemc::seed_rng(mt4, 2);
    for (int i = 0; i < 1000000; ++i) {
        auto res1 = mt1();
        auto res2 = mt2();
        auto res3 = mt3();
        auto res4 = mt4();
        ASSERT_EQ(res1, res2);
        ASSERT_NE(res1, res3);
        ASSERT_NE(res1, res4);
    }
    simplemc::xoshiro256pp xop1, xop2;
    simplemc::seed_rng(xop1, 0);
    simplemc::seed_rng(xop2, 0);
    ASSERT_EQ(xop1, xop2);
    simplemc::seed_rng(xop2, 1);
    ASSERT_NE(xop1, xop2);
}
