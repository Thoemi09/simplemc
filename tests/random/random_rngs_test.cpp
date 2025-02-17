#include "../test_utils.hpp"

#include <simplemc/random/seed_rng.hpp>
#include <simplemc/random/splitmix64.hpp>
#include <simplemc/random/xoshiro256.hpp>

#include <random>
#include <sstream>

// Test random number generators.
template <typename T>
void test_rng() {
    // constructors
    T rng_def {};
    T rng_seed { T::default_seed };
    auto seq = std::seed_seq { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    T rng_seq { seq };
    ASSERT_EQ(rng_def, rng_seed);
    ASSERT_NE(rng_def, rng_seq);

    // discard
    const auto n = 1000000;
    rng_seed.discard(n);
    ASSERT_NE(rng_def, rng_seed);

    // random number generation
    std::uniform_real_distribution dist;
    histogram hist { 0, 1, 10 };
    for (int i = 0; i < n; ++i) {
        hist.add(dist(rng_def));
    }
    hist.print([](auto) { return 1.0; });
    ASSERT_TRUE(hist.check([](auto) { return 1.0; }, 1e-2));
    ASSERT_EQ(rng_def, rng_seed);

    // stream operator
    std::stringstream ss;
    ss << rng_def;
    ss >> rng_seq;
    ASSERT_EQ(rng_def, rng_seq);
}

// Test the splitmix64 RNG.
TEST(SimplemcRandom, Splitmix64) {
    test_rng<simplemc::splitmix64>();
}

// Test xoshiro256 RNGs.
TEST(SimplemcRandom, Xoshiro256) {
    test_rng<simplemc::xoshiro256p>();
    test_rng<simplemc::xoshiro256pp>();
    test_rng<simplemc::xoshiro256ss>();
}

// Test xoshiro256's jump method.
TEST(SimplemcRandom, Xoshiro256Jump) {
    simplemc::xoshiro256p rng {}, rng2 {};
    ASSERT_EQ(rng, rng2);
    rng.jump();
    ASSERT_NE(rng, rng2);
    rng2.jump();
    ASSERT_EQ(rng, rng2);
    rng.long_jump();
    ASSERT_NE(rng, rng2);
    rng2.long_jump();
    ASSERT_EQ(rng, rng2);
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
