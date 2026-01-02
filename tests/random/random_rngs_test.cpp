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
    ASSERT_EQ(rng_def.internal_state(), rng_seed.internal_state());
    ASSERT_NE(rng_def.internal_state(), rng_seq.internal_state());

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

    // min/max
    ASSERT_EQ(T::min(), 0ULL);
    ASSERT_EQ(T::max(), std::numeric_limits<std::uint64_t>::max());
}

// Test the splitmix64 RNG.
TEST(SimplemcRandom, Splitmix64) {
    test_rng<simplemc::splitmix64>();
}

// Test the splitmix64 constexpr RNG.
TEST(SimplemcRandom, Splitmix64Constexpr) {
    constexpr simplemc::splitmix64 rng { 42 };
    constexpr auto state = rng.internal_state();
    static_assert(state == 42);
}

// Test xoshiro256 RNGs.
TEST(SimplemcRandom, Xoshiro256) {
    test_rng<simplemc::xoshiro256p>();
    test_rng<simplemc::xoshiro256pp>();
    test_rng<simplemc::xoshiro256ss>();
}

// Test xoshiro256 constexpr RNG.
TEST(SimplemcRandom, Xoshiro256Constexpr) {
    constexpr simplemc::xoshiro256ss rng_constexpr { 1, 2, 3, 4 };
    constexpr auto cstate = rng_constexpr.internal_state();
    static_assert(cstate[0] == 1);
    static_assert(cstate[1] == 2);
    static_assert(cstate[2] == 3);
    static_assert(cstate[3] == 4);
}

// Test xoshiro256 specific methods.
TEST(SimplemcRandom, Xoshiro256Specific) {
    // test four-value constructor
    simplemc::xoshiro256ss rng1 { 1, 2, 3, 4 };
    const auto& state = rng1.internal_state();
    ASSERT_EQ(state[0], 1ULL);
    ASSERT_EQ(state[1], 2ULL);
    ASSERT_EQ(state[2], 3ULL);
    ASSERT_EQ(state[3], 4ULL);

    // test four-value seed
    simplemc::xoshiro256ss rng2 {};
    rng2.seed(1, 2, 3, 4);
    ASSERT_EQ(rng1, rng2);
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
