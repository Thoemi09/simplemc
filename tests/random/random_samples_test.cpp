#include "../test_utils.hpp"

#include <simplemc/random/samples.hpp>

#include <cmath>
#include <random>

// Test constexpr functions at compile time.
TEST(SimplemcRandom, ConstexprSamples) {
    constexpr double u_sample = simplemc::uniform_sample(0.5, 0.0, 1.0);
    constexpr double u_pdf = simplemc::uniform_pdf(0.0, 1.0);
    ASSERT_DOUBLE_EQ(u_sample, 0.5);
    ASSERT_DOUBLE_EQ(u_pdf, 1.0);

    constexpr double ui_pdf = simplemc::uniform_int_pdf(0, 9);
    ASSERT_DOUBLE_EQ(ui_pdf, 0.1);

    constexpr double eui_pdf = simplemc::exclusive_uniform_int_pdf(0, 10);
    ASSERT_DOUBLE_EQ(eui_pdf, 0.1);
}

// Test uniform distribution on [a, b].
TEST(SimplemcRandom, UniformDistributionSamples) {
    std::mt19937_64 rng {};
    std::uniform_real_distribution<double> dist {};
    const int n = 100000;
    double a {};
    double b {};
    auto pdf = [&](auto) { return simplemc::uniform_pdf(a, b); };

    a = -2.0;
    b = 1.0;
    histogram hist1 { -2, 1, 10 };
    for (int i = 0; i < n; ++i) {
        auto res = simplemc::uniform_sample(dist(rng), a, b);
        ASSERT_TRUE(res >= a && res <= b);
        hist1.add(res);
    }
    hist1.print(pdf);
    ASSERT_TRUE(hist1.check(pdf, 2e-2));

    // edge case: r = 0 gives a
    ASSERT_DOUBLE_EQ(simplemc::uniform_sample(0.0, a, b), a);
}

// Test exponential distribution on [a, \infty).
TEST(SimplemcRandom, ExponentialDistributionSamplesUnbounded) {
    std::mt19937_64 rng {};
    std::uniform_real_distribution<double> dist {};
    const int n = 100000;
    double lambda {};
    double a {};
    auto pdf = [&](auto x) { return simplemc::exponential_pdf(x, lambda, a); };

    lambda = 2;
    a = 3.0;
    histogram hist1 { 3, 6, 10 };
    for (int i = 0; i < n; ++i) {
        auto res = simplemc::exponential_sample(dist(rng), lambda, a);
        ASSERT_TRUE(res >= a);
        hist1.add(res);
    }
    hist1.print(pdf);
    ASSERT_TRUE(hist1.check(pdf, 5e-2));
}

// Test exponential distribution on [a, b].
TEST(SimplemcRandom, ExponentialDistributionSamplesBounded) {
    std::mt19937_64 rng {};
    std::uniform_real_distribution<double> dist {};
    const int n = 100000;
    double lambda {};
    double a {};
    double b {};
    auto pdf = [&](auto x) { return simplemc::exponential_pdf(x, lambda, a, b); };

    lambda = 1.25;
    a = -1.0;
    b = 1.0;
    histogram hist1 { -1, 1, 10 };
    for (int i = 0; i < n; ++i) {
        auto res = simplemc::exponential_sample(dist(rng), lambda, a, b);
        ASSERT_TRUE(res >= a && res <= b);
        hist1.add(res);
    }
    hist1.print(pdf);
    ASSERT_TRUE(hist1.check(pdf, 5e-2));
}

// Test safe exponential distribution.
TEST(SimplemcRandom, SafeExponentialDistributionSamples) {
    std::mt19937_64 rng {};
    std::uniform_real_distribution<double> dist {};
    const int n = 100000;
    double lambda {};
    double a {};
    double b {};
    auto pdf = [&](auto x) { return simplemc::safe_exponential_pdf(x, lambda, a, b); };

    lambda = -1.5;
    a = -5.0;
    b = -2;
    histogram hist1 { -5, -2, 10 };
    for (int i = 0; i < n; ++i) {
        auto res = simplemc::safe_exponential_sample(dist(rng), lambda, a, b);
        ASSERT_TRUE(res >= a && res <= b);
        hist1.add(res);
    }
    hist1.print(pdf);
    ASSERT_TRUE(hist1.check(pdf, 5e-2));

    lambda = 0.0;
    a = 2.0;
    b = 5;
    histogram hist2 { 2, 5, 10 };
    for (int i = 0; i < n; ++i) {
        auto res = simplemc::safe_exponential_sample(dist(rng), lambda, a, b);
        ASSERT_TRUE(res >= a && res <= b);
        hist2.add(res);
    }
    hist2.print(pdf);
    ASSERT_TRUE(hist2.check(pdf, 2e-2));

    lambda = 1.25;
    a = -1.0;
    b = 1.0;
    histogram hist3 { -1, 1, 10 };
    for (int i = 0; i < n; ++i) {
        auto res = simplemc::safe_exponential_sample(dist(rng), lambda, a, b);
        ASSERT_TRUE(res >= a && res <= b);
        hist3.add(res);
    }
    hist3.print(pdf);
    ASSERT_TRUE(hist3.check(pdf, 5e-2));
}

// Test normal distribution.
TEST(SimplemcRandom, NormalDistributionSamples) {
    std::mt19937_64 rng {};
    const int n = 100000;
    double mu {};
    double sigma {};
    auto pdf = [&](auto x) { return simplemc::normal_pdf(x, mu, sigma); };

    mu = -2.0;
    sigma = 1.25;
    histogram hist1 { -5, 1, 10 };
    for (int i = 0; i < n; ++i) {
        hist1.add(simplemc::normal_sample(rng, mu, sigma));
    }
    hist1.print(pdf);
    ASSERT_TRUE(hist1.check(pdf, 1e-1));
}

// Test Cauchy distribution.
TEST(SimplemcRandom, CauchyDistributionSamples) {
    std::mt19937_64 rng {};
    std::uniform_real_distribution<double> dist {};
    const int n = 100000;
    double x0 {};
    double gamma {};
    auto pdf = [&](auto x) { return simplemc::cauchy_pdf(x, x0, gamma); };

    x0 = -2.0;
    gamma = 1.25;
    histogram hist1 { -7, 3, 10 };
    for (int i = 0; i < n; ++i) {
        hist1.add(simplemc::cauchy_sample(dist(rng), x0, gamma));
    }
    hist1.print(pdf);
    ASSERT_TRUE(hist1.check(pdf, 5e-2));
}

// Test uniform int distribution.
TEST(SimplemcRandom, UniformIntDistributionSamples) {
    std::mt19937_64 rng {};
    const int n = 100000;
    int a {};
    int b {};
    auto pdf = [&](auto) { return simplemc::uniform_int_pdf(a, b); };

    a = 5;
    b = 5;
    for (int i = 0; i < 100; ++i) {
        ASSERT_EQ(simplemc::uniform_int_sample(rng, a, b), 5);
    }

    a = 0;
    b = 3;
    histogram hist1 { -0.5, 3.5, 4 };
    for (int i = 0; i < n; ++i) {
        auto res = simplemc::uniform_int_sample(rng, a, b);
        ASSERT_TRUE(res >= a && res <= b);
        hist1.add(res);
    }
    hist1.print(pdf);
    ASSERT_TRUE(hist1.check(pdf, 2e-2));

    a = -2;
    b = 2;
    histogram hist2 { -2.5, 2.5, 5 };
    for (int i = 0; i < n; ++i) {
        auto res = simplemc::uniform_int_sample(rng, a, b);
        ASSERT_TRUE(res >= a && res <= b);
        hist2.add(res);
    }
    hist2.print(pdf);
    ASSERT_TRUE(hist2.check(pdf, 2e-2));
}

// Test exclusive uniform int distribution.
TEST(SimplemcRandom, ExclusiveUniformIntDistributionSamples) {
    std::mt19937_64 rng {};
    const int n = 100000;
    int a {};
    int b {};
    int y {};
    auto pdf = [&](auto x) { return (std::abs(x - y) < 0.1 ? 0.0 : simplemc::exclusive_uniform_int_pdf(a, b)); };

    a = 0;
    b = 3;
    y = 1;
    histogram hist1 { -0.5, 3.5, 4 };
    for (int i = 0; i < n; ++i) {
        auto res = simplemc::exclusive_uniform_int_sample(rng, a, b, y);
        ASSERT_TRUE(res >= a && res <= b && res != y);
        hist1.add(res);
    }
    hist1.print(pdf);
    ASSERT_TRUE(hist1.check(pdf, 2e-2));

    a = -2;
    b = 2;
    y = 1;
    histogram hist2 { -2.5, 2.5, 5 };
    for (int i = 0; i < n; ++i) {
        auto res = simplemc::exclusive_uniform_int_sample(rng, a, b, y);
        ASSERT_TRUE(res >= a && res <= b && res != y);
        hist2.add(res);
    }
    hist2.print(pdf);
    ASSERT_TRUE(hist2.check(pdf, 2e-2));

    a = -4;
    b = -1;
    y = -2;
    histogram hist3 { -4.5, -0.5, 4 };
    for (int i = 0; i < n; ++i) {
        auto res = simplemc::exclusive_uniform_int_sample(rng, a, b, y);
        ASSERT_TRUE(res >= a && res <= b && res != y);
        hist3.add(res);
    }
    hist3.print(pdf);
    ASSERT_TRUE(hist3.check(pdf, 2e-2));
}
