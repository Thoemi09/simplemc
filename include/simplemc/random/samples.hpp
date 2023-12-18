/**
 * @file samples.hpp
 * @brief Often used probability density functions and random samples from distributions.
 */

#ifndef SIMPLEMC_RANDOM_SAMPLES_HPP
#define SIMPLEMC_RANDOM_SAMPLES_HPP

#include <cassert>
#include <cmath>
#include <numbers>
#include <random>

namespace simplemc {

/**
 * @brief Random sample from a uniform distribution defined on [min, max).
 *
 * @param r Uniform random number from [0, 1).
 * @param min Lower limit.
 * @param max Upper limit.
 * @return Uniform random sample from [min, max).
 */
inline double uniform_sample(double r, double min, double max) {
    assert(max > min);
    return min + r * (max - min);
}

/**
 * @brief Value of the uniform distribution defined on [min, max).
 *
 * @param min Lower limit.
 * @param max Upper limit.
 * @return Value of the distribution.
 */
inline double uniform_pdf(double min, double max) {
    assert(max > min);
    return 1.0 / (max - min);
}

/**
 * @brief Random sample from an exponential distribution defined on [min, inf).
 *
 * @param r Uniform random number from [0, 1).
 * @param lambda Parameter of the distribution.
 * @param min Lower limit.
 * @return Exponential random sample from [min, inf).
 */
inline double exponential_sample(double r, double lambda, double min = 0) {
    assert(lambda > 0);
    return min - std::log(r) / lambda;
}

/**
 * @brief Value of the exponential distribution defined on [min, inf).
 *
 * @param x Input variable.
 * @param lambda Parameter of the distribution.
 * @param min Lower limit.
 * @return Value of the distribution.
 */
inline double exponential_pdf(double x, double lambda, double min = 0) {
    assert(lambda > 0);
    return lambda * std::exp(-lambda * (x - min));
}

/**
 * @brief Random sample from an exponential distribution defined on [min, max).
 *
 * @param r Uniform random number from [0, 1).
 * @param lambda Parameter of the distribution.
 * @param min Lower limit.
 * @param max Upper limit.
 * @return Exponential random sample from [min, max).
 */
inline double exponential_sample(double r, double lambda, double min, double max) {
    assert(lambda != 0.0);
    assert(max > min);
    return min - std::log(1.0 - r * (1.0 - std::exp(-lambda * (max - min)))) / lambda;
}

/**
 * @brief Value of the exponential distribution defined on [min, max).
 *
 * @param x Input variable.
 * @param lambda Parameter of the distribution.
 * @param min Lower limit.
 * @param max Upper limit.
 * @return Value of the distribution.
 */
inline double exponential_pdf(double x, double lambda, double min, double max) {
    assert(lambda != 0.0);
    assert(max > min);
    return lambda / (1.0 - std::exp(-lambda * (max - min))) * std::exp(-lambda * (x - min));
}

/**
 * @brief Random sample from an exponential distribution defined on [min, max) with some
 * safety measures depending on the given lambda parameter.
 *
 * @param r Uniform random number from [0, 1).
 * @param lambda Parameter of the distribution.
 * @param min Lower limit.
 * @param max Upper limit.
 * @return Exponential random sample from [min, max).
 */
inline double safe_exponential_sample(double r, double lambda, double min, double max) {
    if (lambda > 0) {
        return exponential_sample(r, lambda, min, max);
    } else if (lambda < 0) {
        return max - exponential_sample(r, -lambda, 0.0, max - min);
    } else {
        return uniform_sample(r, min, max);
    }
}

/**
 * @brief Value of the exponential distribution defined on [min, max) with some safety
 * measures depending on the given lambda parameter.
 *
 * @param x Input variable.
 * @param lambda Parameter of the distribution.
 * @param min Lower limit.
 * @param max Upper limit.
 * @return Value of the distribution.
 */
inline double safe_exponential_pdf(double x, double lambda, double min, double max) {
    if (lambda > 0) {
        return exponential_pdf(x, lambda, min, max);
    } else if (lambda < 0) {
        return exponential_pdf(max - x, -lambda, 0.0, max - min);
    } else {
        return uniform_pdf(min, max);
    }
}

/**
 * @brief Random sample from a normal distribution.
 *
 * @tparam RNG Random number generator.
 * @param rng RNG object.
 * @param mu Mean parameter of the distribution.
 * @param sigma Stddev parameter of the distribution.
 * @return Normal random sample.
 */
template <typename RNG>
inline double normal_sample(RNG& rng, double mu, double sigma) {
    assert(sigma > 0);
    return std::normal_distribution(mu, sigma)(rng);
}

/**
 * @brief Value of the normal distribution.
 *
 * @param x Input variable.
 * @param mu Mean parameter of the distribution.
 * @param sigma Stddev parameter of the distribution.
 * @return Value of the distribution.
 */
inline double normal_pdf(double x, double mu, double sigma) {
    assert(sigma > 0);
    using namespace std::numbers;
    auto tmp = (x - mu) / sigma;
    return std::exp(-0.5 * tmp * tmp) / sigma * inv_sqrtpi / sqrt2;
}

/**
 * @brief Random sample from a Cauchy distribution.
 *
 * @param r Uniform random number from [0, 1).
 * @param x0 Location of the peak.
 * @param gamma Scale parameter.
 * @return Cauchy random sample.
 */
inline double cauchy_sample(double r, double x0, double gamma) {
    assert(gamma > 0);
    using std::numbers::pi;
    return x0 + gamma * std::tan(pi * (r - 0.5));
}

/**
 * @brief Value of the Cauchy distribution.
 *
 * @param x Input variable.
 * @param x0 Location of the peak.
 * @param gamma Scale parameter.
 * @return Value of the distribution.
 */
inline double cauchy_pdf(double x, double x0, double gamma) {
    assert(gamma > 0);
    using std::numbers::pi;
    return 1.0 / (pi * gamma * (1.0 + std::pow((x - x0) / gamma, 2)));
}

/**
 * @brief Random sample from a uniform integer distriubtion defined on [min, max].
 *
 * @tparam T Integer type.
 * @tparam RNG Random number generator.
 * @param rng RNG object.
 * @param min Lower limit.
 * @param max Upper limit.
 * @return Uniform random integer sample from [min, max].
 */
template <typename T, typename RNG>
inline T uniform_int_sample(RNG& rng, T min, T max) {
    assert(max >= min);
    return std::uniform_int_distribution<T>(min, max)(rng);
}

/**
 * @brief Value of the uniform integer distribution.
 *
 * @tparam T Integer type.
 * @param min Lower limit.
 * @param max Upper limit.
 * @return Value of the distribution.
 */
template <typename T>
inline double uniform_int_pdf(T min, T max) {
    assert(max >= min);
    return 1.0 / (max - min + 1);
}

/**
 * @brief Random sample from a uniform integer distriubtion defined on [min, max]
 * excluding a given value on the interval.
 *
 * @tparam T Integer type.
 * @tparam RNG Random number generator.
 * @param rng RNG object.
 * @param min Lower limit.
 * @param max Upper limit.
 * @param exclude Value to be excluded.
 * @return Uniform random integer sample from [min, max] excluding one value.
 */
template <typename T, typename RNG>
inline T exclusive_uniform_int_sample(RNG& rng, T min, T max, T exclude) {
    assert(max > min && exclude >= min && exclude <= max);
    std::uniform_int_distribution<T> dist(1, max - min);
    return ((exclude - min) + dist(rng)) % (max - min + 1) + min;
}

/**
 * @brief Value of the uniform integer distribution excluding a given value.
 *
 * @tparam T Integer type.
 * @param x Input variable.
 * @param min Lower limit.
 * @param max Upper limit.
 * @return Value of the distribution.
 */
template <typename T>
inline double exclusive_uniform_int_pdf(T min, T max) {
    assert(max > min);
    return 1.0 / (max - min);
}

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_SAMPLES_HPP
