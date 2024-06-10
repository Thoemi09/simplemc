/**
 * @file
 * @brief Often used probability density functions and random samples from them.
 */

#ifndef SIMPLEMC_RANDOM_SAMPLES_HPP
#define SIMPLEMC_RANDOM_SAMPLES_HPP

#include <cassert>
#include <cmath>
#include <numbers>
#include <random>

namespace simplemc {

/**
 * @addtogroup simplemc-random-samples
 * @{
 */

/**
 * @brief Random sample from a uniform distribution defined on the interval \f$ [a, b) \f$.
 *
 * @details Using the inversion method on the simplemc::uniform_pdf gives
 * \f[
 *   x = a + r * (b - a) \; ,
 * \f]
 * where \f$ x \f$ is the generated random sample.
 *
 * @note It is assumed that \f$ b > a \f$.
 *
 * @param r Uniform random number from \f$ [0, 1) \f$.
 * @param a Lower limit of the interval.
 * @param b Upper limit of the interval.
 * @return Uniform random sample from \f$ [a, b) \f$.
 */
inline double uniform_sample(double r, double a, double b) {
    assert(b > a);
    return a + r * (b - a);
}

/**
 * @brief Value of the uniform distribution defined on the interval \f$ [a, b) \f$.
 *
 * @details The uniform distribution is defined as the constant function
 * \f[
 *   \mathcal{U}(x; a, b) = \frac{1}{b - a} \; .
 * \f]
 *
 * @note It is assumed that \f$ b > a \f$.
 *
 * @param a Lower limit of the interval.
 * @param b Upper limit of the interval.
 * @return Value of the distribution.
 */
inline double uniform_pdf(double a, double b) {
    assert(b > a);
    return 1.0 / (b - a);
}

/**
 * @brief Random sample from an exponential distribution defined on the interval \f$ [a, \infty) \f$.
 *
 * @details Using the inversion method on the simplemc::exponential_pdf gives
 * \f[
 *   x = a - \frac{\log(r)}{\lambda} \; ,
 * \f]
 * where \f$ x \f$ is the generated random sample.
 *
 * @note It is assumed that \f$ \lambda > 0 \f$.
 *
 * @param r Uniform random number from \f$ [0, 1) \f$.
 * @param lambda Parameter of the simplemc::exponential_pdf.
 * @param a Lower limit of the interval.
 * @return Exponential random sample from \f$ [a, \infty) \f$.
 */
inline double exponential_sample(double r, double lambda, double a = 0) {
    assert(lambda > 0);
    return a - std::log(r) / lambda;
}

/**
 * @brief Value of the exponential distribution defined on the interval \f$ [a, \infty) \f$.
 *
 * @details The exponential distribution is defined as
 * \f[
 *   f(x; \lambda) = \lambda e^{-\lambda (x - a)} \; .
 * \f]
 *
 * @note It is assumed that \f$ \lambda > 0 \f$.
 *
 * @param x Input variable.
 * @param lambda Parameter of the distribution.
 * @param a Lower limit of the interval.
 * @return Value of the distribution at the given input variable.
 */
inline double exponential_pdf(double x, double lambda, double a = 0) {
    assert(lambda > 0);
    return lambda * std::exp(-lambda * (x - a));
}

/**
 * @brief Random sample from an exponential distribution defined on the interval \f$ [a, b) \f$.
 *
 * @details Using the inversion method on the simplemc::exponential_pdf(double, double, double,
 * double) gives
 * \f[
 *   x = a - \frac{\log \left( 1 - r \left( 1 - e^{-\lambda (b - a)} \right)\right)}{\lambda} \; ,
 * \f]
 * where \f$ x \f$ is the generated random sample.
 *
 * @note It is assumed that \f$ \lambda \neq 0 \f$ and \f$ b > a \f$.
 *
 * @param r Uniform random number from \f$ [0, 1) \f$.
 * @param lambda Parameter of the simplemc::exponential_pdf(double, double, double, double).
 * @param a Lower limit of the interval.
 * @param b Upper limit of the interval.
 * @return Exponential random sample from \f$ [a, b) \f$.
 */
inline double exponential_sample(double r, double lambda, double a, double b) {
    assert(lambda != 0.0);
    assert(b > a);
    return a - std::log(1.0 - r * (1.0 - std::exp(-lambda * (b - a)))) / lambda;
}

/**
 * @brief Value of the exponential distribution defined on the interval \f$ [a, b) \f$.
 *
 * @details The exponential distribution is defined as
 * \f[
 *   f(x; \lambda) = \frac{\lambda}{1  - e^{-\lambda (b - a)}} e^{-\lambda (x - a)} \; .
 * \f]
 *
 * @note It is assumed that \f$ \lambda \neq 0 \f$ and \f$ b > a \f$.
 *
 * @param x Input variable.
 * @param lambda Parameter of the distribution.
 * @param a Lower limit of the interval.
 * @param b Upper limit of the interval.
 * @return Value of the distribution at the given input variable.
 */
inline double exponential_pdf(double x, double lambda, double a, double b) {
    assert(lambda != 0.0);
    assert(b > a);
    return lambda / (1.0 - std::exp(-lambda * (b - a))) * std::exp(-lambda * (x - a));
}

/**
 * @brief Random sample from an exponential distribution defined on the interval \f$ [a, b) \f$ with
 * some safety measures depending on the value of given \f$ \lambda \f$ parameter.
 *
 * @details There is no restriction on the value of \f$ \lambda \f$.
 *
 * @param r Uniform random number from \f$ [0, 1) \f$.
 * @param lambda Parameter of the simplemc::exponential_pdf(double, double, double, double).
 * @param a Lower limit of the interval.
 * @param b Upper limit of the interval.
 * @return Exponential random sample from \f$ [a, b) \f$.
 */
inline double safe_exponential_sample(double r, double lambda, double a, double b) {
    if (lambda > 0) {
        return exponential_sample(r, lambda, a, b);
    } else if (lambda < 0) {
        return b - exponential_sample(r, -lambda, 0.0, b - a);
    } else {
        return uniform_sample(r, a, b);
    }
}

/**
 * @brief Value of the exponential distribution defined on the interval \f$ [a, b) \f$ with some
 * safety measures depending on the value of the given \f$ \lambda \f$  parameter.
 *
 * @details There is no restriction on the value of \f$ \lambda \f$.
 *
 * @param x Input variable.
 * @param lambda Parameter of the simplemc::exponential_pdf(double, double, double, double).
 * @param a Lower limit of the interval.
 * @param b Upper limit of the interval.
 * @return Value of the distribution at the given input variable.
 */
inline double safe_exponential_pdf(double x, double lambda, double a, double b) {
    if (lambda > 0) {
        return exponential_pdf(x, lambda, a, b);
    } else if (lambda < 0) {
        return exponential_pdf(b - x, -lambda, 0.0, b - a);
    } else {
        return uniform_pdf(a, b);
    }
}

/**
 * @brief Random sample from a normal distribution.
 *
 * @details It simply uses `std::normal_distribution`.
 *
 * @note It is assumed that \f$ \sigma > 0 \f$.
 *
 * @tparam RNG Random number generator type.
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
 * @details The normal distribution is defined as
 * \f[
 *   \mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}}
 *   e^{-\frac{(x - \mu)^2}{2 \sigma^2}} \; .
 * \f]
 *
 * @note It is assumed that \f$ \sigma > 0 \f$.
 *
 * @param x Input variable.
 * @param mu Mean parameter of the distribution.
 * @param sigma Stddev parameter of the distribution.
 * @return Value of the distribution at the given input value.
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
 * @details Using the inversion method on the simplemc::cauchy_pdf gives
 * \f[
 *   x = x_0 + \gamma * \tan(\pi * (r - 0.5)) \; ,
 * \f]
 * where \f$ x \f$ is the generated random sample.
 *
 * @note It is assumed that \f$ \gamma > 0 \f$.
 *
 * @param r Uniform random number from \f$ [0, 1) \f$.
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
 * @details The Cauchy distribution is defined as
 * \f[
 *   f(x; x_0, \gamma) = \frac{1}{\pi \gamma \left[ 1 + \left( \frac{x - x_0}{\gamma} \right)^2
 *   \right]} \; .
 * \f]
 *
 * @note It is assumed that \f$ \gamma > 0 \f$.
 *
 * @param x Input variable.
 * @param x0 Location of the peak.
 * @param gamma Scale parameter.
 * @return Value of the distribution at the given input variable.
 */
inline double cauchy_pdf(double x, double x0, double gamma) {
    assert(gamma > 0);
    using std::numbers::pi;
    return 1.0 / (pi * gamma * (1.0 + std::pow((x - x0) / gamma, 2)));
}

/**
 * @brief Random sample from a uniform integer distriubtion defined on the interval \f$ [a, b] \f$.
 *
 * @details It simply uses `std::uniform_int_distribution`.
 *
 * @note It is assumed that \f$ b \geq a \f$.
 *
 * @tparam T Integer type.
 * @tparam RNG Random number generator type.
 * @param rng RNG object.
 * @param a Lower limit of the interval.
 * @param b Upper limit of the interval.
 * @return Uniform random integer sample from \f$ [a, b] \f$.
 */
template <typename T, typename RNG>
inline T uniform_int_sample(RNG& rng, T a, T b) {
    assert(b >= a);
    return std::uniform_int_distribution<T>(a, b)(rng);
}

/**
 * @brief Value of the uniform integer distribution defined on the interval \f$ [a, b] \f$.
 *
 * @details The uniform distribution is defined as the discrete constant function
 * \f[
 *   P(i; a, b) = \frac{1}{b - a + 1} \; .
 * \f]
 *
 * @note It is assumed that \f$ b \geq a \f$.
 *
 * @tparam T Integer type.
 * @param a Lower limit of the interval.
 * @param b Upper limit of the interval.
 * @return Value of the distribution.
 */
template <typename T>
inline double uniform_int_pdf(T a, T b) {
    assert(b >= a);
    return 1.0 / (b - a + 1);
}

/**
 * @brief Random sample from a uniform integer distriubtion defined on the interval \f$ [a, b] \f$
 * excluding a given value on the interval.
 *
 * @details It is similar to simplemc::uniform_int_sample but excludes a given value.
 *
 * @note It is assumed that \f$ b > a \f$ and \f$ y \in [a, b] \f$.
 *
 * @tparam T Integer type.
 * @tparam RNG Random number generator type.
 * @param rng RNG object.
 * @param a Lower limit of the interval.
 * @param b Upper limit of the interval.
 * @param y Value to be excluded.
 * @return Uniform random integer sample from \f$ [a, b] \f$ excluding one value.
 */
template <typename T, typename RNG>
inline T exclusive_uniform_int_sample(RNG& rng, T a, T b, T y) {
    assert(b > a && y >= a && y <= b);
    std::uniform_int_distribution<T> dist(1, b - a);
    return ((y - a) + dist(rng)) % (b - a + 1) + a;
}

/**
 * @brief Value of the uniform integer distribution defined on the interval \f$ [a, b] \f$ excluding
 * a given value on the interval.
 *
 * @details The uniform distribution with one value excluded is defined as the discrete constant
 * function
 * \f[
 *   P(i; a, b) = \frac{1}{b - a} \; .
 * \f]
 *
 * @note It is assumed that \f$ b > a \f$.
 *
 * @tparam T Integer type.
 * @param a Lower limit of the interval.
 * @param b Upper limit of the interval.
 * @return Value of the distribution.
 */
template <typename T>
inline double exclusive_uniform_int_pdf(T a, T b) {
    assert(b > a);
    return 1.0 / (b - a);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_SAMPLES_HPP
