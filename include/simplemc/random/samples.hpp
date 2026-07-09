/**
 * @file
 * @brief Probability density functions and random sampling utilities.
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
 * @param a Lower limit \f$ a \f$ of the interval.
 * @param b Upper limit \f$ b \f$ of the interval.
 * @return Uniform random sample from \f$ [a, b) \f$.
 */
[[nodiscard]] constexpr double uniform_sample(double r, double a, double b) {
    assert(b > a);
    return a + r * (b - a);
}

/**
 * @brief Probability density function of the uniform distribution defined on the interval
 * \f$ [a, b) \f$.
 *
 * @details The PDF of the uniform distribution is defined as the constant function
 * \f[
 *   \mathcal{U}(x; a, b) = \frac{1}{b - a} \; .
 * \f]
 *
 * @note It is assumed that \f$ b > a \f$.
 *
 * @param a Lower limit \f$ a \f$ of the interval.
 * @param b Upper limit \f$ b \f$ of the interval.
 * @return Value of the PDF.
 */
[[nodiscard]] constexpr double uniform_pdf(double a, double b) {
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
 * @param lambda Parameter \f$ \lambda \f$ of the exponential distribution.
 * @param a Lower limit \f$ a \f$ of the interval.
 * @return Exponential random sample from \f$ [a, \infty) \f$.
 */
[[nodiscard]] constexpr double exponential_sample(double r, double lambda, double a = 0) {
    assert(lambda > 0);
    return a - std::log(r) / lambda;
}

/**
 * @brief Probability density function of the exponential distribution defined on the interval
 * \f$ [a, \infty) \f$.
 *
 * @details The PDF of the exponential distribution is defined as
 * \f[
 *   f(x; \lambda) = \lambda e^{-\lambda (x - a)} \; .
 * \f]
 *
 * @note It is assumed that \f$ \lambda > 0 \f$.
 *
 * @param x Input variable \f$ x \f$.
 * @param lambda Parameter \f$ \lambda \f$ of the exponential distribution.
 * @param a Lower limit \f$ a \f$ of the interval.
 * @return Value of the PDF at the given input variable \f$ x \f$.
 */
[[nodiscard]] constexpr double exponential_pdf(double x, double lambda, double a = 0) {
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
 * @param lambda Parameter \f$ \lambda \f$ of the exponential distribution.
 * @param a Lower limit \f$ a \f$ of the interval.
 * @param b Upper limit \f$ b \f$ of the interval.
 * @return Exponential random sample from \f$ [a, b) \f$.
 */
[[nodiscard]] constexpr double exponential_sample(double r, double lambda, double a, double b) {
    assert(lambda != 0.0);
    assert(b > a);
    return a - std::log(1.0 - r * (1.0 - std::exp(-lambda * (b - a)))) / lambda;
}

/**
 * @brief Probability density function of the exponential distribution defined on the interval
 * \f$ [a, b) \f$.
 *
 * @details The PDF of the exponential distribution is defined as
 * \f[
 *   f(x; \lambda, a, b) = \frac{\lambda}{1 - e^{-\lambda (b - a)}} e^{-\lambda (x - a)} \; .
 * \f]
 *
 * @note It is assumed that \f$ \lambda \neq 0 \f$, \f$ b > a \f$ and \f$ a \leq x \leq b \f$.
 *
 * @param x Input variable \f$ x \f$.
 * @param lambda Parameter \f$ \lambda \f$ of the exponential distribution.
 * @param a Lower limit \f$ a \f$ of the interval.
 * @param b Upper limit \f$ b \f$ of the interval.
 * @return Value of the PDF at the given input variable \f$ x \f$.
 */
[[nodiscard]] constexpr double exponential_pdf(double x, double lambda, double a, double b) {
    assert(lambda != 0.0);
    assert(b > a);
    return lambda / (1.0 - std::exp(-lambda * (b - a))) * std::exp(-lambda * (x - a));
}

/**
 * @brief Random sample from a safe exponential distribution defined on the interval \f$ [a, b) \f$.
 *
 * @details Depending on the value of \f$ \lambda \f$, the random sample is either generated from a
 * uniform (simplemc::uniform_sample) or an exponential (simplemc::exponential_sample(double, double,
 * double, double)) distribution.
 *
 * See simplemc::safe_exponential_pdf.
 *
 * @note It is assumed that \f$ b > a \f$. There is no restriction on the value of \f$ \lambda \f$.
 *
 * @param r Uniform random number from \f$ [0, 1) \f$.
 * @param lambda Parameter \f$ \lambda \f$ of the exponential distribution.
 * @param a Lower limit \f$ a \f$ of the interval.
 * @param b Upper limit \f$ b \f$ of the interval.
 * @return Exponential random sample from \f$ [a, b) \f$.
 */
[[nodiscard]] constexpr double safe_exponential_sample(double r, double lambda, double a, double b) {
    if (lambda > 0) {
        return exponential_sample(r, lambda, a, b);
    } else if (lambda < 0) {
        return b - exponential_sample(r, -lambda, 0.0, b - a);
    } else {
        return uniform_sample(r, a, b);
    }
}

/**
 * @brief Probability density function of the safe exponential distribution defined on the interval
 * \f$ [a, b) \f$.
 *
 * @details Depending on the value of \f$ \lambda \f$, we define the PDF of the safe exponential
 * distribution as
 * \f[
 *   f(x; \lambda) =
 *   \begin{cases}
 *   \frac{\lambda}{1 - e^{-\lambda (b - a)}} e^{-\lambda (x - a)} & \text{if } \lambda > 0 \\
 *   \frac{1}{b - a} & \text{if } \lambda = 0 \\
 *   \frac{-\lambda}{1 - e^{\lambda (b - a)}} e^{\lambda (b - x)} & \text{if } \lambda < 0
 *   \end{cases}
 *   \; .
 * \f]
 *
 * @note It is assumed that \f$ b > a \f$ and \f$ a \leq x \leq b \f$. There is no restriction on the
 * value of \f$ \lambda \f$.
 *
 * @param x Input variable \f$ x \f$.
 * @param lambda Parameter \f$ \lambda \f$ of the exponential distribution.
 * @param a Lower limit \f$ a \f$ of the interval.
 * @param b Upper limit \f$ b \f$ of the interval.
 * @return Value of the PDF at the given input variable \f$ x \f$.
 */
[[nodiscard]] constexpr double safe_exponential_pdf(double x, double lambda, double a, double b) {
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
 * @details It uses `std::normal_distribution` internally. See simplemc::normal_pdf.
 *
 * @note It is assumed that \f$ \sigma > 0 \f$.
 *
 * @tparam RNG Random number generator type.
 * @param rng RNG object.
 * @param mu Mean parameter \f$ \mu \f$ of the distribution.
 * @param sigma Standard deviation parameter \f$ \sigma \f$ of the distribution.
 * @return Normal random sample.
 */
template <typename RNG>
[[nodiscard]] double normal_sample(RNG& rng, double mu, double sigma) {
    assert(sigma > 0);
    return std::normal_distribution { mu, sigma }(rng);
}

/**
 * @brief Probability density function of the normal distribution.
 *
 * @details The PDF of the normal distribution is defined as
 * \f[
 *   \mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}}
 *   e^{-\frac{(x - \mu)^2}{2 \sigma^2}} \; .
 * \f]
 *
 * @note It is assumed that \f$ \sigma > 0 \f$.
 *
 * @param x Input variable \f$ x \f$.
 * @param mu Mean parameter \f$ \mu \f$ of the distribution.
 * @param sigma Standard deviation parameter \f$ \sigma \f$ of the distribution.
 * @return Value of the PDF at the given input variable \f$ x \f$.
 */
[[nodiscard]] constexpr double normal_pdf(double x, double mu, double sigma) {
    assert(sigma > 0);
    using namespace std::numbers;
    const double tmp = (x - mu) / sigma;
    return std::exp(-0.5 * tmp * tmp) / sigma * inv_sqrtpi / sqrt2;
}

/**
 * @brief Random sample from a Cauchy distribution.
 *
 * @details Using the inversion method on the simplemc::cauchy_pdf gives
 * \f[
 *   x = x_0 + \gamma \tan(\pi (r - 0.5)) \; ,
 * \f]
 * where \f$ x \f$ is the generated random sample.
 *
 * @note It is assumed that \f$ \gamma > 0 \f$.
 *
 * @param r Uniform random number from \f$ [0, 1) \f$.
 * @param x0 Location \f$ x_0 \f$ of the peak.
 * @param gamma Scale parameter \f$ \gamma \f$.
 * @return Cauchy random sample.
 */
[[nodiscard]] constexpr double cauchy_sample(double r, double x0, double gamma) {
    assert(gamma > 0);
    using std::numbers::pi;
    return x0 + gamma * std::tan(pi * (r - 0.5));
}

/**
 * @brief Probability density function of the Cauchy distribution.
 *
 * @details The PDF of the Cauchy distribution is defined as
 * \f[
 *   f(x; x_0, \gamma) = \frac{1}{\pi \gamma \left[ 1 + \left( \frac{x - x_0}{\gamma} \right)^2
 *   \right]} \; .
 * \f]
 *
 * @note It is assumed that \f$ \gamma > 0 \f$.
 *
 * @param x Input variable \f$ x \f$.
 * @param x0 Location \f$ x_0 \f$ of the peak.
 * @param gamma Scale parameter \f$ \gamma \f$.
 * @return Value of the PDF at the given input variable \f$ x \f$.
 */
[[nodiscard]] constexpr double cauchy_pdf(double x, double x0, double gamma) {
    assert(gamma > 0);
    using std::numbers::pi;
    return 1.0 / (pi * gamma * (1.0 + std::pow((x - x0) / gamma, 2)));
}

/**
 * @brief Random sample from a uniform integer distribution defined on the set \f$ \{ a, a+1, \dots,
 * b \} \f$.
 *
 * @details It uses `std::uniform_int_distribution` internally. See simplemc::uniform_int_pdf.
 *
 * @note It is assumed that \f$ b \geq a \f$.
 *
 * @tparam T Integer type.
 * @tparam RNG Random number generator type.
 * @param rng RNG object.
 * @param a Smallest integer \f$ a \f$ of the domain.
 * @param b Largest integer \f$ b \f$ of the domain.
 * @return Uniform random integer sample from the set \f$ \{ a, a+1, \dots, b \} \f$.
 */
template <typename T, typename RNG>
[[nodiscard]] T uniform_int_sample(RNG& rng, T a, T b) {
    assert(b >= a);
    return std::uniform_int_distribution<T> { a, b }(rng);
}

/**
 * @brief Discrete probabilities of the uniform integer distribution defined on the set \f$ \{ a, a+1,
 * \dots, b \} \f$.
 *
 * @details The discrete probabilities of the uniform integer distribution are defined as
 * \f[
 *   P(i; a, b) = \frac{1}{b - a + 1} \; .
 * \f]
 *
 * @note It is assumed that \f$ b \geq a \f$.
 *
 * @tparam T Integer type.
 * @param a Smallest integer \f$ a \f$ of the domain.
 * @param b Largest integer \f$ b \f$ of the domain.
 * @return Probability for choosing any integer in the set \f$ \{ a, a+1, \dots, b \} \f$.
 */
template <typename T>
[[nodiscard]] constexpr double uniform_int_pdf(T a, T b) {
    assert(b >= a);
    return 1.0 / (b - a + 1);
}

/**
 * @brief Random sample from an exclusive uniform integer distribution defined on the set \f$ D = \{
 * i : i \in \{ a, a+1, \dots, b \} \wedge i \neq y \} \f$.
 *
 * @details It is similar to simplemc::uniform_int_sample but excludes a given integer \f$ y \f$. See
 * simplemc::exclusive_uniform_int_pdf.
 *
 * It draws a uniform offset \f$ o \in \{ 1, \dots, b - a \} \f$ and returns \f$ a + ((y - a) + o) \mod
 * (b - a + 1) \f$, i.e. it steps forward from \f$ y \f$ by a non-zero offset and wraps around within
 * \f$ \{ a, a + 1, \dots, b \} \f$, which always lands on an integer of \f$ D \f$ with uniform
 * probability.
 *
 * @note It is assumed that \f$ b > a \f$ and \f$ y \in \{ a, a+1, \dots, b \} \f$.
 *
 * @tparam T Integer type.
 * @tparam RNG Random number generator type.
 * @param rng RNG object.
 * @param a Smallest integer \f$ a \f$ of the domain.
 * @param b Largest integer \f$ b \f$ of the domain.
 * @param y Integer to be excluded.
 * @return Uniform random integer sample from the set \f$ D \f$.
 */
template <typename T, typename RNG>
[[nodiscard]] T exclusive_uniform_int_sample(RNG& rng, T a, T b, T y) {
    assert(b > a && y >= a && y <= b);
    std::uniform_int_distribution<T> dist { 1, b - a };
    return ((y - a) + dist(rng)) % (b - a + 1) + a;
}

/**
 * @brief Discrete probabilities of the exclusive uniform integer distribution defined on the set \f$
 * D = \{ i : i \in \{ a, a+1, \dots, b \} \wedge i \neq y \} \f$.
 *
 * @details The discrete probabilities of the exclusive uniform integer distribution are defined as
 * \f[
 *   P(i; a, b) = \frac{1}{b - a} \; .
 * \f]
 *
 * @note It is assumed that \f$ b > a \f$ and \f$ y \in \{ a, a+1, \dots, b \} \f$.
 *
 * @tparam T Integer type.
 * @param a Smallest integer \f$ a \f$ of the domain.
 * @param b Largest integer \f$ b \f$ of the domain.
 * @return Probability for choosing any integer in the set \f$ D \f$.
 */
template <typename T>
[[nodiscard]] constexpr double exclusive_uniform_int_pdf(T a, T b) {
    assert(b > a);
    return 1.0 / (b - a);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_SAMPLES_HPP
