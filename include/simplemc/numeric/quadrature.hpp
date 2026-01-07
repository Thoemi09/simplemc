/**
 * @file
 * @brief Basic numeric quadrature routines in 1D.
 */

#ifndef SIMPLEMC_NUMERIC_QUADRATURE_HPP
#define SIMPLEMC_NUMERIC_QUADRATURE_HPP

#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-quadrature
 * @{
 */

/**
 * @brief Class implementing the extended trapezoidal quadrature rule.
 *
 * @details It stores
 * - \f$ f \f$ - the integrand,
 * - \f$ a \f$ and \f$ b \f$ - the upper and lower limits of integration,
 * - \f$ N \f$ - the current level of approximation,
 * - \f$ I_N \f$ - the current approximate value of the integral and
 * - \f$ I_{N-1} \f$ - the last approximate value of the integral.
 *
 * The extended trapezoidal quadrature rule on a uniform grid with \f$ M + 1 \f$ grid points can be
 * written as
 * \f[
 *   \int_a^b dx f(x) \approx \frac{\Delta x}{2} \sum_{i=1}^M \left[ f(x_{i-1}) + f(x_i) \right] =
 *   \Delta x \left( \frac{f(b) + f(a)}{2} + \sum_{i=1}^{M-1} f(x_i) \right)\; ,
 * \f]
 * where \f$ \Delta x = (b - a) / M \f$, \f$ x_0 = a \f$ and \f$ x_{M+1} = b \f$.
 *
 * By increasing \f$ M \f$, we can get better approximations to the exact integral. Here, this is done
 * iteratively, i.e. at each level we add \f$ L = 2^{N-2} \f$ new grid points, where \f$ N \f$ denotes
 * the current level of approximation.
 *
 * The first level, \f$ N = 1 \f$, starts with 2 grid points. The integral is approximated by
 * \f[
 *   \int_a^b dx f(x) \approx (b - a) \frac{f(b) + f(a)}{2} = I_1 \; .
 * \f]
 *
 * All higher levels can then be calculated recursively by repeatedly calling the next() method:
 * \f[
 *   I_N = \frac{I_{N-1}}{2} + \frac{b - a}{2L} \sum_{i=0}^{L-1} f(x_{2i+1}) \; .
 * \f]
 *
 * This class is used in simplemc::trapez_quadrature and simplemc::simpson_quadrature.
 *
 * @note See p. 163 in Press, W. H., et al. "Numerical Recipes in C++: The Art of Scientific
 * Computing", 3rd ed., Cambridge University Press, 2007.
 *
 * @tparam F Callable type representing the integrand.
 */
template <typename F>
class basic_quadrature {
public:
    /**
     * @brief Construct a basic quadrature object with the function to integrate \f$ f \f$ and the
     * integration limits \f$ [a, b] \f$.
     *
     * @details It sets \f$ N = 1 \f$ and calculates the first approximation to the integral, \f$ I_1
     * = (b - a) \frac{f(b) + f(a)}{2} \f$. Note that \f$ I_0 = 0.0 \f$.
     *
     * @param f Callable object representing the integrand \f$ f \f$.
     * @param a Lower integration limit \f$ a \f$.
     * @param b Upper integration limit \f$ b \f$.
     */
    constexpr basic_quadrature(F f, double a, double b) :
        f_(std::move(f)),
        a_(a),
        b_(b),
        curr_(0.5 * (b_ - a_) * (f_(a_) + f_(b_))) {}

    /**
     * @brief Get the current approximate value of the integral.
     *
     * @return Current approximation \f$ I_N \f$.
     */
    [[nodiscard]] constexpr double current() const noexcept { return curr_; }

    /**
     * @brief Get the last approximate value of the integral.
     *
     * @return Last approximation \f$ I_{N-1} \f$.
     */
    [[nodiscard]] constexpr double last() const noexcept { return last_; }

    /**
     * @brief Get the current approximation level.
     *
     * @return Current level \f$ N \f$.
     */
    [[nodiscard]] constexpr int level() const noexcept { return n_; }

    /**
     * @brief Get the integration limits.
     *
     * @return `std::array<double, 2>` containing the integration limits \f$ [a, b] \f$.
     */
    [[nodiscard]] constexpr std::array<double, 2> integration_limits() const noexcept { return std::array { a_, b_ }; }

    /**
     * @brief Get the integrand \f$ f \f$.
     *
     * @return Const reference to the function object to be integrated.
     */
    [[nodiscard]] constexpr const F& integrand() const noexcept { return f_; }

    /**
     * @brief Basic convergence test.
     *
     * @details Checks if the relative difference between the current and last approximation is
     * smaller than the given threshold \f$ \epsilon \f$.
     *
     * @param eps Convergence threshold \f$ \epsilon \f$.
     * @return True, if \f$ N > 1 \f$ and \f$ \left| I_N - I_{N-1} \right| \leq \epsilon |I_N| \f$,
     * otherwise return false.
     */
    [[nodiscard]] constexpr bool is_converged(double eps = 1e-7) const noexcept {
        if (n_ == 1) {
            return false;
        }
        return std::abs(curr_ - last_) <= eps * std::abs(curr_);
    }

    /**
     * @brief Calculate the next approximation level \f$ I_{N+1} \f$.
     *
     * @details Let \f$ N \f$ be the current level. It adds \f$ L = 2^{N-1} \f$ additional, linearly
     * spaced grid points, \f$ x_1, \dots, x_{2(L-1) + 1} \f$, and calculates the new approximation
     * \f[
     *   I_{N+1} = \frac{I_N}{2} + \frac{b - a}{2L} \sum_{i=0}^{L-1} f(x_{2i+1}) \; .
     * \f]
     *
     * @return Approximate value of the integral \f$ I_{N+1} \f$.
     */
    constexpr double next() {
        ++n_;
        const auto L = static_cast<std::uint64_t>(1) << (n_ - 2);
        const double step = (b_ - a_) / static_cast<double>(L);
        double x = a_ + 0.5 * step;
        double sum = 0.0;
        for (std::uint64_t i = 0; i < L; ++i) {
            sum += f_(x);
            x += step;
        }
        last_ = curr_;
        curr_ = 0.5 * (last_ + (b_ - a_) * sum / static_cast<double>(L));
        return curr_;
    }

private:
    F f_;
    double a_;
    double b_;
    double curr_;
    double last_ { 0.0 };
    int n_ { 1 };
};

/**
 * @brief Numerical quadrature with the extended trapezoidal rule.
 *
 * @details It uses simplemc::basic_quadrature to integrate a function iteratively until convergence
 * w.r.t. some threshold \f$ \epsilon \f$ is achieved or the maximum number of iterations is reached.
 *
 * The iteration stops when \f$ \left| I_N - I_{N-1} \right| \leq \epsilon |I_N| \f$, where \f$ I_N
 * \f$ is the approximate value of the integral in iteration \f$ N \f$ and \f$ \epsilon > 0 \f$ is the
 * convergence threshold.
 *
 * @note See p. 164 in Press, W. H., et al. "Numerical Recipes in C++: The Art of Scientific
 * Computing", 3rd ed., Cambridge University Press, 2007.
 *
 * @tparam F Callable type representing the integrand.
 * @param f Callable object representing the integrand \f$ f \f$.
 * @param a Lower integration limit \f$ a \f$.
 * @param b Upper integration limit \f$ b \f$.
 * @param eps Convergence threshold \f$ \epsilon \f$.
 * @param max Maximum number of iterations.
 * @param min Minimum number of iterations.
 * @return A `std::tuple` containing the approximate value of the integral \f$ I_N \f$, the absolute
 * difference \f$ |I_N - I_{N-1}| \f$ and the number of iterations performed \f$ N - 1 \f$.
 */
template <typename F>
[[nodiscard]] constexpr auto trapez_quadrature(
    F&& f, double a, double b, double eps = 1e-7, int max = 20, int min = 5) {
    basic_quadrature quad(std::forward<F>(f), a, b);

    // do minimum number of iterations
    for (int i = 0; i < min; ++i) {
        quad.next();
    }

    // iterate until convergence or max number of iterations is reached
    for (int i = quad.level() - 1; i < max && !quad.is_converged(eps); ++i) {
        quad.next();
    }
    return std::make_tuple(quad.current(), std::abs(quad.current() - quad.last()), quad.level() - 1);
}

/**
 * @brief Numerical quadrature with the extended Simpson rule.
 *
 * @details It uses simplemc::basic_quadrature to integrate a function iteratively until convergence
 * w.r.t. some threshold \f$ \epsilon \f$ is achieved or the maximum number of iterations is reached.
 *
 * The iteration stops when \f$ \left| I_N - I_{N-1} \right| \leq \epsilon |I_N| \f$, where \f$ I_N
 * \f$ is the approximate value of the integral in iteration \f$ N \f$ and \f$ \epsilon > 0 \f$ is the
 * convergence threshold.
 *
 * The extended Simpson rule can be written in terms of the extended trapezoidal rule. Let \f$
 * I_N^{(t)} \f$ be the approximate value of the integral using the extended trapezoidal rule (see
 * simplemc::basic_quadrature). Then the extended Simpson rule approximation at iteration \f$ N \f$ is
 * given by
 * \f[
 *   I_N = \frac{4 I_N^{(t)} - I_{N-1}^{(t)}}{3}
 * \f]
 *
 * @note See p. 164 in Press, W. H., et al. "Numerical Recipes in C++: The Art of Scientific
 * Computing", 3rd ed., Cambridge University Press, 2007.
 *
 * @tparam F Callable type representing the integrand.
 * @param f Callable object representing the integrand \f$ f \f$.
 * @param a Lower integration limit \f$ a \f$.
 * @param b Upper integration limit \f$ b \f$.
 * @param eps Convergence threshold \f$ \epsilon \f$.
 * @param max Maximum number of iterations.
 * @param min Minimum number of iterations.
 * @return A `std::tuple` containing the approximate value of the integral \f$ I_N \f$, the absolute
 * difference \f$ |I_N - I_{N-1}| \f$ and the number of iterations performed \f$ N - 1 \f$.
 */
template <typename F>
[[nodiscard]] constexpr auto simpson_quadrature(
    F&& f, double a, double b, double eps = 1e-7, int max = 20, int min = 5) {
    basic_quadrature quad(std::forward<F>(f), a, b);

    // lambda to check for convergence
    auto is_converged = [&quad](double curr, double prev, double threshold) constexpr noexcept {
        if (quad.level() == 1) {
            return false;
        }
        return std::abs(curr - prev) <= threshold * std::abs(curr);
    };

    // do minimum number of iterations
    double current = 0.0;
    double last = 0.0;
    for (int i = 0; i < min; ++i) {
        quad.next();
        last = current;
        current = (4.0 * quad.current() - quad.last()) / 3.0;
    }

    // iterate until convergence or max number of iterations is reached
    for (int i = quad.level() - 1; i < max && !is_converged(current, last, eps); ++i) {
        quad.next();
        last = current;
        current = (4.0 * quad.current() - quad.last()) / 3.0;
    }
    return std::make_tuple(current, std::abs(current - last), quad.level() - 1);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_QUADRATURE_HPP
