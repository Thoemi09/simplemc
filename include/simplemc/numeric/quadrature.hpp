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
 * @details It stores the upper and lower limits of integration, the current approximate value of the
 * integral and the current level of approximation.
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
 * the current level.
 *
 * Level 1 starts with 2 grid points. The integral is approximated by
 * \f[
 *   \int_a^b dx f(x) \approx (b - a) \frac{f(b) + f(a)}{2} = I_1 \; .
 * \f]
 *
 * All higher levels can then be calculated recursively by repeatedly calling the next() method:
 * \f[
 *   I_N = \frac{I_{N-1}}{2} + \frac{b - a}{2L} \sum_{i=0}^{L} f(x_{2i+1}) \; .
 * \f]
 *
 * This class is used in simplemc::trapez_quadrature and simplemc::simpson_quadrature.
 *
 * @note See p. 163 in Press, Numerical Recipes.
 */
class basic_quadrature {
public:
    /**
     * @brief Construct a basic quadrature object with the integration limits \f$ [a, b] \f$.
     *
     * @param a Lower integration limit.
     * @param b Upper integration limit.
     */
    basic_quadrature(double a, double b) : a_(a), b_(b) {}

    /**
     * @brief Get the current approximate value of the integral.
     *
     * @return Current approximate value of the integral.
     */
    [[nodiscard]] double current() const { return curr_; }

    /**
     * @brief Get the current approximation level.
     *
     * @return Current level.
     */
    [[nodiscard]] double level() const { return n_; }

    /**
     * @brief Get the integration limits.
     *
     * @return `std::array<double, 2>` containing the integration limits.
     */
    [[nodiscard]] auto integration_limits() const { return std::array { a_, b_ }; }

    /**
     * @brief Calculate the next approximation level by adding additional grid points.
     *
     * @details The function to be integrated is passed as an argument. The user is responsible to
     * make sure that the same function object is used consistently.
     *
     * @tparam F Callable type.
     * @param f Callable object to be integrated.
     * @return Improved approximate value of the integral.
     */
    template <typename F>
    double next(F&& f) {
        n_++;
        if (n_ == 1) {
            curr_ = 0.5 * (b_ - a_) * (std::forward<F>(f)(a_) + std::forward<F>(f)(b_));
        } else {
            int np = 1;
            for (int i = 1; i < n_ - 1; ++i) {
                np <<= 1;
            }
            double step = (b_ - a_) / np;
            double x = a_ + 0.5 * step;
            double sum = 0.0;
            for (int i = 0; i < np; ++i) {
                sum += std::forward<F>(f)(x);
                x += step;
            }
            curr_ = 0.5 * (curr_ + (b_ - a_) * sum / np);
        }
        return curr_;
    }

private:
    double a_;
    double b_;
    double curr_ { 0.0 };
    int n_ { 0 };
};

/**
 * @brief Numerical quadrature with the extended trapezoidal rule.
 *
 * @details It uses simplemc::basic_quadrature to integrate a function iteratively until a certain
 * accuracy is achieved or the maximum number of iterations is reached.
 *
 * The stopping criterium is \f$ \left| I_N - I_{N-1} \right| < \epsilon |I_N| \f$, where \f$ I_N \f$
 * is the approximate value of the integral in iteration \f$ N \f$.
 *
 * The integrand in passed as a callabe object.
 *
 * @note See p. 164 in Press, Numerical Recipes.
 *
 * @tparam F Callable type.
 * @param f Callable object to be integrated.
 * @param a Lower limit of integration.
 * @param b Upper limit of integration.
 * @param eps Required accuracy to stop the iteration.
 * @param max Maximum number of steps.
 * @param min Minimum number of steps.
 * @return A tuple containing the approximate value of the integral, the difference between the
 * current and previous approximation and the number of iterations performed.
 */
template <typename F>
auto trapez_quadrature(F&& f, double a, double b, double eps = 1e-7, int max = 20, int min = 5) {
    double current = 0.0;
    double old = 0.0;
    basic_quadrature quad(a, b);
    int i = 0;
    for (; i < min; ++i) {
        current = quad.next(std::forward<F>(f));
    }
    old = current;
    for (; i < max; ++i) {
        current = quad.next(std::forward<F>(f));
        if (std::abs(current - old) < eps * std::abs(old) || (current == 0.0 && old == 0.0)) {
            return std::make_tuple(current, std::abs(current - old), i);
        }
        old = current;
    }
    return std::make_tuple(current, std::abs(current - old), i);
}

/**
 * @brief Numerical quadrature with the extended Simpson rule.
 *
 * @details It uses simplemc::basic_quadrature to integrate a function iteratively until a certain
 * accuracy is achieved or the maximum number of iterations is reached.
 *
 * The stopping criterium is \f$ \left| I_N - I_{N-1} \right| < \epsilon |I_N| \f$, where \f$ I_N \f$
 * is the approximate value of the integral in iteration \f$ N \f$.
 *
 * The integrand in passed as a callabe object.
 *
 * The extended Simpson rule can be written in terms of the extended trapezodial rule. Let
 * \f$ I_N^{(t)} \f$ be the approximate value of the integral using the extended trapezoidal rule (see
 * simplemc::basic_quadrature). Then the extended Simpson rule approximation at iteration \f$ N \f$ is
 * given by
 * \f[
 *   I_N = \frac{4 I_N^{(t)} - I_{N-1}^{(t)}}{3}
 * \f]
 *
 * @note See p. 164 in Press, Numerical Recipes.
 *
 * @tparam F Callable type.
 * @param f Callable object to be integrated.
 * @param a Lower limit of integration.
 * @param b Upper limit of integration.
 * @param eps Required accuracy to stop the iteration.
 * @param max Maximum number of steps.
 * @param min Minimum number of steps.
 * @return A tuple containing the approximate value of the integral, the difference between the
 * current and previous approximation and the number of iterations performed.
 */
template <typename F>
auto simpson_quadrature(F&& f, double a, double b, double eps = 1e-7, int max = 20, int min = 5) {
    double current = 0.0;
    double old = 0.0;
    double tra_current = 0.0;
    double tra_old = 0.0;
    basic_quadrature quad(a, b);
    int i = 0;
    for (; i < min; ++i) {
        tra_current = quad.next(std::forward<F>(f));
        current = (4.0 * tra_current - tra_old) / 3.0;
        tra_old = tra_current;
    }
    old = current;
    for (; i < max; ++i) {
        tra_current = quad.next(std::forward<F>(f));
        current = (4.0 * tra_current - tra_old) / 3.0;
        if (std::abs(current - old) < eps * std::abs(old) || (current == 0.0 && old == 0.0)) {
            return std::make_tuple(current, std::abs(current - old), i);
        }
        old = current;
        tra_old = tra_current;
    }
    return std::make_tuple(current, std::abs(current - old), i);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_QUADRATURE_HPP
