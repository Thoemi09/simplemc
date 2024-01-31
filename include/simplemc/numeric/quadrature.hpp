/**
 * @file quadrature.hpp
 * @brief Basic numeric quadrature routines.
 */

#ifndef SIMPLEMC_NUMERIC_QUADRATURE_HPP
#define SIMPLEMC_NUMERIC_QUADRATURE_HPP

#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cmath>
#include <tuple>
#include <utility>

namespace simplemc {

/**
 * @brief Simple class used in trapezoidal and Simpson's quadrature.
 *
 * @details It implements the extended trapezoidal quadrature rule, stores the upper and lower
 * limits of integration, the approximate value of the integral and the current level of approximation.
 *
 * See Press, Numerical Recipes, p. 163.
 */
class basic_quadrature {
public:
    /**
     * @brief Construct a basic quadrature object.
     *
     * @param a Lower integration limit.
     * @param b Upper integration limit.
     */
    basic_quadrature(double a, double b);

    /**
     * @brief Get current value of approximation.
     *
     * @return Current approximate value of the integral.
     */
    [[nodiscard]] double current() const { return curr_; }

    /**
     * @brief Get current level of approximation.
     *
     * @return Current level.
     */
    [[nodiscard]] double level() const { return n_; }

    /**
     * @brief Calculate next level of accuracy by doubling the points.
     *
     * @tparam F Function type.
     * @param f Function object to be integrated (can be called with f(x)).
     * @return Current approximate value of the integral.
     */
    template <typename F>
    double next(F&& f);

    /**
     * @brief Get lower limit of integration.
     *
     * @return Lower limit of integration.
     */
    [[nodiscard]] double lower_limit() const { return a_; }

    /**
     * @brief Get upper limit of integration.
     *
     * @return Upper limit of integration.
     */
    [[nodiscard]] double upper_limit() const { return b_; }

private:
    double a_;
    double b_;
    double curr_ { 0.0 };
    int n_ { 0 };
};

basic_quadrature::basic_quadrature(double a, double b) : a_(a), b_(b) {}

template <typename F>
double basic_quadrature::next(F&& f) {
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

/**
 * @brief Uses extended trapezoidal quadrature to integrate a function until a certain
 * accuracy is achieved or the maximum number of iterations is reached.
 *
 * @details See Numerical Recipes, Press, p. 164.
 *
 * @tparam F Function type.
 * @param f Function object to be integrated (can be called with f(x)).
 * @param a Lower limit of integration.
 * @param b Upper limit of integration.
 * @param eps Required accuracy.
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
        if (std::abs(current - old) < eps * std::abs(old) || (current == 0.0 && old == 0.0))
            return std::make_tuple(current, std::abs(current - old), i);
        old = current;
    }
    return std::make_tuple(current, std::abs(current - old), i);
}

/**
 * @brief Uses extended Simpson quadrature to integrate a function until a certain
 * accuracy is achieved or the maximum number of iterations is reached.
 *
 * @details See Numerical Recipes, Press, p. 165.
 *
 * @tparam F Function type.
 * @param f Function object to be integrated (can be called with f(x)).
 * @param a Lower limit of integration.
 * @param b Upper limit of integration.
 * @param eps Required accuracy.
 * @param max Maximum number of steps.
 * @param min Minimum number of steps.
 * @return A tuple containing the approximate value of the integral, the difference between the
 * current and previous approximation and the number of iterations performed.
 */
template <typename F>
auto simpson_quadrature(F&& f, double a, double b, double eps = 1e-7, int max = 30, int min = 5) {
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
        if (std::abs(current - old) < eps * std::abs(old) || (current == 0.0 && old == 0.0))
            return std::make_tuple(current, std::abs(current - old), i);
        old = current;
        tra_old = tra_current;
    }
    return std::make_tuple(current, std::abs(current - old), i);
}

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_QUADRATURE_HPP
