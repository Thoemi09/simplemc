/**
 * @file
 * @brief Implementation of orthogonal polynomials for estimating coefficients in a generalized Fourier series.
 */

#ifndef SIMPLEMC_NUMERIC_ORTHOGONAL_POLYNOMIALS_HPP
#define SIMPLEMC_NUMERIC_ORTHOGONAL_POLYNOMIALS_HPP

#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>

namespace simplemc {

/**
 * @brief Base class for certain orthogonal polynomials.
 *
 * @details For implementation details see:
 * https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.f
 *
 * This class and derived classes are intented to be used for estimating coefficients in a generalized
 * Fourier series: \f$f(x) = sum_l f_l p_l(x)\f$, with the coefficients \f$f_l\f$ and the l-th order polynomial
 * \f$p_l(x)\f$. It is instantiated at a specific x.
 */
class orthogonal_polynomial {
public:
    /**
     * @brief Construct an orthogonal polynomial for a specific x value.
     *
     * @param x Evaluate the polynomial at this value.
     * @param y0 Value of the 0-th order polynomial evaluated at x.
     * @param y1 Value of the 1-st order polynomial evaluated at x.
     * @param dy0 Value of the first derivative of the 0-th order polynomial evaluated at x.
     * @param dy1 Value of the first derivative of the 1-st order polynomial evaluated at x.
     */
    orthogonal_polynomial(double x, double y0, double y1, double dy0, double dy1);

    /**
     * @brief Virtual destructor.
     */
    virtual ~orthogonal_polynomial() = default;

    /**
     * @brief Get lower bound of the domain.
     *
     * @return Lower bound.
     */
    [[nodiscard]] virtual double lower_bound() const { return -1.0; }

    /**
     * @brief Get upper bound of the domain.
     *
     * @return Upper bound.
     */
    [[nodiscard]] virtual double upper_bound() const { return 1.0; }

    /**
     * @brief Get the normalization factor of the l-th order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor.
     */
    [[nodiscard]] virtual double normalization([[maybe_unused]] int l) const { return 1.0; }

    /**
     * @brief Get the normalization factor of the currently processed order.
     *
     * @return Normalization factor.
     */
    [[nodiscard]] double normalization() const { return normalization(l_); }

    /**
     * @brief Get the value of the weight function evaluated at a given x.
     *
     * @param x x value.
     * @return Value of the weight function.
     */
    [[nodiscard]] virtual double weight([[maybe_unused]] double x) const { return 1.0; }

    /**
     * @brief Get the value of the weight function evaluated at the stored x value.
     *
     * @return Value of the weight function.
     * */
    [[nodiscard]] double weight() const { return weight(x_); }

    /**
     * @brief Get the order of the currently processed polynomial.
     *
     * @return Order of the polynomial.
     */
    [[nodiscard]] int order() const { return l_; }

    /**
     * @brief Get the x value with which the orthogonal polynomial was instantiated.
     *
     * @return x value.
     */
    [[nodiscard]] double x() const { return x_; }

    /**
     * @brief Increase the order from l to l+1.
     *
     * @return Value of the l-th order polynomial evaluated at the stored x.
     */
    virtual double next() = 0;

    /**
     * @brief Get the value of the currently processed polynomial.
     *
     * @return Value of the polynomial evaluated at the stored x.
     */
    [[nodiscard]] virtual double current_value() const { return yl_; }

    /**
     * @brief Get the value of the previously processed polynomial.
     *
     * @return Value of the polynomial evaluated at the stored x.
     */
    [[nodiscard]] virtual double last_value() const {
        assert(l_ > 0);
        return ylm1_;
    }

    /**
     * @brief Get the derivative of the currently processed polynomial.
     *
     * @return First derivative of the polynomial evaluated at the stored x.
     */
    [[nodiscard]] virtual double current_derivative() const { return dyl_; }

    /**
     * @brief Get the derivative of the previously processed polynomial evaluated at the stored x.
     *
     * @return First derivative of the polynomial evaluated at the stored x.
     */
    [[nodiscard]] virtual double last_derivative() const {
        assert(l_ > 0);
        return dylm1_;
    }

    /**
     * @brief Get the value of the polynomial of a given order at a given x.
     *
     * @param l Order of the polynomial.
     * @param x x value.
     * @return Value of polynomial.
     */
    virtual double value(int l, double x) = 0;

    /**
     * @brief Get the derivative of the polynomial of a given order at a given x.
     *
     * @param l Order of the polynomial.
     * @param x x value.
     * @return Derivative of the polynomial.
     */
    virtual double derivative(int l, double x);

protected:
    int l_;
    double x_;
    double y0_;
    double y1_;
    double yl_;
    double ylm1_;
    double dy0_;
    double dy1_;
    double dyl_;
    double dylm1_;
};

/**
 * @brief Recursive implementation of Legendre polynomials for a specific x.
 */
class legendre_polynomial : public orthogonal_polynomial {
public:
    /**
     * @brief Construct a Legendre polynomial for a specific x value.
     *
     * @param x Evaluate the polynomial at this value.
     */
    legendre_polynomial(double x = 0.0);

    /**
     * @brief Get the normalization factor of the l-th order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor.
     */
    [[nodiscard]] double normalization(int l) const override {
        assert(l >= 0);
        return 2.0 / (2.0 * l + 1);
    }

    /**
     * @brief Increase the order from l to l+1.
     *
     * @return Value of the l-th order polynomial evaluated at the stored x.
     */
    double next() override;

    /**
     * @brief Get the derivative of the currently processed polynomial.
     *
     * @return First derivative of the polynomial evaluated at the stored x.
     */
    [[nodiscard]] double current_derivative() const override;

    /**
     * @brief Get the derivative of the previously processed polynomial evaluated at the stored x.
     *
     * @return First derivative of the polynomial evaluated at the stored x.
     */
    [[nodiscard]] double last_derivative() const override;

    /**
     * @brief Get the value of the polynomial of a given order at a given x.
     *
     * @param l Order of the polynomial.
     * @param x x value.
     * @return Value of polynomial.
     */
    double value(int l, double x) override;

private:
    double ylm2_;
};

/**
 * @brief Recursive implementation of Chebyshev polynomials of the first kind for a specific x.
 */
class chebyshev_polynomial_first : public orthogonal_polynomial {
public:
    /**
     * @brief Construct a Chebyshev polynomial of the first kind for a specific x.
     *
     * @param x Evaluate the polynomial at this value.
     */
    chebyshev_polynomial_first(double x = 0.0);

    /**
     * @brief Get the normalization factor of the l-th order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor.
     */
    [[nodiscard]] double normalization(int l) const override;

    /**
     * @brief Get the value of the weight function evaluated at a given x.
     *
     * @param x x value.
     * @return Value of the weight function.
     */
    [[nodiscard]] double weight(double x) const override {
        assert(std::abs(x) <= 1.0);
        return 1.0 / std::sqrt(1 - x * x);
    }

    /**
     * @brief Increase the order from l to l+1.
     *
     * @return Value of the l-th order polynomial evaluated at the stored x.
     */
    double next() override;

    /**
     * @brief Get the value of the polynomial of a given order at a given x.
     *
     * @param l Order of the polynomial.
     * @param x x value.
     * @return Value of polynomial.
     */
    double value(int l, double x) override;
};

/**
 * @brief Recursive implementation of Chebyshev polynomials of the second kind for a specific x.
 */
class chebyshev_polynomial_second : public orthogonal_polynomial {
public:
    /**
     * @brief Construct a Chebyshev polynomial of the second kind for a specific x.
     *
     * @param x Evaluate the polynomial at this value.
     */
    chebyshev_polynomial_second(double x = 0.0);

    /**
     * @brief Get the normalization factor of the l-th order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor.
     */
    [[nodiscard]] double normalization([[maybe_unused]] int l) const override { return 2.0 * std::numbers::pi; }

    /**
     * @brief Get the value of the weight function evaluated at a given x.
     *
     * @param x x value.
     * @return Value of the weight function.
     */
    [[nodiscard]] double weight(double x) const override {
        assert(std::abs(x) <= 1.0);
        return std::sqrt(1 - x * x);
    }

    /**
     * @brief Increase the order from l to l+1.
     *
     * @return Value of the l-th order polynomial evaluated at the stored x.
     */
    double next() override;

    /**
     * @brief Get the value of the polynomial of a given order at a given x.
     *
     * @param l Order of the polynomial.
     * @param x x value.
     * @return Value of polynomial.
     */
    double value(int l, double x) override;
};

/**
 * @brief Recursive implementation of Laguerre polynomials for a specific x.
 */
class laguerre_polynomial : public orthogonal_polynomial {
public:
    /**
     * @brief Construct a Laguerre polynomial for a specific x.
     *
     * @param x Evaluate the polynomial at this value.
     */
    laguerre_polynomial(double x = 0.0);

    /**
     * @brief Get lower bound of the domain.
     *
     * @return Lower bound.
     */
    [[nodiscard]] double lower_bound() const override { return 0.0; }

    /**
     * @brief Get upper bound of the domain.
     *
     * @return Upper bound.
     */
    [[nodiscard]] double upper_bound() const override { return std::numeric_limits<double>::infinity(); }

    /**
     * @brief Get the value of the weight function evaluated at a given x.
     *
     * @param x x value.
     * @return Value of the weight function.
     */
    [[nodiscard]] double weight(double x) const override {
        assert(x >= 0.0);
        return std::exp(-x);
    }

    /**
     * @brief Increase the order from l to l+1.
     *
     * @return Value of the l-th order polynomial evaluated at the stored x.
     */
    double next() override;

    /**
     * @brief Get the value of the polynomial of a given order at a given x.
     *
     * @param l Order of the polynomial.
     * @param x x value.
     * @return Value of polynomial.
     */
    double value(int l, double x) override;
};

/**
 * @brief Recursive implementation of Hermite polynomials for a specific x.
 */
class hermite_polynomial : public orthogonal_polynomial {
public:
    /**
     * @brief Construct a Hermite polynomial for a specific x.
     *
     * @param x Evaluate the polynomial at this value.
     */
    hermite_polynomial(double x = 0.0);

    /**
     * @brief Get lower bound of the domain.
     *
     * @return Lower bound.
     */
    [[nodiscard]] double lower_bound() const override { return -std::numeric_limits<double>::infinity(); }

    /**
     * @brief Get upper bound of the domain.
     *
     * @return Upper bound.
     */
    [[nodiscard]] double upper_bound() const override { return std::numeric_limits<double>::infinity(); }

    /**
     * @brief Get the normalization factor of the l-th order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor.
     */
    [[nodiscard]] double normalization([[maybe_unused]] int l) const override {
        assert(l >= 0);
        return std::pow(2, l) * std::tgamma(l + 1) / std::numbers::inv_sqrtpi;
    }

    /**
     * @brief Get the value of the weight function evaluated at a given x.
     *
     * @param x x value.
     * @return Value of the weight function.
     */
    [[nodiscard]] double weight(double x) const override { return std::exp(-x * x); }

    /**
     * @brief Increase the order from l to l+1.
     *
     * @return Value of the l-th order polynomial evaluated at the stored x.
     */
    double next() override;

    /**
     * @brief Get the value of the polynomial of a given order at a given x.
     *
     * @param l Order of the polynomial.
     * @param x x value.
     * @return Value of polynomial.
     */
    double value(int l, double x) override;
};

/**
 * @brief Recursive implementation of the cosine for a specific x.
 */
class cosine : public orthogonal_polynomial {
public:
    /**
     * @brief Construct a cosine for a specific x.
     *
     * @param x Evaluate the polynomial at this value.
     */
    cosine(double x = 0.0);

    /**
     * @brief Get lower bound of the domain.
     *
     * @return Lower bound.
     */
    [[nodiscard]] double lower_bound() const override { return -std::numbers::pi; }

    /**
     * @brief Get upper bound of the domain.
     *
     * @return Upper bound.
     */
    [[nodiscard]] double upper_bound() const override { return std::numbers::pi; }

    /**
     * @brief Get the normalization factor of the l-th order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor.
     */
    [[nodiscard]] double normalization(int l) const override;

    /**
     * @brief Increase the order from l to l+1.
     *
     * @return Value of the l-th order polynomial evaluated at the stored x.
     */
    double next() override;

    /**
     * @brief Get the derivative of the currently processed polynomial.
     *
     * @return First derivative of the polynomial evaluated at the stored x.
     */
    [[nodiscard]] double current_derivative() const override { return -static_cast<double>(l_) * dyl_ * sinx_; }

    /**
     * @brief Get the derivative of the previously processed polynomial evaluated at the stored x.
     *
     * @return First derivative of the polynomial evaluated at the stored x.
     */
    [[nodiscard]] double last_derivative() const override {
        assert(l_ > 0);
        return -static_cast<double>(l_ - 1) * dylm1_ * sinx_;
    }

    /**
     * @brief Get the value of the polynomial of a given order at a given x.
     *
     * @param l Order of the polynomial.
     * @param x x value.
     * @return Value of polynomial.
     */
    double value(int l, double x) override;

private:
    double cosx_;
    double sinx_;
};

/**
 * @brief Recursive implementation of the sine for a specific x.
 */
class sine : public orthogonal_polynomial {
public:
    /**
     * @brief Construct a sine for a specific x.
     *
     * @param x x-value.
     */
    sine(double x = 0.0);

    /**
     * @brief Get lower bound of the domain.
     *
     * @return Lower bound.
     */
    [[nodiscard]] double lower_bound() const override { return -std::numbers::pi; }

    /**
     * @brief Get upper bound of the domain.
     *
     * @return Upper bound.
     */
    [[nodiscard]] double upper_bound() const override { return std::numbers::pi; }

    /**
     * @brief Get the normalization factor of the l-th order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor.
     */
    [[nodiscard]] double normalization(int l) const override;

    /**
     * @brief Increase the order from l to l+1.
     *
     * @return Value of the l-th order polynomial evaluated at the stored x.
     */
    double next() override;

    /**
     * @brief Get the value of the currently processed polynomial.
     *
     * @return Value of the polynomial evaluated at the stored x.
     */
    [[nodiscard]] double current_value() const override { return yl_ * sinx_; }

    /**
     * @brief Get the value of the previously processed polynomial.
     *
     * @return Value of the polynomial evaluated at the stored x.
     */
    [[nodiscard]] double last_value() const override {
        assert(l_ > 0);
        return ylm1_ * sinx_;
    }

    /**
     * @brief Get the derivative of the currently processed polynomial.
     *
     * @return First derivative of the polynomial evaluated at the stored x.
     */
    [[nodiscard]] double current_derivative() const override { return static_cast<double>(l_) * dyl_; }

    /**
     * @brief Get the derivative of the previously processed polynomial evaluated at the stored x.
     *
     * @return First derivative of the polynomial evaluated at the stored x.
     */
    [[nodiscard]] double last_derivative() const override {
        assert(l_ > 0);
        return static_cast<double>(l_ - 1) * dylm1_;
    }

    /**
     * @brief Get the value of the polynomial of a given order at a given x.
     *
     * @param l Order of the polynomial.
     * @param x x value.
     * @return Value of polynomial.
     */
    double value(int l, double x) override;

private:
    double cosx_;
    double sinx_;
};

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_ORTHOGONAL_POLYNOMIALS_HPP
