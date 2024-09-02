/**
 * @file
 * @brief Base class for implementations of orthogonal polynomials.
 */

#ifndef SIMPLEMC_NUMERIC_ORTHOGONAL_POLYNOMIAL_HPP
#define SIMPLEMC_NUMERIC_ORTHOGONAL_POLYNOMIAL_HPP

#include <array>
#include <cassert>

namespace simplemc {

/**
 * @addtogroup simplemc-numeric-functions
 * @{
 */

/**
 * @brief Base class for implementations of orthogonal polynomials.
 *
 * @details This class and derived classes are intented to be used for estimating coefficients in a
 * generalized Fourier series, i.e. \f$ f(x) = \sum_l f_l p_l(x) \f$, with the coefficients \f$ f_l \f$
 * and the l<sup>th</sup> order polynomial \f$ p_l(x) \f$.
 *
 * It is instantiated with the function values of the 0<sup>th</sup> and 1<sup>st</sup> order
 * polynomial evaluated at a specific \f$ x \f$ as well as the corresponding first derivatives. Higher
 * order polynomials are calculated iteratively using a recurrence relation (see next()).
 *
 * The inner product between two polynomials is defined as
 * \f[
 *   \langle p_l, p_k \rangle = \int_a^b p_l(x) p_k(x) W(x) dx \; ,
 * \f]
 * where \f$ [a, b] \f$ is the domain on which the polynomial is defined and \f$ W(x) \f$ is some
 * weight function specific to a certain polynomial.
 *
 * Orthogonality means that the inner product between two polynomials of different order vanishes,
 * i.e. \f$ \langle p_l, p_k \rangle = N_l \delta_{lk} \f$. Here, \f$ N_l \f$ is some order dependent
 * normalization factor.
 */
class orthogonal_polynomial {
public:
    /**
     * @brief Construct an orthogonal polynomial for a specific \f$ x \f$ value.
     *
     * @param x Value at which the polynomials are evaluated.
     * @param y0 Value of the 0<sup>th</sup> order polynomial evaluated at \f$ x \f$.
     * @param y1 Value of the 1<sup>th</sup> order polynomial evaluated at \f$ x \f$.
     * @param dy0 Value of the first derivative of the 0<sup>th</sup> order polynomial evaluated at
     * \f$ x \f$.
     * @param dy1 Value of the first derivative of the 1<sup>th</sup> order polynomial evaluated at
     * \f$ x \f$.
     */
    orthogonal_polynomial(double x, double y0, double y1, double dy0, double dy1) :
        x_ { x },
        y0_ { y0 },
        y1_ { y1 },
        yl_ { y0 },
        dy0_ { dy0 },
        dy1_ { dy1 },
        dyl_ { dy0 } {}

    /**
     * @brief Virtual destructor.
     */
    virtual ~orthogonal_polynomial() = default;

    /**
     * @brief Get the domain on which the polynomial is defined.
     *
     * @return `std::array<double, 2>` representing the domain.
     */
    [[nodiscard]] virtual std::array<double, 2> domain() const { return { -1.0, 1.0 }; }

    /**
     * @brief Get the normalization factor of the l<sup>th</sup> order polynomial.
     *
     * @param l Order of the polynomial.
     * @return Normalization factor.
     */
    [[nodiscard]] virtual double normalization([[maybe_unused]] int l) const { return 1.0; }

    /**
     * @brief Get the normalization factor of the currently processed polynomial.
     *
     * @return Normalization factor.
     */
    [[nodiscard]] double normalization() const { return normalization(l_); }

    /**
     * @brief Get the value of the weight function evaluated at a given \f$ x \f$.
     *
     * @param x Value at which the weight function is evaluated.
     * @return Value of the weight function.
     */
    [[nodiscard]] virtual double weight([[maybe_unused]] double x) const { return 1.0; }

    /**
     * @brief Get the value of the weight function evaluated at the stored \f$ x \f$ value.
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
     * @brief Get the \f$ x \f$ value with which the orthogonal polynomial was instantiated.
     *
     * @return \f$ x \f$ value.
     */
    [[nodiscard]] double x() const { return x_; }

    /**
     * @brief Increase the order from \f$ l \f$ to \f$ l + 1 \f$.
     *
     * @return Value of the l<sup>th</sup> order polynomial evaluated at the stored \f$ x \f$.
     */
    virtual double next() = 0;

    /**
     * @brief Get the value of the currently processed polynomial.
     *
     * @return Value of the polynomial evaluated at the stored \f$ x \f$.
     */
    [[nodiscard]] virtual double current_value() const { return yl_; }

    /**
     * @brief Get the value of the previously processed polynomial.
     *
     * @return Value of the polynomial evaluated at the stored \f$ x \f$.
     */
    [[nodiscard]] virtual double last_value() const {
        assert(l_ > 0);
        return ylm1_;
    }

    /**
     * @brief Get the derivative of the currently processed polynomial.
     *
     * @return First derivative of the polynomial evaluated at the stored \f$ x \f$.
     */
    [[nodiscard]] virtual double current_derivative() const { return dyl_; }

    /**
     * @brief Get the derivative of the previously processed polynomial.
     *
     * @return First derivative of the polynomial evaluated at the stored \f$ x \f$.
     */
    [[nodiscard]] virtual double last_derivative() const {
        assert(l_ > 0);
        return dylm1_;
    }

protected:
    int l_ { 0 };
    double x_;
    double y0_;
    double y1_;
    double yl_;
    double ylm1_ { 0.0 };
    double dy0_;
    double dy1_;
    double dyl_;
    double dylm1_ { 0.0 };
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_ORTHOGONAL_POLYNOMIAL_HPP
