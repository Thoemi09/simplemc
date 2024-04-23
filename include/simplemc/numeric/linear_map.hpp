/**
 * @file
 * @brief Linear map from a range [a, b] to another range [c, d] (and vice versa).
 */

#ifndef SIMPLEMC_NUMERIC_LINEAR_MAP_HPP
#define SIMPLEMC_NUMERIC_LINEAR_MAP_HPP

#include <array>
#include <utility>

namespace simplemc {

/**
 * @ingroup simplemc-numeric
 * @brief Map a range [a, b] to another range [c, d] (and vice versa) via a linear function.
 *
 * @details The linear function is defined by `y(x) = alpha * x + beta` such that
 * `y(a) = c, y(b) = d, y^-1(c) = a, y^-1(d) = b`. The ranges have to be increasing,
 * i.e. a < b and c < d.
 */
class linear_map {
public:
    /**
     * @brief Range type.
     */
    using range_type = std::array<double, 2>;

    /**
     * @brief Default constructor.
     */
    linear_map() = default;

    /**
     * @brief Construct the linear map from [a, b] to [c, d].
     *
     * @param range1 Range #1 [a, b].
     * @param range2 Range #2 [c, d].
     */
    linear_map(const range_type& range1, const range_type& range2);

    /**
     * @brief Set the range and domain of the linear map.
     *
     * @param range1 Range #1 [a, b].
     * @param range2 Range #2 [c, d].
     */
    void set(const range_type& range1, const range_type& range2);

    /**
     * @brief Map a value x from [a, b] to [c, d].
     *
     * @param x Input value from [a, b].
     * @return Mapped value in [c, d].
     */
    [[nodiscard]] double map(double x) const;

    /**
     * @brief Map a value y from [c, d] to [a, b].
     *
     * @param y Input value from [c, d].
     * @return Mapped value in [a, b].
     */
    [[nodiscard]] double inverse_map(double y) const;

    /**
     * @brief Get the range and domain of the linear map.
     *
     * @return std::pair containing the ranges.
     */
    [[nodiscard]] auto ranges() const { return std::make_pair(range1_, range2_); }

    /**
     * @brief Get the linear function parameters.
     *
     * @return std::pair containing the alpha and beta parameter
     */
    [[nodiscard]] auto params() const { return std::make_pair(alpha_, beta_); }

private:
    range_type range1_ { 0.0, 1.0 };
    range_type range2_ { 0.0, 1.0 };
    double alpha_ { 1.0 };
    double beta_ { 0.0 };
};

} // namespace simplemc

#endif // SIMPLEMC_NUMERIC_LINEAR_MAP_HPP
