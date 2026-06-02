/**
 * @file
 * @brief 1-dimensional power grid.
 */

#ifndef SIMPLEMC_GRIDS_POWER_GRID_HPP
#define SIMPLEMC_GRIDS_POWER_GRID_HPP

#include <simplemc/grids/grid_base.hpp>
#include <simplemc/grids/grid_iterator.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

namespace simplemc {

/**
 * @addtogroup simplemc-grids-1d
 * @{
 */

/**
 * @brief 1-dimensional power grid.
 *
 * @details In the following, we use the notation from @ref simplemc-grids-1d.
 *
 * This class inherits from simplemc::grid_base and satisfies the simplemc::grid_1d concept. It
 * represents a power grid of size \f$ M \geq 2 \f$ on the interval
 * - \f$ \mathrm{R} = [a, b] \f$ for increasing grids, i.e. \f$ a < b \f$, or
 * - \f$ \mathrm{R} = [b, a] \f$ for decreasing grids, i.e. \f$ a > b \f$.
 *
 * It uses the non-linear function
 * \f[
 *   g(i) = a + i^p * \sigma \;,
 * \f]
 * to map an index \f$ i \in \{ 0, 1, \dots, M-1 \} \f$ to its grid point \f$ g(i) \f$, where \f$ p >
 * 0 \f$ is the power parameter and \f$ \sigma = (b - a) / (M - 1)^p \f$ is the scale factor.
 *
 * The corresponding inverse function
 * \f[
 *   \tilde{g}^{-1}(x) = \left\lfloor \left( \frac{x - a}{\sigma} \right)^{1/p} \right\rfloor \;,
 * \f]
 * maps every value \f$ x \in [a, b] \f$ to the index \f$ i \f$ such that \f$ x \in b_i \f$.
 *
 * @include grids/doc_power_grid.cpp
 *
 * Output:
 *
 * ```
 * Grid points: [0, 0.0625, 0.25, 0.5625, 1]
 * Bin centers: [0.03125, 0.15625, 0.40625, 0.78125]
 * Bin volumes: [0.0625, 0.1875, 0.3125, 0.4375]
 * ```
 */
class power_grid : public grid_base<power_grid> {
public:
    /**
     * @brief Type of the CRTP base class.
     */
    using base_type = grid_base<power_grid>;

    /**
     * @brief Default constructor constructs a power grid of size \f$ M = 2 \f$ on the interval
     * \f$ [0, 1] \f$ with \f$ p = 1 \f$.
     */
    constexpr power_grid() noexcept = default;

    /**
     * @brief Construct a power grid by specifying its first and last grid points, \f$ a \f$ and \f$
     * b \f$, its size \f$ M \f$ and the power parameter \f$ p \f$.
     *
     * @details It simply forwards the arguments to reset().
     *
     * @param a First value of the grid, \f$ a = g(0) \f$.
     * @param b Last value of the grid, \f$ b = g(M-1) \f$.
     * @param m Number of grid points \f$ M \f$.
     * @param p Power parameter \f$ p \f$.
     */
    constexpr power_grid(value_type a, value_type b, size_type m, value_type p) { reset(a, b, m, p); }

    /**
     * @brief Reset the power grid by specifying its first and last grid points, \f$ a \f$ and \f$ b
     * \f$, its size \f$ M \f$ and the power parameter \f$ p \f$.
     *
     * @details It forwards the arguments \f$ a \f$, \f$ b \f$ and \f$ M \f$ to base_type::reset and
     * sets the power parameter \f$ p > 0 \f$ and the scale factor \f$ \sigma = (b - a) / (M - 1)^p
     * \f$.
     *
     * @param a First value of the grid, \f$ a = g(0) \f$.
     * @param b Last value of the grid, \f$ b = g(M-1) \f$.
     * @param m Number of grid points \f$ M \f$.
     * @param p Power parameter \f$ p \f$.
     */
    constexpr void reset(value_type a, value_type b, size_type m, value_type p) {
        if (p <= 0.0) {
            throw simplemc_exception("Power parameter must be > 0", "power_grid::reset");
        }
        base_type::reset(a, b, m);
        power_ = p;
        scale_ = (last_ - first_) / std::pow(size_ - 1, power_);
    }

    /**
     * @brief Get the grid point \f$ g(i) \f$ at a given index \f$ i \f$.
     *
     * @details The mapping is defined as \f$ g(i) = a + i^p * \sigma\f$.
     *
     * @param idx Index \f$ i \f$ of the grid point.
     * @return Grid point \f$ g(i) \f$.
     */
    [[nodiscard]] constexpr value_type at(size_type idx) const {
        assert(idx >= 0 && idx < size_);
        return first_ + scale_ * std::pow(idx, power_);
    }

    /**
     * @brief Get the index \f$ i \f$ such that a given value \f$ x \in [a, b] \f$ lies in bin \f$
     * b_i \f$.
     *
     * @details The inverse mapping is defined as \f$ \tilde{g}^{-1}(x) = \left\lfloor \left[ (x - a)
     * / \sigma \right]^{1/p} \right\rfloor \f$.
     *
     * @param value Some value \f$ x \in [a, b] \f$.
     * @return Index \f$ i = \tilde{g}^{-1}(x) \f$ of the bin \f$ b_i \f$ such that \f$ x \in b_i \f$.
     */
    [[nodiscard]] constexpr size_type index(value_type value) const {
        assert((first_ <= value && value <= last_) || (first_ >= value && value >= last_));
        return static_cast<size_type>(std::pow((value - first_) / scale_, 1.0 / power_));
    }

    /**
     * @brief Get the power parameter of the grid.
     *
     * @return Power parameter \f$ p \f$.
     */
    [[nodiscard]] constexpr value_type power() const noexcept { return power_; }

    /**
     * @brief Get the scale factor of the grid.
     *
     * @return Scale factor \f$ \sigma = (b - a) / (M - 1)^p \f$.
     */
    [[nodiscard]] constexpr value_type scale() const noexcept { return scale_; }

    /**
     * @brief Get an iterator to the first grid point.
     *
     * @return Iterator to the first grid point \f$ g(0) \f$.
     */
    [[nodiscard]] constexpr auto begin() const noexcept { return grid_iterator<power_grid> { *this }; }

    /**
     * @brief Get a const iterator to the first grid point.
     *
     * @return Const iterator to the first grid point \f$ g(0) \f$.
     */
    [[nodiscard]] constexpr auto cbegin() const noexcept { return begin(); }

    /**
     * @brief Get an iterator to one past the last grid point.
     *
     * @return Iterator to one past the last grid point.
     */
    [[nodiscard]] constexpr auto end() const noexcept { return grid_iterator<power_grid> { *this, size_ }; }

    /**
     * @brief Get a const iterator to one past the last grid point.
     *
     * @return Const iterator to one past the last grid point.
     */
    [[nodiscard]] constexpr auto cend() const noexcept { return end(); }

private:
    value_type power_ { 1.0 };
    value_type scale_ { 1.0 };
};

/**
 * @brief Serialize a simplemc::power_grid.
 *
 * @details It serializes the first and last grid points, \f$ a \f$ and \f$ b \f$, the number of grid 
 * points \f$ M \f$ and the power exponent \f$ p \f$.
 *
 * @tparam S simplemc::serializer type.
 * @param s Serializer object.
 * @param g Power grid to serialize.
 */
template <serializer S>
void simplemc_save(S& s, const power_grid& g) {
    s.save_at("first", g.first());
    s.save_at("last", g.last());
    s.save_at("size", g.size());
    s.save_at("power", g.power());
}

/**
 * @brief Deserialize a simplemc::power_grid.
 *
 * @details It first deserializes the first and last grid points, \f$ a \f$ and \f$ b \f$, the number 
 * of grid points \f$ M \f$ and the power exponent \f$ p \f$. It then uses them to reset the grid 
 * (see simplemc::power_grid::reset).
 *
 * @tparam S simplemc::deserializer type.
 * @param s Deserializer object.
 * @param g Power grid to deserialize into.
 */
template <deserializer S>
void simplemc_load(const S& s, power_grid& g) {
    double first = 0;
    double last = 0;
    long size = 2;
    double power = 1.0;
    s.load_at("first", first);
    s.load_at("last", last);
    s.load_at("size", size);
    s.load_at("power", power);
    g.reset(first, last, size, power);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_GRIDS_POWER_GRID_HPP
