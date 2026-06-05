/**
 * @file
 * @brief 1-dimensional symmetric power grid.
 */

#ifndef SIMPLEMC_GRIDS_SYMMETRIC_POWER_GRID_HPP
#define SIMPLEMC_GRIDS_SYMMETRIC_POWER_GRID_HPP

#include <simplemc/grids/grid_iterator.hpp>
#include <simplemc/grids/power_grid.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

namespace simplemc {

/**
 * @addtogroup simplemc-grids-1d
 * @{
 */

/**
 * @brief 1-dimensional symmetric power grid.
 *
 * @details In the following, we use the notation from @ref simplemc-grids-1d.
 *
 * This class inherits from simplemc::grid_base and satisfies the simplemc::grid_1d concept. It
 * represents a symmetric power grid of odd size \f$ M > 2 \f$ on the interval
 * - \f$ \mathrm{R} = [a, b] \f$ for increasing grids, i.e. \f$ a < b \f$, or
 * - \f$ \mathrm{R} = [b, a] \f$ for decreasing grids, i.e. \f$ a > b \f$.
 *
 * It uses the function
 * \f[
 *   g(i) =
 *   \begin{cases}
 *   g_1(i) & \text{if } i \leq i_c, \\
 *   g_2(M - 1 - i) & \text{if } i > i_c
 *   \end{cases}
 *   \; ,
 * \f]
 * to map an index \f$ i \in \{ 0, 1, \dots, M-1 \} \f$ to its grid point \f$ g(i) \f$. Here, \f$ c =
 * (a + b) / 2 \f$ is the midpoint of the grid and \f$ i_c = \left\lfloor M / 2 \right\rfloor \f$ is
 * the corresponding index. \f$ g_1 \f$ and \f$ g_2 \f$ are power grids (see simplemc::power_grid) of
 * size \f$ \left\lfloor M / 2 \right\rfloor + 1 \f$ with the range \f$ [a, c] \f$ and \f$ [b, c] \f$,
 * respectively. Note that for increasing (decreasing) grids, \f$ g_1 \f$ is increasing (decreasing),
 * while \f$ g_2 \f$ is decreasing (increasing).
 *
 * The corresponding inverse function
 * \f[
 *   \tilde{g}^{-1}(x) =
 *   \begin{cases}
 *   g_1^{-1}(x) & \text{if increasing and } x \leq c \text{ or if decreasing and } x \geq c \\
 *   M - 2 - g_2^{-1}(x) & \text{otherwise}
 *   \end{cases}
 *   \; ,
 * \f]
 * maps every value \f$ x \in [a, b] \f$ to the index \f$ i \f$ such that \f$ x \in b_i \f$.
 *
 * @include grids/doc_symmetric_power_grid.cpp
 *
 * Output:
 *
 * ```
 * Grid points: [0, 0.03125, 0.125, 0.28125, 0.5, 0.71875, 0.875, 0.96875, 1]
 * Bin centers: [0.015625, 0.078125, 0.203125, 0.390625, 0.609375, 0.796875, 0.921875, 0.984375]
 * Bin volumes: [0.03125, 0.09375, 0.15625, 0.21875, 0.21875, 0.15625, 0.09375, 0.03125]
 * ```
 */
class symmetric_power_grid : public grid_base<symmetric_power_grid> {
public:
    /**
     * @brief Type of the CRTP base class.
     */
    using base_type = grid_base<symmetric_power_grid>;

    /**
     * @brief Default constructor constructs a symmetric power grid of size \f$ M = 3 \f$ on the
     * interval \f$ [0, 1] \f$ with \f$ p = 2 \f$.
     */
    constexpr symmetric_power_grid() { reset(0.0, 1.0, 3, 2); }

    /**
     * @brief Construct a symmetric power grid by specifying its first and last grid points, \f$ a \f$
     * and \f$ b \f$, its size \f$ M \f$ and the power parameter \f$ p \f$.
     *
     * @details It simply forwards the arguments to reset().
     *
     * @param a First value of the grid, \f$ a = g(0) \f$.
     * @param b Last value of the grid, \f$ b = g(M-1) \f$.
     * @param m Number of grid points \f$ M \f$.
     * @param p Power parameter \f$ p \f$.
     */
    constexpr symmetric_power_grid(value_type a, value_type b, size_type m, value_type p) { reset(a, b, m, p); }

    /**
     * @brief Reset a symmetric power grid by specifying its first and last grid points, \f$ a \f$
     * and \f$ b \f$, its size \f$ M \f$ and the power parameter \f$ p \f$.
     *
     * @details It sets the midpoint \f$ c = (a + b) / 2 \f$ and initializes the power grids \f$ g_1
     * \f$ and \f$ g_2 \f$ of size \f$ \left\lfloor M / 2 \right\rfloor + 1 \f$.
     *
     * It throws an exception if the size is not odd.
     *
     * @param a First value of the grid, \f$ a = g(0) \f$.
     * @param b Last value of the grid, \f$ b = g(M-1) \f$.
     * @param m Number of grid points \f$ M \f$.
     * @param p Power parameter \f$ p \f$.
     */
    constexpr void reset(value_type a, value_type b, size_type m, value_type p) {
        if (m % 2 != 1) {
            throw simplemc_exception(
                "Number of grid points needs to be odd in symmetric_power_grid", "symmetric_power_grid::reset");
        }
        base_type::reset(a, b, m);
        midpoint_ = (first_ + last_) / 2;
        g1_.reset(first_, midpoint_, static_cast<size_type>(size_ / 2) + 1, p);
        g2_.reset(last_, midpoint_, static_cast<size_type>(size_ / 2) + 1, p);
    }

    /**
     * @brief Get the grid point \f$ g(i) \f$ at a given index \f$ i \f$.
     *
     * @details If \f$ i \leq i_c \f$, it returns \f$ g_1(i) \f$. Otherwise, it returns
     * \f$ g_2(M - 1 - i) \f$.
     *
     * @param idx Index \f$ i \f$ of the grid point.
     * @return Grid point \f$ g(i) \f$.
     */
    [[nodiscard]] constexpr value_type at(size_type idx) const {
        assert(idx >= 0 && idx < size_);
        if (idx <= (size_ - 1) / 2) {
            return g1_.at(idx);
        } else {
            return g2_.at(size_ - 1 - idx);
        }
    }

    /**
     * @brief Get the index \f$ i \f$ such that a given value \f$ x \in [a, b] \f$ lies in bin \f$
     * b_i \f$.
     *
     * @details If the grid is increasing and \f$ x \leq c \f$ or if the grid is decreasing and \f$ x
     * \geq c \f$, it returns \f$ \tilde{g}_1^{-1}(x) \f$.
     *
     * Otherwise, it returns \f$ M - 2 - \tilde{g}_2^{-1}(x) \f$.
     *
     * @param value Some value \f$ x \in [a, b] \f$.
     * @return Index \f$ i = \tilde{g}^{-1}(x) \f$ of the bin \f$ b_i \f$ such that \f$ x \in b_i \f$.
     */
    [[nodiscard]] constexpr size_type index(value_type value) const {
        assert((first_ <= value && value <= last_) || (first_ >= value && value >= last_));
        if ((value <= midpoint_ && first_ < last_) || (value >= midpoint_ && first_ > last_)) {
            return g1_.index(value);
        } else {
            return size_ - 2 - g2_.index(value);
        }
    }

    /**
     * @brief Get the midpoint of the grid.
     *
     * @return Center of the grid, \f$ c = (a + b) / 2 \f$.
     */
    [[nodiscard]] constexpr value_type midpoint() const noexcept { return midpoint_; }

    /**
     * @brief Get the power grid defined on the interval \f$ [a, c] \f$.
     *
     * @return Power grid \f$ g_1 \f$.
     */
    [[nodiscard]] constexpr const auto& grid1() const noexcept { return g1_; }

    /**
     * @brief Get the power grid defined on the interval \f$ [b, c] \f$.
     *
     * @return Power grid \f$ g_2 \f$.
     */
    [[nodiscard]] constexpr const auto& grid2() const noexcept { return g2_; }

    /**
     * @brief Get an iterator to the first grid point.
     *
     * @return Iterator to the first grid point \f$ g(0) \f$.
     */
    [[nodiscard]] constexpr auto begin() const noexcept { return grid_iterator<symmetric_power_grid> { *this }; }

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
    [[nodiscard]] constexpr auto end() const noexcept { return grid_iterator<symmetric_power_grid> { *this, size_ }; }

    /**
     * @brief Get a const iterator to one past the last grid point.
     *
     * @return Const iterator to one past the last grid point.
     */
    [[nodiscard]] constexpr auto cend() const noexcept { return end(); }

private:
    value_type midpoint_ { 0.5 };
    power_grid g1_ { 0.0, 0.5, 2, 1 };
    power_grid g2_ { 1.0, 0.5, 2, 1 };
};

/**
 * @brief Serialize a simplemc::symmetric_power_grid.
 *
 * @details It serializes the first and last grid points, \f$ a \f$ and \f$ b \f$, the number of grid
 * points \f$ M \f$ and the power exponent \f$ p \f$.
 *
 * @tparam S simplemc::serializer type.
 * @param s Serializer object.
 * @param g Symmetric power grid to serialize.
 */
template <serializer S>
void simplemc_save(S& s, const symmetric_power_grid& g) {
    s.save_at("first", g.first());
    s.save_at("last", g.last());
    s.save_at("size", g.size());
    s.save_at("power", g.grid1().power());
}

/**
 * @brief Deserialize a simplemc::symmetric_power_grid.
 *
 * @details It first deserializes the first and last grid points, \f$ a \f$ and \f$ b \f$, the number
 * of grid points \f$ M \f$ and the power exponent \f$ p \f$. It then uses them to reset the grid
 * (see simplemc::symmetric_power_grid::reset).
 *
 * @tparam S simplemc::serializer type.
 * @param s Serializer object.
 * @param g Symmetric power grid to deserialize into.
 */
template <serializer S>
void simplemc_load(const S& s, symmetric_power_grid& g) {
    double first = 0;
    double last = 0;
    long size = 3;
    double power = 1.0;
    s.load_at("first", first);
    s.load_at("last", last);
    s.load_at("size", size);
    s.load_at("power", power);
    g.reset(first, last, size, power);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_GRIDS_SYMMETRIC_POWER_GRID_HPP
