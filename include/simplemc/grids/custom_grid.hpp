/**
 * @file
 * @brief 1-dimensional custom grid.
 */

#ifndef SIMPLEMC_GRIDS_CUSTOM_GRID_HPP
#define SIMPLEMC_GRIDS_CUSTOM_GRID_HPP

#include <simplemc/grids/grid_base.hpp>
#include <simplemc/grids/grid_iterator.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <utility>
#include <vector>

namespace simplemc {

/**
 * @ingroup simplemc-grids-1d
 * @brief 1-dimensional custom grid.
 *
 * @details In the following, we use the notation from @ref simplemc-grids-1d.
 *
 * This class inherits from simplemc::grid_base and satisfies the simplemc::grid_1d concept. It
 * represents a custom grid of size \f$ M \geq 2 \f$ on the interval
 * - \f$ \mathrm{R} = [a, b] \f$ for increasing grids, i.e. \f$ a < b \f$, or
 * - \f$ \mathrm{R} = [b, a] \f$ for decreasing grids, i.e. \f$ a > b \f$.
 *
 * It is defined by an ordered array of grid points, \f$ \mathbf{x} = (x_0 = a, \dots, x_{M-1} = b)
 * \f$, such that for all \f$ i < j \f$
 * - \f$ x_i < x_j \f$ holds for increasing grids and
 * - \f$ x_i > x_j \f$ holds for decreasing grids.
 *
 * It simply uses \f$ g(i) = \mathbf{x}(i) = x_i \f$ to map an index \f$ i \in \{ 0, 1, \dots, M-1 \}
 * \f$ to its grid point \f$ g(i) \f$.
 *
 * The corresponding inverse function, \f$ \tilde{g}^{-1}(x) = i \f$, that maps every value \f$ x \in
 * [a, b] \f$ to the index \f$ i \f$ such that \f$ x \in b_i \f$ uses a binary search, i.e. this
 * operation is \f$ \mathcal{O}(\log M) \f$.
 *
 * @include grids/doc_custom_grid.cpp
 *
 * Output:
 *
 * ```
 * Grid points: [1, 2.3, 2.4, 5.7, 100]
 * Bin centers: [1.65, 2.35, 4.05, 52.85]
 * Bin volumes: [1.3, 0.1, 3.3, 94.3]
 * ```
 */
class custom_grid : public grid_base<custom_grid> {
public:
    /**
     * @brief Type of the CRTP base class.
     */
    using base_type = grid_base<custom_grid>;

    /**
     * @brief Default constructor constructs a custom grid of size \f$ M = 2 \f$ on the interval \f$
     * [0, 1] \f$.
     */
    constexpr custom_grid() noexcept = default;

    /**
     * @brief Construct a custom grid with the given ordered array as its grid points.
     *
     * @details It simply forwards the arguments to reset().
     *
     * @note The grid points \f$ \mathbf{x} \f$ must be ordered (either increasing or decreasing). If
     * not, the behavior is undefined.
     *
     * @param x Ordered array of grid points.
     */
    constexpr custom_grid(std::vector<value_type> x) { reset(std::move(x)); }

    /**
     * @brief Reset the custom grid by specifying the grid points.
     *
     * @details It stores the grid points \f$ \mathbf{x} \f$ and calls base_type::reset with the first
     * and last grid points and the number of grid points.
     *
     * @note The grid points \f$ \mathbf{x} \f$ must be ordered (either increasing or decreasing). If
     * not, the behavior is undefined.
     *
     * @param x Ordered array of grid points.
     */
    constexpr void reset(std::vector<value_type> x) {
        assert(std::ranges::is_sorted(x) || std::ranges::is_sorted(x, std::greater<>()));
        x_ = std::move(x);
        base_type::reset(x_.front(), x_.back(), static_cast<size_type>(x_.size()));
    }

    /**
     * @brief Get the grid point \f$ g(i) \f$ at a given index \f$ i \f$.
     *
     * @param idx Index \f$ i \f$ of the grid point.
     * @return Grid point \f$ g(i) \f$.
     */
    [[nodiscard]] constexpr value_type at(size_type idx) const noexcept {
        assert(idx >= 0 && idx < size_);
        return x_[idx];
    }

    /**
     * @brief Get the index \f$ i \f$ such that a given value \f$ x \in [a, b] \f$ lies in bin \f$
     * b_i \f$.
     *
     * @param value Some value \f$ x \in [a, b] \f$.
     * @return Index \f$ i = \tilde{g}^{-1}(x) \f$ of the bin \f$ b_i \f$ such that \f$ x \in b_i \f$.
     */
    [[nodiscard]] constexpr size_type index(value_type value) const noexcept {
        assert((first_ <= value && value <= last_) || (first_ >= value && value >= last_));
        if (first_ < last_) {
            return std::distance(x_.cbegin(), --std::ranges::upper_bound(x_, value));
        } else {
            return std::distance(x_.cbegin(), --std::ranges::upper_bound(x_, value, std::greater<>()));
        }
    }

    /**
     * @brief Get the array of grid points.
     *
     * @return `std::vector<value_type>` of grid points \f$ \mathbf{x} \f$.
     */
    [[nodiscard]] constexpr auto const& grid_points() const noexcept { return x_; }

    /**
     * @brief Get an iterator to the first grid point.
     *
     * @return Iterator to the first grid point \f$ g(0) \f$.
     */
    [[nodiscard]] constexpr auto begin() const noexcept { return grid_iterator<custom_grid> { *this }; }

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
    [[nodiscard]] constexpr auto end() const noexcept { return grid_iterator<custom_grid> { *this, size_ }; }

    /**
     * @brief Get a const iterator to one past the last grid point.
     *
     * @return Const iterator to one past the last grid point.
     */
    [[nodiscard]] constexpr auto cend() const noexcept { return end(); }

private:
    std::vector<value_type> x_ { 0.0, 1.0 };
};

} // namespace simplemc

#endif // SIMPLEMC_GRIDS_CUSTOM_GRID_HPP
