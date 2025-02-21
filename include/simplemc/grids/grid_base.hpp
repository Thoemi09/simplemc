/**
 * @file
 * @brief Base class for 1-dimensional grid types.
 */

#ifndef SIMPLEMC_GRIDS_GRID_BASE_HPP
#define SIMPLEMC_GRIDS_GRID_BASE_HPP

#include <simplemc/utils/indexing.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cassert>
#include <cstddef>
#include <cmath>

namespace simplemc {

/**
 * @ingroup simplemc-grids-1d
 * @brief Base class for various 1-dimensional grid types.
 *
 * @details In the following, we use the notation from @ref simplemc-grids-1d.
 *
 * A basic grid is defined by its first and last grid points, \f$ a \f$ and \f$ b \f$, and
 * its size \f$ M \f$, i.e. the number of grid points.
 *
 * Specific grids should inherit from this class and provide implementations for the pure virtual
 * grid_base::at and grid_base::index methods. They correspond to the maps \f$ g : \mathrm{I} \to [a,
 * b] \f$ and \f$ \tilde{g}^{-1} : [a, b] \to \mathrm{I} \f$, respectively.
 *
 * All other virtual methods have reasonable default implementations although they can be overriden if
 * necessary.
 */
class grid_base {
public:
    /**
     * @brief Get number of dimensions.
     *
     * @return Number of dimensions (\f$ = 1\f$).
     */
    static constexpr std::size_t dim() { return 1; }

    /**
     * @brief Type of the grid's range, i.e. the type of the grid points.
     */
    using value_type = double;

    /**
     * @brief Type of the grid's domain, i.e. the type of the grid indices.
     */
    using size_type = long;

    /**
     * @brief Default constructor constructs a basic grid of size \f$ M = 2 \f$ on the interval \f$
     * [0, 1] \f$.
     */
    grid_base() = default;

    /**
     * @brief Construct a base grid by specifying its first and last grid points, \f$ a \f$ and \f$ b
     * \f$, and its size \f$ M \f$.
     *
     * @details Throws a simplemc::simplemc_exception if
     * - \f$ a = b \f$ or
     * - \f$ M < 2 \f$.
     *
     * @param a First value of the grid, \f$ a = g(0) \f$.
     * @param b Last value of the grid, \f$ b = g(M-1) \f$.
     * @param m Number of grid points \f$ M \f$.
     */
    grid_base(value_type a, value_type b, size_type m) { reset(a, b, m); }

    /**
     * @brief Reset the base grid by specifying its first and last grid points, \f$ a \f$ and \f$ b
     * \f$, and its size \f$ M \f$.
     *
     * @details Throws a simplemc::simplemc_exception if
     * - \f$ a = b \f$ or
     * - \f$ M < 2 \f$.
     *
     * @param a First value of the grid, \f$ a = g(0) \f$.
     * @param b Last value of the grid, \f$ b = g(M-1) \f$.
     * @param m Number of grid points \f$ M \f$.
     */
    void reset(value_type a, value_type b, size_type m) {
        if (m < 2) {
            throw simplemc_exception("Grid size must be >= 2", "grid_base::reset");
        }
        if (a == b) {
            throw simplemc_exception("First and last grid value have to be different", "grid_base::reset");
        }
        first_ = a;
        last_ = b;
        size_ = m;
    }

    /**
     * @brief Default virtual destructor.
     */
    virtual ~grid_base() = default;

    /**
     * @brief Get the first grid point \f$ a \f$.
     *
     * @return Value of \f$ g(0) \f$.
     */
    [[nodiscard]] value_type first() const { return first_; }

    /**
     * @brief Get the last grid point \f$ b \f$.
     *
     * @return Value of \f$ g(M-1) \f$.
     */
    [[nodiscard]] value_type last() const { return last_; }

    /**
     * @brief Get the grid size.
     *
     * @return Number of grid points \f$ M \f$ on the grid.
     */
    [[nodiscard]] size_type size() const { return size_; }

    /**
     * @brief Get the grid point \f$ g(i) \f$ at a given index \f$ i \f$.
     *
     * @param idx Index \f$ i \f$ of the grid point.
     * @return Grid point \f$ g(i) \f$.
     */
    [[nodiscard]] virtual value_type at(size_type idx) const = 0;

    /**
     * @brief Get the index \f$ i \f$ such that a given value \f$ x \in [a, b] \f$ lies in bin \f$
     * b_i \f$.
     *
     * @param value Some value \f$ x \in [a, b] \f$.
     * @return Index \f$ i = \tilde{g}^{-1}(x) \f$ of the bin \f$ b_i \f$ such that \f$ x \in b_i \f$.
     */
    [[nodiscard]] virtual size_type index(value_type value) const = 0;

    /**
     * @brief Find a subrange of consecutive grid points which contain a given value \f$ x \f$.
     *
     * @details The subrange consists of \f$ m \f$ consecutive grid points and is chosen such that the
     * given value \f$ x \in [a, b] \f$ lies, as far as possible, in the middle of the interval
     * - \f$ [g(i), g(i+m-1)] \f$ for increasing grids or
     * - \f$ [g(i+m-1), g(i)] \f$ for decreasing grids.
     *
     * It first calls grid_base::index for the given \f$ x \f$ and then forwards the result to
     * simplemc::integer_subrange.
     *
     * @param m Size of the wanted subrange.
     * @param value Some value \f$ x \in [a, b] \f$.
     * @return Index of bin \f$ i \f$ such that \f$ x \f$ lies in the middle of \f$ [g(i), g(i+m-1)]
     * \f$ (or \f$ [g(i+m-1), g(i)] \f$ for decreasing grids).
     */
    [[nodiscard]] virtual size_type index_subrange(size_type m, value_type value) const {
        return integer_subrange(index(value), size(), m);
    }

    /**
     * @brief Get the volume (size) of the bin at a given index \f$ i \f$.
     *
     * @param idx Bin index \f$ i \f$.
     * @return Volume (size) of the bin \f$ b_i \f$.
     */
    [[nodiscard]] virtual value_type bin_volume(size_type idx) const {
        assert(idx >= 0 && idx + 1 < size_);
        return std::abs(at(idx + 1) - at(idx));
    }

    /**
     * @brief Get the center of the bin at a given index \f$ i \f$.
     *
     * @param idx Bin index \f$ i \f$.
     * @return Center of the bin \f$ b_i \f$.
     */
    [[nodiscard]] virtual value_type center(size_type idx) const {
        assert(idx >= 0 && idx + 1 < size_);
        return (at(idx) + at(idx + 1)) / 2.;
    }

    /**
     * @brief Get a lazy view on the grid.
     *
     * @details
     * @code{.cpp}
     * // create a linear grid of size 5 on [0, 1]
     * simplemc::linear_grid lg(0.0, 1.0, 5);
     *
     * // print the grid points as a view
     * fmt::println("{}", lg.view());
     * @endcode
     *
     * Output:
     *
     * ```
     * [0, 0.25, 0.5, 0.75, 1]
     * ```
     *
     * @return View on the grid points.
     */
    [[nodiscard]] auto view() const {
        return ranges::iota_view<size_type, size_type>(0, size_) |
            ranges::views::transform([this](auto idx) { return at(idx); });
    }

    /**
     * @brief Get a lazy view on the bin centers.
     *
     * @details
     * @code{.cpp}
     * // create a linear grid of size 5 on [0, 1]
     * simplemc::linear_grid lg(0.0, 1.0, 5);
     *
     * // print the bin centers as a view
     * fmt::println("{}", lg.view_center());
     * @endcode
     *
     * Output:
     *
     * ```
     * [0.125, 0.375, 0.625, 0.875]
     * ```
     *
     * @return View on the bin centers.
     */
    [[nodiscard]] auto view_center() const {
        return ranges::iota_view<size_type, size_type>(0, size_ - 1) |
            ranges::views::transform([this](auto idx) { return center(idx); });
    }

    /**
     * @brief Get a lazy view on the bin volumes.
     *
     * @details
     * @code{.cpp}
     * // create a linear grid of size 5 on [0, 1]
     * simplemc::linear_grid lg(0.0, 1.0, 5);
     *
     * // print the bin volumes as a view
     * fmt::println("{}", lg.view_bin_volumes());
     * @endcode
     *
     * Output:
     *
     * ```
     * [0.25, 0.25, 0.25, 0.25]
     * ```
     *
     * @return View on the bin volumes.
     */
    [[nodiscard]] auto view_bin_volumes() const {
        return ranges::iota_view<size_type, size_type>(0, size_ - 1) |
            ranges::views::transform([this](auto idx) { return bin_volume(idx); });
    }

protected:
    value_type first_ { 0.0 };
    value_type last_ { 1.0 };
    size_type size_ { 2 };
};

} // namespace simplemc

#endif // SIMPLEMC_GRIDS_GRID_BASE_HPP
