/**
 * @file
 * @brief CRTP base class for 1-dimensional grid types.
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
 * @brief CRTP base class for various 1-dimensional grid types.
 *
 * @details In the following, we use the notation from @ref simplemc-grids-1d.
 *
 * A basic grid is defined by its first and last grid points, \f$ a \f$ and \f$ b \f$, and
 * its size \f$ M \f$, i.e. the number of grid points.
 *
 * Specific grids should inherit from this class and provide implementations for the member functions
 * - `at(size_type) -> value_type` and
 * - `index(value_type) -> size_type`.
 *
 * They correspond to the maps \f$ g : \mathrm{I} \to [a, b] \f$ and \f$ \tilde{g}^{-1} : [a, b] \to
 * \mathrm{I} \f$, respectively.
 *
 * All other methods have reasonable default implementations although they can be overriden if
 * necessary.
 *
 * @tparam Derived 1-dimensional grid type.
 */
template <typename Derived>
class grid_base {
public:
    /**
     * @brief Get number of dimensions.
     *
     * @return Number of dimensions (\f$ = 1\f$).
     */
    static constexpr std::size_t dim() { return 1; }

    /**
     * @brief Type of the grid's range \f$ \mathrm{R} \f$, i.e. the type of the grid points.
     */
    using value_type = double;

    /**
     * @brief Type of the grid's domain \f$ \mathrm{I} \f$, i.e. the type of the grid indices.
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
     * @brief Get the volume (size) of the bin at a given index \f$ i \f$.
     *
     * @param idx Bin index \f$ i \f$.
     * @return Volume (size) of the bin \f$ b_i \f$.
     */
    [[nodiscard]] value_type volume(size_type idx) const {
        assert(idx >= 0 && idx + 1 < size_);
        auto& derived = static_cast<const Derived&>(*this);
        return std::abs(derived.at(idx + 1) - derived.at(idx));
    }

    /**
     * @brief Get the center of the bin at a given index \f$ i \f$.
     *
     * @param idx Bin index \f$ i \f$.
     * @return Center of the bin \f$ b_i \f$.
     */
    [[nodiscard]] value_type center(size_type idx) const {
        assert(idx >= 0 && idx + 1 < size_);
        auto& derived = static_cast<const Derived&>(*this);
        return (derived.at(idx) + derived.at(idx + 1)) / 2.;
    }

protected:
    value_type first_ { 0.0 };
    value_type last_ { 1.0 };
    size_type size_ { 2 };
};

} // namespace simplemc

#endif // SIMPLEMC_GRIDS_GRID_BASE_HPP
