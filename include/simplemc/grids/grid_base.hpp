/**
 * @file
 * @brief Base class for 1-dimensional grid types.
 */

#ifndef SIMPLEMC_GRIDS_GRID_BASE_HPP
#define SIMPLEMC_GRIDS_GRID_BASE_HPP

#include <simplemc/utils/indexing.hpp>

#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>

#include <cassert>
#include <cstddef>
#include <cmath>

namespace simplemc {

/**
 * @ingroup simplemc-grids
 * @brief Base class for various 1-dimensional grid types.
 *
 * @details A grid is a strictly monotone map \f$ g \f$ from the integer set \f$ \{0, 1, \ldots, N-1\}
 * \f$ to a closed, real interval \f$ [a, b] \f$.
 *
 * It is defined by
 * - the first grid point, \f$ g(0) = a \f$,
 * - the last grid point, \f$ g(N-1) = b \neq a \f$,
 * - the size of the grid, \f$ N \geq 2 \f$ and
 * - the map \f$ g \f$.
 *
 * A grid with \f$ N \f$ grid points can also be thought of as \f$ N - 1 \f$ bins. Two consecutive
 * grid points, \f$ g(i) \f$ and \f$ g(i+1) \f$, then enclose the bin \f$ b_i \f$, i.e. \f$ b_i =
 * [g(i), g(i+1)) \f$.
 *
 * Since the map is monotone, one can always define the inverse map \f$ g^{-1} \f$, which maps a
 * given point on the grid to the corresponding grid index.
 *
 * The above is also true for decreasing grids, i.e. in the case that \f$ b < a \f$. The range of the
 * grid is then \f$ [b, a] \f$ and the bins are the intervals \f$ b_i = (g(i+1), g(i)] \f$.
 *
 * Depending on the map, one can define various grids, see e.g. simplemc::linear_grid or
 * simplemc::power_grid.
 */
class grid_base {
public:
    /**
     * @brief Get number of dimensions.
     *
     * @return Number of dimensions.
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
     * @brief Default constructor leaves the grid in an empty state with a size of zero.
     */
    grid_base() = default;

    /**
     * @brief Construct a base grid by specifying its first and last grid points and its size.
     *
     * @param first First value of the grid.
     * @param last Last value of the grid.
     * @param size Number of grid points.
     */
    grid_base(value_type first, value_type last, size_type size);

    /**
     * @brief Reset the base grid by specifying its first and last grid points and its size.
     *
     * @param first First value of the grid.
     * @param last Last value of the grid.
     * @param size Number of grid points.
     */
    void reset(value_type first, value_type last, size_type size);

    /**
     * @brief Default virtual destructor.
     */
    virtual ~grid_base() = default;

    /**
     * @brief Get the first grid point.
     *
     * @return First value of the grid.
     */
    [[nodiscard]] value_type first() const { return first_; }

    /**
     * @brief Get the last grid point.
     *
     * @return Last value of the grid.
     */
    [[nodiscard]] value_type last() const { return last_; }

    /**
     * @brief Get the grid size.
     *
     * @return Number of grid points on the grid.
     */
    [[nodiscard]] size_type size() const { return size_; }

    /**
     * @brief Is the grid empty?
     *
     * @return True if the number of grid points is zero, otherwise false.
     */
    [[nodiscard]] bool empty() const { return size_ == 0; }

    /**
     * @brief Get the grid point at a given index.
     *
     * @param idx Index of the grid point.
     * @return Grid point at the given index.
     */
    [[nodiscard]] virtual value_type at(size_type idx) const = 0;

    /**
     * @brief Get the index of the bin to which a given value belongs.
     *
     * @param value A value in the range of the grid.
     * @return Index of the bin which contains the given value.
     */
    [[nodiscard]] virtual size_type index(value_type value) const = 0;

    /**
     * @brief Find the subrange of \f$ m \f$ consecutive grid points such that the given value lies
     * more or less in the middle of the interval \f$ [g(i), g(i+m-1)] \f$.
     *
     * @details It first calls grid_base::index with the given value and then forwards the result to
     * simplemc::integer_subrange.
     *
     * @param m Size of the wanted subrange.
     * @param value A value in the range of the grid.
     * @return Index of bin \f$ i \f$ such that the value lies in the "middle" of \f$ [g(i),
     * g(i+m-1)] \f$.
     */
    [[nodiscard]] virtual size_type index_subrange(size_type m, value_type value) const {
        return integer_subrange(index(value), size(), m);
    }

    /**
     * @brief Get the volume (size) of the bin at a given index.
     *
     * @param idx Bin index.
     * @return Bin volume (size) at the given index.
     */
    [[nodiscard]] virtual value_type bin_volume(size_type idx) const {
        assert(idx >= 0 && idx + 1 < size_);
        return std::abs(at(idx + 1) - at(idx));
    }

    /**
     * @brief Get the center of the bin at a given index.
     *
     * @param idx Bin index
     * @return Center of the bin at the given index.
     */
    [[nodiscard]] virtual value_type center(size_type idx) const {
        assert(idx >= 0 && idx + 1 < size_);
        return (at(idx) + at(idx + 1)) / 2.;
    }

    /**
     * @brief Get a lazy view on the grid.
     *
     * @return `ranges::view` on the grid points.
     */
    [[nodiscard]] auto view() const {
        return ranges::iota_view<size_type, size_type>(0, size_) |
            ranges::views::transform([this](auto idx) { return at(idx); });
    }

    /**
     * @brief Get a lazy view on the bin centers.
     *
     * @return `ranges::view` on the bin centers.
     */
    [[nodiscard]] auto view_center() const {
        return ranges::iota_view<size_type, size_type>(0, size_ - 1) |
            ranges::views::transform([this](auto idx) { return center(idx); });
    }

    /**
     * @brief Get a lazy view on the bin volumes.
     *
     * @return `ranges::view` on the bin volumes.
     */
    [[nodiscard]] auto view_bin_volumes() const {
        return ranges::iota_view<size_type, size_type>(0, size_ - 1) |
            ranges::views::transform([this](auto idx) { return bin_volume(idx); });
    }

protected:
    value_type first_ { 0.0 };
    value_type last_ { 0.0 };
    size_type size_ { 0 };
};

} // namespace simplemc

#endif // SIMPLEMC_GRIDS_GRID_BASE_HPP
