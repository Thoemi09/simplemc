/**
 * @file grid_base.hpp
 * @brief Base class used for all 1-dimensional grid types.
 */

#ifndef SIMPLEMC_GRIDS_GRID_BASE_HPP
#define SIMPLEMC_GRIDS_GRID_BASE_HPP

#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>

#include <cassert>
#include <cmath>

namespace simplemc {

/**
 * @brief Base class for various 1-dimensional grid types.
 *
 * @details A grid is a collection of points on a real, closed interval. It is defined by
 * the value of the first grid point, the value of the last grid point (!= first) and its
 * size (>= 2).
 */
class grid_base {
public:
    /**
     * @brief Value type of the grid.
     */
    using value_type = double;

    /**
     * @brief Size type of the grid.
     */
    using size_type = long;

    /**
     * @brief Default constructor.
     */
    grid_base() = default;

    /**
     * @brief Constructor for a base class grid.
     *
     * @param first First value of the grid.
     * @param last Last value of the grid.
     * @param size Number of grid points.
     */
    grid_base(value_type first, value_type last, size_type size);

    /**
     * @brief Reset the grid.
     *
     * @param first First value of the grid.
     * @param last Last value of the grid.
     * @param size Number of grid points.
     */
    void reset(value_type first, value_type last, size_type size);

    /**
     * @brief Virtual destructor.
     */
    virtual ~grid_base() = default;

    /**
     * @brief Get first grid point.
     *
     * @return First value of grid.
     */
    [[nodiscard]] value_type first() const { return first_; }

    /**
     * @brief Get last grid point.
     *
     * @return Last value of grid.
     */
    [[nodiscard]] value_type last() const { return last_; }

    /**
     * @brief Get grid size.
     *
     * @return Number of grid points.
     */
    [[nodiscard]] size_type size() const { return size_; }

    /**
     * @brief Get grid point at a certain index.
     *
     * @param idx Index of grid point.
     * @return Value at that index.
     */
    [[nodiscard]] virtual value_type at(size_type idx) const = 0;

    /**
     * @brief Get index of bin to which the value belongs.
     *
     * @param value Input value.
     * @return Index of bin, which contains the supplied value.
     */
    [[nodiscard]] virtual size_type index(value_type value) const = 0;

    /**
     * @brief Get volume of the bin at a certain index.
     *
     * @param idx Index of bin.
     * @return Bin volume at that index.
     */
    [[nodiscard]] virtual value_type bin_volume(size_type idx) const {
        assert(idx >= 0 && idx + 1 < size_);
        return std::abs(at(idx + 1) - at(idx));
    }

    /**
     * @brief Get the center of the bin at a certain index.
     *
     * @param idx Index of bin.
     * @return Center of bin at that index.
     */
    [[nodiscard]] virtual value_type center(size_type idx) const {
        assert(idx >= 0 && idx + 1 < size_);
        return (at(idx) + at(idx + 1)) / 2.;
    }

    /**
     * @brief Get a view on the grid.
     *
     * @return View on the grid.
     */
    [[nodiscard]] auto view() const {
        return ranges::iota_view<size_type, size_type>(0, size_) |
            ranges::views::transform([this](auto idx) { return at(idx); });
    }

    /**
     * @brief Get a view on the bin centers.
     *
     * @return View on the bin centers.
     */
    [[nodiscard]] auto view_center() const {
        return ranges::iota_view<size_type, size_type>(0, size_ - 1) |
            ranges::views::transform([this](auto idx) { return center(idx); });
    }

    /**
     * @brief Get a view on the bin volumes.
     *
     * @return View on the bin volumes.
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
