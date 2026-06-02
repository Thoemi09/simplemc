/**
 * @file
 * @brief 1-dimensional linear grid.
 */

#ifndef SIMPLEMC_GRIDS_LINEAR_GRID_HPP
#define SIMPLEMC_GRIDS_LINEAR_GRID_HPP

#include <simplemc/grids/grid_base.hpp>
#include <simplemc/grids/grid_iterator.hpp>
#include <simplemc/serialize/concepts.hpp>

namespace simplemc {

/**
 * @ingroup simplemc-grids-1d
 * @brief 1-dimensional linear grid.
 *
 * @details In the following, we use the notation from @ref simplemc-grids-1d.
 *
 * This class inherits from simplemc::grid_base and satisfies the simplemc::grid_1d concept. It
 * represents a uniform grid of size \f$ M \geq 2 \f$ on the interval
 * - \f$ \mathrm{R} = [a, b] \f$ for increasing grids, i.e. \f$ a < b \f$, or
 * - \f$ \mathrm{R} = [b, a] \f$ for decreasing grids, i.e. \f$ a > b \f$.
 *
 * It uses the linear function
 * \f[
 *   g(i) = a + i * \Delta \;,
 * \f]
 * to map an index \f$ i \in \{ 0, 1, \dots, M-1 \} \f$ to its grid point \f$ g(i) \f$, where
 * \f$ \Delta = (b - a) / (M - 1) \f$ is the step size or the size of each bin \f$ b_i \f$.
 *
 * The corresponding inverse function
 * \f[
 *   \tilde{g}^{-1}(x) = \left\lfloor \frac{x - a}{\Delta} \right\rfloor \;,
 * \f]
 * maps every value \f$ x \in [a, b] \f$ to the index \f$ i \f$ such that \f$ x \in b_i \f$.
 *
 * @include grids/doc_linear_grid.cpp
 *
 * Output:
 *
 * ```
 * Grid points: [0, 0.25, 0.5, 0.75, 1]
 * Bin centers: [0.125, 0.375, 0.625, 0.875]
 * Bin volumes: [0.25, 0.25, 0.25, 0.25]
 * ```
 */
class linear_grid : public grid_base<linear_grid> {
public:
    /**
     * @brief Type of the CRTP base class.
     */
    using base_type = grid_base<linear_grid>;

    /**
     * @brief Default constructor constructs a linear grid of size \f$ M = 2 \f$ on the interval \f$
     * [0, 1] \f$.
     */
    constexpr linear_grid() noexcept = default;

    /**
     * @brief Construct a linear grid by specifying its first and last grid points, \f$ a \f$ and \f$
     * b \f$, and its size \f$ M \f$.
     *
     * @details It simply forwards the arguments to reset().
     *
     * @param a First value of the grid, \f$ a = g(0) \f$.
     * @param b Last value of the grid, \f$ b = g(M-1) \f$.
     * @param m Number of grid points \f$ M \f$.
     */
    constexpr linear_grid(value_type a, value_type b, size_type m) { reset(a, b, m); }

    /**
     * @brief Reset the linear grid by specifying its first and last grid points, \f$ a \f$ and \f$ b
     * \f$, and its size \f$ M \f$.
     *
     * @details It forwards the arguments to base_type::reset and sets the step size \f$ \Delta =
     * (b - a) / (M - 1) \f$.
     *
     * @param a First value of the grid, \f$ a = g(0) \f$.
     * @param b Last value of the grid, \f$ b = g(M-1) \f$.
     * @param m Number of grid points \f$ M \f$.
     */
    constexpr void reset(value_type a, value_type b, size_type m) {
        base_type::reset(a, b, m);
        step_ = (last_ - first_) / (static_cast<double>(size_) - 1.0);
    }

    /**
     * @brief Get the grid point \f$ g(i) \f$ at a given index \f$ i \f$.
     *
     * @details The mapping is defined as \f$ g(i) = a + i * \Delta \f$.
     *
     * @param idx Index \f$ i \f$ of the grid point.
     * @return Grid point \f$ g(i) \f$.
     */
    [[nodiscard]] constexpr value_type at(size_type idx) const {
        assert(idx >= 0 && idx < size_);
        return first_ + step_ * static_cast<double>(idx);
    }

    /**
     * @brief Get the index \f$ i \f$ such that a given value \f$ x \in [a, b] \f$ lies in bin \f$
     * b_i \f$.
     *
     * @details The inverse mapping is defined as \f$ \tilde{g}^{-1}(x) = \left\lfloor (x - a) /
     * \Delta \right\rfloor \f$.
     *
     * @param value Some value \f$ x \in [a, b] \f$.
     * @return Index \f$ i = \tilde{g}^{-1}(x) \f$ of the bin \f$ b_i \f$ such that \f$ x \in b_i \f$.
     */
    [[nodiscard]] constexpr size_type index(value_type value) const {
        assert((first_ <= value && value <= last_) || (first_ >= value && value >= last_));
        return static_cast<size_type>((value - first_) / step_);
    }

    /**
     * @brief Get the volume (size) of the bin at a given index \f$ i \f$.
     *
     * @param idx Bin index \f$ i \f$.
     * @return Volume (size) of the bin \f$ b_i \f$.
     */
    [[nodiscard]] constexpr value_type volume([[maybe_unused]] size_type idx) const {
        assert(idx >= 0 && idx + 1 < size_);
        return std::abs(step_);
    }

    /**
     * @brief Get the center of the bin at a given index \f$ i \f$.
     *
     * @param idx Bin index \f$ i \f$.
     * @return Center of the bin \f$ b_i \f$.
     */
    [[nodiscard]] constexpr value_type center(size_type idx) const {
        assert(idx >= 0 && idx + 1 < size_);
        return at(idx) + step_ / 2.;
    }

    /**
     * @brief Get the step size between two adjacent grid points.
     *
     * @return Step size \f$ \Delta = (b - a) / (M - 1) \f$.
     */
    [[nodiscard]] constexpr value_type step() const noexcept { return step_; }

    /**
     * @brief Get an iterator to the first grid point.
     *
     * @return Iterator to the first grid point \f$ g(0) \f$.
     */
    [[nodiscard]] constexpr auto begin() const noexcept { return grid_iterator<linear_grid> { *this }; }

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
    [[nodiscard]] constexpr auto end() const noexcept { return grid_iterator<linear_grid> { *this, size_ }; }

    /**
     * @brief Get a const iterator to one past the last grid point.
     *
     * @return Const iterator to one past the last grid point.
     */
    [[nodiscard]] constexpr auto cend() const noexcept { return end(); }

private:
    value_type step_ { 1.0 };
};

/**
 * @brief Serialize a linear_grid by its (first, last, size) parameters.
 *
 * @tparam S Serializer type.
 * @param s Serializer.
 * @param g Linear grid to save.
 */
template <serializer S>
void simplemc_save(S& s, const linear_grid& g) {
    s.save_at("first", g.first());
    s.save_at("last", g.last());
    s.save_at("size", g.size());
}

/**
 * @brief Deserialize a linear_grid (inverse of @ref simplemc_save).
 *
 * @tparam S Deserializer type.
 * @param s Deserializer.
 * @param g Linear grid to populate.
 */
template <deserializer S>
void simplemc_load(const S& s, linear_grid& g) {
    double first = 0;
    double last = 0;
    long size = 2;
    s.load_at("first", first);
    s.load_at("last", last);
    s.load_at("size", size);
    g.reset(first, last, size);
}

} // namespace simplemc

#endif // SIMPLEMC_GRIDS_LINEAR_GRID_HPP
