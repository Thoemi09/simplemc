// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Iterator for grid types.
 */

#ifndef SIMPLEMC_GRIDS_GRID_ITERATOR_HPP
#define SIMPLEMC_GRIDS_GRID_ITERATOR_HPP

#include <simplemc/grids/concepts.hpp>
#include <simplemc/utils/nd_indexing.hpp>

#include <cassert>
#include <cstddef>
#include <concepts>
#include <iterator>

namespace simplemc {

/**
 * @addtogroup simplemc-grids
 * @{
 */

/**
 * @brief CRTP iterator base class for 1- and N-dimensional grids.
 *
 * @details In the following, we use the notation from @ref simplemc-grids-1d and 
 * @ref simplemc-grids-nd.
 * 
 * This class provides random access iteration for all grid types satisfying the
 * simplemc::grid_common concept.
 *
 * It stores the current index \f$ i \f$ (\f$ \equiv i_{\mathrm{lin}} \f$ for N-dimensional grids), a 
 * pointer to the grid object being iterated over and the size \f$ M \f$ of the grid. That means the 
 * underlying grid must outlive the iterator.
 *
 * Dereferencing the iterator calls the grid's `at` method with the current index. For N-dimensional
 * grids, the index is first converted to a multi-dimensional index array using simplemc::nd_index.
 *
 * The iterator is considered out of bounds when \f$ i < 0 \f$ or \f$ i \geq M \f$. In this case,
 * dereferencing the iterator results in undefined behavior.
 *
 * @note For N-dimensional grids, the current implementation calls simplemc::nd_index on each
 * dereference. See simplemc::grid_view for an alternative approach using cartesian product and
 * transformed views.
 *
 * @tparam G Grid type.
 */
template <grid_common G>
class grid_iterator {
public:
    /**
     * @brief Value type (same as the value type of the grid).
     */
    using value_type = G::value_type;

    /**
     * @brief Difference type for iterator arithmetic.
     */
    using difference_type = std::ptrdiff_t;

    /**
     * @brief Reference type.
     */
    using reference = value_type;

    /**
     * @brief Iterator category.
     */
    using iterator_category = std::random_access_iterator_tag;

    /**
     * @brief Grid type.
     */
    using grid_type = G;

    /**
     * @brief Number of dimensions of the underlying grid.
     */
    static constexpr std::size_t dim() noexcept { return G::dim(); }

    /**
     * @brief Default constructor creates an invalid iterator with no associated grid.
     */
    constexpr grid_iterator() noexcept = default;

    /**
     * @brief Construct an iterator for the given grid at the specified (flat) index.
     *
     * @details By default, the index is \f$ i = 0 \f$. For N-dimensional grids, the index corresponds
     * to the flat index, i.e. \f$ i \equiv i_{\mathrm{lin}} \f$.
     *
     * The size is set to \f$ 0 \f$ if the grid pointer is `nullptr`.
     *
     * @param grid Grid object to iterate over.
     * @param idx Index \f$ i \f$.
     */
    constexpr grid_iterator(const grid_type& grid, difference_type idx = 0) noexcept :
        grid_ptr_(&grid),
        idx_(idx),
        size_(grid_ptr_ ? grid_ptr_->size() : 0) {}

    /**
     * @brief Dereference operator to get the grid point \f$ g(i) \f$/\f$ g(\mathbf{i}) \f$ at the
     * current index.
     *
     * @details If the iterator is out of bounds, i.e. \f$ i < 0 \f$ or \f$ i \geq M \f$,
     * dereferencing is undefined behavior.
     *
     * @return Grid point at the current index.
     */
    [[nodiscard]] constexpr value_type operator*() const { return dereference(idx_); }

    /**
     * @brief Pre-increment operator increases the index \f$ i \f$ by \f$ 1 \f$.
     *
     * @return Reference to this iterator after incrementing.
     */
    constexpr grid_iterator& operator++() noexcept {
        ++idx_;
        return *this;
    }

    /**
     * @brief Post-increment operator increases the index \f$ i \f$ by \f$ 1 \f$.
     *
     * @details It calls operator++() after making a copy of the current iterator.
     *
     * @return Copy of the iterator before incrementing.
     */
    constexpr grid_iterator operator++(int) noexcept {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    /**
     * @brief Pre-decrement operator decreases the index \f$ i \f$ by \f$ 1 \f$.
     *
     * @return Reference to this iterator after decrementing.
     */
    constexpr grid_iterator& operator--() noexcept {
        --idx_;
        return *this;
    }

    /**
     * @brief Post-decrement operator decreases the index \f$ i \f$ by \f$ 1 \f$.
     *
     * @details It calls operator--() after making a copy of the current iterator.
     *
     * @return Copy of the iterator before decrementing.
     */
    constexpr grid_iterator operator--(int) noexcept {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    /**
     * @brief Addition assignment operator increases the index \f$ i \f$ by \f$ n \f$.
     *
     * @param n Increment for the index.
     * @return Reference to this iterator.
     */
    constexpr grid_iterator& operator+=(difference_type n) noexcept {
        idx_ += n;
        return *this;
    }

    /**
     * @brief Subtraction assignment operator decreases the index \f$ i \f$ by \f$ n \f$.
     *
     * @param n Decrement for the index.
     * @return Reference to this iterator.
     */
    constexpr grid_iterator& operator-=(difference_type n) noexcept {
        idx_ -= n;
        return *this;
    }

    /**
     * @brief Addition operator increases the index \f$ i \f$ by \f$ n \f$.
     *
     * @details In contrast to operator+=(difference_type), it makes a copy of the current iterator
     * and updates the copy.
     *
     * @param n Increment for the index.
     * @return Copy of this iterator with increased index.
     */
    [[nodiscard]] constexpr grid_iterator operator+(difference_type n) const noexcept {
        auto tmp = *this;
        tmp += n;
        return tmp;
    }

    /**
     * @brief Addition operator increases the index \f$ i \f$ by \f$ n \f$.
     *
     * @details See grid_iterator::operator+ for more information.
     *
     * @param n Increment for the index.
     * @param it Iterator to advance.
     * @return Copy of the given iterator with increased index.
     */
    [[nodiscard]] friend constexpr grid_iterator operator+(difference_type n, const grid_iterator& it) noexcept {
        return it + n;
    }

    /**
     * @brief Subtraction operator decreases the index \f$ i \f$ by \f$ n \f$.
     *
     * @details In contrast to operator-=(difference_type), it makes a copy of the current iterator
     * and updates the copy.
     *
     * @param n Decrement for the index.
     * @return Copy of this iterator with decreased index.
     */
    [[nodiscard]] constexpr grid_iterator operator-(difference_type n) const noexcept {
        auto tmp = *this;
        tmp -= n;
        return tmp;
    }

    /**
     * @brief Subtraction operator between two iterators.
     *
     * @details It calculates the difference between the indices of the two iterators.
     *
     * Both iterators are assumed to point to the same grid, otherwise the result is undefined.
     *
     * @param other Right-hand side iterator.
     * @return Distance between this iterator and the other.
     */
    [[nodiscard]] constexpr difference_type operator-(const grid_iterator& other) const noexcept {
        assert(grid_ptr_ == other.grid_ptr_);
        return idx_ - other.idx_;
    }

    /**
     * @brief Subscript operator to access the grid point at an offset from the current iterator.
     *
     * @details It computes the grid point at \f$ g(i + n) \f$.
     *
     * @param n Index offset.
     * @return Grid point at the index\f$ i + n \f$.
     */
    [[nodiscard]] constexpr value_type operator[](difference_type n) const { return dereference(idx_ + n); }

    /**
     * @brief Equality comparison operator compares the grid pointer and the indices of the two
     * iterators.
     *
     * @param other Right-hand side iterator.
     * @return True if iterators point to the same grid and have the same index.
     */
    [[nodiscard]] constexpr bool operator==(const grid_iterator& other) const noexcept {
        return grid_ptr_ == other.grid_ptr_ && idx_ == other.idx_;
    }

    /**
     * @brief Three-way comparison operator defines an ordering between two iterators.
     *
     * @details The ordering is the same as that of their indices using `<=>` and only makes sense if
     * both iterators point to the same grid.
     *
     * @param other Right-hand side iterator.
     * @return Ordering between the two iterators.
     */
    [[nodiscard]] constexpr auto operator<=>(const grid_iterator& other) const noexcept {
        assert(grid_ptr_ == other.grid_ptr_);
        return idx_ <=> other.idx_;
    }

    /**
     * @brief Get the pointer to the underlying grid.
     *
     * @return Grid pointer.
     */
    [[nodiscard]] constexpr const grid_type* grid_ptr() const noexcept { return grid_ptr_; }

    /**
     * @brief Get the current flat index \f$ i \equiv i_{\mathrm{lin}} \f$.
     *
     * @return Current flat index \f$ i \f$.
     */
    [[nodiscard]] constexpr difference_type flat_index() const noexcept { return idx_; }

    /**
     * @brief Get the current multi-dimensional index \f$ \mathbf{i} \f$.
     *
     * @details For 1-dimensional grids, this returns the flat index \f$ i \f$. For N-dimensional
     * grids, it converts the flat index to a multi-dimensional index array using simplemc::nd_index
     * with row-major ordering.
     *
     * @return Current multi-dimensional index \f$ \mathbf{i} \f$ (or \f$ i \f$ for 1D grids).
     */
    [[nodiscard]] constexpr auto nd_index() const {
        assert(grid_ptr_ && size_ > 0 && idx_ >= 0 && idx_ < size_);
        if constexpr (dim() == 1) {
            return idx_;
        } else {
            return simplemc::nd_index(idx_, grid_ptr_->shape(), row_major {});
        }
    }

private:
    // Helper function calls at() method of the underlying grid.
    constexpr auto dereference(difference_type idx) const {
        assert(grid_ptr_ && size_ > 0 && idx >= 0 && idx < size_);
        if constexpr (dim() == 1) {
            return grid_ptr_->at(idx);
        } else {
            return grid_ptr_->at(simplemc::nd_index(idx, grid_ptr_->shape(), row_major {}));
        }
    }

private:
    const grid_type* grid_ptr_ { nullptr };
    difference_type idx_ { 0 };
    long size_ { 0 };
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_GRIDS_GRID_ITERATOR_HPP
