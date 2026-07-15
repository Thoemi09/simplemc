// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Wrapper accumulator to add multiple values at once.
 */

#ifndef SIMPLEMC_ACCS_MULTIVALUE_ACC_HPP
#define SIMPLEMC_ACCS_MULTIVALUE_ACC_HPP

#include <simplemc/utils/ranges.hpp>

#include <cassert>
#include <concepts>
#include <vector>

namespace simplemc {

/**
 * @ingroup simplemc-accs-wrappers
 * @brief Wrapper for various accumulators to add multiple values at once.
 *
 * @details It holds a reference to an accumulator and can be used to add multiple data points to the
 * accumulator that together form a single sample. The values are buffered and only committed to the
 * accumulator (as one sample) when the user calls the multivalue_acc::commit function.
 *
 * In addition, if the wrapped accumulator is a simplemc::block_acc or simplemc::batch_acc, the user
 * has to call simplemc::block_acc::check_and_add_block or simplemc::batch_acc::check_and_advance as
 * well.
 *
 * All accumulator types mentioned above provide a factory function, `make_mva()`, that returns a
 * simplemc::multivalue_acc object:
 * - simplemc::mean_acc::make_mva(),
 * - simplemc::var_acc<X, A>::make_mva() and simplemc::var_acc<Z, A>::make_mva(),
 * - simplemc::covar_acc<X, A>::make_mva() and simplemc::covar_acc<Z, A>::make_mva(),
 * - simplemc::block_acc::make_mva(),
 * - simplemc::batch_acc::make_mva().
 *
 * @include accs/doc_multivalue_acc.cpp
 *
 * Output:
 *
 * ```
 * Mean: [3, 0, 3]
 * ```
 *
 * @tparam A Accumulator type.
 */
template <typename A>
class multivalue_acc {
public:
    /**
     * @brief Type of the wrapped accumulator.
     */
    using acc_type = A;

    /**
     * @brief Type of accumulated samples (simplemc::sample_type).
     */
    using sample_type = typename acc_type::sample_type;

    /**
     * @brief Underlying scalar type of accumulated samples (simplemc::double_or_complex).
     */
    using value_type = typename acc_type::value_type;

    /**
     * @brief Type for counting the number of accumulated samples.
     */
    using count_type = typename acc_type::count_type;

    /**
     * @brief Size type of the accumulator.
     */
    using size_type = typename acc_type::size_type;

public:
    /**
     * @brief Construct a multi-value accumulator for a given accumulator.
     *
     * @param acc Accumulator to be wrapped.
     */
    multivalue_acc(acc_type& acc) : acc_(acc) {}

    /**
     * @brief Subscript operator sets the index \f$ i \f$ and returns a reference to `this` object.
     *
     * @details The index is *sticky*: it persists until changed by another call to operator[](). For
     * scalar accumulators (size \f$ M = 1 \f$), the index should remain at 0.
     *
     * @param i Index \f$ i \f$.
     * @return Reference to `this` object.
     */
    multivalue_acc& operator[](size_type i) noexcept {
        idx_ = i;
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a single (scalar) value \f$ x \f$.
     *
     * @details For accumulators with size \f$ M > 1 \f$, the value is added to the element at the
     * current index \f$ i \f$ (see operator[]()).
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @note The value is buffered. Call commit() to finalize the sample.
     *
     * @tparam U Type of the value to be accumulated.
     * @param x Value \f$ x \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename U>
        requires std::convertible_to<U, value_type>
    multivalue_acc& operator<<(const U& x) {
        buffered_vals_.push_back(static_cast<value_type>(x));
        buffered_idxs_.push_back(idx_);
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a vector \f$ \mathbf{v} \f$.
     *
     * @details See also @ref simplemc-accs-accs-how.
     *
     * @note The components of the vector are buffered. Call commit() to finalize the sample.
     *
     * @tparam W Eigen vector/array/expression type.
     * @param v Vector/Array/Expression \f$ \mathbf{v} \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename W>
    multivalue_acc& operator<<(const W& v) {
        assert(v.size() == size());
        for (size_type j = 0; j < size(); ++j) {
            buffered_vals_.push_back(v(j));
            buffered_idxs_.push_back(j);
        }
        return *this;
    }

    /**
     * @brief Accumulate a range of values to consecutive elements in the accumulator.
     *
     * @details The values are added to consecutive elements in the accumulator starting with the
     * element at the given index \f$ i \f$. The size \f$ n \f$ of the range is assumed to satisfy \f$
     * n \leq M - i \f$.
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @note The values in range are buffered. Call commit() to finalize the sample.
     *
     * @tparam R Input range of values.
     * @param rg Range of values to be accumulated.
     * @param i Index \f$ i \f$ of the first element in the accumulator that a value will be added to.
     */
    template <ranges::input_range R>
    void accumulate(R&& rg, size_type i = 0) { // NOLINT (ranges need not be forwarded)
        for (auto val : rg) {
            buffered_vals_.push_back(val);
            buffered_idxs_.push_back(i);
            ++i;
        }
    }

    /**
     * @brief Accumulate a range of values to arbitrary elements with the given indices.
     *
     * @details Each value \f$ v_{i_j} \f$ of the given value range is accumulated at the
     * corresponding index \f$ i_j \f$ in the accumulator. Every index \f$ i_j \f$ should only appear
     * once in index range. Both ranges are assumed to be of the same size \f$ n \leq M \f$.
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @note The values in range are buffered. Call commit() to finalize the sample.
     *
     * @tparam R1 Input range of values.
     * @tparam R2 Input range of indices.
     * @param rg Range of values to be accumulated.
     * @param idxs Range of indices at which the values should be accumulated.
     */
    template <ranges::input_range R1, ranges::input_range R2>
    void accumulate(R1&& rg, R2&& idxs) { // NOLINT (ranges need not be forwarded)
        for (auto [val, idx] : ranges::views::zip(rg, idxs)) {
            buffered_vals_.push_back(val);
            buffered_idxs_.push_back(idx);
        }
    }

    /**
     * @brief Commit the buffered values as a single sample to the wrapped accumulator.
     *
     * @details All values accumulated since the last commit() call form one sample (the values at the
     * touched indices, with implicit zeros elsewhere). They are flushed into the wrapped accumulator
     * as a single sample by calling its `accumulate()` function with the buffered values and indices.
     *
     * The buffer is cleared.
     *
     * Committing an empty buffer contributes an all-zero sample.
     */
    void commit() {
        acc_.accumulate(buffered_vals_, buffered_idxs_);
        buffered_vals_.clear();
        buffered_idxs_.clear();
    }

    /**
     * @brief Get the size \f$ M \f$ of the wrapped accumulator.
     *
     * @return Number of elements.
     */
    [[nodiscard]] auto size() const noexcept { return acc_.size(); }

    /**
     * @brief Get the total number of accumulated samples \f$ N \f$.
     *
     * @return Number of accumulated samples.
     */
    [[nodiscard]] auto count() const noexcept { return acc_.count(); }

    /**
     * @brief Check if the accumulator is empty.
     *
     * @return True if the count() is zero, i.e. \f$ N = 0 \f$, false otherwise.
     */
    [[nodiscard]] bool empty() const noexcept { return count() == 0; }

    /**
     * @brief Get the wrapped accumulator object.
     *
     * @return Reference to the wrapped accumulator.
     */
    acc_type& accumulator() noexcept { return acc_; }

    /**
     * @brief Get the wrapped accumulator object.
     *
     * @return Const reference to the wrapped accumulator.
     */
    const acc_type& accumulator() const noexcept { return acc_; }

private:
    acc_type& acc_; // NOLINT (reference is wanted here)
    size_type idx_ = 0;
    std::vector<value_type> buffered_vals_;
    std::vector<size_type> buffered_idxs_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_MULTIVALUE_ACC_HPP
