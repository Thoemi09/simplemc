/**
 * @file
 * @brief Wrapper for accumulators to add multiple values to them.
 */

#ifndef SIMPLEMC_ACCS_MULTIVALUE_ACC_HPP
#define SIMPLEMC_ACCS_MULTIVALUE_ACC_HPP

#include <range/v3/range/concepts.hpp>
#include <range/v3/view/zip.hpp>

namespace simplemc {

/**
 * @ingroup simplemc-accs-wrappers
 * @brief Multi value accumulator for various accumulators.
 *
 * @details It holds a reference to a accumulator and can be used to add multiple data points to the
 * accumulator without increasing the count automatically. This has to be done manually by the user
 * by calling the multivalue_acc::increment_count function.
 *
 * The intended use is for simplemc::mean_acc and simplemc::var_acc. Both of them provide a factory
 * function, `make_mva()`, that wraps the current object and returns a multi value accumlator.
 *
 * The user should not need to create a multivalue_acc on their own.
 *
 * @code{.cpp}
 * simplemc::mean_acc_dynamic<double> acc(size);
 * // ...
 * for (int i = 0; i < num_samples; ++i) {
 *     // ...
 *     auto mva = acc.make_mva();
 *     mva[idx_1] << val_1;
 *     // ...
 *     mva[idx_n] << val_n;
 *     mva.increment_count();
 *     // ...
 * }
 * // ...
 * @endcode
 *
 * @tparam A Accumulator type.
 */
template <typename A>
class multivalue_acc {
public:
    /**
     * @brief Type of the accumulator.
     */
    using acc_type = A;

    /**
     * @brief Type of accumulated values.
     */
    using value_type = typename acc_type::value_type;

    /**
     * @brief Type for counting the number of accumulated values.
     */
    using count_type = typename acc_type::count_type;

    /**
     * @brief Size type of the accumulator.
     */
    using size_type = typename acc_type::size_type;

    /**
     * @brief Construct a multi value accumulator for a given accumulator.
     *
     * @param acc Accumulator to be wrapped.
     */
    multivalue_acc(acc_type& acc) : acc_(acc) {}

    /**
     * @brief Subscript operator sets the index of the wrapped accumulator and returns a reference to
     * this object.
     *
     * @param idx Index.
     * @return Reference to this object.
     */
    multivalue_acc& operator[](size_type idx) {
        acc_.idx_ = idx;
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a single value.
     *
     * @details For accumulators with size > 1, the value is added to the element at the current
     * index.
     *
     * @param val Value to be accumulated.
     * @return Reference to this object.
     */
    multivalue_acc& operator<<(value_type val) {
        acc_.add_value(val, acc_.idx_, acc_.count_ + 1);
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a vector.
     *
     * @details It simply calls the corresponding streaming operator of the wrapped accumulator and
     * then decreases its count by one.
     *
     * @tparam W Eigen vector/array/expression type/.
     * @param vec Vector/Array/Expression to be accumulated.
     * @return Reference to this object.
     */
    template <typename W>
    multivalue_acc& operator<<(const W& vec) {
        acc_ << vec;
        --acc_.count_;
        return *this;
    }

    /**
     * @brief Accumulate a range of values to consecutive elements in the accumulator.
     *
     * @details The values are added to consecutive elements in the accumulator starting with the
     * element at the given index. The size of the range is assumed to be <= size() - `idx`.
     *
     * @tparam R Input range of values.
     * @param rg Range of values to be accumulated.
     * @param idx Starting index for the accumulator.
     */
    template <ranges::input_range R>
    void accumulate(R&& rg, size_type idx = 0) { // NOLINT (ranges need not be forwarded)
        for (auto val : rg) {
            acc_.add_value(val, idx, acc_.count_ + 1);
            ++idx;
        }
    }

    /**
     * @brief Accumulate a range of values to arbitrary elements but with different indices.
     *
     * @details Each value of the given value range is accumulated at the corresponding index of the
     * given index range. Every index should only appear once.
     *
     * @tparam R1 Input range of values.
     * @tparam R2 Input range of indices.
     * @param rg Range of values to be accumulated.
     * @param idxs Range of indices.
     */
    template <ranges::input_range R1, ranges::input_range R2>
    void accumulate(R1&& rg, R2&& idxs) { // NOLINT (ranges need not be forwarded)
        for (auto [val, idx] : ranges::views::zip(rg, idxs)) {
            acc_.add_value(val, idx, acc_.count_ + 1);
        }
    }

    /**
     * @brief Increment the count of the wrapped accumulator by a given increment.
     *
     * @param inc Increment.
     */
    void increment_count(count_type inc = 1) { acc_.count_ += inc; }

private:
    acc_type& acc_; // NOLINT (reference is wanted here)
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_MULTIVALUE_ACC_HPP
