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
 * @ingroup simplemc-accs
 * @brief Multi value accumulator for various accumulators.
 *
 * @details It holds a reference to a accumulator and can be used to add multiple data points
 * to the accumulator without increasing the count automatically. This has to be done manually!
 * The intended use is for simplemc::mean_acc and simplemc::var_acc.
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
     * @brief Construct a multi value accumulator for a given accumulator and a given index.
     *
     * @param acc Accumulator to be wrapped.
     */
    multivalue_acc(acc_type& acc) : acc_(acc) {}

    /**
     * @brief Set index of the wrapped accumulator and return this object.
     *
     * @param idx Index.
     * @return Reference to this object.
     */
    multivalue_acc& operator[](size_type idx) {
        acc_.idx_ = idx;
        return *this;
    }

    /**
     * @brief Accumulate a single value.
     *
     * @details It simply calls the `add_value` method of the wrapped accumulator to add a single
     * value without increasing the count.
     *
     * @param val Value to be accumulated.
     * @return Reference to this object.
     */
    multivalue_acc& operator<<(value_type val) {
        acc_.add_value(val, acc_.idx_, acc_.count_ + 1);
        return *this;
    }

    /**
     * @brief Accumulate a range of values.
     *
     * @details The size of the range is assumed to be <= size() - idx. It simply calls the `add_value`
     * method of the wrapped accumulator for every value in the range without increasing the count.
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
     * @brief Accumulate a range of values.
     *
     * @details The size of the range is assumed to be <= size() - idx. It simply calls the `add_value`
     * method of the wrapped accumulator for every value-index pair in the ranges without increasing the
     * count.
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
     * @brief Increment the count of the wrapped accumulator.
     *
     * @param inc Increment.
     */
    void increment_count(count_type inc = 1) { acc_.count_ += inc; }

private:
    acc_type& acc_; // NOLINT (reference is wanted here)
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_MULTIVALUE_ACC_HPP
