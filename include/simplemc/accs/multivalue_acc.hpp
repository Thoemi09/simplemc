/**
 * @file
 * @brief Wrapper accumulator to add multiple values at once.
 */

#ifndef SIMPLEMC_ACCS_MULTIVALUE_ACC_HPP
#define SIMPLEMC_ACCS_MULTIVALUE_ACC_HPP

#include <simplemc/utils/ranges.hpp>

namespace simplemc {

/**
 * @ingroup simplemc-accs-wrappers
 * @brief Wrapper for simplemc::mean_acc, simplemc::var_acc, simplemc::block_acc and
 * simplemc::batch_acc to add multiple values at once.
 *
 * @details It holds a reference to an accumulator and can be used to add multiple data points to the
 * accumulator without increasing the count automatically. This has to be done manually by the user
 * by calling the multivalue_acc::increment_count function. In addition, if the wrapped accumulator is
 * a simplemc::block_acc or simplemc::batch_acc, the user has to call
 * simplemc::block_acc::check_and_add_block or simplemc::batch_acc::check_and_advance as well.
 *
 * The intended use is for mean, variance, blocked variance and batch accumulators. All of them
 * provide a factory function, `make_mva()`, that wraps the current accumulator object and returns a
 * simplemc::multivalue_acc object:
 * - simplemc::mean_acc::make_mva(),
 * - simplemc::var_acc<X, A>::make_mva() and simplemc::var_acc<Z, A>::make_mva(),
 * - simplemc::block_acc::make_mva(),
 * - simplemc::batch_acc::make_mva().
 *
 * @code{.cpp}
 * #include <fmt/ranges.h>
 * #include <simplemc/accs.hpp>
 *
 * #include <vector>
 *
 * int main() {
 *     // data to be sampled
 *     std::vector<double> data = { 1.0, 2.0, 3.0, 4.0, 5.0 };
 *
 *     // accumulate samples into the first and third elements of a mean accumulator of size 3
 *     simplemc::mean_acc_dynamic<double> acc(3);
 *     for (auto& val : data) {
 *         auto mva = acc.make_mva();
 *         mva[0] << val;
 *         mva[2] << val;
 *         mva.increment_count();
 *     }
 *
 *     // print the mean of the accumulated data
 *     fmt::print("Mean: {}\n", acc.mean());
 * }
 * @endcode
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

public:
    /**
     * @brief Construct a multi-value accumulator for a given accumulator.
     *
     * @param acc Accumulator to be wrapped.
     */
    multivalue_acc(acc_type& acc) : acc_(acc) {}

    /**
     * @brief Subscript operator sets the index \f$ i \f$ of the wrapped accumulator and returns a
     * reference to `this` object.
     *
     * @param idx Index \f$ i \f$.
     * @return Reference to `this` object.
     */
    multivalue_acc& operator[](size_type idx) {
        acc_.idx_ = idx;
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a single value \f$ x \f$.
     *
     * @details For accumulators with size \f$ M > 1 \f$, the value is added to the element at the
     * current index \f$ i \f$ (see operator[]()).
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @note The sample count is not incremented. Manually use increment_count().
     *
     * @tparam T Type of the value to be accumulated.
     * @param x Value \f$ x \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename T>
        requires std::convertible_to<T, value_type>
    multivalue_acc& operator<<(const T& x) {
        acc_.add_value(x, acc_.idx_, acc_.count_ + 1);
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a vector \f$ \mathbf{v} \f$.
     *
     * @details See also @ref simplemc-accs-accs-how.
     *
     * @note The sample count is not incremented. Manually use increment_count().
     *
     * @tparam W Eigen vector/array/expression type.
     * @param v Vector/Array/Expression \f$ \mathbf{v} \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename W>
    multivalue_acc& operator<<(const W& v) {
        acc_ << v;
        --acc_.count_;
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
     * @note The sample count is not incremented. Manually use increment_count().
     *
     * @tparam R Input range of values.
     * @param rg Range of values to be accumulated.
     * @param i Index \f$ i \f$ of the first element in the accumulator that a value will be added to.
     */
    template <ranges::input_range R>
    void accumulate(R&& rg, size_type i = 0) { // NOLINT (ranges need not be forwarded)
        for (auto val : rg) {
            acc_.add_value(val, i, acc_.count_ + 1);
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
     * @note The sample count is not incremented. Manually use increment_count().
     *
     * @tparam R1 Input range of values.
     * @tparam R2 Input range of indices.
     * @param rg Range of values to be accumulated.
     * @param idxs Range of indices at which the values should be accumulated.
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
