/**
 * @file block_acc.hpp
 * @brief Wrapper for simplemc::var_acc and simplemc::covar_acc to block data.
 */

#ifndef SIMPLEMC_ACCS_BLOCK_ACC_HPP
#define SIMPLEMC_ACCS_BLOCK_ACC_HPP

#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <utility>

namespace simplemc {

/**
 * @brief Wrapper class for the simplemc::var_acc and simplemc::covar_acc accumulators to put data
 * into blocks of a given size to reduce correlations.
 *
 * @details Functionality and usage is similar to the supported wrapped accumulators.
 *
 * Multi value accumulation can only be used for wrapped simplemc::var_acc. Remember to manually
 * call the `increment_count` method of the multi value accumulator as well as the `check_and_add_block`
 * method of the block accumulator after using them, otherwise the number of effective samples will not
 * be correct:
 * @code{.cpp}
 * auto mva = acc.make_mva();
 * mva[idx1] << val1;
 * mva[idx2] << val2;
 * mva.increment_count();
 * acc.check_and_add_block();
 * @endcode
 *
 * Results can be obtained from the wrapped accumulators directly. For example:
 * @code{.cpp}
 * auto mean = acc.accumulator().mean();
 * auto covar = acc.accumulator().covariance();
 * @endcode
 *
 * @tparam A Accumulator type.
 */
template <typename A>
class block_acc {
public:
    /**
     * @brief Accumulator type.
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
     * @brief Get the algorithm.
     */
    static constexpr auto varalg() { return A::varalg(); }

    /**
     * @brief Mean accumulator type for accumulating block data.
     */
    using mean_acc_type = mean_acc<value_type, varalg()>;

public:
    /**
     * @brief Construct a block accumulator with a given number of elements, a constant shift and a block size.
     *
     * @param size Number of elements.
     * @param shift A single constant shift applied to the accumulated values.
     * @param blsize Block size.
     */
    explicit block_acc(size_type size = 1, value_type shift = 0.0, count_type blsize = 1) :
        acc_(size, shift),
        block_(size, shift),
        blsize_(blsize) {
        if (blsize_ == 0) {
            throw simplemc_exception("Block size == 0", "block_acc::block_acc");
        }
    }

    /**
     * @brief Construct a block accumulator with a given constant shift and block size.
     *
     * @details The size of the accumulator will be the same as the size of the shift vector.
     *
     * @param shift Constant shift applied to the accumulated values.
     * @param blsize Block size.
     */
    explicit block_acc(const Eigen::VectorX<value_type>& shift, count_type blsize = 1) :
        acc_(shift),
        block_(shift),
        blsize_(blsize) {
        if (blsize_ == 0) {
            throw simplemc_exception("Block size == 0", "block_acc::block_acc");
        }
    }

    /**
     * @brief Construct a block accumulator from a given accumulator and block size.
     *
     * @param acc Accumulator.
     * @param blsize Block size.
     */
    explicit block_acc(acc_type acc, count_type blsize = 1) :
        acc_(std::move(acc)),
        block_(acc_.size(), acc_.shift()),
        blsize_(blsize) {
        if (blsize_ == 0) {
            throw simplemc_exception("Block size == 0", "block_acc::block_acc");
        }
    }

    /**
     * @brief Construct a block accumulator from a given accumulator, block data (mean accumulator) and block size.
     *
     * @param acc Accumulator.
     * @param bldata Block data (mean accumulator).
     * @param blsize Block size.
     */
    explicit block_acc(acc_type acc, mean_acc_type bldata, count_type blsize = 1) :
        acc_(std::move(acc)),
        block_(std::move(bldata)),
        blsize_(blsize) {
        if (acc_.size() != block_.size()) {
            throw simplemc_exception("Size mismatch between accumulator and block data", "block_acc::block_acc");
        }
        if (acc_.shift() != block_.shift()) {
            throw simplemc_exception("Shift mismatch between accumulator and block data", "block_acc::block_acc");
        }
        if (blsize_ == 0) {
            throw simplemc_exception("Block size == 0", "block_acc::block_acc");
        }
    }

    /**
     * @brief Reset the accumulator to its initial state, i.e. with no accumulated values.
     */
    void reset() {
        acc_.reset();
        block_.reset();
    }

    /**
     * @brief Subscript operator sets the index of the mean accumulator for the block data.
     *
     * @param idx Index.
     * @return Reference to this object.
     */
    block_acc& operator[](size_type idx) {
        block_[idx];
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a single value.
     *
     * @param val Value to be accumulated.
     * @return Reference to this object.
     */
    block_acc& operator<<(value_type val) {
        block_ << val;
        check_and_add_block();
        return *this;
    }

    /**
     * @brief Stream operator for incorporating the data from another block accumulator.
     *
     * @details Ignores the data in the current blocks.
     *
     * @param acc Block accumulator to be incorporated.
     * @return Reference to this object.
     */
    block_acc& operator<<(const block_acc& acc) {
        assert(block_size() == acc.block_size());
        acc_ << acc.acc_;
        return *this;
    }

    /**
     * @brief Accumulate a range of values.
     *
     * @details The size of the range is assumed to be <= size() - idx.
     *
     * @tparam R Input range of values.
     * @param rg Range of values to be accumulated.
     * @param idx Starting index for the accumulator.
     */
    template <ranges::input_range R>
    void accumulate(R&& rg, size_type idx = 0) {
        block_.accumulate(std::forward<R>(rg), idx);
        check_and_add_block();
    }

    /**
     * @brief Accumulate a range of values.
     *
     * @details The size of the range is assumed to be <= size() - idx and every index should
     * only appear at most once.
     *
     * @tparam R1 Input range of values.
     * @tparam R2 Input range of indices.
     * @param rg Range of values to be accumulated.
     * @param idxs Range of indices.
     */
    template <ranges::input_range R1, ranges::input_range R2>
    void accumulate(R1&& rg, R2&& idxs) {
        block_.accumulate(std::forward<R1>(rg), std::forward<R2>(idxs));
        check_and_add_block();
    }

    /**
     * @brief Create a multi value accumulator.
     *
     * @return Multi value accumulator.
     */
    [[nodiscard]] auto make_mva() { return block_.make_mva(); }

    /**
     * @brief Get the size of the accumulator.
     *
     * @return Number of elements.
     */
    [[nodiscard]] auto size() const { return acc_.size(); }

    /**
     * @brief Get the block size of the accumulator.
     *
     * @return Block size.
     */
    [[nodiscard]] auto block_size() const { return blsize_; }

    /**
     * @brief Get the effective number of accumulated values, i.e. number of accumulated blocks.
     *
     * @return Number of accumulated blocks.
     */
    [[nodiscard]] auto count() const { return acc_.count(); }

    /**
     * @brief Get the constant shift.
     *
     * @return Constant shift vector applied to the accumulated values.
     */
    [[nodiscard]] const auto& shift() const { return acc_.shift(); }

    /**
     * @brief Get the wrapped accumulator.
     *
     * @return Wrapped accumulator.
     */
    [[nodiscard]] const auto& accumulator() const { return acc_; }

    /**
     * @brief Get the mean accumulator used to accumulate block data.
     *
     * @return Mean accumulator.
     */
    [[nodiscard]] const auto& block() const { return block_; }

    /**
     * @brief Check if the current block is full?
     *
     * @return True if the current block is full, false otherwise.
     */
    [[nodiscard]] auto is_full() const { return block_.count() >= blsize_; }

    /**
     * @brief Check if the current block is full and add it to the effective samples if so.
     */
    void check_and_add_block() {
        if (is_full()) {
            acc_.accumulate(block_.mean());
            block_.reset();
        }
    }

private:
    acc_type acc_;
    mean_acc_type block_;
    count_type blsize_ { 1 };
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_BLOCK_ACC_HPP
