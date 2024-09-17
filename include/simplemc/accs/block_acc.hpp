/**
 * @file
 * @brief Wrapper for simplemc::var_acc and simplemc::covar_acc to block random samples.
 */

#ifndef SIMPLEMC_ACCS_BLOCK_ACC_HPP
#define SIMPLEMC_ACCS_BLOCK_ACC_HPP

#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <range/v3/range/concepts.hpp>

#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-accs-wrappers
 * @{
 */

/**
 * @brief Wrapper for simplemc::var_acc and simplemc::covar_acc to block random samples.
 *
 * @details It groups the random samples into blocks of a given size before accumulating them into the
 * wrapped accumulator object. This helps to decorrelate consecutive samples.
 *
 * A block is simply a simplemc::mean_acc object. When a new random sample is accumlated, it is added
 * to the mean accumulator. When the number of samples in the block is equal the specified block size,
 * the mean of the block is accumulated into the wrapped accumulator.
 *
 * Functionality and usage is similar to the supported wrapped accumulators.
 *
 * Multi value accumulation can only be used with simplemc::var_acc accumulators:
 * @code{.cpp}
 * auto mva = blacc.make_mva();
 * mva[idx1] << val1;
 * mva[idx2] << val2;
 * mva.increment_count();
 * blacc.check_and_add_block();
 * @endcode
 * @note The user is responsible for calling the simplemc::multivalue_acc::increment_count method as
 * well as the check_and_add_block() method, otherwise the number of effective samples will not be
 * correct.
 *
 * Results can be obtained from the wrapped accumulators directly by calling accumulator(). For
 * example:
 * @code{.cpp}
 * auto mean = blacc.accumulator().mean();
 * auto covar = blacc.accumulator().covariance();
 * @endcode
 *
 * @tparam A Accumulator type.
 */
template <typename A>
class block_acc {
public:
    /**
     * @brief Type of the wrapped accumulator.
     */
    using acc_type = A;

    /**
     * @brief Static size of the accumulator.
     */
    static constexpr int static_size = acc_type::static_size;

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
     * @brief Get the algorithm used to accumulate the data.
     *
     * @return The simplemc::varalg tag.
     */
    static constexpr auto varalg() { return A::varalg(); }

    /**
     * @brief Mean accumulator type for accumulating block data.
     */
    using mean_acc_type = mean_acc<Eigen::Matrix<value_type, static_size, 1>, varalg()>;

public:
    /**
     * @brief Construct a block accumulator with a given number of elements and block size.
     *
     * @details It throws a simplemc::simplemc_exception if the block size is zero.
     *
     * @param blsize Block size.
     * @param num Number of elements.
     */
    explicit block_acc(count_type blsize, size_type num = 1) : acc_(num), block_(num), blsize_(blsize) {
        if (blsize_ == 0) {
            throw simplemc_exception("Block size == 0", "block_acc::block_acc");
        }
    }

    /**
     * @brief Construct a block accumulator from a given accumulator and with a given block size.
     *
     * @details It throws a simplemc::simplemc_exception if the block size is zero.
     *
     * @param blsize Block size.
     * @param acc Accumulator.
     */
    explicit block_acc(const acc_type& acc, count_type blsize) : acc_(acc), block_(acc_.size()), blsize_(blsize) {
        if (blsize_ == 0) {
            throw simplemc_exception("Block size == 0", "block_acc::block_acc");
        }
    }

    /**
     * @brief Construct a block accumulator from a given accumulator, block data (mean accumulator)
     * and block size.
     *
     * @details The size of the accumulator and block data should be the same and the blocks size
     * should be > 0. Otherwise, it throws a simplemc::simplemc_exception.
     *
     * @param acc Accumulator.
     * @param bldata Block data (mean accumulator).
     * @param blsize Block size.
     */
    explicit block_acc(const acc_type& acc, const mean_acc_type& bldata, count_type blsize) :
        acc_(acc),
        block_(bldata),
        blsize_(blsize) {
        if (acc_.size() != block_.size()) {
            throw simplemc_exception("Size mismatch between accumulator and block data", "block_acc::block_acc");
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
     * @details See the corresponding method of the wrapped accumulator for more details.
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
     * @brief Stream operator for accumulating a vector.
     *
     * @details See the corresponding method of the wrapped accumulator for more details.
     *
     * @tparam W Eigen vector/array/expression type.
     * @param vec Vector/Array/Expression to be accumulated.
     * @return Reference to this object.
     */
    template <typename W>
    block_acc& operator<<(const W& vec) {
        assert(vec.size() == size());
        block_ << vec;
        check_and_add_block();
        return *this;
    }

    /**
     * @brief Stream operator for incorporating the data from another block accumulator.
     *
     * @details Ignores the data in the current blocks.
     *
     * See the corresponding method of the wrapped accumulator for more details.
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
     * @brief Accumulate a range of values to consecutive elements in the accumulator.
     *
     * @details See the corresponding method of the wrapped accumulator for more details.
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
     * @brief Accumulate a range of values to arbitrary elements but with different indices.
     *
     * @details For wrapped covariance accumulators, the indices should be sorted in ascending order.
     *
     * See the corresponding method of the wrapped accumulator for more details.
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
     * @brief Create a multi-value accumulator.
     *
     * @details See simplemc::multivalue_acc.
     *
     * @return Multi-value accumulator.
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
     * @brief Get the total number of accumulated values.
     *
     * @return Number of accumulated blocks times the block size plus what is left in the current
     * block.
     */
    [[nodiscard]] auto total_count() const { return count() * blsize_ + block_.count(); }

    /**
     * @brief Get the wrapped accumulator.
     *
     * @return Wrapped accumulator.
     */
    [[nodiscard]] const auto& accumulator() const { return acc_; }

    /**
     * @brief Get the mean accumulator used to accumulate block data.
     *
     * @return simplemc::mean_acc object.
     */
    [[nodiscard]] const auto& block() const { return block_; }

    /**
     * @brief Check if the current block is full.
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

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_BLOCK_ACC_HPP
