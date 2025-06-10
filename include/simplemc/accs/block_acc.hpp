/**
 * @file
 * @brief Wrapper accumulator to block random samples.
 */

#ifndef SIMPLEMC_ACCS_BLOCK_ACC_HPP
#define SIMPLEMC_ACCS_BLOCK_ACC_HPP

#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-accs-wrappers
 * @{
 */

/**
 * @brief Wrapper for simplemc::var_acc and simplemc::covar_acc to block random samples.
 *
 * @details It groups the random samples into blocks of a given size \f$ B \f$ before accumulating
 * them into the wrapped accumulator object. This helps to decorrelate consecutive samples.
 *
 * A block is simply a simplemc::mean_acc object. When a new random sample is accumlated, it is added
 * to the mean accumulator. When the number of samples in the block is equal to \f$ B \f$, the mean of
 * the block is accumulated into the underlying variance/covariance accumulator.
 *
 * Functionality and usage is similar to the supported wrapped accumulators. Results can be obtained
 * from the wrapped accumulators directly by calling accumulator().
 *
 * @code{.cpp}
 * #include <fmt/ranges.h>
 * #include <simplemc/accs.hpp>
 *
 * #include <random>
 *
 * int main() {
 *     // normal distribution to be sampled: mu = 5, sigma = 1
 *     std::mt19937_64 rng;
 *     std::normal_distribution<double> normal_dist(5.0, 1.0);
 *
 *     // accumulate samples into a block accumulator with block size B = 10
 *     simplemc::block_acc<simplemc::var_acc_single<double>> blacc{10};
 *     for (int i = 0; i < 100000; ++i) {
 *         blacc << normal_dist(rng);
 *     }
 *
 *     // print the mean and variance of the accumulated data
 *     fmt::println("Mean: {}", blacc.accumulator().mean());
 *     fmt::println("Variance: {}", blacc.accumulator().variance_of_data());
 * }
 * @endcode
 *
 * Output:
 *
 * ```
 * Mean: 5.002072302074504
 * Variance: 0.10090101293580174
 * ```
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
     * @return simplemc::varalg tag of the wrapped accumulator.
     */
    static constexpr auto varalg() { return A::varalg(); }

    /**
     * @brief Mean accumulator type for accumulating block data.
     */
    using mean_acc_type = mean_acc<Eigen::Matrix<value_type, static_size, 1>, varalg()>;

public:
    /**
     * @brief Construct a block accumulator with a given block size \f$ B \f$ and number of elements
     * \f$ M \f$.
     *
     * @details The wrapped accumulator is constructed with \f$ M \f$ elements. It throws a 
     * simplemc::simplemc_exception if \f$ B = 0 \f$.
     *
     * @param b Block size \f$ B \f$.
     * @param m Number of elements \f$ M \f$.
     */
    explicit block_acc(count_type b, size_type m = 1) : acc_(m), block_(m), blsize_(b) {
        if (blsize_ == 0) {
            throw simplemc_exception("Block size == 0", "block_acc::block_acc");
        }
    }

    /**
     * @brief Construct a block accumulator from a given accumulator and with a given block size \f$ B
     * \f$.
     *
     * @details It throws a simplemc::simplemc_exception if \f$ B = 0 \f$.
     *
     * @param acc Accumulator to be wrapped.
     * @param b Block size \f$ B \f$.
     */
    explicit block_acc(const acc_type& acc, count_type b) : acc_(acc), block_(acc_.size()), blsize_(b) {
        if (blsize_ == 0) {
            throw simplemc_exception("Block size == 0", "block_acc::block_acc");
        }
    }

    /**
     * @brief Construct a block accumulator from a given accumulator, block data (mean accumulator)
     * and block size \f$ B \f$.
     *
     * @details The size \f$ M \f$ of the accumulator and block data should be the same and \f$ B > 0
     * \f$. Otherwise, it throws a simplemc::simplemc_exception.
     *
     * @param acc Accumulator to be wrapped.
     * @param block_data Mean accumulator for the block data.
     * @param b Block size \f$ B \f$.
     */
    explicit block_acc(const acc_type& acc, const mean_acc_type& block_data, count_type b) :
        acc_(acc),
        block_(block_data),
        blsize_(b) {
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
     * @brief Subscript operator sets the index \f$ i \f$ of the mean accumulator for the block data
     * and returns a reference to `this` object.
     *
     * @param i Index \f$ i \f$.
     * @return Reference to `this` object.
     */
    block_acc& operator[](size_type i) {
        block_[i];
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a single value \f$ x \f$.
     *
     * @details The value is first added to the current block using
     * simplemc::mean_acc::operator<<(const T&) and then check_and_add_block() is called.
     *
     * @tparam T Type of the value to be accumulated.
     * @param x Value \f$ x \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename T>
        requires std::convertible_to<T, value_type>
    block_acc& operator<<(const T& x) {
        block_ << x;
        check_and_add_block();
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a vector \f$ \mathbf{v} \f$.
     *
     * @details The vector is first added to the current block using
     * simplemc::mean_acc::operator<<(const W&) and then check_and_add_block() is called.
     *
     * @tparam W Eigen vector/array/expression type.
     * @param v Vector/Array/Expression \f$ \mathbf{v} \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename W>
    block_acc& operator<<(const W& v) {
        assert(v.size() == size());
        block_ << v;
        check_and_add_block();
        return *this;
    }

    /**
     * @brief Stream operator for incorporating the data from another block accumulator.
     *
     * @details The data is added by simply calling `operator<<` with the wrapped accumulator objects.
     *
     * @note Data in the current block is ignored and not incorporated.
     *
     * @param acc_other simplemc::block_acc to be incorporated.
     * @return Reference to `this` object.
     */
    block_acc& operator<<(const block_acc& acc_other) {
        assert(block_size() == acc_other.block_size());
        acc_ << acc_other.acc_;
        return *this;
    }

    /**
     * @brief Accumulate a range of values to consecutive elements in the accumulator.
     *
     * @details The values are first added to the current block using
     * simplemc::mean_acc::accumulate() and then check_and_add_block() is called.
     *
     * @tparam R Input range of values.
     * @param rg Range of values to be accumulated.
     * @param i Index \f$ i \f$ of the first element in the accumulator that a value will be added to.
     */
    template <ranges::input_range R>
    void accumulate(R&& rg, size_type i = 0) {
        block_.accumulate(std::forward<R>(rg), i);
        check_and_add_block();
    }

    /**
     * @brief Accumulate a range of values to arbitrary elements with the given indices.
     *
     * @details The values are first added to the current block using
     * simplemc::mean_acc::accumulate(R1 &&, R2 &&) and then check_and_add_block() is called.
     *
     * @tparam R1 Input range of values.
     * @tparam R2 Input range of indices.
     * @param rg Range of values to be accumulated.
     * @param idxs Range of indices at which the values should be accumulated.
     */
    template <ranges::input_range R1, ranges::input_range R2>
    void accumulate(R1&& rg, R2&& idxs) {
        block_.accumulate(std::forward<R1>(rg), std::forward<R2>(idxs));
        check_and_add_block();
    }

    /**
     * @brief Create a multi-value accumulator (only for variance accumulators).
     *
     * @details The user is responsible for calling simplemc::multivalue_acc::increment_count as well 
     * as check_and_add_block() after all values have been added, otherwise the number of samples will 
     * not be correct. For example,
     * @code{.cpp}
     * auto mva = acc.make_mva();
     * mva[i_1] << v_i_1;
     * // ...
     * mva[i_n] << v_i_n;
     * mva.increment_count();
     * acc.check_and_add_block();
     * @endcode
     *
     * @return Multi-value accumulator wrapping `this` object.
     */
    [[nodiscard]] auto make_mva() { return block_.make_mva(); }

    /**
     * @brief Get the size \f$ M \f$ of the accumulator.
     *
     * @return Number of elements.
     */
    [[nodiscard]] auto size() const { return acc_.size(); }

    /**
     * @brief Get the block size \f$ B \f$ of the accumulator.
     *
     * @return Block size.
     */
    [[nodiscard]] auto block_size() const { return blsize_; }

    /**
     * @brief Get the effective number of accumulated samples \f$ N_{\mathrm{eff}} = \lfloor N / B 
     * \rfloor \f$, i.e. the number of accumulated blocks.
     *
     * @return Number of accumulated blocks.
     */
    [[nodiscard]] auto count() const { return acc_.count(); }

    /**
     * @brief Get the total number of accumulated samples \f$ N \f$.
     *
     * @return Number of accumulated blocks times the block size plus what is left in the current
     * block.
     */
    [[nodiscard]] auto total_count() const { return count() * blsize_ + block_.count(); }

    /**
     * @brief Get the wrapped accumulator.
     *
     * @return Underlying variance/covariance accumulator object.
     */
    [[nodiscard]] const auto& accumulator() const { return acc_; }

    /**
     * @brief Get the simplemc::mean_acc object used to accumulate block data.
     *
     * @return Current block data.
     */
    [[nodiscard]] const auto& block() const { return block_; }

    /**
     * @brief Check if the current block is full.
     *
     * @return True if the number of accumulated samples in the block is \f$ \geq B \f$, false
     * otherwise.
     */
    [[nodiscard]] auto is_full() const { return block_.count() >= blsize_; }

    /**
     * @brief Check if the current block is_full() and add it to the effective samples if so.
     */
    void check_and_add_block() {
        if (is_full()) {
            acc_ << block_.mean();
            block_.reset();
        }
    }

    /**
     * @brief Collect block accumulators from different MPI processes.
     *
     * @details It simply calls `mpi_collect` for the wrapped accumulator. Accumulated data in current
     * blocks is ignored.
     *
     * It throws a simplemc::simplemc_exception, if the block size \f$ B \f$ or if the size \f$ M \f$
     * of the wrapped accumulator is not equal on all processes.
     *
     * @param comm simplemc::mpi::communicator object.
     * @param acc Block accumulator.
     * @return Block accumulator with the reduced data from all processes.
     */
    friend block_acc mpi_collect(const mpi::communicator& comm, const block_acc& acc) {
        if (!all_equal(comm, acc.block_size())) {
            throw simplemc_exception("Block size is not equal on all processes", "mpi_collect");
        }
        block_acc res(acc.blsize_, acc.size());
        res.acc_ = mpi_collect(comm, acc.acc_);
        return res;
    }

private:
    acc_type acc_;
    mean_acc_type block_;
    count_type blsize_ { 1 };
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_BLOCK_ACC_HPP
