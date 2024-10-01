/**
 * @file
 * @brief Accumulator for grouping samples of a random vector into a fixed number of batches.
 */

#ifndef SIMPLEMC_ACCS_BATCH_ACC_HPP
#define SIMPLEMC_ACCS_BATCH_ACC_HPP

#include <simplemc/accs/covar_acc.hpp>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/accs/multivalue_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/accs/var_acc.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/concepts.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <vector>

namespace simplemc {

/**
 * @addtogroup simplemc-accs-accs
 * @{
 */

/**
 * @brief Accumulator for grouping samples of a random vector into a fixed number of batches.
 *
 * @details Random samples are accumulated into a fixed number of batches. The batches correspond to
 * effective and uncorrelated (if their count is large enough) random samples. That means that once
 * the accumlation is done, they can be used for further statistical analysis, i.e. calculating the
 * sample mean and its (blocked) variance, estimate the autocorrelation time and so on. Furthermore,
 * the batches can form the basis for the Jackknife tools provided by **simplemc**.
 *
 * The order of the random samples in the batches is preserved. To do this efficiently and in a simple
 * way, we have two vectors of `N` batches:
 * - The first vector stores currently full batches. They all contain the same number of samples.
 * - The second vector is used for actually accumulating new the data into batches. Once, all the
 * batches contain the same number of samples as the batches in the first vector, they are merged
 * together into the first vector and batches in the second vector are reset.
 *
 * @note If the number of batches in the two vectors is \f$ N \f$, respectively, then the resulting
 * number of full batches ready to be used for further analysis will be somewhere between \f$ [N, 2N)
 * \f$ depending on the number of random samples accumulated.
 *
 * A batch accumulator is therefore similar to a simplemc::block_acc. The main difference is that
 * instead of specifying the block size, one specifies the number of blocks/batches. An advantage of
 * the batch accumulator is that one has access to the individual blocks/batches, which is not the
 * case for the block accumulator.
 *
 * Multi-value accumulation is possible but has the same requirements as for the simplemc::block_acc:
 * @code{.cpp}
 * auto mva = bacc.make_mva();
 * mva[idx1] << val1;
 * mva[idx2] << val2;
 * mva.increment_count();
 * blacc.check_and_advance();
 * @endcode
 * @note The user is responsible for calling the simplemc::multivalue_acc::increment_count method as
 * well as the check_and_advance() method, otherwise the number of samples will not be correct.
 *
 * Under the hood, a single batch is just a mean accumulator.
 *
 * The accumulator takes two template parameters:
 * - the type of the random samples (a simplemc::eigen_vector type) and
 * - the algorithm (simplemc::varalg) that should be used to accumulate the data.
 * Both of them determine the type of simplemc::mean_acc accumulators to be used for storing the batch
 * data.
 *
 * @tparam V simplemc::eigen_vector type.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <eigen_vector V, varalg A = varalg::welford>
class batch_acc {
public:
    /**
     * @brief Type of accumulated values.
     */
    using value_type = typename V::value_type;

    /**
     * @brief Static size of the accumulator.
     */
    static constexpr int static_size = V::RowsAtCompileTime;

    /**
     * @brief Is the accumulator dynamically sized?
     */
    static constexpr bool is_dynamic = (V::RowsAtCompileTime == Eigen::Dynamic);

    /**
     * @brief Does the accumulator return scalars or vectors/matrices?
     */
    static constexpr bool returns_scalar = (!is_dynamic && static_size == 1);

    /**
     * @brief Type for counting the number of accumulated values.
     */
    using count_type = std::uint64_t;

    /**
     * @brief Size type of the accumulator.
     */
    using size_type = long;

    /**
     * @brief Vector type.
     */
    using vec_type = V;

    /**
     * @brief Get the algorithm used to accumulate the data.
     *
     * @return The simplemc::varalg tag.
     */
    static constexpr auto varalg() { return A; }

    /**
     * @brief Mean accumulator type for accumulating batch data.
     */
    using mean_acc_type = mean_acc<vec_type, varalg()>;

    /**
     * @brief Multi-value accumulator type.
     */
    using mva_type = multivalue_acc<mean_acc_type>;

private:
    // Check if the first merge has not happened yet.
    [[nodiscard]] auto is_first_merge() const { return batch_count() < bcount_; }

    // Check if the current accumulating batch is full.
    [[nodiscard]] auto is_batch_full() const { return acc_batches_[bidx_].count() >= bcount_; }

    // Merge all the batches together while preserving the order of the samples.
    void merge_batches() {
        // the first merge is special
        if (is_first_merge()) {
            for (std::size_t i = 0; i < full_batches_.size(); ++i) {
                full_batches_[i] << acc_batches_[i];
                acc_batches_[i].reset();
            }
            bidx_ = 0;
            return;
        }

        // merge the batches of the first vector
        std::size_t midx = 0;
        for (std::size_t i = 0; i < full_batches_.size(); i += 2) {
            full_batches_[midx++] = (full_batches_[i] << full_batches_[i + 1]);
        }

        // merge the batches of the second vector
        for (std::size_t i = 0; i < acc_batches_.size(); i += 2) {
            full_batches_[midx++] = (acc_batches_[i] << acc_batches_[i + 1]);
            acc_batches_[i].reset();
            acc_batches_[i + 1].reset();
        }

        // double the batch count and reset the batch index
        bcount_ *= 2;
        bidx_ = 0;
    }

public:
    /**
     * @brief Construct a batch accumulator with a given number of elements and batches.
     *
     * @details It throws a simplemc::simplemc_exception if the given number of batches is < 2 or odd.
     *
     * For dynamically sized accumulators, it throws a simplemc::simplemc_exception if the given size
     * is <= 0.
     *
     * For static sized accumulators, the `num` parameter is ignored.
     *
     * @param num Number of elements.
     * @param num_batches Number of batches.
     */
    explicit batch_acc(size_type num = 1, std::size_t num_batches = 64) :
        full_batches_(num_batches, mean_acc_type { num }),
        acc_batches_(num_batches, mean_acc_type { num }) {
        check_batches();
    }

    /**
     * @brief Construct a batch accumulator from the given data storages.
     *
     * @details It throws a simplemc::simplemc_exception if the given data storages are inconsistent.
     * See simplemc::batch_acc::check_batches for more details.
     *
     * @param full_batches Vector of mean accumulators for storing full batches.
     * @param acc_batches Vector of mean accumulators for accumulating new batches.
     */
    batch_acc(std::vector<mean_acc_type> full_batches, std::vector<mean_acc_type> acc_batches) :
        full_batches_(std::move(full_batches)),
        acc_batches_(std::move(acc_batches)),
        bcount_(full_batches_[0].count == 0 ? 1 : full_batches_[0].count()) {
        auto it = std::upper_bound(acc_batches_.begin(), acc_batches_.end(), bcount_,
            [](const auto& acc, auto val) { return acc.count() < val; });
        bidx_ = static_cast<std::size_t>(std::distance(acc_batches_.begin(), it));
        check_batches();
    }

    /**
     * @brief Reset the accumulator to its initial state, i.e. with no accumulated values.
     */
    void reset() {
        for (auto& acc : full_batches_) {
            acc.reset();
        }
        for (auto& acc : acc_batches_) {
            acc.reset();
        }
        bcount_ = 1;
        bidx_ = 0;
    }

    /**
     * @brief Subscript operator sets the index of the current accumulating batch and returns a
     * reference to this object.
     *
     * @param idx Index.
     * @return Reference to this object.
     */
    batch_acc& operator[](size_type idx) {
        acc_batches_[bidx_][idx];
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a single value.
     *
     * @details It simply calls simplemc::mean_acc::operator<<(value_type) of the mean accumulator of
     * the current accumulating batch.
     *
     * @param val Value to be accumulated.
     * @return Reference to this object.
     */
    batch_acc& operator<<(value_type val) {
        acc_batches_[bidx_] << val;
        check_and_advance();
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a vector.
     *
     * @details It simply calls simplemc::mean_acc::operator<<(const W&) of the mean accumulator of
     * the current accumulating batch.
     *
     * @tparam W Eigen vector/array/expression type.
     * @param vec Vector/Array/Expression to be accumulated.
     * @return Reference to this object.
     */
    template <typename W>
    batch_acc& operator<<(const W& vec) {
        acc_batches_[bidx_] << vec;
        check_and_advance();
        return *this;
    }

    /**
     * @brief Accumulate a range of values to consecutive elements in the accumulator.
     *
     * @details It simply calls simplemc::mean_acc::accumulate(R&&, size_type) of the mean accumulator
     * of the current accumulating batch.
     *
     * @tparam R Input range of values.
     * @param rg Range of values to be accumulated.
     * @param idx Starting index for the accumulator.
     */
    template <ranges::input_range R>
    void accumulate(R&& rg, size_type idx = 0) {
        acc_batches_[bidx_].accumulate(std::forward<R>(rg), idx);
        check_and_advance();
    }

    /**
     * @brief Accumulate a range of values to arbitrary elements but with different indices.
     *
     * @details It simply calls simplemc::mean_acc::accumulate(R1&&, R2&&) of the mean accumulator of
     * the current accumulating batch.
     *
     * @tparam R1 Input range of values.
     * @tparam R2 Input range of indices.
     * @param rg Range of values to be accumulated.
     * @param idxs Range of indices.
     */
    template <ranges::input_range R1, ranges::input_range R2>
    void accumulate(R1&& rg, R2&& idxs) {
        acc_batches_[bidx_].accumulate(std::forward<R1>(rg), std::forward<R2>(idxs));
        check_and_advance();
    }

    /**
     * @brief Create a multi-value accumulator.
     *
     * @details See simplemc::multivalue_acc.
     *
     * @return Multi-value accumulator.
     */
    mva_type make_mva() { return mva_type(acc_batches_[bidx_]); }

    /**
     * @brief Get the size of the accumulator.
     *
     * @return Number of elements.
     */
    [[nodiscard]] auto size() const { return full_batches_[0].size(); }

    /**
     * @brief Get the total number of accumulated values.
     *
     * @return Number of accumulated values.
     */
    [[nodiscard]] auto count() const {
        auto op = [](count_type sum, const mean_acc_type& acc) { return sum + acc.count(); };
        count_type sum = std::accumulate(full_batches_.begin(), full_batches_.end(), count_type { 0 }, op);
        sum += std::accumulate(acc_batches_.begin(), acc_batches_.end(), count_type { 0 }, op);
        return sum;
    }

    /**
     * @brief Get the count in the full batches.
     *
     * @return Number of accumulated values in a single full batch.
     */
    [[nodiscard]] auto batch_count() const { return full_batches_[0].count(); }

    /**
     * @brief Get all full batches after combining a given number of them.
     *
     * @details It takes all the batches which are currently considered to be full and combines a
     * given number of batches. The resulting batches are then returned.
     *
     * It returns an empty vector if no merge has been performed so far or if the number of batches
     * to combine is zero or greater than the number of full batches.
     *
     * @param num Number of batches to combine.
     * @return `std::vector` of batch_acc::mean_acc_type objects.
     */
    [[nodiscard]] auto batches(std::size_t num = 1) const {
        if (is_first_merge() || num == 0 || num > full_batches_.size() + bidx_) {
            return std::vector<mean_acc_type> {};
        }
        // get all full batches
        std::vector<mean_acc_type> res = full_batches_;
        const auto it = std::next(acc_batches_.begin(), bidx_);
        std::transform(acc_batches_.begin(), it, std::back_inserter(res), [](const auto& acc) { return acc; });

        // return early if no merging is needed
        if (num == 1) {
            return res;
        }

        // combine the batches
        const auto sz = res.size() / num;
        for (std::size_t i = 0; i < sz; ++i) {
            res[i] = res[i * num];
            for (std::size_t j = 1; j < num; ++j) {
                res[i] << full_batches_[i * num + j];
            }
        }
        res.resize(sz);
        return res;
    }

    /**
     * @brief Get the vector of batches used for storing full batches.
     *
     * @return `std::vector` of batch_acc::mean_acc_type objects.
     */
    [[nodiscard]] const auto& batch_vector_full() const { return full_batches_; }

    /**
     * @brief Get the vector of batches used for accumulating new batches.
     *
     * @return `std::vector` of batch_acc::mean_acc_type objects.
     */
    [[nodiscard]] const auto& batch_vector_accumulating() const { return acc_batches_; }

    /**
     * @brief Get a mean accumulator with all the accumulated data.
     *
     * @return batch_acc::mean_acc_type object as if all the data was accumulated in a single batch.
     */
    [[nodiscard]] auto mean_accumulator() const {
        mean_acc_type res { size() };
        for (const auto& acc : full_batches_) {
            res << acc;
        }
        for (const auto& acc : acc_batches_) {
            res << acc;
        }
        return res;
    }

    /**
     * @brief Get a variance accumulator with full batches as effective samples.
     *
     * @details Full batches can optionally (`num > 1`) be combined before being used as effective
     * samples.
     *
     * The resulting variance accumulator is identical to a simplemc::block_acc with a block size of
     * `batch_count() * num`.
     *
     * @param num Number of batches to be combined into an effective sample.
     * @return simplemc::var_acc object.
     */
    [[nodiscard]] auto var_accumulator(std::size_t num = 1) const {
        simplemc::var_acc<vec_type, varalg()> res { size() };
        for (const auto& acc : batches(num)) {
            res << acc.mean();
        }
        return res;
    }

    /**
     * @brief Get a covariance accumulator with full batches as effective samples.
     *
     * @details Full batches can optionally (`num > 1`) be combined before being used as effective
     * samples.
     *
     * The resulting variance accumulator is identical to a simplemc::block_acc with a block size of
     * `batch_count() * num`.
     *
     * @param num Number of batches to be combined into an effective sample.
     * @return simplemc::covar_acc object.
     */
    [[nodiscard]] auto covar_accumulator(std::size_t num = 1) const {
        simplemc::covar_acc<vec_type, varalg()> res { size() };
        for (const auto& acc : batches(num)) {
            res << acc.mean();
        }
        return res;
    }

    /**
     * @brief Check if the current batch is full and advance to the next one if so.
     */
    void check_and_advance() {
        if (is_batch_full()) {
            if (bidx_ == acc_batches_.size() - 1) {
                merge_batches();
            } else {
                ++bidx_;
            }
        }
    }

    /**
     * @brief Check batches and parameters for consistency.
     *
     * @details It throws a simplemc::simplemc_exception in case of inconsistencies.
     */
    void check_batches() {
        // check the batch size
        if (bcount_ < 1 || !std::has_single_bit(bcount_)) {
            throw simplemc_exception("Incorrect batch size", "batch_acc::check_batches");
        }

        // check the batch index
        if (bidx_ >= acc_batches_.size()) {
            throw simplemc_exception("Batch index out of bounds", "batch_acc::check_batches");
        }

        // check batches
        if (full_batches_.size() < 2 || full_batches_.size() % 2 != 0) {
            throw simplemc_exception("Minimum number of batches < 2 or odd", "batch_acc::check_batches");
        }
        if (full_batches_.size() != acc_batches_.size()) {
            throw simplemc_exception(
                "Number of storing batches != number of accumulating batches", "batch_acc::check_batches");
        }
        auto check_size = [this](const auto& acc) {
            if (acc.size() != size()) {
                throw simplemc_exception("Size mismatch in batches", "batch_acc::check_batches");
            }
        };
        auto check_full_batch_count = [this](const auto& acc) {
            if (acc.count() != bcount_) {
                throw simplemc_exception("Full batch count != current batch size", "batch_acc::check_batches");
            }
        };
        auto check_empty_batch_count = [](const auto& acc) {
            if (acc.count() != 0) {
                throw simplemc_exception("Empty batch count != 0", "batch_acc::check_batches");
            }
        };
        // check full batches
        for (const auto& acc : full_batches_) {
            check_size(acc);
            if (is_first_merge()) {
                check_empty_batch_count(acc);
            } else {
                check_full_batch_count(acc);
            }
        }
        // check full accumulating batches
        for (std::size_t i = 0; i < bidx_; ++i) {
            check_size(acc_batches_[i]);
            check_full_batch_count(acc_batches_[i]);
        }
        // check the current accumulating batch
        check_size(acc_batches_[bidx_]);
        if (acc_batches_[bidx_].count() >= bcount_) {
            throw simplemc_exception("Accumulating batch count >= current batch size", "batch_acc::check_batches");
        }
        // check empty accumulating batches
        for (std::size_t i = bidx_ + 1; i < acc_batches_.size(); ++i) {
            check_size(acc_batches_[i]);
            check_empty_batch_count(acc_batches_[i]);
        }
    }

private:
    std::vector<mean_acc_type> full_batches_;
    std::vector<mean_acc_type> acc_batches_;
    count_type bcount_ { 1 };
    std::size_t bidx_ { 0 };
};

/**
 * @brief Alias for a statically sized batch accumulator of size 1.
 *
 * @tparam T Type of accumulated data.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <double_or_complex T, varalg A = varalg::welford>
using batch_acc_single = batch_acc<Eigen::Matrix<T, 1, 1>, A>;

/**
 * @brief Alias for a statically sized batch accumulator.
 *
 * @tparam T Type of accumulated data.
 * @tparam M Size of the accumulator.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <double_or_complex T, int M, varalg A = varalg::welford>
    requires(M >= 1)
using batch_acc_static = batch_acc<Eigen::Matrix<T, M, 1>, A>;

/**
 * @brief Alias for a dynamically sized batch accumulator.
 *
 * @tparam T Type of accumulated data.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <double_or_complex T, varalg A = varalg::welford>
using batch_acc_dynamic = batch_acc<Eigen::Matrix<T, Eigen::Dynamic, 1>, A>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_BATCH_ACC_HPP
