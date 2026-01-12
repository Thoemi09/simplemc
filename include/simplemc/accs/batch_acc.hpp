/**
 * @file
 * @brief Accumulator for grouping samples of a random vector into a fixed number of batches.
 */

#ifndef SIMPLEMC_ACCS_BATCH_ACC_HPP
#define SIMPLEMC_ACCS_BATCH_ACC_HPP

#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/covar_acc.hpp>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/accs/multivalue_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/accs/var_acc.hpp>
#include <simplemc/accs/varalg.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <optional>
#include <vector>

namespace simplemc {

/**
 * @ingroup simplemc-accs-utils
 * @brief Merge batches together.
 *
 * @details It takes a vector of \f$ M_b \f$ batches (simplemc::mean_acc) and combines them into \f$
 * M_b' = \lfloor M_b / c \rfloor \f$ new batches, where \f$ c \f$ is the given number of batches that
 * should be merged together. If \f$ c \f$ is not a divisor of \f$ M_b \f$, the left over batches are
 * simply discarded.
 *
 * If \f$ c < 2 \f$ or \f$ c > M_b \f$, no merges are performed and batches are left unchanged.
 *
 * It is assumed, that the batches all have the same size \f$ M \f$.
 *
 * @param c Number of batches to merge together.
 * @param batches `std::vector` containing the \f$ M_b \f$ batches before the merge.
 */
template <eigen_vector V, varalg A>
void merge_batches(std::size_t c, std::vector<mean_acc<V, A>>& batches) {
    // early return if the number of batches to combine is < 2 or > M_b
    if (c < 2 || c > batches.size()) {
        return;
    }

    // combine the batches
    const auto mb_p = batches.size() / c;
    for (std::size_t i = 0; i < mb_p; ++i) {
        batches[i] = batches[i * c];
        for (std::size_t j = 1; j < c; ++j) {
            batches[i] << batches[i * c + j];
        }
    }
    batches.resize(mb_p);
}

/**
 * @addtogroup simplemc-accs-accs-batch
 * @{
 */

/**
 * @brief Accumulator for grouping samples of a random vector into a fixed number of batches.
 *
 * @details The accumulator takes two template parameters:
 * - the type of the random samples (a simplemc::eigen_vector type) and
 * - the algorithm (simplemc::varalg) that should be used to accumulate the data.
 *
 * Both of them determine how the accumulation is actually done and what is stored in the accumulator.
 *
 * @code{.cpp}
 * #include <fmt/ranges.h>
 * #include <simplemc/accs.hpp>
 *
 * #include <random>
 *
 * int main() {
 *     // AR(1) parameters and white noise distribution: phi = 0.9, sigma = 1.0
 *     const double phi = 0.9;
 *     const double sigma = 1.0;
 *     std::mt19937_64 rng;
 *     std::normal_distribution<double> normal_dist(0.0, sigma);
 *
 *     // accumulate samples into a batch accumulator
 *     simplemc::batch_acc_single<double> acc {};
 *     double x_t = 0.0;
 *     for (int i = 0; i < 1000000; ++i) {
 *         acc << x_t;
 *         x_t = phi * x_t + normal_dist(rng);
 *     }
 *
 *     // print the mean and variance of the accumulated data
 *     auto vacc = acc.var_accumulator();
 *     fmt::println("Mean: {}", vacc.mean());
 *     fmt::println("Variance: {}", vacc.variance());
 * }
 * @endcode
 *
 * Output:
 *
 * ```
 * Mean: 0.002233435848225748
 * Variance: 0.00010741776938402939
 * ```
 *
 * @tparam V simplemc::eigen_vector type.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <eigen_vector V, varalg A = varalg::welford>
class batch_acc {
public:
    /**
     * @brief Type of accumulated values (simplemc::double_or_complex).
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
     * @return Its simplemc::varalg tag.
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
    void combine_batches() {
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
     * @brief Construct a batch accumulator with a given number of elements \f$ M \f$ and batches \f$
     * M_b \f$.
     *
     * @details For dynamically sized accumulators, it throws a simplemc::simplemc_exception if the
     * given size \f$ M \leq 0 \f$.
     *
     * For static sized accumulators, \f$ M \f$ is ignored.
     *
     * It calls check_batches() to check the given input parameters.
     *
     * @param m Number of elements \f$ M \f$.
     * @param m_b Number of batches \f$ M_b \f$.
     */
    explicit batch_acc(size_type m = 1, std::size_t m_b = 256) :
        full_batches_(m_b, mean_acc_type { m }),
        acc_batches_(m_b, mean_acc_type { m }) {
        check_batches();
    }

    /**
     * @brief Construct a batch accumulator with the given vectors of full and accumulating batches.
     *
     * @details It calls check_batches() to check the given batches.
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
     * @brief Subscript operator sets the index \f$ i \f$ of the current accumulating batch and
     * returns a reference to `this` object.
     *
     * @param i Index \f$ i \f$.
     * @return Reference to `this` object.
     */
    batch_acc& operator[](size_type i) {
        acc_batches_[bidx_][i];
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a single value \f$ x \f$.
     *
     * @details It simply calls
     * - simplemc::mean_acc::operator<<(const T &) of the mean accumulator of the current
     * accumulating batch and
     * - check_and_advance() to check if the batch is full.
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @tparam T Type of the value to be accumulated.
     * @param x Value \f$ x \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename T>
        requires std::convertible_to<T, value_type>
    batch_acc& operator<<(const T& x) {
        acc_batches_[bidx_] << x;
        check_and_advance();
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a vector \f$ \mathbf{v} \f$.
     *
     * @details It simply calls
     * - simplemc::mean_acc::operator<<(const W&) of the mean accumulator of the current accumulating
     * batch and
     * - check_and_advance() to check if the batch is full.
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @tparam W Eigen vector/array/expression type.
     * @param v Vector/Array/Expression \f$ \mathbf{v} \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename W>
    batch_acc& operator<<(const W& v) {
        acc_batches_[bidx_] << v;
        check_and_advance();
        return *this;
    }

    /**
     * @brief Accumulate a range of values to consecutive elements in the accumulator.
     *
     * @details It simply calls
     * - simplemc::mean_acc::accumulate(R&&, size_type) of the mean accumulator of the current
     * accumulating batch and
     * - check_and_advance() to check if the batch is full.
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @tparam R Input range of values.
     * @param rg Range of values to be accumulated.
     * @param i Index \f$ i \f$ of the first element in the batch that a value will be added to.
     */
    template <ranges::input_range R>
    void accumulate(R&& rg, size_type i = 0) {
        acc_batches_[bidx_].accumulate(std::forward<R>(rg), i);
        check_and_advance();
    }

    /**
     * @brief Accumulate a range of values to arbitrary elements with the given indices.
     *
     * @details It simply calls
     * - simplemc::mean_acc::accumulate(R1&&, R2&&) of the mean accumulator of the current
     * accumulating batch and
     * - check_and_advance() to check if the batch is full.
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @tparam R1 Input range of values.
     * @tparam R2 Input range of indices.
     * @param rg Range of values to be accumulated.
     * @param idxs Range of indices at which the values should be accumulated.
     */
    template <ranges::input_range R1, ranges::input_range R2>
    void accumulate(R1&& rg, R2&& idxs) {
        acc_batches_[bidx_].accumulate(std::forward<R1>(rg), std::forward<R2>(idxs));
        check_and_advance();
    }

    /**
     * @brief Create a multi-value accumulator.
     *
     * @details The user is responsible for calling simplemc::multivalue_acc::increment_count as well
     * as check_and_advance() after all values have been added, otherwise the number of samples will
     * not be correct. For example,
     * @code{.cpp}
     * auto mva = acc.make_mva();
     * mva[i_1] << v_i_1;
     * // ...
     * mva[i_n] << v_i_n;
     * mva.increment_count();
     * acc.check_and_advance();
     * @endcode
     *
     * @return Multi-value accumulator wrapping `this` object.
     */
    [[nodiscard]] mva_type make_mva() { return mva_type(acc_batches_[bidx_]); }

    /**
     * @brief Get the size \f$ M \f$ of the accumulator.
     *
     * @return Number of elements.
     */
    [[nodiscard]] auto size() const { return full_batches_[0].size(); }

    /**
     * @brief Get the total number of accumulated samples \f$ N \f$.
     *
     * @return Number of accumulated samples.
     */
    [[nodiscard]] auto count() const {
        auto op = [](count_type sum, const mean_acc_type& acc) { return sum + acc.count(); };
        count_type sum = std::accumulate(full_batches_.begin(), full_batches_.end(), count_type { 0 }, op);
        sum += std::accumulate(acc_batches_.begin(), acc_batches_.end(), count_type { 0 }, op);
        return sum;
    }

    /**
     * @brief Get the count \f$ N_b \f$ in the full batches.
     *
     * @return Number of accumulated values in a single full batch.
     */
    [[nodiscard]] auto batch_count() const { return full_batches_[0].count(); }

    /**
     * @brief Get the number of full batches \f$ \tilde{M}_b \in [M_b, 2 M_b) \f$.
     *
     * @details This is equal to calling `batches().%size()`.
     *
     * @return Number of currently full batches.
     */
    [[nodiscard]] auto num_full_batches() const { return is_first_merge() ? 0 : full_batches_.size() + bidx_; }

    /**
     * @brief Get all full batches after merging a given number of them together.
     *
     * @details It takes all the batches which are currently considered to be full and uses
     * simplemc::merge_batches to merge a given number of them together.
     *
     * It returns an empty vector
     * - if the current batch count is zero, i.e. \f$ N_b = 0 \f$, or
     * - if \f$ c > \tilde{M}_b \f$, i.e. there are not enough full batches to merge.
     *
     * @param c Number of batches to merge together.
     * @return `std::vector` containing \f$ \tilde{M}_b \in [M_b, 2M_b) \f$ full (and merged) batches.
     */
    [[nodiscard]] auto batches(std::size_t c = 1) const {
        // return an empty vector if we don't have enough full batches
        if (is_first_merge() || c > full_batches_.size() + bidx_) {
            return std::vector<mean_acc_type> {};
        }

        // get all full batches
        std::vector<mean_acc_type> res = full_batches_;
        const auto it = std::next(acc_batches_.begin(), bidx_);
        std::transform(acc_batches_.begin(), it, std::back_inserter(res), [](const auto& acc) { return acc; });

        // merge the batches
        merge_batches(c, res);
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
     * @brief Get a mean accumulator with all \f$ N \f$ accumulated samples.
     *
     * @return batch_acc::mean_acc_type object as if all the data was accumulated into a single batch.
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
     * @brief Get a variance accumulator with all \f$ \tilde{M}_b \f$ full batches as effective
     * samples.
     *
     * @details Full batches can optionally be merged together before being used as effective samples.
     *
     * The resulting variance accumulator is identical to a simplemc::block_acc with a block size of
     * \f$ N_b * c \f$.
     *
     * @param c Number of batches to be combined into an effective sample.
     * @return simplemc::var_acc object.
     */
    [[nodiscard]] auto var_accumulator(std::size_t c = 1) const {
        simplemc::var_acc<vec_type, varalg()> res { size() };
        for (const auto& acc : batches(c)) {
            res << acc.mean();
        }
        return res;
    }

    /**
     * @brief Get a covariance accumulator with all \f$ \tilde{M}_b \f$ full batches as effective
     * samples.
     *
     * @details Full batches can optionally be merged together before being used as effective samples.
     *
     * The resulting covariance accumulator is identical to a simplemc::block_acc with a block size of
     * \f$ N_b * c \f$.
     *
     * @param c Number of batches to be combined into an effective sample.
     * @return simplemc::covar_acc object.
     */
    [[nodiscard]] auto covar_accumulator(std::size_t c = 1) const {
        simplemc::covar_acc<vec_type, varalg()> res { size() };
        for (const auto& acc : batches(c)) {
            res << acc.mean();
        }
        return res;
    }

    /**
     * @brief Check if the current accumulating batch is full and advance to the next one if so.
     *
     * @details If all accumulating batches are full, the full and accumulating batches are merged
     * together and the accumulating batches are reset.
     */
    void check_and_advance() {
        if (is_batch_full()) {
            if (bidx_ == acc_batches_.size() - 1) {
                combine_batches();
            } else {
                ++bidx_;
            }
        }
    }

    /**
     * @brief Check batches and parameters for consistency.
     *
     * @details It throws a simplemc::simplemc_exception if
     * - the index \f$ i_b \f$ of the current accumulating batch is \f$ \geq M_b \f$,
     * - \f$ M_b < 2 \f$ or if \f$ M_b \f$ is not even,
     * - the number of full batches is not equal to the number of accumulating batches,
     * - the number of samples in full batches is
     *   - \f$ \neq 0 \f$ in case no merge has happened yet,
     *   - \f$ \neq N_b \f$ otherwise,
     * - the number of samples in an accumulating batch with index \f$ i \f$ is
     *   - \f$ \neq N_b \f$ in case \f$ i < i_b \f$,
     *   - \f$ < N_b \f$ in case \f$ i = i_b \f$,
     *   - \f$ \neq 0 \f$ in case \f$ i > i_b \f$,
     * - the sizes of all batches are \f$ \neq M \f$.
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

    /**
     * @brief Collect batch accumulators from different MPI processes.
     *
     * @details Batches from different processes are not merged together. Instead, they are simply
     * gathered into a vector.
     *
     * If `same_count` is true, it will enforce that all gathered batches have the same number of
     * accumulated samples, i.e. it will merge batches together if necessary.
     *
     * It asserts that the size of the accumulator is equal on all processes.
     *
     * @param comm simplemc::mpi::communicator object.
     * @param acc Batch accumulator.
     * @param same_count Should we perform merges to enforce the same count on each process?
     * @return `std::vector` containing the batches gatherered from all processes.
     */
    friend std::vector<mean_acc_type> mpi_collect(
        const mpi::communicator& comm, const batch_acc& acc, bool same_count) {
        assert(all_equal(acc.size(), comm));

        // get full batches
        auto bs = acc.batches();

        // should we enforce the same count on each process?
        if (same_count) {
            // perform merges if necessary
            auto max_count = acc.batch_count();
            mpi::all_reduce(acc.batch_count(), max_count, MPI_MAX, comm);
            merge_batches(max_count / acc.batch_count(), bs);

            // if a process does not have enough samples, discard its batches
            if (!bs.empty() && bs[0].count() != max_count) {
                bs.clear();
            }
        }

        // number of batches on each process
        std::vector<std::size_t> vec_sz(comm.size());
        mpi::all_gather(bs.size(), vec_sz, comm);

        // gather the batches
        const auto sz = std::accumulate(vec_sz.begin(), vec_sz.end(), static_cast<std::size_t>(0));
        std::vector<mean_acc_type> res(sz, mean_acc_type { acc.size() });

        // each process broadcasts its batches to all other processes
        std::size_t prev_idx = 0;
        for (int i = 0; i < comm.size(); ++i) {
            auto view = ranges::views::drop(res, prev_idx) | ranges::views::take(vec_sz[i]);
            if (comm.rank() == i) {
                std::ranges::copy(bs, ranges::begin(view));
            }
            for (auto& macc : view) {
                macc.broadcast(i, comm);
            }
            prev_idx += vec_sz[i];
        }
        return res;
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

/**
 * @brief Accumulate (complex) random samples in a simplemc::batch_acc.
 *
 * @details See simplemc::batch_acc for more details on how the random samples are accumulated.
 *
 * It throws a simplemc::simplemc_exception if the range is empty.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam R simplemc::random_sample_range type.
 * @param rg Range containing the random samples \f$ \left\{ \mathbf{z}^{(j)} : j = 1, \dots, N
 * \right\} \f$.
 * @param m_b Number of batches \f$ M_b \f$.
 * @param t Optional vector/scalar shift \f$ \mathbf{t} \f$ that is applied when accumulating the
 * data.
 * @return simplemc::batch_acc containing the accumulated random samples from the given range.
 */
template <varalg A = varalg::welford, random_sample_range R>
[[nodiscard]] auto make_batch_acc(R&& rg, std::size_t m_b, // NOLINT (ranges need not be forwarded)
    std::optional<ranges::range_value_t<R>> t = std::nullopt) {
    using value_type = ranges::range_value_t<R>;

    auto const sz = detail::random_sample_size(*ranges::begin(rg));
    if constexpr (double_or_complex<value_type>) {
        return detail::make_acc<batch_acc_single<value_type, A>>(rg, t, sz, m_b);
    } else {
        return detail::make_acc<batch_acc<value_type, A>>(rg, t, sz, m_b);
    }
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_BATCH_ACC_HPP
