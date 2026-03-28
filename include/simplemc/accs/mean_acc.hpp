/**
 * @file
 * @brief Accumulator for calculating the sample mean.
 */

#ifndef SIMPLEMC_ACCS_MEAN_ACC_HPP
#define SIMPLEMC_ACCS_MEAN_ACC_HPP

#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/multivalue_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/accs/varalg.hpp>
#include <simplemc/mpi.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <Eigen/Dense>

#include <cassert>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace simplemc {

/**
 * @addtogroup simplemc-accs-accs-mean
 * @{
 */

/// @cond

/* Forward declarations. */
template <eigen_vector V, varalg A>
class batch_acc;

/// @endcond

/**
 * @brief Accumulator for calculating the sample mean.
 *
 * @details The accumulator satisfies simplemc::mean_accumulator and takes two template parameters:
 * - the type of the data samples (a simplemc::sample_type) and
 * - the algorithm (simplemc::varalg) that should be used to accumulate the data.
 *
 * Both of them determine how the accumulation is actually done and what is stored in the accumulator.
 * The accumulated data is stored in a single vector \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$ and its
 * content depends on the algorithm:
 * - `standard`: The mean data is accumulated with
 *   \f[
 *     \mathbf{m}^{(N)} = \mathbf{m}^{(N-1)} + \mathbf{z}^{(N)} - \mathbf{t} =
 *     \sum_{j=1}^N \left( \mathbf{z}^{(j)} - \mathbf{t} \right) \; ,
 *   \f]
 *   such that the sample mean is given by
 *   \f[
 *     \overline{\mathbf{z}}^{(N)} = \frac{\mathbf{m}^{(N)}}{N} + \mathbf{t} =
 *     \frac{1}{N} \sum_{j=1}^N \mathbf{z}^{(j)} \; .
 *   \f]
 *
 * - `welford`: The mean data is accumulated with
 *   \f[
 *     \mathbf{n}^{(N)} = \mathbf{n}^{(N-1)} + \frac{1}{N} \left( \mathbf{z}^{(N)} -
 *     \mathbf{t} - \mathbf{n}^{(N-1)} \right) =
 *     \frac{1}{N} \sum_{j=1}^N \left( \mathbf{z}^{(j)} - \mathbf{t} \right) \; ,
 *   \f]
 *   such that the sample mean is given by
 *   \f[
 *     \overline{\mathbf{z}}^{(N)} = \mathbf{n}^{(N)} + \mathbf{t} =
 *     \frac{1}{N} \sum_{j=1}^N \mathbf{z}^{(j)} \; .
 *   \f]
 *
 * Here, \f$ \mathbf{t} \f$ is a constant vector/scalar that can optionally (and manually by the user)
 * be applied to the data samples to increase numerical accuracy. See also
 * @ref simplemc-accs-stats-mean.
 *
 * @include accs/doc_mean_acc.cpp
 *
 * Output:
 *
 * ```
 * Mean: 3
 * ```
 *
 * @tparam T simplemc::sample_type to accumulate.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <sample_type T, varalg A = varalg::welford>
class mean_acc {
public:
    /**
     * @brief Type of accumulated samples (simplemc::sample_type).
     */
    using sample_type = T;

    /**
     * @brief Vector type for storing accumulated mean data.
     */
    using vec_type = std::conditional_t<sample_scalar<T>, Eigen::Matrix<T, 1, 1>, T>;

    /**
     * @brief Underlying scalar type of accumulated samples (simplemc::double_or_complex).
     */
    using value_type = typename vec_type::value_type;

    /**
     * @brief Type for counting the number of accumulated samples.
     */
    using count_type = std::uint64_t;

    /**
     * @brief Size type of the accumulator.
     */
    using size_type = long;

    /**
     * @brief Static size of the accumulator.
     */
    static constexpr int static_size = vec_type::RowsAtCompileTime;

    /**
     * @brief Is the accumulator dynamically sized?
     */
    static constexpr bool is_dynamic = (vec_type::RowsAtCompileTime == Eigen::Dynamic);

    /**
     * @brief Get the algorithm used to accumulate the data.
     *
     * @return Its simplemc::varalg tag.
     */
    static constexpr auto varalg() noexcept { return A; }

    /* Friend declarations. */
    friend class multivalue_acc<mean_acc>;
    friend std::vector<mean_acc> mpi_collect(const mpi::communicator&, const batch_acc<vec_type, varalg()>&, bool);

private:
    // Add a single value to the accumulator without increasing the count (the given count is assumed
    // to be already increased by one).
    void add_value(value_type val, size_type idx, count_type count) {
        assert(idx >= 0 && idx < size());
        if constexpr (varalg() == varalg::standard) {
            mdata_(idx) += val;
        } else {
            mdata_(idx) += (val - mdata_(idx)) / static_cast<double>(count);
        }
    }

    // Broadcast into/from the current mean accumulator. Only used in mpi_collect with batch_acc.
    void broadcast(int root, const mpi::communicator& comm) {
        mpi::broadcast(make_span(mdata_), root, comm);
        mpi::broadcast(count_, root, comm);
    }

public:
    /**
     * @brief Construct a mean accumulator with a given number of elements \f$ M \f$.
     *
     * @details For dynamically sized accumulators, it throws a simplemc::simplemc_exception if the
     * given size \f$ M \leq 0 \f$.
     *
     * For statically sized accumulators, \f$ M \f$ is ignored.
     *
     * @param m Number of elements \f$ M \f$.
     */
    explicit mean_acc(size_type m = 1) : mdata_(vec_type::Zero(is_dynamic ? m : static_size)), count_(0), idx_(0) {
        if constexpr (is_dynamic) {
            if (m <= 0) {
                throw simplemc_exception("Dynamic size <= 0");
            }
        }
    }

    /**
     * @brief Construct a mean accumulator with the given accumulated data \f$ \mathbf{m}^{(N)}/
     * \mathbf{n}^{(N)} \f$ and number of samples \f$ N \f$.
     *
     * @details For dynamically sized accumulators, the size \f$ M \f$ of the data storage must be
     * \f$ \geq 1 \f$. Otherwise, it throws a simplemc::simplemc_exception.
     *
     * @param md Accumulated mean data \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$.
     * @param n Number of accumulated samples \f$ N \f$.
     */
    mean_acc(const vec_type& md, count_type n) : mdata_(md), count_(n), idx_(0) {
        if constexpr (is_dynamic) {
            if (mdata_.size() <= 0) {
                throw simplemc_exception("Dynamic size <= 0");
            }
        }
    }

    /**
     * @brief Reset the accumulator to its initial state, i.e. with no accumulated values.
     */
    void reset() {
        mdata_.setZero();
        count_ = 0;
        idx_ = 0;
    }

    /**
     * @brief Subscript operator sets the index \f$ i \f$ and returns a reference to `this` object.
     *
     * @details The index is *sticky*: it persists until changed by another call to operator[]() or
     * until reset() is called. For scalar accumulators (size \f$ M = 1 \f$), the index should
     * remain at 0.
     *
     * @param i Index \f$ i \f$.
     * @return Reference to `this` object.
     */
    mean_acc& operator[](size_type i) noexcept {
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
     * @tparam U Type of the value to be accumulated.
     * @param x Value \f$ x \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename U>
        requires std::convertible_to<U, value_type>
    mean_acc& operator<<(const U& x) {
        ++count_;
        add_value(x, idx_, count_);
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a vector \f$ \mathbf{v} \f$.
     *
     * @details See also @ref simplemc-accs-accs-how.
     *
     * @tparam W Eigen vector/array/expression type.
     * @param v Vector/Array/Expression \f$ \mathbf{v} \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename W>
    mean_acc& operator<<(const W& v) {
        assert(v.size() == size());
        ++count_;
        if constexpr (varalg() == varalg::standard) {
            mdata_ += v.matrix();
        } else {
            mdata_ += (v.matrix() - mdata_) / static_cast<double>(count_);
        }
        return *this;
    }

    /**
     * @brief Stream operator for incorporating the data from another mean accumulator.
     *
     * @details Let the subscripts 1 and 2 correspond to the two accumulators we want to combine such
     * that \f$ N = N_1 + N_2 \f$ is the total number of accumulated values. Then, depending on the
     * simplemc::varalg, the combined accumulated data can be calculated as follows:
     * - `standard`:
     *   - \f$ \mathbf{m}^{(N)} = \mathbf{m}_{1}^{(N_1)} + \mathbf{m}_{2}^{(N_2)} \f$ .
     *
     * - `welford`:
     *   - \f$ \mathbf{n}^{(N)} = \frac{N_1}{N}\mathbf{n}_{1}^{(N_1)} + \frac{N_2}{N}
     *     \mathbf{n}_{2}^{(N_2)} \f$.
     *
     * If the accumulator to be incorporated is empty(), nothing is done.
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @param acc_other simplemc::mean_acc to be incorporated.
     * @return Reference to `this` object.
     */
    mean_acc& operator<<(const mean_acc& acc_other) {
        assert(size() == acc_other.size());

        // early return if the other accumulator is empty
        if (acc_other.empty()) {
            return *this;
        }

        // incorporate the data and count
        if constexpr (varalg() == varalg::standard) {
            mdata_ += acc_other.mdata_;
        } else {
            const auto n1 = static_cast<double>(count_);
            const auto n2 = static_cast<double>(acc_other.count_);
            const auto ratio = n2 / (n1 + n2);
            mdata_ = mdata_ + (acc_other.mdata_ - mdata_) * ratio;
        }
        count_ += acc_other.count_;

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
     * @tparam R Input range of values.
     * @param rg Range of values to be accumulated.
     * @param i Index \f$ i \f$ of the first element in the accumulator that a value will be added to.
     */
    template <ranges::input_range R>
    void accumulate(R&& rg, size_type i = 0) { // NOLINT (ranges need not be forwarded)
        ++count_;
        for (auto val : rg) {
            add_value(val, i, count_);
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
     * @tparam R1 Input range of values.
     * @tparam R2 Input range of indices.
     * @param rg Range of values to be accumulated.
     * @param idxs Range of indices at which the values should be accumulated.
     */
    template <ranges::input_range R1, ranges::input_range R2>
    void accumulate(R1&& rg, R2&& idxs) { // NOLINT (ranges need not be forwarded)
        ++count_;
        for (auto [val, idx] : ranges::views::zip(rg, idxs)) {
            add_value(val, idx, count_);
        }
    }

    /**
     * @brief Create a multi-value accumulator.
     *
     * @details See simplemc::multivalue_acc.
     *
     * @return Multi-value accumulator wrapping `this` object.
     */
    [[nodiscard]] auto make_mva() noexcept { return multivalue_acc<mean_acc>(*this); }

    /**
     * @brief Get the size \f$ M \f$ of the accumulator.
     *
     * @return Number of elements.
     */
    [[nodiscard]] auto size() const noexcept { return mdata_.size(); }

    /**
     * @brief Get the total number of accumulated samples \f$ N \f$.
     *
     * @return Number of accumulated samples.
     */
    [[nodiscard]] auto count() const noexcept { return count_; }

    /**
     * @brief Check if the accumulator is empty.
     *
     * @return True if the count() is zero, i.e. \f$ N = 0 \f$, false otherwise.
     */
    [[nodiscard]] bool empty() const noexcept { return count_ == 0; }

    /**
     * @brief Get the accumulated mean data \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$.
     *
     * @return simplemc::eigen_vector of size \f$ M \f$ containing \f$ \mathbf{m}^{(N)}/
     * \mathbf{n}^{(N)} \f$.
     */
    [[nodiscard]] const vec_type& mdata() const noexcept { return mdata_; }

    /**
     * @brief Calculate the sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$.
     *
     * @details It calls simplemc::accs::mean with the accumulated mean data \f$ \mathbf{m}^{(N)}/
     * \mathbf{n}^{(N)} \f$ and the count \f$ N \f$.
     *
     * @return Sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$.
     */
    [[nodiscard]] auto mean() const {
        using simplemc::accs::mean;
        return detail::scalar_or_matrix<sample_scalar<sample_type>>(mean<varalg()>(mdata_, count_));
    }

    /**
     * @brief Collect mean accumulators from different MPI processes.
     *
     * @details It constructs a new mean accumulator with the reduced accumulated mean data and counts
     * from all MPI processes.
     *
     * The reduction operation depends on the simplemc::varalg algorithm used to accumulate the data.
     * See operator<<(const mean_acc&) for how it is done in the case of 2 accumulators.
     *
     * If the reduced count() of all accumulators is zero, no data is collected and the returned
     * accumulator will be empty().
     *
     * It asserts that the size of the accumulator is equal on all processes.
     *
     * @param comm simplemc::mpi::communicator object.
     * @param acc Mean accumulator.
     * @return Mean accumulator with the reduced data from all processes.
     */
    [[nodiscard]] friend mean_acc mpi_collect(const mpi::communicator& comm, const mean_acc& acc) {
        assert(all_equal(acc.size(), comm));
        mean_acc res { acc.size() };

        // reduce the count
        mpi::all_reduce(acc.count_, res.count_, MPI_SUM, comm);

        // early return if the reduced count is zero
        if (res.count_ == 0) {
            return res;
        }

        // reduce the accumulated data
        if constexpr (mean_acc::varalg() == varalg::standard) {
            mpi::all_reduce(make_span(acc.mdata_), make_span(res.mdata_), MPI_SUM, comm);
        } else {
            const auto n1 = static_cast<double>(acc.count_);
            const auto n = static_cast<double>(res.count_);
            const vec_type tmp_mdata = acc.mdata_ * n1 / n;
            mpi::all_reduce(make_span(tmp_mdata), make_span(res.mdata_), MPI_SUM, comm);
        }

        return res;
    }

private:
    vec_type mdata_;
    count_type count_;
    size_type idx_;
};

/**
 * @brief Alias for a statically sized mean accumulator.
 *
 * @tparam T Underlying scalar type of accumulated values (simplemc::double_or_complex).
 * @tparam M Size of the accumulator.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <double_or_complex T, int M, varalg A = varalg::welford>
    requires(M >= 1)
using mean_acc_static = mean_acc<Eigen::Matrix<T, M, 1>, A>;

/**
 * @brief Alias for a dynamically sized mean accumulator.
 *
 * @tparam T Underlying scalar type of accumulated values (simplemc::double_or_complex).
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <double_or_complex T, varalg A = varalg::welford>
using mean_acc_dynamic = mean_acc<Eigen::Matrix<T, Eigen::Dynamic, 1>, A>;

/**
 * @brief Accumulate data samples in a simplemc::mean_acc.
 *
 * @details See simplemc::mean_acc for more details on how the data samples are accumulated.
 *
 * It throws a simplemc::simplemc_exception if the range is empty.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam R simplemc::sample_range type.
 * @param rg Range containing the data samples \f$ \left\{ \mathbf{z}^{(j)} : j = 1, \dots, N
 * \right\} \f$.
 * @param t Optional vector/scalar shift \f$ \mathbf{t} \f$ that is applied when accumulating the
 * data.
 * @return simplemc::mean_acc containing the accumulated data samples from the given range.
 */
template <varalg A = varalg::welford, sample_range R>
[[nodiscard]] auto make_mean_acc(R&& rg, // NOLINT (ranges need not be forwarded)
    std::optional<ranges::range_value_t<R>> t = std::nullopt) {
    using value_type = ranges::range_value_t<R>;

    auto const sz = detail::random_sample_size(*ranges::begin(rg));
    return detail::make_acc<mean_acc<value_type, A>>(rg, t, sz);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_MEAN_ACC_HPP
