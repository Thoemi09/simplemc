/**
 * @file
 * @brief Accumulator for calculating the sample mean of a random vector.
 */

#ifndef SIMPLEMC_ACCS_MEAN_ACC_HPP
#define SIMPLEMC_ACCS_MEAN_ACC_HPP

#include <simplemc/accs/multivalue_acc.hpp>
#include <simplemc/accs/utils.hpp>
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
 * @brief Accumulator for calculating the sample mean of a random vector.
 *
 * @details The accumulator takes two template parameters:
 * - the type of the random samples (a simplemc::eigen_vector type) and
 * - the algorithm (simplemc::varalg) that should be used to accumulate the data.
 *
 * Both of them determine how the accumulation is actually done and what is stored in the accumulator.
 * The accumulated data is stored in a single vector \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$. Please
 * see simplemc::accs::mean for more details.
 *
 * @code{.cpp}
 * #include <fmt/ranges.h>
 * #include <simplemc/accs/mean_acc.hpp>
 *
 * #include <vector>
 *
 * int main() {
 *     // data to be sampled
 *     std::vector<double> data = { 1.0, 2.0, 3.0, 4.0, 5.0 };
 *
 *     // accumulate samples into a mean accumulator of size 1
 *     simplemc::mean_acc_single<double> acc;
 *     for (auto& val : data) {
 *         acc << val;
 *     }
 *
 *     // print the mean of the accumulated data
 *     fmt::println("Mean: {}", acc.mean());
 * }
 * @endcode
 *
 * Output:
 *
 * ```
 * Mean: 3
 * ```
 *
 * @tparam V simplemc::eigen_vector type.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <eigen_vector V, varalg A = varalg::welford>
class mean_acc {
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
     * @brief Type for counting the number of accumulated samples.
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
     * @brief Multi-value accumulator type.
     */
    using mva_type = multivalue_acc<mean_acc>;

    /**
     * @brief Get the algorithm used to accumulate the data.
     *
     * @return Its simplemc::varalg tag.
     */
    static constexpr auto varalg() { return A; }

    /* Friend declarations. */
    friend class multivalue_acc<mean_acc>;
    friend std::vector<mean_acc> mpi_collect(const mpi::communicator&, const batch_acc<V, varalg()>&, bool);

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
    void broadcast(const mpi::communicator& comm, int root) {
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
                throw simplemc_exception("Dynamic size <= 0", "mean_acc::mean_acc");
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
                throw simplemc_exception("Dynamic size <= 0", "mean_acc::mean_acc");
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
     * @param i Index \f$ i \f$.
     * @return Reference to `this` object.
     */
    mean_acc& operator[](size_type i) {
        idx_ = i;
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
     * @tparam T Type of the value to be accumulated.
     * @param x Value \f$ x \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename T>
        requires std::convertible_to<T, value_type>
    mean_acc& operator<<(const T& x) {
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
     * See also @ref simplemc-accs-accs-how.
     *
     * @param acc_other simplemc::mean_acc to be incorporated.
     * @return Reference to `this` object.
     */
    mean_acc& operator<<(const mean_acc& acc_other) {
        assert(size() == acc_other.size());
        if constexpr (varalg() == varalg::standard) {
            mdata_ += acc_other.mdata_;
        } else {
            const auto n1 = static_cast<double>(count_);
            const auto n2 = static_cast<double>(acc_other.count_);
            mdata_ = mdata_ * n1 / (n1 + n2) + acc_other.mdata_ * n2 / (n1 + n2);
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
    mva_type make_mva() { return mva_type(*this); }

    /**
     * @brief Get the size \f$ M \f$ of the accumulator.
     *
     * @return Number of elements.
     */
    [[nodiscard]] auto size() const { return mdata_.size(); }

    /**
     * @brief Get the total number of accumulated samples \f$ N \f$.
     *
     * @return Number of accumulated samples.
     */
    [[nodiscard]] auto count() const { return count_; }

    /**
     * @brief Get the accumulated data \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$ used for estimating
     * the mean.
     *
     * @return simplemc::eigen_vector of size \f$ M \f$ containing \f$ \mathbf{m}^{(N)}/
     * \mathbf{n}^{(N)} \f$ (content depends on the algorithm, see simplemc::accs::mean).
     */
    [[nodiscard]] const vec_type& mdata() const { return mdata_; }

    /**
     * @brief Calculate the sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$.
     *
     * @details It calls simplemc::accs::mean with the accumulated mean data and the count.
     *
     * For statically sized accumulators with \f$ M = 1 \f$, it returns a scalar of type
     * mean_acc::value_type. Otherwise, it returns a mean_acc::vec_type object.
     *
     * @return Sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$.
     */
    [[nodiscard]] auto mean() const {
        using simplemc::accs::mean;
        return detail::scalar_or_matrix<returns_scalar>(mean<varalg()>(mdata_, count_));
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
     * It asserts that the size of the accumulator is equal on all processes.
     *
     * @param comm simplemc::mpi::communicator object.
     * @param acc Mean accumulator.
     * @return Mean accumulator with the reduced data from all processes.
     */
    friend mean_acc mpi_collect(const mpi::communicator& comm, const mean_acc& acc) {
        assert(all_equal(acc.size(), comm));
        mean_acc res(acc.size());
        mpi::all_reduce(acc.count_, res.count_, MPI_SUM, comm);
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
 * @brief Alias for a statically sized mean accumulator of size 1.
 *
 * @tparam T Type of accumulated values (simplemc::double_or_complex).
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <double_or_complex T, varalg A = varalg::welford>
using mean_acc_single = mean_acc<Eigen::Matrix<T, 1, 1>, A>;

/**
 * @brief Alias for a statically sized mean accumulator.
 *
 * @tparam T Type of accumulated values (simplemc::double_or_complex).
 * @tparam M Size of the accumulator.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <double_or_complex T, int M, varalg A = varalg::welford>
    requires(M >= 1)
using mean_acc_static = mean_acc<Eigen::Matrix<T, M, 1>, A>;

/**
 * @brief Alias for a dynamically sized mean accumulator.
 *
 * @tparam T Type of accumulated values (simplemc::double_or_complex).
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <double_or_complex T, varalg A = varalg::welford>
using mean_acc_dynamic = mean_acc<Eigen::Matrix<T, Eigen::Dynamic, 1>, A>;

/**
 * @brief Accumulate (complex) random samples in a simplemc::mean_acc.
 *
 * @details See simplemc::accs::mean and simplemc::mean_acc for more details on how the random samples
 * are accumulated.
 *
 * It throws a simplemc::simplemc_exception if the range is empty.
 *
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 * @tparam R simplemc::random_sample_range type.
 * @param rg Range containing the random samples \f$ \left\{ \mathbf{z}^{(j)} : j = 1, \dots, N
 * \right\} \f$.
 * @param t Optional vector/scalar shift \f$ \mathbf{t} \f$ that is applied when accumulating the
 * data.
 * @return simplemc::mean_acc containing the accumulated random samples from the given range.
 */
template <varalg A = varalg::welford, random_sample_range R>
[[nodiscard]] auto make_mean_acc(R&& rg, // NOLINT (ranges need not be forwarded)
    std::optional<ranges::range_value_t<R>> t = std::nullopt) {
    using value_type = ranges::range_value_t<R>;

    auto const sz = detail::random_sample_size(*ranges::begin(rg));
    if constexpr (double_or_complex<value_type>) {
        return detail::make_acc<mean_acc_single<value_type, A>>(rg, t, sz);
    } else {
        return detail::make_acc<mean_acc<value_type, A>>(rg, t, sz);
    }
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_MEAN_ACC_HPP
