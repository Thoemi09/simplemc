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

namespace simplemc {

/**
 * @addtogroup simplemc-accs-accs
 * @{
 */

/* Forward declarations. */
template <eigen_vector V, varalg A>
class batch_acc;

/**
 * @brief Accumulator for calculating the sample mean of a random vector.
 *
 * @details The sample mean is used as an approximation to the exact expectation value. See
 * @ref simplemc-accs for a definition of the sample mean and expectation value.
 *
 * The accumulator takes two template parameters:
 * - the type of the random samples (a simplemc::eigen_vector type) and
 * - the algorithm (simplemc::varalg) that should be used to accumulate the data.
 *
 * Both of them determine how the accumulation is actually done and what is stored in the accumulator.
 * The accumulated data is stored in a single vector. Please see simplemc::accs::mean for more
 * details.
 *
 * @code{.cpp}
 * std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
 * simplemc::mean_acc<Eigen::Vector<double, 1>> acc;
 * for (auto& val : data) {
 *     acc << val;
 * }
 * fmt::print("Mean: {}\n", acc.mean());
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
     * @brief Multi-value accumulator type.
     */
    using mva_type = multivalue_acc<mean_acc>;

    /**
     * @brief Get the algorithm used to accumulate the data.
     *
     * @return The simplemc::varalg tag.
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
        mpi::broadcast(comm, make_span(mdata_), root);
        mpi::broadcast(comm, count_, root);
    }

public:
    /**
     * @brief Construct a mean accumulator with a given number of elements.
     *
     * @details For dynamically sized accumulators, it throws a simplemc::simplemc_exception if the
     * given size is <= 0.
     *
     * For static sized accumulators, the `num` parameter is ignored.
     *
     * @param num Number of elements.
     */
    explicit mean_acc(size_type num = 1) : mdata_(vec_type::Zero(is_dynamic ? num : static_size)), count_(0), idx_(0) {
        if constexpr (is_dynamic) {
            if (num <= 0) {
                throw simplemc_exception("Dynamic size <= 0", "mean_acc::mean_acc");
            }
        }
    }

    /**
     * @brief Construct a mean accumulator with a given data storage and count.
     *
     * @details For dynamically sized accumulators, the size of the data storage must be >= 1.
     * Otherwise, it throws a simplemc::simplemc_exception.
     *
     * @param md Accumulated mean data.
     * @param ct Number of accumulated values.
     */
    mean_acc(const vec_type& md, count_type ct) : mdata_(md), count_(ct), idx_(0) {
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
     * @brief Subscript operator sets the index and returns a reference to this object.
     *
     * @param idx Index.
     * @return Reference to this object.
     */
    mean_acc& operator[](size_type idx) {
        idx_ = idx;
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a single value.
     *
     * @details For accumulators with size > 1, the value is added to the element at the current
     * index.
     *
     * See @ref simplemc-accs-accs for a code example.
     *
     * @tparam T Type of the value to be accumulated.
     * @param val Value to be accumulated.
     * @return Reference to this object.
     */
    template <typename T>
        requires std::convertible_to<T, value_type>
    mean_acc& operator<<(const T& val) {
        ++count_;
        add_value(val, idx_, count_);
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a vector.
     *
     * @details See @ref simplemc-accs-accs for a code example.
     *
     * @tparam W Eigen vector/array/expression type.
     * @param vec Vector/Array/Expression to be accumulated.
     * @return Reference to this object.
     */
    template <typename W>
    mean_acc& operator<<(const W& vec) {
        assert(vec.size() == size());
        ++count_;
        if constexpr (varalg() == varalg::standard) {
            mdata_ += vec.matrix();
        } else {
            mdata_ += (vec.matrix() - mdata_) / static_cast<double>(count_);
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
     *     \mathbf{n}_{2}^{(N_2)} \f$ .
     *
     * @param acc simplemc::mean_acc to be incorporated.
     * @return Reference to this object.
     */
    mean_acc& operator<<(const mean_acc& acc) {
        assert(size() == acc.size());
        if constexpr (varalg() == varalg::standard) {
            mdata_ += acc.mdata_;
        } else {
            const auto n1 = static_cast<double>(count_);
            const auto n2 = static_cast<double>(acc.count_);
            mdata_ = mdata_ * n1 / (n1 + n2) + acc.mdata_ * n2 / (n1 + n2);
        }
        count_ += acc.count_;
        return *this;
    }

    /**
     * @brief Accumulate a range of values to consecutive elements in the accumulator.
     *
     * @details The values are added to consecutive elements in the accumulator starting with the
     * element at the given index. The size of the range is assumed to be <= size() - `idx`.
     *
     * See @ref simplemc-accs-accs for a code example.
     *
     * @tparam R Input range of values.
     * @param rg Range of values to be accumulated.
     * @param idx Starting index for the accumulator.
     */
    template <ranges::input_range R>
    void accumulate(R&& rg, size_type idx = 0) { // NOLINT (ranges need not be forwarded)
        ++count_;
        for (auto val : rg) {
            add_value(val, idx, count_);
            ++idx;
        }
    }

    /**
     * @brief Accumulate a range of values to arbitrary elements but with different indices.
     *
     * @details Each value of the given value range is accumulated at the corresponding index of the
     * given index range. Every index should only appear once.
     *
     * See @ref simplemc-accs-accs for a code example.
     *
     * @tparam R1 Input range of values.
     * @tparam R2 Input range of indices.
     * @param rg Range of values to be accumulated.
     * @param idxs Range of indices.
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
     * @return Multi-value accumulator.
     */
    mva_type make_mva() { return mva_type(*this); }

    /**
     * @brief Get the size of the accumulator.
     *
     * @return Number of elements.
     */
    [[nodiscard]] auto size() const { return mdata_.size(); }

    /**
     * @brief Get the total number of accumulated values.
     *
     * @return Number of accumulated values.
     */
    [[nodiscard]] auto count() const { return count_; }

    /**
     * @brief Get accumulated data used for estimating the mean.
     *
     * @return Data storage (content depends on the algorithm, see simplemc::accs::mean).
     */
    [[nodiscard]] const vec_type& mdata() const { return mdata_; }

    /**
     * @brief Calculate the sample mean from the accumulated data.
     *
     * @details Calls simplemc::accs::mean with the accumulated mean data and the count.
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a mean_acc::vec_type object.
     *
     * @return Sample mean.
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
     * It throws an exception, if the size of the accumulator is not equal on all processes.
     *
     * @param comm simplemc::mpi::communicator object.
     * @param acc Mean accumulator.
     * @return Mean accumulator with the reduced data from all processes.
     */
    friend mean_acc mpi_collect(const mpi::communicator& comm, const mean_acc& acc) {
        if (!all_equal(comm, acc.size())) {
            throw simplemc_exception("mean_acc size is not equal on all processes", "mpi_collect");
        }
        mean_acc res(acc.size());
        mpi::all_reduce(comm, acc.count_, res.count_, MPI_SUM);
        if constexpr (mean_acc::varalg() == varalg::standard) {
            mpi::all_reduce(comm, make_span(acc.mdata_), make_span(res.mdata_), MPI_SUM);
        } else {
            const auto n1 = static_cast<double>(acc.count_);
            const auto n = static_cast<double>(res.count_);
            const vec_type tmp_mdata = acc.mdata_ * n1 / n;
            mpi::all_reduce(comm, make_span(tmp_mdata), make_span(res.mdata_), MPI_SUM);
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
 * @tparam T Type of accumulated data.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <double_or_complex T, varalg A = varalg::welford>
using mean_acc_single = mean_acc<Eigen::Matrix<T, 1, 1>, A>;

/**
 * @brief Alias for a statically sized mean accumulator.
 *
 * @tparam T Type of accumulated data.
 * @tparam M Size of the accumulator.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <double_or_complex T, int M, varalg A = varalg::welford>
    requires(M >= 1)
using mean_acc_static = mean_acc<Eigen::Matrix<T, M, 1>, A>;

/**
 * @brief Alias for a dynamically sized mean accumulator.
 *
 * @tparam T Type of accumulated data.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <double_or_complex T, varalg A = varalg::welford>
using mean_acc_dynamic = mean_acc<Eigen::Matrix<T, Eigen::Dynamic, 1>, A>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_MEAN_ACC_HPP
