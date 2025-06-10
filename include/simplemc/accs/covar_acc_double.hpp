/**
 * @file
 * @brief Specialization of simplemc::covar_acc for real random vectors.
 */

#ifndef SIMPLEMC_ACCS_COVAR_ACC_DOUBLE_HPP
#define SIMPLEMC_ACCS_COVAR_ACC_DOUBLE_HPP

#include <simplemc/accs/covar_acc_fwd.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/mpi.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <Eigen/Dense>

#include <array>
#include <cassert>
#include <cstdint>
#include <utility>

namespace simplemc {

/**
 * @ingroup simplemc-accs-accs
 * @brief Specialization of simplemc::covar_acc for real random vectors \f$ \mathbf{X} \f$.
 *
 * @details The accumulated data is stored in two objects:
 * - a real vector \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$ for the mean data and
 * - a real matrix \f$ \mathbf{C}^{(N)}/\mathbf{D}^{(N)} \f$ for the covariance data.
 *
 * See simplemc::accs::mean and simplemc::accs::covariance.
 *
 * @code{.cpp}
 * #include <fmt/ranges.h>
 * #include <simplemc/accs.hpp>
 * #include <simplemc/utils/to_string.hpp>
 *
 * #include <random>
 *
 * int main() {
 *     // normal distribution to be sampled: mu = 2, sigma = 1
 *     std::mt19937_64 rng;
 *     std::normal_distribution<double> normal_dist(2.0, 1.0);
 *
 *     // accumulate samples into a covariance accumulator of size 2
 *     simplemc::covar_acc_dynamic<double> acc{2};
 *     for (int i = 0; i < 100000; ++i) {
 *         auto sample = normal_dist(rng);
 *         acc << Eigen::Vector2d{ sample, 2.0 * sample };
 *     }
 *
 *     // print the mean and covariance matrix of the accumulated data
 *     fmt::println("Mean: {}", acc.mean());
 *     fmt::println("Covariance:\n{}", simplemc::to_string(acc.covariance_of_data()));
 * }
 * @endcode
 *
 * Output:
 *
 * ```
 * Mean: [2.002072302074497, 4.004144604148994]
 * Covariance:
 * 1.00378 2.00756
 * 2.00756 4.01513
 * ```
 *
 * @tparam X simplemc::eigen_vector_dbl type.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <eigen_vector_dbl X, varalg A>
class covar_acc<X, A> {
public:
    /**
     * @brief Type of accumulated values.
     */
    using value_type = double;

    /**
     * @brief Static size of the accumulator.
     */
    static constexpr int static_size = X::RowsAtCompileTime;

    /**
     * @brief Is the accumulator dynamically sized?
     */
    static constexpr bool is_dynamic = (X::RowsAtCompileTime == Eigen::Dynamic);

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
    using vec_type = X;

    /**
     * @brief Matrix type.
     */
    using mat_type = Eigen::Matrix<value_type, static_size, static_size>;

    /**
     * @brief Get the algorithm used to accumulate the data.
     *
     * @return Its simplemc::varalg tag.
     */
    static constexpr auto varalg() { return A; }

private:
    // Add values to the accumulator without increasing the count (the given count is assumed to be
    // already increased by one).
    // The range of indices is assumed to be sorted and that each index is unique.
    // Only the lower triangular part of the covariance matrix is updated.
    template <ranges::input_range R1, ranges::input_range R2>
    void add_values(R1&& rg, R2&& idxs, count_type count) { // NOLINT (ranges need not be forwarded)
        int drop = 1;
        if constexpr (varalg() == varalg::standard) {
            for (auto [idx1, val1] : ranges::views::zip(idxs, rg)) {
                assert(idx1 >= 0 && idx1 < size());
                // mean data and diagonal elements of covariance matrix
                mdata_(idx1) += val1;
                cdata_(idx1, idx1) += val1 * val1;
                // off-diagonal elements of covariance matrix
                for (auto [idx2, val2] : ranges::views::drop(ranges::views::zip(idxs, rg), drop)) {
                    assert(idx2 > idx1 && idx2 < size());
                    // idx2 > idx1 for sorted indices -> only lower triangular matrix
                    cdata_(idx2, idx1) += val1 * val2;
                }
                ++drop;
            }
        } else {
            for (auto [idx1, val1] : ranges::views::zip(idxs, rg)) {
                assert(idx1 >= 0 && idx1 < size());
                // mean data and diagonal elements of covariance matrix
                const auto tmp_old = val1 - mdata_(idx1);
                mdata_(idx1) += tmp_old / static_cast<double>(count);
                const auto tmp = val1 - mdata_(idx1);
                cdata_(idx1, idx1) += tmp_old * tmp;
                // off-diagonal elements of covariance matrix
                for (auto [idx2, val2] : ranges::views::drop(ranges::views::zip(idxs, rg), drop)) {
                    assert(idx2 > idx1 && idx2 < size());
                    // idx2 > idx1 for sorted indices -> only lower triangular matrix
                    // mdata_(idx2) has not been updated yet
                    cdata_(idx2, idx1) += tmp * (val2 - mdata_(idx2));
                }
                ++drop;
            }
        }
    }

public:
    /**
     * @brief Construct a covariance accumulator with a given number of elements \f$ M \f$.
     *
     * @details For dynamically sized accumulators, it throws a simplemc::simplemc_exception if the
     * given size \f$ M \leq 0 \f$.
     *
     * For statically sized accumulators, \f$ M \f$ is ignored.
     *
     * @param m Number of elements \f$ M \f$.
     */
    explicit covar_acc(size_type m = 1) :
        mdata_(vec_type::Zero(is_dynamic ? m : static_size)),
        cdata_(mat_type::Zero(is_dynamic ? m : static_size, is_dynamic ? m : static_size)),
        count_(0),
        idx_(0) {
        if constexpr (is_dynamic) {
            if (m <= 0) {
                throw simplemc_exception("Dynamic size <= 0", "covar_acc::covar_acc");
            }
        }
    }

    /**
     * @brief Construct a covariance accumulator with the given accumulated data and number of
     * samples \f$ N \f$.
     *
     * @details For dynamically sized accumulators, the size \f$ M \f$ of the data storages must be
     * \f$ \geq 1 \f$. Otherwise, it throws a simplemc::simplemc_exception.
     *
     * @param md Accumulated mean data \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$.
     * @param cd Accumulated covariance data \f$ \mathbf{C}^{(N)}/\mathbf{D}^{(N)} \f$.
     * @param n Number of accumulated samples \f$ N \f$.
     */
    covar_acc(const vec_type& md, const mat_type& cd, count_type n) : mdata_(md), cdata_(cd), count_(n), idx_(0) {
        if constexpr (is_dynamic) {
            if (mdata_.size() <= 0) {
                throw simplemc_exception("Dynamic size <= 0", "covar_acc::covar_acc");
            }
            if (mdata_.size() != cdata_.rows() || mdata_.size() != cdata_.cols()) {
                throw simplemc_exception("Sizes of data storages do not match", "covar_acc::covar_acc");
            }
        }
    }

    /**
     * @brief Reset the accumulator to its initial state, i.e. with no accumulated values.
     */
    void reset() {
        mdata_.setZero();
        cdata_.setZero();
        count_ = 0;
        idx_ = 0;
    }

    /**
     * @brief Subscript operator sets the index \f$ i \f$ and returns a reference to `this` object.
     *
     * @param i Index \f$ i \f$.
     * @return Reference to `this` object.
     */
    covar_acc& operator[](size_type i) {
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
    covar_acc& operator<<(const T& x) {
        ++count_;
        add_values(std::array<value_type, 1> { x }, std::array<size_type, 1> { idx_ }, count_);
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
    covar_acc& operator<<(const W& v) {
        assert(v.size() == size());
        ++count_;
        if constexpr (varalg() == varalg::standard) {
            mdata_ += v.matrix();
            cdata_ += (v.matrix() * v.matrix().transpose()).template triangularView<Eigen::Lower>();
        } else {
            const auto tmp = (v.matrix() - mdata_).eval();
            mdata_ += tmp / static_cast<double>(count_);
            cdata_ += (tmp * (v.matrix() - mdata_).transpose()).template triangularView<Eigen::Lower>();
        }
        return *this;
    }

    /**
     * @brief Stream operator for incorporating the data from another covariance accumulator.
     *
     * @details Let the subscripts 1 and 2 correspond to the two accumulators we want to combine such
     * that \f$ N = N_1 + N_2 \f$ is the total number of accumulated values. Then, depending on the
     * simplemc::varalg, the combined accumulated data can be calculated as follows:
     * - `standard`:
     *   - \f$ \mathbf{m}^{(N)} = \mathbf{m}_{1}^{(N_1)} + \mathbf{m}_{2}^{(N_2)} \f$ and
     *   - \f$ \mathbf{C}^{(N)} = \mathbf{C}_{1}^{(N_1)} + \mathbf{C}_{2}^{(N_2)} \f$ .
     *
     * - `welford`:
     *   - \f$ \mathbf{n}^{(N)} = \frac{N_1}{N} \mathbf{n}_{1}^{(N_1)} +
     *     \frac{N_2}{N} \mathbf{n}_{2}^{(N_2)} \f$ and
     *   - \f$ \mathbf{D}^{(N)} = \mathbf{D}_{1}^{(N_1)} + \mathbf{D}_{2}^{(N_2)} + N_1 \left(
     *     \mathbf{n}_{1}^{(N_1)} - \mathbf{n}^{(N)} \right) \left( \mathbf{n}_{1}^{(N_1)} -
     *     \mathbf{n}^{(N)} \right)^T + N_2 \left( \mathbf{n}_{2}^{(N_2)} - \mathbf{n}^{(N)} \right)
     *     \left( \mathbf{n}_{2}^{(N_2)} - \mathbf{n}^{(N)} \right)^T \f$.
     * 
     * See also @ref simplemc-accs-accs-how.
     *
     * @param acc_other simplemc::covar_acc<X, A> to be incorporated.
     * @return Reference to `this` object.
     */
    covar_acc& operator<<(const covar_acc& acc_other) {
        assert(size() == acc_other.size());
        if constexpr (varalg() == varalg::standard) {
            mdata_ += acc_other.mdata_;
            cdata_ += acc_other.cdata_.template triangularView<Eigen::Lower>();
        } else {
            const auto n1 = static_cast<double>(count_);
            const auto n2 = static_cast<double>(acc_other.count_);
            const auto m = mdata_ * n1 / (n1 + n2) + acc_other.mdata_ * n2 / (n1 + n2);
            cdata_ += (acc_other.cdata_ + n1 * (mdata_ - m) * (mdata_ - m).transpose() +
                n2 * (acc_other.mdata_ - m) * (acc_other.mdata_ - m).transpose())
                          .template triangularView<Eigen::Lower>();
            mdata_ = m;
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
    void accumulate(R&& rg, size_type i = 0) {
        ++count_;
        add_values(std::forward<R>(rg), ranges::views::iota(i), count_);
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
     * @note For performance reasonses, we assume that the range of indices is sorted.
     *
     * @tparam R1 Input range of values.
     * @tparam R2 Input range of indices.
     * @param rg Range of values to be accumulated.
     * @param idxs (Sorted) range of indices at which the values should be accumulated.
     */
    template <ranges::input_range R1, ranges::input_range R2>
    void accumulate(R1&& rg, R2&& idxs) {
        ++count_;
        add_values(std::forward<R1>(rg), std::forward<R2>(idxs), count_);
    }

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
     * @brief Get the accumulated data \f$ \mathbf{m}^{(N)}/ \mathbf{n}^{(N)} \f$ used for estimating 
     * the mean.
     *
     * @return simplemc::eigen_vector_dbl of size \f$ M \f$ containing \f$ \mathbf{m}^{(N)}/
     * \mathbf{n}^{(N)} \f$ (content depends on the algorithm, see simplemc::accs::mean).
     */
    [[nodiscard]] const vec_type& mdata() const { return mdata_; }

    /**
     * @brief Get the accumulated data \f$ \mathbf{C}^{(N)}/\mathbf{D}^{(N)} \f$ used for estimating 
     * the covariance.
     *
     * @note Only the lower triangular part of the covariance matrix is valid. Use
     * `selfadjointView<Eigen::Lower>()` to get the full covariance matrix.
     *
     * @return simplemc::eigen_matrix_dbl of size \f$ M \times M \f$ containing \f$ \mathbf{C}^{(N)}/
     * \mathbf{D}^{(N)} \f$ (content depends on the algorithm, see simplemc::accs::covariance).
     */
    [[nodiscard]] const mat_type& cdata() const { return cdata_; }

    /**
     * @brief Calculate the sample mean \f$ \overline{\mathbf{x}}^{(N)} \f$.
     *
     * @details It calls simplemc::accs::mean with the accumulated mean data and the count.
     *
     * For statically sized accumulators with \f$ M = 1 \f$, it returns a real scalar. Otherwise, it
     * returns a vec_type object.
     *
     * @return Sample mean \f$ \overline{\mathbf{x}}^{(N)} \f$.
     */
    [[nodiscard]] auto mean() const {
        return detail::scalar_or_matrix<returns_scalar>(simplemc::accs::mean<varalg()>(mdata_, count_));
    }

    /**
     * @brief Calculate the sample covariance matrix \f$ s_{\overline{\mathbf{X}}
     * \overline{\mathbf{X}}}^2 \f$ of the mean.
     *
     * @details It calls covariance_of_data() and divides the result by count().
     *
     * For statically sized accumulators with \f$ M = 1 \f$, it returns a real scalar. Otherwise, it
     * returns a mat_type object.
     *
     * @return Sample covariance matrix of the mean \f$ s_{\overline{\mathbf{X}}
     * \overline{\mathbf{X}}}^2 \f$.
     */
    [[nodiscard]] auto covariance() const {
        auto res = covariance_of_data() / static_cast<double>(count_);
        if constexpr (returns_scalar) {
            return res;
        } else {
            return res.eval();
        }
    }

    /**
     * @brief Calculate the sample covariance matrix \f$ s_{\mathbf{X} \mathbf{X}}^2 \f$.
     *
     * @details It calls simplemc::accs::covariance with the accumulated data and the count.
     *
     * For statically sized accumulators with \f$ M = 1 \f$, it returns a real scalar. Otherwise, it
     * returns a mat_type object.
     *
     * @return Sample covariance matrix \f$ s_{\mathbf{X} \mathbf{X}}^2 \f$.
     */
    [[nodiscard]] auto covariance_of_data() const {
        using simplemc::accs::covariance;
        mat_type cdata_full = cdata_.template selfadjointView<Eigen::Lower>();
        return detail::scalar_or_matrix<returns_scalar>(covariance<varalg()>(mdata_, mdata_, cdata_full, count_));
    }

    /**
     * @brief Collect covariance accumulators from different MPI processes.
     *
     * @details It constructs a new covariance accumulator with the reduced accumulated mean data,
     * covariance data and counts from all MPI processes.
     *
     * The reduction operation depends on the simplemc::varalg algorithm used to accumulate the data.
     * See operator<<(const covar_acc&) for how it is done in the case of 2 accumulators.
     *
     * It throws an exception, if the size of the accumulator is not equal on all processes.
     *
     * @param comm simplemc::mpi::communicator object.
     * @param acc Covariance accumulator.
     * @return Covariance accumulator with the reduced data from all processes.
     */
    friend covar_acc mpi_collect(const mpi::communicator& comm, const covar_acc& acc) {
        if (!all_equal(comm, acc.size())) {
            throw simplemc_exception("covar_acc size is not equal on all processes", "mpi_collect");
        }
        covar_acc res(acc.size());
        mpi::all_reduce(comm, acc.count_, res.count_, MPI_SUM);
        if constexpr (covar_acc::varalg() == varalg::standard) {
            mpi::all_reduce(comm, make_span(acc.mdata_), make_span(res.mdata_), MPI_SUM);
            mpi::all_reduce(comm, make_span(acc.cdata_), make_span(res.cdata_), MPI_SUM);
        } else {
            const auto n1 = static_cast<double>(acc.count_);
            const auto n = static_cast<double>(res.count_);
            const vec_type tmp_mdata = acc.mdata_ * n1 / n;
            mpi::all_reduce(comm, make_span(tmp_mdata), make_span(res.mdata_), MPI_SUM);
            // we cannot add the triangular view to the full matrix (only when we assign)
            mat_type tmp_cdata = acc.cdata_;
            tmp_cdata += (n1 * (acc.mdata_ - res.mdata_) * (acc.mdata_ - res.mdata_).transpose())
                             .template triangularView<Eigen::Lower>();
            mpi::all_reduce(comm, make_span(tmp_cdata), make_span(res.cdata_), MPI_SUM);
        }
        return res;
    }

private:
    vec_type mdata_;
    mat_type cdata_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_COVAR_ACC_DOUBLE_HPP
