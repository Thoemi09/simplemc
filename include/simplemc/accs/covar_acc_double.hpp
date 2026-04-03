/**
 * @file
 * @brief Accumulator for calculating the sample mean and sample covariance matrix of a real data set.
 */

#ifndef SIMPLEMC_ACCS_COVAR_ACC_DOUBLE_HPP
#define SIMPLEMC_ACCS_COVAR_ACC_DOUBLE_HPP

#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/covar_acc_fwd.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/accs/varalg.hpp>
#include <simplemc/mpi.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <Eigen/Dense>

#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace simplemc {

/**
 * @ingroup simplemc-accs-accs-covar
 * @brief Accumulator for calculating the sample mean and sample covariance matrix of a real data set.
 *
 * @details The accumulator satisfies simplemc::covariance_accumulator and takes two template
 * parameters:
 * - the type of the data samples (real simplemc::sample_type) and
 * - the algorithm (simplemc::varalg) that should be used to accumulate the data.
 *
 * Both of them determine how the accumulation is actually done and what is stored in the accumulator.
 * The mean data \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$ is accumulated as described in
 * simplemc::mean_acc. The covariance data \f$ \mathbf{C}^{(N)}/\mathbf{D}^{(N)} \f$ is stored in a
 * single matrix and its content depends on the algorithm:
 * - `standard`: The covariance data is accumulated with
 *   \f[
 *     \mathbf{C}^{(N)} = \mathbf{C}^{(N-1)} + \left( \mathbf{x}^{(N)} - \mathbf{t} \right)
 *     \left( \mathbf{x}^{(N)} - \mathbf{t} \right)^T =
 *     \sum_{j=1}^N \left( \mathbf{x}^{(j)} - \mathbf{t} \right)
 *     \left( \mathbf{x}^{(j)} - \mathbf{t} \right)^T \; ,
 *   \f]
 *   such that the sample covariance matrix is given by
 *   \f[
 *     s_{\mathbf{X}\mathbf{X}}^2 = \frac{1}{N - 1} \left\{ \mathbf{C}^{(N)} -
 *     \frac{\mathbf{m}^{(N)} \left( \mathbf{m}^{(N)} \right)^T}{N} \right\} \; .
 *   \f]
 *
 * - `welford`: The covariance data is accumulated with
 *   \f[
 *     \mathbf{D}^{(N)} = \mathbf{D}^{(N-1)} + \left( \mathbf{x}^{(N)} - \mathbf{t} -
 *     \mathbf{n}^{(N-1)} \right) \left( \mathbf{x}^{(N)} - \mathbf{t} -
 *     \mathbf{n}^{(N)} \right)^T \; ,
 *   \f]
 *   such that the sample covariance matrix is given by
 *   \f[
 *     s_{\mathbf{X}\mathbf{X}}^2 = \frac{\mathbf{D}^{(N)}}{N - 1} \; .
 *   \f]
 *
 * Here, \f$ \mathbf{t} \f$ is a constant vector that can optionally be applied to the data samples to
 * increase numerical accuracy. See also @ref simplemc-accs-stats-covar.
 *
 * @include accs/doc_covar_acc_dbl.cpp
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
 * @tparam X Real simplemc::sample_type.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <sample_type X, varalg A>
    requires(std::same_as<X, double> || eigen_vector_dbl<X>)
class covar_acc<X, A> {
public:
    /**
     * @brief Type of accumulated samples (real simplemc::sample_type).
     */
    using sample_type = X;

    /**
     * @brief Real vector type for storing accumulated mean data.
     */
    using dbl_vec_type = std::conditional_t<sample_scalar<X>, Eigen::Matrix<X, 1, 1>, X>;

    /**
     * @brief Underlying scalar type of accumulated samples.
     */
    using value_type = double;

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
    static constexpr int static_size = dbl_vec_type::RowsAtCompileTime;

    /**
     * @brief Is the accumulator dynamically sized?
     */
    static constexpr bool is_dynamic = (dbl_vec_type::RowsAtCompileTime == Eigen::Dynamic);

    /**
     * @brief Real matrix type for storing accumulated covariance data.
     */
    using dbl_mat_type = Eigen::Matrix<value_type, static_size, static_size>;

    /**
     * @brief Get the algorithm used to accumulate the data.
     *
     * @return Its simplemc::varalg tag.
     */
    static constexpr auto varalg() noexcept { return A; }

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
        mdata_(dbl_vec_type::Zero(is_dynamic ? m : static_size)),
        cdata_(dbl_mat_type::Zero(is_dynamic ? m : static_size, is_dynamic ? m : static_size)),
        count_(0),
        idx_(0) {
        if constexpr (is_dynamic) {
            if (m <= 0) {
                throw simplemc_exception("Dynamic size <= 0");
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
    covar_acc(const dbl_vec_type& md, const dbl_mat_type& cd, count_type n) :
        mdata_(md),
        cdata_(cd),
        count_(n),
        idx_(0) {
        if constexpr (is_dynamic) {
            if (mdata_.size() <= 0) {
                throw simplemc_exception("Dynamic size <= 0");
            }
            if (mdata_.size() != cdata_.rows() || mdata_.size() != cdata_.cols()) {
                throw simplemc_exception("Sizes of data storages do not match");
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
     * @details The index is *sticky*: it persists until changed by another call to operator[]() or
     * until reset() is called. For scalar accumulators (size \f$ M = 1 \f$), the index should
     * remain at 0.
     *
     * @param i Index \f$ i \f$.
     * @return Reference to `this` object.
     */
    covar_acc& operator[](size_type i) noexcept {
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
    covar_acc& operator<<(const U& x) {
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

        // early return if the other accumulator is empty
        if (acc_other.empty()) {
            return *this;
        }

        // incorporate the data and count
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
     * @note For performance reasons, we assume that the range of indices is sorted.
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
     * @return simplemc::eigen_vector_dbl of size \f$ M \f$ containing \f$ \mathbf{m}^{(N)}/
     * \mathbf{n}^{(N)} \f$.
     */
    [[nodiscard]] const dbl_vec_type& mdata() const noexcept { return mdata_; }

    /**
     * @brief Get the accumulated covariance data \f$ \mathbf{C}^{(N)}/\mathbf{D}^{(N)} \f$.
     *
     * @note Only the lower triangular part of the covariance matrix is valid. Use
     * `selfadjointView<Eigen::Lower>()` to get the full covariance matrix.
     *
     * @return simplemc::eigen_matrix_dbl of size \f$ M \times M \f$ containing \f$ \mathbf{C}^{(N)}/
     * \mathbf{D}^{(N)} \f$.
     */
    [[nodiscard]] const dbl_mat_type& cdata() const noexcept { return cdata_; }

    /**
     * @brief Calculate the sample mean \f$ \overline{\mathbf{x}}^{(N)} \f$.
     *
     * @details It calls simplemc::accs::mean with the accumulated mean data \f$ \mathbf{m}^{(N)}/
     * \mathbf{n}^{(N)} \f$ and the count \f$ N \f$.
     *
     * @return Sample mean \f$ \overline{\mathbf{x}}^{(N)} \f$.
     */
    [[nodiscard]] auto mean() const {
        return detail::scalar_or_matrix<sample_scalar<sample_type>>(simplemc::accs::mean<varalg()>(mdata_, count_));
    }

    /**
     * @brief Calculate the sample covariance matrix \f$ s_{\overline{\mathbf{X}}
     * \overline{\mathbf{X}}}^2 \f$ of the mean.
     *
     * @details It calls covariance_of_data() and divides the result by count().
     *
     * @return Sample covariance matrix of the mean \f$ s_{\overline{\mathbf{X}}
     * \overline{\mathbf{X}}}^2 \f$.
     */
    [[nodiscard]] auto covariance() const {
        auto res = covariance_of_data() / static_cast<double>(count_);
        if constexpr (sample_scalar<sample_type>) {
            return res;
        } else {
            return res.eval();
        }
    }

    /**
     * @brief Calculate the sample covariance matrix \f$ s_{\mathbf{X} \mathbf{X}}^2 \f$.
     *
     * @details It calls simplemc::accs::covariance with the accumulated mean data \f$
     * \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$, the covariance data \f$ \mathbf{C}^{(N)}/
     * \mathbf{D}^{(N)} \f$ and the count \f$ N \f$.
     *
     * @return Sample covariance matrix \f$ s_{\mathbf{X} \mathbf{X}}^2 \f$.
     */
    [[nodiscard]] auto covariance_of_data() const {
        dbl_mat_type cdata_full = cdata_.template selfadjointView<Eigen::Lower>();
        return detail::scalar_or_matrix<sample_scalar<sample_type>>(
            simplemc::accs::covariance<varalg()>(mdata_, mdata_, cdata_full, count_));
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
     * It asserts that the size of the accumulator is equal on all processes.
     *
     * @param comm simplemc::mpi::communicator object.
     * @param acc Covariance accumulator.
     * @return Covariance accumulator with the reduced data from all processes.
     */
    friend covar_acc mpi_collect(const mpi::communicator& comm, const covar_acc& acc) {
        assert(all_equal(acc.size(), comm));
        covar_acc res(acc.size());

        // reduce the count
        mpi::all_reduce(acc.count_, res.count_, MPI_SUM, comm);

        // early return if the reduced count is zero
        if (res.count_ == 0) {
            return res;
        }

        // reduce the accumulated data
        if constexpr (covar_acc::varalg() == varalg::standard) {
            mpi::all_reduce(make_span(acc.mdata_), make_span(res.mdata_), MPI_SUM, comm);
            mpi::all_reduce(make_span(acc.cdata_), make_span(res.cdata_), MPI_SUM, comm);
        } else {
            const auto n1 = static_cast<double>(acc.count_);
            const auto n = static_cast<double>(res.count_);
            const dbl_vec_type tmp_mdata = acc.mdata_ * n1 / n;
            mpi::all_reduce(make_span(tmp_mdata), make_span(res.mdata_), MPI_SUM, comm);
            // we cannot add the triangular view to the full matrix (only when we assign)
            dbl_mat_type tmp_cdata = acc.cdata_;
            tmp_cdata += (n1 * (acc.mdata_ - res.mdata_) * (acc.mdata_ - res.mdata_).transpose())
                             .template triangularView<Eigen::Lower>();
            mpi::all_reduce(make_span(tmp_cdata), make_span(res.cdata_), MPI_SUM, comm);
        }

        return res;
    }

private:
    dbl_vec_type mdata_;
    dbl_mat_type cdata_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_COVAR_ACC_DOUBLE_HPP
