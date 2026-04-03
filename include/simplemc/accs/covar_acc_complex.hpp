/**
 * @file
 * @brief Accumulator for calculating the sample mean and sample covariance matrix of a complex data
 * set.
 */

#ifndef SIMPLEMC_ACCS_COVAR_ACC_COMPLEX_HPP
#define SIMPLEMC_ACCS_COVAR_ACC_COMPLEX_HPP

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
#include <complex>
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace simplemc {

/**
 * @ingroup simplemc-accs-accs-covar
 * @brief Accumulator for calculating the sample mean and sample covariance matrix of a complex data
 * set.
 *
 * @details The accumulator satisfies simplemc::covariance_accumulator and takes two template
 * parameters:
 * - the type of the data samples (complex simplemc::sample_type) and
 * - the algorithm (simplemc::varalg) that should be used to accumulate the data.
 *
 * Both of them determine how the accumulation is actually done and what is stored in the accumulator.
 * The accumulated data is stored in four objects:
 * - a complex vector \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$ for the mean data,
 * - a real matrix \f$ \mathbf{C}_r^{(N)}/\mathbf{D}_r^{(N)} \f$ for the covariance data of the real
 * part of the complex samples,
 * - a real matrix \f$ \mathbf{C}_i^{(N)}/\mathbf{D}_i^{(N)} \f$ for the covariance data of the
 * imaginary part of the complex samples and
 * - a real matrix \f$ \mathbf{C}_{ri}^{(N)}/\mathbf{D}_{ri}^{(N)} \f$ for the cross-covariance data
 * between the real and imaginary parts.
 *
 * The mean data is accumulated as described in simplemc::mean_acc and the accumulation of the
 * covariance data follows @ref "simplemc::covar_acc< X, A >" "simplemc::covar_acc for real
 * samples".
 *
 * @include accs/doc_covar_acc_cplx.cpp
 *
 * Output:
 *
 * ```
 * Mean: [(5.00441-2.99916i), (10.00882-5.99832i)]
 * Covariance of real part:
 * 1.00475 2.00949
 * 2.00949 4.01898
 * Covariance of imaginary part:
 * 0.250734 0.501468
 * 0.501468  1.00294
 * ```
 *
 * @tparam Z Complex simplemc::sample_type.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <sample_type Z, varalg A>
    requires(std::same_as<Z, std::complex<double>> || eigen_vector_cplx<Z>)
class covar_acc<Z, A> {
public:
    /**
     * @brief Type of accumulated samples (complex simplemc::sample_type).
     */
    using sample_type = Z;

    /**
     * @brief Complex vector type for storing accumulated mean data.
     */
    using cplx_vec_type = std::conditional_t<sample_scalar<Z>, Eigen::Matrix<Z, 1, 1>, Z>;

    /**
     * @brief Underlying scalar type of accumulated samples.
     */
    using value_type = std::complex<double>;

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
    static constexpr int static_size = cplx_vec_type::RowsAtCompileTime;

    /**
     * @brief Is the accumulator dynamically sized?
     */
    static constexpr bool is_dynamic = (cplx_vec_type::RowsAtCompileTime == Eigen::Dynamic);

    /**
     * @brief Real matrix type for storing accumulated (cross-)covariance data.
     */
    using dbl_mat_type = Eigen::Matrix<double, static_size, static_size>;

    /**
     * @brief Get the algorithm used to accumulate the data.
     *
     * @return Its simplemc::varalg tag.
     */
    static constexpr auto varalg() noexcept { return A; }

private:
    // Real vector type for storing real/imaginary parts of the mean data.
    using dbl_vec_type = Eigen::Matrix<double, cplx_vec_type::RowsAtCompileTime, 1>;

    // Add values to the accumulator without increasing the count (the given count is assumed to be
    // already increased by one).
    // The range of indices is assumed to be sorted and that each index is unique.
    // Only the lower triangular part of the rdata_ and idata_ matrices are updated.
    template <ranges::input_range R1, ranges::input_range R2>
    void add_values(R1&& rg, R2&& idxs, count_type count) { // NOLINT (ranges need not be forwarded)
        int drop = 1;
        if constexpr (varalg() == varalg::standard) {
            for (auto [idx1, val1] : ranges::views::zip(idxs, rg)) {
                assert(idx1 >= 0 && idx1 < size());
                // mean data and diagonal elements of (cross-)covariance matrices
                mdata_(idx1) += val1;
                rdata_(idx1, idx1) += std::real(val1) * std::real(val1);
                idata_(idx1, idx1) += std::imag(val1) * std::imag(val1);
                cdata_(idx1, idx1) += std::real(val1) * std::imag(val1);
                // off-diagonal elements of (cross-)covariance matrices
                for (auto [idx2, val2] : ranges::views::drop(ranges::views::zip(idxs, rg), drop)) {
                    assert(idx2 > idx1 && idx2 < size());
                    // idx2 > idx1 for sorted indices -> only lower triangular matrix
                    rdata_(idx2, idx1) += std::real(val1) * std::real(val2);
                    idata_(idx2, idx1) += std::imag(val1) * std::imag(val2);
                    // the first index corresponds to the real part, the second to the imaginary part
                    cdata_(idx1, idx2) += std::real(val1) * std::imag(val2);
                    cdata_(idx2, idx1) += std::imag(val1) * std::real(val2);
                }
                ++drop;
            }
        } else {
            for (auto [idx1, val1] : ranges::views::zip(idxs, rg)) {
                assert(idx1 >= 0 && idx1 < size());
                // mean data and diagonal elements of covariance matrices
                const auto tmp_old = val1 - mdata_(idx1);
                mdata_(idx1) += tmp_old / static_cast<double>(count);
                const auto tmp = val1 - mdata_(idx1);
                rdata_(idx1, idx1) += std::real(tmp_old) * std::real(tmp);
                idata_(idx1, idx1) += std::imag(tmp_old) * std::imag(tmp);
                cdata_(idx1, idx1) += std::real(tmp_old) * std::imag(tmp);
                // off-diagonal elements of (cross-)covariance matrices
                for (auto [idx2, val2] : ranges::views::drop(ranges::views::zip(idxs, rg), drop)) {
                    assert(idx2 > idx1 && idx2 < size());
                    // idx2 > idx1 for sorted indices -> only lower triangular matrix
                    const auto other_tmp = val2 - mdata_(idx2);
                    rdata_(idx2, idx1) += std::real(tmp) * std::real(other_tmp);
                    idata_(idx2, idx1) += std::imag(tmp) * std::imag(other_tmp);
                    // the first index corresponds to the real part, the second to the imaginary part
                    cdata_(idx1, idx2) += std::real(tmp) * std::imag(other_tmp);
                    cdata_(idx2, idx1) += std::imag(tmp) * std::real(other_tmp);
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
        mdata_(cplx_vec_type::Zero(is_dynamic ? m : static_size)),
        rdata_(dbl_mat_type::Zero(is_dynamic ? m : static_size, is_dynamic ? m : static_size)),
        idata_(dbl_mat_type::Zero(is_dynamic ? m : static_size, is_dynamic ? m : static_size)),
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
     * @param rd Accumulated covariance data \f$ \mathbf{C}_r^{(N)}/\mathbf{D}_r^{(N)} \f$.
     * @param id Accumulated covariance data \f$ \mathbf{C}_i^{(N)}/\mathbf{D}_i^{(N)} \f$.
     * @param cd Accumulated cross-covariance data \f$ \mathbf{C}_{ri}^{(N)}/\mathbf{D}_{ri}^{(N)}
     * \f$.
     * @param n Number of accumulated samples \f$ N \f$.
     */
    covar_acc(
        const cplx_vec_type& md, const dbl_mat_type& rd, const dbl_mat_type& id, const dbl_mat_type& cd, count_type n) :
        mdata_(md),
        rdata_(rd),
        idata_(id),
        cdata_(cd),
        count_(n),
        idx_(0) {
        if constexpr (is_dynamic) {
            if (mdata_.size() <= 0) {
                throw simplemc_exception("Dynamic size <= 0");
            }
            if (mdata_.size() != rdata_.rows() || mdata_.size() != rdata_.cols() || mdata_.size() != idata_.rows() ||
                mdata_.size() != idata_.cols() || mdata_.size() != cdata_.rows() || mdata_.size() != cdata_.cols()) {
                throw simplemc_exception("Sizes of data storages do not match");
            }
        }
    }

    /**
     * @brief Reset the accumulator to its initial state, i.e. with no accumulated values.
     */
    void reset() {
        mdata_.setZero();
        rdata_.setZero();
        idata_.setZero();
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
     * @brief Stream operator for accumulating a single (scalar) value \f$ z \f$.
     *
     * @details For accumulators with size \f$ M > 1 \f$, the value is added to the element at the
     * current index \f$ i \f$ (see operator[]()).
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @tparam U Type of the value to be accumulated.
     * @param z Value \f$ z \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename U>
        requires std::convertible_to<U, value_type>
    covar_acc& operator<<(const U& z) {
        ++count_;
        add_values(std::array<value_type, 1> { z }, std::array<size_type, 1> { idx_ }, count_);
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
            rdata_ += (v.matrix().real() * v.matrix().real().transpose()).template triangularView<Eigen::Lower>();
            idata_ += (v.matrix().imag() * v.matrix().imag().transpose()).template triangularView<Eigen::Lower>();
            cdata_ += (v.matrix().real() * v.matrix().imag().transpose());
        } else {
            const auto tmp = (v.matrix() - mdata_).eval();
            mdata_ += tmp / static_cast<double>(count_);
            const auto tmp2 = v.matrix() - mdata_;
            rdata_ += (tmp.real() * tmp2.real().transpose()).template triangularView<Eigen::Lower>();
            idata_ += (tmp.imag() * tmp2.imag().transpose()).template triangularView<Eigen::Lower>();
            cdata_ += (tmp.real() * tmp2.imag().transpose());
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
     *   Here, \f$ \mathbf{C} \f$ stands for any of the accumulated covariance data, i.e. for \f$
     *   \mathbf{C}_r^{(N)} \f$, \f$ \mathbf{C}_i^{(N)} \f$ and for \f$ \mathbf{C}_{ri}^{(N)} \f$.
     *
     * - `welford`:
     *   - \f$ \mathbf{n}^{(N)} = \frac{N_1}{N} \mathbf{n}_{1}^{(N_1)} +
     *     \frac{N_2}{N} \mathbf{n}_{2}^{(N_2)} \f$ and
     *   - \f$ \mathbf{D}_r^{(N)} = \mathbf{D}_{r1}^{(N_1)} + \mathbf{D}_{r2}^{(N_2)} + N_1 \Re\left(
     *     \mathbf{n}_{1}^{(N_1)} - \mathbf{n}^{(N)} \right) \Re\left( \mathbf{n}_{1}^{(N_1)} -
     *     \mathbf{n}^{(N)} \right)^T + N_2 \Re\left( \mathbf{n}_{2}^{(N_2)} - \mathbf{n}^{(N)}
     *     \right) \Re\left( \mathbf{n}_{2}^{(N_2)} - \mathbf{n}^{(N)} \right)^T \f$ .
     *
     *   Similar expressions hold for the other accumulated covariance data, \f$ \mathbf{D}_i^{(N)}
     *   \f$ and \f$ \mathbf{D}_{ri}^{(N)} \f$.
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @param acc_other simplemc::covar_acc<Z, A> to be incorporated.
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
            rdata_ += acc_other.rdata_.template triangularView<Eigen::Lower>();
            idata_ += acc_other.idata_.template triangularView<Eigen::Lower>();
            cdata_ += acc_other.cdata_;
        } else {
            const auto n1 = static_cast<double>(count_);
            const auto n2 = static_cast<double>(acc_other.count_);
            const auto m = mdata_ * n1 / (n1 + n2) + acc_other.mdata_ * n2 / (n1 + n2);
            rdata_ += (acc_other.rdata_ + n1 * (mdata_ - m).real() * (mdata_ - m).real().transpose() +
                n2 * (acc_other.mdata_ - m).real() * (acc_other.mdata_ - m).real().transpose())
                          .template triangularView<Eigen::Lower>();
            idata_ += (acc_other.idata_ + n1 * (mdata_ - m).imag() * (mdata_ - m).imag().transpose() +
                n2 * (acc_other.mdata_ - m).imag() * (acc_other.mdata_ - m).imag().transpose())
                          .template triangularView<Eigen::Lower>();
            cdata_ += acc_other.cdata_ + n1 * (mdata_ - m).real() * (mdata_ - m).imag().transpose() +
                n2 * (acc_other.mdata_ - m).real() * (acc_other.mdata_ - m).imag().transpose();
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
     * @return simplemc::eigen_vector_cplx of size \f$ M \f$ containing \f$ \mathbf{m}^{(N)}/
     * \mathbf{n}^{(N)} \f$.
     */
    [[nodiscard]] const cplx_vec_type& mdata() const noexcept { return mdata_; }

    /**
     * @brief Get the accumulated covariance data \f$ \mathbf{C}_r^{(N)}/\mathbf{D}_r^{(N)} \f$ of
     * the real part of the complex samples.
     *
     * @note Only the lower triangular part of the covariance matrix is valid. Use
     * `selfadjointView<Eigen::Lower>()` to get the full covariance matrix.
     *
     * @return simplemc::eigen_matrix_dbl of size \f$ M \times M \f$ containing \f$ \mathbf{C}_r^{(N)}
     * /\mathbf{D}_r^{(N)} \f$.
     */
    [[nodiscard]] const dbl_mat_type& rdata() const noexcept { return rdata_; }

    /**
     * @brief Get the accumulated covariance data \f$ \mathbf{C}_i^{(N)}/\mathbf{D}_i^{(N)} \f$ of
     * the imaginary part of the complex samples.
     *
     * @note Only the lower triangular part of the covariance matrix is valid. Use
     * `selfadjointView<Eigen::Lower>()` to get the full covariance matrix.
     *
     * @return simplemc::eigen_matrix_dbl of size \f$ M \times M \f$ containing \f$ \mathbf{C}_i^{(N)}
     * /\mathbf{D}_i^{(N)} \f$.
     */
    [[nodiscard]] const dbl_mat_type& idata() const noexcept { return idata_; }

    /**
     * @brief Get the accumulated cross-covariance data \f$ \mathbf{C}_{ri}^{(N)}/
     * \mathbf{D}_{ri}^{(N)} \f$ between the real and imaginary parts of the complex samples.
     *
     * @details The real part is the row index and the imaginary part is the column index.
     *
     * @return simplemc::eigen_matrix_dbl of size \f$ M \times M \f$ containing \f$
     * \mathbf{C}_{ri}^{(N)}/\mathbf{D}_{ri}^{(N)} \f$.
     */
    [[nodiscard]] const dbl_mat_type& cdata() const noexcept { return cdata_; }

    /**
     * @brief Calculate the sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$.
     *
     * @details It calls simplemc::accs::mean with the accumulated mean data \f$ \mathbf{m}^{(N)}/
     * \mathbf{n}^{(N)} \f$ and the count \f$ N \f$.
     *
     * @return Sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$.
     */
    [[nodiscard]] auto mean() const {
        return detail::scalar_or_matrix<sample_scalar<sample_type>>(simplemc::accs::mean<varalg()>(mdata_, count_));
    }

    /**
     * @brief Calculate the sample variance of the mean \f$ s_{\overline{\mathbf{Z}}}^2 \f$.
     *
     * @details It returns the diagonal of covariance().
     *
     * @return Sample variance of the mean \f$ s_{\overline{\mathbf{Z}}}^2 \f$.
     */
    [[nodiscard]] auto variance() const {
        if constexpr (sample_scalar<sample_type>) {
            return covariance();
        } else {
            return covariance().diagonal().eval();
        }
    }

    /**
     * @brief Calculate the sample covariance matrix \f$ s_{\overline{\mathbf{Z}}
     * \overline{\mathbf{Z}}}^2 \f$ of the mean.
     *
     * @details It calls covariance_of_data() and divides the result by count().
     *
     * @return Sample covariance matrix of the mean \f$ s_{\overline{\mathbf{Z}}
     * \overline{\mathbf{Z}}}^2 \f$.
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
     * @brief Calculate the sample covariance matrix \f$ s_{\mathbf{Z} \mathbf{Z}}^2 \f$.
     *
     * @details It combines the different covariance matrices (real, imaginary, real-imaginary) to
     * form the full covariance matrix (see @ref simplemc-accs-stats-covar for the formula).
     *
     * @return Sample covariance matrix \f$ s_{\mathbf{Z} \mathbf{Z}}^2 \f$.
     */
    [[nodiscard]] auto covariance_of_data() const {
        using namespace std::complex_literals;
        if constexpr (sample_scalar<sample_type>) {
            auto res = covariance_of_real_data() + covariance_of_imag_data();
            return res;
        } else {
            auto res = covariance_of_real_data() + covariance_of_imag_data() +
                (0.0 + 1.0i) * (covariance_of_real_and_imag_data().transpose() - covariance_of_real_and_imag_data());
            return res.eval();
        }
    }

    /**
     * @brief Calculate the sample covariance matrix \f$ s_{\mathbf{X}\mathbf{X}}^2 \f$ of the real
     * part of the data set.
     *
     * @details It calls simplemc::accs::covariance with the real part of the accumulated mean data
     * \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$, the covariance data \f$ \mathbf{C}_r^{(N)}/
     * \mathbf{D}_r^{(N)} \f$ and the count \f$ N \f$.
     *
     * @return Sample covariance matrix of the real part of the data set, i.e. \f$ s_{\mathbf{X}
     * \mathbf{X}}^2 \f$.
     */
    [[nodiscard]] auto covariance_of_real_data() const {
        dbl_mat_type rdata_full = rdata_.template selfadjointView<Eigen::Lower>();
        dbl_vec_type mdata_r = mdata_.real();
        return detail::scalar_or_matrix<sample_scalar<sample_type>>(
            simplemc::accs::covariance<varalg()>(mdata_r, mdata_r, rdata_full, count_));
    }

    /**
     * @brief Calculate the sample covariance matrix \f$ s_{\mathbf{Y}\mathbf{Y}}^2 \f$ of the
     * imaginary part of the data set.
     *
     * @details It calls simplemc::accs::covariance with the imaginary part of the accumulated mean
     * data \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$, the covariance data \f$ \mathbf{C}_i^{(N)}/
     * \mathbf{D}_i^{(N)} \f$ and the count \f$ N \f$.
     *
     * @return Sample covariance matrix of the imaginary part of the data set, i.e. \f$
     * s_{\mathbf{Y}\mathbf{Y}}^2 \f$.
     */
    [[nodiscard]] auto covariance_of_imag_data() const {
        dbl_mat_type idata_full = idata_.template selfadjointView<Eigen::Lower>();
        dbl_vec_type mdata_i = mdata_.imag();
        return detail::scalar_or_matrix<sample_scalar<sample_type>>(
            simplemc::accs::covariance<varalg()>(mdata_i, mdata_i, idata_full, count_));
    }

    /**
     * @brief Calculate the sample cross-covariance matrix \f$ s_{\mathbf{X}\mathbf{Y}}^2 \f$
     * between the real and imaginary parts of the data set.
     *
     * @details It calls simplemc::accs::covariance with the real and imaginary parts of the
     * accumulated mean data \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$, the cross-covariance data
     * \f$ \mathbf{C}_{ri}^{(N)}/\mathbf{D}_{ri}^{(N)} \f$ and the count \f$ N \f$.
     *
     * @return Sample cross-covariance matrix between the real and imaginary parts of the data set,
     * i.e. \f$ s_{\mathbf{X}\mathbf{Y}}^2 \f$.
     */
    [[nodiscard]] auto covariance_of_real_and_imag_data() const {
        dbl_vec_type mdata_r = mdata_.real();
        dbl_vec_type mdata_i = mdata_.imag();
        return detail::scalar_or_matrix<sample_scalar<sample_type>>(
            simplemc::accs::covariance<varalg()>(mdata_r, mdata_i, cdata_, count_));
    }

    /**
     * @brief Collect covariance accumulators from different MPI processes.
     *
     * @details It constructs a new covariance accumulator with the reduced accumulated mean data,
     * (cross-)covariance data and counts from all MPI processes.
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
            mpi::all_reduce(make_span(acc.rdata_), make_span(res.rdata_), MPI_SUM, comm);
            mpi::all_reduce(make_span(acc.idata_), make_span(res.idata_), MPI_SUM, comm);
            mpi::all_reduce(make_span(acc.cdata_), make_span(res.cdata_), MPI_SUM, comm);
        } else {
            const auto n1 = static_cast<double>(acc.count_);
            const auto n = static_cast<double>(res.count_);
            const cplx_vec_type tmp_mdata = acc.mdata_ * n1 / n;
            mpi::all_reduce(make_span(tmp_mdata), make_span(res.mdata_), MPI_SUM, comm);
            const dbl_mat_type tmp_cdata =
                acc.cdata_ + n1 * (acc.mdata_ - res.mdata_).real() * (acc.mdata_ - res.mdata_).imag().transpose();
            mpi::all_reduce(make_span(tmp_cdata), make_span(res.cdata_), MPI_SUM, comm);
            // we cannot add the triangular view to the full matrix (only when we assign)
            dbl_mat_type tmp_rdata = acc.rdata_;
            tmp_rdata += (n1 * (acc.mdata_ - res.mdata_).real() * (acc.mdata_ - res.mdata_).real().transpose())
                             .template triangularView<Eigen::Lower>();
            mpi::all_reduce(make_span(tmp_rdata), make_span(res.rdata_), MPI_SUM, comm);
            dbl_mat_type tmp_idata = acc.idata_;
            tmp_idata += (n1 * (acc.mdata_ - res.mdata_).imag() * (acc.mdata_ - res.mdata_).imag().transpose())
                             .template triangularView<Eigen::Lower>();
            mpi::all_reduce(make_span(tmp_idata), make_span(res.idata_), MPI_SUM, comm);
        }

        return res;
    }

private:
    cplx_vec_type mdata_;
    dbl_mat_type rdata_;
    dbl_mat_type idata_;
    dbl_mat_type cdata_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_COVAR_ACC_COMPLEX_HPP
