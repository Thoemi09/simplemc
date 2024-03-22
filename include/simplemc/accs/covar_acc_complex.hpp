/**
 * @file
 * @brief Covariance accumulator for complex values.
 */

#ifndef SIMPLEMC_ACCS_COVAR_ACC_COMPLEX_HPP
#define SIMPLEMC_ACCS_COVAR_ACC_COMPLEX_HPP

#include <simplemc/accs/covar_acc_fwd.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <range/v3/range/concepts.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/zip.hpp>

#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <utility>

namespace simplemc {

/**
 * @brief Covariance accumulator specialized for accumulating complex values.
 *
 * @tparam A Algorithm (either standard or welford).
 */
template <accs::varalg A>
class covar_acc<std::complex<double>, A> {
public:
    /**
     * @brief Type of accumulated values.
     */
    using value_type = std::complex<double>;

    /**
     * @brief Type for counting the number of accumulated values.
     */
    using count_type = std::uint64_t;

    /**
     * @brief Size type of the accumulator.
     */
    using size_type = long;

    /**
     * @brief Complex vector type.
     */
    using cplx_vec_type = Eigen::VectorX<value_type>;

    /**
     * @brief Double vector type.
     */
    using dbl_vec_type = Eigen::VectorX<double>;

    /**
     * @brief Complex matrix type.
     */
    using cplx_mat_type = Eigen::MatrixX<value_type>;

    /**
     * @brief Double matrix type.
     */
    using dbl_mat_type = Eigen::MatrixX<double>;

    /**
     * @brief Get the algorithm.
     */
    static constexpr auto varalg() { return A; }

private:
    /**
     * @brief Add a range of values to the accumulator without increasing the count.
     *
     * @details For performance reasons, we assume that the range of indices is sorted and each index is unique.
     *
     * @tparam R1 Input range of values.
     * @tparam R2 Input range of indices.
     * @param rg Range of values to be accumulated.
     * @param idxs Range of indices.
     * @param count Already increased count.
     */
    template <ranges::input_range R1, ranges::input_range R2>
    void add_values(R1&& rg, R2&& idxs, count_type count) { // NOLINT (ranges need not be forwarded)
        int drop = 1;
        if constexpr (varalg() == accs::varalg::standard) {
            for (auto [idx1, val1] : ranges::views::zip(idxs, rg)) {
                assert(idx1 >= 0 && idx1 < size());
                // mean data and diagonal elements of (cross-)covariance matrices
                const auto tmp = val1 - shift_(idx1);
                mdata_(idx1) += tmp;
                rdata_(idx1, idx1) += std::real(tmp) * std::real(tmp);
                idata_(idx1, idx1) += std::imag(tmp) * std::imag(tmp);
                cdata_(idx1, idx1) += std::real(tmp) * std::imag(tmp);
                // off-diagonal elements of (cross-)covariance matrices
                for (auto [idx2, val2] : ranges::views::drop(ranges::views::zip(idxs, rg), drop)) {
                    assert(idx2 > idx1 && idx2 < size());
                    // idx2 > idx1 for sorted indices -> only lower triangular matrix
                    const auto other_tmp = val2 - shift_(idx2);
                    rdata_(idx2, idx1) += std::real(tmp) * std::real(other_tmp);
                    idata_(idx2, idx1) += std::imag(tmp) * std::imag(other_tmp);
                    // idx1 is the index of the real part, idx2 is the index of the imaginary part
                    cdata_(idx1, idx2) += std::real(tmp) * std::imag(other_tmp);
                    cdata_(idx2, idx1) += std::imag(tmp) * std::real(other_tmp);
                }
                ++drop;
            }
        } else {
            for (auto [idx1, val1] : ranges::views::zip(idxs, rg)) {
                assert(idx1 >= 0 && idx1 < size());
                // mean data and diagonal elements of covariance matrices
                const auto tmp_old = val1 - shift_(idx1) - mdata_(idx1);
                mdata_(idx1) += tmp_old / static_cast<double>(count);
                const auto tmp = val1 - shift_(idx1) - mdata_(idx1);
                rdata_(idx1, idx1) += std::real(tmp_old) * std::real(tmp);
                idata_(idx1, idx1) += std::imag(tmp_old) * std::imag(tmp);
                cdata_(idx1, idx1) += std::real(tmp_old) * std::imag(tmp);
                // off-diagonal elements of (cross-)covariance matrices
                for (auto [idx2, val2] : ranges::views::drop(ranges::views::zip(idxs, rg), drop)) {
                    assert(idx2 > idx1 && idx2 < size());
                    // idx2 > idx1 for sorted indices -> only lower triangular matrix
                    const auto other_tmp = val2 - shift_(idx2) - mdata_(idx2);
                    rdata_(idx2, idx1) += std::real(tmp) * std::real(other_tmp);
                    idata_(idx2, idx1) += std::imag(tmp) * std::imag(other_tmp);
                    // idx1 is the index of the real part, idx2 is the index of the imaginary part
                    cdata_(idx1, idx2) += std::real(tmp) * std::imag(other_tmp);
                    cdata_(idx2, idx1) += std::imag(tmp) * std::real(other_tmp);
                }
                ++drop;
            }
        }
    }

public:
    /**
     * @brief Construct a covariance accumulator with a given number of elements and a constant shift.
     *
     * @param size Number of elements.
     * @param shift A single constant shift applied to the accumulated values.
     */
    explicit covar_acc(size_type size = 1, value_type shift = 0.0) :
        mdata_(cplx_vec_type::Zero(size)),
        rdata_(dbl_mat_type::Zero(size, size)),
        idata_(dbl_mat_type::Zero(size, size)),
        cdata_(dbl_mat_type::Zero(size, size)),
        shift_(cplx_vec_type::Constant(size, shift)),
        count_(0),
        idx_(0) {
        if (size <= 0) {
            throw simplemc_exception("Size <= 0", "covar_acc::covar_acc");
        }
    }

    /**
     * @brief Construct a covariance accumulator with a given constant shift.
     *
     * @details The size of the accumulator will be the same as the size of the shift vector.
     *
     * @param shift Constant shift applied to the accumulated values.
     */
    explicit covar_acc(cplx_vec_type shift) :
        mdata_(cplx_vec_type::Zero(shift.size())),
        rdata_(dbl_mat_type::Zero(shift.size(), shift.size())),
        idata_(dbl_mat_type::Zero(shift.size(), shift.size())),
        cdata_(dbl_mat_type::Zero(shift.size(), shift.size())),
        shift_(std::move(shift)),
        count_(0),
        idx_(0) {
        if (mdata_.size() <= 0) {
            throw simplemc_exception("Size <= 0", "covar_acc::covar_acc");
        }
    }

    /**
     * @brief Construct a covariance accumulator with given data storages, a constant shift and a count.
     *
     * @param mdata Accumulated mean data.
     * @param rdata Accumulated covariance data of the real part.
     * @param idata Accumulated covariance data of the imaginary part.
     * @param cdata Accumulated cross-covariance data between the real and imaginary part.
     * @param shift Constant shift applied to the accumulated values.
     * @param count Number of accumulated values.
     */
    covar_acc(cplx_vec_type mdata, dbl_mat_type rdata, dbl_mat_type idata, dbl_mat_type cdata, cplx_vec_type shift,
        count_type count) :
        mdata_(std::move(mdata)),
        rdata_(std::move(rdata)),
        idata_(std::move(idata)),
        cdata_(std::move(cdata)),
        shift_(std::move(shift)),
        count_(count),
        idx_(0) {
        if (mdata_.size() == 0) {
            throw simplemc_exception("Size == 0", "covar_acc::covar_acc");
        }
        if (mdata_.size() != rdata_.rows() || mdata_.size() != rdata_.cols() || mdata_.size() != idata_.rows() ||
            mdata_.size() != idata_.cols() || mdata_.size() != cdata_.rows() || mdata_.size() != cdata_.cols() ||
            mdata_.size() != shift_.size()) {
            throw simplemc_exception("Sizes of data storages do not match", "covar_acc::covar_acc");
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
     * @brief Subscript operator sets the index and returns a reference to this object.
     *
     * @param idx Index.
     * @return Reference to this object.
     */
    covar_acc& operator[](size_type idx) {
        idx_ = idx;
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a single value.
     *
     * @param val Value to be accumulated.
     * @return Reference to this object.
     */
    covar_acc& operator<<(value_type val) {
        ++count_;
        add_values(std::array<value_type, 1> { val }, std::array<size_type, 1> { idx_ }, count_);
        return *this;
    }

    /**
     * @brief Stream operator for incorporating the data from another covariance accumulator.
     *
     * @details We have to take care of the fact that the shift vectors might be different.
     *
     * @param acc Variance accumulator to be incorporated.
     * @return Reference to this object.
     */
    covar_acc& operator<<(const covar_acc& acc) {
        assert(size() == acc.size());
        if constexpr (varalg() == accs::varalg::standard) {
            const auto tmp = acc.shift_ - shift_;
            mdata_ += acc.mdata_ + acc.count_ * (acc.shift_ - shift_);
            rdata_ += (acc.rdata_ + acc.mdata_.real() * tmp.real().transpose() +
                tmp.real() * acc.mdata_.real().transpose() + acc.count_ * tmp.real() * tmp.real().transpose())
                          .template triangularView<Eigen::Lower>();
            idata_ += (acc.idata_ + acc.mdata_.imag() * tmp.imag().transpose() +
                tmp.imag() * acc.mdata_.imag().transpose() + acc.count_ * tmp.imag() * tmp.imag().transpose())
                          .template triangularView<Eigen::Lower>();
            ;
            cdata_ += acc.cdata_ + acc.mdata_.real() * tmp.imag().transpose() +
                tmp.real() * acc.mdata_.imag().transpose() + acc.count_ * tmp.real() * tmp.imag().transpose();
        } else {
            const auto n1 = static_cast<value_type>(count_);
            const auto n2 = static_cast<value_type>(acc.count_);
            const auto tmp = acc.mdata_ + acc.shift_ - shift_;
            const auto m = mdata_ * n1 / (n1 + n2) + tmp * n2 / (n1 + n2);
            rdata_ += (acc.rdata_ + n1 * (mdata_ - m).real() * (mdata_ - m).real().transpose() +
                n2 * (tmp - m).real() * (tmp - m).real().transpose())
                          .template triangularView<Eigen::Lower>();
            idata_ += (acc.idata_ + n1 * (mdata_ - m).imag() * (mdata_ - m).imag().transpose() +
                n2 * (tmp - m).imag() * (tmp - m).imag().transpose())
                          .template triangularView<Eigen::Lower>();
            cdata_ += acc.cdata_ + n1 * (mdata_ - m).real() * (mdata_ - m).imag().transpose() +
                n2 * (tmp - m).real() * (tmp - m).imag().transpose();
            mdata_ = m;
        }
        count_ += acc.count_;
        return *this;
    }

    /**
     * @brief Accumulate a range of values.
     *
     * @details The size of the range is assumed to be <= size() - idx.
     *
     * @tparam R Input range of values.
     * @param rg Range of values to be accumulated.
     * @param idx Starting index for the accumulator.
     */
    template <ranges::input_range R>
    void accumulate(R&& rg, size_type idx = 0) {
        ++count_;
        add_values(std::forward<R>(rg), ranges::views::iota(idx), count_);
    }

    /**
     * @brief Accumulate a range of values.
     *
     * @details The size of the range is assumed to be <= size() - idx. For performance reasonse,
     * we further assume that the range of indices is sorted and each index is unique.
     *
     * @tparam R1 Input range of values.
     * @tparam R2 Input range of indices.
     * @param rg Range of values to be accumulated.
     * @param idxs Range of indices.
     */
    template <ranges::input_range R1, ranges::input_range R2>
    void accumulate(R1&& rg, R2&& idxs) {
        ++count_;
        add_values(std::forward<R1>(rg), std::forward<R2>(idxs), count_);
    }

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
     * @brief Get the constant shift.
     *
     * @return Constant shift vector applied to accumulated values.
     */
    [[nodiscard]] const cplx_vec_type& shift() const { return shift_; }

    /**
     * @brief Get accumulated data used for estimating the mean.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const cplx_vec_type& mdata() const { return mdata_; }

    /**
     * @brief Get accumulated data used for estimating the covariance matrix of the real part.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const dbl_mat_type& rdata() const { return rdata_; }

    /**
     * @brief Get accumulated data used for estimating the covariance matrix of the imaginary part.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const dbl_mat_type& idata() const { return idata_; }

    /**
     * @brief Get accumulated data used for estimating the cross-covariance matrix of the real and
     * imaginary part.
     *
     * @details The real part is the row index and the imaginary part is the column index.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const dbl_mat_type& cdata() const { return cdata_; }

    /**
     * @brief Calculate the sample mean from the accumulated data.
     *
     * @return Sample mean.
     */
    [[nodiscard]] cplx_vec_type mean() const {
        return simplemc::accs::mean<value_type, varalg()>(mdata_, count_, shift_);
    }

    /**
     * @brief Calculate the sample covariance matrix of the mean.
     *
     * @return Sample covariance matrix of the mean.
     */
    [[nodiscard]] cplx_mat_type covariance() const {
        using namespace std::complex_literals;
        return covariance_of_data() / static_cast<value_type>(count_);
    }

    /**
     * @brief Calculate the sample covariance matrix of the accumulated data.
     *
     * @return Sample covariance matrix of the data.
     */
    [[nodiscard]] cplx_mat_type covariance_of_data() const {
        using namespace std::complex_literals;
        return covariance_of_real_data() + covariance_of_imag_data() +
            (0.0 + 1.0i) * (covariance_of_real_and_imag_data().transpose() - covariance_of_real_and_imag_data());
    }

    /**
     * @brief Calculate the sample covariance matrix of the real part of the accumulated data.
     *
     * @return Sample covariance matrix of the real part of the accumulated data.
     */
    [[nodiscard]] dbl_mat_type covariance_of_real_data() const {
        return simplemc::accs::covariance<varalg()>(
            mdata_.real(), mdata_.real(), rdata_.selfadjointView<Eigen::Lower>(), count_);
    }

    /**
     * @brief Calculate the sample covariance matrix of the imaginary part of the accumulated data.
     *
     * @return Sample covariance matrix of the imaginary part of the accumulated data.
     */
    [[nodiscard]] dbl_mat_type covariance_of_imag_data() const {
        return simplemc::accs::covariance<varalg()>(
            mdata_.imag(), mdata_.imag(), idata_.selfadjointView<Eigen::Lower>(), count_);
    }

    /**
     * @brief Calculate the sample cross-covariance matrix between the real and imaginary part of the
     * accumulated data.
     *
     * @return Sample cross-covariance matrix between the real and imaginary part of the accumulated data.
     */
    [[nodiscard]] dbl_mat_type covariance_of_real_and_imag_data() const {
        return simplemc::accs::covariance<varalg()>(mdata_.real(), mdata_.imag(), cdata_, count_);
    }

private:
    cplx_vec_type mdata_;
    dbl_mat_type rdata_;
    dbl_mat_type idata_;
    dbl_mat_type cdata_;
    cplx_vec_type shift_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_COVAR_ACC_COMPLEX_HPP
