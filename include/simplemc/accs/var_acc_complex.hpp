/**
 * @file
 * @brief Variance accumulator for complex values.
 */

#ifndef SIMPLEMC_ACCS_VAR_ACC_COMPLEX_HPP
#define SIMPLEMC_ACCS_VAR_ACC_COMPLEX_HPP

#include <simplemc/accs/multivalue_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/accs/var_acc_fwd.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <range/v3/range/concepts.hpp>
#include <range/v3/view/zip.hpp>

#include <cassert>
#include <complex>
#include <cstdint>
#include <utility>

namespace simplemc {

/**
 * @ingroup simplemc-accs
 * @brief Variance accumulator specialized for accumulating complex values.
 *
 * @details The variance of a complex random variable is determined by the variance of its real and
 * imaginary parts as well as their covariance (see https://en.wikipedia.org/wiki/Complex_random_vector).
 * Note that in the following we only talk about the covariance of the real and imaginary part of a
 * single random complex variable, not the full covariance matrix of a complex random vector. For
 * calulating the full covariance matrix, please use simplemc::covar_acc.
 *
 * @tparam A Algorithm (either standard or welford).
 */
template <accs::varalg A>
class var_acc<std::complex<double>, A> {
public:
    /**
     * @brief Type of accumulated value.
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
     * @brief Multi value accumulator type.
     */
    using mva_type = multivalue_acc<var_acc>;

    /**
     * @brief Get the algorithm.
     */
    static constexpr auto varalg() { return A; }

    /* Friend declarations. */
    friend class multivalue_acc<var_acc>;

private:
    /**
     * @brief Add a single value to the accumulator without increasing the count.
     *
     * @param val Value to be added.
     * @param idx Index.
     * @param count Already increased count.
     */
    void add_value(value_type val, size_type idx, count_type count) {
        assert(idx >= 0 && idx < size());
        if constexpr (varalg() == accs::varalg::standard) {
            const auto tmp = val - shift_(idx);
            mdata_(idx) += tmp;
            rdata_(idx) += std::real(tmp) * std::real(tmp);
            idata_(idx) += std::imag(tmp) * std::imag(tmp);
            cdata_(idx) += std::real(tmp) * std::imag(tmp);
        } else {
            const auto tmp = val - shift_(idx) - mdata_(idx);
            mdata_(idx) += tmp / static_cast<double>(count);
            const auto tmp2 = val - shift_(idx) - mdata_(idx);
            rdata_(idx) += std::real(tmp) * std::real(tmp2);
            idata_(idx) += std::imag(tmp) * std::imag(tmp2);
            cdata_(idx) += std::real(tmp) * std::imag(tmp2);
        }
    }

public:
    /**
     * @brief Construct a variance accumulator with a given number of elements and a constant shift.
     *
     * @param size Number of elements.
     * @param shift A single constant shift applied to the accumulated values.
     */
    explicit var_acc(size_type size = 1, value_type shift = 0.0) :
        mdata_(cplx_vec_type::Zero(size)),
        rdata_(dbl_vec_type::Zero(size)),
        idata_(dbl_vec_type::Zero(size)),
        cdata_(dbl_vec_type::Zero(size)),
        shift_(cplx_vec_type::Constant(size, shift)),
        count_(0),
        idx_(0) {
        if (size <= 0) {
            throw simplemc_exception("Size <= 0", "var_acc::var_acc");
        }
    }

    /**
     * @brief Construct a variance accumulator with a given constant shift.
     *
     * @details The size of the accumulator will be the same as the size of the shift vector.
     *
     * @param shift Constant shift applied to the accumulated values.
     */
    explicit var_acc(cplx_vec_type shift) :
        mdata_(cplx_vec_type::Zero(shift.size())),
        rdata_(dbl_vec_type::Zero(shift.size())),
        idata_(dbl_vec_type::Zero(shift.size())),
        cdata_(dbl_vec_type::Zero(shift.size())),
        shift_(std::move(shift)),
        count_(0),
        idx_(0) {
        if (mdata_.size() <= 0) {
            throw simplemc_exception("Size <= 0", "var_acc::var_acc");
        }
    }

    /**
     * @brief Construct a variance accumulator with given data storages, a constant shift and a count.
     *
     * @param mdata Accumulated mean data.
     * @param rdata Accumulated variance data of the real part.
     * @param idata Accumulated variance data of the imaginary part.
     * @param cdata Accumulated covariance data between the real and imaginary part.
     * @param shift Constant shift applied to the accumulated values.
     * @param count Number of accumulated values.
     */
    var_acc(cplx_vec_type mdata, dbl_vec_type rdata, dbl_vec_type idata, dbl_vec_type cdata, cplx_vec_type shift,
        count_type count) :
        mdata_(std::move(mdata)),
        rdata_(std::move(rdata)),
        idata_(std::move(idata)),
        cdata_(std::move(cdata)),
        shift_(std::move(shift)),
        count_(count),
        idx_(0) {
        if (mdata_.size() == 0) {
            throw simplemc_exception("Size == 0", "var_acc::var_acc");
        }
        if (mdata_.size() != rdata_.size() || mdata_.size() != idata_.size() || mdata_.size() != cdata_.size() ||
            mdata_.size() != shift_.size()) {
            throw simplemc_exception("Sizes of data storages do not match", "var_acc::var_acc");
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
    var_acc& operator[](size_type idx) {
        idx_ = idx;
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a single value.
     *
     * @param val Value to be accumulated.
     * @return Reference to this object.
     */
    var_acc& operator<<(value_type val) {
        ++count_;
        add_value(val, idx_, count_);
        return *this;
    }

    /**
     * @brief Stream operator for incorporating the data from another variance accumulator.
     *
     * @details We have to take care of the fact that the shift vectors might be different.
     *
     * @param acc Variance accumulator to be incorporated.
     * @return Reference to this object.
     */
    var_acc& operator<<(const var_acc& acc) {
        assert(size() == acc.size());
        if constexpr (varalg() == accs::varalg::standard) {
            const auto tmp = acc.shift_ - shift_;
            mdata_ += acc.mdata_ + acc.count_ * tmp;
            rdata_ += acc.rdata_ + 2.0 * tmp.real().cwiseProduct(acc.mdata_.real()) +
                acc.count_ * tmp.real().cwiseProduct(tmp.real());
            idata_ += acc.idata_ + 2.0 * tmp.imag().cwiseProduct(acc.mdata_.imag()) +
                acc.count_ * tmp.imag().cwiseProduct(tmp.imag());
            cdata_ += acc.cdata_ + tmp.imag().cwiseProduct(acc.mdata_.real()) +
                tmp.real().cwiseProduct(acc.mdata_.imag()) + acc.count_ * tmp.real().cwiseProduct(tmp.imag());
        } else {
            const auto n1 = static_cast<double>(count_);
            const auto n2 = static_cast<double>(acc.count_);
            const auto tmp = acc.shift_ - shift_ + acc.mdata_;
            const auto m = mdata_ * n1 / (n1 + n2) + tmp * n2 / (n1 + n2);
            rdata_ += acc.rdata_ + n1 * (mdata_ - m).real().cwiseProduct((mdata_ - m).real()) +
                n2 * (tmp - m).real().cwiseProduct((tmp - m).real());
            idata_ += acc.idata_ + n1 * (mdata_ - m).imag().cwiseProduct((mdata_ - m).imag()) +
                n2 * (tmp - m).imag().cwiseProduct((tmp - m).imag());
            cdata_ += acc.cdata_ + n1 * (mdata_ - m).real().cwiseProduct((mdata_ - m).imag()) +
                n2 * (tmp - m).real().cwiseProduct((tmp - m).imag());
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
    void accumulate(R&& rg, size_type idx = 0) { // NOLINT (ranges need not be forwarded)
        ++count_;
        for (auto val : rg) {
            add_value(val, idx, count_);
            ++idx;
        }
    }

    /**
     * @brief Accumulate a range of values.
     *
     * @details The size of the range is assumed to be <= size() - idx and every index should
     * only appear at most once.
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
     * @brief Create a multi value accumulator.
     *
     * @return Multi value accumulator.
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
     * @brief Get the constant shift.
     *
     * @return Constant shift applied to accumulated values.
     */
    [[nodiscard]] const cplx_vec_type& shift() const { return shift_; }

    /**
     * @brief Get accumulated data used for estimating the mean.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const cplx_vec_type& mdata() const { return mdata_; }

    /**
     * @brief Get accumulated data used for estimating the variance of the real part.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const dbl_vec_type& rdata() const { return rdata_; }

    /**
     * @brief Get accumulated data used for estimating the variance of the imaginary part.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const dbl_vec_type& idata() const { return idata_; }

    /**
     * @brief Get accumulated data used for estimating the cross-covariance between the real and imaginary parts.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const dbl_vec_type& cdata() const { return cdata_; }

    /**
     * @brief Calculate the sample mean from the accumulated data.
     *
     * @return Sample mean.
     */
    [[nodiscard]] cplx_vec_type mean() const {
        return simplemc::accs::mean<value_type, varalg()>(mdata_, count_, shift_);
    }

    /**
     * @brief Calculate the sample variance of the mean.
     *
     * @return Diagonal of the sample covariance matrix of the mean.
     */
    [[nodiscard]] dbl_vec_type variance() const {
        return (variance_of_real_data() + variance_of_imag_data()) / static_cast<value_type>(count_);
    }

    /**
     * @brief Calculate the sample variance of the accumulated data.
     *
     * @return Diagonal of the sample covariance matrix of the accumulated data.
     */
    [[nodiscard]] dbl_vec_type variance_of_data() const { return variance_of_real_data() + variance_of_imag_data(); }

    /**
     * @brief Calculate the sample variance of the real part of the accumulated data.
     *
     * @return Diagonal of the sample covariance matrix of the real part of the accumulated data.
     */
    [[nodiscard]] dbl_vec_type variance_of_real_data() const {
        return simplemc::accs::diag_covariance<varalg()>(mdata_.real(), mdata_.real(), rdata_, count_);
    }

    /**
     * @brief Calculate the sample variance of the imaginary part of the accumulated data.
     *
     * @return Diagonal of the sample covariance matrix of the imaginary part of the accumulated data.
     */
    [[nodiscard]] dbl_vec_type variance_of_imag_data() const {
        return simplemc::accs::diag_covariance<varalg()>(mdata_.imag(), mdata_.imag(), idata_, count_);
    }

    /**
     * @brief Calculate the sample cross-covariance between the real and imaginary part of the accumulated data.
     *
     * @return Diagonal of the sample cross-covariance matrix between the real and imaginary part of the
     * accumulated data.
     */
    [[nodiscard]] dbl_vec_type covariance_of_real_and_imag_data() const {
        return simplemc::accs::diag_covariance<varalg()>(mdata_.real(), mdata_.imag(), cdata_, count_);
    }

private:
    cplx_vec_type mdata_;
    dbl_vec_type rdata_;
    dbl_vec_type idata_;
    dbl_vec_type cdata_;
    cplx_vec_type shift_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_VAR_ACC_COMPLEX_HPP
