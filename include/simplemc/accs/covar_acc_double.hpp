/**
 * @file covar_acc_double.hpp
 * @brief Covariance accumulator for double values.
 */

#ifndef SIMPLEMC_ACCS_COVAR_ACC_DOUBLE_HPP
#define SIMPLEMC_ACCS_COVAR_ACC_DOUBLE_HPP

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
#include <cstdint>
#include <utility>

namespace simplemc {

/**
 * @brief Covariance accumulator specialized for accumulating double values.
 *
 * @tparam A Algorithm (either standard or welford).
 */
template <accs::varalg A>
class covar_acc<double, A> {
public:
    /**
     * @brief Type of accumulated values.
     */
    using value_type = double;

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
    using vec_type = Eigen::VectorX<value_type>;

    /**
     * @brief Matrix type.
     */
    using mat_type = Eigen::MatrixX<value_type>;

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
                // mean data and diagonal elements of covariance matrix
                const auto tmp = val1 - shift_(idx1);
                mdata_(idx1) += tmp;
                cdata_(idx1, idx1) += tmp * tmp;
                // off-diagonal elements of covariance matrix
                for (auto [idx2, val2] : ranges::views::drop(ranges::views::zip(idxs, rg), drop)) {
                    assert(idx2 > idx1 && idx2 < size());
                    // idx2 > idx1 for sorted indices -> only lower triangular matrix
                    cdata_(idx2, idx1) += tmp * (val2 - shift_(idx2));
                }
                ++drop;
            }
        } else {
            for (auto [idx1, val1] : ranges::views::zip(idxs, rg)) {
                assert(idx1 >= 0 && idx1 < size());
                // mean data and diagonal elements of covariance matrix
                const auto tmp_old = val1 - shift_(idx1) - mdata_(idx1);
                mdata_(idx1) += tmp_old / static_cast<value_type>(count);
                const auto tmp = val1 - shift_(idx1) - mdata_(idx1);
                cdata_(idx1, idx1) += tmp_old * tmp;
                // off-diagonal elements of covariance matrix
                for (auto [idx2, val2] : ranges::views::drop(ranges::views::zip(idxs, rg), drop)) {
                    assert(idx2 > idx1 && idx2 < size());
                    // idx2 > idx1 for sorted indices -> only lower triangular matrix
                    cdata_(idx2, idx1) += tmp * (val2 - shift_(idx2) - mdata_(idx2));
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
        mdata_(vec_type::Zero(size)),
        cdata_(mat_type::Zero(size, size)),
        shift_(vec_type::Constant(size, shift)),
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
    explicit covar_acc(vec_type shift) :
        mdata_(vec_type::Zero(shift.size())),
        cdata_(mat_type::Zero(shift.size(), shift.size())),
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
     * @param cdata Accumulated covariance data.
     * @param shift Constant shift applied to the accumulated values.
     * @param count Number of accumulated values.
     */
    covar_acc(vec_type mdata, mat_type cdata, vec_type shift, count_type count) :
        mdata_(std::move(mdata)),
        cdata_(std::move(cdata)),
        shift_(std::move(shift)),
        count_(count),
        idx_(0) {
        if (mdata_.size() == 0) {
            throw simplemc_exception("Size == 0", "covar_acc::covar_acc");
        }
        if (mdata_.size() != cdata_.rows() || mdata_.size() != cdata_.cols() || mdata_.size() != shift_.size()) {
            throw simplemc_exception("Sizes of data storages do not match", "covar_acc::covar_acc");
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
            mdata_ += acc.mdata_ + acc.count_ * tmp;
            cdata_ += (acc.cdata_ + acc.mdata_ * tmp.transpose() + tmp * acc.mdata_.transpose() +
                acc.count_ * tmp * tmp.transpose())
                          .template triangularView<Eigen::Lower>();
        } else {
            const auto n1 = static_cast<value_type>(count_);
            const auto n2 = static_cast<value_type>(acc.count_);
            const auto tmp = acc.mdata_ + acc.shift_ - shift_;
            const auto m = mdata_ = mdata_ * n1 / (n1 + n2) + tmp * n2 / (n1 + n2);
            cdata_ +=
                (acc.cdata_ + n1 * (mdata_ - m) * (mdata_ - m).transpose() + n2 * (tmp - m) * (tmp - m).transpose())
                    .template triangularView<Eigen::Lower>();
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
    [[nodiscard]] const vec_type& shift() const { return shift_; }

    /**
     * @brief Get accumulated data used for estimating the mean.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const vec_type& mdata() const { return mdata_; }

    /**
     * @brief Get accumulated data used for estimating the covariance.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const mat_type& cdata() const { return cdata_; }

    /**
     * @brief Calculate the sample mean from the accumulated data.
     *
     * @return Sample mean.
     */
    [[nodiscard]] vec_type mean() const { return simplemc::accs::mean<value_type, varalg()>(mdata_, count_, shift_); }

    /**
     * @brief Calculate the sample covariance matrix of the mean.
     *
     * @return Sample covariance matrix of the mean.
     */
    [[nodiscard]] mat_type covariance() const {
        return simplemc::accs::covariance<varalg()>(mdata_, mdata_, cdata_.selfadjointView<Eigen::Lower>(), count_) /
            static_cast<value_type>(count_);
    }

    /**
     * @brief Calculate the sample covariance matrix of the accumulated data.
     *
     * @return Sample covariance matrix of the accumulated data.
     */
    [[nodiscard]] mat_type covariance_of_data() const {
        return simplemc::accs::covariance<varalg()>(mdata_, mdata_, cdata_.selfadjointView<Eigen::Lower>(), count_);
    }

private:
    vec_type mdata_;
    mat_type cdata_;
    vec_type shift_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_COVAR_ACC_DOUBLE_HPP
