/**
 * @file
 * @brief Variance accumulator for double values.
 */

#ifndef SIMPLEMC_ACCS_VAR_ACC_DOUBLE_HPP
#define SIMPLEMC_ACCS_VAR_ACC_DOUBLE_HPP

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
#include <cstdint>
#include <utility>

namespace simplemc {

/**
 * @brief Variance accumulator specialized for accumulating double values.
 *
 * @tparam A Algorithm (either standard or welford).
 */
template <accs::varalg A>
class var_acc<double, A> {
public:
    /**
     * @brief Type of accumulated value.
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
            vdata_(idx) += tmp * tmp;
        } else {
            const auto tmp = val - shift_(idx) - mdata_(idx);
            mdata_(idx) += tmp / static_cast<value_type>(count);
            vdata_(idx) += tmp * (val - shift_(idx) - mdata_(idx));
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
        mdata_(vec_type::Zero(size)),
        vdata_(vec_type::Zero(size)),
        shift_(vec_type::Constant(size, shift)),
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
    explicit var_acc(vec_type shift) :
        mdata_(vec_type::Zero(shift.size())),
        vdata_(vec_type::Zero(shift.size())),
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
     * @param vdata Accumulated variance data.
     * @param shift Constant shift applied to the accumulated values.
     * @param count Number of accumulated values.
     */
    var_acc(vec_type mdata, vec_type vdata, vec_type shift, count_type count) :
        mdata_(std::move(mdata)),
        vdata_(std::move(vdata)),
        shift_(std::move(shift)),
        count_(count),
        idx_(0) {
        if (mdata_.size() == 0) {
            throw simplemc_exception("Size == 0", "var_acc::var_acc");
        }
        if (mdata_.size() != vdata_.size() || mdata_.size() != shift_.size()) {
            throw simplemc_exception("Sizes of data storages do not match", "var_acc::var_acc");
        }
    }

    /**
     * @brief Reset the accumulator to its initial state, i.e. with no accumulated values.
     */
    void reset() {
        mdata_.setZero();
        vdata_.setZero();
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
            vdata_ += acc.vdata_ + 2.0 * tmp.cwiseProduct(acc.mdata_) + acc.count_ * tmp.cwiseProduct(tmp);
        } else {
            const auto n1 = static_cast<value_type>(count_);
            const auto n2 = static_cast<value_type>(acc.count_);
            const auto tmp = acc.shift_ - shift_ + acc.mdata_;
            const auto m = mdata_ * n1 / (n1 + n2) + tmp * n2 / (n1 + n2);
            vdata_ += acc.vdata_ + n1 * (mdata_ - m).cwiseProduct(mdata_ - m) + n2 * (tmp - m).cwiseProduct(tmp - m);
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
     * @brief Get accumulated data used for estimating the variance.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const vec_type& vdata() const { return vdata_; }

    /**
     * @brief Calculate the sample mean from the accumulated data.
     *
     * @return Sample mean.
     */
    [[nodiscard]] vec_type mean() const { return simplemc::accs::mean<value_type, varalg()>(mdata_, count_, shift_); }

    /**
     * @brief Calculate the sample variance of the mean.
     *
     * @return Diagonal of the sample covariance matrix of the mean.
     */
    [[nodiscard]] vec_type variance() const {
        return simplemc::accs::diag_covariance<varalg()>(mdata_, mdata_, vdata_, count_) /
            static_cast<value_type>(count_);
    }

    /**
     * @brief Calculate the sample variance of the accumulated data.
     *
     * @return Diagonal of the sample covariance matrix of the accumulated data.
     */
    [[nodiscard]] vec_type variance_of_data() const {
        return simplemc::accs::diag_covariance<varalg()>(mdata_, mdata_, vdata_, count_);
    }

private:
    vec_type mdata_;
    vec_type vdata_;
    vec_type shift_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_VAR_ACC_DOUBLE_HPP
