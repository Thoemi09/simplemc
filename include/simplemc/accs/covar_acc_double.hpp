/**
 * @file covar_acc_double.hpp
 * @brief Accumulator for calculating the mean and covariance matrix for double values.
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

#include <cassert>
#include <cstdint>
#include <utility>

namespace simplemc {

/**
 * @brief Covariance accumulator specialized for accumulating double values.
 *
 * @tparam A Algorithm for accumulating data.
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
     * @brief 1-dimensional array type.
     */
    using array_type = Eigen::ArrayX<value_type>;

    /**
     * @brief 2-dimensional array type.
     */
    using array2d_type = Eigen::ArrayXX<value_type>;

    /**
     * @brief Get the algorithm for accumulating data.
     */
    static constexpr auto varalg() { return A; }

private:
    /**
     * @brief Multi value accumulator for the covariance accumulator.
     *
     * @details It holds a reference to a covariance accumulator. It can be used to add multiple data points
     * to the accumulator but only increase the count once (when it goes out of scope).
     */
    class covar_mva {
    public:
        /**
         * @brief Construct a multi value accumulator for a given covariance accumulator and a given index.
         *
         * @param acc Covariance accumulator.
         * @param idx Index.
         */
        covar_mva(covar_acc& acc, size_type idx) : acc_(acc), idx_(idx) { assert(idx_ >= 0 && idx_ < acc_.size()); }

        /**
         * @brief Set index and return this object.
         *
         * @param idx Index.
         * @return Reference to this object.
         */
        covar_mva& operator[](size_type idx) {
            assert(idx >= 0 && idx < acc_.size());
            idx_ = idx;
            return *this;
        }

        /**
         * @brief Accumulate data.
         *
         * @param val Value to be accumulated.
         * @return Reference to this object.
         */
        covar_mva& operator<<(value_type val) {
            if constexpr (varalg() == accs::varalg::standard) {
                const auto tmp = val - acc_.shift_[idx_];
                acc_.mdata_[idx_] += tmp;
                acc_.vdata_[idx_] += tmp * tmp;
            } else {
                const auto tmp = val - acc_.shift_[idx_] - acc_.mdata_[idx_];
                acc_.mdata_[idx_] += tmp / static_cast<value_type>(acc_.count_ + 1);
                acc_.vdata_[idx_] += tmp * (val - acc_.shift_[idx_] - acc_.mdata_[idx_]);
            }
            return *this;
        }

        /**
         * @brief Destructor increases the count.
         */
        ~covar_mva() { acc_.count_ += 1; }

    private:
        covar_acc& acc_; // NOLINT (reference is wanted here)
        size_type idx_;
    };

public:
    /**
     * @brief Construct a covariance accumulator with a given number of elements and a constant shift.
     *
     * @param size Number of elements.
     * @param shift A single constant shift applied to the accumulated values.
     */
    explicit covar_acc(size_type size = 1, value_type shift = 0.0) :
        mdata_(array_type::Zero(size)),
        cdata_(array2d_type::Zero(size, size)),
        shift_(array_type::Constant(size, shift)),
        count_(0),
        idx_(0) {
        if (size <= 0) {
            throw simplemc_exception("Size <= 0 in variance accumulator", "var_acc::var_acc");
        }
    }

    /**
     * @brief Construct a covariance accumulator with a given constant shift.
     *
     * @details The size of the accumulator will be the same as the size of the shift vector.
     *
     * @param shift Constant shift applied to the accumulated values.
     */
    explicit covar_acc(array_type shift) :
        mdata_(array_type::Zero(shift.size())),
        cdata_(array2d_type::Zero(shift.size(), shift.size())),
        shift_(std::move(shift)),
        count_(0),
        idx_(0) {
        if (mdata_.size() <= 0) {
            throw simplemc_exception("Size <= 0 in variance accumulator", "var_acc::var_acc");
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
    covar_acc(array_type mdata, array2d_type cdata, array_type shift, count_type count) :
        mdata_(std::move(mdata)),
        cdata_(std::move(cdata)),
        shift_(std::move(shift)),
        count_(count),
        idx_(0) {
        if (mdata_.size() == 0) {
            throw simplemc_exception("Size == 0 in variance accumulator", "var_acc::var_acc");
        }
        if (mdata_.size() != cdata_.rows() || mdata_.size() != cdata_.cols() || mdata_.size() != shift_.size()) {
            throw simplemc_exception("Sizes of data storages do not match", "var_acc::var_acc");
        }
    }

    /**
     * @brief Subscript operator sets the index and returns a reference to this object.
     *
     * @param idx Index.
     * @return Reference to this object.
     */
    covar_acc& operator[](size_type idx) {
        assert(idx >= 0 && idx < size());
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
        if constexpr (varalg() == accs::varalg::standard) {
            const auto tmp = val - shift_(idx_);
            mdata_(idx_) += tmp;
            cdata_(idx_, idx_) += tmp * tmp;
        } else {
            const auto tmp = val - shift_(idx_) - mdata_(idx_);
            mdata_(idx_) += tmp / static_cast<value_type>(count_);
            cdata_(idx_) += tmp * (val - shift_(idx_) - mdata_(idx_));
        }
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
    covar_acc& operator<<(const covar_acc& acc) {
        assert(size() == acc.size());
        if constexpr (varalg() == accs::varalg::standard) {
            mdata_ += acc.mdata_ + acc.count_ * (acc.shift_ - shift_);
            cdata_ +=
                acc.vdata_ + 2.0 * (acc.shift_ - shift_) * acc.mdata_ + acc.count_ * (acc.shift_ - shift_).square();
        } else {
            const auto n1 = static_cast<value_type>(count_);
            const auto n2 = static_cast<value_type>(acc.count_);
            cdata_ += acc.vdata_ + (acc.mdata_ + acc.shift_ - shift_ - mdata_).square() * n1 * n2 / (n1 + n2);
            mdata_ = mdata_ * n1 / (n1 + n2) + (acc.mdata_ + acc.shift_ - shift_) * n2 / (n1 + n2);
        }
        count_ += acc.count_;
        return *this;
    }

    /**
     * @brief Accumulate a range of values.
     *
     * @details The size of the range is assumed to be <= size() - idx.
     *
     * @tparam R Input range.
     * @param rg Range of values to be accumulated.
     * @param idx Starting index for the accumulator.
     */
    template <ranges::input_range R>
    void accumulate(R&& rg, size_type idx = 0) { // NOLINT (ranges need not be forwarded)
        assert(idx >= 0 && idx < size());
        ++count_;
        for (auto val : rg) {
            if constexpr (varalg() == accs::varalg::standard) {
                const auto tmp = val - shift_[idx];
                mdata_[idx] += tmp;
                cdata_[idx] += tmp * tmp;
            } else {
                const auto tmp = val - shift_[idx_] - mdata_[idx];
                mdata_[idx] += tmp / static_cast<value_type>(count_);
                cdata_[idx] += tmp * (val - shift_[idx_] - mdata_[idx]);
            }
            ++idx;
        }
    }

    /**
     * @brief Create a multi value accumulator for a given index.
     *
     * @param idx Index.
     * @return Multi value accumulator.
     */
    covar_mva make_mva(size_type idx = 0) { return covar_mva(*this, idx); }

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
    [[nodiscard]] const array_type& shift() const { return shift_; }

    /**
     * @brief Get accumulated data used for estimating the mean.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const array_type& mdata() const { return mdata_; }

    /**
     * @brief Get accumulated data used for estimating the covariance.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const array2d_type& cdata() const { return cdata_; }

    /**
     * @brief Calculate the sample mean from the accumulated data.
     *
     * @return Data storage with mean values.
     */
    [[nodiscard]] array_type mean() const { return simplemc::accs::mean<value_type, varalg()>(mdata_, count_, shift_); }

    /**
     * @brief Calculate the sample covariance matrix of the mean.
     *
     * @return Data storage with variances.
     */
    [[nodiscard]] array_type covariance() const {
        return simplemc::accs::diag_covariance<varalg()>(mdata_, mdata_, cdata_, count_) /
            static_cast<value_type>(count_);
    }

    /**
     * @brief Calculate the sample covariance matrix of the accumulated data.
     *
     * @return Data storage with covariances.
     */
    [[nodiscard]] array_type variance_of_data() const {
        return simplemc::accs::diag_covariance<varalg()>(mdata_, mdata_, cdata_, count_);
    }

private:
    array_type mdata_;
    array2d_type cdata_;
    array_type shift_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_COVAR_ACC_DOUBLE_HPP
