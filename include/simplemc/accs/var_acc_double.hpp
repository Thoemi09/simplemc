/**
 * @file var_acc_double.hpp
 * @brief Variance accumulator for calculating arithmetic means and variances.
 */

#ifndef SIMPLEMC_ACCS_VAR_ACC_DOUBLE_HPP
#define SIMPLEMC_ACCS_VAR_ACC_DOUBLE_HPP

#include <simplemc/accs/var_acc_fwd.hpp>
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
 * @brief Variance accumulator specialized for accumulating double values.
 *
 * @tparam A Algorithm used to calculate the mean/variance.
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
     * @brief Type for storing accumulated values.
     */
    using storage_type = Eigen::ArrayX<value_type>;

    /**
     * @brief Get the algorithm used to calculate the mean/variance.
     */
    static constexpr auto varalg() { return A; }

private:
    /**
     * @brief Multi value accumulator for the variance accumulator.
     *
     * @details It holds a reference to a variance accumulator. It can be used to add multiple data points
     * to the accumulator but only increase the count once (when it goes out of scope).
     */
    class var_mva {
    public:
        /**
         * @brief Construct a multi value accumulator for a given variance accumulator and a given index.
         *
         * @param acc Variance accumulator.
         * @param idx Index.
         */
        var_mva(var_acc& acc, size_type idx) : acc_(acc), idx_(idx) { assert(idx_ >= 0 && idx_ < acc_.size()); }

        /**
         * @brief Set index and return this object.
         *
         * @param idx Index.
         * @return Reference to this object.
         */
        var_mva& operator[](size_type idx) {
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
        var_mva& operator<<(value_type val) {
            if constexpr (varalg() == accs::varalg::standard) {
                acc_.data_[idx_] += (val - acc_.shift_[idx_]);
                acc_.data2_[idx_] += (val - acc_.shift_[idx_]) * (val - acc_.shift_[idx_]);
            } else {
                const auto tmp = val - acc_.shift_[idx_] - acc_.data_[idx_];
                acc_.data_[idx_] += tmp / static_cast<value_type>(acc_.count_ + 1);
                acc_.data2_[idx_] += tmp * (val - acc_.shift_[idx_] - acc_.data_[idx_]);
            }
            return *this;
        }

        /**
         * @brief Destructor increases the count.
         */
        ~var_mva() { acc_.count_ += 1; }

    private:
        var_acc& acc_; // NOLINT (reference is wanted here)
        size_type idx_;
    };

public:
    /**
     * @brief Construct a variance accumulator with a given number of elements and a constant shift.
     *
     * @param size Number of elements.
     * @param shift A single constant shift applied to the accumulated values.
     */
    explicit var_acc(size_type size = 1, value_type shift = 0.0) :
        data_(storage_type::Zero(size)),
        data2_(storage_type::Zero(size)),
        shift_(storage_type::Constant(size, shift)),
        count_(0),
        idx_(0) {
        if (size <= 0) {
            throw simplemc_exception("Size <= 0 in variance accumulator", "var_acc::var_acc");
        }
    }

    /**
     * @brief Construct a variance accumulator with a given constant shift.
     *
     * @param shift Constant shift applied to the accumulated values.
     */
    explicit var_acc(storage_type shift) :
        data_(storage_type::Zero(shift.size())),
        data2_(storage_type::Zero(shift.size())),
        shift_(std::move(shift)),
        count_(0),
        idx_(0) {
        if (data_.size() <= 0) {
            throw simplemc_exception("Size <= 0 in variance accumulator", "var_acc::var_acc");
        }
    }

    /**
     * @brief Construct a variance accumulator with given data storages, a count and constant shift.
     *
     * @param data Accumulated data.
     * @param data2 Accumulated squared data.
     * @param shift Constant shift applied to the accumulated values.
     * @param count Number of accumulated values.
     */
    var_acc(storage_type data, storage_type data2, storage_type shift, count_type count) :
        data_(std::move(data)),
        data2_(std::move(data2)),
        shift_(std::move(shift)),
        count_(count),
        idx_(0) {
        if (data_.size() == 0) {
            throw simplemc_exception("Size == 0 in variance accumulator", "var_acc::var_acc");
        }
        if (data_.size() != data2_.size() || data_.size() != shift_.size()) {
            throw simplemc_exception("Sizes of data storages do not match", "var_acc::var_acc");
        }
    }

    /**
     * @brief Subscript operator sets the index and returns a reference to this object.
     *
     * @param idx Index.
     * @return Reference to this object.
     */
    var_acc& operator[](size_type idx) {
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
    var_acc& operator<<(value_type val) {
        ++count_;
        if constexpr (varalg() == accs::varalg::standard) {
            data_[idx_] += (val - shift_[idx_]);
            data2_[idx_] += (val - shift_[idx_]) * (val - shift_[idx_]);
        } else {
            const auto tmp = val - shift_[idx_] - data_[idx_];
            data_[idx_] += tmp / static_cast<value_type>(count_);
            data2_[idx_] += tmp * (val - shift_[idx_] - data_[idx_]);
        }
        return *this;
    }

    /**
     * @brief Stream operator for incorporating the data from another variance accumulator.
     *
     * @param acc Variance accumulator to be incorporated.
     * @return Reference to this object.
     */
    var_acc& operator<<(const var_acc& acc) {
        assert(size() == acc.size());
        auto shifted_data = acc.data_ + acc.shift_ - shift_;
        if constexpr (varalg() == accs::varalg::standard) {
            data_ += shifted_data;
            data2_ += (acc.data2_ - 2.0 * (shift_ - acc.shift_) * (acc.data_ + acc.shift_) +
                acc.count_ * (shift_.square() - acc.shift_.square()));
        } else {
            const auto n1 = static_cast<value_type>(count_);
            const auto n2 = static_cast<value_type>(acc.count_);
            data2_ += acc.data2_ + (shifted_data - data_).square() * n1 * n2 / (n1 + n2);
            data_ = data_ * n1 / (n1 + n2) + (acc.data_ + acc.shift_ - shift_) * n2 / (n1 + n2);
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
                data_[idx] += (val - shift_[idx_]);
                data2_[idx] += (val - shift_[idx_]) * (val - shift_[idx_]);
            } else {
                const auto tmp = val - shift_[idx_] - data_[idx];
                data_[idx] += tmp / static_cast<value_type>(count_);
                data2_[idx] += tmp * (val - shift_[idx_] - data_[idx]);
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
    var_mva make_mva(size_type idx = 0) { return var_mva(*this, idx); }

    /**
     * @brief Get the size of the accumulator.
     *
     * @return Number of elements.
     */
    [[nodiscard]] auto size() const { return data_.size(); }

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
    [[nodiscard]] const storage_type& shift() const { return shift_; }

    /**
     * @brief Get accumulated data.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const storage_type& data() const { return data_; }

    /**
     * @brief Get accumulated squared data.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const storage_type& data2() const { return data2_; }

    /**
     * @brief Calculate the mean from the accumulated data.
     *
     * @return Data storage with mean values.
     */
    [[nodiscard]] storage_type mean() const {
        return simplemc::accs::mean<value_type, varalg()>(data_, count_, shift_);
    }

    /**
     * @brief Calculate the standard error from the accumulated data.
     *
     * @details The standard error \f$s_{\bar{X}}\f$ of the mean is calculated. It is related to the
     * variance of the mean by \f$s_{\bar{X}} = \sqrt{s^2_{\bar{X}}}\f$ and to the standard error of
     * the actual data by \f$s_{X} = s_{\bar{X}} * N\f$ where \f$N\f$ is the number of accumulated
     * values.
     *
     * @return Data storage with standard errors.
     */
    [[nodiscard]] storage_type stderror() const {
        return (simplemc::accs::variance<varalg()>(data_, data2_, count_) / static_cast<value_type>(count_)).sqrt();
    }

private:
    storage_type data_;
    storage_type data2_;
    storage_type shift_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_VAR_ACC_DOUBLE_HPP
