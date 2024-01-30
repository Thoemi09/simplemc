/**
 * @file var_acc.hpp
 * @brief Variance accumulator for calculating arithmetic means and variances.
 */

#ifndef SIMPLEMC_ACCS_VAR_ACC_HPP
#define SIMPLEMC_ACCS_VAR_ACC_HPP

#include <simplemc/accs/utils.hpp>

#include <range/v3/range/concepts.hpp>

#include <cassert>

namespace simplemc {

/**
 * @brief Variance accumulator for calculating arithmetic means and variances.
 * 
 * @tparam T Type of accumulated values (either double or std::complex<double>).
 * @tparam A Algorithm used to calculate the mean/variance.
 */
template <double_or_complex T, accs::varalg A = accs::varalg::standard>
class var_acc;

/**
 * @brief Variance accumulator specialized for accumulating double values.
 *
 * @details Use this accumulator for calculating means and variances/standard deviations.
 * No blocking/batching, covariance or estimation of the autocorrelation time is available. 
 *
 * If the size of the accumulator is 1, then values can be added with the stream operator:
 *
 *     acc << val;
 *
 * If the size of the accumulator is > 1 and only a single value should be added at once,
 * then one can use the subscript operator together with the stream operator:
 *
 *     acc[idx] << val;
 *
 * If the size of the accumulator is > 1 and multiple values should be added at once,
 * then one can use a multi value accumulator together with the stream operator:
 *
 *     {
 *         auto mva = acc.make_mva();
 *         mva[idx1] << val1;
 *         mva[idx2] << val2;
 *     }
 *
 * If a range of values should be added at once, one can use the accumulate function:
 *
 *     acc.accumulate(range, idx, count);
 *
 * The mean (stderror, tau) is always returned as an Eigen::ArrayX object. If e.g. the size of the
 * accumulator is 1, then we still need to access the array:
 *
 *     auto mean = acc.mean()[0];
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
     * @brief Type for counting the number of samples.
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
     * @brief Get algorithm used to calculate the mean/variance.
     */
    static constexpr auto varalg() { return A; }

private:
    /**
     * @brief Multi value accumulator for var_acc.
     *
     * @details It holds a reference to a variance accumulator. It can be used to add multiple data points
     * to the accumulator but only increase the count once (when it goes out of scope).
     */
    class var_mva {
    public:
        /**
         * @brief Construct a var_mva for a given var_acc and a given index.
         *
         * @param acc Variance accumulator associated with this var_mva.
         * @param idx Index.
         */
        var_mva(var_acc& acc, size_type idx) : acc_(acc), idx_(idx) { assert(idx_ >= 0 && idx_ < acc_.size()); }

        /**
         * @brief Set index and return this object.
         *
         * @param idx Index.
         * @return Reference to this object.
         */
        var_mva& operator[](std::size_t idx) {
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
                acc_.data_[idx_] += (val - acc_.shift_);
                acc_.data2_[idx_] += (val - acc_.shift_) * (val - acc_.shift_);
            } else {
                const auto tmp = acc_.data_[idx_];
                acc_.data_[idx_] += (val - acc_.shift_ - acc_.data_[idx_]) / static_cast<value_type>(acc_.count_ + 1);
                acc_.data2_[idx_] += (val - acc_.shift_ - tmp) * (val - acc_.shift_ - acc_.data_[idx_]);
            }
            return *this;
        }

        /**
         * @brief Destructor increases the count.
         */
        ~var_mva() { acc_.count_ += 1; }

    private:
        var_acc& acc_; // NOLINT (reference is wanted here)
        std::size_t idx_;
    };

public:
    /**
     * @brief Construct a variance accumulator with a given number of elements and a constant shift.
     *
     * @param size Number of elements.
     * @param shift Constant shift applied to the accumulated samples.
     */
    var_acc(size_type size = 1, value_type shift = 0.0);

    /**
     * @brief Construct a variance accumulator with given data storages, a count and constant shift.
     *
     * @param data Accumulated data.
     * @param data2 Accumulated squared data.
     * @param count Number of accumulated samples.
     * @param shift Constant shift applied to the accumulated samples.
     */
    var_acc(const storage_type& data, const storage_type& data2, count_type count, value_type shift = 0.0);

    /**
     * @brief Set the data storages, count and shift.
     *
     * @param data Accumulated data.
     * @param data2 Accumulated squared data.
     * @param count Number of accumulated samples.
     * @param shift Constant shift applied to the accumulated samples.
     */
    void set(const storage_type& data, const storage_type& data2, count_type count, value_type shift = 0.0);

    /**
     * @brief Reset the current accumulator by setting everything to zero and resizing the number of
     * elements if a size is specified.
     *
     * @param size New number of elements. If <= 0, no resizing is performed.
     * @param shift New constant shift. If it is not finite, the old value is kept.
     */
    void reset(size_type size = 0, value_type shift = std::numeric_limits<double>::infinity());

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
     * @brief Stream operator for accumulating single values.
     *
     * @details It adds the value to the current index and increases the count by one.
     *
     * @param val Value to be accumulated.
     * @return Reference to this object.
     */
    var_acc& operator<<(value_type val) {
        ++count_;
        if constexpr (varalg() == accs::varalg::standard) {
            data_[idx_] += (val - shift_);
            data2_[idx_] += (val - shift_) * (val - shift_);
        } else {
            const auto tmp = data_[idx_];
            data_[idx_] += (val - shift_ - data_[idx_]) / static_cast<value_type>(count_);
            data2_[idx_] += (val - shift_ - tmp) * (val - shift_ - data_[idx_]);
        }
        return *this;
    }

    /**
     * @brief Stream operator for incorporating the data from another var_acc.
     *
     * @details It simply adds the (reshifted) data and the count of the other accumulator to this one.
     *
     * @param acc Variance accumulator to be incorporated.
     * @return Reference to this object.
     */
    var_acc& operator<<(const var_acc& acc);

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
                data_[idx] += (val - shift_);
                data2_[idx] += (val - shift_) * (val - shift_);
            } else {
                const auto tmp = data_[idx];
                data_[idx] += (val - shift_ - data_[idx]) / static_cast<value_type>(count_);
                data2_[idx] += (val - shift_ - tmp) * (val - shift_ - data_[idx]);
            }
            ++idx;
        }
    }

    /**
     * @brief Create a multi value accumulator for a given index.
     *
     * @param idx Index.
     * @return Multi value accumulator var_mva.
     */
    var_mva make_mva(std::size_t idx = 0) { return var_mva(*this, idx); }

    /**
     * @brief Get the size of the accumulator.
     *
     * @return Number of elements.
     */
    [[nodiscard]] auto size() const { return data_.size(); }

    /**
     * @brief Get total number of samples.
     *
     * @return Number of accumulated data points.
     */
    [[nodiscard]] auto count() const { return count_; }

    /**
     * @brief Get constant shift.
     *
     * @return Constant shift applied to accumulated samples.
     */
    [[nodiscard]] auto shift() const { return shift_; }

    /**
     * @brief Get accumulated data.
     *
     * @return Data storage.
     */
    [[nodiscard]] const storage_type& data() const { return data_; }

    /**
     * @brief Get accumulated squared data.
     *
     * @return Data storage.
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

private:
    storage_type data_;
    storage_type data2_;
    count_type count_;
    size_type idx_;
    value_type shift_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_VAR_ACC_HPP
