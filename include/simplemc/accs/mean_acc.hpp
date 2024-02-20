/**
 * @file mean_acc.hpp
 * @brief Mean accumulator for calculating arithmetic means.
 */

#ifndef SIMPLEMC_ACCS_MEAN_ACC_HPP
#define SIMPLEMC_ACCS_MEAN_ACC_HPP

#include <simplemc/accs/utils.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <range/v3/range/concepts.hpp>

#include <cassert>
#include <cstdint>

namespace simplemc {

/**
 * @brief Mean accumulator for calculating arithmetic means.
 *
 * @details No error estimation is available.
 *
 * The user can choose between the standard and the more stable Welford algorithm and whether
 * to apply a constant shift to the accumulated data. This can sometimes improve the numerical
 * accuracy.
 *
 * If the size of the accumulator is 1, then values can be added with the stream operator:
 * @code{.cpp}
 * acc << val;
 * @endcode
 *
 * If the size of the accumulator is > 1 and only a single value should be added at once,
 * then one can use the subscript operator together with the stream operator:
 * @code{.cpp}
 * acc[idx] << val;
 * @endcode
 *
 * If the size of the accumulator is > 1 and multiple values should be added at once,
 * then one can use a multi value accumulator together with the stream operator:
 * @code{.cpp}
 * {
 *     auto mva = acc.make_mva();
 *     mva[idx1] << val1;
 *     mva[idx2] << val2;
 * }
 * @endcode
 *
 * If a range of values should be added at once, one can use the accumulate function:
 * @code{.cpp}
 * acc.accumulate(range, idx, count);
 * @endcode
 *
 * Results (mean, stderror, tau) are always returned as an Eigen::ArrayX object. If e.g.
 * the size of the accumulator is 1, then we still need to access the array:
 * @code{.cpp}
 * auto mean = acc.mean()[0];
 * @endcode
 *
 * @tparam T Type of accumulated values (either double or std::complex<double>).
 * @tparam A Algorithm used to calculate the mean.
 */
template <double_or_complex T, accs::varalg A = accs::varalg::standard>
class mean_acc {
public:
    /**
     * @brief Type of accumulated value.
     */
    using value_type = T;

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
     * @brief Get the algorithm used to calculate the mean.
     */
    static constexpr auto varalg() { return A; }

private:
    /**
     * @brief Multi value accumulator for the mean accumulator.
     *
     * @details It holds a reference to a mean accumulator. It can be used to add multiple data points
     * to the accumulator but only increase the count once (when it goes out of scope).
     */
    class mean_mva {
    public:
        /**
         * @brief Construct a multi value accumulator for a given mean accumulator and a given index.
         *
         * @param acc Mean accumulator.
         * @param idx Index.
         */
        mean_mva(mean_acc& acc, size_type idx) : acc_(acc), idx_(idx) { assert(idx_ >= 0 && idx_ < acc_.size()); }

        /**
         * @brief Set index and return this object.
         *
         * @param idx Index.
         * @return Reference to this object.
         */
        mean_mva& operator[](size_type idx) {
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
        mean_mva& operator<<(value_type val) {
            if constexpr (varalg() == accs::varalg::standard) {
                acc_.data_[idx_] += (val - acc_.shift_[idx_]);
            } else {
                acc_.data_[idx_] +=
                    (val - acc_.shift_[idx_] - acc_.data_[idx_]) / static_cast<value_type>(acc_.count_ + 1);
            }
            return *this;
        }

        /**
         * @brief Destructor increases the count.
         */
        ~mean_mva() { acc_.count_ += 1; }

    private:
        mean_acc& acc_; // NOLINT (reference is wanted here)
        size_type idx_;
    };

public:
    /**
     * @brief Construct a mean accumulator with a given number of elements.
     *
     * @param size Number of elements.
     * @param shift A single constant shift applied to the accumulated values.
     */
    explicit mean_acc(size_type size = 1, value_type shift = 0.0) :
        data_(storage_type::Zero(size)),
        shift_(storage_type::Constant(size, shift)),
        count_(0),
        idx_(0) {
        if (size <= 0) {
            throw simplemc_exception("Size <= 0 in mean accumulator", "mean_acc::mean_acc");
        }
    }

    /**
     * @brief Construct a mean accumulator with a given constant shift.
     *
     * @param shift Constant shift applied to the accumulated values.
     */
    explicit mean_acc(storage_type shift) :
        data_(storage_type::Zero(shift.size())),
        shift_(std::move(shift)),
        count_(0),
        idx_(0) {
        if (data_.size() <= 0) {
            throw simplemc_exception("Size <= 0 in mean accumulator", "mean_acc::mean_acc");
        }
    }

    /**
     * @brief Construct a mean accumulator with a given data storage, a constant shift and a count.
     *
     * @param data Accumulated data.
     * @param shift Constant shift applied to the accumulated values.
     * @param count Number of accumulated values.
     */
    mean_acc(storage_type data, storage_type shift, count_type count) :
        data_(std::move(data)),
        shift_(std::move(shift)),
        count_(count),
        idx_(0) {
        if (data_.size() == 0) {
            throw simplemc_exception("Size == 0 in mean accumulator", "mean_acc::mean_acc");
        }
        if (data_.size() != shift_.size()) {
            throw simplemc_exception("Size of data != size of shift in mean accumulator", "mean_acc::mean_acc");
        }
    }

    /**
     * @brief Subscript operator sets the index and returns a reference to this object.
     *
     * @param idx Index.
     * @return Reference to this object.
     */
    mean_acc& operator[](size_type idx) {
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
    mean_acc& operator<<(value_type val) {
        ++count_;
        if constexpr (varalg() == accs::varalg::standard) {
            data_[idx_] += (val - shift_[idx_]);
        } else {
            data_[idx_] += (val - shift_[idx_] - data_[idx_]) / static_cast<value_type>(count_);
        }
        return *this;
    }

    /**
     * @brief Stream operator for incorporating the data from another mean accumulator.
     *
     * @param acc Mean accumulator to be incorporated.
     * @return Reference to this object.
     */
    mean_acc& operator<<(const mean_acc& acc) {
        assert(size() == acc.size());
        if constexpr (varalg() == accs::varalg::standard) {
            data_ += acc.data_ + acc.count_ * (acc.shift_ - shift_);
        } else {
            const auto n1 = static_cast<value_type>(count_);
            const auto n2 = static_cast<value_type>(acc.count_);
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
                data_[idx] += (val - shift_[idx]);
            } else {
                data_[idx] += (val - shift_[idx] - data_[idx]) / static_cast<value_type>(count_);
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
    mean_mva make_mva(size_type idx = 0) { return mean_mva(*this, idx); }

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
     * @return Constant shift applied to the accumulated values.
     */
    [[nodiscard]] const storage_type& shift() const { return shift_; }

    /**
     * @brief Get accumulated data.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const storage_type& data() const { return data_; }

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
    storage_type shift_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_MEAN_ACC_HPP
