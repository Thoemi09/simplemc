/**
 * @file var_acc_double.hpp
 * @brief Variance accumulator for calculating arithmetic means and variances for double values.
 */

#ifndef SIMPLEMC_ACCS_VAR_ACC_DOUBLE_HPP
#define SIMPLEMC_ACCS_VAR_ACC_DOUBLE_HPP

#include <simplemc/accs/utils.hpp>
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
 * @details Let \f$ \{ \mathbf{x}^{(i)}: i = 1,\dots,N \} \f$ be the set of random vectors to accumulate 
 * and let \f$ \mathbf{t} \f$ be the constant shift. We denote the data storage for the mean data by 
 * \f$ \mathbf{m}^{(N)} \f$ and for the variance data by \f$ \mathbf{v}^{(N)} \f$. The mean data and the 
 * sample mean are calculated as in simplemc::mean_acc and the calculated variance corresponds to the
 * diagonal of the sample covariance matrix:
 *
 * - Standard algorithm: The variance data stores \f$ \mathbf{v}^{(N)}_j = \sum_{i=1}^N (\mathbf{x}^{(i)}_j
 * - \mathbf{t}_j)^2 = \mathbf{v}^{(N-1)}_j + (\mathbf{x}^{(N)}_j - \mathbf{t}_j)^2 \f$. Then the sample 
 * variance is estimated with \f$ \mathbf{s}^2_{\mathbf{X}j} = \frac{1}{N - 1} \left( \mathbf{v}^{(N)}_j - 
 * \mathbf{m}^{(N)}_j * \mathbf{m}^{(N)}_j / N \right) \f$.
 *
 * - Welford algorithm: The variance data stores \f$ \mathbf{v}^{(N)}_j = \mathbf{v}^{(N-1)}_j + \left( 
 * \mathbf{x}^{(N)}_j - \mathbf{t}_j - \mathbf{m}^{(N)}_j \right) \left( \mathbf{x}^{(N)}_j - \mathbf{t}_j 
 * - \mathbf{m}^{(N-1)}_j \right) \f$. Then the sample variance is estimated with \f$ \mathbf{s}^2_{\mathbf{X}j} 
 * = \frac{1}{N - 1} \mathbf{v}^{(N)}_j \f$.
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
        mdata_(storage_type::Zero(size)),
        vdata_(storage_type::Zero(size)),
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
     * @details The size of the accumulator will be the same as the size of the shift vector.
     *
     * @param shift Constant shift applied to the accumulated values.
     */
    explicit var_acc(storage_type shift) :
        mdata_(storage_type::Zero(shift.size())),
        vdata_(storage_type::Zero(shift.size())),
        shift_(std::move(shift)),
        count_(0),
        idx_(0) {
        if (mdata_.size() <= 0) {
            throw simplemc_exception("Size <= 0 in variance accumulator", "var_acc::var_acc");
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
    var_acc(storage_type mdata, storage_type vdata, storage_type shift, count_type count) :
        mdata_(std::move(mdata)),
        vdata_(std::move(vdata)),
        shift_(std::move(shift)),
        count_(count),
        idx_(0) {
        if (mdata_.size() == 0) {
            throw simplemc_exception("Size == 0 in variance accumulator", "var_acc::var_acc");
        }
        if (mdata_.size() != vdata_.size() || mdata_.size() != shift_.size()) {
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
            const auto tmp = val - shift_[idx_];
            mdata_[idx_] += tmp;
            vdata_[idx_] += tmp * tmp;
        } else {
            const auto tmp = val - shift_[idx_] - mdata_[idx_];
            mdata_[idx_] += tmp / static_cast<value_type>(count_);
            vdata_[idx_] += tmp * (val - shift_[idx_] - mdata_[idx_]);
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
    var_acc& operator<<(const var_acc& acc) {
        assert(size() == acc.size());
        if constexpr (varalg() == accs::varalg::standard) {
            mdata_ += acc.mdata_ + acc.count_ * (acc.shift_ - shift_);
            vdata_ +=
                acc.vdata_ + 2.0 * (acc.shift_ - shift_) * acc.mdata_ + acc.count_ * (acc.shift_ - shift_).square();
        } else {
            const auto n1 = static_cast<value_type>(count_);
            const auto n2 = static_cast<value_type>(acc.count_);
            vdata_ += acc.vdata_ + (acc.mdata_ + acc.shift_ - shift_ - mdata_).square() * n1 * n2 / (n1 + n2);
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
                vdata_[idx] += tmp * tmp;
            } else {
                const auto tmp = val - shift_[idx_] - mdata_[idx];
                mdata_[idx] += tmp / static_cast<value_type>(count_);
                vdata_[idx] += tmp * (val - shift_[idx_] - mdata_[idx]);
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
    [[nodiscard]] const storage_type& shift() const { return shift_; }

    /**
     * @brief Get accumulated data used for estimating the mean.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const storage_type& mdata() const { return mdata_; }

    /**
     * @brief Get accumulated data used for estimating the variance.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const storage_type& vdata() const { return vdata_; }

    /**
     * @brief Calculate the sample mean from the accumulated data.
     *
     * @return Data storage with mean values.
     */
    [[nodiscard]] storage_type mean() const {
        return simplemc::accs::mean<value_type, varalg()>(mdata_, count_, shift_);
    }

    /**
     * @brief Calculate the diagonal of the sample covariance matrix of the mean.
     *
     * @details The sample variance of the mean \f$ \mathbf{s}^2_{\bar{\mathbf{X}}j} \f$ is related to the
     * standard error of the mean \f$ \mathbf{s}_{\bar{\mathbf{X}}j} = \sqrt{\mathbf{s}^2_{\mathbf{X}j}} \f$
     * and to the variance of the data \f$ \mathbf{s}^2_{\mathbf{X}j} = \mathbf{s}^2_{\bar{\mathbf{X}j}} * N \f$.
     *
     * @return Data storage with variances.
     */
    [[nodiscard]] storage_type variance() const {
        return simplemc::accs::diag_covariance<varalg()>(mdata_, mdata_, vdata_, count_) /
            static_cast<value_type>(count_);
    }

    /**
     * @brief Calculate the diagonal of the sample covariance matrix of the accumulated data.
     *
     * @details See var_acc::variance.
     *
     * @return Data storage with variances.
     */
    [[nodiscard]] storage_type variance_of_data() const {
        return simplemc::accs::diag_covariance<varalg()>(mdata_, mdata_, vdata_, count_);
    }

private:
    storage_type mdata_;
    storage_type vdata_;
    storage_type shift_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_VAR_ACC_DOUBLE_HPP
