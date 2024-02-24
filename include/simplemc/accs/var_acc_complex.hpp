/**
 * @file var_acc_complex.hpp
 * @brief Variance accumulator for calculating arithmetic means and variances for complex values.
 */

#ifndef SIMPLEMC_ACCS_VAR_ACC_COMPLEX_HPP
#define SIMPLEMC_ACCS_VAR_ACC_COMPLEX_HPP

#include <simplemc/accs/var_acc_fwd.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <range/v3/range/concepts.hpp>

#include <cassert>
#include <complex>
#include <cstdint>
#include <utility>

namespace simplemc {

/**
 * @brief Variance accumulator specialized for accumulating complex values.
 *
 * @details The variance of a complex random variable is determined by the variance of its real and
 * imaginary parts as well as their covariance (see https://en.wikipedia.org/wiki/Complex_random_vector).
 *
 * Let \f$ \{ \mathbf{z}_i = \mathbf{x}_i + i \mathbf{y}_i : i = 1,\dots,N \} \f$ be the set of complex
 * random vectors to accumulate and let \f$ \mathbf{t} = \mathbf{u} + i \mathbf{v} \f$ be the constant shift.
 * We denote the data storage for the mean data by \f$ \mathbf{m}_N \f$ and for the real variance, imaginary
 * variance and covariance data by \f$ \mathbf{r}_N \f$, \f$ \mathbf{j}_N \f$ and \f$ \mathbf{c}_N \f$,
 * respectively. The mean and variance of the real and imaginary parts are calculated as in simplemc::var_acc.
 * To get the covariance, we do the following:
 *
 * - Standard algorithm: The covariance data stores \f$ \mathbf{c}_N = \sum_{i=1}^N (\mathbf{x}_i -
 * \mathbf{u})(\mathbf{y}_i - \mathbf{v}) = \mathbf{c}_{N-1} + (\mathbf{x}_N - \mathbf{u})(\mathbf{y}_N -
 * \mathbf{v}) \f$. Then the sample covariance is estimated with \f$ \mathrm{Cov}(\mathrm{Re}[\mathbf{Z}],
 * \mathrm{Im}[\mathbf{Z}]) = \frac{1}{N - 1} \left( \mathbf{c}_N - \mathrm{Re}[\mathbf{m}_N] *
 * \mathrm{Im}[\mathbf{m}_N] / N \right) \f$.
 *
 * - Welford algorithm: The covariance data stores \f$ \mathbf{c}_N = \mathbf{c}_{N-1} + \left( \mathbf{x}_N -
 * \mathbf{u} - \mathrm{Re}[\mathbf{m}_{N-1}] \right) \left( \mathbf{y}_N - \mathbf{v} -
 * \mathrm{Im}[\mathbf{m}_{N}] \right) \f$. Then the sample covariance is estimated with \f$
 * \mathrm{Cov}(\mathrm{Re}[\mathbf{Z}], \mathrm{Im}[\mathbf{Z}]) = \frac{1}{N - 1} \mathbf{c}_N \f$.
 *
 * @tparam A Algorithm used to calculate the mean/variance.
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
     * @brief Type for storing accumulated values.
     */
    using storage_type = Eigen::ArrayX<value_type>;

    /**
     * @brief Type for storing accumulated double values.
     */
    using dbl_storage_type = Eigen::ArrayX<double>;

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
                acc_.rdata_[idx_] += std::real(tmp) * std::real(tmp);
                acc_.idata_[idx_] += std::imag(tmp) * std::imag(tmp);
                acc_.cdata_[idx_] += std::real(tmp) * std::imag(tmp);
            } else {
                const auto tmp = val - acc_.shift_[idx_] - acc_.mdata_[idx_];
                acc_.mdata_[idx_] += tmp / static_cast<double>(acc_.count_ + 1);
                const auto tmp2 = val - acc_.shift_[idx_] - acc_.mdata_[idx_];
                acc_.rdata_[idx_] += std::real(tmp) * std::real(tmp2);
                acc_.idata_[idx_] += std::imag(tmp) * std::imag(tmp2);
                acc_.cdata_[idx_] += std::real(tmp) * std::imag(tmp2);
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
        rdata_(dbl_storage_type::Zero(size)),
        idata_(dbl_storage_type::Zero(size)),
        cdata_(dbl_storage_type::Zero(size)),
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
        rdata_(dbl_storage_type::Zero(shift.size())),
        idata_(dbl_storage_type::Zero(shift.size())),
        cdata_(dbl_storage_type::Zero(shift.size())),
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
     * @param rdata Accumulated real variance data.
     * @param idata Accumulated imaginary variance data.
     * @param cdata Accumulated covariance data.
     * @param shift Constant shift applied to the accumulated values.
     * @param count Number of accumulated values.
     */
    var_acc(storage_type mdata, dbl_storage_type rdata, dbl_storage_type idata, dbl_storage_type cdata,
        storage_type shift, count_type count) :
        mdata_(std::move(mdata)),
        rdata_(std::move(rdata)),
        idata_(std::move(idata)),
        cdata_(std::move(cdata)),
        shift_(std::move(shift)),
        count_(count),
        idx_(0) {
        if (mdata_.size() == 0) {
            throw simplemc_exception("Size == 0 in variance accumulator", "var_acc::var_acc");
        }
        if (mdata_.size() != rdata_.size() || mdata_.size() != idata_.size() || mdata_.size() != cdata_.size() ||
            mdata_.size() != shift_.size()) {
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
            rdata_[idx_] += std::real(tmp) * std::real(tmp);
            idata_[idx_] += std::imag(tmp) * std::imag(tmp);
            cdata_[idx_] += std::real(tmp) * std::imag(tmp);
        } else {
            const auto tmp = val - shift_[idx_] - mdata_[idx_];
            mdata_[idx_] += tmp / static_cast<double>(count_);
            const auto tmp2 = val - shift_[idx_] - mdata_[idx_];
            rdata_[idx_] += std::real(tmp) * std::real(tmp2);
            idata_[idx_] += std::imag(tmp) * std::imag(tmp2);
            cdata_[idx_] += std::real(tmp) * std::imag(tmp2);
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
            rdata_ += acc.rdata_ + 2.0 * (acc.shift_ - shift_).real() * acc.mdata_.real() +
                acc.count_ * (acc.shift_ - shift_).real().square();
            idata_ += acc.idata_ + 2.0 * (acc.shift_ - shift_).imag() * acc.mdata_.imag() +
                acc.count_ * (acc.shift_ - shift_).imag().square();
            cdata_ += acc.cdata_ + (acc.shift_.imag() - shift_.imag()) * acc.mdata_.real() +
                (acc.shift_.real() - shift_.real()) * acc.mdata_.imag() +
                acc.count_ * (acc.shift_.real() - shift_.real()) * (acc.shift_.imag() - shift_.imag());
        } else {
            const auto n1 = static_cast<double>(count_);
            const auto n2 = static_cast<double>(acc.count_);
            rdata_ += acc.rdata_ + (acc.mdata_ + acc.shift_ - shift_ - mdata_).real().square() * n1 * n2 / (n1 + n2);
            idata_ += acc.idata_ + (acc.mdata_ + acc.shift_ - shift_ - mdata_).imag().square() * n1 * n2 / (n1 + n2);
            cdata_ += acc.cdata_ +
                (acc.mdata_ + acc.shift_ - shift_ - mdata_).real() *
                    (acc.mdata_ + acc.shift_ - shift_ - mdata_).imag() * n1 * n2 / (n1 + n2);
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
                rdata_[idx] += std::real(tmp) * std::real(tmp);
                idata_[idx] += std::imag(tmp) * std::imag(tmp);
                cdata_[idx] += std::real(tmp) * std::imag(tmp);
            } else {
                const auto tmp = val - shift_[idx] - mdata_[idx];
                mdata_[idx] += tmp / static_cast<double>(count_);
                const auto tmp2 = val - shift_[idx] - mdata_[idx];
                rdata_[idx] += std::real(tmp) * std::real(tmp2);
                idata_[idx] += std::imag(tmp) * std::imag(tmp2);
                cdata_[idx] += std::real(tmp) * std::imag(tmp2);
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
     * @return Constant shift applied to accumulated values.
     */
    [[nodiscard]] const storage_type& shift() const { return shift_; }

    /**
     * @brief Get accumulated data used for estimating the mean.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const storage_type& mdata() const { return mdata_; }

    /**
     * @brief Get accumulated data used for estimating the variance of the real part.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const dbl_storage_type& rdata() const { return rdata_; }

    /**
     * @brief Get accumulated data used for estimating the variance of the imaginary part.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const dbl_storage_type& idata() const { return idata_; }

    /**
     * @brief Get accumulated data used for estimating the covariance between the real and imaginary parts.
     *
     * @return Data storage (content depends on the algorithm).
     */
    [[nodiscard]] const dbl_storage_type& cdata() const { return cdata_; }

    /**
     * @brief Calculate the sample mean from the accumulated data.
     *
     * @return Data storage with mean values.
     */
    [[nodiscard]] storage_type mean() const {
        return simplemc::accs::mean<value_type, varalg()>(mdata_, count_, shift_);
    }

    /**
     * @brief Calculate the sample variance of the mean.
     *
     * @details The sample variance of the mean \f$ \mathbf{s}^2_{\bar{\mathbf{Z}}} \f$ is related to the
     * standard error of the mean \f$ \mathbf{s}_{\bar{\mathbf{Z}}} = \sqrt{\mathbf{s}^2_{\bar{\mathbf{Z}}}} \f$
     * and to the variance of the data \f$ \mathbf{s}^2_{\mathbf{Z}} = \mathbf{s}^2_{\bar{\mathbf{Z}}} * N \f$.
     *
     * @return Data storage with variances.
     */
    [[nodiscard]] storage_type variance() const {
        return (variance_of_real_data() + variance_of_imag_data()) / static_cast<value_type>(count_);
    }

    /**
     * @brief Calculate the sample variance of the accumulated data.
     *
     * @details See var_acc::variance.
     *
     * @return Data storage with variances.
     */
    [[nodiscard]] dbl_storage_type variance_of_data() const {
        return variance_of_real_data() + variance_of_imag_data();
    }

    /**
     * @brief Calculate the sample variance of the real part of the accumulated data.
     *
     * @return Data storage with variances.
     */
    [[nodiscard]] dbl_storage_type variance_of_real_data() const {
        return simplemc::accs::variance<varalg()>(mdata_.real(), mdata_.real(), rdata_, count_);
    }

    /**
     * @brief Calculate the sample variance of the imaginary part of the accumulated data.
     *
     * @return Data storage with variances.
     */
    [[nodiscard]] dbl_storage_type variance_of_imag_data() const {
        return simplemc::accs::variance<varalg()>(mdata_.imag(), mdata_.imag(), idata_, count_);
    }

    /**
     * @brief Calculate the sample covariance of the real and imaginary parts of the accumulated data.
     *
     * @return Data storage with covariances.
     */
    [[nodiscard]] dbl_storage_type covariance_of_real_and_imag_data() const {
        return simplemc::accs::variance<varalg()>(mdata_.real(), mdata_.imag(), cdata_, count_);
    }

private:
    storage_type mdata_;
    dbl_storage_type rdata_;
    dbl_storage_type idata_;
    dbl_storage_type cdata_;
    storage_type shift_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_VAR_ACC_COMPLEX_HPP
