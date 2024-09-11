/**
 * @file
 * @brief Accumulator for calculating the sample mean of a random vector.
 */

#ifndef SIMPLEMC_ACCS_MEAN_ACC_HPP
#define SIMPLEMC_ACCS_MEAN_ACC_HPP

#include <simplemc/accs/multivalue_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <Eigen/Dense>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/zip.hpp>

#include <cassert>
#include <cstdint>

namespace simplemc {

/**
 * @ingroup simplemc-accs-accs
 * @brief Accumulator for calculating the sample mean of a random vector.
 *
 * @details The accumulator takes two template parameters:
 * - the value type of the random vector and
 * - the algorithm (simplemc::accs::varalg) that should be used to accumulate the data.
 *
 * Both of them determine how the accumulation is actually done and what is stored in the accumulator.
 * Please see simplemc::accs::mean for more details.
 *
 * @code{.cpp}
 *
 * @endcode
 *
 * Output:
 *
 * ```
 *
 * ```
 *
 * @tparam T Type of accumulated values (either `double` or `std::complex<double>`).
 * @tparam M Static size of the accumulator/random vector (use `Eigen::Dynamic` if dynamic).
 * @tparam A Algorithm (either `standard` or `welford`).
 */
template <double_or_complex T, int M = Eigen::Dynamic, accs::varalg A = accs::varalg::welford>
    requires accs::static_extent<M>
class mean_acc {
public:
    /**
     * @brief Type of accumulated values.
     */
    using value_type = T;

    /**
     * @brief Static size of the accumulator.
     */
    static constexpr int static_size = M;

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
    using vec_type = Eigen::Matrix<value_type, static_size, 1>;

    /**
     * @brief Multi value accumulator type.
     */
    using mva_type = multivalue_acc<mean_acc>;

    /**
     * @brief Get the algorithm used to accumulate the data.
     *
     * @return The simplemc::accs::varalg tag.
     */
    static constexpr auto varalg() { return A; }

    /* Friend declarations. */
    friend class multivalue_acc<mean_acc>;

private:
    // Add a single value to the accumulator without increasing the count (the given count is assumed
    // to be already increased by one). The constant shift if set, is ignored.
    void add_value(value_type val, size_type idx, count_type count) {
        assert(shift_ == vec_type::Zero(size()));
        assert(idx >= 0 && idx < size());
        if constexpr (varalg() == accs::varalg::standard) {
            mdata_(idx) += val;
        } else {
            mdata_(idx) += (val - mdata_(idx)) / static_cast<value_type>(count);
        }
    }

public:
    /**
     * @brief Construct a mean accumulator with a given number of elements.
     *
     * @details The constant shift vector is set to zero.
     *
     * For dynamically sized accumulators, it throws a simplemc::simplemc_exception if the given size
     * is <= 0.
     *
     * @note For static sized accumulators, the `size` parameter is ignored.
     *
     * @param size Number of elements.
     */
    explicit mean_acc(size_type size = 1) :
        mdata_(vec_type::Zero(size)),
        shift_(vec_type::Zero(size)),
        count_(0),
        idx_(0) {
        if constexpr (static_size == Eigen::Dynamic) {
            if (size <= 0) {
                throw simplemc_exception("Dynamic size <= 0", "mean_acc::mean_acc");
            }
        }
    }

    /**
     * @brief Construct a mean accumulator with a given constant shift.
     *
     * @details For dynamically sized accumulators, i.e. `static_size == Eigen::Dynamic`, the size is
     * set to the size of the shift vector. It throws a simplemc::simplemc_exception if the size of
     * the shift vector is <= 0.
     *
     * @param shift Constant shift applied to the accumulated values.
     */
    explicit mean_acc(const vec_type& shift) : mdata_(vec_type::Zero(shift.size())), shift_(shift), count_(0), idx_(0) {
        if constexpr (static_size == Eigen::Dynamic) {
            if (shift.size() <= 0) {
                throw simplemc_exception("Dynamic size <= 0", "mean_acc::mean_acc");
            }
        }
    }

    /**
     * @brief Construct a mean accumulator with a given data storage, constant shift and count.
     *
     * @details For dynamically sized accumulators, i.e. `static_size == Eigen::Dynamic`, the sizes of
     * the data storage and the shift vector must match and must be >= 0. Otherwise, it throws a
     * simplemc::simplemc_exception.
     *
     * @param mdata Accumulated mean data.
     * @param shift Constant shift applied to the accumulated values.
     * @param count Number of accumulated values.
     */
    mean_acc(const vec_type& mdata, const vec_type& shift, count_type count) :
        mdata_(mdata),
        shift_(shift),
        count_(count),
        idx_(0) {
        if constexpr (static_size == Eigen::Dynamic) {
            if (mdata_.size() <= 0) {
                throw simplemc_exception("Dynamic size <= 0", "mean_acc::mean_acc");
            }
            if (mdata_.size() != shift_.size()) {
                throw simplemc_exception("Size of data != size of constant shift", "mean_acc::mean_acc");
            }
        }
    }

    /**
     * @brief Reset the accumulator to its initial state, i.e. with no accumulated values.
     */
    void reset() {
        mdata_.setZero();
        count_ = 0;
        idx_ = 0;
    }

    /**
     * @brief Subscript operator sets the index and returns a reference to this object.
     *
     * @param idx Index.
     * @return Reference to this object.
     */
    mean_acc& operator[](size_type idx) {
        idx_ = idx;
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a single value.
     *
     * @note No constant shift is applied, even if it was set during construction.
     *
     * @param val Value to be accumulated.
     * @return Reference to this object.
     */
    mean_acc& operator<<(value_type val) {
        ++count_;
        add_value(val, idx_, count_);
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a vector.
     *
     * @param vec Vector to be accumulated.
     * @return Reference to this object.
     */
    mean_acc& operator<<(const vec_type& vec) {
        ++count_;
        if constexpr (varalg() == accs::varalg::standard) {
            mdata_ += (vec - shift_);
        } else {
            mdata_ += (vec - shift_ - mdata_) / static_cast<value_type>(count_);
        }
        return *this;
    }

    /**
     * @brief Stream operator for incorporating the data from another mean accumulator.
     *
     * @details We have to take care of the fact that the shift vectors might be different. This
     * results in the following equations for the two supported algorithms (see simplemc::accs::mean):
     * - `standard`:
     *   \f[
     *     \mathbf{m}_{1}^{(N)} = \mathbf{m}_{1}^{(N_1)} + \mathbf{m}_{2}^{(N_2)} +
     *     N_2 \left( \mathbf{t}_2 - \mathbf{t}_1 \right) \; .
     *   \f]
     * - `welford`:
     *   \f[
     *     \mathbf{m}_{1}^{(N)} = \frac{N_1}{N}\mathbf{m}_{1}^{(N_1)} +
     *     \frac{N_2}{N} \left( \mathbf{m}_{2}^{(N_2)} + \mathbf{t}_2 - \mathbf{t}_1 \right) \; .
     *   \f]
     *
     * @param acc Mean accumulator to be incorporated.
     * @return Reference to this object.
     */
    mean_acc& operator<<(const mean_acc& acc) {
        assert(size() == acc.size());
        if constexpr (varalg() == accs::varalg::standard) {
            mdata_ += acc.mdata_ + acc.count_ * (acc.shift_ - shift_);
        } else {
            const auto n1 = static_cast<value_type>(count_);
            const auto n2 = static_cast<value_type>(acc.count_);
            mdata_ = mdata_ * n1 / (n1 + n2) + (acc.mdata_ + acc.shift_ - shift_) * n2 / (n1 + n2);
        }
        count_ += acc.count_;
        return *this;
    }

    /**
     * @brief Accumulate a range of values.
     *
     * @details The given index denotes the starting index for the accumulator and the size of the
     * range is assumed to be `<= size() - idx`.
     *
     * For example,
     * @code{.cpp}
     * mean_acc<double> acc(3);
     * acc.accumlate(std::array<double, 2>{1.0, 2.0}, 1);
     * @endcode
     * will accumulate the values 1.0 and 2.0 at indices 1 and 2, respectively.
     *
     * @note No constant shift is applied, even if it was set during construction.
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
     * @details Each value of the given value range is accumulated at the corresponding index of the
     * given index range. Every index should only appear once.
     *
     * For example,
     * @code{.cpp}
     * mean_acc<double> acc(3);
     * acc.accumlate(std::array<double, 2>{1.0, 2.0}, std::array<long, 2>{0, 2});
     * @endcode
     * will accumulate the values 1.0 and 2.0 at indices 0 and 2, respectively.
     *
     * @note No constant shift is applied, even if it was set during construction.
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
     * @brief Create a multi-value accumulator.
     *
     * @details See simplemc::multivalue_acc.
     *
     * @return Multi-value accumulator.
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
     * @return Constant shift vector applied to the accumulated values.
     */
    [[nodiscard]] const vec_type& shift() const { return shift_; }

    /**
     * @brief Get accumulated data used for estimating the mean.
     *
     * @return Data storage (content depends on the algorithm, see simplemc::accs::mean).
     */
    [[nodiscard]] const vec_type& mdata() const { return mdata_; }

    /**
     * @brief Calculate the sample mean from the accumulated data.
     *
     * @details Calls simplemc::accs::mean with the accumulated mean data, count and shift vector.
     *
     * @return Sample mean.
     */
    [[nodiscard]] vec_type mean() const { return simplemc::accs::mean<value_type, varalg()>(mdata_, count_, shift_); }

private:
    vec_type mdata_;
    vec_type shift_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_MEAN_ACC_HPP
