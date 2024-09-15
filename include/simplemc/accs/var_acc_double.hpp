/**
 * @file
 * @brief Specialization of simplemc::var_acc for real random vectors.
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

#include <Eigen/Dense>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/zip.hpp>

#include <cassert>
#include <cstdint>

namespace simplemc {

/**
 * @ingroup simplemc-accs-accs
 * @brief Specialization of simplemc::var_acc for real random vectors.
 *
 * @details The accumulated data is stored in two vectors: (i) one for the mean and (ii) one for the
 * variance (see simplemc::accs::mean and simplemc::accs::diag_covariance).
 *
 * @code{.cpp}
 * std::mt19937_64 rng;
 * std::normal_distribution<double> normal_dist(5.0, 1.0);
 * simplemc::var_acc<Eigen::Vector<double, 1>> acc;
 * for (int i = 0; i < 100000; ++i) {
 *     acc << normal_dist(rng);
 * }
 * fmt::print("Mean: {}\n", acc.mean());
 * fmt::print("Variance: {}\n", acc.variance_of_data());
 * @endcode
 *
 * Output:
 *
 * ```
 * Mean: 5.002072302074473
 * Variance: 1.0037814573268022
 * ```
 *
 * @tparam X simplemc::eigen_vector_dbl type.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <eigen_vector_dbl X, varalg A>
class var_acc<X, A> {
public:
    /**
     * @brief Type of accumulated value.
     */
    using value_type = double;

    /**
     * @brief Static size of the accumulator.
     */
    static constexpr int static_size = X::RowsAtCompileTime;

    /**
     * @brief Is the accumulator dynamically sized?
     */
    static constexpr bool is_dynamic = (X::RowsAtCompileTime == Eigen::Dynamic);

    /**
     * @brief Does the accumulator return scalars or vectors/matrices?
     */
    static constexpr bool returns_scalar = (!is_dynamic && static_size == 1);

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
    using vec_type = X;

    /**
     * @brief Multi value accumulator type.
     */
    using mva_type = multivalue_acc<var_acc>;

    /**
     * @brief Get the algorithm used to accumulate the data.
     *
     * @return The simplemc::varalg tag.
     */
    static constexpr auto varalg() { return A; }

    /* Friend declarations. */
    friend class multivalue_acc<var_acc>;

private:
    // Add a single value to the accumulator without increasing the count (the given count is assumed
    // to be already increased by one).
    void add_value(value_type val, size_type idx, count_type count) {
        assert(idx >= 0 && idx < size());
        if constexpr (varalg() == varalg::standard) {
            mdata_(idx) += val;
            vdata_(idx) += val * val;
        } else {
            const auto tmp = val - mdata_(idx);
            mdata_(idx) += tmp / static_cast<value_type>(count);
            vdata_(idx) += tmp * (val - mdata_(idx));
        }
    }

public:
    /**
     * @brief Construct a variance accumulator with a given number of elements.
     *
     * @details For dynamically sized accumulators, it throws a simplemc::simplemc_exception if the
     * given size is <= 0.
     *
     * For static sized accumulators, the `num` parameter is ignored.
     *
     * @param num Number of elements.
     */
    explicit var_acc(size_type num = 1) : mdata_(vec_type::Zero(num)), vdata_(vec_type::Zero(num)), count_(0), idx_(0) {
        if constexpr (is_dynamic) {
            if (num <= 0) {
                throw simplemc_exception("Dynamic size <= 0", "var_acc::var_acc");
            }
        }
    }

    /**
     * @brief Construct a variance accumulator with given data storages and count.
     *
     * @details For dynamically sized accumulators, the size of the data storages must match and be
     * >= 0. Otherwise, it throws a simplemc::simplemc_exception.
     *
     * @param md Accumulated mean data.
     * @param vd Accumulated variance data.
     * @param ct Number of accumulated values.
     */
    var_acc(const vec_type& md, const vec_type& vd, count_type ct) : mdata_(md), vdata_(vd), count_(ct), idx_(0) {
        if constexpr (is_dynamic) {
            if (mdata_.size() <= 0) {
                throw simplemc_exception("Dynamic size <= 0", "var_acc::var_acc");
            }
            if (mdata_.size() != vdata_.size()) {
                throw simplemc_exception("Sizes of data storages do not match", "var_acc::var_acc");
            }
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
     * @details For accumulators with size > 1, the value is added to the element at the current
     * index.
     *
     * See @ref simplemc-accs-accs for a code example.
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
     * @brief Stream operator for accumulating a vector.
     *
     * @details See @ref simplemc-accs-accs for a code example.
     *
     * @tparam W Eigen vector/array/expression type.
     * @param vec Vector/Array/Expression to be accumulated.
     * @return Reference to this object.
     */
    template <typename W>
    var_acc& operator<<(const W& vec) {
        assert(vec.size() == size());
        ++count_;
        if constexpr (varalg() == varalg::standard) {
            mdata_ += vec.matrix();
            vdata_ += vec.matrix().cwiseProduct(vec.matrix());
        } else {
            const auto tmp = (vec.matrix() - mdata_).eval();
            mdata_ += tmp / static_cast<value_type>(count_);
            vdata_ += tmp.cwiseProduct(vec.matrix() - mdata_);
        }
        return *this;
    }

    /**
     * @brief Stream operator for incorporating the data from another variance accumulator.
     *
     * @details Let the subscripts 1 and 2 correspond to the two accumulators we want to combine such
     * that \f$ N = N_1 + N_2 \f$ is the total number of accumulated values. Then, depending on the
     * simplemc::varalg, the combined accumulated data can be calculated as follows:
     * - `standard`:
     *   - \f$ \mathbf{m}^{(N)} = \mathbf{m}_{1}^{(N_1)} + \mathbf{m}_{2}^{(N_2)} \f$ and
     *   - \f$ \mathbf{c}^{(N)} = \mathbf{c}_{1}^{(N_1)} + \mathbf{c}_{2}^{(N_2)} \f$ .
     *
     * - `welford`: Let \f$ \mathbf{a} \odot \mathbf{b} \f$ denote the element-wise (Hadamard) product
     *   of two vectors \f$ \mathbf{a} \f$ and \f$ \mathbf{b} \f$. Then,
     *   - \f$ \mathbf{n}^{(N)} = \frac{N_1}{N} \mathbf{n}_{1}^{(N_1)} +
     *     \frac{N_2}{N} \mathbf{n}_{2}^{(N_2)} \f$ and
     *   - \f$ \mathbf{d}^{(N)} = \mathbf{d}_{1}^{(N_1)} + \mathbf{d}_{2}^{(N_2)} + N_1 \left(
     *     \mathbf{n}_{1}^{(N_1)} - \mathbf{n}^{(N)} \right) \odot \left( \mathbf{n}_{1}^{(N_1)} -
     *     \mathbf{n}^{(N)} \right) + N_2 \left( \mathbf{n}_{2}^{(N_2)} - \mathbf{n}^{(N)} \right)
     *     \odot \left( \mathbf{n}_{2}^{(N_2)} - \mathbf{n}^{(N)} \right) \f$ .
     *
     * @param acc simplemc::var_acc to be incorporated.
     * @return Reference to this object.
     */
    var_acc& operator<<(const var_acc& acc) {
        assert(size() == acc.size());
        if constexpr (varalg() == varalg::standard) {
            mdata_ += acc.mdata_;
            vdata_ += acc.vdata_;
        } else {
            const auto n1 = static_cast<value_type>(count_);
            const auto n2 = static_cast<value_type>(acc.count_);
            const auto m = mdata_ * n1 / (n1 + n2) + acc.mdata_ * n2 / (n1 + n2);
            vdata_ += acc.vdata_ + n1 * (mdata_ - m).cwiseProduct(mdata_ - m) +
                n2 * (acc.mdata_ - m).cwiseProduct(acc.mdata_ - m);
            mdata_ = m;
        }
        count_ += acc.count_;
        return *this;
    }

    /**
     * @brief Accumulate a range of values to consecutive elements in the accumulator.
     *
     * @details The values are added to consecutive elements in the accumulator starting with the
     * element at the given index. The size of the range is assumed to be <= size() - `idx`.
     *
     * See @ref simplemc-accs-accs for a code example.
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
     * @brief Accumulate a range of values to arbitrary elements but with different indices.
     *
     * @details Each value of the given value range is accumulated at the corresponding index of the
     * given index range. Every index should only appear once.
     *
     * See @ref simplemc-accs-accs for a code example.
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
     * @brief Get accumulated data used for estimating the mean.
     *
     * @return Data storage (content depends on the algorithm, see simplemc::accs::mean).
     */
    [[nodiscard]] const vec_type& mdata() const { return mdata_; }

    /**
     * @brief Get accumulated data used for estimating the variance.
     *
     * @return Data storage (content depends on the algorithm, see simplemc::accs::diag_covariance).
     */
    [[nodiscard]] const vec_type& vdata() const { return vdata_; }

    /**
     * @brief Calculate the sample mean from the accumulated data.
     *
     * @details Calls simplemc::accs::mean with the accumulated mean data and the count.
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a vector.
     *
     * @return Sample mean.
     */
    [[nodiscard]] auto mean() const {
        return detail::scalar_or_matrix<returns_scalar>(simplemc::accs::mean<varalg()>(mdata_, count_));
    }

    /**
     * @brief Calculate the sample variance of the mean.
     *
     * @details Calls simplemc::accs::diag_covariance with the accumulated data and the count and
     * divides the result by the count.
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a vector.
     *
     * @return Diagonal of the sample covariance matrix of the mean.
     */
    [[nodiscard]] auto variance() const {
        auto res = variance_of_data() / static_cast<value_type>(count_);
        if constexpr (returns_scalar) {
            return res;
        } else {
            return res.eval();
        }
    }

    /**
     * @brief Calculate the sample variance of the accumulated data.
     *
     * @details Calls simplemc::accs::diag_covariance with the accumulated data and the count.
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a vector.
     *
     * @return Diagonal of the sample covariance matrix of the accumulated data.
     */
    [[nodiscard]] auto variance_of_data() const {
        using simplemc::accs::diag_covariance;
        return detail::scalar_or_matrix<returns_scalar>(diag_covariance<varalg()>(mdata_, mdata_, vdata_, count_));
    }

private:
    vec_type mdata_;
    vec_type vdata_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_VAR_ACC_DOUBLE_HPP
