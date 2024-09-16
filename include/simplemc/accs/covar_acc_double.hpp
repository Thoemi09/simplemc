/**
 * @file
 * @brief Specialization of simplemc::covar_acc for real random vectors.
 */

#ifndef SIMPLEMC_ACCS_COVAR_ACC_DOUBLE_HPP
#define SIMPLEMC_ACCS_COVAR_ACC_DOUBLE_HPP

#include <simplemc/accs/covar_acc_fwd.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <Eigen/Dense>
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
 * @ingroup simplemc-accs-accs
 * @brief Specialization of simplemc::var_acc for real random vectors.
 *
 * @details The accumulated data is stored in two objects:
 * - a real vector for the mean data and
 * - a real matrix for the covariance data.
 *
 * See simplemc::accs::mean and simplemc::accs::covariance).
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
class covar_acc<X, A> {
public:
    /**
     * @brief Type of accumulated values.
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
     * @brief Matrix type.
     */
    using mat_type = Eigen::Matrix<value_type, static_size, static_size>;

    /**
     * @brief Get the algorithm used to accumulate the data.
     *
     * @return The simplemc::varalg tag.
     */
    static constexpr auto varalg() { return A; }

private:
    // Add a single values to the accumulator without increasing the count (the given count is assumed
    // to be already increased by one).
    // The range of indices is assumed to be sorted and that each index is unique.
    // Only the lower triangular part of the covariance matrix is updated.
    template <ranges::input_range R1, ranges::input_range R2>
    void add_values(R1&& rg, R2&& idxs, count_type count) { // NOLINT (ranges need not be forwarded)
        int drop = 1;
        if constexpr (varalg() == varalg::standard) {
            for (auto [idx1, val1] : ranges::views::zip(idxs, rg)) {
                assert(idx1 >= 0 && idx1 < size());
                // mean data and diagonal elements of covariance matrix
                mdata_(idx1) += val1;
                cdata_(idx1, idx1) += val1 * val1;
                // off-diagonal elements of covariance matrix
                for (auto [idx2, val2] : ranges::views::drop(ranges::views::zip(idxs, rg), drop)) {
                    assert(idx2 > idx1 && idx2 < size());
                    // idx2 > idx1 for sorted indices -> only lower triangular matrix
                    cdata_(idx2, idx1) += val1 * val2;
                }
                ++drop;
            }
        } else {
            for (auto [idx1, val1] : ranges::views::zip(idxs, rg)) {
                assert(idx1 >= 0 && idx1 < size());
                // mean data and diagonal elements of covariance matrix
                const auto tmp_old = val1 - mdata_(idx1);
                mdata_(idx1) += tmp_old / static_cast<double>(count);
                const auto tmp = val1 - mdata_(idx1);
                cdata_(idx1, idx1) += tmp_old * tmp;
                // off-diagonal elements of covariance matrix
                for (auto [idx2, val2] : ranges::views::drop(ranges::views::zip(idxs, rg), drop)) {
                    assert(idx2 > idx1 && idx2 < size());
                    // idx2 > idx1 for sorted indices -> only lower triangular matrix
                    // mdata_(idx2) has not been updated yet
                    cdata_(idx2, idx1) += tmp * (val2 - mdata_(idx2));
                }
                ++drop;
            }
        }
    }

public:
    /**
     * @brief Construct a covariance accumulator with a given number of elements.
     *
     * @details For dynamically sized accumulators, it throws a simplemc::simplemc_exception if the
     * given size is <= 0.
     *
     * For static sized accumulators, the `num` parameter is ignored.
     *
     * @param num Number of elements.
     */
    explicit covar_acc(size_type num = 1) :
        mdata_(vec_type::Zero(num)),
        cdata_(mat_type::Zero(num, num)),
        count_(0),
        idx_(0) {
        if constexpr (is_dynamic) {
            if (num <= 0) {
                throw simplemc_exception("Dynamic size <= 0", "covar_acc::covar_acc");
            }
        }
    }

    /**
     * @brief Construct a covariance accumulator with given data storages and count.
     *
     * @details For dynamically sized accumulators, the size of the data storages must match and be
     * >= 0. Otherwise, it throws a simplemc::simplemc_exception.
     *
     * @param md Accumulated mean data.
     * @param cd Accumulated covariance data.
     * @param ct Number of accumulated values.
     */
    covar_acc(const vec_type& md, const mat_type& cd, count_type ct) : mdata_(md), cdata_(cd), count_(ct), idx_(0) {
        if constexpr (is_dynamic) {
            if (mdata_.size() <= 0) {
                throw simplemc_exception("Dynamic size <= 0", "covar_acc::covar_acc");
            }
            if (mdata_.size() != cdata_.rows() || mdata_.size() != cdata_.cols()) {
                throw simplemc_exception("Sizes of data storages do not match", "covar_acc::covar_acc");
            }
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
     * @details For accumulators with size > 1, the value is added to the element at the current
     * index.
     *
     * See @ref simplemc-accs-accs for a code example.
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
     * @brief Stream operator for accumulating a vector.
     *
     * @details See @ref simplemc-accs-accs for a code example.
     *
     * @tparam W Eigen vector/array/expression type.
     * @param vec Vector/Array/Expression to be accumulated.
     * @return Reference to this object.
     */
    template <typename W>
    covar_acc& operator<<(const W& vec) {
        assert(vec.size() == size());
        ++count_;
        if constexpr (varalg() == varalg::standard) {
            mdata_ += vec.matrix();
            cdata_ += (vec.matrix() * vec.matrix().transpose()).template triangularView<Eigen::Lower>();
        } else {
            const auto tmp = (vec.matrix() - mdata_).eval();
            mdata_ += tmp / static_cast<double>(count_);
            cdata_ += (tmp * (vec.matrix() - mdata_).transpose()).template triangularView<Eigen::Lower>();
        }
        return *this;
    }

    /**
     * @brief Stream operator for incorporating the data from another covariance accumulator.
     *
     * @details Let the subscripts 1 and 2 correspond to the two accumulators we want to combine such
     * that \f$ N = N_1 + N_2 \f$ is the total number of accumulated values. Then, depending on the
     * simplemc::varalg, the combined accumulated data can be calculated as follows:
     * - `standard`:
     *   - \f$ \mathbf{m}^{(N)} = \mathbf{m}_{1}^{(N_1)} + \mathbf{m}_{2}^{(N_2)} \f$ and
     *   - \f$ \mathbf{C}^{(N)} = \mathbf{C}_{1}^{(N_1)} + \mathbf{C}_{2}^{(N_2)} \f$ .
     *
     * - `welford`:
     *   - \f$ \mathbf{n}^{(N)} = \frac{N_1}{N} \mathbf{n}_{1}^{(N_1)} +
     *     \frac{N_2}{N} \mathbf{n}_{2}^{(N_2)} \f$ and
     *   - \f$ \mathbf{D}^{(N)} = \mathbf{D}_{1}^{(N_1)} + \mathbf{D}_{2}^{(N_2)} + N_1 \left(
     *     \mathbf{n}_{1}^{(N_1)} - \mathbf{n}^{(N)} \right) \left( \mathbf{n}_{1}^{(N_1)} -
     *     \mathbf{n}^{(N)} \right)^T + N_2 \left( \mathbf{n}_{2}^{(N_2)} - \mathbf{n}^{(N)} \right)
     *     \left( \mathbf{n}_{2}^{(N_2)} - \mathbf{n}^{(N)} \right)^T \f$ .
     *
     * @param acc simplemc::covar_acc to be incorporated.
     * @return Reference to this object.
     */
    covar_acc& operator<<(const covar_acc& acc) {
        assert(size() == acc.size());
        if constexpr (varalg() == varalg::standard) {
            mdata_ += acc.mdata_;
            cdata_ += acc.cdata_.template triangularView<Eigen::Lower>();
        } else {
            const auto n1 = static_cast<double>(count_);
            const auto n2 = static_cast<double>(acc.count_);
            const auto m  = mdata_ * n1 / (n1 + n2) + acc.mdata_ * n2 / (n1 + n2);
            cdata_ += (acc.cdata_ + n1 * (mdata_ - m) * (mdata_ - m).transpose() +
                n2 * (acc.mdata_ - m) * (acc.mdata_ - m).transpose())
                          .template triangularView<Eigen::Lower>();
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
    void accumulate(R&& rg, size_type idx = 0) {
        ++count_;
        add_values(std::forward<R>(rg), ranges::views::iota(idx), count_);
    }

    /**
     * @brief Accumulate a range of values to arbitrary elements but with different indices.
     *
     * @details For performance reasonses, we assume that the range of indices is sorted and that each
     * index is unique.
     *
     * See @ref simplemc-accs-accs for a code example.
     *
     * @tparam R1 Input range of values.
     * @tparam R2 Input range of indices.
     * @param rg Range of values to be accumulated.
     * @param idxs Range of indices (sorted).
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
     * @brief Get accumulated data used for estimating the mean.
     *
     * @return Data storage (content depends on the algorithm, see simplemc::accs::mean).
     */
    [[nodiscard]] const vec_type& mdata() const { return mdata_; }

    /**
     * @brief Get accumulated data used for estimating the covariance.
     *
     * @warning Only the lower triangular part of the covariance matrix is valid. Use
     * `selfadjointView<Eigen::Lower>()` to get the full covariance matrix.
     *
     * @return Data storage (content depends on the algorithm, see simplemc::accs::covariance).
     */
    [[nodiscard]] const mat_type& cdata() const { return cdata_; }

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
     * @brief Calculate the sample covariance matrix of the mean.
     *
     * @details Calls covariance_of_data() with the accumulated data and the count and divides the
     * result by the count.
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a vector.
     *
     * @return Sample covariance matrix of the mean.
     */
    [[nodiscard]] auto covariance() const {
        auto res = covariance_of_data() / static_cast<double>(count_);
        if constexpr (returns_scalar) {
            return res;
        } else {
            return res.eval();
        }
    }

    /**
     * @brief Calculate the sample covariance matrix of the accumulated data.
     *
     * @details Calls simplemc::accs::covariance with the accumulated data and the count.
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a vector.
     *
     * @return Sample covariance matrix of the accumulated data.
     */
    [[nodiscard]] auto covariance_of_data() const {
        using simplemc::accs::covariance;
        mat_type cdata_full = cdata_.template selfadjointView<Eigen::Lower>();
        return detail::scalar_or_matrix<returns_scalar>(covariance<varalg()>(mdata_, mdata_, cdata_full, count_));
    }

private:
    vec_type mdata_;
    mat_type cdata_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_COVAR_ACC_DOUBLE_HPP
