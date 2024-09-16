/**
 * @file
 * @brief Specialization of simplemc::var_acc for complex random vectors.
 */

#ifndef SIMPLEMC_ACCS_VAR_ACC_COMPLEX_HPP
#define SIMPLEMC_ACCS_VAR_ACC_COMPLEX_HPP

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
#include <complex>
#include <cstdint>

namespace simplemc {

/**
 * @ingroup simplemc-accs-accs
 * @brief Specialization of simplemc::var_acc for complex random vectors.
 *
 * @details The accumulated data is stored in four vectors:
 * - a complex vector for the mean data,
 * - a real vector for the variance data of the real part of the random vector,
 * - a real vector for the variance data of the imaginary part of the random vector and
 * - a real vector for the covariance data between the real and imaginary parts of the random vector.
 *
 * See simplemc::accs::mean and simplemc::accs::diag_covariance.
 *
 * @code{.cpp}
 * std::mt19937_64 rng;
 * std::normal_distribution<double> normal_dist_r(5.0, 1.0);
 * std::normal_distribution<double> normal_dist_i(-3.0, 0.5);
 * simplemc::var_acc<Eigen::Vector<std::complex<double>, 1>> acc;
 * for (int i = 0; i < 100000; ++i) {
 *     acc << std::complex<double>{ normal_dist_r(rng), normal_dist_i(rng) };
 * }
 * fmt::print("Mean: ({:.5f}, {:.5f})\n", acc.mean().real(), acc.mean().imag());
 * fmt::print("Variance of real part: {:.5f}\n", acc.variance_of_real_data());
 * fmt::print("Variance of imag part: {:.5f}\n", acc.variance_of_imag_data());
 * @endcode
 *
 * Output:
 *
 * ```
 * Mean: (5.00441, -2.99916)
 * Variance of real part: 1.00475
 * Variance of imag part: 0.25073
 * ```
 *
 * @tparam Z simplemc::eigen_vector_cplx type.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <eigen_vector_cplx Z, varalg A>
class var_acc<Z, A> {
public:
    /**
     * @brief Type of accumulated value.
     */
    using value_type = std::complex<double>;

    /**
     * @brief Static size of the accumulator.
     */
    static constexpr int static_size = Z::RowsAtCompileTime;

    /**
     * @brief Is the accumulator dynamically sized?
     */
    static constexpr bool is_dynamic = (Z::RowsAtCompileTime == Eigen::Dynamic);

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
     * @brief Complex vector type.
     */
    using cplx_vec_type = Z;

    /**
     * @brief Double vector type.
     */
    using dbl_vec_type = Eigen::Matrix<double, static_size, 1>;

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
            rdata_(idx) += std::real(val) * std::real(val);
            idata_(idx) += std::imag(val) * std::imag(val);
            cdata_(idx) += std::real(val) * std::imag(val);
        } else {
            const auto tmp = val - mdata_(idx);
            mdata_(idx) += tmp / static_cast<double>(count);
            const auto tmp2 = val - mdata_(idx);
            rdata_(idx) += std::real(tmp) * std::real(tmp2);
            idata_(idx) += std::imag(tmp) * std::imag(tmp2);
            cdata_(idx) += std::real(tmp) * std::imag(tmp2);
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
    explicit var_acc(size_type num = 1) :
        mdata_(cplx_vec_type::Zero(num)),
        rdata_(dbl_vec_type::Zero(num)),
        idata_(dbl_vec_type::Zero(num)),
        cdata_(dbl_vec_type::Zero(num)),
        count_(0),
        idx_(0) {
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
     * >= 1. Otherwise, it throws a simplemc::simplemc_exception.
     *
     * @param md Accumulated mean data.
     * @param rd Accumulated variance data of the real part.
     * @param id Accumulated variance data of the imaginary part.
     * @param cd Accumulated covariance data between the real and imaginary part.
     * @param ct Number of accumulated values.
     */
    var_acc(const cplx_vec_type& md, const dbl_vec_type& rd, const dbl_vec_type& id, const dbl_vec_type& cd,
        count_type ct) :
        mdata_(md),
        rdata_(rd),
        idata_(id),
        cdata_(cd),
        count_(ct),
        idx_(0) {
        if constexpr (is_dynamic) {
            if (mdata_.size() <= 0) {
                throw simplemc_exception("Dynamic size <= 0", "var_acc::var_acc");
            }
            if (mdata_.size() != rdata_.size() || mdata_.size() != idata_.size() || mdata_.size() != cdata_.size()) {
                throw simplemc_exception("Sizes of data storages do not match", "var_acc::var_acc");
            }
        }
    }

    /**
     * @brief Reset the accumulator to its initial state, i.e. with no accumulated values.
     */
    void reset() {
        mdata_.setZero();
        rdata_.setZero();
        idata_.setZero();
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
            rdata_ += vec.matrix().real().cwiseProduct(vec.matrix().real());
            idata_ += vec.matrix().imag().cwiseProduct(vec.matrix().imag());
            cdata_ += vec.matrix().real().cwiseProduct(vec.matrix().imag());
        } else {
            const auto tmp = (vec.matrix() - mdata_).eval();
            mdata_ += tmp / static_cast<double>(count_);
            const auto tmp2 = vec.matrix() - mdata_;
            rdata_ += tmp.real().cwiseProduct(tmp2.real());
            idata_ += tmp.imag().cwiseProduct(tmp2.imag());
            cdata_ += tmp.real().cwiseProduct(tmp2.imag());
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
     *   Here, \f$ \mathbf{c} \f$ stands for any of the accumulated (co)variance data, i.e. the
     *   variance of the real part, the variance of the imaginary part or the covariance between the
     *   real and imaginary parts.
     *
     * - `welford`: Let \f$ \mathbf{a} \odot \mathbf{b} \f$ denote the element-wise (Hadamard) product
     *   of two vectors \f$ \mathbf{a} \f$ and \f$ \mathbf{b} \f$. Then,
     *   - \f$ \mathbf{n}^{(N)} = \frac{N_1}{N} \mathbf{n}_{1}^{(N_1)} +
     *     \frac{N_2}{N} \mathbf{n}_{2}^{(N_2)} \f$ and
     *   - \f$ \mathbf{d}^{(N)} = \mathbf{d}_{1}^{(N_1)} + \mathbf{d}_{2}^{(N_2)} + N_1 \Re\left(
     *     \mathbf{n}_{1}^{(N_1)} - \mathbf{n}^{(N)} \right) \odot \Re\left( \mathbf{n}_{1}^{(N_1)} -
     *     \mathbf{n}^{(N)} \right) + N_2 \Re\left( \mathbf{n}_{2}^{(N_2)} - \mathbf{n}^{(N)} \right)
     *     \odot \Re\left( \mathbf{n}_{2}^{(N_2)} - \mathbf{n}^{(N)} \right) \f$ .
     *
     *   Here, \f$ \mathbf{d} \f$ stands for the accumulated variance data of the real part. For the
     *   variance data of the imaginary part and the covariance data between the real and imaginary
     *   parts, similar expressions hold.
     *
     * @param acc simplemc::var_acc to be incorporated.
     * @return Reference to this object.
     */
    var_acc& operator<<(const var_acc& acc) {
        assert(size() == acc.size());
        if constexpr (varalg() == varalg::standard) {
            mdata_ += acc.mdata_;
            rdata_ += acc.rdata_;
            idata_ += acc.idata_;
            cdata_ += acc.cdata_;
        } else {
            const auto n1 = static_cast<double>(count_);
            const auto n2 = static_cast<double>(acc.count_);
            const auto m = mdata_ * n1 / (n1 + n2) + acc.mdata_ * n2 / (n1 + n2);
            rdata_ += acc.rdata_ + n1 * (mdata_ - m).real().cwiseProduct((mdata_ - m).real()) +
                n2 * (acc.mdata_ - m).real().cwiseProduct((acc.mdata_ - m).real());
            idata_ += acc.idata_ + n1 * (mdata_ - m).imag().cwiseProduct((mdata_ - m).imag()) +
                n2 * (acc.mdata_ - m).imag().cwiseProduct((acc.mdata_ - m).imag());
            cdata_ += acc.cdata_ + n1 * (mdata_ - m).real().cwiseProduct((mdata_ - m).imag()) +
                n2 * (acc.mdata_ - m).real().cwiseProduct((acc.mdata_ - m).imag());
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
    [[nodiscard]] const cplx_vec_type& mdata() const { return mdata_; }

    /**
     * @brief Get accumulated data used for estimating the variance of the real part.
     *
     * @return Data storage (content depends on the algorithm, see simplemc::accs::diag_covariance).
     */
    [[nodiscard]] const dbl_vec_type& rdata() const { return rdata_; }

    /**
     * @brief Get accumulated data used for estimating the variance of the imaginary part.
     *
     * @return Data storage (content depends on the algorithm, see simplemc::accs::diag_covariance).
     */
    [[nodiscard]] const dbl_vec_type& idata() const { return idata_; }

    /**
     * @brief Get accumulated data used for estimating the cross-covariance between the real and
     * imaginary parts.
     *
     * @return Data storage (content depends on the algorithm, see simplemc::accs::diag_covariance).
     */
    [[nodiscard]] const dbl_vec_type& cdata() const { return cdata_; }

    /**
     * @brief Calculate the sample mean from the accumulated data.
     *
     * @details Calls simplemc::accs::mean with the accumulated mean data and the count.
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a vector object.
     *
     * @return Sample mean.
     */
    [[nodiscard]] auto mean() const {
        return detail::scalar_or_matrix<returns_scalar>(simplemc::accs::mean<varalg()>(mdata_, count_));
    }

    /**
     * @brief Calculate the sample variance of the mean.
     *
     * @details It adds the results variance_of_real_data() and variance_of_imag_data() and divides
     * by the count.
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a vector object.
     *
     * @return Diagonal of the sample covariance matrix of the mean.
     */
    [[nodiscard]] auto variance() const {
        auto res = (variance_of_real_data() + variance_of_imag_data()) / static_cast<double>(count_);
        if constexpr (returns_scalar) {
            return res;
        } else {
            return res.eval();
        }
    }

    /**
     * @brief Calculate the sample variance of the accumulated data.
     *
     * @details It adds the results from variance_of_real_data() and variance_of_imag_data().
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a vector object.
     *
     * @return Diagonal of the sample covariance matrix of the accumulated data.
     */
    [[nodiscard]] auto variance_of_data() const {
        auto res = variance_of_real_data() + variance_of_imag_data();
        if constexpr (returns_scalar) {
            return res;
        } else {
            return res.eval();
        }
    }

    /**
     * @brief Calculate the sample variance of the real part of the accumulated data.
     *
     * @details Calls simplemc::accs::diag_covariance with the accumulated real data and the count.
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a vector object.
     *
     * @return Diagonal of the sample covariance matrix of the real part of the accumulated data.
     */
    [[nodiscard]] auto variance_of_real_data() const {
        using simplemc::accs::diag_covariance;
        dbl_vec_type mdata_r = mdata_.real();
        return detail::scalar_or_matrix<returns_scalar>(diag_covariance<varalg()>(mdata_r, mdata_r, rdata_, count_));
    }

    /**
     * @brief Calculate the sample variance of the imaginary part of the accumulated data.
     *
     * @details Calls simplemc::accs::diag_covariance with the accumulated imaginary data and the
     * count.
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a vector object.
     *
     * @return Diagonal of the sample covariance matrix of the imaginary part of the accumulated data.
     */
    [[nodiscard]] auto variance_of_imag_data() const {
        using simplemc::accs::diag_covariance;
        dbl_vec_type mdata_i = mdata_.imag();
        return detail::scalar_or_matrix<returns_scalar>(diag_covariance<varalg()>(mdata_i, mdata_i, idata_, count_));
    }

    /**
     * @brief Calculate the sample cross-covariance between the real and imaginary part of the
     * accumulated data.
     *
     * @details Calls simplemc::accs::diag_covariance with the accumulated real and imaginary data and
     * the count.
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a vector object.
     *
     * @return Diagonal of the sample cross-covariance matrix between the real and imaginary part of
     * the accumulated data.
     */
    [[nodiscard]] auto covariance_of_real_and_imag_data() const {
        using simplemc::accs::diag_covariance;
        dbl_vec_type mdata_r = mdata_.real();
        dbl_vec_type mdata_i = mdata_.imag();
        return detail::scalar_or_matrix<returns_scalar>(diag_covariance<varalg()>(mdata_r, mdata_i, cdata_, count_));
    }

private:
    cplx_vec_type mdata_;
    dbl_vec_type rdata_;
    dbl_vec_type idata_;
    dbl_vec_type cdata_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_VAR_ACC_COMPLEX_HPP
