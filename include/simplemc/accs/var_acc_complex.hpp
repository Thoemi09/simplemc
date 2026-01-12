/**
 * @file
 * @brief Accumulator for calculating the sample mean and sample variance of a complex random vector.
 */

#ifndef SIMPLEMC_ACCS_VAR_ACC_COMPLEX_HPP
#define SIMPLEMC_ACCS_VAR_ACC_COMPLEX_HPP

#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/multivalue_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/accs/varalg.hpp>
#include <simplemc/accs/var_acc_fwd.hpp>
#include <simplemc/mpi.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <Eigen/Dense>

#include <cassert>
#include <complex>
#include <cstdint>

namespace simplemc {

/**
 * @ingroup simplemc-accs-accs-var
 * @brief Accumulator for calculating the sample mean and sample variance of a complex random vector
 * \f$ \mathbf{Z} = \mathbf{X} + i \mathbf{Y} \f$.
 *
 * @details The accumulator takes two template parameters:
 * - the type of the random samples (a simplemc::eigen_vector_cplx type) and
 * - the algorithm (simplemc::varalg) that should be used to accumulate the data.
 *
 * Both of them determine how the accumulation is actually done and what is stored in the accumulator.
 * The accumulated data is stored in four vectors:
 * - a complex vector \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$ for the mean data,
 * - a real vector \f$ \mathbf{c}_r^{(N)}/\mathbf{d}_r^{(N)} \f$ for the variance data of the real
 * part \f$ \mathbf{X} \f$ of the random vector,
 * - a real vector \f$ \mathbf{c}_i^{(N)}/\mathbf{d}_i^{(N)} \f$ for the variance data of the
 * imaginary part \f$ \mathbf{Y} \f$ of the random vector and
 * - a real vector \f$ \mathbf{c}_{ri}^{(N)}/\mathbf{d}_{ri}^{(N)} \f$ for the cross-covariance data
 * between the real and imaginary parts of the random vector.
 *
 * See simplemc::accs::mean and simplemc::accs::diag_covariance.
 *
 * @code{.cpp}
 * #include <fmt/ranges.h>
 * #include <fmt/std.h>
 * #include <simplemc/accs.hpp>
 *
 * #include <complex>
 * #include <random>
 *
 * int main() {
 *     // complex distribution to be sampled:
 *     // - real part: normal distributed with mu = 5 and sigma = 1
 *     // - imaginary part: normal distributed with mu = -3 and sigma = 0.5
 *     std::mt19937_64 rng;
 *     std::normal_distribution<double> normal_dist_r(5.0, 1.0);
 *     std::normal_distribution<double> normal_dist_i(-3.0, 0.5);
 *
 *     // accumulate samples into a variance accumulator of size 1
 *     simplemc::var_acc_single<std::complex<double>> acc;
 *     for (int i = 0; i < 100000; ++i) {
 *         acc << std::complex<double> { normal_dist_r(rng), normal_dist_i(rng) };
 *     }
 *
 *     // print the mean and variance of the real/imaginary parts of the accumulated data
 *     fmt::println("Mean: {:.5f}", acc.mean());
 *     fmt::println("Variance of real part: {:.5f}", acc.variance_of_real_data());
 *     fmt::println("Variance of imag part: {:.5f}", acc.variance_of_imag_data());
 * }
 * @endcode
 *
 * Output:
 *
 * ```
 * Mean: (5.00441-2.99916i)
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
     * @brief Multi-value accumulator type.
     */
    using mva_type = multivalue_acc<var_acc>;

    /**
     * @brief Get the algorithm used to accumulate the data.
     *
     * @return Its simplemc::varalg tag.
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
     * @brief Construct a variance accumulator with a given number of elements \f$ M \f$.
     *
     * @details For dynamically sized accumulators, it throws a simplemc::simplemc_exception if the
     * given size \f$ M \leq 0 \f$.
     *
     * For statically sized accumulators, \f$ M \f$ is ignored.
     *
     * @param m Number of elements \f$ M \f$.
     */
    explicit var_acc(size_type m = 1) :
        mdata_(cplx_vec_type::Zero(is_dynamic ? m : static_size)),
        rdata_(dbl_vec_type::Zero(is_dynamic ? m : static_size)),
        idata_(dbl_vec_type::Zero(is_dynamic ? m : static_size)),
        cdata_(dbl_vec_type::Zero(is_dynamic ? m : static_size)),
        count_(0),
        idx_(0) {
        if constexpr (is_dynamic) {
            if (m <= 0) {
                throw simplemc_exception("Dynamic size <= 0", "var_acc::var_acc");
            }
        }
    }

    /**
     * @brief Construct a variance accumulator with the given accumulated data and number of samples
     * \f$ N \f$.
     *
     * @details For dynamically sized accumulators, the size \f$ M \f$ of the data storages must be
     * \f$ \geq 1 \f$. Otherwise, it throws a simplemc::simplemc_exception.
     *
     * @param md Accumulated mean data \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$.
     * @param rd Accumulated variance data \f$ \mathbf{c}_r^{(N)}/\mathbf{d}_r^{(N)} \f$ of the real
     * part \f$ \mathbf{X} \f$ of the random vector.
     * @param id Accumulated variance data \f$ \mathbf{c}_i^{(N)}/\mathbf{d}_i^{(N)} \f$ of the
     * imaginary part \f$ \mathbf{Y} \f$ of the random vector.
     * @param cd Accumulated covariance data \f$ \mathbf{c}_{ri}^{(N)}/\mathbf{d}_{ri}^{(N)} \f$
     * between the real and imaginary parts of the random vector.
     * @param n Number of accumulated samples \f$ N \f$.
     */
    var_acc(
        const cplx_vec_type& md, const dbl_vec_type& rd, const dbl_vec_type& id, const dbl_vec_type& cd, count_type n) :
        mdata_(md),
        rdata_(rd),
        idata_(id),
        cdata_(cd),
        count_(n),
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
     * @brief Subscript operator sets the index \f$ i \f$ and returns a reference to `this` object.
     *
     * @param i Index \f$ i \f$.
     * @return Reference to `this` object.
     */
    var_acc& operator[](size_type i) {
        idx_ = i;
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a single value \f$ z \f$.
     *
     * @details For accumulators with size \f$ M > 1 \f$, the value is added to the element at the
     * current index \f$ i \f$ (see operator[]()).
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @tparam T Type of the value to be accumulated.
     * @param z Value \f$ z \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename T>
        requires std::convertible_to<T, value_type>
    var_acc& operator<<(const T& z) {
        ++count_;
        add_value(z, idx_, count_);
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a vector \f$ \mathbf{v} \f$.
     *
     * @details See also @ref simplemc-accs-accs-how.
     *
     * @tparam W Eigen vector/array/expression type.
     * @param v Vector/Array/Expression \f$ \mathbf{v} \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename W>
    var_acc& operator<<(const W& v) {
        assert(v.size() == size());
        ++count_;
        if constexpr (varalg() == varalg::standard) {
            mdata_ += v.matrix();
            rdata_ += v.matrix().real().cwiseProduct(v.matrix().real());
            idata_ += v.matrix().imag().cwiseProduct(v.matrix().imag());
            cdata_ += v.matrix().real().cwiseProduct(v.matrix().imag());
        } else {
            const auto tmp = (v.matrix() - mdata_).eval();
            mdata_ += tmp / static_cast<double>(count_);
            const auto tmp2 = v.matrix() - mdata_;
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
     *   Here, \f$ \mathbf{c} \f$ stands for any of the accumulated (co)variance data, i.e. for \f$
     *   \mathbf{c}_r^{(N)} \f$, \f$ \mathbf{c}_i^{(N)} \f$ and for \f$ \mathbf{c}_{ri}^{(N)} \f$.
     *
     * - `welford`: Let \f$ \mathbf{a} \odot \mathbf{b} \f$ denote the element-wise (Hadamard) product
     *   of two vectors \f$ \mathbf{a} \f$ and \f$ \mathbf{b} \f$. Then,
     *   - \f$ \mathbf{n}^{(N)} = \frac{N_1}{N} \mathbf{n}_{1}^{(N_1)} +
     *     \frac{N_2}{N} \mathbf{n}_{2}^{(N_2)} \f$ and
     *   - \f$ \mathbf{d}_r^{(N)} = \mathbf{d}_{r1}^{(N_1)} + \mathbf{d}_{r2}^{(N_2)} + N_1 \Re\left(
     *     \mathbf{n}_{1}^{(N_1)} - \mathbf{n}^{(N)} \right) \odot \Re\left( \mathbf{n}_{1}^{(N_1)} -
     *     \mathbf{n}^{(N)} \right) + N_2 \Re\left( \mathbf{n}_{2}^{(N_2)} - \mathbf{n}^{(N)} \right)
     *     \odot \Re\left( \mathbf{n}_{2}^{(N_2)} - \mathbf{n}^{(N)} \right) \f$ .
     *
     *   Similar expressions hold for the other accumulated (co)variance data, \f$ \mathbf{d}_i^{(N)}
     *   \f$ and \f$ \mathbf{d}_{ri}^{(N)} \f$.
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @param acc_other simplemc::var_acc<Z, A> to be incorporated.
     * @return Reference to `this` object.
     */
    var_acc& operator<<(const var_acc& acc_other) {
        assert(size() == acc_other.size());
        if constexpr (varalg() == varalg::standard) {
            mdata_ += acc_other.mdata_;
            rdata_ += acc_other.rdata_;
            idata_ += acc_other.idata_;
            cdata_ += acc_other.cdata_;
        } else {
            const auto n1 = static_cast<double>(count_);
            const auto n2 = static_cast<double>(acc_other.count_);
            const auto m = mdata_ * n1 / (n1 + n2) + acc_other.mdata_ * n2 / (n1 + n2);
            rdata_ += acc_other.rdata_ + n1 * (mdata_ - m).real().cwiseProduct((mdata_ - m).real()) +
                n2 * (acc_other.mdata_ - m).real().cwiseProduct((acc_other.mdata_ - m).real());
            idata_ += acc_other.idata_ + n1 * (mdata_ - m).imag().cwiseProduct((mdata_ - m).imag()) +
                n2 * (acc_other.mdata_ - m).imag().cwiseProduct((acc_other.mdata_ - m).imag());
            cdata_ += acc_other.cdata_ + n1 * (mdata_ - m).real().cwiseProduct((mdata_ - m).imag()) +
                n2 * (acc_other.mdata_ - m).real().cwiseProduct((acc_other.mdata_ - m).imag());
            mdata_ = m;
        }
        count_ += acc_other.count_;
        return *this;
    }

    /**
     * @brief Accumulate a range of values to consecutive elements in the accumulator.
     *
     * @details The values are added to consecutive elements in the accumulator starting with the
     * element at the given index \f$ i \f$. The size \f$ n \f$ of the range is assumed to satisfy \f$
     * n \leq M - i \f$.
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @tparam R Input range of values.
     * @param rg Range of values to be accumulated.
     * @param i Index \f$ i \f$ of the first element in the accumulator that a value will be added to.
     */
    template <ranges::input_range R>
    void accumulate(R&& rg, size_type i = 0) { // NOLINT (ranges need not be forwarded)
        ++count_;
        for (auto val : rg) {
            add_value(val, i, count_);
            ++i;
        }
    }

    /**
     * @brief Accumulate a range of values to arbitrary elements with the given indices.
     *
     * @details Each value \f$ v_{i_j} \f$ of the given value range is accumulated at the
     * corresponding index \f$ i_j \f$ in the accumulator. Every index \f$ i_j \f$ should only appear
     * once in index range. Both ranges are assumed to be of the same size \f$ n \leq M \f$.
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @tparam R1 Input range of values.
     * @tparam R2 Input range of indices.
     * @param rg Range of values to be accumulated.
     * @param idxs Range of indices at which the values should be accumulated.
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
     * @return Multi-value accumulator wrapping `this` object.
     */
    mva_type make_mva() { return mva_type(*this); }

    /**
     * @brief Get the size \f$ M \f$ of the accumulator.
     *
     * @return Number of elements.
     */
    [[nodiscard]] auto size() const { return mdata_.size(); }

    /**
     * @brief Get the total number of accumulated samples \f$ N \f$.
     *
     * @return Number of accumulated samples.
     */
    [[nodiscard]] auto count() const { return count_; }

    /**
     * @brief Get the accumulated data \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$ used for estimating
     * the mean.
     *
     * @return simplemc::eigen_vector_cplx of size \f$ M \f$ containing \f$ \mathbf{m}^{(N)}/
     * \mathbf{n}^{(N)} \f$ (content depends on the algorithm, see simplemc::accs::mean).
     */
    [[nodiscard]] const cplx_vec_type& mdata() const { return mdata_; }

    /**
     * @brief Get the accumulated data \f$ \mathbf{c}_r^{(N)}/\mathbf{d}_r^{(N)} \f$ used for
     * estimating the variance of the real part of the random vector.
     *
     * @return simplemc::eigen_vector_dbl of size \f$ M \f$ containing \f$ \mathbf{c}_{r}^{(N)}/
     * \mathbf{d}_{r}^{(N)} \f$ (content depends on the algorithm, see
     * simplemc::accs::diag_covariance).
     */
    [[nodiscard]] const dbl_vec_type& rdata() const { return rdata_; }

    /**
     * @brief Get the accumulated data \f$ \mathbf{c}_i^{(N)}/\mathbf{d}_i^{(N)} \f$ used for
     * estimating the variance of the imaginary part of the random vector.
     *
     * @return simplemc::eigen_vector_dbl of size \f$ M \f$ containing \f$ \mathbf{c}_{i}^{(N)}/
     * \mathbf{d}_{i}^{(N)} \f$ (content depends on the algorithm, see
     * simplemc::accs::diag_covariance).
     */
    [[nodiscard]] const dbl_vec_type& idata() const { return idata_; }

    /**
     * @brief Get the accumulated data \f$ \mathbf{c}_{ri}^{(N)}/\mathbf{d}_{ri}^{(N)} \f$ used for
     * estimating the cross-covariance between the real and imaginary parts of the random vector.
     *
     * @return simplemc::eigen_vector_dbl of size \f$ M \f$ containing \f$ \mathbf{c}_{ri}^{(N)}/
     * \mathbf{d}_{ri}^{(N)} \f$ (content depends on the algorithm, see
     * simplemc::accs::diag_covariance).
     */
    [[nodiscard]] const dbl_vec_type& cdata() const { return cdata_; }

    /**
     * @brief Calculate the sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$.
     *
     * @details It calls simplemc::accs::mean with the accumulated mean data and the count.
     *
     * For statically sized accumulators with \f$ M = 1 \f$, it returns a complex scalar. Otherwise,
     * it returns a cplx_vec_type object.
     *
     * @return Sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$.
     */
    [[nodiscard]] auto mean() const {
        return detail::scalar_or_matrix<returns_scalar>(simplemc::accs::mean<varalg()>(mdata_, count_));
    }

    /**
     * @brief Calculate the sample variance \f$ s_{\overline{\mathbf{Z}}}^2 \f$ of the mean.
     *
     * @details It adds variance_of_real_data() and variance_of_imag_data() and divides the result by
     * count().
     *
     * For statically sized accumulators with \f$ M = 1 \f$, it returns a real scalar. Otherwise, it
     * returns a dbl_vec_type object.
     *
     * @return Sample variance of the mean \f$ s_{\overline{\mathbf{Z}}}^2 \f$.
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
     * @brief Calculate the sample variance \f$ s_{\mathbf{Z}}^2 \f$.
     *
     * @details It adds the results from variance_of_real_data() and variance_of_imag_data().
     *
     * For statically sized accumulators with \f$ M = 1 \f$, it returns a real scalar. Otherwise, it
     * returns a dbl_vec_type object.
     *
     * @return Sample variance \f$ s_{\mathbf{Z}}^2 \f$.
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
     * @brief Calculate the sample variance \f$ s_{\mathbf{X}}^2 \f$ of the real part \f$ \mathbf{X}
     * \f$ of the random vector.
     *
     * @details It calls simplemc::accs::diag_covariance with the accumulated real data and the count.
     *
     * For statically sized accumulators with \f$ M = 1 \f$, it returns a real scalar. Otherwise, it
     * returns a dbl_vec_type object.
     *
     * @return Sample variance of the real part of the random vector, i.e. \f$ s_{\mathbf{X}}^2 \f$.
     */
    [[nodiscard]] auto variance_of_real_data() const {
        using simplemc::accs::diag_covariance;
        dbl_vec_type mdata_r = mdata_.real();
        return detail::scalar_or_matrix<returns_scalar>(diag_covariance<varalg()>(mdata_r, mdata_r, rdata_, count_));
    }

    /**
     * @brief Calculate the sample variance \f$ s_{\mathbf{Y}}^2 \f$ of the imaginary part \f$
     * \mathbf{Y} \f$ of the random vector.
     *
     * @details It calls simplemc::accs::diag_covariance with the accumulated imaginary data and the
     * count.
     *
     * For statically sized accumulators with \f$ M = 1 \f$, it returns a real scalar. Otherwise, it
     * returns a dbl_vec_type object.
     *
     * @return Sample variance of the imaginary part of the random vector, i.e. \f$ s_{\mathbf{Y}}^2
     * \f$.
     */
    [[nodiscard]] auto variance_of_imag_data() const {
        using simplemc::accs::diag_covariance;
        dbl_vec_type mdata_i = mdata_.imag();
        return detail::scalar_or_matrix<returns_scalar>(diag_covariance<varalg()>(mdata_i, mdata_i, idata_, count_));
    }

    /**
     * @brief Calculate the sample covariance \f$ \mathrm{diag}(s_{\mathbf{XY}}^2) \f$ between the
     * real (\f$ \mathbf{X} \f$) and imaginary (\f$ \mathbf{Y} \f$) parts of the random vector.
     *
     * @details It calls simplemc::accs::diag_covariance with the accumulated real and imaginary data
     * and the count.
     *
     * For statically sized accumulators with \f$ M = 1 \f$, it returns a real scalar. Otherwise, it
     * returns a dbl_vec_type object.
     *
     * @return Sample covariance between the real and imaginary parts of the random vector, i.e. \f$
     * \mathrm{diag}(s_{\mathbf{XY}}^2) \f$.
     */
    [[nodiscard]] auto covariance_of_real_and_imag_data() const {
        using simplemc::accs::diag_covariance;
        dbl_vec_type mdata_r = mdata_.real();
        dbl_vec_type mdata_i = mdata_.imag();
        return detail::scalar_or_matrix<returns_scalar>(diag_covariance<varalg()>(mdata_r, mdata_i, cdata_, count_));
    }

    /**
     * @brief Collect variance accumulators from different MPI processes.
     *
     * @details It constructs a new variance accumulator with the reduced accumulated mean data,
     * variance data and counts from all MPI processes.
     *
     * The reduction operation depends on the simplemc::varalg algorithm used to accumulate the data.
     * See operator<<(const var_acc&) for how it is done in the case of 2 accumulators.
     *
     * It asserts that the size of the accumulator is equal on all processes.
     *
     * @param comm simplemc::mpi::communicator object.
     * @param acc Variance accumulator.
     * @return Variance accumulator with the reduced data from all processes.
     */
    friend var_acc mpi_collect(const mpi::communicator& comm, const var_acc& acc) {
        assert(all_equal(acc.size(), comm));
        var_acc res(acc.size());
        mpi::all_reduce(acc.count_, res.count_, MPI_SUM, comm);
        if constexpr (var_acc::varalg() == varalg::standard) {
            mpi::all_reduce(make_span(acc.mdata_), make_span(res.mdata_), MPI_SUM, comm);
            mpi::all_reduce(make_span(acc.rdata_), make_span(res.rdata_), MPI_SUM, comm);
            mpi::all_reduce(make_span(acc.idata_), make_span(res.idata_), MPI_SUM, comm);
            mpi::all_reduce(make_span(acc.cdata_), make_span(res.cdata_), MPI_SUM, comm);
        } else {
            const auto n1 = static_cast<double>(acc.count_);
            const auto n = static_cast<double>(res.count_);
            const cplx_vec_type tmp_mdata = acc.mdata_ * n1 / n;
            mpi::all_reduce(make_span(tmp_mdata), make_span(res.mdata_), MPI_SUM, comm);
            const dbl_vec_type tmp_rdata =
                acc.rdata_ + n1 * (acc.mdata_ - res.mdata_).real().cwiseProduct((acc.mdata_ - res.mdata_).real());
            mpi::all_reduce(make_span(tmp_rdata), make_span(res.rdata_), MPI_SUM, comm);
            const dbl_vec_type tmp_idata =
                acc.idata_ + n1 * (acc.mdata_ - res.mdata_).imag().cwiseProduct((acc.mdata_ - res.mdata_).imag());
            mpi::all_reduce(make_span(tmp_idata), make_span(res.idata_), MPI_SUM, comm);
            const dbl_vec_type tmp_cdata =
                acc.cdata_ + n1 * (acc.mdata_ - res.mdata_).real().cwiseProduct((acc.mdata_ - res.mdata_).imag());
            mpi::all_reduce(make_span(tmp_cdata), make_span(res.cdata_), MPI_SUM, comm);
        }
        return res;
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
