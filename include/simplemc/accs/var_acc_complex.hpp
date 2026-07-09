/**
 * @file
 * @brief Accumulator for calculating the sample mean and sample variance of a complex data set.
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
#include <concepts>
#include <cstdint>
#include <type_traits>

namespace simplemc {

/**
 * @ingroup simplemc-accs-accs-var
 * @brief Accumulator for calculating the sample mean and sample variance of a complex data set.
 *
 * @details The accumulator satisfies simplemc::variance_accumulator and takes two template
 * parameters:
 * - the type of the data samples (complex simplemc::sample_type) and
 * - the algorithm (simplemc::varalg) that should be used to accumulate the data.
 *
 * Both of them determine how the accumulation is actually done and what is stored in the accumulator.
 * The accumulated data is stored in four vectors:
 * - a complex vector \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$ for the mean data,
 * - a real vector \f$ \mathbf{c}_r^{(N)}/\mathbf{d}_r^{(N)} \f$ for the variance data of the real
 * part of the complex samples,
 * - a real vector \f$ \mathbf{c}_i^{(N)}/\mathbf{d}_i^{(N)} \f$ for the variance data of the
 * imaginary part of the complex samples and
 * - a real vector \f$ \mathbf{c}_{ri}^{(N)}/\mathbf{d}_{ri}^{(N)} \f$ for the cross-covariance data
 * between the real and imaginary parts.
 *
 * The mean data is accumulated as described in simplemc::mean_acc and the accumulation of the
 * variance data follows @ref "simplemc::var_acc< X, A >" "simplemc::var_acc for real samples".
 *
 * @include accs/doc_var_acc_cplx.cpp
 *
 * Output:
 *
 * ```
 * Mean: (5.00441-2.99916i)
 * Variance of real part: 1.00475
 * Variance of imag part: 0.25073
 * ```
 *
 * @tparam Z Complex simplemc::sample_type.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <sample_type Z, varalg A>
    requires(std::same_as<Z, std::complex<double>> || eigen_vector_cplx<Z>)
class var_acc<Z, A> {
public:
    /**
     * @brief Type of accumulated samples (complex simplemc::sample_type).
     */
    using sample_type = Z;

    /**
     * @brief Complex vector type for storing accumulated mean data.
     */
    using cplx_vec_type = std::conditional_t<sample_scalar<Z>, Eigen::Matrix<Z, 1, 1>, Z>;

    /**
     * @brief Underlying scalar type of accumulated samples.
     */
    using value_type = std::complex<double>;

    /**
     * @brief Type for counting the number of accumulated samples.
     */
    using count_type = std::uint64_t;

    /**
     * @brief Size type of the accumulator.
     */
    using size_type = long;

    /**
     * @brief Static size of the accumulator.
     */
    static constexpr int static_size = cplx_vec_type::RowsAtCompileTime;

    /**
     * @brief Is the accumulator dynamically sized?
     */
    static constexpr bool is_dynamic = (cplx_vec_type::RowsAtCompileTime == Eigen::Dynamic);

    /**
     * @brief Real vector type for storing accumulated (cross-co)variance data.
     */
    using dbl_vec_type = Eigen::Matrix<double, static_size, 1>;

    /**
     * @brief Get the algorithm used to accumulate the data.
     *
     * @return Its simplemc::varalg tag.
     */
    static constexpr auto varalg() noexcept { return A; }

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
                throw simplemc_exception("Dynamic size <= 0");
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
     * @param rd Accumulated variance data \f$ \mathbf{c}_r^{(N)}/\mathbf{d}_r^{(N)} \f$.
     * @param id Accumulated variance data \f$ \mathbf{c}_i^{(N)}/\mathbf{d}_i^{(N)} \f$.
     * @param cd Accumulated cross-covariance data \f$ \mathbf{c}_{ri}^{(N)}/\mathbf{d}_{ri}^{(N)}
     * \f$.
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
                throw simplemc_exception("Dynamic size <= 0");
            }
            if (mdata_.size() != rdata_.size() || mdata_.size() != idata_.size() || mdata_.size() != cdata_.size()) {
                throw simplemc_exception("Sizes of data storages do not match");
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
     * @details The index is *sticky*: it persists until changed by another call to operator[]() or
     * until reset() is called. For scalar accumulators (size \f$ M = 1 \f$), the index should
     * remain at 0.
     *
     * @param i Index \f$ i \f$.
     * @return Reference to `this` object.
     */
    var_acc& operator[](size_type i) noexcept {
        idx_ = i;
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a single (scalar) value \f$ z \f$.
     *
     * @details For accumulators with size \f$ M > 1 \f$, the value is added to the element at the
     * current index \f$ i \f$ (see operator[]()).
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @tparam U Type of the value to be accumulated.
     * @param z Value \f$ z \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename U>
        requires std::convertible_to<U, value_type>
    var_acc& operator<<(const U& z) {
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
     *   Here, \f$ \mathbf{c} \f$ stands for any of the accumulated (cross-co)variance data, i.e. for
     *   \f$ \mathbf{c}_r^{(N)} \f$, \f$ \mathbf{c}_i^{(N)} \f$ and for \f$ \mathbf{c}_{ri}^{(N)} \f$.
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
     *   Similar expressions hold for the other accumulated (cross-co)variance data, \f$
     *   \mathbf{d}_i^{(N)} \f$ and \f$ \mathbf{d}_{ri}^{(N)} \f$.
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @param acc_other simplemc::var_acc<Z, A> to be incorporated.
     * @return Reference to `this` object.
     */
    var_acc& operator<<(const var_acc& acc_other) {
        assert(size() == acc_other.size());

        // early return if the other accumulator is empty
        if (acc_other.empty()) {
            return *this;
        }

        // incorporate the data and count
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
    [[nodiscard]] auto make_mva() noexcept { return multivalue_acc<var_acc>(*this); }

    /**
     * @brief Get the size \f$ M \f$ of the accumulator.
     *
     * @return Number of elements.
     */
    [[nodiscard]] auto size() const noexcept { return mdata_.size(); }

    /**
     * @brief Get the total number of accumulated samples \f$ N \f$.
     *
     * @return Number of accumulated samples.
     */
    [[nodiscard]] auto count() const noexcept { return count_; }

    /**
     * @brief Check if the accumulator is empty.
     *
     * @return True if the count() is zero, i.e. \f$ N = 0 \f$, false otherwise.
     */
    [[nodiscard]] bool empty() const noexcept { return count_ == 0; }

    /**
     * @brief Get the accumulated mean data \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$.
     *
     * @return simplemc::eigen_vector_cplx of size \f$ M \f$ containing \f$ \mathbf{m}^{(N)}/
     * \mathbf{n}^{(N)} \f$.
     */
    [[nodiscard]] const cplx_vec_type& mdata() const noexcept { return mdata_; }

    /**
     * @brief Get the accumulated variance data \f$ \mathbf{c}_r^{(N)}/\mathbf{d}_r^{(N)} \f$ of the
     * real part of the complex samples.
     *
     * @return simplemc::eigen_vector_dbl of size \f$ M \f$ containing \f$ \mathbf{c}_{r}^{(N)}/
     * \mathbf{d}_{r}^{(N)} \f$.
     */
    [[nodiscard]] const dbl_vec_type& rdata() const noexcept { return rdata_; }

    /**
     * @brief Get the accumulated variance data \f$ \mathbf{c}_i^{(N)}/\mathbf{d}_i^{(N)} \f$ of the
     * imaginary part of the complex samples.
     *
     * @return simplemc::eigen_vector_dbl of size \f$ M \f$ containing \f$ \mathbf{c}_{i}^{(N)}/
     * \mathbf{d}_{i}^{(N)} \f$.
     */
    [[nodiscard]] const dbl_vec_type& idata() const noexcept { return idata_; }

    /**
     * @brief Get the accumulated cross-covariance data \f$ \mathbf{c}_{ri}^{(N)}/
     * \mathbf{d}_{ri}^{(N)} \f$ between the real and imaginary parts of the complex samples.
     *
     * @return simplemc::eigen_vector_dbl of size \f$ M \f$ containing \f$ \mathbf{c}_{ri}^{(N)}/
     * \mathbf{d}_{ri}^{(N)} \f$.
     */
    [[nodiscard]] const dbl_vec_type& cdata() const noexcept { return cdata_; }

    /**
     * @brief Calculate the sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$.
     *
     * @details It calls simplemc::accs::mean with the accumulated mean data \f$ \mathbf{m}^{(N)}/
     * \mathbf{n}^{(N)} \f$ and the count \f$ N \f$.
     *
     * @return Sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$.
     */
    [[nodiscard]] auto mean() const {
        return detail::scalar_or_matrix<sample_scalar<sample_type>>(simplemc::accs::mean<varalg()>(mdata_, count_));
    }

    /**
     * @brief Calculate the sample variance \f$ s_{\overline{\mathbf{Z}}}^2 \f$ of the mean.
     *
     * @details It adds variance_of_real_data() and variance_of_imag_data() and divides the result by
     * count().
     *
     * @return Sample variance of the mean \f$ s_{\overline{\mathbf{Z}}}^2 \f$.
     */
    [[nodiscard]] auto variance() const {
        // make sure that Eigen expressions don't have dangling reference
        if constexpr (sample_scalar<sample_type>) {
            return (variance_of_real_data() + variance_of_imag_data()) / static_cast<double>(count_);
        } else {
            return ((variance_of_real_data() + variance_of_imag_data()) / static_cast<double>(count_)).eval();
        }
    }

    /**
     * @brief Calculate the sample variance \f$ s_{\mathbf{Z}}^2 \f$.
     *
     * @details It adds the results from variance_of_real_data() and variance_of_imag_data().
     *
     * @return Sample variance \f$ s_{\mathbf{Z}}^2 \f$.
     */
    [[nodiscard]] auto variance_of_data() const {
        if constexpr (sample_scalar<sample_type>) {
            return variance_of_real_data() + variance_of_imag_data();
        } else {
            // Same dangling-reference fix as in variance() above.
            return (variance_of_real_data() + variance_of_imag_data()).eval();
        }
    }

    /**
     * @brief Calculate the sample variance \f$ s_{\mathbf{X}}^2 \f$ of the real part of the data set.
     *
     * @details It calls simplemc::accs::diag_covariance with the real part of the accumulated mean
     * data \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$, the variance data \f$ \mathbf{c}_r^{(N)}/
     * \mathbf{d}_r^{(N)} \f$ and the count \f$ N \f$.
     *
     * @return Sample variance of the real part of the data set, i.e. \f$ s_{\mathbf{X}}^2 \f$.
     */
    [[nodiscard]] auto variance_of_real_data() const {
        dbl_vec_type mdata_r = mdata_.real();
        return detail::scalar_or_matrix<sample_scalar<sample_type>>(
            simplemc::accs::diag_covariance<varalg()>(mdata_r, mdata_r, rdata_, count_));
    }

    /**
     * @brief Calculate the sample variance \f$ s_{\mathbf{Y}}^2 \f$ of the imaginary part of the data
     * set.
     *
     * @details It calls simplemc::accs::diag_covariance with the imaginary part of the accumulated
     * mean data \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$, the variance data \f$ \mathbf{c}_i^{(N)}/
     * \mathbf{d}_i^{(N)} \f$ and the count \f$ N \f$.
     *
     * @return Sample variance of the imaginary part of the data set, i.e. \f$ s_{\mathbf{Y}}^2 \f$.
     */
    [[nodiscard]] auto variance_of_imag_data() const {
        dbl_vec_type mdata_i = mdata_.imag();
        return detail::scalar_or_matrix<sample_scalar<sample_type>>(
            simplemc::accs::diag_covariance<varalg()>(mdata_i, mdata_i, idata_, count_));
    }

    /**
     * @brief Calculate the sample cross-covariance \f$ \mathrm{diag}(s_{\mathbf{XY}}^2) \f$ between
     * the real and imaginary parts of the data set.
     *
     * @details It calls simplemc::accs::diag_covariance with the real and imaginary parts of the
     * accumulated mean data \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$, the cross-covariance data \f$
     * \mathbf{c}_{ri}^{(N)}/\mathbf{d}_{ri}^{(N)} \f$ and the count \f$ N \f$.
     *
     * @return Sample cross-covariance between the real and imaginary parts of the data set, i.e. \f$
     * \mathrm{diag}(s_{\mathbf{XY}}^2) \f$.
     */
    [[nodiscard]] auto covariance_of_real_and_imag_data() const {
        dbl_vec_type mdata_r = mdata_.real();
        dbl_vec_type mdata_i = mdata_.imag();
        return detail::scalar_or_matrix<sample_scalar<sample_type>>(
            simplemc::accs::diag_covariance<varalg()>(mdata_r, mdata_i, cdata_, count_));
    }

    /**
     * @brief Collect variance accumulators from different MPI processes.
     *
     * @details It reduces the accumulated mean data, (cross-co)variance data and counts from all MPI
     * processes **in place**. After the call, the accumulator on every rank holds the combined data.
     * The sticky index (see operator[]()) is reset to \f$ 0 \f$.
     *
     * The reduction operation depends on the simplemc::varalg algorithm used to accumulate the data.
     * See operator<<(const var_acc&) for how it is done in the case of 2 accumulators.
     *
     * It asserts that the size of the accumulator is equal on all processes.
     *
     * @param comm simplemc::mpi::communicator object.
     * @param acc Variance accumulator to reduce in place.
     */
    friend void simplemc_mpi_collect(const mpi::communicator& comm, var_acc& acc) {
        assert(all_equal(acc.size(), comm));
        acc.idx_ = 0;

        // reduce the count, keeping the local count for the welford scaling
        [[maybe_unused]] const auto nloc = acc.count_;
        mpi::all_reduce_in_place(acc.count_, MPI_SUM, comm);

        // early return if the reduced count is zero
        if (acc.count_ == 0) {
            return;
        }

        // reduce the accumulated data
        if constexpr (var_acc::varalg() == varalg::standard) {
            mpi::all_reduce_in_place(make_span(acc.mdata_), MPI_SUM, comm);
            mpi::all_reduce_in_place(make_span(acc.rdata_), MPI_SUM, comm);
            mpi::all_reduce_in_place(make_span(acc.idata_), MPI_SUM, comm);
            mpi::all_reduce_in_place(make_span(acc.cdata_), MPI_SUM, comm);
        } else {
            const auto n1 = static_cast<double>(nloc);
            const auto n = static_cast<double>(acc.count_);
            const cplx_vec_type m1 = acc.mdata_;
            acc.mdata_ *= n1 / n;
            mpi::all_reduce_in_place(make_span(acc.mdata_), MPI_SUM, comm);
            acc.rdata_ += n1 * (m1 - acc.mdata_).real().cwiseProduct((m1 - acc.mdata_).real());
            mpi::all_reduce_in_place(make_span(acc.rdata_), MPI_SUM, comm);
            acc.idata_ += n1 * (m1 - acc.mdata_).imag().cwiseProduct((m1 - acc.mdata_).imag());
            mpi::all_reduce_in_place(make_span(acc.idata_), MPI_SUM, comm);
            acc.cdata_ += n1 * (m1 - acc.mdata_).real().cwiseProduct((m1 - acc.mdata_).imag());
            mpi::all_reduce_in_place(make_span(acc.cdata_), MPI_SUM, comm);
        }
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
