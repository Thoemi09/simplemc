/**
 * @file
 * @brief Specialization of simplemc::covar_acc for real random vectors.
 */

#ifndef SIMPLEMC_ACCS_COVAR_ACC_COMPLEX_HPP
#define SIMPLEMC_ACCS_COVAR_ACC_COMPLEX_HPP

#include <simplemc/accs/covar_acc_fwd.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/mpi.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <range/v3/range/concepts.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/zip.hpp>

#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <utility>

namespace simplemc {

/**
 * @ingroup simplemc-accs-accs
 * @brief Specialization of simplemc::covar_acc for complex random vectors.
 *
 * @details The accumulated data is stored in four objects:
 * - a complex vector for the mean data,
 * - a real matrix for the covariance data of the real part of the random vector,
 * - a real matrix for the covariance data of the imaginary part of the random vector and
 * - a real matrix for the covariance data between the real and imaginary parts of the random vector.
 *
 * See simplemc::accs::mean and simplemc::accs::covariance.
 *
 * @code{.cpp}
 * using vec_type = Eigen::Vector<std::complex<double>, 2>;
 * std::mt19937_64 rng;
 * std::uniform_real_distribution<double> uni_dist_r(-1.0, 1.0);
 * std::normal_distribution<double> normal_dist_i(-3.0, 0.5);
 * simplemc::covar_acc<vec_type> acc;
 * for (int i = 0; i < 100000; ++i) {
 *     auto z1 = std::complex<double>{ uni_dist_r(rng), normal_dist_i(rng) };
 *     auto z2 = std::complex<double>{ uni_dist_r(rng), normal_dist_i(rng) };
 *     acc << vec_type{ z1, z2 };
 * }
 * fmt::print("Mean: [{:.5f}, {:.5f}]\n", acc.mean()(0), acc.mean()(1));
 * fmt::print("Covariance of real part:\n{}\n", simplemc::to_string(acc.covariance_of_real_data()));
 * @endcode
 *
 * Output:
 *
 * ```
 * Mean: [(0.00170,-3.00357), (-0.00070,-2.99948)]
 * Covariance of real part:
 *     0.332981 -0.000434833
 * -0.000434833     0.332989
 * ```
 *
 * @tparam Z simplemc::eigen_vector_cplx type.
 * @tparam A simplemc::varalg algorithm used to accumulate the data.
 */
template <eigen_vector_cplx Z, varalg A>
class covar_acc<Z, A> {
public:
    /**
     * @brief Type of accumulated values.
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
     * @brief Complex matrix type.
     */
    using cplx_mat_type = Eigen::Matrix<value_type, static_size, static_size>;

    /**
     * @brief Double matrix type.
     */
    using dbl_mat_type = Eigen::Matrix<double, static_size, static_size>;

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
    // Only the lower triangular part of the rdata_ and idata_ matrices are updated.
    template <ranges::input_range R1, ranges::input_range R2>
    void add_values(R1&& rg, R2&& idxs, count_type count) { // NOLINT (ranges need not be forwarded)
        int drop = 1;
        if constexpr (varalg() == varalg::standard) {
            for (auto [idx1, val1] : ranges::views::zip(idxs, rg)) {
                assert(idx1 >= 0 && idx1 < size());
                // mean data and diagonal elements of (cross-)covariance matrices
                mdata_(idx1) += val1;
                rdata_(idx1, idx1) += std::real(val1) * std::real(val1);
                idata_(idx1, idx1) += std::imag(val1) * std::imag(val1);
                cdata_(idx1, idx1) += std::real(val1) * std::imag(val1);
                // off-diagonal elements of (cross-)covariance matrices
                for (auto [idx2, val2] : ranges::views::drop(ranges::views::zip(idxs, rg), drop)) {
                    assert(idx2 > idx1 && idx2 < size());
                    // idx2 > idx1 for sorted indices -> only lower triangular matrix
                    rdata_(idx2, idx1) += std::real(val1) * std::real(val2);
                    idata_(idx2, idx1) += std::imag(val1) * std::imag(val2);
                    // the first index corresponds to the real part, the seconde to the imaginary part
                    cdata_(idx1, idx2) += std::real(val1) * std::imag(val2);
                    cdata_(idx2, idx1) += std::imag(val1) * std::real(val2);
                }
                ++drop;
            }
        } else {
            for (auto [idx1, val1] : ranges::views::zip(idxs, rg)) {
                assert(idx1 >= 0 && idx1 < size());
                // mean data and diagonal elements of covariance matrices
                const auto tmp_old = val1 - mdata_(idx1);
                mdata_(idx1) += tmp_old / static_cast<double>(count);
                const auto tmp = val1 - mdata_(idx1);
                rdata_(idx1, idx1) += std::real(tmp_old) * std::real(tmp);
                idata_(idx1, idx1) += std::imag(tmp_old) * std::imag(tmp);
                cdata_(idx1, idx1) += std::real(tmp_old) * std::imag(tmp);
                // off-diagonal elements of (cross-)covariance matrices
                for (auto [idx2, val2] : ranges::views::drop(ranges::views::zip(idxs, rg), drop)) {
                    assert(idx2 > idx1 && idx2 < size());
                    // idx2 > idx1 for sorted indices -> only lower triangular matrix
                    const auto other_tmp = val2 - mdata_(idx2);
                    rdata_(idx2, idx1) += std::real(tmp) * std::real(other_tmp);
                    idata_(idx2, idx1) += std::imag(tmp) * std::imag(other_tmp);
                    // the first index corresponds to the real part, the seconde to the imaginary part
                    cdata_(idx1, idx2) += std::real(tmp) * std::imag(other_tmp);
                    cdata_(idx2, idx1) += std::imag(tmp) * std::real(other_tmp);
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
        mdata_(cplx_vec_type::Zero(num)),
        rdata_(dbl_mat_type::Zero(num, num)),
        idata_(dbl_mat_type::Zero(num, num)),
        cdata_(dbl_mat_type::Zero(num, num)),
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
     * >= 1. Otherwise, it throws a simplemc::simplemc_exception.
     *
     * @param md Accumulated mean data.
     * @param rd Accumulated covariance data of the real part.
     * @param id Accumulated covariance data of the imaginary part.
     * @param cd Accumulated cross-covariance data between the real and imaginary part.
     * @param ct Number of accumulated values.
     */
    covar_acc(const cplx_vec_type& md, const dbl_mat_type& rd, const dbl_mat_type& id, const dbl_mat_type& cd,
        count_type ct) :
        mdata_(md),
        rdata_(rd),
        idata_(id),
        cdata_(cd),
        count_(ct),
        idx_(0) {
        if constexpr (is_dynamic) {
            if (mdata_.size() <= 0) {
                throw simplemc_exception("Dynamic size <= 0", "covar_acc::covar_acc");
            }
            if (mdata_.size() != rdata_.rows() || mdata_.size() != rdata_.cols() || mdata_.size() != idata_.rows() ||
                mdata_.size() != idata_.cols() || mdata_.size() != cdata_.rows() || mdata_.size() != cdata_.cols()) {
                throw simplemc_exception("Sizes of data storages do not match", "covar_acc::covar_acc");
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
            rdata_ += (vec.matrix().real() * vec.matrix().real().transpose()).template triangularView<Eigen::Lower>();
            idata_ += (vec.matrix().imag() * vec.matrix().imag().transpose()).template triangularView<Eigen::Lower>();
            cdata_ += (vec.matrix().real() * vec.matrix().imag().transpose());
        } else {
            const auto tmp = (vec.matrix() - mdata_).eval();
            mdata_ += tmp / static_cast<double>(count_);
            const auto tmp2 = vec.matrix() - mdata_;
            rdata_ += (tmp.real() * tmp2.real().transpose()).template triangularView<Eigen::Lower>();
            idata_ += (tmp.imag() * tmp2.imag().transpose()).template triangularView<Eigen::Lower>();
            cdata_ += (tmp.real() * tmp2.imag().transpose());
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
     *   - \f$ \mathbf{c}^{(N)} = \mathbf{c}_{1}^{(N_1)} + \mathbf{c}_{2}^{(N_2)} \f$ .
     *
     *   Here, \f$ \mathbf{c} \f$ stands for any of the accumulated covariance data, i.e. the
     *   covariance of the real part, the covariance of the imaginary part or the covariance between
     *   the real and imaginary parts.
     *
     * - `welford`:
     *   - \f$ \mathbf{n}^{(N)} = \frac{N_1}{N} \mathbf{n}_{1}^{(N_1)} +
     *     \frac{N_2}{N} \mathbf{n}_{2}^{(N_2)} \f$ and
     *   - \f$ \mathbf{D}^{(N)} = \mathbf{D}_{1}^{(N_1)} + \mathbf{D}_{2}^{(N_2)} + N_1 \left(
     *     \mathbf{n}_{1}^{(N_1)} - \mathbf{n}^{(N)} \right) \left( \mathbf{n}_{1}^{(N_1)} -
     *     \mathbf{n}^{(N)} \right)^T + N_2 \left( \mathbf{n}_{2}^{(N_2)} - \mathbf{n}^{(N)} \right)
     *     \left( \mathbf{n}_{2}^{(N_2)} - \mathbf{n}^{(N)} \right)^T \f$ .
     *
     *   Here, \f$ \mathbf{D} \f$ stands for the accumulated covariance data of the real part. For the
     *   covariance data of the imaginary part and the covariance data between the real and imaginary
     *   parts, similar expressions hold.
     *
     * @param acc Variance accumulator to be incorporated.
     * @return Reference to this object.
     */
    covar_acc& operator<<(const covar_acc& acc) {
        assert(size() == acc.size());
        if constexpr (varalg() == varalg::standard) {
            mdata_ += acc.mdata_;
            rdata_ += acc.rdata_.template triangularView<Eigen::Lower>();
            idata_ += acc.idata_.template triangularView<Eigen::Lower>();
            cdata_ += acc.cdata_;
        } else {
            const auto n1 = static_cast<double>(count_);
            const auto n2 = static_cast<double>(acc.count_);
            const auto m = mdata_ * n1 / (n1 + n2) + acc.mdata_ * n2 / (n1 + n2);
            rdata_ += (acc.rdata_ + n1 * (mdata_ - m).real() * (mdata_ - m).real().transpose() +
                n2 * (acc.mdata_ - m).real() * (acc.mdata_ - m).real().transpose())
                          .template triangularView<Eigen::Lower>();
            idata_ += (acc.idata_ + n1 * (mdata_ - m).imag() * (mdata_ - m).imag().transpose() +
                n2 * (acc.mdata_ - m).imag() * (acc.mdata_ - m).imag().transpose())
                          .template triangularView<Eigen::Lower>();
            cdata_ += acc.cdata_ + n1 * (mdata_ - m).real() * (mdata_ - m).imag().transpose() +
                n2 * (acc.mdata_ - m).real() * (acc.mdata_ - m).imag().transpose();
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
    [[nodiscard]] const cplx_vec_type& mdata() const { return mdata_; }

    /**
     * @brief Get accumulated data used for estimating the covariance matrix of the real part.
     *
     * @warning Only the lower triangular part of the covariance matrix is valid. Use
     * `selfadjointView<Eigen::Lower>()` to get the full covariance matrix.
     *
     * @return Data storage (content depends on the algorithm, see simplemc::accs::covariance).
     */
    [[nodiscard]] const dbl_mat_type& rdata() const { return rdata_; }

    /**
     * @brief Get accumulated data used for estimating the covariance matrix of the imaginary part.
     *
     * @warning Only the lower triangular part of the covariance matrix is valid. Use
     * `selfadjointView<Eigen::Lower>()` to get the full covariance matrix.
     *
     * @return Data storage (content depends on the algorithm, see simplemc::accs::covariance).
     */
    [[nodiscard]] const dbl_mat_type& idata() const { return idata_; }

    /**
     * @brief Get accumulated data used for estimating the cross-covariance matrix of the real and
     * imaginary part.
     *
     * @details The real part is the row index and the imaginary part is the column index.
     *
     * @return Data storage (content depends on the algorithm, see simplemc::accs::covariance).
     */
    [[nodiscard]] const dbl_mat_type& cdata() const { return cdata_; }

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
     * @details It calls covariance_of_data() and divides the result by the count.
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a vector object.
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
     * @details It combines the different covariance matrices (real, imaginary, real-imaginary) to
     * form the full covariance matrix (see @ref simplemc-accs for the formula).
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a vector object.
     *
     * @return Sample covariance matrix of the data.
     */
    [[nodiscard]] auto covariance_of_data() const {
        using namespace std::complex_literals;
        auto res = covariance_of_real_data() + covariance_of_imag_data() +
            (0.0 + 1.0i) * (covariance_of_real_and_imag_data().transpose() - covariance_of_real_and_imag_data());
        if constexpr (returns_scalar) {
            return res;
        } else {
            return res.eval();
        }
    }

    /**
     * @brief Calculate the sample covariance matrix of the real part of the accumulated data.
     *
     * @details Calls simplemc::accs::covariance with the accumulated data and the count.
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a vector.
     *
     * @return Sample covariance matrix of the real part of the accumulated data.
     */
    [[nodiscard]] auto covariance_of_real_data() const {
        using simplemc::accs::covariance;
        dbl_mat_type rdata_full = rdata_.template selfadjointView<Eigen::Lower>();
        dbl_vec_type mdata_r = mdata_.real();
        return detail::scalar_or_matrix<returns_scalar>(covariance<varalg()>(mdata_r, mdata_r, rdata_full, count_));
    }

    /**
     * @brief Calculate the sample covariance matrix of the imaginary part of the accumulated data.
     *
     * @details Calls simplemc::accs::covariance with the accumulated data and the count.
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a vector.
     *
     * @return Sample covariance matrix of the imaginary part of the accumulated data.
     */
    [[nodiscard]] auto covariance_of_imag_data() const {
        using simplemc::accs::covariance;
        dbl_mat_type idata_full = idata_.template selfadjointView<Eigen::Lower>();
        dbl_vec_type mdata_i = mdata_.imag();
        return detail::scalar_or_matrix<returns_scalar>(covariance<varalg()>(mdata_i, mdata_i, idata_full, count_));
    }

    /**
     * @brief Calculate the sample cross-covariance matrix between the real and imaginary part of the
     * accumulated data.
     *
     * @details Calls simplemc::accs::covariance with the accumulated data and the count.
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a vector.
     *
     * @return Sample cross-covariance matrix between the real and imaginary part of the accumulated data.
     */
    [[nodiscard]] auto covariance_of_real_and_imag_data() const {
        using simplemc::accs::covariance;
        dbl_vec_type mdata_r = mdata_.real();
        dbl_vec_type mdata_i = mdata_.imag();
        return detail::scalar_or_matrix<returns_scalar>(covariance<varalg()>(mdata_r, mdata_i, cdata_, count_));
    }

    /**
     * @brief Collect covariance accumulators from different MPI processes.
     *
     * @details It constructs a new covariance accumulator with the reduced accumulated mean data,
     * covariance data and counts from all MPI processes.
     *
     * The reduction operation depends on the simplemc::varalg algorithm used to accumulate the data.
     * See operator<<(const covar_acc&) for how it is done in the case of 2 accumulators.
     *
     * @param comm simplemc::mpi::communicator object.
     * @param acc Covariance accumulator.
     * @return Covariance accumulator with the reduced data from all processes.
     */
    friend covar_acc mpi_collect(const mpi::communicator& comm, const covar_acc& acc) {
        covar_acc res(acc.size());
        mpi::all_reduce(comm, acc.count_, res.count_, MPI_SUM);
        if constexpr (covar_acc::varalg() == varalg::standard) {
            mpi::all_reduce(comm, make_span(acc.mdata_), make_span(res.mdata_), MPI_SUM);
            mpi::all_reduce(comm, make_span(acc.rdata_), make_span(res.rdata_), MPI_SUM);
            mpi::all_reduce(comm, make_span(acc.idata_), make_span(res.idata_), MPI_SUM);
            mpi::all_reduce(comm, make_span(acc.cdata_), make_span(res.cdata_), MPI_SUM);
        } else {
            const auto n1 = static_cast<double>(acc.count_);
            const auto n = static_cast<double>(res.count_);
            const cplx_vec_type tmp_mdata = acc.mdata_ * n1 / n;
            mpi::all_reduce(comm, make_span(tmp_mdata), make_span(res.mdata_), MPI_SUM);
            const dbl_mat_type tmp_cdata =
                acc.cdata_ + n1 * (acc.mdata_ - res.mdata_).real() * (acc.mdata_ - res.mdata_).imag().transpose();
            mpi::all_reduce(comm, make_span(tmp_cdata), make_span(res.cdata_), MPI_SUM);
            // we cannot add the triangular view to the full matrix (only when we assign)
            dbl_mat_type tmp_rdata = acc.rdata_;
            tmp_rdata += (n1 * (acc.mdata_ - res.mdata_).real() * (acc.mdata_ - res.mdata_).real().transpose())
                             .template triangularView<Eigen::Lower>();
            mpi::all_reduce(comm, make_span(tmp_rdata), make_span(res.rdata_), MPI_SUM);
            dbl_mat_type tmp_idata = acc.idata_;
            tmp_idata += (n1 * (acc.mdata_ - res.mdata_).imag() * (acc.mdata_ - res.mdata_).imag().transpose())
                             .template triangularView<Eigen::Lower>();
            mpi::all_reduce(comm, make_span(tmp_idata), make_span(res.idata_), MPI_SUM);
        }
        return res;
    }

private:
    cplx_vec_type mdata_;
    dbl_mat_type rdata_;
    dbl_mat_type idata_;
    dbl_mat_type cdata_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_COVAR_ACC_COMPLEX_HPP
