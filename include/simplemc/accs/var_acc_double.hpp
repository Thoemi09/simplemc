/**
 * @file
 * @brief Accumulator for calculating the sample mean and sample variance of a real random vector.
 */

#ifndef SIMPLEMC_ACCS_VAR_ACC_DOUBLE_HPP
#define SIMPLEMC_ACCS_VAR_ACC_DOUBLE_HPP

#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/multivalue_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/accs/var_acc_fwd.hpp>
#include <simplemc/accs/varalg.hpp>
#include <simplemc/mpi.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <Eigen/Dense>

#include <cassert>
#include <cstdint>

namespace simplemc {

/**
 * @ingroup simplemc-accs-accs-var
 * @brief Accumulator for calculating the sample mean and sample variance of a real random vector
 * \f$ \mathbf{X} \f$.
 *
 * @details The accumulator takes two template parameters:
 * - the type of the random samples (a simplemc::eigen_vector_dbl type) and
 * - the algorithm (simplemc::varalg) that should be used to accumulate the data.
 *
 * Both of them determine how the accumulation is actually done and what is stored in the
 * accumulator. The mean data \f$ \mathbf{m}^{(N)}/\mathbf{n}^{(N)} \f$ is accumulated as
 * described in simplemc::mean_acc. The variance data \f$ \mathbf{c}^{(N)}/\mathbf{d}^{(N)} \f$
 * depends on the algorithm:
 * - `standard`: The variance data is accumulated with
 *   \f[
 *     \mathbf{c}^{(N)} = \mathbf{c}^{(N-1)} + \left( \mathbf{x}^{(N)} - \mathbf{t} \right)
 *     \odot \left( \mathbf{x}^{(N)} - \mathbf{t} \right) =
 *     \sum_{j=1}^N \left( \mathbf{x}^{(j)} - \mathbf{t} \right)^{\odot 2} \; ,
 *   \f]
 *   such that the sample variance is given by
 *   \f[
 *     s_{\mathbf{X}}^2 = \frac{1}{N - 1} \left\{
 *     \mathbf{c}^{(N)} - \frac{\mathbf{m}^{(N)} \odot \mathbf{m}^{(N)}}{N}
 *     \right\} \; .
 *   \f]
 *
 * - `welford`: The variance data is accumulated with
 *   \f[
 *     \mathbf{d}^{(N)} = \mathbf{d}^{(N-1)} + \left( \mathbf{x}^{(N)} - \mathbf{t} -
 *     \mathbf{n}^{(N-1)} \right) \odot \left( \mathbf{x}^{(N)} - \mathbf{t} -
 *     \mathbf{n}^{(N)} \right) \; ,
 *   \f]
 *   such that the sample variance is given by
 *   \f[
 *     s_{\mathbf{X}}^2 = \frac{\mathbf{d}^{(N)}}{N - 1} \; .
 *   \f]
 *
 * Here, \f$ \mathbf{t} \f$ is a constant vector that can optionally be applied to the random
 * samples to increase numerical accuracy and \f$ \mathbf{a} \odot \mathbf{b} \f$ denotes the
 * Hadamard (element-wise) product. See also @ref simplemc-accs-stats-var.
 *
 * @code{.cpp}
 * #include <fmt/ranges.h>
 * #include <simplemc/accs.hpp>
 *
 * #include <random>
 *
 * int main() {
 *     // normal distribution to be sampled: mu = 5, sigma = 1
 *     std::mt19937_64 rng;
 *     std::normal_distribution<double> normal_dist(5.0, 1.0);
 *
 *     // accumulate samples into a variance accumulator of size 1
 *     simplemc::var_acc_single<double> acc;
 *     for (int i = 0; i < 100000; ++i) {
 *         acc << normal_dist(rng);
 *     }
 *
 *     // print the mean and variance of the accumulated data
 *     fmt::println("Mean: {}", acc.mean());
 *     fmt::println("Variance: {}", acc.variance_of_data());
 * }
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
            cdata_(idx) += val * val;
        } else {
            const auto tmp = val - mdata_(idx);
            mdata_(idx) += tmp / static_cast<double>(count);
            cdata_(idx) += tmp * (val - mdata_(idx));
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
        mdata_(vec_type::Zero(is_dynamic ? m : static_size)),
        cdata_(vec_type::Zero(is_dynamic ? m : static_size)),
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
     * @param cd Accumulated variance data \f$ \mathbf{c}^{(N)}/\mathbf{d}^{(N)} \f$.
     * @param n Number of accumulated samples \f$ N \f$.
     */
    var_acc(const vec_type& md, const vec_type& cd, count_type n) : mdata_(md), cdata_(cd), count_(n), idx_(0) {
        if constexpr (is_dynamic) {
            if (mdata_.size() <= 0) {
                throw simplemc_exception("Dynamic size <= 0", "var_acc::var_acc");
            }
            if (mdata_.size() != cdata_.size()) {
                throw simplemc_exception("Sizes of data storages do not match", "var_acc::var_acc");
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
     * @brief Stream operator for accumulating a single value \f$ x \f$.
     *
     * @details For accumulators with size \f$ M > 1 \f$, the value is added to the element at the
     * current index \f$ i \f$ (see operator[]()).
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @tparam T Type of the value to be accumulated.
     * @param x Value \f$ x \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename T>
        requires std::convertible_to<T, value_type>
    var_acc& operator<<(const T& x) {
        ++count_;
        add_value(x, idx_, count_);
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
            cdata_ += v.matrix().cwiseProduct(v.matrix());
        } else {
            const auto tmp = (v.matrix() - mdata_).eval();
            mdata_ += tmp / static_cast<double>(count_);
            cdata_ += tmp.cwiseProduct(v.matrix() - mdata_);
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
     *     \odot \left( \mathbf{n}_{2}^{(N_2)} - \mathbf{n}^{(N)} \right) \f$.
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @param acc_other simplemc::var_acc<X, A> to be incorporated.
     * @return Reference to `this` object.
     */
    var_acc& operator<<(const var_acc& acc_other) {
        assert(size() == acc_other.size());
        if constexpr (varalg() == varalg::standard) {
            mdata_ += acc_other.mdata_;
            cdata_ += acc_other.cdata_;
        } else {
            const auto n1 = static_cast<double>(count_);
            const auto n2 = static_cast<double>(acc_other.count_);
            const auto m = mdata_ * n1 / (n1 + n2) + acc_other.mdata_ * n2 / (n1 + n2);
            cdata_ += acc_other.cdata_ + n1 * (mdata_ - m).cwiseProduct(mdata_ - m) +
                n2 * (acc_other.mdata_ - m).cwiseProduct(acc_other.mdata_ - m);
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
     * @return simplemc::eigen_vector_dbl of size \f$ M \f$ containing \f$ \mathbf{m}^{(N)}/
     * \mathbf{n}^{(N)} \f$ (content depends on the algorithm, see simplemc::mean_acc).
     */
    [[nodiscard]] const vec_type& mdata() const { return mdata_; }

    /**
     * @brief Get the accumulated data \f$ \mathbf{c}^{(N)}/\mathbf{d}^{(N)} \f$ used for estimating
     * the variance.
     *
     * @return simplemc::eigen_vector_dbl of size \f$ M \f$ containing \f$ \mathbf{c}^{(N)}/
     * \mathbf{d}^{(N)} \f$ (content depends on the algorithm, see the class documentation).
     */
    [[nodiscard]] const vec_type& cdata() const { return cdata_; }

    /**
     * @brief Calculate the sample mean \f$ \overline{\mathbf{x}}^{(N)} \f$.
     *
     * @details It calls simplemc::accs::mean with the accumulated mean data and the
     * count.
     *
     * For statically sized accumulators with \f$ M = 1 \f$, it returns a real scalar. Otherwise, it
     * returns a vec_type object.
     *
     * @return Sample mean \f$ \overline{\mathbf{x}}^{(N)} \f$.
     */
    [[nodiscard]] auto mean() const {
        return detail::scalar_or_matrix<returns_scalar>(simplemc::accs::mean<varalg()>(mdata_, count_));
    }

    /**
     * @brief Calculate the sample variance \f$ s_{\overline{\mathbf{X}}}^2 \f$ of the mean.
     *
     * @details It calls variance_of_data() and divides the result by count().
     *
     * For statically sized accumulators with \f$ M = 1 \f$, it returns a real scalar. Otherwise, it
     * returns a vec_type object.
     *
     * @return Sample variance of the mean \f$ s_{\overline{\mathbf{X}}}^2 \f$.
     */
    [[nodiscard]] auto variance() const {
        auto res = variance_of_data() / static_cast<double>(count_);
        if constexpr (returns_scalar) {
            return res;
        } else {
            return res.eval();
        }
    }

    /**
     * @brief Calculate the sample variance \f$ s_{\mathbf{X}}^2 \f$.
     *
     * @details It calls simplemc::accs::diag_covariance with the accumulated data and
     * the count.
     *
     * For statically sized accumulators with \f$ M = 1 \f$, it returns a real scalar. Otherwise, it
     * returns a vec_type object.
     *
     * @return Sample variance \f$ s_{\mathbf{X}}^2 \f$.
     */
    [[nodiscard]] auto variance_of_data() const {
        using simplemc::accs::diag_covariance;
        return detail::scalar_or_matrix<returns_scalar>(diag_covariance<varalg()>(mdata_, mdata_, cdata_, count_));
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
            mpi::all_reduce(make_span(acc.cdata_), make_span(res.cdata_), MPI_SUM, comm);
        } else {
            const auto n1 = static_cast<double>(acc.count_);
            const auto n = static_cast<double>(res.count_);
            const vec_type tmp_mdata = acc.mdata_ * n1 / n;
            mpi::all_reduce(make_span(tmp_mdata), make_span(res.mdata_), MPI_SUM, comm);
            const vec_type tmp_vdata =
                acc.cdata_ + n1 * (acc.mdata_ - res.mdata_).cwiseProduct(acc.mdata_ - res.mdata_);
            mpi::all_reduce(make_span(tmp_vdata), make_span(res.cdata_), MPI_SUM, comm);
        }
        return res;
    }

private:
    vec_type mdata_;
    vec_type cdata_;
    count_type count_;
    size_type idx_;
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_VAR_ACC_DOUBLE_HPP
