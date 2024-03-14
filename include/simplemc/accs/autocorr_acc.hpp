/**
 * @file autocorr_acc.hpp
 * @brief Wrapper for simplemc::var_acc and simplemc::covar_acc to estimate the autocorrelation time.
 */

#ifndef SIMPLEMC_ACCS_AUTOCORR_ACC_HPP
#define SIMPLEMC_ACCS_AUTOCORR_ACC_HPP

#include <simplemc/accs/covar_acc_complex.hpp>
#include <simplemc/accs/covar_acc_double.hpp>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/accs/var_acc_complex.hpp>
#include <simplemc/accs/var_acc_double.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <range/v3/range/concepts.hpp>
#include <range/v3/view/iota.hpp>

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

namespace simplemc {

/**
 * @brief Wrapper class for the simplemc::var_acc and simplemc::covar_acc accumulators to estimate
 * the autocorrelation time using the blocking method.
 *
 * @details It uses blocks of increasing size to decorrelate the individual samples. The (co)variance
 * and the integrated autocorrelation time will increase with the block size until it reaches a plateau.
 * The value at the plateau should give you a good estimate of both quantities.
 *
 * Functionality and usage is similar to the supported wrapped accumulators. The autocorrelation time
 * is calculated for each element of the wrapped accumulator, e.g. if you wrap a simplemc::covar_acc,
 * you will get back a matrix.
 *
 * Multi value accumulation is not supported right now (please use the accumulate(..) function instead).
 *
 * @tparam A Accumulator type.
 */
template <typename A>
class autocorr_acc {
public:
    /**
     * @brief Accumulator type.
     */
    using acc_type = A;

    /**
     * @brief Type of accumulated values.
     */
    using value_type = typename acc_type::value_type;

    /**
     * @brief Type for counting the number of accumulated values.
     */
    using count_type = typename acc_type::count_type;

    /**
     * @brief Size type of the accumulator.
     */
    using size_type = typename acc_type::size_type;

    /**
     * @brief Get the algorithm.
     */
    static constexpr auto varalg() { return A::varalg(); }

    /**
     * @brief Mean accumulator type for accumulating block data.
     */
    using mean_acc_type = mean_acc<value_type, varalg()>;

    /**
     * @brief Calculate the integrated autocorrelation time for the diagonal elements of a
     * (cross)-covariance matrix.
     *
     * @param c0 Naive (unblocked) estimate of the diagonal of the (cross)-covariance matrix.
     * @param cb Blocked estimate of the diagonal of the (cross)-covariance matrix.
     * @param blsize Block size of cb w.r.t. c0.
     * @return Vector of integrated autocorrelation times.
     */
    [[nodiscard]] static Eigen::VectorXd tau(const Eigen::VectorXd& c0, const Eigen::VectorXd& cb, count_type blsize) {
        assert(c0.size() == cb.size());
        return ((cb.array() * blsize / c0.array() - 1.0) * 0.5).matrix();
    }

    /**
     * @brief Calculate the integrated autocorrelation time for the elements of a (cross)-covariance
     * matrix.
     *
     * @param c0 Naive (unblocked) estimate of the (cross)-covariance matrix.
     * @param cb Blocked estimate of the (cross)-covariance matrix.
     * @param blsize Block size of cb w.r.t. c0.
     * @return Matrix of integrated autocorrelation times.
     */
    template <typename T>
    [[nodiscard]] static Eigen::MatrixXd tau(const Eigen::MatrixXd& c0, const Eigen::MatrixXd& cb, count_type blsize) {
        assert(c0.rows() == cb.rows());
        assert(c0.cols() == cb.cols());
        return ((cb.array() * blsize / c0.array() - 1.0) * 0.5).matrix();
    }

private:
    /**
     * @brief Check if level i is full?
     *
     * @param i Level index.
     * @return True if the block in level i is full, false otherwise.
     */
    [[nodiscard]] auto is_full(std::size_t i) const {
        assert(i < accs_.size());
        return blocks_[i].count() >= blsizes_[i];
    }

    /**
     * @brief Add the block in level i to its effective samples and reset its block data.
     *
     * @param i Level index.
     */
    void add_block(std::size_t i) {
        assert(i < accs_.size());
        accs_[i].accumulate(blocks_[i].mean());
        blocks_[i].reset();
    }

    /**
     * @brief Propagate the given level to the next higher level recursively.
     *
     * @param i Level index.
     */
    void propagate(std::size_t i) {
        assert(i < accs_.size());
        if (is_full(i)) {
            if (i == accs_.size() - 1) {
                add_level();
            }
            blocks_[i + 1] << blocks_[i];
            add_block(i);
            propagate(i + 1);
        }
    }

    /**
     * @brief Add a new level to the accumulator.
     */
    void add_level() {
        accs_.emplace_back(shift());
        blocks_.emplace_back(shift());
        blsizes_.emplace_back(blsizes_.back() * fac_);
    }

    /**
     * @brief Create the vector of increasing block sizes.
     *
     * @param nl Number of levels.
     * @param fac Multiplication factor.
     * @return Vector of increasing block sizes.
     */
    [[nodiscard]] auto make_blsizes(std::size_t nl, count_type fac) const {
        std::vector<count_type> res(nl, 1);
        for (std::size_t i = 1; i < nl; ++i) {
            res[i] = res[i - 1] * fac;
        }
        return res;
    }

public:
    /**
     * @brief Construct an autocorrelation accumulator with a given number of elements, a constant shift, a
     * multiplication factor for increasing block sizes and a minimum number of levels.
     *
     * @param size Number of elements.
     * @param shift A single constant shift applied to the accumulated values.
     * @param fac Multiplication factor.
     * @param min_levels Minimum number of levels.
     */
    explicit autocorr_acc(size_type size = 1, value_type shift = 0.0, count_type fac = 2, std::size_t min_levels = 2) :
        accs_(min_levels, acc_type(size, shift)),
        blocks_(min_levels, mean_acc_type(size, shift)),
        blsizes_(make_blsizes(min_levels, fac)),
        idx_(0),
        fac_(fac),
        min_levels_(min_levels) {
        check_levels();
    }

    /**
     * @brief Construct an autocorrelation accumulator with a given constant shift, multiplication factor for
     * increasing block sizes and a minimum number of levels.
     *
     * @details The size of the accumulator will be the same as the size of the shift vector.
     *
     * @param shift Constant shift applied to the accumulated values.
     * @param fac Multiplication factor.
     * @param min_levels Minimum number of levels.
     */
    explicit autocorr_acc(const Eigen::VectorX<value_type>& shift, count_type fac = 2, std::size_t min_levels = 2) :
        accs_(min_levels, acc_type(shift)),
        blocks_(min_levels, mean_acc_type(shift)),
        blsizes_(make_blsizes(min_levels, fac)),
        idx_(0),
        fac_(fac),
        min_levels_(min_levels) {
        check_levels();
    }

    /**
     * @brief Construct an autocorrelation accumulator from a given vector of (co)variance accumulators, a vector of
     * mean accumulators for blocking, a multiplication factor for increasing block sizes and a minimum number of
     * levels.
     *
     * @param accs Vector of (co)variance accumulators.
     * @param blocks Vector of mean accumulators.
     * @param fac Multiplication factor.
     * @param min_levels Minimum number of levels.
     */
    explicit autocorr_acc(
        std::vector<acc_type> accs, std::vector<mean_acc_type> blocks, count_type fac, std::size_t min_levels) :
        accs_(std::move(accs)),
        blocks_(std::move(blocks)),
        blsizes_(make_blsizes(accs_.size(), fac)),
        idx_(0),
        fac_(fac),
        min_levels_(min_levels) {
        check_levels();
    }

    /**
     * @brief Reset the accumulator to its initial state, i.e. with no accumulated values.
     */
    void reset() {
        accs_ = std::vector<acc_type>(min_levels_, acc_type(shift()));
        blocks_ = std::vector<mean_acc_type>(min_levels_, mean_acc_type(shift()));
        blsizes_ = make_blsizes(min_levels_, fac_);
    }

    /**
     * @brief Subscript operator sets the index and returns a reference to this object.
     *
     * @param idx Index.
     * @return Reference to this object.
     */
    autocorr_acc& operator[](size_type idx) {
        idx_ = idx;
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a single value.
     *
     * @param val Value to be accumulated.
     * @return Reference to this object.
     */
    autocorr_acc& operator<<(value_type val) {
        accumulate(std::array<value_type, 1> { val }, std::array<size_type, 1> { idx_ });
        return *this;
    }

    /**
     * @brief Accumulate a range of values.
     *
     * @details The size of the range is assumed to be <= size() - idx.
     *
     * @tparam R Input range of values.
     * @param rg Range of values to be accumulated.
     * @param idx Starting index for the accumulator.
     */
    template <ranges::input_range R>
    void accumulate(R&& rg, size_type idx = 0) {
        accumulate(std::forward<R>(rg), ranges::views::iota(idx));
    }

    /**
     * @brief Accumulate a range of values.
     *
     * @details The size of the range is assumed to be <= size() - idx. In case that the wrapped accumulator is
     *
     * @tparam R1 Input range of values.
     * @tparam R2 Input range of indices.
     * @param rg Range of values to be accumulated.
     * @param idxs Range of indices.
     */
    template <ranges::input_range R1, ranges::input_range R2>
    void accumulate(R1&& rg, R2&& idxs) {
        // always accumulate directly in the (co)variance accumulator of level 0
        accs_[0].accumulate(std::forward<R1>(rg), std::forward<R2>(idxs));

        // accumulate in all levels < min_levels_ - 1
        for (std::size_t i = 1; i < min_levels_ - 1; ++i) {
            blocks_[i].accumulate(std::forward<R1>(rg), std::forward<R2>(idxs));
            if (is_full(i)) {
                add_block(i);
            }
        }

        // accumulate in the last min. level and propagate
        blocks_[min_levels_ - 1].accumulate(std::forward<R1>(rg), std::forward<R2>(idxs));
        propagate(min_levels_ - 1);
    }

    /**
     * @brief Get the size of the accumulator.
     *
     * @return Number of elements.
     */
    [[nodiscard]] auto size() const { return accs_[0].size(); }

    /**
     * @brief Get the multiplication factor for increasing block sizes.
     *
     * @return Multiplication factor.
     */
    [[nodiscard]] auto factor() const { return fac_; }

    /**
     * @brief Get number of current levels.
     *
     * @return Number of levels.
     */
    [[nodiscard]] auto num_levels() const { return accs_.size(); }

    /**
     * @brief Get effective number of samples taken in level i.
     *
     * @param i Level index.
     * @return Number of accumulated samples in level i.
     */
    [[nodiscard]] auto count(std::size_t i = 0) const {
        assert(i < accs_.size());
        return accs_[i].count();
    }

    /**
     * @brief Get the block size of level i.
     *
     * @param i Level index.
     * @return Block size of level i.
     */
    [[nodiscard]] auto block_size(std::size_t i = 0) const {
        assert(i < accs_.size());
        return blsizes_[i];
    }

    /**
     * @brief Get the constant shift.
     *
     * @return Constant shift vector applied to the accumulated values.
     */
    [[nodiscard]] const auto& shift() const { return accs_[0].shift(); }

    /**
     * @brief Get the vector of (co)variance accumulators.
     *
     * @return Vector of (co)variance accumulators.
     */
    [[nodiscard]] const auto& accumulators() const { return accs_; }

    /**
     * @brief Get the vector of block data.
     *
     * @return Vector of block data.
     */
    [[nodiscard]] const auto& blocks() const { return accs_; }

    /**
     * @brief Get the vector of block sizes.
     *
     * @return Vector of block sizes.
     */
    [[nodiscard]] const auto& block_sizes() const { return blsizes_; }

    /**
     * @brief Find the highest level with at least a given number of effective samples.
     *
     * @param min Minimum number of effective samples.
     * @return Level index.
     */
    [[nodiscard]] std::size_t find_level(count_type min) const {
        for (auto i = accs_.size(); i > 1; --i) {
            if (accs_[i - 1].count() >= min) {
                return i - 1;
            }
        }
        return 0;
    }

    /**
     * @brief Calculate the sample mean from the accumulated data in level 0.
     *
     * @return Sample mean.
     */
    [[nodiscard]] auto mean() const { return accs_[0].mean(); }

    /**
     * @brief Check accumulators and parameters for consistency, otherwise throw an exception.
     */
    void check_levels() const {
        if (fac_ <= 1) {
            throw simplemc_exception("Multiplication factor for block sizes <= 1", "autocorr_acc::check_levels");
        }
        if (min_levels_ < 2) {
            throw simplemc_exception("Minimum number of levels < 2", "autocorr_acc::check_levels");
        }
        if (accs_.size() != blocks_.size()) {
            throw simplemc_exception(
                "Number of (co)variance accs != number of mean accs", "autocorr_acc::check_levels");
        }
        if (accs_.size() < min_levels_) {
            throw simplemc_exception("Number of levels < minimum number of levels", "autocorr_acc::check_levels");
        }
        count_type bs = 1;
        auto ct = accs_[0].count();
        for (std::size_t i = 0; i < accs_.size(); ++i) {
            if (accs_[i].size() != size()) {
                throw simplemc_exception("Size mismatch between accumulators", "autocorr_acc::check_levels");
            }
            if (blocks_[i].size() != size()) {
                throw simplemc_exception("Size mismatch between blocks", "autocorr_acc::check_levels");
            }
            if (accs_[i].count() != ct) {
                throw simplemc_exception("Wrong sample count", "autocorr_acc::check_levels");
            }
            if (blsizes_[i] != bs) {
                throw simplemc_exception("Wrong block size", "autocorr_acc::check_levels");
            }
            ct /= fac_;
            bs *= fac_;
        }
    }

private:
    std::vector<acc_type> accs_;
    std::vector<mean_acc_type> blocks_;
    std::vector<count_type> blsizes_;
    size_type idx_ { 0 };
    count_type fac_ { 2 };
    std::size_t min_levels_ { 2 };
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_AUTOCORR_ACC_HPP
