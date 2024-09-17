/**
 * @file
 * @brief Wrapper for simplemc::var_acc and simplemc::covar_acc to estimate the integrated
 * autocorrelation time.
 */

#ifndef SIMPLEMC_ACCS_AUTOCORR_ACC_HPP
#define SIMPLEMC_ACCS_AUTOCORR_ACC_HPP

#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <range/v3/range/concepts.hpp>
#include <range/v3/view/iota.hpp>

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

namespace simplemc {

/**
 * @ingroup simplemc-accs-wrappers
 * @brief Wrapper for simplemc::var_acc and simplemc::covar_acc to estimate the integrated
 * autocorrelation time.
 *
 * @details It uses blocks of increasing size to decorrelate the individual samples. The (co)variance
 * and the integrated autocorrelation time will increase with the block size until it reaches a
 * plateau. The value at the plateau should give you a good estimate of both quantities.
 *
 * Functionality and usage is similar to the supported wrapped accumulators. Multi-value accumulation
 * is not supported right now (please use the `accumulate()` instead).
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
     * @brief Static size of the accumulator.
     */
    static constexpr int static_size = acc_type::static_size;

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
     * @brief Get the algorithm used to accumulate the data.
     *
     * @return The simplemc::varalg tag.
     */
    static constexpr auto varalg() { return A::varalg(); }

    /**
     * @brief Mean accumulator type for accumulating block data.
     */
    using mean_acc_type = mean_acc<Eigen::Matrix<value_type, static_size, 1>, varalg()>;

private:
    // Check if the given level is full?
    [[nodiscard]] auto is_full(std::size_t lvl) const {
        assert(lvl < accs_.size());
        return blocks_[lvl].count() >= blsizes_[lvl];
    }

    // Add the block in the given level to its effective samples and reset its block data.
    void add_block(std::size_t lvl) {
        assert(lvl < accs_.size());
        accs_[lvl].accumulate(blocks_[lvl].mean());
        blocks_[lvl].reset();
    }

    // Propagate the given level to the next higher level recursively.
    void propagate(std::size_t lvl) {
        if (is_full(lvl)) {
            if (lvl == accs_.size() - 1) {
                add_level();
            }
            blocks_[lvl + 1] << blocks_[lvl];
            add_block(lvl);
            propagate(lvl + 1);
        }
    }

    // Add a new level to the accumulator.
    void add_level() {
        accs_.emplace_back(size());
        blocks_.emplace_back(size());
        blsizes_.emplace_back(blsizes_.back() * fac_);
    }

    // Create a vector of increasing block sizes for the number of given levels.
    [[nodiscard]] auto make_blsizes(std::size_t nl, count_type fac) const {
        std::vector<count_type> res(nl, 1);
        for (std::size_t i = 1; i < nl; ++i) {
            res[i] = res[i - 1] * fac;
        }
        return res;
    }

    // Add values to the accumulator using a callable object.
    template <typename F, typename... Args>
    void add_values(F&& f, Args&&... args) {
        // always accumulate directly in the (co)variance accumulator of level 0
        std::forward<F>(f)(accs_[0], std::forward<Args>(args)...);

        // accumulate in all levels < min_levels_ - 1
        for (std::size_t i = 1; i < min_levels_ - 1; ++i) {
            std::forward<F>(f)(blocks_[i], std::forward<Args>(args)...);
            if (is_full(i)) {
                add_block(i);
            }
        }

        // accumulate in the last min. level and propagate
        std::forward<F>(f)(blocks_[min_levels_ - 1], std::forward<Args>(args)...);
        propagate(min_levels_ - 1);
    }

public:
    /**
     * @brief Construct an autocorrelation accumulator with a given number of elements.
     *
     * @details It further sets the multiplication factor for increasing block sizes and the minimum
     * number of levels to which new samples are always added directly without propagation.
     *
     * @param num Number of elements.
     * @param fac Multiplication factor.
     * @param min_levels Minimum number of levels.
     */
    explicit autocorr_acc(size_type num = 1, count_type fac = 2, std::size_t min_levels = 2) :
        accs_(min_levels, acc_type(num)),
        blocks_(min_levels, mean_acc_type(num)),
        blsizes_(make_blsizes(min_levels, fac)),
        idx_(0),
        fac_(fac),
        min_levels_(min_levels) {
        check_levels();
    }

    /**
     * @brief Construct an autocorrelation accumulator from given vectors of accumulators.
     *
     * @details It further sets the multiplication factor for increasing block sizes and the minimum
     * number of levels to which new samples are always added directly without propagation.
     *
     * @param accs Vector of (co)variance accumulators.
     * @param blocks Vector of mean accumulators (for blocking).
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
        accs_ = std::vector<acc_type>(min_levels_, acc_type { size() });
        blocks_ = std::vector<mean_acc_type>(min_levels_, mean_acc_type { size() });
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
     * @details See the corresponding method of the wrapped accumulator for more details.
     *
     * @param val Value to be accumulated.
     * @return Reference to this object.
     */
    autocorr_acc& operator<<(value_type val) {
        auto f = [](auto& acc, auto idx_, auto val) { acc[idx_] << val; };
        add_values(f, idx_, val);
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a vector.
     *
     * @details See the corresponding method of the wrapped accumulator for more details.
     *
     * @tparam W Eigen vector/array/expression type.
     * @param vec Vector/Array/Expression to be accumulated.
     * @return Reference to this object.
     */
    template <typename W>
    autocorr_acc& operator<<(const W& vec) {
        auto f = [](auto& acc, const auto& vec) { acc << vec; };
        add_values(f, vec);
        return *this;
    }

    /**
     * @brief Accumulate a range of values to consecutive elements in the accumulator.
     *
     * @details See the corresponding method of the wrapped accumulator for more details.
     *
     * @tparam R Input range of values.
     * @param rg Range of values to be accumulated.
     * @param idx Starting index for the accumulator.
     */
    template <ranges::input_range R>
    void accumulate(R&& rg, size_type idx = 0) {
        auto f = [](auto& acc, auto&& rg, size_type idx) { acc.accumulate(std::forward<R>(rg), idx); };
        add_values(f, std::forward<R>(rg), idx);
    }

    /**
     * @brief Accumulate a range of values to arbitrary elements but with different indices.
     *
     * @details For wrapped covariance accumulators, the indices should be sorted in ascending order.
     *
     * See the corresponding method of the wrapped accumulator for more details.
     *
     * @tparam R1 Input range of values.
     * @tparam R2 Input range of indices.
     * @param rg Range of values to be accumulated.
     * @param idxs Range of indices.
     */
    template <ranges::input_range R1, ranges::input_range R2>
    void accumulate(R1&& rg, R2&& idxs) {
        auto f = [](auto& acc, auto&& rg, auto&& idxs) {
            acc.accumulate(std::forward<R1>(rg), std::forward<R2>(idxs));
        };
        add_values(f, std::forward<R1>(rg), std::forward<R2>(idxs));
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
     * @brief Get effective number of samples in level \f$ i \f$.
     *
     * @param i Level index.
     * @return Number of accumulated samples in level \f$ i \f$.
     */
    [[nodiscard]] auto count(std::size_t i = 0) const {
        assert(i < accs_.size());
        return accs_[i].count();
    }

    /**
     * @brief Get the block size of level \f$ i \f$.
     *
     * @param i Level index.
     * @return Block size of level \f$ i \f$.
     */
    [[nodiscard]] auto block_size(std::size_t i = 0) const {
        assert(i < accs_.size());
        return blsizes_[i];
    }

    /**
     * @brief Get the vector of (co)variance accumulators.
     *
     * @return Vector of (co)variance accumulators.
     */
    [[nodiscard]] const auto& accumulators() const { return accs_; }

    /**
     * @brief Get the vector of block data.
     *
     * @return Vector of mean accumulators used for blocking.
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
     * @param min_samples Minimum number of effective samples.
     * @return Level index.
     */
    [[nodiscard]] std::size_t find_level(count_type min_samples) const {
        for (auto i = accs_.size(); i > 1; --i) {
            if (accs_[i - 1].count() >= min_samples) {
                return i - 1;
            }
        }
        return 0;
    }

    /**
     * @brief Calculate the sample mean from the accumulated data in level 0.
     *
     * @details Calls the `mean()` method of the wrapped accumulator in level 0.
     *
     * For statically sized accumulators with a size() == 1, it returns a single value. Otherwise, it
     * returns a vector.
     *
     * @return Sample mean.
     */
    [[nodiscard]] auto mean() const { return accs_[0].mean(); }

    /**
     * @brief Check accumulators and parameters for consistency.
     *
     * @details It throws a simplemc::simplemc_exception in case of inconsistencies.
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
