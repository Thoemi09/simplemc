/**
 * @file autocorr_acc.hpp
 * @brief Wrapper for accumulators to estimate the autocorrelation time.
 */

#ifndef SIMPLEMC_ACCS_AUTOCORR_ACC_HPP
#define SIMPLEMC_ACCS_AUTOCORR_ACC_HPP

#include <simplemc/accs/block_acc.hpp>

#include <vector>

namespace simplemc {

/**
 * @brief Wrapper class for other accumulators to estimate the autocorrelation time using the blocking method.
 *
 * @details It makes use of the simplemc::block_acc to block data into blocks of increasing size. The variance
 * and the integrated autocorrelation time will increase with the block size until it reaches a plateau. The
 * value at the plateau should give you a good estimate of both quantities.
 *
 * Functionality and usage is similar to the supported wrapped accumulators (simplemc::var_acc
 * and simplemc::covar_acc). The autocorrelation time is calculated for each element of the wrapped
 * accumulator, e.g. if you wrap a simplemc::covar_acc, you will get back a matrix.
 *
 * Multi value accumulators can be used for wrapped simplemc::var_acc. Remember to manually propagate
 * full blocks after using them, otherwise the autocorrelation time will not be correct:
 * * @code{.cpp}
 * auto mva = acc.make_mva();
 * mva[idx1] << val1;
 * mva[idx2] << val2;
 * mva.increment_count();
 * acc.propagate();
 * @endcode
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
     * @brief Block accumulator type.
     */
    using block_acc_type = block_acc<acc_type>;

public:
    /**
     * @brief Construct an autocorrelation accumulator with a given number of elements, a constant shift and a
     * multiplication factor for increasing block sizes.
     *
     * @param size Number of elements.
     * @param shift A single constant shift applied to the accumulated values.
     * @param fac Multiplication factor.
     */
    explicit autocorr_acc(size_type size = 1, value_type shift = 0.0, count_type fac = 2) :
        levels_ { block_acc_type(size, shift, 1) },
        fac_(fac) {
        check_levels();
    }

    /**
     * @brief Construct an autocorrelation accumulator with a given constant shift and multiplication factor for
     * increasing block sizes.
     *
     * @details The size of the accumulator will be the same as the size of the shift vector.
     *
     * @param shift Constant shift applied to the accumulated values.
     * @param fac Multiplication factor.
     */
    explicit autocorr_acc(const Eigen::VectorX<value_type>& shift, count_type fac = 2) :
        levels_ { block_acc_type(shift, 1) },
        fac_(fac) {
        check_levels();
    }

    /**
     * @brief Construct an autocorrelation accumulator from a given vector of simplemc::block_acc accumulators and a
     * multiplication factor for increasing block sizes.
     *
     * @param levels Vector of simplemc::block_acc accumulators.
     * @param fac Multiplication factor.
     */
    explicit autocorr_acc(std::vector<block_acc_type> levels, count_type fac) : levels_(std::move(levels)), fac_(fac) {
        check_levels();
    }

    /**
     * @brief Create a multi value accumulator for a given index.
     *
     * @param idx Index.
     * @return Multi value accumulator.
     */
    [[nodiscard]] auto make_mva(size_type idx = 0) { return levels_[0].make_mva(idx); }

    /**
     * @brief Get the size of the accumulator.
     *
     * @return Number of elements.
     */
    [[nodiscard]] auto size() const { return levels_[0].size(); }

    /**
     * @brief Get the multiplication factor for increasing block sizes.
     *
     * @return Multiplication factor.
     */
    [[nodiscard]] auto factor() const { return fac_; }

    /**
     * @brief Get effective number of samples taken in level i.
     *
     * @param i Level index.
     * @return Number of accumulated samples in level i.
     */
    [[nodiscard]] auto count(std::size_t i) const {
        assert(i < levels_.size());
        return levels_[i].count();
    }

    /**
     * @brief Get the constant shift.
     *
     * @return Constant shift vector applied to the accumulated values.
     */
    [[nodiscard]] const auto& shift() const { return levels_[0].shift(); }

    /**
     * @brief Get the vector of block accumulators.
     *
     * @return Vector of block accumulators.
     */
    [[nodiscard]] const auto& levels() const { return levels_; }

    /**
     * @brief Check levels for consistency, otherwise throw an exception.
     */
    void check_levels() const {
        if (fac_ <= 1) {
            throw simplemc_exception("Factor <= 1 in autocorrelation accumulator", "autocorr_acc::check_levels");
        }
        if (levels_.empty()) {
            throw simplemc_exception("No levels in autocorrelation accumulator", "autocorr_acc::check_levels");
        }
        if (levels_[0].block_size() != 1) {
            throw simplemc_exception("Block size != 1 in level 0", "autocorr_acc::check_levels");
        }
        for (std::size_t i = 1; i < levels_.size(); ++i) {
            if (levels_[i].size() != size()) {
                throw simplemc_exception("Size mismatch between levels", "autocorr_acc::check_levels");
            }
            if (levels_[i].block_size() != levels_[i - 1].block_size() * fac_) {
                throw simplemc_exception("Block size mismatch between levels", "autocorr_acc::check_levels");
            }
        }
    }

private:
    std::vector<block_acc_type> levels_;
    count_type fac_ { 2 };
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_AUTOCORR_ACC_HPP
