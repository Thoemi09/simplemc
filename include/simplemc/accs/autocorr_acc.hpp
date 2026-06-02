/**
 * @file
 * @brief Wrapper accumulator to estimate the integrated autocorrelation time.
 */

#ifndef SIMPLEMC_ACCS_AUTOCORR_ACC_HPP
#define SIMPLEMC_ACCS_AUTOCORR_ACC_HPP

#include <simplemc/accs/concepts.hpp>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/accs/utils.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <utility>
#include <vector>

namespace simplemc {

/**
 * @addtogroup simplemc-accs-wrappers
 * @{
 */

/**
 * @brief Wrapper for simplemc::var_acc and simplemc::covar_acc to estimate the integrated
 * autocorrelation time.
 *
 * @details Following the derivation in @ref simplemc-accs-stats-tau, we can write the integrated
 * autocorrelation time as
 * \f[
 *   \tau_{\mathbf{X}\mathbf{Y}} = \frac{1}{2} \left( \frac{\mathrm{Cov}[\overline{\mathbf{X}}^{(N)},
 *   \overline{\mathbf{Y}}^{(N)}]}{\mathrm{Cov}[\mathbf{X}, \mathbf{Y}]} N - 1 \right) \approx
 *   \frac{1}{2} \left( \frac{s_{\overline{\mathbf{X}}^{(N)}\overline{\mathbf{Y}}^{(N)}}^2}
 *   {s_{\mathbf{X}\mathbf{Y}}^2} N - 1 \right) \; .
 * \f]
 * If we use a blocking method to determine the block size \f$ B \f$ at which the block averages,
 * \f$ \overline{\mathbf{X}}^{(B)} \f$ and \f$ \overline{\mathbf{Y}}^{(B)} \f$, become uncorrelated,
 * then this equation can be further approximated with
 * \f[
 *   \tau_{\mathbf{X}\mathbf{Y}} \approx \frac{1}{2} \left( \frac{s_{\overline{\mathbf{X}}^{(B)}
 *   \overline{\mathbf{Y}}^{(B)}}^2}{s_{\mathbf{X}\mathbf{Y}}^2} B - 1 \right) \; ,
 * \f]
 * where \f$ s_{\mathbf{X}\mathbf{Y}}^2 \f$ is the naive (unblocked) sample (cross)-covariance matrix
 * and \f$ s_{\overline{\mathbf{X}}^{(B)}\overline{\mathbf{Y}}^{(B)}}^2 =
 * s_{\overline{\mathbf{X}}^{(N)}\overline{\mathbf{Y}}^{(N)}}^2 N_{\mathrm{eff}} \f$ is the sample
 * (cross)-covariance matrix of the blocked samples with block size \f$ B = N / N_{\mathrm{eff}} \f$.
 *
 * A similar equation holds for \f$ \tau_{\mathbf{X}} \f$.
 *
 * This class uses blocks of increasing size to decorrelate individual samples. The (co)variance and
 * the integrated autocorrelation time will increase with the block size until it reaches a plateau.
 * The value at the plateau should give you a good estimate of both quantities.
 *
 * The block sizes are given by \f$ B_{l+1} = B_l * c \f$, where \f$ l \f$ is the level index and \f$
 * c > 1 \f$ is the multiplication factor for increasing block sizes. The first level has a block size
 * of 1, i.e. \f$ B_0 = 1 \f$.
 *
 * New samples are always added to the block data of levels with \f$ l < L_{\text{min}} \f$, where
 * \f$ L_{\text{min}} \f$ is the minimum number of levels in the autocorrelation accumulator. Once a
 * block in level \f$ l \f$ is full, the block average is accumulated in a simplemc::var_acc or
 * simplemc::covar_acc and for \f$ l \geq L_{\text{min}} - 1 \f$, the block data is propagated to
 * level \f$ l+1 \f$. If level \f$ l+1 \f$ does not exist yet, it is created with a block size \f$
 * B_{l+1} = B_l * c \f$.
 *
 * Functionality and usage is similar to the wrapped accumulator except that multi-value accumulation
 * is not supported right now (please use accumulate(R1&&, R2&&) instead).
 *
 * @note The accumulator only groups the data into levels with increasing block sizes. It does not
 * give a final estimate of the integrated autocorrelation time. It is the users responsibility to
 * inspect the blocked data and decide what to do with it.
 *
 * Here is a usual workflow, that
 * - accumulates samples from an AR(1) process into a simplemc::autocorr_acc,
 * - finds the highest level \f$ l' \f$ with at least 256 effective samples using find_level() and
 * - prints \f$ s_{\overline{X}}^2 \f$ and the integrated autocorrelation time \f$ \tau_X \f$ for
 * each level \f$ l \leq l' \f$:
 *
 * @include accs/doc_autocorr_acc.cpp
 *
 * Output:
 *
 * ```
 * Count          Block size     Variance       tau
 * 1000000        1              0.00000527     0.00000000
 * 500000         2              0.00001000     0.44994963
 * 250000         4              0.00001856     1.26242973
 * 125000         8              0.00003261     2.59613923
 * 62500          16             0.00005184     4.42247644
 * 31250          32             0.00007154     6.29331284
 * 15625          64             0.00008576     7.64325130
 * 7812           128            0.00009447     8.47016291
 * 3906           256            0.00009829     8.83266946
 * 1953           512            0.00010079     9.07009176
 * 976            1024           0.00010675     9.63041320
 * 488            2048           0.00010742     9.69411582
 * ```
 *
 * @tparam A Accumulator type.
 */
template <typename A>
class autocorr_acc {
public:
    /**
     * @brief Type of the wrapped accumulator.
     */
    using acc_type = A;

    /**
     * @brief Type of accumulated samples (simplemc::sample_type).
     */
    using sample_type = typename acc_type::sample_type;

    /**
     * @brief Underlying scalar type of accumulated samples (simplemc::double_or_complex).
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
     * @brief Static size of the accumulator.
     */
    static constexpr int static_size = acc_type::static_size;

    /**
     * @brief Is the accumulator dynamically sized?
     */
    static constexpr bool is_dynamic = acc_type::is_dynamic;

    /**
     * @brief Get the algorithm used to accumulate the data.
     *
     * @return simplemc::varalg tag of the wrapped accumulator.
     */
    static constexpr auto varalg() noexcept { return A::varalg(); }

    /**
     * @brief Mean accumulator type for accumulating block data.
     */
    using mean_acc_type = mean_acc<typename acc_type::sample_type, varalg()>;

private:
    // Check if the given level is full?
    [[nodiscard]] auto is_full(std::size_t lvl) const {
        assert(lvl < accs_.size());
        return blocks_[lvl].count() >= blsizes_[lvl];
    }

    // Add the block in the given level to its effective samples and reset its block data.
    void add_block(std::size_t lvl) {
        assert(lvl < accs_.size());
        accs_[lvl] << blocks_[lvl].mean();
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
     * @brief Construct an autocorrelation accumulator with the given number of elements \f$ M \f$,
     * multiplication factor \f$ c \f$ and minimum number of levels \f$ L_{\text{min}} \f$.
     *
     * @details It calls check_levels() to check the given input parameters.
     *
     * @param m Number of elements \f$ M \f$.
     * @param c Multiplication factor \f$ c \f$.
     * @param min_levels Minimum number of levels \f$ L_{\text{min}} \f$.
     */
    explicit autocorr_acc(size_type m = 1, count_type c = 2, std::size_t min_levels = 2) :
        accs_(min_levels, acc_type(m)),
        blocks_(min_levels, mean_acc_type(m)),
        blsizes_(make_blsizes(min_levels, c)),
        idx_(0),
        fac_(c),
        min_levels_(min_levels) {
        check_levels();
    }

    /**
     * @brief Construct an autocorrelation accumulator with the given vectors of accumulators,
     * multiplication factor \f$ c \f$ and minimum number of levels \f$ L_{\text{min}} \f$.
     *
     * @details It calls check_levels() to check the given accumulators and input parameters.
     *
     * @param accs Vector of (co)variance accumulators for accumulating effective (blocked) samples.
     * @param blocks Vector of mean accumulators for blocking individual samples.
     * @param c Multiplication factor \f$ c \f$.
     * @param min_levels Minimum number of levels.
     */
    explicit autocorr_acc(
        std::vector<acc_type> accs, std::vector<mean_acc_type> blocks, count_type c, std::size_t min_levels) :
        accs_(std::move(accs)),
        blocks_(std::move(blocks)),
        blsizes_(make_blsizes(accs_.size(), c)),
        idx_(0),
        fac_(c),
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
    autocorr_acc& operator[](size_type i) {
        idx_ = i;
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a single (scalar) value \f$ x \f$.
     *
     * @details The value is first added to all levels with \f$ l < L_{\text{min}} \f$ using
     * simplemc::mean_acc::operator<<(const U&). If the block in level \f$ L_{\text{min}} - 1 \f$ is
     * full, it is recursively propagated to higher levels.
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @tparam U Type of the value to be accumulated.
     * @param x Value \f$ x \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename U>
        requires std::convertible_to<U, value_type>
    autocorr_acc& operator<<(const U& x) {
        auto f = [](auto& acc, auto idx_, auto val) { acc[idx_] << val; };
        add_values(f, idx_, x);
        return *this;
    }

    /**
     * @brief Stream operator for accumulating a vector \f$ \mathbf{v} \f$.
     *
     * @details The vector is first added to all levels with \f$ l < L_{\text{min}} \f$ using
     * simplemc::mean_acc::operator<<(const W&). If the block in level \f$ L_{\text{min}} - 1 \f$ is
     * full, it is recursively propagated to higher levels.
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @tparam W Eigen vector/array/expression type.
     * @param v Vector/Array/Expression \f$ \mathbf{v} \f$ to be accumulated.
     * @return Reference to `this` object.
     */
    template <typename W>
    autocorr_acc& operator<<(const W& v) {
        assert(v.size() == size());
        auto f = [](auto& acc, const auto& vec) { acc << vec; };
        add_values(f, v);
        return *this;
    }

    /**
     * @brief Accumulate a range of values to consecutive elements in the accumulator.
     *
     * @details The values are first added to all levels with \f$ l < L_{\text{min}} \f$ using
     * simplemc::mean_acc::accumulate(). If the block in level \f$ L_{\text{min}} - 1 \f$ is full, it
     * is recursively propagated to higher levels.
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @tparam R Input range of values.
     * @param rg Range of values to be accumulated.
     * @param i Index \f$ i \f$ of the first element in the accumulator that a value will be added to.
     */
    template <ranges::input_range R>
    void accumulate(R&& rg, size_type i = 0) {
        auto f = [](auto& acc, auto&& rg, size_type idx) { acc.accumulate(std::forward<R>(rg), idx); };
        add_values(f, std::forward<R>(rg), i);
    }

    /**
     * @brief Accumulate a range of values to arbitrary elements with the given indices.
     *
     * @details The values are first added to all levels with \f$ l < L_{\text{min}} \f$ using
     * simplemc::mean_acc::accumulate(R1 &&, R2 &&). If the block in level \f$ L_{\text{min}} - 1 \f$
     * is full, it is recursively propagated to higher levels.
     *
     * See also @ref simplemc-accs-accs-how.
     *
     * @tparam R1 Input range of values.
     * @tparam R2 Input range of indices.
     * @param rg Range of values to be accumulated.
     * @param idxs Range of indices at which the values should be accumulated.
     */
    template <ranges::input_range R1, ranges::input_range R2>
    void accumulate(R1&& rg, R2&& idxs) {
        auto f = [](auto& acc, auto&& rg, auto&& idxs) {
            acc.accumulate(std::forward<R1>(rg), std::forward<R2>(idxs));
        };
        add_values(f, std::forward<R1>(rg), std::forward<R2>(idxs));
    }

    /**
     * @brief Get the size \f$ M \f$ of the accumulator.
     *
     * @return Number of elements.
     */
    [[nodiscard]] auto size() const noexcept { return accs_[0].size(); }

    /**
     * @brief Get the multiplication factor \f$ c \f$ for increasing block sizes.
     *
     * @return Multiplication factor.
     */
    [[nodiscard]] auto factor() const noexcept { return fac_; }

    /**
     * @brief Get the minimum number of levels \f$ L_{\text{min}} \f$.
     *
     * @return Minimum number of levels.
     */
    [[nodiscard]] auto min_levels() const noexcept { return min_levels_; }

    /**
     * @brief Get the number \f$ L \f$ of levels.
     *
     * @return Current number of levels.
     */
    [[nodiscard]] auto num_levels() const noexcept { return accs_.size(); }

    /**
     * @brief Get the effective number \f$ N_l \f$ of samples in level \f$ l \f$.
     *
     * @param l Level index.
     * @return Number of accumulated samples in level \f$ l \f$.
     */
    [[nodiscard]] auto count(std::size_t l = 0) const noexcept {
        assert(l < accs_.size());
        return accs_[l].count();
    }

    /**
     * @brief Check if the accumulator is empty.
     *
     * @return True if the count() is zero, i.e. \f$ N_0 = 0 \f$, false otherwise.
     */
    [[nodiscard]] bool empty() const noexcept { return count() == 0; }

    /**
     * @brief Get the block size \f$ B_l \f$ of level \f$ l \f$.
     *
     * @param l Level index.
     * @return Block size of level \f$ l \f$.
     */
    [[nodiscard]] auto block_size(std::size_t l = 0) const noexcept {
        assert(l < accs_.size());
        return blsizes_[l];
    }

    /**
     * @brief Get the (co)variance accumulators used to accumulate the effective (blocked)
     * samples.
     *
     * @return `std::vector` of (co)variance accumulators.
     */
    [[nodiscard]] const auto& accumulators() const noexcept { return accs_; }

    /**
     * @brief Get the mean accumulators used to accumulate individual samples and to block
     * them into effective samples.
     *
     * @return `std::vector` of mean accumulators.
     */
    [[nodiscard]] const auto& blocks() const noexcept { return blocks_; }

    /**
     * @brief Get the block sizes of all levels.
     *
     * @return `std::vector` containing the block sizes \f$ B_l \f$.
     */
    [[nodiscard]] const auto& block_sizes() const noexcept { return blsizes_; }

    /**
     * @brief Find the highest level \f$ l' \f$ with at least a given number of effective samples \f$
     * N_{\text{min}} \f$.
     *
     * @param n_min Minimum number of effective samples \f$ N_{\text{min}} \f$.
     * @return Level index \f$ l' = \max_l \{ l : N_l \geq N_{\text{min}} \} \f$.
     */
    [[nodiscard]] std::size_t find_level(count_type n_min) const noexcept {
        for (auto i = accs_.size(); i > 1; --i) {
            if (accs_[i - 1].count() >= n_min) {
                return i - 1;
            }
        }
        return 0;
    }

    /**
     * @brief Calculate the sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$.
     *
     * @details It calls the wrapped accumulator's `%mean()` method in level \f$ 0 \f$.
     *
     * @return Sample mean \f$ \overline{\mathbf{z}}^{(N)} \f$.
     */
    [[nodiscard]] auto mean() const { return accs_[0].mean(); }

    /**
     * @brief Calculate the sample variance of the mean \f$ s_{\overline{\mathbf{Z}}}^2 \f$.
     *
     * @details It uses find_level() to determine the highest level with at least \f$ N_{\text{min}}
     * \f$ effective samples and it calls that level's `%variance()` method.
     *
     * @note Only available when the wrapped accumulator satisfies simplemc::variance_accumulator.
     *
     * @param n_min Minimum number of effective samples \f$ N_{\text{min}} \f$. If zero, level \f$ 0
     * \f$ is used.
     * @return Sample variance of the mean \f$ s_{\overline{\mathbf{Z}}}^2 \f$.
     */
    [[nodiscard]] auto variance(count_type n_min = 0) const
        requires variance_accumulator<acc_type>
    {
        return accs_[n_min == 0 ? 0 : find_level(n_min)].variance();
    }

    /**
     * @brief Calculate the sample covariance matrix of the mean \f$ s_{\overline{\mathbf{Z}}
     * \overline{\mathbf{Z}}}^2 \f$.
     *
     * @details It uses find_level() to determine the highest level with at least \f$ N_{\text{min}}
     * \f$ effective samples and it calls that level's `%covariance()` method.
     *
     * @note Only available when the wrapped accumulator satisfies simplemc::covariance_accumulator.
     *
     * @param n_min Minimum number of effective samples \f$ N_{\text{min}} \f$. If zero, level \f$ 0
     * \f$ is used.
     * @return Sample covariance matrix of the mean \f$ s_{\overline{\mathbf{Z}}
     * \overline{\mathbf{Z}}}^2 \f$.
     */
    [[nodiscard]] auto covariance(count_type n_min = 0) const
        requires covariance_accumulator<acc_type>
    {
        return accs_[n_min == 0 ? 0 : find_level(n_min)].covariance();
    }

    /**
     * @brief Check accumulators and parameters for consistency.
     *
     * @details It throws a simplemc::simplemc_exception if
     * - \f$ c \leq 1 \f$,
     * - \f$ L_{\text{min}} < 2 \f$,
     * - the number \f$ L \f$ of (co)variance accumulators and mean accumulators for blocking is not
     * the same or if \f$ L < L_{\text{min}} \f$,
     * - the size of an individual accumulator is \f$ \neq M \f$,
     * - in level \f$ l \f$
     *   - the number \f$ N_l \f$ of accumulated effective (blocked) samples is \f$ \neq \lfloor N_0 /
     *   c^l \rfloor \f$,
     *   - \f$ B_l \neq B_{l-1} * c \f$ with \f$ B_0 = 1 \f$.
     */
    void check_levels() const {
        if (fac_ <= 1) {
            throw simplemc_exception("Multiplication factor for block sizes <= 1");
        }
        if (min_levels_ < 2) {
            throw simplemc_exception("Minimum number of levels < 2");
        }
        if (accs_.size() != blocks_.size()) {
            throw simplemc_exception("Number of (co)variance accs != number of mean accs");
        }
        if (accs_.size() < min_levels_) {
            throw simplemc_exception("Number of levels < minimum number of levels");
        }
        count_type b_l = 1;
        auto n_l = accs_[0].count();
        for (std::size_t i = 0; i < accs_.size(); ++i) {
            if (accs_[i].size() != size()) {
                throw simplemc_exception("Size mismatch between accumulators");
            }
            if (blocks_[i].size() != size()) {
                throw simplemc_exception("Size mismatch between blocks");
            }
            if (accs_[i].count() != n_l) {
                throw simplemc_exception("Wrong sample count");
            }
            if (blsizes_[i] != b_l) {
                throw simplemc_exception("Wrong block size");
            }
            n_l /= fac_;
            b_l *= fac_;
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

/**
 * @brief Serialize a simplemc::autocorr_acc.
 *
 * @details It serializes the multiplication factor \f$ c \f$, the minimum number of levels \f$
 * L_{\text{min}} \f$ and the current number of levels \f$ L \f$ together with the (co)variance
 * accumulator and the mean accumulator of each level.
 *
 * @tparam S simplemc::serializer type.
 * @tparam A Wrapped (co)variance accumulator type of the autocorrelation accumulator.
 * @param s Serializer object.
 * @param acc Autocorrelation accumulator to serialize.
 */
template <serializer S, typename A>
void simplemc_save(S& s, const autocorr_acc<A>& acc) {
    const auto num_levels = acc.num_levels();
    s.save_at("factor", acc.factor());
    s.save_at("min_levels", acc.min_levels());
    s.save_at("num_levels", num_levels);
    for (std::size_t l = 0; l < num_levels; ++l) {
        auto sub = s[std::to_string(l)];
        sub.save_at("acc", acc.accumulators()[l]);
        sub.save_at("block", acc.blocks()[l]);
    }
}

/**
 * @brief Deserialize a simplemc::autocorr_acc.
 *
 * @details It first deserializes the multiplication factor \f$ c \f$, the minimum number of levels
 * \f$ L_{\text{min}} \f$ and the current number of levels \f$ L \f$ together with the (co)variance
 * accumulator and the mean accumulator of each level. Then it uses them to construct the
 * autocorrelation accumulator (see simplemc::autocorr_acc(std::vector<acc_type>,
 * std::vector<mean_acc_type>, count_type, std::size_t)).
 *
 * @tparam S simplemc::deserializer type.
 * @tparam A Wrapped (co)variance accumulator type of the autocorrelation accumulator.
 * @param s Deserializer object.
 * @param acc Autocorrelation accumulator to deserialize into.
 */
template <deserializer S, typename A>
void simplemc_load(const S& s, autocorr_acc<A>& acc) {
    using wrapped_acc_type = typename autocorr_acc<A>::acc_type;
    using mean_acc_type = typename autocorr_acc<A>::mean_acc_type;
    auto factor = typename autocorr_acc<A>::count_type {};
    std::size_t min_levels = 0;
    std::size_t num_levels = 0;
    s.load_at("factor", factor);
    s.load_at("min_levels", min_levels);
    s.load_at("num_levels", num_levels);
    auto accs = std::vector<wrapped_acc_type>(num_levels, wrapped_acc_type { acc.size() });
    auto blocks = std::vector<mean_acc_type>(num_levels, mean_acc_type { acc.size() });
    for (std::size_t l = 0; l < num_levels; ++l) {
        const auto sub = s[std::to_string(l)];
        sub.load_at("acc", accs[l]);
        sub.load_at("block", blocks[l]);
    }
    acc = autocorr_acc<A> { std::move(accs), std::move(blocks), factor, min_levels };
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_ACCS_AUTOCORR_ACC_HPP
