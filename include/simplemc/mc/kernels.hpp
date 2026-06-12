/**
 * @file
 * @brief Concrete Monte Carlo kernels that satisfy simplemc::mc_kernel.
 */

#ifndef SIMPLEMC_MC_KERNELS_HPP
#define SIMPLEMC_MC_KERNELS_HPP

#include <simplemc/mc/serializer.hpp>
#include <simplemc/mc/update_set.hpp>

#include <cstddef>
#include <random>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-kernels
 * @{
 */

/**
 * @brief Default Metropolis step kernel.
 *
 * @details Holds a non-owning pointer to a simplemc::update_set and a private
 * `std::uniform_real_distribution<double>` for the acceptance draw. The `step(rng)` method samples
 * an update via the set's discrete distribution, calls `attempt()` on the wrapper, multiplies the
 * result by the entry's detailed-balance `ratio`, and applies the standard Metropolis branch:
 *
 * - if `attempt() * ratio <= 0`, the proposal is counted as impossible (`nimps`) and `reject()` is
 *   called so any speculative state set up by `attempt()` is rolled back;
 * - if `attempt() * ratio >= uniform_draw`, the proposal is accepted (`naccs`) and `accept()` is
 *   called on the wrapped update;
 * - otherwise it is rejected and `reject()` is called.
 *
 * Every `attempt()` is therefore followed by exactly one of `accept()` / `reject()`. In every case
 * the proposal counter (`nprops`) is bumped. The counters live on the simplemc::update entries
 * themselves.
 *
 * Lifetime: the kernel stores a non-owning pointer to the set; the caller must keep the set alive
 * for the kernel's lifetime.
 */
class metropolis_kernel {
public:
    using update_set_type = update_set;

    /**
     * @brief Construct with a back-pointer to the update set this kernel will drive.
     */
    explicit metropolis_kernel(update_set_type& set) noexcept : set_ { &set } {}

    /**
     * @brief Build the selection distribution from current weights.
     *
     * @details Detected by simplemc::run via `if constexpr (requires { kernel.prepare(); })` and
     * invoked once at the start of each run. Equivalent to calling
     * simplemc::update_set::rebuild_distribution on the owning set.
     */
    void prepare() { set_->rebuild_distribution(); }

    /**
     * @brief Run a single Metropolis step.
     *
     * @tparam RNG Random number generator type.
     * @param rng Random number generator driving the selection and acceptance draws.
     */
    template <class RNG>
    void step(RNG& rng) {
        const std::size_t idx = set_->select(rng);
        auto& u = set_->at(idx);
        ++u.nprops;
        const double p = u.wrapped.attempt() * u.ratio;
        if (p <= 0.0) {
            ++u.nimps;
            u.wrapped.reject();
        } else if (p >= uni01_(rng)) {
            u.wrapped.accept();
            ++u.naccs;
        } else {
            u.wrapped.reject();
        }
    }

    /**
     * @brief Access the owning update set.
     */
    [[nodiscard]] update_set_type& updates() noexcept { return *set_; }

    /**
     * @brief Const access to the owning update set.
     */
    [[nodiscard]] const update_set_type& updates() const noexcept { return *set_; }

private:
    update_set_type* set_;
    std::uniform_real_distribution<double> uni01_ { 0.0, 1.0 };
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_KERNELS_HPP
