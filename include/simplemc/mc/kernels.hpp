/**
 * @file
 * @brief Concrete Monte Carlo kernels that satisfy simplemc::mc_kernel.
 */

#ifndef SIMPLEMC_MC_KERNELS_HPP
#define SIMPLEMC_MC_KERNELS_HPP

#include <simplemc/mc/update_set.hpp>

#include <cstddef>
#include <random>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-kernels
 * @{
 */

/**
 * @brief Metropolis kernel.
 *
 * @details Implementation of the standard Metropolis algorithm (see step()).
 *
 * It holds a non-owning pointer to a simplemc::update_set, which it uses to select and execute updates
 * to advance the Markov chain.
 *
 * Since simplemc::run is not aware of the update set, it is the kernel's responsibility to make sure
 * that the selection distribution is valid (see prepare()).
 *
 * @note The stored update set is required to outlive the kernel.
 *
 * @tparam Us User update types (each satisfying simplemc::mc_update) of the driven update set.
 */
template <mc_update... Us>
class metropolis_kernel {
public:
    /**
     * @brief Construct a Metropolis kernel with a pointer to the given update set.
     *
     * @param ups Set of updates.
     */
    explicit metropolis_kernel(update_set<Us...>& ups) noexcept : ups_ { &ups } {}

    /**
     * @brief Build the update selection distribution from the updates' weights.
     *
     * @details It calls simplemc::update_set::rebuild_distribution on the stored update set.
     *
     * The function is executed automatically by simplemc::run before the main MC loop starts.
     */
    void prepare() { ups_->rebuild_distribution(); }

    /**
     * @brief Run a single Metropolis step.
     *
     * @details Standard implementation of a Metropolis algorithm:
     *
     * - Select an update from the set with probability proportional to its update_stats::weight.
     * - Propose a new MC configuration by calling update::attempt() of the selected update.
     * - Calculate the acceptance probability \f$ p \f$ by multiplying the return value of `attempt()`
     * with the detailed-balance update_stats::ratio.
     * - Acceptance/Rejection step:
     *   - if \f$ p < 0 \f$: The proposed MC configuration is not in the allowed state space.
     *   update::mark_impossible() and update::reject() are called.
     *   - if \f$ p >= u \f$, where \f$ u \f$ is a uniform random number on \f$ [0, 1) \f$: The
     *   proposed MC configuration is accepted. update::accept() is called.
     *   - Otherwise, the proposed MC configuration is rejected. update::reject() is called.
     *
     * @tparam RNG Random number generator type.
     * @param rng Random number generator driving the update selection and acceptance/rejection.
     */
    template <typename RNG>
    void step(RNG& rng) {
        const std::size_t idx = ups_->select(rng);
        ups_->visit_at(idx, [&](auto& up) {
            const double p = up.attempt() * up.stats().ratio;
            if (p < 0.0) {
                // impossible --> rejected
                up.mark_impossible();
                up.reject();
            } else if (p >= uni01_(rng)) {
                // accepted
                up.accept();
            } else {
                // rejected
                up.reject();
            }
        });
    }

private:
    update_set<Us...>* ups_;
    std::uniform_real_distribution<double> uni01_ { 0.0, 1.0 };
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_KERNELS_HPP
