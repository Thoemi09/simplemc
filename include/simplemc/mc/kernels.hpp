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
 * It holds a non-owning pointer to a simplemc::update_set which it uses to select and execute updates
 * to advance the Markov chain. Since simplemc::run is not aware of the update set, it is the kernel's
 * responsibility to make sure that the selection distribution is valid (see prepare()) and to also
 * keep track of update statistics (number of proposed, accepted and impossible updates).
 *
 * The kernel is templated on the concrete update-set type; a deduction guide lets it be constructed as
 * `metropolis_kernel kernel { updates }` without spelling the type.
 *
 * @note The stored update set is required to outlive the kernel.
 *
 * @tparam UpdateSet Concrete simplemc::update_set instantiation.
 */
template <typename UpdateSet>
class metropolis_kernel {
public:
    /**
     * @brief Construct a Metropolis kernel with a pointer to the given update set.
     *
     * @param ups Set of updates.
     */
    explicit metropolis_kernel(UpdateSet& ups) noexcept : ups_ { &ups } {}

    /**
     * @brief Build the update selection distribution from their weights.
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
     * - Select an update from the set with probability proportional to its update::weight.
     * - Propose a new MC configuration by calling `attempt()` of the selected update and by increasing
     * the counter for *proposed* updates, update::nprops.
     * - Calculate the acceptance probability \f$ p \f$ by multiplying the return value of `attempt()`
     * with the detailed-balance update::ratio.
     * - Acceptance/Rejection step:
     *   - if \f$ p < 0 \f$: The proposed MC configuration is not \f$ \in \f$ the allowed state
     *   space. The counter for *impossible* updates, update::nimps, is increased and `reject()` is
     *   called.
     *   - \f$ p >= u \f$, where \f$ u \f$ is a uniform random number on \f$ [0, 1) \f$: The proposed
     *   MC configuration is accepted. The counter for *accepted* updates, update::naccs, is increased
     *   and `accept()` is called.
     *   - Otherwise, the proposed MC configuration is rejected. `reject()` is called.
     *
     * @tparam RNG Random number generator type.
     * @param rng Random number generator driving the update selection and acceptance/rejection.
     */
    template <typename RNG>
    void step(RNG& rng) {
        const std::size_t idx = ups_->select(rng);
        ups_->visit_at(idx, [&](auto& up) {
            ++up.nprops;
            const double p = up.attempt() * up.ratio;
            if (p < 0.0) {
                // impossible --> rejected
                ++up.nimps;
                up.reject();
            } else if (p >= uni01_(rng)) {
                // accepted
                up.accept();
                ++up.naccs;
            } else {
                // rejected
                up.reject();
            }
        });
    }

private:
    UpdateSet* ups_;
    std::uniform_real_distribution<double> uni01_ { 0.0, 1.0 };
};

/// Deduction guide: deduce the update-set type from the constructor argument.
template <typename UpdateSet>
metropolis_kernel(UpdateSet&) -> metropolis_kernel<UpdateSet>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_KERNELS_HPP
