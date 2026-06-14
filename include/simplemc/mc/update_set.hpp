/**
 * @file
 * @brief Set of MC updates.
 */

#ifndef SIMPLEMC_MC_UPDATE_SET_HPP
#define SIMPLEMC_MC_UPDATE_SET_HPP

#include <simplemc/mc/named_set.hpp>
#include <simplemc/mc/serializer.hpp>
#include <simplemc/mc/update.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <ranges>
#include <string_view>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-entities-updates
 * @{
 */

/**
 * @brief Set of MC updates.
 *
 * @details The set holds registered simplemc::update objects and a `std::discrete_distribution` built
 * from their update::weight values. The distribution is initialized by calling 
 * rebuild_distribution(), which also checks that the weights are valid and recomputes the 
 * detailed-balance update::ratio for every update.
 *
 * Monte Carlo kernels usually perform the following actions (see e.g. simplemc::metropolis_kernel):
 *
 * - call select() to sample an update index according to the cached distribution,
 * - get the corresponding update using named_set::operator[](),
 * - call basic_update::attempt() to propose a new MC configuration and get the acceptance ratio, and
 * - call basic_update::accept() or basic_update::reject() to commit or roll back the proposal.
 *
 * Registration, lookup and typed access are provided by the base class simplemc::named_set. The same
 * API is shared with simplemc::measurement_set.
 */
class update_set : public named_set<update> {
public:
    /**
     * @brief Serializer type used for both checkpoint and input-config serialization.
     */
    using serializer_type = mc_serializer;

    /**
     * @brief Register a self-inverse simplemc::update in the set.
     *
     * @details It calls named_set::add_checked(). A simplemc::simplemc_exception is thrown if the
     * name is already registered.
     *
     * @param u Update to register.
     */
    void add(update u) { this->add_checked(std::move(u), "update"); }

    /**
     * @brief Register an inverse pair of simplemc::update objects in the set.
     *
     * @details It cross-links update::inv_name and calls named_set::add_checked() for both updates.
     *
     * It validates that the two names are distinct and not yet registered, and that the weights are
     * either both zero or both non-zero. Otherwise, it throws a simplemc::simplemc_exception.
     *
     * The detailed-balance update::ratio is not computed here — rebuild_distribution() derives it
     * from the current weights before every run.
     *
     * @param u1 First update to register.
     * @param u2 Second update to register.
     */
    void add_pair(update u1, update u2) {
        if ((u1.weight == 0.0) != (u2.weight == 0.0)) {
            throw simplemc_exception(fmt::format("inverse pair must have both weights zero or both non-zero "
                                                 "(got {} on '{}' and {} on '{}')",
                u1.weight, u1.name, u2.weight, u2.name));
        }
        if (u1.name == u2.name) {
            throw simplemc_exception(fmt::format("paired updates must have distinct names (both are '{}')", u1.name));
        }
        if (this->find(u1.name)) {
            throw simplemc_exception(fmt::format("update '{}' is already registered", u1.name));
        }
        if (this->find(u2.name)) {
            throw simplemc_exception(fmt::format("update '{}' is already registered", u2.name));
        }

        u1.inv_name = u2.name;
        u2.inv_name = u1.name;

        this->add_checked(std::move(u1), "update");
        this->add_checked(std::move(u2), "update");
    }

    /**
     * @brief Change the selection weight of a registered update.
     *
     * @details It validates \f$ w >= 0 \f$, looks up the update by name, and writes the new weight. A
     * simplemc::simplemc_exception is thrown if the name is not registered or if the weight is
     * negative.
     *
     * @param name Name of the update whose weight to change.
     * @param w New selection weight (must be \f$ >= 0 \f$).
     */
    void set_weight(std::string_view name, double w) {
        if (w < 0.0) {
            throw simplemc_exception(fmt::format("update weight must be >= 0 (got {} on '{}')", w, name));
        }
        const auto idx = this->find(name);
        if (!idx) {
            throw simplemc_exception(fmt::format("update '{}' is not registered", name));
        }
        this->entries_[*idx].weight = w;
    }

    /**
     * @brief Build the selection distribution from the update::weight values and recompute the
     * detailed-balance update::ratio for every update.
     *
     * @details It first recomputes every update::ratio and then initializes the discrete
     * distribution. The ratio is set to \f$ 1.0 \f$ for self-inverse updates and to
     * \f$ w_{\text{inv}} / w \f$ otherwise.
     *
     * It throws a simplemc::simplemc_exception if an update::inv_name is not registered, if exactly
     * one weight of an inverse-pair is zero, or if all weights are zero.
     */
    void rebuild_distribution() {
        // recompute detailed-balance ratios and validate inverse pairs
        for (auto& u : this->entries_) {
            // self-inverse
            if (u.inv_name == u.name) {
                u.ratio = 1.0;
                continue;
            }

            // find inverse
            const auto idx = this->find(u.inv_name);
            if (!idx) {
                throw simplemc_exception(
                    fmt::format("inverse update '{}' of '{}' is not registered", u.inv_name, u.name));
            }
            const auto& inv = this->entries_[*idx];

            // validate weights and compute ratio
            if ((u.weight == 0.0) != (inv.weight == 0.0)) {
                throw simplemc_exception(fmt::format("inverse pair must have both weights zero or both non-zero "
                                                     "(got {} on '{}' and {} on '{}')",
                    u.weight, u.name, inv.weight, inv.name));
            }
            u.ratio = u.weight != 0.0 ? inv.weight / u.weight : 1.0;
        }

        // view on the update weights (at least one must be > 0)
        auto weights = this->entries_ | std::views::transform(&update::weight);
        if (!std::ranges::any_of(weights, [](double w) { return w > 0.0; })) {
            throw simplemc_exception("at least one update weight must be > 0");
        }

        // rebuild the distribution
        dist_ = std::discrete_distribution<std::size_t> { weights.begin(), weights.end() };
    }

    /**
     * @brief Sample one update according to the cached selection distribution.
     *
     * @tparam RNG Random number generator type.
     * @param rng Random number generator.
     * @return Index of the selected update.
     */
    template <typename RNG>
    [[nodiscard]] std::size_t select(RNG& rng) {
        return dist_(rng);
    }

    /**
     * @brief Zero the current-run counters by calling update::reset_run_counters() for all registered
     * updates.
     */
    void reset_run_counters() noexcept {
        for (auto& u : this->entries_) {
            u.reset_run_counters();
        }
    }

    /**
     * @brief Accumulate the current-run counters by calling update::accumulate_counters() for all
     * registered updates.
     */
    void accumulate_counters() noexcept {
        for (auto& u : this->entries_) {
            u.accumulate_counters();
        }
    }

    /**
     * @brief Serialize a simplemc::update_set.
     *
     * @details It serializes all registered updates by calling named_set::save_entries().
     *
     * @param s Serializer handle.
     * @param us Update set to serialize.
     */
    friend void simplemc_save(serializer_type& s, const update_set& us) { us.save_entries(s); }

    /**
     * @brief Deserialize a simplemc::update_set.
     *
     * @details It deserializes all registered updates by calling named_set::load_entries().
     *
     * @param s Serializer handle.
     * @param us Update set to deserialize into.
     */
    friend void simplemc_load(const serializer_type& s, update_set& us) { us.load_entries(s, "update"); }

    /**
     * @brief Serialize the user-input config of a simplemc::update_set.
     *
     * @details It serializes all registered updates by calling
     * named_set::save_input_config_entries().
     *
     * @param s Serializer handle.
     * @param us Update set to serialize.
     */
    friend void simplemc_save_input_config(serializer_type& s, const update_set& us) {
        us.save_input_config_entries(s);
    }

    /**
     * @brief Deserialize the user-input config of a simplemc::update_set.
     *
     * @details It deserializes all registered updates by calling
     * named_set::load_input_config_entries().
     *
     * @param s Serializer handle.
     * @param us Update set to deserialize into.
     */
    friend void simplemc_load_input_config(const serializer_type& s, update_set& us) {
        us.load_input_config_entries(s, "update");
    }

    /**
     * @brief Collect simplemc::update_set objects from different MPI processes.
     *
     * @details It simply calls named_set::mpi_collect_entries(). Weights, names, the discrete
     * selection distribution, and detailed-balance ratios are local registration data assumed
     * identical across ranks and are not touched.
     *
     * @param comm simplemc::mpi::communicator object.
     * @param us Update set to reduce in place.
     */
    friend void simplemc_mpi_collect(const mpi::communicator& comm, update_set& us) { us.mpi_collect_entries(comm); }

private:
    std::discrete_distribution<std::size_t> dist_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_UPDATE_SET_HPP
