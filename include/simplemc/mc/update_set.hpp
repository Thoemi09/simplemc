/**
 * @file
 * @brief Owns a collection of simplemc::update entries plus their selection distribution.
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
 * @brief Owns a collection of simplemc::update entries plus their selection distribution.
 *
 * @details simplemc::update_set is the component a Monte Carlo kernel reads from to pick the next
 * update to attempt. It owns the registered simplemc::update entries (each carrying its own wrapper,
 * metadata, and counters), the `std::discrete_distribution<std::size_t>` built from per-entry
 * weights, and the registration / lookup APIs (the latter shared with simplemc::measurement_set
 * through simplemc::named_set).
 *
 * The detailed-balance `ratio` on each entry is **derived state**: rebuild_distribution()
 * recomputes it from the current weights (via the `inv_name` cross-links) before building the
 * selection distribution, so weight changes through set_weight() or input-config loading can never
 * leave ratios stale.
 *
 * The set is kernel-agnostic: it stores the `ratio` field on each entry but never applies it. The
 * default simplemc::metropolis_kernel applies it during `step()`; other kernels may ignore it.
 *
 */
class update_set : public named_set<update> {
    using base_type = named_set<update>;

public:
    using serializer_type = mc_serializer;
    using update_type = update;

    /**
     * @brief Register a self-inverse update.
     *
     * @details The simplemc::update constructor already validated `name` non-empty and
     * `weight >= 0`. This method additionally validates that the name is unique within the set.
     *
     * @param u Update to register (constructed and moved in).
     */
    void add(update_type u) { this->add_checked(std::move(u), "update"); }

    /**
     * @brief Register an inverse pair of updates.
     *
     * @details Validates that the two names are distinct and not yet registered, and that the
     * weights are either both zero or both non-zero. Cross-links `inv_name`, then pushes both
     * entries. The detailed-balance ratios are not computed here — rebuild_distribution() derives
     * them from the current weights before every run.
     *
     * @param u1 First update (constructed and moved in).
     * @param u2 Second update (constructed and moved in).
     */
    void add_pair(update_type u1, update_type u2) {
        if ((u1.weight == 0.0) != (u2.weight == 0.0)) {
            throw simplemc_exception(fmt::format("inverse pair must have both weights zero or both non-zero "
                                                 "(got {} on '{}' and {} on '{}')",
                u1.weight, u1.name, u2.weight, u2.name));
        }
        if (u1.name == u2.name) {
            throw simplemc_exception(fmt::format("paired updates must have distinct names (both are '{}')", u1.name));
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
     * @details Validates `w >= 0`, looks up the update by name (throwing if not found), and
     * writes the new weight. Neither the internal selection distribution nor the detailed-balance
     * ratios are recomputed here — call rebuild_distribution() (done automatically at run entry)
     * before the next sampling pass for the change to take effect.
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
     * @brief Derive the detailed-balance ratios and build the selection distribution from the
     * current per-entry weights.
     *
     * @details First recomputes every entry's `ratio`: `1.0` for self-inverse entries,
     * \f$ w_{\text{inv}} / w \f$ for paired ones (looked up via `inv_name`; `1.0` when both weights
     * of the pair are zero). Throws simplemc::simplemc_exception if an `inv_name` is not registered
     * or if exactly one weight of a pair is zero (a state set_weight() can produce after
     * registration). Then builds the distribution; throws if no entry has a positive weight.
     * Zero-weight entries are passed through to `std::discrete_distribution` unchanged (they get
     * probability 0).
     */
    void rebuild_distribution() {
        for (auto& u : this->entries_) {
            if (u.inv_name == u.name) {
                u.ratio = 1.0;
                continue;
            }
            const auto idx = this->find(u.inv_name);
            if (!idx) {
                throw simplemc_exception(
                    fmt::format("inverse update '{}' of '{}' is not registered", u.inv_name, u.name));
            }
            const auto& inv = this->entries_[*idx];
            if ((u.weight == 0.0) != (inv.weight == 0.0)) {
                throw simplemc_exception(fmt::format("inverse pair must have both weights zero or both non-zero "
                                                     "(got {} on '{}' and {} on '{}')",
                    u.weight, u.name, inv.weight, inv.name));
            }
            u.ratio = u.weight != 0.0 ? inv.weight / u.weight : 1.0;
        }

        auto weights = this->entries_ | std::views::transform(&update_type::weight);
        if (!std::ranges::any_of(weights, [](double w) { return w > 0.0; })) {
            throw simplemc_exception("at least one update weight must be > 0");
        }
        dist_ = std::discrete_distribution<std::size_t> { weights.begin(), weights.end() };
    }

    /**
     * @brief Sample one update index according to the cached selection distribution.
     */
    template <class RNG>
    [[nodiscard]] std::size_t select(RNG& rng) {
        return dist_(rng);
    }

    /**
     * @brief Zero the current-run counters on every entry. Cumulative counters are left untouched.
     */
    void reset_run_counters() noexcept {
        for (auto& u : this->entries_) {
            u.reset_run_counters();
        }
    }

    /**
     * @brief Fold current-run counters into cumulative on every entry, then reset the current ones.
     */
    void accumulate_counters() noexcept {
        for (auto& u : this->entries_) {
            u.accumulate_counters();
        }
    }

    /**
     * @brief Serialize all registered updates into the passed serializer, keyed by name.
     *
     * @details Writes one sub-entry per registered update, with the per-entity schema produced by
     * the simplemc::update overload of simplemc_save (stats fields plus a `"user"` sub-tree). The
     * caller chooses what key to nest this under (typically `s["updates"]`).
     */
    friend void simplemc_save(serializer_type& s, const update_set& us) { us.save_entries(s); }

    /**
     * @brief Restore registered updates from the serializer.
     *
     * @details The destination set must already have the same set of updates registered (matching
     * names) before this call. An entry present in the set but missing in the serialized data
     * throws simplemc::simplemc_exception. The internal selection distribution is not rebuilt.
     */
    friend void simplemc_load(const serializer_type& s, update_set& us) {
        us.load_entries(s, "update");
    }

    friend void simplemc_save_input_config(serializer_type& s, const update_set& us) {
        us.save_input_config_entries(s);
    }

    friend void simplemc_load_input_config(const serializer_type& s, update_set& us) {
        us.load_input_config_entries(s, "update");
    }

    /**
     * @brief All-reduce every registered update across MPI ranks (counters + wrapped payload).
     *
     * @details Iterates the registered updates and forwards to the per-`update<>` `simplemc_mpi_collect`,
     * which allreduces the six counter fields and the wrapper. Weights, names, the discrete
     * selection distribution, and detailed-balance ratios are local registration data assumed
     * identical across ranks and are not touched.
     *
     * @param comm MPI communicator over which to reduce.
     * @param us Update set to reduce in place.
     */
    friend void simplemc_mpi_collect(const mpi::communicator& comm, update_set& us) {
        us.mpi_collect_entries(comm);
    }

private:
    std::discrete_distribution<std::size_t> dist_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_UPDATE_SET_HPP
