/**
 * @file
 * @brief Owns a collection of simplemc::update entries plus their selection distribution.
 */

#ifndef SIMPLEMC_MC_UPDATE_SET_HPP
#define SIMPLEMC_MC_UPDATE_SET_HPP

#include <simplemc/mc/update.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace simplemc {

/**
 * @addtogroup simplemc-mc
 * @{
 */

/**
 * @brief Owns a collection of simplemc::update entries plus their selection distribution.
 *
 * @details simplemc::update_set is the component a Monte Carlo kernel reads from to pick the next
 * update to attempt. It owns the registered simplemc::update entries (each carrying its own wrapper,
 * metadata, and counters), the `std::discrete_distribution<std::size_t>` built from per-entry
 * weights, and the registration / lookup APIs.
 *
 * The set is kernel-agnostic: it stores the detailed-balance `ratio` field on each entry but never
 * applies it. The default simplemc::metropolis_kernel applies it during `step()`; other kernels may
 * ignore it.
 *
 * @tparam S1 Serializer type used for checkpoint serialization.
 * @tparam S2 Serializer type used for input-config serialization.
 */
template <serializer S1 = json_serializer, serializer S2 = json_serializer>
class update_set {
public:
    using cp_serializer_type = S1;
    using ic_serializer_type = S2;
    using update_type = update<S1, S2>;

    /**
     * @brief Register a self-inverse update.
     *
     * @details The simplemc::update constructor already validated `name` non-empty and
     * `weight >= 0`. This method additionally validates that the name is unique within the set.
     *
     * @param u Update to register (constructed and moved in).
     */
    void add(update_type u) {
        if (find(u.name).has_value()) {
            throw simplemc_exception(fmt::format("update '{}' is already registered", u.name));
        }
        updates_.push_back(std::move(u));
    }

    /**
     * @brief Register an inverse pair of updates.
     *
     * @details Validates that the two names are distinct and not yet registered, and that the
     * weights are either both zero or both non-zero. Computes detailed-balance ratios
     * \f$ w_2 / w_1 \f$ and \f$ w_1 / w_2 \f$, cross-links `inv_name`, then pushes both entries.
     * If both weights are zero, the ratios stay at `1.0`; the entries are never picked anyway.
     *
     * @param u1 First update (constructed and moved in).
     * @param u2 Second update (constructed and moved in).
     */
    void add_pair(update_type u1, update_type u2) {
        if ((u1.weight == 0.0) != (u2.weight == 0.0)) {
            throw simplemc_exception(fmt::format(
                "inverse pair must have both weights zero or both non-zero "
                "(got {} on '{}' and {} on '{}')",
                u1.weight, u1.name, u2.weight, u2.name));
        }
        if (u1.name == u2.name) {
            throw simplemc_exception(
                fmt::format("paired updates must have distinct names (both are '{}')", u1.name));
        }
        if (find(u1.name)) {
            throw simplemc_exception(fmt::format("update '{}' is already registered", u1.name));
        }
        if (find(u2.name)) {
            throw simplemc_exception(fmt::format("update '{}' is already registered", u2.name));
        }

        if (u1.weight != 0.0) {
            u1.ratio = u2.weight / u1.weight;
            u2.ratio = u1.weight / u2.weight;
        }
        u1.inv_name = u2.name;
        u2.inv_name = u1.name;

        updates_.push_back(std::move(u1));
        updates_.push_back(std::move(u2));
    }

    /**
     * @brief Change the selection weight of a registered update.
     *
     * @details Validates `w >= 0`, looks up the update by name (throwing if not found), and
     * writes the new weight. The internal selection distribution is not rebuilt — call
     * rebuild_distribution() before the next sampling pass for the change to take effect.
     */
    void set_weight(std::string_view name, double w) {
        if (w < 0.0) {
            throw simplemc_exception(fmt::format("update weight must be >= 0 (got {} on '{}')", w, name));
        }
        const auto idx = find(name);
        if (!idx) {
            throw simplemc_exception(fmt::format("update '{}' is not registered", name));
        }
        updates_[*idx].weight = w;
    }

    /**
     * @brief Look up an update by name.
     */
    [[nodiscard]] std::optional<std::size_t> find(std::string_view name) const noexcept {
        const auto it =
            std::ranges::find_if(updates_, [name](const update_type& u) { return u.name == name; });
        if (it == updates_.end()) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(std::distance(updates_.begin(), it));
    }

    [[nodiscard]] std::size_t size() const noexcept { return updates_.size(); }
    [[nodiscard]] bool empty() const noexcept { return updates_.empty(); }

    /**
     * @brief Mutable access to the i-th update entry. Used by kernels to write counters.
     */
    [[nodiscard]] update_type& at(std::size_t i) noexcept { return updates_[i]; }

    /**
     * @brief Const access to the i-th update entry.
     */
    [[nodiscard]] const update_type& at(std::size_t i) const noexcept { return updates_[i]; }

    /**
     * @brief Read-only view over all registered update entries.
     */
    [[nodiscard]] std::span<const update_type> data() const noexcept { return updates_; }

    /**
     * @brief Recover a typed pointer to a registered update's wrapped user value.
     *
     * @tparam T Concrete user type.
     * @param name Update name.
     * @return Pointer to the wrapped user value, or `nullptr` if the name is unknown or the type
     *         does not match.
     */
    template <class T>
    [[nodiscard]] T* get(std::string_view name) noexcept {
        const auto idx = find(name);
        return idx ? updates_[*idx].template get<T>() : nullptr;
    }

    template <class T>
    [[nodiscard]] const T* get(std::string_view name) const noexcept {
        const auto idx = find(name);
        return idx ? updates_[*idx].template get<T>() : nullptr;
    }

    /**
     * @brief Build the internal selection distribution from the current per-entry weights.
     *
     * @details Throws simplemc::simplemc_exception if no entry has a positive weight; zero-weight
     * entries are passed through to `std::discrete_distribution` unchanged (they get probability 0).
     */
    void rebuild_distribution() {
        auto weights = updates_ | std::views::transform(&update_type::weight);
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
        for (auto& u : updates_) {
            u.reset_run_counters();
        }
    }

    /**
     * @brief Fold current-run counters into cumulative on every entry, then reset the current ones.
     */
    void accumulate_counters() noexcept {
        for (auto& u : updates_) {
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
    friend void simplemc_save(cp_serializer_type& s, const update_set& us) {
        for (const auto& u : us.updates_) {
            auto entry = s[u.name];
            simplemc_save(entry, u);
        }
    }

    /**
     * @brief Restore registered updates from the serializer.
     *
     * @details The destination set must already have the same set of updates registered (matching
     * names) before this call. An entry present in the set but missing in the serialized data
     * throws simplemc::simplemc_exception. The internal selection distribution is not rebuilt.
     */
    friend void simplemc_load(const cp_serializer_type& s, update_set& us) {
        for (auto& u : us.updates_) {
            if (!s.has(u.name)) {
                throw simplemc_exception(fmt::format("update '{}' not found in serialized data", u.name));
            }
            const auto entry = s[u.name];
            simplemc_load(entry, u);
        }
    }

    friend void simplemc_save_input_config(ic_serializer_type& s, const update_set& us) {
        for (const auto& u : us.updates_) {
            auto entry = s[u.name];
            simplemc_save_input_config(entry, u);
        }
    }

    friend void simplemc_load_input_config(const ic_serializer_type& s, update_set& us) {
        for (auto& u : us.updates_) {
            if (!s.has(u.name)) {
                throw simplemc_exception(fmt::format("update '{}' not found in input config", u.name));
            }
            const auto entry = s[u.name];
            simplemc_load_input_config(entry, u);
        }
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
        for (auto& u : us.updates_) {
            simplemc_mpi_collect(comm, u);
        }
    }

private:
    std::vector<update_type> updates_;
    std::discrete_distribution<std::size_t> dist_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_UPDATE_SET_HPP
