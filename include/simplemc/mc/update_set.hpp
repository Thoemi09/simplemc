// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Set of MC updates.
 */

#ifndef SIMPLEMC_MC_UPDATE_SET_HPP
#define SIMPLEMC_MC_UPDATE_SET_HPP

#include <simplemc/mc/tuple_set.hpp>
#include <simplemc/mc/update.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-entities-updates
 * @{
 */

/**
 * @brief Set of MC updates.
 *
 * @details The set inherits from simplemc::tuple_set to store simplemc::update entries in a
 * `std::tuple`, so the update *types* are fixed at construction. It also holds a
 * `std::discrete_distribution` built from the update_stats::weight values. The distribution is
 * (re)built by rebuild_distribution(), which validates the inverse update pairs and recomputes the
 * detailed-balance update_stats::ratio for every update.
 *
 * Update sets are used directly by Monte Carlo kernels (see simplemc::metropolis_kernel).
 *
 * Individual updates are registered at construction-time, e.g.
 *
 * @code{.cpp}
 * simplemc::update_set {
 *     simplemc::update { foo { }, "foo", 1.0 },
 *     simplemc::update { bar { }, "bar", 2.0 }
 * };
 * @endcode
 *
 * The weights and inverse pairing stay mutable at runtime via set_weight() and link_pair(),
 * respectively. Lookup and typed access are provided by the base class simplemc::tuple_set.
 *
 * @tparam Us User update types (each satisfying simplemc::mc_update).
 */
template <mc_update... Us>
class update_set : public tuple_set<update<Us>...> {
public:
    /**
     * @brief Construct the set from the given updates.
     *
     * @param us Updates to store, in order.
     */
    explicit update_set(update<Us>... us) : tuple_set<update<Us>...> { std::move(us)... } {}

    /**
     * @brief Link two registered updates as a detailed-balance inverse pair.
     *
     * @details It cross-links update_stats::inv_name of the two named updates. Passing the same name
     * twice declares the update self-inverse.
     *
     * The detailed-balance update_stats::ratio is not computed here — rebuild_distribution() derives
     * it from the current weights before every run.
     *
     * It throws a simplemc::simplemc_exception if either name is not registered.
     *
     * @param name1 First update name.
     * @param name2 Second update name.
     */
    void link_pair(std::string_view name1, std::string_view name2) {
        // check name1 up front (visit_at() checks name2), so a throw leaves the set untouched
        const auto ia = this->find(name1);
        if (!ia) {
            throw simplemc_exception(fmt::format("no entry named '{}' in the set", name1));
        }

        // cross-link the two updates
        this->visit_at(name2, [&](auto& u) { u.set_inv_name(std::string { name1 }); });
        this->visit_at(*ia, [&](auto& u) { u.set_inv_name(std::string { name2 }); });
    }

    /**
     * @brief Change the selection weight of a registered update.
     *
     * @details It finds the update by name and calls update::set_weight() on it.
     *
     * It throws a simplemc::simplemc_exception if the name is not registered.
     *
     * @param name Name of the update whose weight to change.
     * @param w New selection weight (must be \f$ \geq 0 \f$).
     */
    void set_weight(std::string_view name, double w) {
        this->visit_at(name, [&](auto& u) { u.set_weight(w); });
    }

    /**
     * @brief Build the selection distribution from the update_stats::weight values and recompute the
     * detailed-balance update_stats::ratio for every update.
     *
     * @details It validates the correctness of all inverse pairs, recomputes every
     * update_stats::ratio and then initializes the discrete distribution.
     *
     * It throws a simplemc::simplemc_exception:
     *
     * - if an update_stats::inv_name is not registered,
     * - if the inverse pairing is asymmetric,
     * - if exactly one weight of an inverse pair is zero, or
     * - if all weights are zero.
     */
    void rebuild_distribution() {
        // recompute detailed-balance ratios and validate the inverse pairs
        this->for_each([&](auto& u) {
            // early return for self-inverse updates
            if (u.stats().inv_name == u.name()) {
                u.set_ratio(1.0);
                return;
            }

            // find the inverse update by name
            const auto idx = this->find(u.stats().inv_name);
            if (!idx) {
                throw simplemc_exception(
                    fmt::format("inverse update '{}' of '{}' is not registered", u.stats().inv_name, u.name()));
            }

            // get the inverse's weight and inverse name
            double w_inv = 0.0;
            std::string_view inv_inv;
            this->visit_at(*idx, [&](const auto& inv) {
                w_inv = inv.stats().weight;
                inv_inv = inv.stats().inv_name;
            });

            // validate the inverse pairing and weight consistency
            if (inv_inv != u.name()) {
                throw simplemc_exception(fmt::format("inverse pairing is asymmetric: '{}' names '{}' as its "
                                                     "inverse, but '{}' names '{}'",
                    u.name(), u.stats().inv_name, u.stats().inv_name, inv_inv));
            }
            if ((u.stats().weight == 0.0) != (w_inv == 0.0)) {
                throw simplemc_exception(fmt::format("inverse pair must have both weights zero or both non-zero "
                                                     "(got {} on '{}' and {} on '{}')",
                    u.stats().weight, u.name(), w_inv, u.stats().inv_name));
            }

            // compute the detailed-balance ratio (1.0 if both weights are zero)
            u.set_ratio(u.stats().weight != 0.0 ? w_inv / u.stats().weight : 1.0);
        });

        // gather weights (at least one must be > 0) and rebuild the distribution
        std::array<double, sizeof...(Us)> weights {};
        this->for_each([&weights, k = std::size_t { 0 }](const auto& u) mutable { weights[k++] = u.stats().weight; });
        if (!std::ranges::any_of(weights, [](double w) { return w > 0.0; })) {
            throw simplemc_exception("at least one update weight must be > 0");
        }
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
     * @brief Collect the metadata of every update into a vector of simplemc::update_stats.
     *
     * @return Snapshot of the update statistics, one entry per update, in set order.
     */
    [[nodiscard]] std::vector<update_stats> stats() const {
        std::vector<update_stats> out;
        out.reserve(this->size());
        this->for_each([&](const auto& u) { out.push_back(u.stats()); });
        return out;
    }

    /**
     * @brief Zero the counters of every update via update::reset_counters().
     *
     * @details This discards all update statistics gathered so far. The typical use is dropping the 
     * statistics of a warm-up run before the measurement run starts.
     */
    void reset_counters() noexcept {
        this->for_each([](auto& u) { u.reset_counters(); });
    }

    /**
     * @brief Serialize a simplemc::update_set.
     *
     * @details It dispatches to tuple_set::save_entries, which serializes all updates in the set.
     *
     * @tparam S Serializer type.
     * @param s Serializer handle.
     * @param us Update set to serialize.
     */
    template <serializer S>
    friend void simplemc_save(S s, const update_set& us) {
        us.save_entries(s);
    }

    /**
     * @brief Deserialize a simplemc::update_set.
     *
     * @details It dispatches to tuple_set::load_entries, which deserializes all updates in the set.
     *
     * @tparam S Serializer type.
     * @param s Serializer handle.
     * @param us Update set to deserialize.
     */
    template <serializer S>
    friend void simplemc_load(const S& s, update_set& us) {
        us.load_entries(s, "update");
    }

    /**
     * @brief Serialize the user-input config of a simplemc::update_set.
     *
     * @details It dispatches to tuple_set::save_input_config_entries, which serializes the
     * input-config of all updates in the set.
     *
     * @tparam S Serializer type.
     * @param s Serializer handle.
     * @param us Update set to serialize.
     */
    template <serializer S>
    friend void simplemc_save_input_config(S s, const update_set& us) {
        us.save_input_config_entries(s);
    }

    /**
     * @brief Deserialize the user-input config of a simplemc::update_set.
     *
     * @details It dispatches to tuple_set::load_input_config_entries, which deserializes the
     * input-config of all updates in the set.
     *
     * @tparam S Serializer type.
     * @param s Serializer handle.
     * @param us Update set to deserialize.
     */
    template <serializer S>
    friend void simplemc_load_input_config(const S& s, update_set& us) {
        us.load_input_config_entries(s, "update");
    }

    /**
     * @brief Collect a simplemc::update_set from different MPI processes.
     *
     * @details It dispatches to tuple_set::mpi_collect_entries.
     *
     * @param comm simplemc::mpi::communicator object.
     * @param us Update set to reduce in place.
     */
    friend void simplemc_mpi_collect(const mpi::communicator& comm, update_set& us) { us.mpi_collect_entries(comm); }

private:
    std::discrete_distribution<std::size_t> dist_;
};

/// Deduction guide: deduce the update types from the constructor arguments.
template <mc_update... Us>
update_set(update<Us>...) -> update_set<Us...>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_UPDATE_SET_HPP
