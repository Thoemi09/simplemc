/**
 * @file
 * @brief Run-loop driver that owns all state for a single-process Monte Carlo simulation.
 */

#ifndef SIMPLEMC_MC_SIMULATION_HPP
#define SIMPLEMC_MC_SIMULATION_HPP

#include <simplemc/mc/measurement.hpp>
#include <simplemc/mc/measurement_stats.hpp>
#include <simplemc/mc/simulation_params.hpp>
#include <simplemc/mc/simulation_stats.hpp>
#include <simplemc/mc/update.hpp>
#include <simplemc/mc/update_stats.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>
#include <simplemc/utils/timer.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
 * @brief Single-process Monte Carlo run-loop driver.
 *
 * @details The driver owns the RNG, the updates, the per-update statistics, the measurements,
 * the per-measurement metadata, and the aggregate simulation statistics. Users build the
 * simulation up by calling simplemc::simulation::add_update and simplemc::simulation::add_measurement,
 * then invoke simplemc::simulation::run once or several times.
 *
 * The same loop is used for warm-up and production runs — call `run()` twice with two different
 * `simulation_params` objects. Measurements are still invoked during warm-up so the user can
 * monitor thermalization or autocorrelation; if their cost is unacceptable during warm-up the user
 * runs a separate `simulation` instance without them.
 *
 * Selection of updates uses `std::discrete_distribution<std::size_t>` built from the `weight`
 * field on each `update_stats` entry. The Metropolis comparison uses a separate
 * `std::uniform_real_distribution<double>` over \f$ [0, 1) \f$.
 *
 * Impossibility is signaled by an `attempt()` return value of \f$ \leq 0 \f$. In that case the
 * proposal is counted in `update_stats::nimps` and neither `accept()` nor `reject()` is invoked
 * on the wrapped update.
 *
 * The driver carries two independent backend parameters for serialization: `S1` for state
 * checkpointing and `S2` for input-config (user-facing config files). They default independently to
 * simplemc::json_serializer, so picking e.g. `simulation<hdf5_serializer>` keeps state checkpointing
 * in HDF5 while keeping input config in JSON without any extra typing.
 *
 * @tparam S1 Serializer type for state checkpointing (default: simplemc::json_serializer).
 * @tparam S2 Serializer type for input-config (default: simplemc::json_serializer).
 * @tparam RNG Random number generator type (default: simplemc::xoshiro256ss).
 */
template <serializer S1 = json_serializer, serializer S2 = json_serializer, class RNG = xoshiro256ss>
class simulation {
public:
    /**
     * @brief Type used for checkpoint serialization.
     */
    using cp_serializer_type = S1;

    /**
     * @brief Type used for input-config serialization.
     */
    using ic_serializer_type = S2;

    /**
     * @brief Default constructor.
     *
     * @details Default-constructs the RNG and leaves all containers empty. The simulation cannot
     * be run until at least one update with positive weight is registered via add_update().
     */
    simulation() = default;

    /**
     * @brief Construct with a user-supplied RNG.
     *
     * @details Moves `rng` into the driver. Use this to resume from a checkpointed RNG state.
     *
     * @param rng RNG to move in.
     */
    explicit simulation(RNG rng) : rng_ { std::move(rng) } {}

    /**
     * @brief Register a self-inverse update.
     *
     * @details Constructs an `update_stats` entry with `name`, `inv_name = name`, the given
     * weight, and `ratio = 1.0`. Throws simplemc::simplemc_exception if `name` is already used or
     * if `weight < 0`.
     *
     * @param u Update to register.
     * @param name Identifier (must be unique within this simulation).
     * @param weight Unnormalized selection weight (\f$ \ge 0 \f$).
     */
    void add_update(update<cp_serializer_type, ic_serializer_type> u, std::string name, double weight);

    /**
     * @brief Register an inverse pair of updates.
     *
     * @details Appends two `update_stats` entries with cross-linked `inv_name` and detailed-balance
     * ratios \f$ w_2 / w_1 \f$ and \f$ w_1 / w_2 \f$. Throws simplemc::simplemc_exception if either
     * name is already used (including the other one in the pair), if any weight is negative, or if
     * exactly one of the two weights is zero (the symmetric-zero invariant from new-simplemc). The
     * both-zero case is accepted and the ratios stay at `1.0`; the entries are never picked
     * anyway.
     *
     * @param u1 First update.
     * @param name1 Identifier for the first update.
     * @param weight1 Weight of the first update.
     * @param u2 Second update.
     * @param name2 Identifier for the second update.
     * @param weight2 Weight of the second update.
     */
    void add_update(update<cp_serializer_type, ic_serializer_type> u1, std::string name1, double weight1,
        update<cp_serializer_type, ic_serializer_type> u2, std::string name2, double weight2);

    /**
     * @brief Register a measurement.
     *
     * @details Appends a measurement and a corresponding `measurement_stats { name, is_active }`.
     * Throws simplemc::simplemc_exception if `name` is already used. Inactive measurements are
     * skipped during `cycle()` but otherwise stay in the simulation (visible via
     * measurement_stats_data(), serialized into checkpoints, toggleable via
     * set_measurement_active).
     *
     * @param m Measurement to register.
     * @param name Identifier (must be unique within this simulation).
     * @param is_active Initial activation state. Defaults to `true`.
     */
    void add_measurement(
        measurement<cp_serializer_type, ic_serializer_type> m, std::string name, bool is_active = true);

    /**
     * @brief Look up an update by name.
     *
     * @details Linear scan over the registered updates.
     *
     * @param name Update name.
     * @return Index of the update in the order it was added, or `std::nullopt` if no update with
     * the given name is registered.
     */
    [[nodiscard]] std::optional<std::size_t> find_update(std::string_view name) const noexcept;

    /**
     * @brief Look up a measurement by name.
     *
     * @details Linear scan over the registered measurements.
     *
     * @param name Measurement name.
     * @return Index of the measurement in the order it was added, or `std::nullopt` if no
     * measurement with the given name is registered.
     */
    [[nodiscard]] std::optional<std::size_t> find_measurement(std::string_view name) const noexcept;

    /**
     * @brief Change the selection weight of a registered update.
     *
     * @details Validates `w >= 0`, looks up the update by name (throwing if not found), and
     * writes the new weight. The internal selection distribution is not rebuilt — the user must
     * call run() (which rebuilds at entry) or initialize_update_distribution() before the next
     * sampling pass for the change to take effect.
     *
     * @param name Update to modify.
     * @param w New unnormalized selection weight.
     */
    void set_update_weight(std::string_view name, double w);

    /**
     * @brief Toggle the active state of a registered measurement.
     *
     * @details Looks up the measurement by name (throwing if not found) and writes the new flag.
     * Takes effect on the next `cycle()`.
     *
     * @param name Measurement to modify.
     * @param is_active New activation state.
     */
    void set_measurement_active(std::string_view name, bool is_active);

    /**
     * @brief Run the main loop until a stop criterion fires.
     *
     * @details Validates `p`, ensures at least one update has positive weight, calls
     * simplemc::simulation::initialize_update_distribution, then calls
     * simplemc::simulation::reset_stats so `p.max_steps` and `p.max_time` are interpreted as
     * "in this call". Cumulative counters are left untouched — call accumulate_stats() to fold
     * them in.
     *
     * The loop is nested: an outer while-loop polls the stop criteria every
     * `p.cycles_per_check` cycles; an inner for-loop runs that many cycles; each cycle runs
     * `p.steps_per_cycle` Metropolis steps followed by a single sweep over the active
     * measurements. The active set is sampled once at entry from `measurement_stats::is_active`;
     * if `p.skip_measurements` is `true` the active set is empty for the whole call. Toggling
     * `is_active` after `run()` starts has no effect (and is in any case not possible from the
     * outside while the blocking `run()` is executing).
     *
     * The stop criteria are `stats.steps_done < p.max_steps` and `stats.last_runtime <
     * p.max_time`. Because they are only re-checked at cycle-block boundaries, `steps_done`
     * may slightly exceed `p.max_steps` and `last_runtime` may slightly exceed `p.max_time` by
     * up to one block of `p.cycles_per_check * p.steps_per_cycle` steps.
     *
     * @param p Parameters for this run.
     */
    void run(const simulation_params& p);

    /**
     * @brief Build the internal selection distribution from the current update weights.
     *
     * @details Called automatically by run() at entry.
     *
     * Throws simplemc::simplemc_exception if no update has a positive weight; zero-weight entries
     * are passed through to `std::discrete_distribution` unchanged (they get probability 0).
     */
    void initialize_update_distribution();

    /**
     * @brief Fold the current-run counters into the cumulative counters on every owned aggregate.
     *
     * @details Calls simplemc::accumulate_simulation_stats on the simulation_stats and
     * simplemc::accumulate_update_stats on each entry of update_stats.
     */
    void accumulate_stats() noexcept;

    /**
     * @brief Zero the current-run counters on every owned aggregate.
     *
     * @details Calls simplemc::reset_simulation_stats on the simulation_stats and
     * simplemc::reset_update_stats on each entry of update_stats. Cumulative counters are left
     * untouched.
     */
    void reset_stats() noexcept;

    /**
     * @brief Read-only access to the aggregate simulation statistics.
     */
    [[nodiscard]] const simulation_stats& stats() const noexcept { return stats_; }

    /**
     * @brief Read-only view over the per-update statistics.
     */
    [[nodiscard]] std::span<const update_stats> update_stats_data() const noexcept { return update_stats_; }

    /**
     * @brief Read-only view over the per-measurement metadata.
     */
    [[nodiscard]] std::span<const measurement_stats> measurement_stats_data() const noexcept {
        return measurement_stats_;
    }

    /**
     * @brief Const access to the RNG.
     */
    [[nodiscard]] const RNG& rng() const noexcept { return rng_; }

    /**
     * @brief Mutable access to the RNG.
     *
     * @details Mutable because drawing from the RNG mutates its internal state. Users may also seed
     * it via this accessor.
     */
    [[nodiscard]] RNG& rng() noexcept { return rng_; }

    /**
     * @brief Serialize the persistent state of the simulation.
     *
     * @details Writes the RNG state, the cumulative simulation counters via simplemc_save on the
     * simulation_stats aggregate, and for each registered update / measurement its persistent
     * fields keyed by name via simplemc_save on the corresponding stats aggregate (see
     * @ref simplemc-mc for the schema). Per-run state (`steps_done`, `last_runtime`, per-update
     * current-run counters) is intentionally not written; `simulation_params` is per-call and not
     * part of the checkpoint either — serialize it separately if you need to persist user inputs.
     *
     * Each entry's user state is written under a `"user"` sub-key via the wrapper's `save_at`,
     * which routes through `s.save_at(key, concrete)` — picking up either ADL `simplemc_save<S1>`
     * or the backend's native fallback, or silently no-op'ing if neither is available.
     *
     * See also simplemc_save_input_config() for the parallel user-input flavor.
     */
    friend void simplemc_save(cp_serializer_type& s, const simulation& sim) {
        s.save_at("rng", sim.rng_);
        simplemc_save(s, sim.stats_);

        auto updates = s["updates"];
        for (std::size_t i = 0; i < sim.updates_.size(); ++i) {
            const auto& us = sim.update_stats_[i];
            auto entry = updates[us.name];
            simplemc_save(entry, us);
            sim.updates_[i].save_at(entry, "user");
        }

        auto meas = s["measurements"];
        for (std::size_t i = 0; i < sim.measurements_.size(); ++i) {
            const auto& ms = sim.measurement_stats_[i];
            auto entry = meas[ms.name];
            simplemc_save(entry, ms);
            sim.measurements_[i].save_at(entry, "user");
        }
    }

    /**
     * @brief Restore the persistent state of the simulation from a checkpoint.
     *
     * @details The destination simulation must already have the same set of updates / measurements
     * registered (matching names) before this call. Load patches the persistent fields onto the
     * existing entries; an entry present in the simulation but missing in the checkpoint throws
     * simplemc::simplemc_exception. Per-run state and the internal selection distribution are not
     * touched — the user calls `run()` (which rebuilds the distribution at entry) or
     * `initialize_update_distribution()` before sampling.
     */
    friend void simplemc_load(const cp_serializer_type& s, simulation& sim) {
        s.load_at("rng", sim.rng_);
        simplemc_load(s, sim.stats_);

        if (!sim.update_stats_.empty()) {
            const auto updates = s["updates"];
            for (std::size_t i = 0; i < sim.update_stats_.size(); ++i) {
                auto& us = sim.update_stats_[i];
                if (!updates.has(us.name)) {
                    throw simplemc_exception(fmt::format("checkpoint missing update '{}'", us.name));
                }
                const auto entry = updates[us.name];
                simplemc_load(entry, us);
                sim.updates_[i].load_at(entry, "user");
            }
        }

        if (!sim.measurement_stats_.empty()) {
            const auto meas = s["measurements"];
            for (std::size_t i = 0; i < sim.measurement_stats_.size(); ++i) {
                auto& ms = sim.measurement_stats_[i];
                if (!meas.has(ms.name)) {
                    throw simplemc_exception(fmt::format("checkpoint missing measurement '{}'", ms.name));
                }
                const auto entry = meas[ms.name];
                simplemc_load(entry, ms);
                sim.measurements_[i].load_at(entry, "user");
            }
        }
    }

    /**
     * @brief Serialize the user-facing input config of the simulation.
     *
     * @details Writes per-update `weight` and per-measurement `is_active`, keyed by name, plus
     * each wrapper's input-config user state under a `"user"` sub-key when the wrapped type opts
     * in. Omits the RNG state, cumulative counters, `inv_name`, and `ratio` — those are runtime
     * state or derived from the registration call.
     *
     * `simulation_params` is intentionally not included; the user holds it separately and
     * serializes it side-by-side via its own simplemc_save_input_config() free-function hook.
     *
     * The produced JSON is hand-editable as a config file: change a `weight` or toggle
     * `is_active`, reload via simplemc_load_input_config(), and the simulation picks up the new
     * settings.
     */
    friend void simplemc_save_input_config(ic_serializer_type& s, const simulation& sim) {
        auto updates = s["updates"];
        for (std::size_t i = 0; i < sim.updates_.size(); ++i) {
            const auto& us = sim.update_stats_[i];
            auto entry = updates[us.name];
            simplemc_save_input_config(entry, us);
            sim.updates_[i].save_input_config_at(entry, "user");
        }

        auto meas = s["measurements"];
        for (std::size_t i = 0; i < sim.measurements_.size(); ++i) {
            const auto& ms = sim.measurement_stats_[i];
            auto entry = meas[ms.name];
            simplemc_save_input_config(entry, ms);
            sim.measurements_[i].save_input_config_at(entry, "user");
        }
    }

    /**
     * @brief Restore the user-facing input config of the simulation.
     *
     * @details The destination simulation must already have the same set of updates / measurements
     * registered (matching names) before this call. An entry present in the simulation but missing
     * in the input config throws simplemc::simplemc_exception, mirroring simplemc_load. Per-field
     * presence is optional inside each entry (each stats hook uses `has()` guards), so an input
     * file may legitimately omit fields it doesn't override.
     *
     * The simulation's internal selection distribution is not rebuilt — call run() (which rebuilds
     * at entry) or initialize_update_distribution() before sampling if you changed any weight.
     */
    friend void simplemc_load_input_config(const ic_serializer_type& s, simulation& sim) {
        if (!sim.update_stats_.empty()) {
            const auto updates = s["updates"];
            for (std::size_t i = 0; i < sim.update_stats_.size(); ++i) {
                auto& us = sim.update_stats_[i];
                if (!updates.has(us.name)) {
                    throw simplemc_exception(fmt::format("input config missing update '{}'", us.name));
                }
                const auto entry = updates[us.name];
                simplemc_load_input_config(entry, us);
                sim.updates_[i].load_input_config_at(entry, "user");
            }
        }

        if (!sim.measurement_stats_.empty()) {
            const auto meas = s["measurements"];
            for (std::size_t i = 0; i < sim.measurement_stats_.size(); ++i) {
                auto& ms = sim.measurement_stats_[i];
                if (!meas.has(ms.name)) {
                    throw simplemc_exception(fmt::format("input config missing measurement '{}'", ms.name));
                }
                const auto entry = meas[ms.name];
                simplemc_load_input_config(entry, ms);
                sim.measurements_[i].load_input_config_at(entry, "user");
            }
        }
    }

private:
    /**
     * @brief Run a single Metropolis step.
     *
     * @details Samples an update via the internal `std::discrete_distribution`, increments the
     * proposal counter, calls `attempt()`, and applies the standard branch (impossible if
     * `attempt() * ratio <= 0`, accept if it beats a uniform draw, otherwise reject).
     */
    void step();

    /**
     * @brief Run `steps_per_cycle` consecutive steps, then invoke `measure()` on every active
     * measurement.
     *
     * @param steps_per_cycle Number of Metropolis steps before the measurement sweep.
     * @param active Indices of the measurements to invoke this cycle. Built once at the start of
     *               `run()` to avoid re-checking `measurement_stats::is_active` on every cycle.
     */
    void cycle(std::uint64_t steps_per_cycle, std::span<const std::size_t> active);

    RNG rng_ {};
    std::vector<update<cp_serializer_type, ic_serializer_type>> updates_;
    std::vector<update_stats> update_stats_;
    std::vector<measurement<cp_serializer_type, ic_serializer_type>> measurements_;
    std::vector<measurement_stats> measurement_stats_;
    simulation_stats stats_;

    std::discrete_distribution<std::size_t> dist_;
    std::uniform_real_distribution<double> uni01_ { 0.0, 1.0 };
    timer<> clock_;
};

template <serializer S1, serializer S2, class RNG>
std::optional<std::size_t> simulation<S1, S2, RNG>::find_update(std::string_view name) const noexcept {
    const auto it = std::ranges::find_if(update_stats_, [name](const update_stats& us) { return us.name == name; });
    if (it == update_stats_.end()) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(std::distance(update_stats_.begin(), it));
}

template <serializer S1, serializer S2, class RNG>
std::optional<std::size_t> simulation<S1, S2, RNG>::find_measurement(std::string_view name) const noexcept {
    const auto it =
        std::ranges::find_if(measurement_stats_, [name](const measurement_stats& ms) { return ms.name == name; });
    if (it == measurement_stats_.end()) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(std::distance(measurement_stats_.begin(), it));
}

template <serializer S1, serializer S2, class RNG>
void simulation<S1, S2, RNG>::add_update(update<S1, S2> u, std::string name, double weight) {
    if (weight < 0.0) {
        throw simplemc_exception(fmt::format("update weight must be >= 0 (got {} on '{}')", weight, name));
    }
    if (find_update(name).has_value()) {
        throw simplemc_exception(fmt::format("update '{}' is already registered", name));
    }

    updates_.push_back(std::move(u));
    update_stats_.push_back({ .name = name, .inv_name = std::move(name), .weight = weight, .ratio = 1.0 });
}

template <serializer S1, serializer S2, class RNG>
void simulation<S1, S2, RNG>::add_update(
    update<S1, S2> u1, std::string name1, double weight1, update<S1, S2> u2, std::string name2, double weight2) {
    if (weight1 < 0.0 || weight2 < 0.0) {
        throw simplemc_exception(
            fmt::format("update weights must be >= 0 (got {} on '{}' and {} on '{}')", weight1, name1, weight2, name2));
    }
    if ((weight1 == 0.0) != (weight2 == 0.0)) {
        throw simplemc_exception(
            fmt::format("inverse pair must have both weights zero or both non-zero (got {} on '{}' and {} on '{}')",
                weight1, name1, weight2, name2));
    }
    if (name1 == name2) {
        throw simplemc_exception(fmt::format("paired updates must have distinct names (both are '{}')", name1));
    }
    if (find_update(name1)) {
        throw simplemc_exception(fmt::format("update '{}' is already registered", name1));
    }
    if (find_update(name2)) {
        throw simplemc_exception(fmt::format("update '{}' is already registered", name2));
    }

    const double r1 = (weight1 == 0.0) ? 1.0 : weight2 / weight1;
    const double r2 = (weight2 == 0.0) ? 1.0 : weight1 / weight2;

    updates_.push_back(std::move(u1));
    update_stats_.push_back({ .name = name1, .inv_name = name2, .weight = weight1, .ratio = r1 });

    updates_.push_back(std::move(u2));
    update_stats_.push_back({ .name = std::move(name2), .inv_name = std::move(name1), .weight = weight2, .ratio = r2 });
}

template <serializer S1, serializer S2, class RNG>
void simulation<S1, S2, RNG>::add_measurement(measurement<S1, S2> m, std::string name, bool is_active) {
    if (find_measurement(name)) {
        throw simplemc_exception(fmt::format("measurement '{}' is already registered", name));
    }
    measurements_.push_back(std::move(m));
    measurement_stats_.push_back({ .name = std::move(name), .is_active = is_active });
}

template <serializer S1, serializer S2, class RNG>
void simulation<S1, S2, RNG>::set_update_weight(std::string_view name, double w) {
    if (w < 0.0) {
        throw simplemc_exception(fmt::format("update weight must be >= 0 (got {} on '{}')", w, name));
    }
    const auto idx = find_update(name);
    if (!idx) {
        throw simplemc_exception(fmt::format("update '{}' is not registered", name));
    }
    update_stats_[*idx].weight = w;
}

template <serializer S1, serializer S2, class RNG>
void simulation<S1, S2, RNG>::set_measurement_active(std::string_view name, bool is_active) {
    const auto idx = find_measurement(name);
    if (!idx) {
        throw simplemc_exception(fmt::format("measurement '{}' is not registered", name));
    }
    measurement_stats_[*idx].is_active = is_active;
}

template <serializer S1, serializer S2, class RNG>
void simulation<S1, S2, RNG>::initialize_update_distribution() {
    auto weights = update_stats_ | std::views::transform(&update_stats::weight);
    if (!std::ranges::any_of(weights, [](double w) { return w > 0.0; })) {
        throw simplemc_exception("at least one update weight must be > 0");
    }
    dist_ = std::discrete_distribution<std::size_t> { weights.begin(), weights.end() };
}

template <serializer S1, serializer S2, class RNG>
void simulation<S1, S2, RNG>::step() {
    const std::size_t idx = dist_(rng_);
    ++stats_.steps_done;
    ++update_stats_[idx].nprops;
    const double p = updates_[idx].attempt() * update_stats_[idx].ratio;
    if (p <= 0.0) {
        ++update_stats_[idx].nimps;
    } else if (p >= uni01_(rng_)) {
        updates_[idx].accept();
        ++update_stats_[idx].naccs;
    } else {
        updates_[idx].reject();
    }
}

template <serializer S1, serializer S2, class RNG>
void simulation<S1, S2, RNG>::cycle(std::uint64_t steps_per_cycle, std::span<const std::size_t> active) {
    for (std::uint64_t s = 0; s < steps_per_cycle; ++s) {
        step();
    }
    for (std::size_t i : active) {
        measurements_[i].measure();
    }
}

template <serializer S1, serializer S2, class RNG>
void simulation<S1, S2, RNG>::run(const simulation_params& p) {
    validate_simulation_params(p);
    if (updates_.empty()) {
        throw simplemc_exception("no updates registered");
    }
    initialize_update_distribution();
    reset_stats();

    std::vector<std::size_t> active_measurements;
    if (!p.skip_measurements) {
        active_measurements.reserve(measurements_.size());
        for (std::size_t i = 0; i < measurements_.size(); ++i) {
            if (measurement_stats_[i].is_active) {
                active_measurements.push_back(i);
            }
        }
    }

    clock_.start();
    while (stats_.steps_done < p.max_steps && stats_.last_runtime < p.max_time) {
        for (std::uint64_t c = 0; c < p.cycles_per_check; ++c) {
            cycle(p.steps_per_cycle, active_measurements);
        }
        stats_.last_runtime = clock_.elapsed();
    }
}

template <serializer S1, serializer S2, class RNG>
void simulation<S1, S2, RNG>::accumulate_stats() noexcept {
    accumulate_simulation_stats(stats_);
    for (auto& us : update_stats_) {
        accumulate_update_stats(us);
    }
}

template <serializer S1, serializer S2, class RNG>
void simulation<S1, S2, RNG>::reset_stats() noexcept {
    reset_simulation_stats(stats_);
    for (auto& us : update_stats_) {
        reset_update_stats(us);
    }
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_SIMULATION_HPP
