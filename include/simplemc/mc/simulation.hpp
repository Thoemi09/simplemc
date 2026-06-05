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
#include <simplemc/utils/simplemc_exception.hpp>
#include <simplemc/utils/timer.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
 * @tparam RNG Random number generator type (default: `simplemc::xoshiro256ss`).
 */
template <class RNG = xoshiro256ss>
class simulation {
public:
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
    void add_update(update u, std::string name, double weight);

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
    void add_update(update u1, std::string name1, double weight1, update u2, std::string name2, double weight2);

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
    void add_measurement(measurement m, std::string name, bool is_active = true);

    /**
     * @brief Look up an update by name.
     *
     * @details Linear scan over the registered updates. Throws simplemc::simplemc_exception if
     * no update with the given name is registered.
     *
     * @param name Update name.
     * @return Index of the update in the order it was added.
     */
    [[nodiscard]] std::size_t find_update(std::string_view name) const;

    /**
     * @brief Look up a measurement by name.
     *
     * @details Linear scan over the registered measurements. Throws simplemc::simplemc_exception
     * if no measurement with the given name is registered.
     *
     * @param name Measurement name.
     * @return Index of the measurement in the order it was added.
     */
    [[nodiscard]] std::size_t find_measurement(std::string_view name) const;

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
     * `p.steps_per_cycle` Metropolis steps followed by a single sweep over the measurements.
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
     * @brief Run a single Metropolis step.
     *
     * @details Samples an update via the internal `std::discrete_distribution`, increments the
     * proposal counter, calls `attempt()`, and applies the standard branch (impossible if
     * `attempt() * ratio <= 0`, accept if it beats a uniform draw, otherwise reject).
     *
     * @pre The distribution must already be built. `run()` builds it automatically at entry; users
     * who call `step()` or `cycle()` directly must call initialize_update_distribution() first.
     */
    void step();

    /**
     * @brief Run `steps_per_cycle` consecutive steps, then invoke `measure()` on every measurement.
     *
     * @param steps_per_cycle Number of Metropolis steps before the measurement sweep.
     */
    void cycle(std::uint64_t steps_per_cycle);

    /**
     * @brief Build the internal selection distribution from the current update weights.
     *
     * @details Called automatically by run() at entry. Exposed so users who drive step() or
     * cycle() directly can build the distribution manually after their add_update calls.
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

private:
    [[nodiscard]] bool has_update_with_name(std::string_view name) const noexcept;
    [[nodiscard]] bool has_measurement_with_name(std::string_view name) const noexcept;

    RNG rng_ {};
    std::vector<update> updates_;
    std::vector<update_stats> update_stats_;
    std::vector<measurement> measurements_;
    std::vector<measurement_stats> measurement_stats_;
    simulation_stats stats_;

    std::discrete_distribution<std::size_t> dist_;
    std::uniform_real_distribution<double> uni01_ { 0.0, 1.0 };
    timer<> clock_;
};

template <class RNG>
bool simulation<RNG>::has_update_with_name(std::string_view name) const noexcept {
    return std::ranges::any_of(update_stats_, [name](const update_stats& us) { return us.name == name; });
}

template <class RNG>
bool simulation<RNG>::has_measurement_with_name(std::string_view name) const noexcept {
    return std::ranges::any_of(measurement_stats_, [name](const measurement_stats& ms) { return ms.name == name; });
}

template <class RNG>
void simulation<RNG>::add_update(update u, std::string name, double weight) {
    if (weight < 0.0) {
        throw simplemc_exception(fmt::format("update weight must be >= 0 (got {} on '{}')", weight, name));
    }
    if (has_update_with_name(name)) {
        throw simplemc_exception(fmt::format("update '{}' is already registered", name));
    }

    updates_.push_back(std::move(u));
    update_stats_.push_back({ .name = name, .inv_name = std::move(name), .weight = weight, .ratio = 1.0 });
}

template <class RNG>
void simulation<RNG>::add_update(
    update u1, std::string name1, double weight1, update u2, std::string name2, double weight2) {
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
    if (has_update_with_name(name1)) {
        throw simplemc_exception(fmt::format("update '{}' is already registered", name1));
    }
    if (has_update_with_name(name2)) {
        throw simplemc_exception(fmt::format("update '{}' is already registered", name2));
    }

    const double r1 = (weight1 == 0.0) ? 1.0 : weight2 / weight1;
    const double r2 = (weight2 == 0.0) ? 1.0 : weight1 / weight2;

    updates_.push_back(std::move(u1));
    update_stats_.push_back({ .name = name1, .inv_name = name2, .weight = weight1, .ratio = r1 });

    updates_.push_back(std::move(u2));
    update_stats_.push_back({ .name = std::move(name2), .inv_name = std::move(name1), .weight = weight2, .ratio = r2 });
}

template <class RNG>
void simulation<RNG>::add_measurement(measurement m, std::string name, bool is_active) {
    if (has_measurement_with_name(name)) {
        throw simplemc_exception(fmt::format("measurement '{}' is already registered", name));
    }
    measurements_.push_back(std::move(m));
    measurement_stats_.push_back({ .name = std::move(name), .is_active = is_active });
}

template <class RNG>
std::size_t simulation<RNG>::find_update(std::string_view name) const {
    const auto it = std::ranges::find_if(update_stats_, [name](const update_stats& us) { return us.name == name; });
    if (it == update_stats_.end()) {
        throw simplemc_exception(fmt::format("update '{}' is not registered", name));
    }
    return static_cast<std::size_t>(std::distance(update_stats_.begin(), it));
}

template <class RNG>
std::size_t simulation<RNG>::find_measurement(std::string_view name) const {
    const auto it =
        std::ranges::find_if(measurement_stats_, [name](const measurement_stats& ms) { return ms.name == name; });
    if (it == measurement_stats_.end()) {
        throw simplemc_exception(fmt::format("measurement '{}' is not registered", name));
    }
    return static_cast<std::size_t>(std::distance(measurement_stats_.begin(), it));
}

template <class RNG>
void simulation<RNG>::set_update_weight(std::string_view name, double w) {
    if (w < 0.0) {
        throw simplemc_exception(fmt::format("update weight must be >= 0 (got {} on '{}')", w, name));
    }
    update_stats_[find_update(name)].weight = w;
}

template <class RNG>
void simulation<RNG>::set_measurement_active(std::string_view name, bool is_active) {
    measurement_stats_[find_measurement(name)].is_active = is_active;
}

template <class RNG>
void simulation<RNG>::initialize_update_distribution() {
    auto weights = update_stats_ | std::views::transform(&update_stats::weight);
    if (!std::ranges::any_of(weights, [](double w) { return w > 0.0; })) {
        throw simplemc_exception("at least one update weight must be > 0");
    }
    dist_ = std::discrete_distribution<std::size_t> { weights.begin(), weights.end() };
}

template <class RNG>
void simulation<RNG>::step() {
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

template <class RNG>
void simulation<RNG>::cycle(std::uint64_t steps_per_cycle) {
    for (std::uint64_t s = 0; s < steps_per_cycle; ++s) {
        step();
    }
    for (std::size_t i = 0; i < measurements_.size(); ++i) {
        if (measurement_stats_[i].is_active) {
            measurements_[i].measure();
        }
    }
}

template <class RNG>
void simulation<RNG>::run(const simulation_params& p) {
    validate_simulation_params(p);
    if (updates_.empty()) {
        throw simplemc_exception("no updates registered");
    }
    initialize_update_distribution();
    reset_stats();
    clock_.start();
    while (stats_.steps_done < p.max_steps && stats_.last_runtime < p.max_time) {
        for (std::uint64_t c = 0; c < p.cycles_per_check; ++c) {
            cycle(p.steps_per_cycle);
        }
        stats_.last_runtime = clock_.elapsed();
    }
}

template <class RNG>
void simulation<RNG>::accumulate_stats() noexcept {
    accumulate_simulation_stats(stats_);
    for (auto& us : update_stats_) {
        accumulate_update_stats(us);
    }
}

template <class RNG>
void simulation<RNG>::reset_stats() noexcept {
    reset_simulation_stats(stats_);
    for (auto& us : update_stats_) {
        reset_update_stats(us);
    }
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_SIMULATION_HPP
