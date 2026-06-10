/**
 * @file
 * @brief Run-loop driver that owns all state for a single-process Monte Carlo simulation.
 */

#ifndef SIMPLEMC_MC_SIMULATION_HPP
#define SIMPLEMC_MC_SIMULATION_HPP

#include <simplemc/mc/basic_measurement.hpp>
#include <simplemc/mc/basic_update.hpp>
#include <simplemc/mc/measurement.hpp>
#include <simplemc/mc/measurement_set.hpp>
#include <simplemc/mc/kernels.hpp>
#include <simplemc/mc/run.hpp>
#include <simplemc/mc/simulation_params.hpp>
#include <simplemc/mc/simulation_stats.hpp>
#include <simplemc/mc/traits.hpp>
#include <simplemc/mc/update.hpp>
#include <simplemc/mc/update_set.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc
 * @{
 */

/**
 * @brief Single-process Monte Carlo run-loop driver.
 *
 * @details The driver owns the RNG, the registered updates (each carrying its own metadata and
 * counters via simplemc::update), the registered measurements (each carrying its own metadata via
 * simplemc::measurement), and the aggregate simulation statistics. Users build the simulation up by
 * calling simplemc::simulation::add_update and simplemc::simulation::add_measurement, then invoke
 * simplemc::simulation::run once or several times.
 *
 * The same loop is used for warm-up and production runs — call `run()` twice with two different
 * `simulation_params` objects. Measurements are still invoked during warm-up so the user can
 * monitor thermalization or autocorrelation; if their cost is unacceptable during warm-up the user
 * runs a separate `simulation` instance without them, or sets `p.skip_measurements = true`.
 *
 * Selection of updates uses `std::discrete_distribution<std::size_t>` built from the `weight`
 * field on each simplemc::update entry. The Metropolis comparison uses a separate
 * `std::uniform_real_distribution<double>` over \f$ [0, 1) \f$.
 *
 * Impossibility is signaled by an `attempt()` return value of \f$ \leq 0 \f$. In that case the
 * proposal is counted in simplemc::update::nimps and `reject()` is invoked on the wrapped update,
 * so every `attempt()` is followed by exactly one of `accept()` / `reject()`.
 *
 * The driver is a regular movable type: it owns only value members (RNG, sets, stats) and
 * constructs a throwaway simplemc::metropolis_kernel over its own update set for each run.
 *
 * The driver is parameterized on a simplemc::mc_traits_like bundle that carries the two independent
 * serializer backends (`checkpoint_serializer_type` for state checkpointing,
 * `input_config_serializer_type` for user-facing config files) and the `rng_type`. They default
 * independently via simplemc::mc_traits, so picking e.g. `simulation<mc_traits<hdf5_serializer>>`
 * keeps state checkpointing in HDF5 while keeping input config in JSON without any extra typing.
 *
 * @tparam Traits Traits bundle satisfying simplemc::mc_traits_like (default: simplemc::mc_traits<>).
 */
template <mc_traits_like Traits = mc_traits<>>
class simulation {
public:
    /**
     * @brief The traits bundle this simulation was parameterized on.
     */
    using traits_type = Traits;

    /**
     * @brief Type used for checkpoint serialization.
     */
    using checkpoint_serializer_type = typename Traits::checkpoint_serializer_type;

    /**
     * @brief Type used for input-config serialization.
     */
    using input_config_serializer_type = typename Traits::input_config_serializer_type;

    /**
     * @brief Random number generator type.
     */
    using rng_type = typename Traits::rng_type;

    /**
     * @brief Type of one registered update entry.
     */
    using update_type = update<Traits>;

    /**
     * @brief Type of one registered measurement entry.
     */
    using measurement_type = measurement<Traits>;

    /**
     * @brief Default constructor.
     */
    simulation() = default;

    /**
     * @brief Construct with a user-supplied RNG.
     */
    explicit simulation(rng_type rng) : rng_ { std::move(rng) } {}

    /**
     * @brief Register a self-inverse update.
     *
     * @details Forwards `u` (a user value or a pre-built simplemc::basic_update) into a new
     * simplemc::update entry, which validates `name` non-empty and `weight >= 0`. Throws
     * simplemc::simplemc_exception if `name` is already used.
     *
     * @param u Update to register (user value or pre-built simplemc::basic_update).
     * @param name Identifier (must be unique within this simulation).
     * @param weight Unnormalized selection weight (\f$ \ge 0 \f$).
     */
    void add_update(basic_update<Traits> u, std::string name, double weight);

    /**
     * @brief Register an inverse pair of updates.
     *
     * @details Constructs two simplemc::update entries with cross-linked `inv_name` and
     * detailed-balance ratios \f$ w_2 / w_1 \f$ and \f$ w_1 / w_2 \f$. Throws
     * simplemc::simplemc_exception if either name is already used (including the other one in the
     * pair), if any weight is negative, or if exactly one of the two weights is zero (the
     * symmetric-zero invariant from new-simplemc). The both-zero case is accepted and the ratios
     * stay at `1.0`; the entries are never picked anyway.
     *
     * @param u1 First update.
     * @param name1 Identifier for the first update.
     * @param weight1 Weight of the first update.
     * @param u2 Second update.
     * @param name2 Identifier for the second update.
     * @param weight2 Weight of the second update.
     */
    void add_update(basic_update<Traits> u1, std::string name1, double weight1,
        basic_update<Traits> u2, std::string name2, double weight2);

    /**
     * @brief Register a measurement.
     *
     * @details Throws simplemc::simplemc_exception if `name` is already used. Inactive measurements
     * are skipped during `cycle()` but otherwise stay in the simulation (visible via
     * measurement_data(), serialized into checkpoints, toggleable via set_measurement_active).
     *
     * @param m Measurement to register.
     * @param name Identifier (must be unique within this simulation).
     * @param is_active Initial activation state. Defaults to `true`.
     */
    void add_measurement(basic_measurement<Traits> m, std::string name,
        bool is_active = true);

    /**
     * @brief Look up an update by name.
     */
    [[nodiscard]] std::optional<std::size_t> find_update(std::string_view name) const noexcept;

    /**
     * @brief Look up a measurement by name.
     */
    [[nodiscard]] std::optional<std::size_t> find_measurement(std::string_view name) const noexcept;

    /**
     * @brief Change the selection weight of a registered update.
     *
     * @details Validates `w >= 0`, looks up the update by name (throwing if not found), and
     * writes the new weight. The internal selection distribution is not rebuilt — the user must
     * call run() (which rebuilds at entry) or initialize_update_distribution() before the next
     * sampling pass for the change to take effect.
     */
    void set_update_weight(std::string_view name, double w);

    /**
     * @brief Toggle the active state of a registered measurement.
     */
    void set_measurement_active(std::string_view name, bool is_active);

    /**
     * @brief Recover a typed pointer to a registered update's wrapped user value.
     *
     * @tparam T Concrete user type.
     * @param name Update name.
     * @return Pointer to the wrapped user value, or `nullptr` if the name is unknown or the type
     *         does not match.
     */
    template <class T>
    [[nodiscard]] T* get_update(std::string_view name) noexcept {
        return updates_.template get<T>(name);
    }

    /**
     * @brief Const overload of get_update().
     */
    template <class T>
    [[nodiscard]] const T* get_update(std::string_view name) const noexcept {
        return updates_.template get<T>(name);
    }

    /**
     * @brief Recover a typed pointer to a registered measurement's wrapped user value.
     */
    template <class T>
    [[nodiscard]] T* get_measurement(std::string_view name) noexcept {
        return measurements_.template get<T>(name);
    }

    /**
     * @brief Const overload of get_measurement().
     */
    template <class T>
    [[nodiscard]] const T* get_measurement(std::string_view name) const noexcept {
        return measurements_.template get<T>(name);
    }

    /**
     * @brief Run the main loop until a stop criterion fires.
     *
     * @details Forwards to the free simplemc::run with this aggregate's kernel, measurement set,
     * stats, and RNG. The run's final totals overwrite the `last_*` fields of stats at exit;
     * cumulative counters are left untouched — call accumulate_stats() to fold them in. Throws if
     * no updates have been registered.
     *
     * @tparam Cbs Callbacks bundle satisfying simplemc::mc_run_callbacks. Defaults to
     *             simplemc::run_callbacks<> (all no-ops).
     *
     * @param p Parameters for this run.
     * @param cbs Optional callbacks. Defaults to all no-ops.
     */
    template <mc_run_callbacks Cbs = run_callbacks<>>
    void run(const simulation_params& p, const Cbs& cbs = {}) {
        if (updates_.empty()) {
            throw simplemc_exception("no updates registered");
        }
        metropolis_kernel<Traits> kernel { updates_ };
        simplemc::run(kernel, measurements_, p, stats_, rng_, cbs);
    }

    /**
     * @brief Build the internal selection distribution from the current update weights.
     *
     * @details Called automatically by run() at entry. Throws simplemc::simplemc_exception if no
     * update has a positive weight; zero-weight entries are passed through to
     * `std::discrete_distribution` unchanged (they get probability 0).
     */
    void initialize_update_distribution();

    /**
     * @brief Fold the current-run counters into the cumulative counters on every owned aggregate.
     */
    void accumulate_stats() noexcept;

    /**
     * @brief Zero the current-run counters on every owned aggregate.
     */
    void reset_stats() noexcept;

    /**
     * @brief Read-only access to the aggregate simulation statistics.
     */
    [[nodiscard]] const simulation_stats& stats() const noexcept { return stats_; }

    /**
     * @brief Read-only view over the per-update entries.
     */
    [[nodiscard]] std::span<const update_type> update_data() const noexcept { return updates_.data(); }

    /**
     * @brief Const access to the owned update set.
     */
    [[nodiscard]] const update_set<Traits>& updates() const noexcept { return updates_; }

    /**
     * @brief Mutable access to the owned update set.
     *
     * @details Exposed so a kernel can hold a non-owning pointer/reference to it. Mutating updates
     * through this handle has the same effect as calling the corresponding `simulation` method.
     */
    [[nodiscard]] update_set<Traits>& updates() noexcept { return updates_; }

    /**
     * @brief Read-only view over the per-measurement entries.
     */
    [[nodiscard]] std::span<const measurement_type> measurement_data() const noexcept {
        return measurements_.data();
    }

    /**
     * @brief Const access to the owned measurement set.
     */
    [[nodiscard]] const measurement_set<Traits>& measurements() const noexcept { return measurements_; }

    /**
     * @brief Mutable access to the owned measurement set.
     */
    [[nodiscard]] measurement_set<Traits>& measurements() noexcept { return measurements_; }

    /**
     * @brief Const access to the RNG.
     */
    [[nodiscard]] const rng_type& rng() const noexcept { return rng_; }

    /**
     * @brief Mutable access to the RNG.
     */
    [[nodiscard]] rng_type& rng() noexcept { return rng_; }

    /**
     * @brief Serialize the persistent state of the simulation.
     *
     * @details Writes the RNG state, the cumulative simulation counters, and each registered update
     * and measurement keyed by name. Per-update / per-measurement entries delegate to the per-entity
     * `simplemc_save` overload, which preserves the previous on-disk schema (the stats fields under
     * the named key, plus a `"user"` sub-tree from the wrapped user type).
     *
     * Per-run state (`last_steps_done`, `last_runtime`, per-update current-run counters) is
     * intentionally not written; `simulation_params` is per-call and not part of the checkpoint
     * either.
     */
    friend void simplemc_save(checkpoint_serializer_type& s, const simulation& sim) {
        s.save_at("rng", sim.rng_);
        simplemc_save(s, sim.stats_);

        auto updates = s["updates"];
        simplemc_save(updates, sim.updates_);

        auto meas = s["measurements"];
        simplemc_save(meas, sim.measurements_);
    }

    /**
     * @brief Restore the persistent state of the simulation from a checkpoint.
     *
     * @details The destination simulation must already have the same set of updates / measurements
     * registered (matching names) before this call. Load patches the persistent fields onto the
     * existing entries; an entry present in the simulation but missing in the checkpoint throws
     * simplemc::simplemc_exception. Per-run state and the internal selection distribution are not
     * touched.
     */
    friend void simplemc_load(const checkpoint_serializer_type& s, simulation& sim) {
        s.load_at("rng", sim.rng_);
        simplemc_load(s, sim.stats_);

        if (!sim.updates_.empty()) {
            const auto updates = s["updates"];
            simplemc_load(updates, sim.updates_);
        }

        if (!sim.measurements_.empty()) {
            const auto meas = s["measurements"];
            simplemc_load(meas, sim.measurements_);
        }
    }

    /**
     * @brief Serialize the user-facing input config of the simulation.
     *
     * @details Writes per-update `weight` and per-measurement `is_active`, keyed by name, plus
     * each wrapper's input-config user state under a `"user"` sub-key when the wrapped type opts in.
     */
    friend void simplemc_save_input_config(input_config_serializer_type& s, const simulation& sim) {
        auto updates = s["updates"];
        simplemc_save_input_config(updates, sim.updates_);

        auto meas = s["measurements"];
        simplemc_save_input_config(meas, sim.measurements_);
    }

    /**
     * @brief Restore the user-facing input config of the simulation.
     *
     * @details The destination simulation must already have the same set of updates / measurements
     * registered (matching names) before this call. An entry present in the simulation but missing
     * in the input config throws simplemc::simplemc_exception, mirroring simplemc_load.
     */
    friend void simplemc_load_input_config(const input_config_serializer_type& s, simulation& sim) {
        if (!sim.updates_.empty()) {
            const auto updates = s["updates"];
            simplemc_load_input_config(updates, sim.updates_);
        }

        if (!sim.measurements_.empty()) {
            const auto meas = s["measurements"];
            simplemc_load_input_config(meas, sim.measurements_);
        }
    }

    /**
     * @brief All-reduce every component of the simulation across MPI ranks.
     *
     * @details Composite reducer: forwards to simplemc_mpi_collect on the contained
     * simulation_stats, update_set, and measurement_set in turn. The RNG and the internal selection
     * distribution are intentionally not touched. Reduction is in-place, matching the convention of
     * the contained sets.
     *
     * @param comm MPI communicator over which to reduce.
     * @param sim Simulation to reduce in place.
     */
    friend void simplemc_mpi_collect(const mpi::communicator& comm, simulation& sim) {
        simplemc_mpi_collect(comm, sim.stats_);
        simplemc_mpi_collect(comm, sim.updates_);
        simplemc_mpi_collect(comm, sim.measurements_);
    }

private:
    rng_type rng_ {};
    update_set<Traits> updates_;
    measurement_set<Traits> measurements_;
    simulation_stats stats_;
};

template <mc_traits_like Traits>
std::optional<std::size_t> simulation<Traits>::find_update(std::string_view name) const noexcept {
    return updates_.find(name);
}

template <mc_traits_like Traits>
std::optional<std::size_t> simulation<Traits>::find_measurement(std::string_view name) const noexcept {
    return measurements_.find(name);
}

template <mc_traits_like Traits>
void simulation<Traits>::add_update(basic_update<Traits> u, std::string name, double weight) {
    updates_.add(update_type { std::move(u), std::move(name), weight });
}

template <mc_traits_like Traits>
void simulation<Traits>::add_update(basic_update<Traits> u1, std::string name1, double weight1,
    basic_update<Traits> u2, std::string name2, double weight2) {
    updates_.add_pair(update_type { std::move(u1), std::move(name1), weight1 },
        update_type { std::move(u2), std::move(name2), weight2 });
}

template <mc_traits_like Traits>
void simulation<Traits>::add_measurement(basic_measurement<Traits> m, std::string name, bool is_active) {
    measurements_.add(measurement_type { std::move(m), std::move(name), is_active });
}

template <mc_traits_like Traits>
void simulation<Traits>::set_update_weight(std::string_view name, double w) {
    updates_.set_weight(name, w);
}

template <mc_traits_like Traits>
void simulation<Traits>::set_measurement_active(std::string_view name, bool is_active) {
    measurements_.set_active(name, is_active);
}

template <mc_traits_like Traits>
void simulation<Traits>::initialize_update_distribution() {
    updates_.rebuild_distribution();
}

template <mc_traits_like Traits>
void simulation<Traits>::accumulate_stats() noexcept {
    accumulate_simulation_stats(stats_);
    updates_.accumulate_counters();
}

template <mc_traits_like Traits>
void simulation<Traits>::reset_stats() noexcept {
    reset_simulation_stats(stats_);
    updates_.reset_run_counters();
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_SIMULATION_HPP
