/**
 * @file
 * @brief Lifecycle, predicates, and bookkeeping for a single Monte Carlo simulation run.
 */

#ifndef SIMPLEMC_MC_SIMULATION_RUN_HPP
#define SIMPLEMC_MC_SIMULATION_RUN_HPP

#include <simplemc/mc/simulation_params.hpp>
#include <simplemc/mc/simulation_stats.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>
#include <simplemc/utils/timer.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <chrono>
#include <cstdio>

namespace simplemc {

/**
 * @addtogroup simplemc-mc
 * @{
 */

/**
 * @brief Lifecycle, predicates, and bookkeeping for a single Monte Carlo simulation run.
 *
 * @details It owns a `std::stead_clock`-backed simplemc::timer, the parameters, and the running
 * statistics of a single MC simulation run. Parameters are validated on assignment and construction
 * using simplemc::validate_simulation_params.
 *
 * Lifecycle in expected order:
 *
 * 1. Construct with default or user-supplied simplemc::simulation_params.
 * 2. Call start() — resets simulation_stats::steps_done and simulation_stats::last_runtime to zero
 * and start the time.
 * 3. MC driver runs steps, polling has_steps_remaining() and has_time_remaining() for stopping
 * conditions.
 * 4. Call stop() — stops the timer and updates simulation_stats::last_runtime,
 * simulation_stats::cumulative_steps, and simulation_stats::cumulative_time.
 *
 * The driver itself is not part of this class — it lives in its own header. This object only holds
 * the state.
 */
class simulation_run {
public:
    /**
     * @brief Construct with the given simulation parameters and validate them.
     *
     * @param params Simulation parameters.
     */
    explicit simulation_run(simulation_params params = {}) : params_ { params } { validate_simulation_params(params_); }

    /**
     * @brief Replace the simulation parameters.
     *
     * @details It throws a simplemc::simplemc_exception if any field violates its invariant by
     * calling simplemc::validate_simulation_params() on the new parameters before assignment.
     *
     * @param p New simulation parameters.
     */
    void set_params(const simulation_params& p) {
        validate_simulation_params(p);
        params_ = p;
    }

    /**
     * @brief Read-only access to the current parameters.
     *
     * @return Const reference to the parameter set.
     */
    [[nodiscard]] const simulation_params& params() const noexcept { return params_; }

    /**
     * @brief Read-only access to the running statistics.
     *
     * @return Const reference to the statistics.
     */
    [[nodiscard]] const simulation_stats& stats() const noexcept { return stats_; }

    /**
     * @brief Start (or restart) the timer and reset per-run statistics.
     *
     * @details It sets simulation_stats::steps_done and simulation_stats::last_runtime to zero and
     * calls timer::start() on the internal timer object.
     *
     * Cumulative statistics are left untouched.
     */
    void start() {
        stats_.steps_done = 0;
        stats_.last_runtime = 0.0;
        clock_.start();
    }

    /**
     * @brief Stop the timer and finalize bookkeeping for the just-completed simulation run.
     *
     * @details It records the simulation's wall-clock duration in simulation_stats::last_runtime and
     * updates the cumulative statistics, simulation_stats::cumulative_steps and
     * simulation_stats::cumulative_time.
     */
    void stop() {
        clock_.stop();
        stats_.last_runtime = time_passed(clock_.start_time(), clock_.stop_time());
        stats_.cumulative_steps += stats_.steps_done;
        stats_.cumulative_time += stats_.last_runtime;
    }

    /**
     * @brief Reset the simulation statistics to its default state.
     */
    void reset_stats() noexcept { stats_ = simulation_stats {}; }

    /**
     * @brief Live wall-clock time, in seconds, since the most recent start().
     *
     * @details It reads the underlying timer and keeps advancing after stop() until the next start().
     * For a finalized bracket duration use stats() and query `last_runtime`.
     *
     * @return Elapsed seconds.
     */
    [[nodiscard]] double runtime() const { return clock_.elapsed(); }

    /**
     * @brief Whether the driver should keep running based on the step budget.
     *
     * @details It compares simulation_stats::steps_done to simulation_params::max_steps.
     *
     * @return True iff the number of steps done is below the maximum number of steps to do.
     */
    [[nodiscard]] bool has_steps_remaining() const noexcept { return stats_.steps_done < params_.max_steps; }

    /**
     * @brief Whether the driver should keep running based on the time budget.
     *
     * @details It compares the live runtime() to simulation_params::max_time.
     *
     * @return True iff the runtime is below the maximum time to run.
     */
    [[nodiscard]] bool has_time_remaining() const { return runtime() < params_.max_time; }

    /**
     * @brief Print the simulation parameters to a file handle as a human-readable block.
     *
     * @param fp File pointer to write to (default `stdout`).
     */
    void print_params(std::FILE* fp = stdout) const {
        fmt::print(fp,
            "============================\n"
            "SIMULATION PARAMETERS:\n"
            "============================\n"
            "Max. steps        = {}\n"
            "Max. time         = {} sec\n"
            "Steps per cycle   = {}\n"
            "Cycles per check  = {}\n",
            params_.max_steps, params_.max_time, params_.steps_per_cycle, params_.cycles_per_check);
    }

    /**
     * @brief Print the simulation statistics to a file handle as a human-readable block.
     *
     * @param fp File pointer to write to (default `stdout`).
     */
    void print_stats(std::FILE* fp = stdout) const {
        fmt::print(fp,
            "============================\n"
            "SIMULATION STATISTICS:\n"
            "============================\n"
            "Steps done        = {}\n"
            "Last runtime      = {} sec\n"
            "Cumulative steps  = {}\n"
            "Cumulative time   = {} sec\n",
            stats_.steps_done, stats_.last_runtime, stats_.cumulative_steps, stats_.cumulative_time);
    }

    /**
     * @brief Serialize a simplemc::simulation_run.
     *
     * @details It serializes the simulation parameters and statistics. The timer is intentionally not
     * serialized since a resumed run begins with a fresh simulation_run::start().
     *
     * @tparam S Serializer type.
     * @param s Serializer handle.
     * @param r Simulation run to serialize.
     */
    template <serializer S>
    friend void simplemc_save(S& s, const simulation_run& r) {
        s.save_at("params", r.params_);
        s.save_at("stats", r.stats_);
    }

    /**
     * @brief Deserialize a simplemc::simulation_run.
     *
     * @details It deserializes the simulation parameters and statistics directly into the run.
     *
     * @tparam S Deserializer type.
     * @param s Deserializer handle.
     * @param r Simulation run to deserialize into.
     */
    template <deserializer S>
    friend void simplemc_load(const S& s, simulation_run& r) {
        s.load_at("params", r.params_);
        s.load_at("stats", r.stats_);
    }

private:
    simulation_params params_ {};
    simulation_stats stats_ {};
    timer<std::chrono::steady_clock> clock_ {};
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_SIMULATION_RUN_HPP
