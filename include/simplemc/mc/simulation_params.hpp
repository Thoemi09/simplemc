/**
 * @file
 * @brief User-set parameters that control a Monte Carlo simulation run.
 */

#ifndef SIMPLEMC_MC_SIMULATION_PARAMS_HPP
#define SIMPLEMC_MC_SIMULATION_PARAMS_HPP

#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <cstdint>
#include <cstdio>
#include <limits>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-sim
 * @{
 */

/**
 * @brief User-set parameters that control a Monte Carlo simulation run.
 *
 * @details Parameters are validated by calling simplemc::validate_simulation_params.
 */
struct simulation_params {
    /**
     * @brief Stop the run once this many MC steps have been performed.
     */
    std::uint64_t max_steps = std::numeric_limits<std::uint64_t>::max();

    /**
     * @brief Stop the run once this much wall-clock time has elapsed (in seconds).
     */
    double max_time = 10.0;

    /**
     * @brief Number of MC steps between measurements.
     */
    std::uint64_t steps_per_cycle = 5;

    /**
     * @brief Number of MC cycles between checkpoint and stop-criteria polls.
     */
    std::uint64_t cycles_per_check = 1'000'000;

    /**
     * @brief Whether to deactivate all measurements for the whole run.
     */
    bool skip_measurements = false;

    /**
     * @brief Trigger the `on_checkpoint` callback once this many MC steps have elapsed since the
     * last checkpoint.
     */
    std::uint64_t checkpoint_after_steps = std::numeric_limits<std::uint64_t>::max();

    /**
     * @brief Trigger the `on_checkpoint` callback once this much wall-clock time has elapsed since
     * the last checkpoint.
     */
    double checkpoint_after_time = std::numeric_limits<double>::max();
};

/** @} */

/**
 * @relates simplemc::simulation_params
 * @brief Validate a simplemc::simulation_params.
 *
 * @details It throws simplemc::simplemc_exception when any field is out of range:
 *
 * - `max_time < 0`,
 * - `max_steps == 0`,
 * - `steps_per_cycle == 0`,
 * - `cycles_per_check == 0`, or
 * - `checkpoint_after_time < 0`.
 *
 * @param p Parameters to validate.
 */
inline void validate_simulation_params(const simulation_params& p) {
    if (p.max_time < 0) {
        throw simplemc_exception(fmt::format("max_time must be >= 0 (got {})", p.max_time));
    }
    if (p.max_steps == 0) {
        throw simplemc_exception("max_steps must be > 0");
    }
    if (p.steps_per_cycle == 0) {
        throw simplemc_exception("steps_per_cycle must be > 0");
    }
    if (p.cycles_per_check == 0) {
        throw simplemc_exception("cycles_per_check must be > 0");
    }
    if (p.checkpoint_after_time < 0) {
        throw simplemc_exception(fmt::format("checkpoint_after_time must be >= 0 (got {})", p.checkpoint_after_time));
    }
}

/**
 * @relates simplemc::simulation_params
 * @brief Print a simplemc::simulation_params as a human-readable block.
 *
 * @param p Parameters to print.
 * @param fp Destination file pointer (defaults to stdout).
 */
inline void print(const simulation_params& p, std::FILE* fp = stdout) {
    fmt::print(fp,
        "============================\n"
        "SIMULATION PARAMETERS:\n"
        "============================\n"
        "Max. steps              = {}\n"
        "Max. time               = {} sec\n"
        "Steps per cycle         = {}\n"
        "Cycles per check        = {}\n"
        "Skip measurements       = {}\n"
        "Checkpoint after steps  = {}\n"
        "Checkpoint after time   = {} sec\n",
        p.max_steps, p.max_time, p.steps_per_cycle, p.cycles_per_check, p.skip_measurements, p.checkpoint_after_steps,
        p.checkpoint_after_time);
}

/**
 * @relates simplemc::simulation_params
 * @brief Serialize the persistent fields of simplemc::simulation_params.
 *
 * @details Persistent fields include
 *
 * - simulation_params::steps_per_cycle and
 * - simulation_params::cycles_per_check.
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param p Parameters to serialize.
 */
template <serializer S>
void simplemc_save(S s, const simulation_params& p) {
    s.save_at("steps_per_cycle", p.steps_per_cycle);
    s.save_at("cycles_per_check", p.cycles_per_check);
}

/**
 * @relates simplemc::simulation_params
 * @brief Deserialize the persistent fields of simplemc::simulation_params.
 *
 * @details See also simplemc_save(S, const simulation_params&).
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param p Parameters to deserialize into.
 */
template <serializer S>
void simplemc_load(const S& s, simulation_params& p) {
    s.load_at("steps_per_cycle", p.steps_per_cycle);
    s.load_at("cycles_per_check", p.cycles_per_check);
}

/**
 * @relates simplemc::simulation_params
 * @brief Serialize the user-input config of simplemc::simulation_params.
 *
 * @details All fields are part of the user-input conifg.
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param p Parameters to serialize.
 */
template <serializer S>
void simplemc_save_input_config(S s, const simulation_params& p) {
    s.save_at("max_steps", p.max_steps);
    s.save_at("max_time", p.max_time);
    s.save_at("steps_per_cycle", p.steps_per_cycle);
    s.save_at("cycles_per_check", p.cycles_per_check);
    s.save_at("skip_measurements", p.skip_measurements);
    s.save_at("checkpoint_after_steps", p.checkpoint_after_steps);
    s.save_at("checkpoint_after_time", p.checkpoint_after_time);
}

/**
 * @relates simplemc::simulation_params
 * @brief Deserialize the user-input config of a simplemc::simulation_params.
 *
 * @details See also simplemc_save_input_config(S, const simulation_params&).
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param p Parameters to deserialize into.
 */
template <serializer S>
void simplemc_load_input_config(const S& s, simulation_params& p) {
    s.try_load_at("max_steps", p.max_steps);
    s.try_load_at("max_time", p.max_time);
    s.try_load_at("steps_per_cycle", p.steps_per_cycle);
    s.try_load_at("cycles_per_check", p.cycles_per_check);
    s.try_load_at("skip_measurements", p.skip_measurements);
    s.try_load_at("checkpoint_after_steps", p.checkpoint_after_steps);
    s.try_load_at("checkpoint_after_time", p.checkpoint_after_time);
    validate_simulation_params(p);
}

} // namespace simplemc

#endif // SIMPLEMC_MC_SIMULATION_PARAMS_HPP
