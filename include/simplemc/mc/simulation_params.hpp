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
 * @addtogroup simplemc-mc
 * @{
 */

/**
 * @brief Plain aggregate of user-set parameters that control a Monte Carlo simulation run.
 */
struct simulation_params {
    /**
     * @brief Stop the run once this many Monte Carlo steps have been performed. Default: maximum.
     */
    std::uint64_t max_steps = std::numeric_limits<std::uint64_t>::max();

    /**
     * @brief Stop the run once this much wall-clock time has elapsed. Default: \f$ 10 \f$ seconds.
     */
    double max_time = 10.0;

    /**
     * @brief Number of update attempts between measurement opportunities. Default: \f$ 5 \f$.
     */
    std::uint64_t steps_per_cycle = 5;

    /**
     * @brief Number of measurement cycles between stop-criteria polls. Default: \f$ 10^6 \f$.
     */
    std::uint64_t cycles_per_check = 1'000'000;
};

/**
 * @brief Validate a simplemc::simulation_params.
 *
 * @details It throws simplemc::simplemc_exception when any field is out of range:
 * - `max_time < 0 `,
 * - `max_steps == 0`,
 * - `steps_per_cycle == 0`, or
 * - `cycles_per_check == 0`.
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
}

/**
 * @brief Print a simplemc::simulation_params as a human-readable block.
 *
 * @param fp Destination file handle.
 * @param p Parameters to print.
 */
inline void print(std::FILE* fp, const simulation_params& p) {
    fmt::print(fp,
        "============================\n"
        "SIMULATION PARAMETERS:\n"
        "============================\n"
        "Max. steps        = {}\n"
        "Max. time         = {} sec\n"
        "Steps per cycle   = {}\n"
        "Cycles per check  = {}\n",
        p.max_steps, p.max_time, p.steps_per_cycle, p.cycles_per_check);
}

/**
 * @brief Serialize the persistent fields of simplemc::simulation_params.
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param p Parameters to write.
 */
template <serializer S>
void simplemc_save(S& s, const simulation_params& p) {
    s.save_at("steps_per_cycle", p.steps_per_cycle);
    s.save_at("cycles_per_check", p.cycles_per_check);
}

/**
 * @brief Deserialize the persistent fields of simplemc::simulation_params.
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param p Parameters to read into.
 */
template <serializer S>
void simplemc_load(const S& s, simulation_params& p) {
    s.load_at("steps_per_cycle", p.steps_per_cycle);
    s.load_at("cycles_per_check", p.cycles_per_check);
}

/**
 * @brief Serialize the user-input config of simplemc::simulation_params.
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param p Parameters to write.
 */
template <serializer S>
void simplemc_save_input_config(S& s, const simulation_params& p) {
    s.save_at("max_steps", p.max_steps);
    s.save_at("max_time", p.max_time);
    s.save_at("steps_per_cycle", p.steps_per_cycle);
    s.save_at("cycles_per_check", p.cycles_per_check);
}

/**
 * @brief Deserialize the user-input config of a simplemc::simulation_params.
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param p Parameters to read into.
 */
template <serializer S>
void simplemc_load_input_config(const S& s, simulation_params& p) {
    s.try_load_at("max_steps", p.max_steps);
    s.try_load_at("max_time", p.max_time);
    s.try_load_at("steps_per_cycle", p.steps_per_cycle);
    s.try_load_at("cycles_per_check", p.cycles_per_check);
    validate_simulation_params(p);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_SIMULATION_PARAMS_HPP
