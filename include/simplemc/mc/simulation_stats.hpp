/**
 * @file
 * @brief Simulation statistics gathered during a Monte Carlo run.
 */

#ifndef SIMPLEMC_MC_SIMULATION_STATS_HPP
#define SIMPLEMC_MC_SIMULATION_STATS_HPP


#include <fmt/format.h>

#include <cstdint>
#include <cstdio>

namespace simplemc {

/**
 * @addtogroup simplemc-mc
 * @{
 */

/**
 * @brief Plain aggregate of simulation statistics gathered during a Monte Carlo run.
 */
struct simulation_stats {
    /// Number of Monte Carlo steps performed in the current run.
    std::uint64_t steps_done = 0;

    /// Runtime, in seconds, of the most recently finalized simulation.
    double last_runtime = 0.0;

    /// Cumulative number of MC steps performed over multiple simulations.
    std::uint64_t cumulative_steps = 0;

    /// Cumulative runtime, in seconds, over multiple simulations.
    double cumulative_time = 0.0;
};

/**
 * @brief Reset the current-run counters to zero.
 *
 * @details The cumulative fields are left untouched.
 *
 * @param s Simulation statistics to reset.
 */
inline void reset_simulation_stats(simulation_stats& s) noexcept {
    s.steps_done = 0;
    s.last_runtime = 0.0;
}

/**
 * @brief Accumulate the current-run counters into the cumulative ones and reset the current ones.
 *
 * @details It calls simplemc::reset_simulation_stats after accumulating.
 *
 * @param s Simulation statistics to accumulate.
 */
inline void accumulate_simulation_stats(simulation_stats& s) noexcept {
    s.cumulative_steps += s.steps_done;
    s.cumulative_time += s.last_runtime;
    reset_simulation_stats(s);
}

/**
 * @brief Print a simplemc::simulation_stats as a human-readable block.
 *
 * @param fp Destination file handle (default `stdout`).
 * @param s Statistics to print.
 */
inline void print(std::FILE* fp, const simulation_stats& s) {
    fmt::print(fp,
        "============================\n"
        "SIMULATION STATISTICS:\n"
        "============================\n"
        "Steps done        = {}\n"
        "Last runtime      = {} sec\n"
        "Cumulative steps  = {}\n"
        "Cumulative time   = {} sec\n",
        s.steps_done, s.last_runtime, s.cumulative_steps, s.cumulative_time);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_SIMULATION_STATS_HPP
