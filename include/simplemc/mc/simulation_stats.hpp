/**
 * @file
 * @brief Simulation statistics gathered during a Monte Carlo run.
 */

#ifndef SIMPLEMC_MC_SIMULATION_STATS_HPP
#define SIMPLEMC_MC_SIMULATION_STATS_HPP

#include <simplemc/serialize/concepts.hpp>

#include <cstdint>

namespace simplemc {

/**
 * @addtogroup simplemc-mc
 * @{
 */

/**
 * @brief Simulation statistics gathered during a Monte Carlo run.
 *
 * @details It is held by simplemc::simulation_run, which mutates these fields to track the progress
 * and statistics of active and completed simulations.
 *
 * All fields are public for ease of aggregate-initialisation and direct mutation by the driver.
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
 * @brief Serialize a simplemc::simulation_stats.
 *
 * @details It serializes all fields of the struct.
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param st Stats to serialize.
 */
template <serializer S>
void simplemc_save(S& s, const simulation_stats& st) {
    s.save_at("steps_done", st.steps_done);
    s.save_at("last_runtime", st.last_runtime);
    s.save_at("cumulative_steps", st.cumulative_steps);
    s.save_at("cumulative_time", st.cumulative_time);
}

/**
 * @brief Deserialize a simplemc::simulation_stats.
 *
 * @details It deserializes all fields of the struct.
 *
 * @tparam S Deserializer type.
 * @param s Deserializer handle.
 * @param st Stats to deserialize into.
 */
template <deserializer S>
void simplemc_load(const S& s, simulation_stats& st) {
    s.load_at("steps_done", st.steps_done);
    s.load_at("last_runtime", st.last_runtime);
    s.load_at("cumulative_steps", st.cumulative_steps);
    s.load_at("cumulative_time", st.cumulative_time);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_SIMULATION_STATS_HPP
