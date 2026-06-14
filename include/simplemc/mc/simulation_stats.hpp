/**
 * @file
 * @brief Post-run, persistent MC simulation statistics.
 */

#ifndef SIMPLEMC_MC_SIMULATION_STATS_HPP
#define SIMPLEMC_MC_SIMULATION_STATS_HPP

#include <simplemc/mpi/all_reduce.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/serialize/concepts.hpp>

#include <fmt/format.h>

#include <cstdint>
#include <cstdio>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-sim
 * @{
 */

/**
 * @brief Post-run, persistent MC simulation statistics.
 *
 * @details It holds the persistent quantities that survive across runs: the totals of the most
 * recently finalized run and the cumulative totals over all runs.
 *
 * After a run completes, simplemc::accumulate_simulation_stats folds the last-run totals into the
 * cumulative ones and resets the last-run counters, while simplemc::reset_simulation_stats clears
 * only the last-run counters.
 *
 * The live state of a run still in progress (the running step counter and wall-clock) lives
 * separately in simplemc::simulation_ctx.
 *
 * Only the cumulative fields are serialized (see simplemc_save(S&, const simulation_stats&)). All
 * four fields are MPI-reduced in place (see simplemc_mpi_collect(const mpi::communicator&,
 * simulation_stats&)).
 */
struct simulation_stats {
    /**
     * @brief Number of Monte Carlo steps performed in the most recently finalized run.
     */
    std::uint64_t last_steps_done = 0;

    /**
     * @brief Runtime, in seconds, of the most recently finalized run.
     */
    double last_runtime = 0.0;

    /**
     * @brief Cumulative number of MC steps performed over multiple simulations.
     */
    std::uint64_t cumulative_steps = 0;

    /**
     * @brief Cumulative runtime, in seconds, over multiple simulations.
     */
    double cumulative_time = 0.0;
};

/** @} */

/**
 * @relates simplemc::simulation_stats
 * @brief Reset the current-run counters to zero.
 *
 * @details The cumulative fields are left untouched.
 *
 * @param s Simulation statistics to reset.
 */
inline void reset_simulation_stats(simulation_stats& s) noexcept {
    s.last_steps_done = 0;
    s.last_runtime = 0.0;
}

/**
 * @relates simplemc::simulation_stats
 * @brief Accumulate the current-run counters into the cumulative ones and reset the current ones.
 *
 * @details It calls simplemc::reset_simulation_stats after accumulating.
 *
 * @param s Simulation statistics to accumulate.
 */
inline void accumulate_simulation_stats(simulation_stats& s) noexcept {
    s.cumulative_steps += s.last_steps_done;
    s.cumulative_time += s.last_runtime;
    reset_simulation_stats(s);
}

/**
 * @relates simplemc::simulation_stats
 * @brief Print a simplemc::simulation_stats as a human-readable block.
 *
 * @param fp Destination file pointer.
 * @param s Statistics to print.
 */
inline void print(std::FILE* fp, const simulation_stats& s) {
    fmt::print(fp,
        "============================\n"
        "SIMULATION STATISTICS:\n"
        "============================\n"
        "Last steps done   = {}\n"
        "Last runtime      = {} sec\n"
        "Cumulative steps  = {}\n"
        "Cumulative time   = {} sec\n",
        s.last_steps_done, s.last_runtime, s.cumulative_steps, s.cumulative_time);
}

/**
 * @relates simplemc::simulation_stats
 * @brief Serialize the persistent fields of simplemc::simulation_stats.
 * 
 * @details Persistent fields include 
 * 
 * - simulation_stats::cumulative_steps and 
 * - simulation_stats::cumulative_time.
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param st Stats to serialize.
 */
template <serializer S>
void simplemc_save(S& s, const simulation_stats& st) {
    s.save_at("cumulative_steps", st.cumulative_steps);
    s.save_at("cumulative_time", st.cumulative_time);
}

/**
 * @relates simplemc::simulation_stats
 * @brief Deserialize the persistent fields of simplemc::simulation_stats.
 * 
 * @details See also simplemc::simplemc_save(S&, const simulation_stats&).
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param st Stats to deserialize into.
 */
template <serializer S>
void simplemc_load(const S& s, simulation_stats& st) {
    s.load_at("cumulative_steps", st.cumulative_steps);
    s.load_at("cumulative_time", st.cumulative_time);
}

/**
 * @relates simplemc::simulation_stats
 * @brief Collect simplemc::simulation_stats from different MPI processes.
 *
 * @details It uses mpi::all_reduce_in_place with `MPI_SUM` on all fields.
 *
 * @param comm MPI communicator over which to reduce.
 * @param st Stats to reduce in place.
 */
inline void simplemc_mpi_collect(const mpi::communicator& comm, simulation_stats& st) {
    mpi::all_reduce_in_place(st.last_steps_done, MPI_SUM, comm);
    mpi::all_reduce_in_place(st.last_runtime, MPI_SUM, comm);
    mpi::all_reduce_in_place(st.cumulative_steps, MPI_SUM, comm);
    mpi::all_reduce_in_place(st.cumulative_time, MPI_SUM, comm);
}

} // namespace simplemc

#endif // SIMPLEMC_MC_SIMULATION_STATS_HPP
