/**
 * @file
 * @brief MC simulation statistics.
 */

#ifndef SIMPLEMC_MC_SIMULATION_STATS_HPP
#define SIMPLEMC_MC_SIMULATION_STATS_HPP

#include <simplemc/mc/simulation_ctx.hpp>
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
 * @brief MC simulation statistics.
 *
 * @details It holds the total number of MC steps performed and the total runtime (in seconds).
 *
 * After a simplemc::run completes, the returned simplemc::simulation_ctx can be folded into the
 * simulation stats using simplemc::accumulate_simulation_stats. In that way, one can keep track of
 * the cumulative statistics over multiple runs.
 */
struct simulation_stats {
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
 * @brief Accumulate the context of a finished run into simulation statistics.
 *
 * @details The per-run totals are read from the simplemc::simulation_ctx returned by simplemc::run.
 *
 * @param s Simulation statistics to accumulate into.
 * @param ctx Context of the finished run.
 */
inline void accumulate_simulation_stats(simulation_stats& s, const simulation_ctx& ctx) noexcept {
    s.cumulative_steps += ctx.steps_done;
    s.cumulative_time += ctx.runtime;
}

/**
 * @relates simplemc::simulation_stats
 * @brief Print a simplemc::simulation_stats as a human-readable block.
 *
 * @param s Statistics to print.
 * @param fp Destination file pointer (defaults to stdout).
 */
inline void print(const simulation_stats& s, std::FILE* fp = stdout) {
    fmt::print(fp,
        "============================\n"
        "SIMULATION STATISTICS:\n"
        "============================\n"
        "MC steps done = {}\n"
        "Runtime       = {} sec\n",
        s.cumulative_steps, s.cumulative_time);
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
void simplemc_save(S s, const simulation_stats& st) {
    s.save_at("cumulative_steps", st.cumulative_steps);
    s.save_at("cumulative_time", st.cumulative_time);
}

/**
 * @relates simplemc::simulation_stats
 * @brief Deserialize the persistent fields of simplemc::simulation_stats.
 *
 * @details See also simplemc::simplemc_save(S, const simulation_stats&).
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
 * @details It uses mpi::all_reduce_in_place with `MPI_SUM` on both cumulative fields.
 *
 * @param comm MPI communicator over which to reduce.
 * @param st Stats to reduce in place.
 */
inline void simplemc_mpi_collect(const mpi::communicator& comm, simulation_stats& st) {
    mpi::all_reduce_in_place(st.cumulative_steps, MPI_SUM, comm);
    mpi::all_reduce_in_place(st.cumulative_time, MPI_SUM, comm);
}

} // namespace simplemc

#endif // SIMPLEMC_MC_SIMULATION_STATS_HPP
