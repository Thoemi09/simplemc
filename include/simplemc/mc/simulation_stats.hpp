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
 * A simplemc::simulation_ctx acts as the per-run delta to this cumulative record. During and after a
 * run, the context should be folded into the statistics using `operator+=()` or `operator+()`.
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
 * @brief Addition assignment operator to fold a simulation context into the cumulative simulation
 * statistics.
 *
 * @details This updates the statistics in place. Use it at the end of a simulation to keep track of
 * the total number of MC steps and the total runtime across multiple runs.
 *
 * @param s Simulation statistics to fold into.
 * @param ctx Simulation context.
 * @return Reference to simulation statistics after the fold.
 */
constexpr simulation_stats& operator+=(simulation_stats& s, const simulation_ctx& ctx) noexcept {
    s.cumulative_steps += ctx.steps_done;
    s.cumulative_time += ctx.runtime;
    return s;
}

/**
 * @relates simplemc::simulation_stats
 * @brief Addition operator to combine cumulative simulation statistics with a simulation context.
 *
 * @details This adds the MC steps and runtimes contained in the context and the statistics and
 * returns them in a new simplemc::simulation_stats object.
 *
 * @param s Simulation statistics.
 * @param ctx Simulation context.
 * @return Summed statistics.
 */
[[nodiscard]] constexpr simulation_stats operator+(simulation_stats s, const simulation_ctx& ctx) noexcept {
    return s += ctx;
}

/**
 * @relates simplemc::simulation_stats
 * @brief Addition operator to combine a simulation context with cumulative simulation statistics.
 *
 * @details Same as simplemc::operator+(simulation_stats, const simulation_ctx&).
 *
 * @param ctx Simulation context.
 * @param s Simulation statistics.
 * @return Summed statistics.
 */
[[nodiscard]] constexpr simulation_stats operator+(const simulation_ctx& ctx, simulation_stats s) noexcept {
    return s += ctx;
}

/**
 * @relates simplemc::simulation_stats
 * @brief Print a simplemc::simulation_stats as a human-readable block.
 *
 * @details It also reports the derived throughput (steps per second).
 *
 * @param s Statistics to print.
 * @param fp Destination file pointer (defaults to stdout).
 */
inline void print(const simulation_stats& s, std::FILE* fp = stdout) {
    const std::string steps_per_sec = s.cumulative_time > 0.0 ?
        fmt::format("{:.6}", static_cast<double>(s.cumulative_steps) / s.cumulative_time) :
        "--";
    fmt::print(fp,
        "Runtime        = {} sec\n"
        "MC steps done  = {}\n"
        "Steps per sec  = {}\n",
        s.cumulative_time, s.cumulative_steps, steps_per_sec);
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
