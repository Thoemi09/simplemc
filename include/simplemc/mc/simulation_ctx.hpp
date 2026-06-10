/**
 * @file
 * @brief Live, transient state of a single Monte Carlo run.
 */

#ifndef SIMPLEMC_MC_SIMULATION_CTX_HPP
#define SIMPLEMC_MC_SIMULATION_CTX_HPP

#include <simplemc/utils/timer.hpp>

#include <cstdint>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-sim
 * @{
 */

/**
 * @brief Live state of the run currently in progress.
 *
 * @details Holds the transient quantities that evolve while simplemc::run is executing: the live
 * step counter and a wall-clock timer. It is created fresh inside simplemc::run for each run and
 * handed to the run-loop callbacks (see simplemc::mc_run_callbacks), which read the live step count
 * via `steps_done` and the live elapsed seconds via elapsed(). The recorded, post-run statistics
 * (last-run totals and cumulative totals) live separately in simplemc::simulation_stats.
 *
 * This type is intentionally not serialized or MPI-reduced — it is per-run scratch state.
 */
struct simulation_ctx {
    /**
     * @brief Number of Monte Carlo steps performed so far in the current run.
     */
    std::uint64_t steps_done = 0;

    /**
     * @brief Wall-clock timer for the current run; started by simplemc::run before the loop.
     */
    timer<> clk {};

    /**
     * @brief Live elapsed seconds since the current run's timer was started.
     *
     * @return Wall-clock seconds since `clk` was last started.
     */
    [[nodiscard]] double elapsed() const { return clk.elapsed(); }
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_SIMULATION_CTX_HPP
