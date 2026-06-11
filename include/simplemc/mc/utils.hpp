/**
 * @file
 * @brief Free composite save / load / MPI-collect helpers over the simplemc-mc run components.
 */

#ifndef SIMPLEMC_MC_UTILS_HPP
#define SIMPLEMC_MC_UTILS_HPP

#include <simplemc/mc/concepts.hpp>
#include <simplemc/mc/measurement_set.hpp>
#include <simplemc/mc/simulation_params.hpp>
#include <simplemc/mc/simulation_stats.hpp>
#include <simplemc/mc/traits.hpp>
#include <simplemc/mc/update_set.hpp>
#include <simplemc/mpi/communicator.hpp>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-sim
 * @{
 */

/**
 * @brief Serialize the persistent run state held across the four run components.
 *
 * @details Writes the RNG state under `"rng"`, the cumulative simulation counters at the current
 * location, and the registered updates / measurements under `"updates"` / `"measurements"`. Per-run
 * state (current-run counters, last step count / runtime) and the per-call simplemc::simulation_params
 * are intentionally not part of a checkpoint. Mirrors the per-entity schema produced by the
 * component-level `simplemc_save` overloads.
 *
 * @tparam Traits Traits bundle satisfying simplemc::mc_traits_like.
 * @tparam RNG Random number generator type.
 * @param s Checkpoint serializer handle.
 * @param rng Random number generator to persist.
 * @param updates Update set to persist.
 * @param meas Measurement set to persist.
 * @param stats Cumulative simulation statistics to persist.
 */
template <mc_traits_like Traits, class RNG>
void simplemc_save(typename Traits::checkpoint_serializer_type& s, const RNG& rng,
    const update_set<Traits>& updates, const measurement_set<Traits>& meas, const simulation_stats& stats) {
    s.save_at("rng", rng);
    simplemc_save(s, stats);

    auto u = s["updates"];
    simplemc_save(u, updates);

    auto m = s["measurements"];
    simplemc_save(m, meas);
}

/**
 * @brief Restore the persistent run state into the four run components.
 *
 * @details The destination update / measurement sets must already have the same entries (matching
 * names) registered. An entry present in a non-empty set but missing from the checkpoint throws
 * simplemc::simplemc_exception; empty sets are skipped. Per-run state and the internal selection
 * distribution are not touched.
 *
 * @tparam Traits Traits bundle satisfying simplemc::mc_traits_like.
 * @tparam RNG Random number generator type.
 * @param s Const checkpoint serializer handle.
 * @param rng Random number generator to restore.
 * @param updates Update set to patch in place.
 * @param meas Measurement set to patch in place.
 * @param stats Cumulative simulation statistics to restore.
 */
template <mc_traits_like Traits, class RNG>
void simplemc_load(const typename Traits::checkpoint_serializer_type& s, RNG& rng, update_set<Traits>& updates,
    measurement_set<Traits>& meas, simulation_stats& stats) {
    s.load_at("rng", rng);
    simplemc_load(s, stats);

    if (!updates.empty()) {
        const auto u = s["updates"];
        simplemc_load(u, updates);
    }

    if (!meas.empty()) {
        const auto m = s["measurements"];
        simplemc_load(m, meas);
    }
}

/**
 * @brief Serialize the user-facing input config of the run components.
 *
 * @details Writes the simplemc::simulation_params under `"params"`, then per-update `weight` and
 * per-measurement `is_active` (plus any opt-in user input-config) under `"updates"` /
 * `"measurements"`.
 *
 * @tparam Traits Traits bundle satisfying simplemc::mc_traits_like.
 * @param s Input-config serializer handle.
 * @param p Simulation parameters to persist.
 * @param updates Update set whose input config to persist.
 * @param meas Measurement set whose input config to persist.
 */
template <mc_traits_like Traits>
void simplemc_save_input_config(typename Traits::input_config_serializer_type& s, const simulation_params& p,
    const update_set<Traits>& updates, const measurement_set<Traits>& meas) {
    auto params = s["params"];
    simplemc_save_input_config(params, p);

    auto u = s["updates"];
    simplemc_save_input_config(u, updates);

    auto m = s["measurements"];
    simplemc_save_input_config(m, meas);
}

/**
 * @brief Restore the user-facing input config into the run components.
 *
 * @details The destination update / measurement sets must already have the same entries (matching
 * names) registered. Parameters are read leniently (missing fields keep their defaults); a non-empty
 * set with an entry missing from the input config throws simplemc::simplemc_exception, mirroring
 * simplemc_load.
 *
 * @tparam Traits Traits bundle satisfying simplemc::mc_traits_like.
 * @param s Const input-config serializer handle.
 * @param p Simulation parameters to read into.
 * @param updates Update set to patch in place.
 * @param meas Measurement set to patch in place.
 */
template <mc_traits_like Traits>
void simplemc_load_input_config(const typename Traits::input_config_serializer_type& s, simulation_params& p,
    update_set<Traits>& updates, measurement_set<Traits>& meas) {
    if (s.has("params")) {
        const auto params = s["params"];
        simplemc_load_input_config(params, p);
    }

    if (!updates.empty()) {
        const auto u = s["updates"];
        simplemc_load_input_config(u, updates);
    }

    if (!meas.empty()) {
        const auto m = s["measurements"];
        simplemc_load_input_config(m, meas);
    }
}

/**
 * @brief All-reduce every reducible run component across MPI ranks.
 *
 * @details Composite reducer that forwards to simplemc_mpi_collect on the simplemc::simulation_stats,
 * simplemc::update_set, and simplemc::measurement_set in turn. The RNG and the internal selection
 * distribution are intentionally not touched. Reduction is in-place.
 *
 * @tparam Traits Traits bundle satisfying simplemc::mc_traits_like.
 * @param comm MPI communicator over which to reduce.
 * @param updates Update set to reduce in place.
 * @param meas Measurement set to reduce in place.
 * @param stats Simulation statistics to reduce in place.
 */
template <mc_traits_like Traits>
void simplemc_mpi_collect(const mpi::communicator& comm, update_set<Traits>& updates,
    measurement_set<Traits>& meas, simulation_stats& stats) {
    simplemc_mpi_collect(comm, stats);
    simplemc_mpi_collect(comm, updates);
    simplemc_mpi_collect(comm, meas);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_UTILS_HPP
