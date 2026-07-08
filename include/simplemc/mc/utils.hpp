/**
 * @file
 * @brief Utilities for **simplemc-mc** library.
 */

#ifndef SIMPLEMC_MC_UTILS_HPP
#define SIMPLEMC_MC_UTILS_HPP

#include <simplemc/mc/concepts.hpp>
#include <simplemc/mc/measurement_set.hpp>
#include <simplemc/mc/simulation_params.hpp>
#include <simplemc/mc/simulation_stats.hpp>
#include <simplemc/mc/update_set.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/serialize/concepts.hpp>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-utils
 * @{
 */

/**
 * @brief Serialize the persistent run state of a MC simulation.
 *
 * @details Convenience function that serializes the random number generator, update set, measurement
 * set and simulation statistics in a single call.
 *
 * @tparam S Serializer type.
 * @tparam RNG Random number generator type.
 * @tparam Us User update types.
 * @tparam Ms User measurement types.
 * @param s Serializer handle.
 * @param rng Random number generator to serialize.
 * @param updates Update set to serialize.
 * @param meas Measurement set to serialize.
 * @param stats Simulation statistics to serialize.
 */
template <serializer S, typename RNG, mc_update... Us, mc_measurement... Ms>
void simplemc_save(S s, const RNG& rng, const update_set<Us...>& updates, const measurement_set<Ms...>& meas,
    const simulation_stats& stats) {
    s.save_at("rng", rng);
    s.save_at("stats", stats);
    s.save_at("updates", updates);
    s.save_at("measurements", meas);
}

/**
 * @brief Deserialize the persistent run state of a MC simulation.
 *
 * @details Convenience function that deserializes the random number generator, update set,
 * measurement set and simulation statistics in a single call.
 *
 * Update and measurement sets are only deserialized if the destination set is non-empty.
 *
 * @note Persistent-state loads are strict by design: every expected key must be present or a
 * simplemc::simplemc_exception is thrown. This is the opposite of the input-config channel
 * (simplemc_load_input_config), which tolerates missing keys via `try_load_at`.
 *
 * @tparam S Serializer type.
 * @tparam RNG Random number generator type.
 * @tparam Us User update types.
 * @tparam Ms User measurement types.
 * @param s Serializer handle.
 * @param rng Random number generator to deserialize into.
 * @param updates Update set to deserialize into.
 * @param meas Measurement set to deserialize into.
 * @param stats Simulation statistics to deserialize into.
 */
template <serializer S, typename RNG, mc_update... Us, mc_measurement... Ms>
void simplemc_load(
    const S& s, RNG& rng, update_set<Us...>& updates, measurement_set<Ms...>& meas, simulation_stats& stats) {
    s.load_at("rng", rng);
    s.load_at("stats", stats);
    if (!updates.empty()) {
        s.load_at("updates", updates);
    }
    if (!meas.empty()) {
        s.load_at("measurements", meas);
    }
}

/**
 * @brief Serialize the user-facing input config of a MC simulation.
 *
 * @details Convenience function that serializes the user-facing input config of the simulation
 * parameters, update set and measurement set in a single call.
 *
 * @tparam S Serializer type.
 * @tparam Us User update types.
 * @tparam Ms User measurement types.
 * @param s Serializer handle.
 * @param p Simulation parameters to serialize.
 * @param updates Update set to serialize.
 * @param meas Measurement set to serialize.
 */
template <serializer S, mc_update... Us, mc_measurement... Ms>
void simplemc_save_input_config(
    S s, const simulation_params& p, const update_set<Us...>& updates, const measurement_set<Ms...>& meas) {
    simplemc_save_input_config(s["params"], p);
    simplemc_save_input_config(s["updates"], updates);
    simplemc_save_input_config(s["measurements"], meas);
}

/**
 * @brief Deserialize the user-facing input config of a MC simulation.
 *
 * @details Convenience function that deserializes the user-facing input config of the simulation
 * parameters, update set and measurement set in a single call.
 *
 * @note Input-config loads are tolerant by design: missing keys leave the destination untouched
 * (`try_load_at`), so a partial config document is valid input. This is the opposite of the
 * persistent-state channel (simplemc_load), which is strict.
 *
 * @tparam S Serializer type.
 * @tparam Us User update types.
 * @tparam Ms User measurement types.
 * @param s Serializer handle.
 * @param p Simulation parameters to deserialize into.
 * @param updates Update set to deserialize into.
 * @param meas Measurement set to deserialize into.
 */
template <serializer S, mc_update... Us, mc_measurement... Ms>
void simplemc_load_input_config(
    const S& s, simulation_params& p, update_set<Us...>& updates, measurement_set<Ms...>& meas) {
    if (s.has("params")) {
        const auto params = s["params"];
        simplemc_load_input_config(params, p);
    }

    if (!updates.empty() && s.has("updates")) {
        const auto u = s["updates"];
        simplemc_load_input_config(u, updates);
    }

    if (!meas.empty() && s.has("measurements")) {
        const auto m = s["measurements"];
        simplemc_load_input_config(m, meas);
    }
}

/**
 * @brief Collect MC simulation components from different MPI processes.
 *
 * @details Convenience function that collects the update set, measurement set and simulation
 * statistics from different MPI processes in a single call.
 *
 * It simply dispatches to the corresponding ADL `%simplemc_mpi_collect` hooks of the individual
 * components.
 *
 * @note The reduction is **not idempotent**: all counters are summed across ranks, so call it exactly
 * once per collection point (a second call double-counts).
 *
 * @tparam Us User update types.
 * @tparam Ms User measurement types.
 * @param comm MPI communicator over which to reduce.
 * @param updates Update set to reduce in place.
 * @param meas Measurement set to reduce in place.
 * @param stats Simulation statistics to reduce in place.
 */
template <mc_update... Us, mc_measurement... Ms>
void simplemc_mpi_collect(
    const mpi::communicator& comm, update_set<Us...>& updates, measurement_set<Ms...>& meas, simulation_stats& stats) {
    simplemc_mpi_collect(comm, stats);
    simplemc_mpi_collect(comm, updates);
    simplemc_mpi_collect(comm, meas);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_UTILS_HPP
