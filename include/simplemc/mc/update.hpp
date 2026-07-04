/**
 * @file
 * @brief MC update value together with some metadata.
 */

#ifndef SIMPLEMC_MC_UPDATE_HPP
#define SIMPLEMC_MC_UPDATE_HPP

#include <simplemc/mc/concepts.hpp>
#include <simplemc/mpi/all_reduce.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <cstdint>
#include <string>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-entities-updates
 * @{
 */

/**
 * @brief MC update value together with some metadata.
 *
 * @details simplemc::update owns a user update value of type `U` (satisfying simplemc::mc_update)
 * together with its metadata:
 *
 * - a unique name that identifies the update,
 * - the name of the inverse update,
 * - an unnormalized selection weight,
 * - a detailed-balance correction ratio, and
 * - counters tracking the number of proposals, acceptances, and impossible signals over the current
 * run and across multiple runs.
 *
 * All fields are public so kernels and reporting code can read and write them directly. The wrapped
 * type is stored by value; there is no type erasure, so the concrete type is available at compile
 * time (via the public @ref value member and the nested alias @ref value_type).
 *
 * @tparam U User update type satisfying simplemc::mc_update.
 */
template <mc_update U>
struct update {
    /**
     * @brief The concrete wrapped user type.
     */
    using value_type = U;

    /**
     * @brief The wrapped user update.
     */
    U value;

    /**
     * @brief Identifier used in lookups and printed reports.
     */
    std::string name;

    /**
     * @brief Identifier of the inverse update. Equal to `name` for self-inverse updates.
     */
    std::string inv_name;

    /**
     * @brief Unnormalized selection weight (\f$ w \geq 0 \f$).
     */
    double weight;

    /**
     * @brief Detailed-balance correction multiplier applied by the kernel to `attempt()`.
     *
     * @details \f$ 1 \f$ for a self-inverse update; \f$ w_{\text{inv}} / w \f$ for a paired update.
     */
    double ratio = 1.0;

    /**
     * @brief Number of times this update has been proposed in the current run.
     */
    std::uint64_t nprops = 0;

    /**
     * @brief Number of times this update has been accepted in the current run.
     */
    std::uint64_t naccs = 0;

    /**
     * @brief Number of times this update has been signaled as impossible in the current run.
     */
    std::uint64_t nimps = 0;

    /**
     * @brief Cumulative number of proposals across runs.
     */
    std::uint64_t cumulative_nprops = 0;

    /**
     * @brief Cumulative number of acceptances across runs.
     */
    std::uint64_t cumulative_naccs = 0;

    /**
     * @brief Cumulative number of impossible signals across runs.
     */
    std::uint64_t cumulative_nimps = 0;

    /**
     * @brief Constructor stores a user-defined update value.
     *
     * @details It validates that the name is not empty and that the weight is non-negative, throwing a
     * simplemc::simplemc_exception otherwise. It sets `inv_name = name` and `ratio = 1.0`; the counters
     * default to zero.
     *
     * The first parameter is the class template parameter by value, so the implicitly-generated
     * deduction guide lets `update{ my_update {}, "name", 1.0 }` deduce `update<my_update>`.
     *
     * @param value User update value to store.
     * @param name Identifier.
     * @param weight Unnormalized selection weight.
     */
    update(U value, std::string name, double weight) :
        value { std::move(value) },
        name { std::move(name) },
        inv_name { this->name },
        weight { weight } {
        if (this->name.empty()) {
            throw simplemc_exception("update name must be non-empty");
        }
        if (this->weight < 0.0) {
            throw simplemc_exception(
                fmt::format("update weight must be >= 0 (got {} on '{}')", this->weight, this->name));
        }
    }

    /**
     * @brief Propose a change to the simulation state.
     *
     * @return Acceptance probability of the proposed change.
     */
    double attempt() { return value.attempt(); }

    /**
     * @brief Commit the proposed change.
     */
    void accept() { value.accept(); }

    /**
     * @brief Roll back any speculative state set up by attempt().
     *
     * @details If the wrapped user type defines a `%reject()` member it is invoked. Otherwise this is a
     * no-op.
     */
    void reject() {
        if constexpr (requires { value.reject(); }) {
            value.reject();
        }
    }

    /**
     * @brief Zero the current-run counters but leave the cumulative counters untouched.
     */
    void reset_run_counters() noexcept {
        nprops = 0;
        naccs = 0;
        nimps = 0;
    }

    /**
     * @brief Add the current-run counters to the cumulative counters and call reset_run_counters().
     */
    void accumulate_counters() noexcept {
        cumulative_nprops += nprops;
        cumulative_naccs += naccs;
        cumulative_nimps += nimps;
        reset_run_counters();
    }
};

/**
 * @relates simplemc::update
 * @brief Serialize a simplemc::update.
 *
 * @details It serializes all metadata except `name` and the transient current-run counters, then the
 * wrapped value under `"user"` if the value is serializable by `S` (otherwise the value is
 * skipped).
 *
 * @note The value skip is silent: a missing (or misspelled) ADL `%simplemc_save` overload on the
 * user type does not produce a diagnostic — the `"user"` key is simply absent from the output.
 *
 * @tparam S Serializer type.
 * @tparam U User update type.
 * @param s Serializer handle.
 * @param u Update to serialize.
 */
template <serializer S, mc_update U>
void simplemc_save(S& s, const update<U>& u) {
    s.save_at("inv_name", u.inv_name);
    s.save_at("weight", u.weight);
    s.save_at("ratio", u.ratio);
    s.save_at("cumulative_nprops", u.cumulative_nprops);
    s.save_at("cumulative_naccs", u.cumulative_naccs);
    s.save_at("cumulative_nimps", u.cumulative_nimps);
    if constexpr (save_at_all<U, S>) {
        s.save_at("user", u.value);
    }
}

/**
 * @relates simplemc::update
 * @brief Deserialize a simplemc::update.
 *
 * @details Symmetric to simplemc_save(S&, const update<U>&).
 *
 * @note If the value is not deserializable by `S` it is silently skipped and keeps its current
 * value; a missing (or misspelled) ADL `%simplemc_load` overload does not produce a diagnostic.
 *
 * @tparam S Serializer type.
 * @tparam U User update type.
 * @param s Serializer handle.
 * @param u Update to deserialize into.
 */
template <serializer S, mc_update U>
void simplemc_load(const S& s, update<U>& u) {
    s.load_at("inv_name", u.inv_name);
    s.load_at("weight", u.weight);
    s.load_at("ratio", u.ratio);
    s.load_at("cumulative_nprops", u.cumulative_nprops);
    s.load_at("cumulative_naccs", u.cumulative_naccs);
    s.load_at("cumulative_nimps", u.cumulative_nimps);
    if constexpr (load_at_all<U, S>) {
        s.load_at("user", u.value);
    }
}

/**
 * @relates simplemc::update
 * @brief Serialize the user-input config of a simplemc::update.
 *
 * @details It serializes `weight` and, if the value has an input-config serialization, the value
 * under `"user"`.
 *
 * @tparam S Serializer type.
 * @tparam U User update type.
 * @param s Serializer handle.
 * @param u Update to serialize.
 */
template <serializer S, mc_update U>
void simplemc_save_input_config(S& s, const update<U>& u) {
    s.save_at("weight", u.weight);
    if constexpr (has_simplemc_save_input_config<U, S>) {
        auto sub = s["user"];
        simplemc_save_input_config(sub, u.value);
    }
}

/**
 * @relates simplemc::update
 * @brief Deserialize the user-input config of a simplemc::update.
 *
 * @details Symmetric to simplemc_save_input_config(S&, const update<U>&). The `weight` is optional.
 *
 * @tparam S Serializer type.
 * @tparam U User update type.
 * @param s Serializer handle.
 * @param u Update to deserialize into.
 */
template <serializer S, mc_update U>
void simplemc_load_input_config(const S& s, update<U>& u) {
    s.try_load_at("weight", u.weight);
    if constexpr (has_simplemc_load_input_config<U, S>) {
        const auto sub = s["user"];
        simplemc_load_input_config(sub, u.value);
    }
}

/**
 * @relates simplemc::update
 * @brief Collect a simplemc::update from different MPI processes.
 *
 * @details It all-reduces the six counter fields and, if the value supports it, reduces the value
 * via its own `%simplemc_mpi_collect`.
 *
 * @note This reduction is **not idempotent**: both the current-run and the cumulative counters are
 * summed, so call it exactly once per run, at a fixed point relative to update::accumulate_counters()
 * (a second call double-counts).
 *
 * @tparam U User update type.
 * @param comm simplemc::mpi::communicator object.
 * @param u Update to reduce in place.
 */
template <mc_update U>
void simplemc_mpi_collect(const mpi::communicator& comm, update<U>& u) {
    mpi::all_reduce_in_place(u.nprops, MPI_SUM, comm);
    mpi::all_reduce_in_place(u.naccs, MPI_SUM, comm);
    mpi::all_reduce_in_place(u.nimps, MPI_SUM, comm);
    mpi::all_reduce_in_place(u.cumulative_nprops, MPI_SUM, comm);
    mpi::all_reduce_in_place(u.cumulative_naccs, MPI_SUM, comm);
    mpi::all_reduce_in_place(u.cumulative_nimps, MPI_SUM, comm);
    if constexpr (has_simplemc_mpi_collect<U>) {
        u.value = simplemc_mpi_collect(comm, u.value);
    }
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_UPDATE_HPP
