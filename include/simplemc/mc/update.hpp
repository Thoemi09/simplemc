/**
 * @file
 * @brief MC update wrapper class.
 */

#ifndef SIMPLEMC_MC_UPDATE_HPP
#define SIMPLEMC_MC_UPDATE_HPP

#include <simplemc/mc/concepts.hpp>
#include <simplemc/mc/update_stats.hpp>
#include <simplemc/mpi/all_reduce.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <string>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-entities-updates
 * @{
 */

// Forward declarations.
template <mc_update U>
class update;

template <serializer S, mc_update U>
void simplemc_load(const S& s, update<U>& u);

template <mc_update U>
void simplemc_mpi_collect(const mpi::communicator& comm, update<U>& u);

/**
 * @brief MC update wrapper class for a user-defined update type.
 *
 * @details It owns a user-defined update of type @ref value_type (satisfying simplemc::mc_update)
 * together with its metadata, bundled in a simplemc::update_stats.
 *
 * The counters in the update statistics are maintained by the wrapper itself: attempt(), accept() and
 * mark_impossible() do the bookkeeping, so any kernel driving the wrapper gets correct statistics for
 * free.
 *
 * Some metadata can be set at runtime through the setters set_inv_name(), set_weight() and
 * set_ratio().
 *
 * The wrapped type is stored by value and can be accessed via value().
 *
 * @tparam U User update type satisfying simplemc::mc_update.
 */
template <mc_update U>
class update {
public:
    /**
     * @brief The concrete wrapped user type.
     */
    using value_type = U;

    /**
     * @brief Constructor stores a user-defined update value.
     *
     * @details It validates that the name is not empty and that the weight is non-negative, throwing
     * a simplemc::simplemc_exception otherwise.
     *
     * By default, updates are self-inverse, i.e. update_stats::inv_name is set to update_stats::name
     * and update_stats::ratio is set to \f$ 1.0 \f$. To define a paired update, see
     * update_set::link_pair().
     *
     * @param value User update value to store.
     * @param name Identifier.
     * @param weight Unnormalized selection weight.
     */
    update(U value, std::string name, double weight) : value_ { std::move(value) } {
        if (name.empty()) {
            throw simplemc_exception("update name must be non-empty");
        }
        stats_.name = std::move(name);
        stats_.inv_name = stats_.name;
        set_weight(weight);
    }

    /**
     * @brief Get the wrapped user update.
     *
     * @return Reference to the stored user update.
     */
    [[nodiscard]] U& value() noexcept { return value_; }

    /**
     * @brief Get the wrapped user update.
     *
     * @return Const reference to the stored user update.
     */
    [[nodiscard]] const U& value() const noexcept { return value_; }

    /**
     * @brief Identifier used in lookups and printed reports. Fixed at construction.
     *
     * @return Name of the update.
     */
    [[nodiscard]] const std::string& name() const noexcept { return stats_.name; }

    /**
     * @brief All metadata of the update, bundled in a simplemc::update_stats.
     *
     * @return Const reference to the live metadata.
     */
    [[nodiscard]] const update_stats& stats() const noexcept { return stats_; }

    /**
     * @brief Set the identifier of the inverse update.
     *
     * @details It throws a simplemc::simplemc_exception when the given name is empty. This is usually
     * called indirectly via update_set::link_pair().
     *
     * @param inv_name Inverse-update identifier.
     */
    void set_inv_name(std::string inv_name) {
        if (inv_name.empty()) {
            throw simplemc_exception(fmt::format("update inverse name must be non-empty (on '{}')", stats_.name));
        }
        stats_.inv_name = std::move(inv_name);
    }

    /**
     * @brief Set the unnormalized selection weight.
     *
     * @details It throws a simplemc::simplemc_exception when the weight is negative.
     *
     * @param weight Selection weight (\f$ w \geq 0 \f$).
     */
    void set_weight(double weight) {
        if (weight < 0.0) {
            throw simplemc_exception(fmt::format("update weight must be >= 0 (got {} on '{}')", weight, stats_.name));
        }
        stats_.weight = weight;
    }

    /**
     * @brief Set the detailed-balance correction multiplier.
     *
     * @details This is usually called indirectly via update_set::rebuild_distribution(), which
     * recomputes it from the weight ratio of the update and its inverse.
     *
     * @param ratio Detailed-balance correction.
     */
    void set_ratio(double ratio) noexcept { stats_.ratio = ratio; }

    /**
     * @brief Propose a change to the simulation state.
     *
     * @details It increments the proposal counter update_stats::nprops and calls the `%attempt()`
     * member of the wrapped user type.
     *
     * @return Acceptance probability of the proposed change.
     */
    double attempt() {
        ++stats_.nprops;
        return value_.attempt();
    }

    /**
     * @brief Commit the proposed change.
     *
     * @details It increments the acceptance counter update_stats::naccs and calls the `%accept()`
     * member of the wrapped user type.
     */
    void accept() {
        ++stats_.naccs;
        value_.accept();
    }

    /**
     * @brief Roll back any speculative state set up by attempt().
     *
     * @details If the wrapped user type defines a `%reject()` member it is invoked. Otherwise this is
     * a no-op.
     */
    void reject() {
        if constexpr (requires { value_.reject(); }) {
            value_.reject();
        }
    }

    /**
     * @brief Signal that the last proposal was impossible.
     *
     * @details It increments the impossible counter update_stats::nimps. It does *not* call reject();
     * a kernel rejecting an impossible proposal must still do so explicitly.
     */
    void mark_impossible() noexcept { ++stats_.nimps; }

    /**
     * @brief Zero the counters.
     *
     * @details This discards all update statistics gathered so far. The typical use is dropping the 
     * statistics of a warm-up run before the measurement run starts.
     */
    void reset_counters() noexcept {
        stats_.nprops = 0;
        stats_.naccs = 0;
        stats_.nimps = 0;
    }

    // Friend declarations.
    template <serializer S, mc_update V>
    friend void simplemc_load(const S& s, update<V>& u);

    template <mc_update V>
    friend void simplemc_mpi_collect(const mpi::communicator& comm, update<V>& u);

private:
    U value_;
    update_stats stats_;
};

/**
 * @relates simplemc::update
 * @brief Serialize a simplemc::update.
 *
 * @details It serializes all metadata except update_stats::name (which serves as the map key). If the
 * wrapped user update is serializable by `S`, update::value() is also serialized.
 *
 * @tparam S Serializer type.
 * @tparam U User update type.
 * @param s Serializer handle.
 * @param u Update to serialize.
 */
template <serializer S, mc_update U>
void simplemc_save(S s, const update<U>& u) {
    s.save_at("inv_name", u.stats().inv_name);
    s.save_at("weight", u.stats().weight);
    s.save_at("ratio", u.stats().ratio);
    s.save_at("nprops", u.stats().nprops);
    s.save_at("naccs", u.stats().naccs);
    s.save_at("nimps", u.stats().nimps);
    if constexpr (save_at_all<U, S>) {
        s.save_at("user", u.value());
    }
}

/**
 * @relates simplemc::update
 * @brief Deserialize a simplemc::update.
 *
 * @details It deserializes all metadata except update_stats::name (which serves as the map key). If
 * the wrapped user update is deserializable by `S`, update::value() is also deserialized.
 *
 * @tparam S Serializer type.
 * @tparam U User update type.
 * @param s Serializer handle.
 * @param u Update to deserialize into.
 */
template <serializer S, mc_update U>
void simplemc_load(const S& s, update<U>& u) {
    s.load_at("inv_name", u.stats_.inv_name);
    s.load_at("weight", u.stats_.weight);
    s.load_at("ratio", u.stats_.ratio);
    s.load_at("nprops", u.stats_.nprops);
    s.load_at("naccs", u.stats_.naccs);
    s.load_at("nimps", u.stats_.nimps);
    if constexpr (load_at_all<U, S>) {
        s.load_at("user", u.value());
    }
}

/**
 * @relates simplemc::update
 * @brief Serialize the user-input config of a simplemc::update.
 *
 * @details It serializes update_stats::weight and the wrapped user update (if it has an input-config
 * serialization).
 *
 * @tparam S Serializer type.
 * @tparam U User update type.
 * @param s Serializer handle.
 * @param u Update to serialize.
 */
template <serializer S, mc_update U>
void simplemc_save_input_config(S s, const update<U>& u) {
    s.save_at("weight", u.stats().weight);
    if constexpr (has_simplemc_save_input_config<U, S>) {
        simplemc_save_input_config(s["user"], u.value());
    }
}

/**
 * @relates simplemc::update
 * @brief Deserialize the user-input config of a simplemc::update.
 *
 * @details It deserializes update_stats::weight through update::set_weight(), so a negative weight
 * in the input config throws a simplemc::simplemc_exception. If the wrapped user update has an
 * input-config deserialization, update::value() is also deserialized.
 *
 * @tparam S Serializer type.
 * @tparam U User update type.
 * @param s Serializer handle.
 * @param u Update to deserialize into.
 */
template <serializer S, mc_update U>
void simplemc_load_input_config(const S& s, update<U>& u) {
    double weight = u.stats().weight;
    s.try_load_at("weight", weight);
    u.set_weight(weight);
    if constexpr (has_simplemc_load_input_config<U, S>) {
        simplemc_load_input_config(s["user"], u.value());
    }
}

/**
 * @relates simplemc::update
 * @brief Collect a simplemc::update from different MPI processes.
 *
 * @details It all-reduces the three counter fields and, if the user update satisfies
 * simplemc::has_simplemc_mpi_collect, it reduces the value via the ADL hook `%simplemc_mpi_collect`.
 *
 * @note This reduction is **not idempotent**: the counters are summed across ranks, so call it
 * exactly once per collection point (a second call double-counts).
 *
 * @tparam U User update type.
 * @param comm simplemc::mpi::communicator object.
 * @param u Update to reduce in place.
 */
template <mc_update U>
void simplemc_mpi_collect(const mpi::communicator& comm, update<U>& u) {
    mpi::all_reduce_in_place(u.stats_.nprops, MPI_SUM, comm);
    mpi::all_reduce_in_place(u.stats_.naccs, MPI_SUM, comm);
    mpi::all_reduce_in_place(u.stats_.nimps, MPI_SUM, comm);
    if constexpr (has_simplemc_mpi_collect<U>) {
        u.value() = simplemc_mpi_collect(comm, u.value());
    }
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_UPDATE_HPP
