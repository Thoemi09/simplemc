/**
 * @file
 * @brief MC measurement wrapper class.
 */

#ifndef SIMPLEMC_MC_MEASUREMENT_HPP
#define SIMPLEMC_MC_MEASUREMENT_HPP

#include <simplemc/mc/concepts.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <string>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-entities-measurements
 * @{
 */

/**
 * @brief MC measurement wrapper class for a user-defined measurement type.
 *
 * @details It owns a user-defined measurement @ref value of type @ref value_type (satisfying
 * simplemc::mc_measurement) together with its metadata:
 *
 * - a unique @ref name that identifies the measurement, and
 * - a boolean flag indicating whether the measurement @ref is_active during an MC simulation.
 *
 * All fields are public so the driver and reporting code can read and write them directly. The
 * wrapped type is stored by value and accessed via the public @ref value member.
 *
 * @tparam M User measurement type satisfying simplemc::mc_measurement.
 */
template <mc_measurement M>
struct measurement {
    /**
     * @brief The concrete wrapped user type.
     */
    using value_type = M;

    /**
     * @brief The wrapped user measurement.
     */
    M value;

    /**
     * @brief Identifier used in lookups and printed reports.
     */
    std::string name;

    /**
     * @brief Whether the driver invokes measure() during each cycle (see simplemc::run).
     */
    bool is_active;

    /**
     * @brief Constructor stores a user-defined measurement value.
     *
     * @details It validates that the name is not empty, throwing a simplemc::simplemc_exception
     * otherwise.
     *
     * @param value User measurement value to store.
     * @param name Identifier.
     * @param is_active Initial activation state.
     */
    measurement(M value, std::string name, bool is_active = true) :
        value { std::move(value) },
        name { std::move(name) },
        is_active { is_active } {
        if (this->name.empty()) {
            throw simplemc_exception("measurement name must be non-empty");
        }
    }

    /**
     * @brief Perform the measurement 
     * 
     * @details It simply calls the `%measure()` member of the wrapped user type.
     */
    void measure() { value.measure(); }
};

/**
 * @relates simplemc::measurement
 * @brief Serialize a simplemc::measurement.
 *
 * @details It serializes measurement::is_active and, if the wrapped user measurement is serializable
 * by `S`, measurement::value.
 *
 * @tparam S Serializer type.
 * @tparam M User measurement type.
 * @param s Serializer handle.
 * @param m Measurement to serialize.
 */
template <serializer S, mc_measurement M>
void simplemc_save(S& s, const measurement<M>& m) {
    s.save_at("is_active", m.is_active);
    if constexpr (save_at_all<M, S>) {
        s.save_at("user", m.value);
    }
}

/**
 * @relates simplemc::measurement
 * @brief Deserialize a simplemc::measurement.
 *
 * @details It deserializes measurement::is_active and, if the wrapped user measurement is
 * deserializable by `S`, measurement::value.
 *
 * @tparam S Serializer type.
 * @tparam M User measurement type.
 * @param s Serializer handle.
 * @param m Measurement to deserialize into.
 */
template <serializer S, mc_measurement M>
void simplemc_load(const S& s, measurement<M>& m) {
    s.load_at("is_active", m.is_active);
    if constexpr (load_at_all<M, S>) {
        s.load_at("user", m.value);
    }
}

/**
 * @relates simplemc::measurement
 * @brief Serialize the user-input config of a simplemc::measurement.
 *
 * @details It serializes measurement::is_active and, if the wrapped user measurement has an
 * input-config serialization, measurement::value.
 *
 * @tparam S Serializer type.
 * @tparam M User measurement type.
 * @param s Serializer handle.
 * @param m Measurement to serialize.
 */
template <serializer S, mc_measurement M>
void simplemc_save_input_config(S& s, const measurement<M>& m) {
    s.save_at("is_active", m.is_active);
    if constexpr (has_simplemc_save_input_config<M, S>) {
        auto sub = s["user"];
        simplemc_save_input_config(sub, m.value);
    }
}

/**
 * @relates simplemc::measurement
 * @brief Deserialize the user-input config of a simplemc::measurement.
 *
 * @details It deserializes measurement::is_active and, if the wrapped user measurement has an
 * input-config deserialization, measurement::value.
 *
 * @tparam S Serializer type.
 * @tparam M User measurement type.
 * @param s Serializer handle.
 * @param m Measurement to deserialize into.
 */
template <serializer S, mc_measurement M>
void simplemc_load_input_config(const S& s, measurement<M>& m) {
    s.try_load_at("is_active", m.is_active);
    if constexpr (has_simplemc_load_input_config<M, S>) {
        const auto sub = s["user"];
        simplemc_load_input_config(sub, m.value);
    }
}

/**
 * @relates simplemc::measurement
 * @brief Collect a simplemc::measurement from different MPI processes.
 *
 * @details If the user measurement satisfies simplemc::has_simplemc_mpi_collect, it reduces the value
 * via the ADL hook `%simplemc_mpi_collect`.
 *
 * @tparam M User measurement type.
 * @param comm simplemc::mpi::communicator object.
 * @param m Measurement to reduce in place.
 */
template <mc_measurement M>
void simplemc_mpi_collect(const mpi::communicator& comm, measurement<M>& m) {
    if constexpr (has_simplemc_mpi_collect<M>) {
        m.value = simplemc_mpi_collect(comm, m.value);
    }
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_MEASUREMENT_HPP
