// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

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
 * @details It owns a user-defined measurement of type @ref value_type (satisfying
 * simplemc::mc_measurement) together with its metadata:
 *
 * - a unique name() that identifies the measurement, and
 * - a boolean flag indicating whether the measurement is_active() during an MC simulation.
 *
 * The wrapped type is stored by value and can be accessed via value().
 *
 * @tparam M User measurement type satisfying simplemc::mc_measurement.
 */
template <mc_measurement M>
class measurement {
public:
    /**
     * @brief The concrete wrapped user type.
     */
    using value_type = M;

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
        value_ { std::move(value) },
        name_ { std::move(name) },
        is_active_ { is_active } {
        if (name_.empty()) {
            throw simplemc_exception("measurement name must be non-empty");
        }
    }

    /**
     * @brief Get the wrapped user measurement.
     *
     * @return Reference to the stored user measurement.
     */
    [[nodiscard]] M& value() noexcept { return value_; }

    /**
     * @brief Get the wrapped user measurement.
     *
     * @return Const reference to the stored user measurement.
     */
    [[nodiscard]] const M& value() const noexcept { return value_; }

    /**
     * @brief Identifier used in lookups and printed reports. Fixed at construction.
     *
     * @return Name of the measurement.
     */
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    /**
     * @brief Get the activation state.
     *
     * @details It determines whether the driver invokes measure() during each cycle (see
     * simplemc::run).
     *
     * @return Current activation state.
     */
    [[nodiscard]] bool is_active() const noexcept { return is_active_; }

    /**
     * @brief Set the activation state.
     *
     * @details This is usually called indirectly via measurement_set::set_active().
     *
     * @param active New activation state.
     */
    void set_active(bool active) noexcept { is_active_ = active; }

    /**
     * @brief Perform the measurement.
     *
     * @details It simply calls the `%measure()` member of the wrapped user type.
     */
    void measure() { value_.measure(); }

private:
    M value_;
    std::string name_;
    bool is_active_;
};

/**
 * @relates simplemc::measurement
 * @brief Serialize a simplemc::measurement.
 *
 * @details It serializes the current activation state and the wrapped user measurement (if it is
 * serializable by `S`).
 *
 * @tparam S Serializer type.
 * @tparam M User measurement type.
 * @param s Serializer handle.
 * @param m Measurement to serialize.
 */
template <serializer S, mc_measurement M>
void simplemc_save(S s, const measurement<M>& m) {
    s.save_at("is_active", m.is_active());
    if constexpr (save_at_all<M, S>) {
        s.save_at("user", m.value());
    }
}

/**
 * @relates simplemc::measurement
 * @brief Deserialize a simplemc::measurement.
 *
 * @details It deserializes the current activation state and the wrapped user measurement (if it is
 * deserializable by `S`).
 *
 * @tparam S Serializer type.
 * @tparam M User measurement type.
 * @param s Serializer handle.
 * @param m Measurement to deserialize into.
 */
template <serializer S, mc_measurement M>
void simplemc_load(const S& s, measurement<M>& m) {
    bool is_active {};
    s.load_at("is_active", is_active);
    m.set_active(is_active);
    if constexpr (load_at_all<M, S>) {
        s.load_at("user", m.value());
    }
}

/**
 * @relates simplemc::measurement
 * @brief Serialize the user-input config of a simplemc::measurement.
 *
 * @details It serializes the current activation state and the wrapped user measurement (if it has
 * an input-config serialization).
 *
 * @tparam S Serializer type.
 * @tparam M User measurement type.
 * @param s Serializer handle.
 * @param m Measurement to serialize.
 */
template <serializer S, mc_measurement M>
void simplemc_save_input_config(S s, const measurement<M>& m) {
    s.save_at("is_active", m.is_active());
    if constexpr (has_simplemc_save_input_config<M, S>) {
        simplemc_save_input_config(s["user"], m.value());
    }
}

/**
 * @relates simplemc::measurement
 * @brief Deserialize the user-input config of a simplemc::measurement.
 *
 * @details It deserializes the current activation state and the wrapped user measurement (if it has
 * an input-config deserialization).
 *
 * @tparam S Serializer type.
 * @tparam M User measurement type.
 * @param s Serializer handle.
 * @param m Measurement to deserialize into.
 */
template <serializer S, mc_measurement M>
void simplemc_load_input_config(const S& s, measurement<M>& m) {
    bool is_active = m.is_active();
    s.try_load_at("is_active", is_active);
    m.set_active(is_active);
    if constexpr (has_simplemc_load_input_config<M, S>) {
        simplemc_load_input_config(s["user"], m.value());
    }
}

/**
 * @relates simplemc::measurement
 * @brief Collect a simplemc::measurement from different MPI processes.
 *
 * @details If the user measurement satisfies simplemc::has_simplemc_mpi_collect, it reduces the value
 * via the ADL hook `%simplemc_mpi_collect`.
 * 
 * @note To keep the per-rank state, copy the measurement first.
 *
 * @tparam M User measurement type.
 * @param comm simplemc::mpi::communicator object.
 * @param m Measurement to reduce in place.
 */
template <mc_measurement M>
void simplemc_mpi_collect(const mpi::communicator& comm, measurement<M>& m) {
    if constexpr (has_simplemc_mpi_collect<M>) {
        simplemc_mpi_collect(comm, m.value());
    }
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_MEASUREMENT_HPP
