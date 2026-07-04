/**
 * @file
 * @brief MC measurement payload together with some metadata.
 */

#ifndef SIMPLEMC_MC_MEASUREMENT_HPP
#define SIMPLEMC_MC_MEASUREMENT_HPP

#include <simplemc/mc/concepts.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <concepts>
#include <string>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-entities-measurements
 * @{
 */

/**
 * @brief MC measurement payload together with some metadata.
 *
 * @details simplemc::measurement owns a user measurement value of type `M` (satisfying
 * simplemc::mc_measurement) together with its metadata:
 *
 * - a unique name that identifies the measurement, and
 * - a boolean flag indicating whether the measurement is active during an MC simulation.
 *
 * All fields are public so the driver and reporting code can read and write them directly. The wrapped
 * type is stored by value; recover it via get() or the nested alias @ref payload_type.
 *
 * @tparam M User measurement type satisfying simplemc::mc_measurement.
 */
template <mc_measurement M>
struct measurement {
    /**
     * @brief The concrete wrapped user type.
     */
    using payload_type = M;

    /**
     * @brief The wrapped user measurement.
     */
    M payload;

    /**
     * @brief Identifier used in lookups and printed reports.
     */
    std::string name;

    /**
     * @brief Whether the driver invokes `measure()` during each cycle.
     */
    bool is_active;

    /**
     * @brief Constructor stores a user-defined measurement value.
     *
     * @details It validates that the name is not empty, throwing a simplemc::simplemc_exception
     * otherwise. The first parameter is the class template parameter by value, so the
     * implicitly-generated deduction guide lets `measurement{ my_obs {}, "name" }` deduce
     * `measurement<my_obs>`.
     *
     * @param payload User measurement value to store.
     * @param name Identifier.
     * @param is_active Initial activation state.
     */
    measurement(M payload, std::string name, bool is_active = true) :
        payload { std::move(payload) },
        name { std::move(name) },
        is_active { is_active } {
        if (this->name.empty()) {
            throw simplemc_exception("measurement name must be non-empty");
        }
    }

    /**
     * @brief Perform the measurement by calling the `%measure()` member of the wrapped user type.
     */
    void measure() { payload.measure(); }

    /**
     * @brief Recover a pointer to the wrapped user measurement.
     *
     * @tparam T Expected type of the wrapped user measurement.
     * @return Pointer to the wrapped measurement, or `nullptr` if `T` is not the wrapped type.
     */
    template <typename T>
    [[nodiscard]] T* get() noexcept {
        if constexpr (std::same_as<T, M>) {
            return &payload;
        } else {
            return nullptr;
        }
    }

    /**
     * @brief Const overload of get().
     */
    template <typename T>
    [[nodiscard]] const T* get() const noexcept {
        if constexpr (std::same_as<T, M>) {
            return &payload;
        } else {
            return nullptr;
        }
    }
};

/**
 * @relates simplemc::measurement
 * @brief Serialize a simplemc::measurement.
 *
 * @details It serializes `is_active`, then the wrapped payload under `"user"` if the payload is
 * serializable by `S` (otherwise the payload is skipped).
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
        s.save_at("user", m.payload);
    }
}

/**
 * @relates simplemc::measurement
 * @brief Deserialize a simplemc::measurement.
 *
 * @details Symmetric to simplemc_save(S&, const measurement<M>&).
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
        s.load_at("user", m.payload);
    }
}

/**
 * @relates simplemc::measurement
 * @brief Serialize the user-input config of a simplemc::measurement.
 *
 * @details It serializes `is_active` and, if the payload has an input-config serialization, the
 * payload under `"user"`.
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
        simplemc_save_input_config(sub, m.payload);
    }
}

/**
 * @relates simplemc::measurement
 * @brief Deserialize the user-input config of a simplemc::measurement.
 *
 * @details Symmetric to simplemc_save_input_config(S&, const measurement<M>&). The `is_active` flag is
 * optional.
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
        simplemc_load_input_config(sub, m.payload);
    }
}

/**
 * @relates simplemc::measurement
 * @brief Collect a simplemc::measurement from different MPI processes.
 *
 * @details If the payload supports it, reduces the payload via its own `%simplemc_mpi_collect`.
 *
 * @tparam M User measurement type.
 * @param comm simplemc::mpi::communicator object.
 * @param m Measurement to reduce in place.
 */
template <mc_measurement M>
void simplemc_mpi_collect(const mpi::communicator& comm, measurement<M>& m) {
    if constexpr (has_simplemc_mpi_collect<M>) {
        m.payload = simplemc_mpi_collect(comm, m.payload);
    }
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_MEASUREMENT_HPP
