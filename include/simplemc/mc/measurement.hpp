/**
 * @file
 * @brief Type-erased MC measurement together with some metadata.
 */

#ifndef SIMPLEMC_MC_MEASUREMENT_HPP
#define SIMPLEMC_MC_MEASUREMENT_HPP

#include <simplemc/mc/basic_measurement.hpp>
#include <simplemc/mc/concepts.hpp>
#include <simplemc/mc/serializer.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <concepts>
#include <string>
#include <type_traits>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-entities-measurements
 * @{
 */

/**
 * @brief Type-erased MC measurement together with some metadata.
 *
 * @details simplemc::measurement owns a simplemc::basic_measurement together with its metadata:
 *
 * - a unique name that identifies the measurement, and
 * - a boolean flag indicating whether the measurement is active during an MC simulation.
 *
 * All fields are public so the driver and reporting code can read and write them directly.
 */
struct measurement {
    /**
     * @brief Serializer type used for both checkpoint and input-config serialization.
     */
    using serializer_type = mc_serializer;

    /**
     * @brief The wrapped user measurement.
     */
    basic_measurement wrapped;

    /**
     * @brief Identifier used in lookups and printed reports.
     */
    std::string name;

    /**
     * @brief Whether the driver invokes `measure()` during each cycle.
     */
    bool is_active;

    /**
     * @brief Constructor wraps a user-defined measurement value.
     *
     * @details It forwards the given measurement into an internally-constructed
     * simplemc::basic_measurement and validates that the name is not empty. Otherwise, it throws
     * a simplemc::simplemc_exception.
     *
     * The forwarded type must satisfy simplemc::mc_measurement and `std::copy_constructible`.
     *
     * @tparam M User measurement type.
     * @param m User measurement to wrap.
     * @param name Identifier.
     * @param is_active Initial activation state.
     */
    template <typename M>
        requires(!std::same_as<std::remove_cvref_t<M>, measurement>) &&
        (!std::same_as<std::remove_cvref_t<M>, basic_measurement>) && mc_measurement<std::remove_cvref_t<M>> &&
        std::copy_constructible<std::remove_cvref_t<M>>
    measurement(M&& m, std::string name, bool is_active = true) :
        measurement { basic_measurement { std::forward<M>(m) }, std::move(name), is_active } {}

    /**
     * @brief Construct a measurement from a pre-built simplemc::basic_measurement wrapper.
     *
     * @details It validates that the name is not empty. Otherwise, it throws a
     * simplemc::simplemc_exception.
     *
     * @param w Pre-built simplemc::basic_measurement wrapper.
     * @param name Identifier.
     * @param is_active Initial activation state.
     */
    measurement(basic_measurement w, std::string name, bool is_active = true) :
        wrapped { std::move(w) },
        name { std::move(name) },
        is_active { is_active } {
        if (this->name.empty()) {
            throw simplemc_exception("measurement name must be non-empty");
        }
    }

    /**
     * @brief Perform the measurement by calling the `%measure()` member of the wrapped user type.
     */
    void measure() { wrapped.measure(); }

    /**
     * @brief Recover a pointer to the wrapped user measurement.
     *
     * @details It simply calls basic_measurement::get<T>().
     *
     * @tparam T Expected type of the wrapped user measurement.
     * @return Pointer to the wrapped measurement, or `nullptr` on type mismatch.
     */
    template <typename T>
    [[nodiscard]] T* get() noexcept {
        return wrapped.template get<T>();
    }

    /**
     * @brief Const overload of get().
     */
    template <typename T>
    [[nodiscard]] const T* get() const noexcept {
        return wrapped.template get<T>();
    }

    /**
     * @brief Serialize a simplemc::measurement.
     *
     * @details It serializes `is_active` and calls basic_measurement::save_at().
     *
     * @param s Serializer handle.
     * @param m Measurement to serialize.
     */
    friend void simplemc_save(serializer_type& s, const measurement& m) {
        s.save_at("is_active", m.is_active);
        m.wrapped.save_at(s, "user");
    }

    /**
     * @brief Deserialize a simplemc::measurement.
     *
     * @details It deserializes `is_active` and calls basic_measurement::load_at().
     *
     * @param s Serializer handle.
     * @param m Measurement to deserialize into.
     */
    friend void simplemc_load(const serializer_type& s, measurement& m) {
        s.load_at("is_active", m.is_active);
        m.wrapped.load_at(s, "user");
    }

    /**
     * @brief Serialize the user-input config of a simplemc::measurement.
     *
     * @details It serializes `is_active` and calls basic_measurement::save_input_config_at().
     *
     * @param s Serializer handle.
     * @param m Measurement to serialize.
     */
    friend void simplemc_save_input_config(serializer_type& s, const measurement& m) {
        s.save_at("is_active", m.is_active);
        m.wrapped.save_input_config_at(s, "user");
    }

    /**
     * @brief Deserialize the user-input config of a simplemc::measurement.
     *
     * @details It deserializes `is_active` and calls basic_measurement::load_input_config_at().
     *
     * @param s Serializer handle.
     * @param m Measurement to deserialize into.
     */
    friend void simplemc_load_input_config(const serializer_type& s, measurement& m) {
        s.try_load_at("is_active", m.is_active);
        m.wrapped.load_input_config_at(s, "user");
    }

    /**
     * @brief Collect simplemc::measurement objects from different MPI processes.
     *
     * @details It simply calls basic_measurement::mpi_collect().
     *
     * @param comm simplemc::mpi::communicator object.
     * @param m Measurement to reduce in place.
     */
    friend void simplemc_mpi_collect(const mpi::communicator& comm, measurement& m) { m.wrapped.mpi_collect(comm); }
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_MEASUREMENT_HPP
