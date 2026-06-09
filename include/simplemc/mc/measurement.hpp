/**
 * @file
 * @brief Per-measurement aggregate: wrapped user value plus registration metadata.
 */

#ifndef SIMPLEMC_MC_MEASUREMENT_HPP
#define SIMPLEMC_MC_MEASUREMENT_HPP

#include <simplemc/mc/basic_measurement.hpp>
#include <simplemc/mc/concepts.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <concepts>
#include <string>
#include <type_traits>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc
 * @{
 */

/**
 * @brief One Monte Carlo measurement entry: the wrapped user value plus registration metadata.
 *
 * @details simplemc::measurement owns a simplemc::basic_measurement together with the metadata that
 * identifies it (`name`) and controls when it fires (`is_active`). All fields are public so the
 * driver and reporting code can read and write them directly.
 *
 * Construction validates `name` non-empty.
 *
 * @tparam S1 Serializer type used for checkpoint serialization.
 * @tparam S2 Serializer type used for input-config serialization.
 */
template <serializer S1 = json_serializer, serializer S2 = json_serializer>
struct measurement {
    /**
     * @brief Type used for checkpoint serialization.
     */
    using cp_serializer_type = S1;

    /**
     * @brief Type used for input-config serialization.
     */
    using ic_serializer_type = S2;

    /**
     * @brief The wrapped user measurement.
     */
    basic_measurement<S1, S2> wrapped;

    /**
     * @brief Identifier used in lookups and printed reports.
     */
    std::string name;

    /**
     * @brief Whether the driver invokes `measure()` during each cycle.
     */
    bool is_active;

    /**
     * @brief Construct by wrapping a user-defined measurement value.
     *
     * @details Forwards `m` into an internally-constructed simplemc::basic_measurement. Validates
     * `name` non-empty. The forwarded type must satisfy simplemc::mc_measurement and
     * `std::copy_constructible`.
     *
     * @tparam M User measurement type.
     * @param m User value to wrap.
     * @param name Identifier.
     * @param is_active Initial activation state.
     */
    template <class M>
        requires(!std::same_as<std::remove_cvref_t<M>, measurement>) &&
                (!std::same_as<std::remove_cvref_t<M>, basic_measurement<S1, S2>>) &&
                mc_measurement<std::remove_cvref_t<M>> &&
                std::copy_constructible<std::remove_cvref_t<M>>
    measurement(M&& m, std::string name, bool is_active = true) : measurement {
        basic_measurement<S1, S2> { std::forward<M>(m) }, std::move(name), is_active
    } {}

    /**
     * @brief Construct from a pre-built simplemc::basic_measurement wrapper.
     *
     * @details Validates `name` non-empty.
     *
     * @param w Pre-built wrapper.
     * @param name Identifier.
     * @param is_active Initial activation state.
     */
    measurement(basic_measurement<S1, S2> w, std::string name, bool is_active = true) :
        wrapped { std::move(w) }, name { std::move(name) }, is_active { is_active } {
        if (this->name.empty()) {
            throw simplemc_exception("measurement name must be non-empty");
        }
    }

    /**
     * @brief Take a measurement (delegates to the wrapped user type).
     */
    void measure() { wrapped.measure(); }

    /**
     * @brief Recover a pointer to the wrapped user value, checked by type.
     *
     * @tparam T Expected concrete type of the wrapped user value.
     * @return Pointer to the wrapped value, or `nullptr` on type mismatch.
     */
    template <class T>
    [[nodiscard]] T* get() noexcept {
        return wrapped.template get<T>();
    }

    /**
     * @brief Const overload of get().
     */
    template <class T>
    [[nodiscard]] const T* get() const noexcept {
        return wrapped.template get<T>();
    }

    /**
     * @brief Serialize persistent fields of this measurement.
     *
     * @details Writes `is_active` and delegates to the wrapper for the `"user"` sub-key. Schema
     * matches the previous simplemc::measurement_stats layout (plus the wrapper's `"user"` sub-tree).
     */
    friend void simplemc_save(cp_serializer_type& s, const measurement& m) {
        s.save_at("is_active", m.is_active);
        m.wrapped.save_at(s, "user");
    }

    /**
     * @brief Deserialize the persistent fields of this measurement.
     */
    friend void simplemc_load(const cp_serializer_type& s, measurement& m) {
        s.load_at("is_active", m.is_active);
        m.wrapped.load_at(s, "user");
    }

    /**
     * @brief Serialize the user-input config of this measurement.
     *
     * @details Writes `is_active` and delegates to the wrapper for the `"user"` sub-key.
     */
    friend void simplemc_save_input_config(ic_serializer_type& s, const measurement& m) {
        s.save_at("is_active", m.is_active);
        m.wrapped.save_input_config_at(s, "user");
    }

    /**
     * @brief Deserialize the user-input config of this measurement.
     */
    friend void simplemc_load_input_config(const ic_serializer_type& s, measurement& m) {
        s.try_load_at("is_active", m.is_active);
        m.wrapped.load_input_config_at(s, "user");
    }

    /**
     * @brief All-reduce this measurement's wrapped payload across MPI ranks.
     *
     * @details Forwards to the wrapper's `mpi_collect()`, which dispatches via ADL to the wrapped
     * user type's `simplemc_mpi_collect(comm, T&)` overload (silent no-op when the user type does
     * not provide one). The measurement's identification fields (`name`, `is_active`) are local
     * registration data assumed identical across ranks and are not touched.
     *
     * @param comm MPI communicator over which to reduce.
     * @param m Measurement to reduce in place.
     */
    friend void simplemc_mpi_collect(const mpi::communicator& comm, measurement& m) {
        m.wrapped.mpi_collect(comm);
    }
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_MEASUREMENT_HPP
