/**
 * @file
 * @brief Polymorphic value type holding a Monte Carlo measurement.
 */

#ifndef SIMPLEMC_MC_BASIC_MEASUREMENT_HPP
#define SIMPLEMC_MC_BASIC_MEASUREMENT_HPP

#include <simplemc/mc/basic_wrapper.hpp>
#include <simplemc/mc/concepts.hpp>
#include <simplemc/mc/serializer.hpp>

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-entities-measurements
 * @{
 */

/**
 * @brief Role interface of the measurement wrapper: adds `measure()` to the common set.
 *
 * @details Derives from simplemc::basic_interface and adds the Monte Carlo measurement operation
 * (`measure()`). simplemc::basic_measurement owns one of these (through simplemc::basic_wrapper)
 * and forwards its public `measure()` to it.
 */
struct measurement_interface : basic_interface<measurement_interface> {
    /**
     * @brief Take a measurement.
     */
    virtual void measure() = 0;
};

/**
 * @brief Role model of the measurement wrapper: implements `measure()` over the wrapped value.
 *
 * @details Derives from simplemc::basic_model (which supplies the role-agnostic overrides) and adds
 * the `measure()` override.
 *
 * @tparam M Wrapped user type.
 */
template <class M>
struct measurement_model final : basic_model<measurement_interface, M> {
    using basic_model<measurement_interface, M>::basic_model;

    /**
     * @brief Forward to the wrapped value's `measure()`.
     */
    void measure() override { this->concrete.measure(); }

    /**
     * @brief Deep-copy this model, preserving the dynamic type.
     */
    [[nodiscard]] std::unique_ptr<measurement_interface> clone() const override {
        return std::make_unique<measurement_model>(*this);
    }
};

/**
 * @brief A Monte Carlo measurement.
 *
 * @details A measurement is an observer of the simulation state that the driver invokes once per
 * measurement cycle to record an observable, accumulate a sample, write to a histogram, etc.
 *
 * The library accepts any user-defined type that satisfies the simplemc::mc_measurement concept (i.e.
 * exposes a callable `measure()` member) — there is no base class to inherit from. A user value is
 * wrapped by construction:
 *
 * @code{.cpp}
 * struct my_observer {
 *     void measure() { ... }
 * };
 *
 * simplemc::basic_measurement m{ my_observer{} };
 * m.measure();
 * @endcode
 *
 * simplemc::basic_measurement is a regular value type: copies are deep, moves are cheap, and it can
 * be stored directly in containers such as `std::vector<%basic_measurement>` to hold a heterogeneous
 * collection of observers. Because copies are deep, the wrapped user type must be copy-constructible;
 * this is checked at construction time.
 *
 * Serialization runs through the library-wide simplemc::mc_serializer (a backend-erasing
 * simplemc::variant_serializer), used for both the checkpoint and the input-config channels:
 * - If the wrapped user type provides a serialization path through the serializer's
 * `%save_at()` / `%load_at()`, the wrapper's save_at() / load_at() forward to it.
 * - If the wrapped user type satisfies simplemc::has_simplemc_save_input_config with the serializer,
 * the wrapper's save_input_config_at() / load_input_config_at() forward to the user type's ADL hooks.
 * Otherwise they are silent no-ops.
 *
 * @note Because simplemc::mc_serializer is a variant over the shipped backends, its capability is
 * the *intersection* of those backends: with HDF5 enabled a wrapped payload must be serializable by
 * both JSON and HDF5 for the forwards above to engage.
 *
 * The deep-copy/serialization/MPI machinery is shared with simplemc::basic_update through
 * simplemc::basic_wrapper; this class adds only the measurement role. See @ref simplemc-mc
 * for a description of the implementation strategy.
 *
 * @note The wrapper takes the user measurement by value. To access the concrete value (e.g. an
 * accumulator) after a run, use the typed accessor get<T>() — it returns a pointer to the wrapped
 * value when the dynamic type matches `T`, or `nullptr` otherwise.
 */
class basic_measurement : public basic_wrapper<measurement_interface> {
    using base_type = basic_wrapper<measurement_interface>;

public:
    /**
     * @brief Wrap a user-defined measurement.
     *
     * @details The user type is passed by forwarding reference, so lvalues are copied and rvalues
     * are moved into the wrapper.
     *
     * The wrapped type must also be copy-constructible: simplemc::basic_measurement is a value type
     * with deep-copy semantics and the internal model holds an instance by value. This is a storage
     * requirement of the wrapper, not part of the simplemc::mc_measurement role.
     *
     * @tparam M Type satisfying simplemc::mc_measurement and `std::copy_constructible`.
     * @param m Measurement to wrap.
     */
    template <class M>
        requires(!std::same_as<std::remove_cvref_t<M>, basic_measurement>) && mc_measurement<std::remove_cvref_t<M>> &&
        std::copy_constructible<std::remove_cvref_t<M>>
    basic_measurement(M&& m) :
        base_type { std::make_unique<measurement_model<std::remove_cvref_t<M>>>(std::forward<M>(m)) } {}

    /**
     * @brief Take a measurement.
     */
    void measure() { this->impl().measure(); }
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_BASIC_MEASUREMENT_HPP
