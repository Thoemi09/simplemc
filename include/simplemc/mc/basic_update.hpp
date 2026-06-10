/**
 * @file
 * @brief Polymorphic value type holding a Monte Carlo update.
 */

#ifndef SIMPLEMC_MC_BASIC_UPDATE_HPP
#define SIMPLEMC_MC_BASIC_UPDATE_HPP

#include <simplemc/mc/basic_wrapper.hpp>
#include <simplemc/mc/concepts.hpp>
#include <simplemc/mc/traits.hpp>

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc
 * @{
 */

/**
 * @brief Role interface of the update wrapper: adds the update virtuals to the common set.
 *
 * @details Derives from simplemc::basic_interface and adds the Monte Carlo update operations
 * (`attempt()` / `accept()` / `reject()`). simplemc::basic_update owns one of these (through
 * simplemc::basic_wrapper) and forwards its public update methods to it.
 *
 * @tparam Traits Traits bundle satisfying simplemc::mc_traits_like.
 */
template <mc_traits_like Traits>
struct update_interface : basic_interface<Traits, update_interface<Traits>> {
    /**
     * @brief Propose a change to the simulation state.
     *
     * @return Acceptance probability of the proposed change.
     */
    virtual double attempt() = 0;

    /**
     * @brief Commit the proposed change.
     */
    virtual void accept() = 0;

    /**
     * @brief Roll back any speculative state set up by attempt().
     */
    virtual void reject() = 0;
};

/**
 * @brief Role model of the update wrapper: implements the update virtuals over the wrapped value.
 *
 * @details Derives from simplemc::basic_model (which supplies the role-agnostic overrides) and adds
 * the update overrides. `reject()` is optional on the wrapped type; when the wrapped type omits it,
 * the override is a no-op.
 *
 * @tparam Traits Traits bundle satisfying simplemc::mc_traits_like.
 * @tparam U Wrapped user type.
 */
template <mc_traits_like Traits, class U>
struct update_model final : basic_model<Traits, update_interface<Traits>, U> {
    using basic_model<Traits, update_interface<Traits>, U>::basic_model;

    /**
     * @brief Forward to the wrapped value's `attempt()`.
     */
    double attempt() override { return this->concrete.attempt(); }

    /**
     * @brief Forward to the wrapped value's `accept()`.
     */
    void accept() override { this->concrete.accept(); }

    /**
     * @brief Forward to the wrapped value's `reject()` if it has one; otherwise a no-op.
     */
    void reject() override {
        if constexpr (requires { this->concrete.reject(); }) {
            this->concrete.reject();
        }
    }

    /**
     * @brief Deep-copy this model, preserving the dynamic type.
     */
    [[nodiscard]] std::unique_ptr<update_interface<Traits>> clone() const override {
        return std::make_unique<update_model>(*this);
    }
};

/**
 * @brief A Monte Carlo update.
 *
 * @details A Monte Carlo update proposes a change to the simulation state and lets the driver
 * decide whether to commit or roll back the proposal. It has three callbacks:
 *
 * - `attempt()` proposes the change and returns its acceptance probability.
 * - `accept()` commits the proposed change.
 * - `reject()` (optional on the user type) rolls back any speculative state set up by `attempt()`.
 *
 * The library accepts any user-defined type that satisfies the simplemc::mc_update concept — there is
 * no base class to inherit from. The optional `reject()` is detected at wrap time, i.e. when the user
 * type omits it, the wrapper's `reject()` is a no-op. A user value is wrapped by construction:
 *
 * @code{.cpp}
 * struct my_update {
 *     double attempt() { ... return p; }
 *     void accept()    { ... }
 *     void reject()    { ... }   // optional
 * };
 *
 * simplemc::basic_update u{ my_update{} };
 * @endcode
 *
 * simplemc::basic_update is a regular value type: copies are deep, moves are cheap, and it can be
 * stored directly in containers such as `std::vector<%basic_update>` to hold a heterogeneous collection
 * of updates. Because copies are deep, the wrapped user type must be copy-constructible; this is
 * checked at construction time.
 *
 * The wrapper is parameterized on a simplemc::mc_traits_like bundle whose
 * `checkpoint_serializer_type` and `input_config_serializer_type` drive checkpointing and
 * input-config serialization, respectively:
 * - If the wrapped user type provides a serialization path through the checkpoint serializer's
 * `%save_at()` / `%load_at()`, the wrapper's save_at() / load_at() forward to it.
 * - If the wrapped user type satisfies simplemc::has_simplemc_save_input_config with the
 * input-config serializer, the wrapper's save_input_config_at() / load_input_config_at() forward to
 * the user type's ADL hooks.
 *
 * Otherwise they are silent no-ops.
 *
 * The deep-copy/serialization/MPI machinery is shared with simplemc::basic_measurement through
 * simplemc::basic_wrapper; this class adds only the update role. See @ref simplemc-mc for a
 * description of the implementation strategy.
 *
 * @tparam Traits Traits bundle satisfying simplemc::mc_traits_like (default: simplemc::mc_traits<>).
 */
template <mc_traits_like Traits = mc_traits<>>
class basic_update : public basic_wrapper<Traits, update_interface<Traits>> {
    using base_type = basic_wrapper<Traits, update_interface<Traits>>;

public:
    /**
     * @brief Wrap a user-defined update.
     *
     * @details The user type is passed by forwarding reference, so lvalues are copied and rvalues
     * are moved into the wrapper.
     *
     * The wrapped type must also be copy-constructible: simplemc::basic_update is a value type with
     * deep-copy semantics and the internal model holds an instance by value. This is a storage
     * requirement of the wrapper, not part of the simplemc::mc_update role.
     *
     * @tparam U Type satisfying simplemc::mc_update and `std::copy_constructible`.
     * @param u Update to wrap.
     */
    template <class U>
        requires(!std::same_as<std::remove_cvref_t<U>, basic_update>) && mc_update<std::remove_cvref_t<U>> &&
        std::copy_constructible<std::remove_cvref_t<U>>
    basic_update(U&& u) :
        base_type { std::make_unique<update_model<Traits, std::remove_cvref_t<U>>>(std::forward<U>(u)) } {}

    /**
     * @brief Propose a change to the simulation state.
     *
     * @return Acceptance probability of the proposed change.
     */
    double attempt() { return this->impl().attempt(); }

    /**
     * @brief Commit the proposed change.
     */
    void accept() { this->impl().accept(); }

    /**
     * @brief Roll back any speculative state set up by attempt().
     *
     * @details If the wrapped user type defines a `%reject()` member it is invoked; otherwise this
     * is a no-op.
     */
    void reject() { this->impl().reject(); }
};

// Deduction guide: basic_update u { my_update{} } deduces to basic_update<mc_traits<>>.
template <class U>
    requires mc_update<std::remove_cvref_t<U>> && std::copy_constructible<std::remove_cvref_t<U>>
basic_update(U&&) -> basic_update<>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_BASIC_UPDATE_HPP
