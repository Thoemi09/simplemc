/**
 * @file
 * @brief Polymorphic value type holding a Monte Carlo update.
 */

#ifndef SIMPLEMC_MC_BASIC_UPDATE_HPP
#define SIMPLEMC_MC_BASIC_UPDATE_HPP

#include <simplemc/mc/concepts.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>

#include <concepts>
#include <memory>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc
 * @{
 */

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
 * The wrapper is templated on the serializer types `S1` and `S2` used for checkpointing and
 * input-config serialization, respectively:
 * - If the wrapped user type provides a serialization path through `S1`'s `%save_at()` /
 * `%load_at()`, the wrapper's save_at() / load_at() forward to it.
 * - If the wrapped user type satisfies simplemc::has_simplemc_save_input_config with `S2`, the
 * wrapper's save_input_config_at() / load_input_config_at() forward to the user type's ADL hooks.
 *
 * Otherwise they are silent no-ops.
 *
 * See @ref simplemc-mc for a description of the implementation strategy.
 *
 * @tparam S1 Serializer type used for checkpoint serialization.
 * @tparam S2 Serializer type used for input-config serialization.
 */
template <serializer S1 = json_serializer, serializer S2 = json_serializer>
class basic_update {
public:
    /**
     * @brief Type used for checkpoint serialization.
     */
    using cp_serializer_type = S1;

    /**
     * @brief Type used for input-config serialization.
     */
    using ic_serializer_type = S2;

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
    basic_update(U&& u) : pimpl_ { std::make_unique<model<std::remove_cvref_t<U>>>(std::forward<U>(u)) } {}

    /**
     * @brief Copy constructor.
     *
     * @details Performs a deep copy: the new wrapper owns an independent copy of `other`'s wrapped
     * value, produced by the wrapped type's copy constructor. The two wrappers do not share state.
     *
     * @param other Wrapper to copy from.
     */
    basic_update(const basic_update& other) : pimpl_ { other.pimpl_->clone() } {}

    /**
     * @brief Default move constructor.
     */
    basic_update(basic_update&& other) noexcept = default;

    /**
     * @brief Copy assignment.
     *
     * @details Performs a deep copy of `other`'s wrapped value into `*this` (via the copy-and-swap
     * pattern), discarding `*this`'s previous wrapped value. After the call, the two wrappers do
     * not share state.
     *
     * @param other Wrapper to copy from.
     * @return Reference to `*this`.
     */
    basic_update& operator=(const basic_update& other) {
        basic_update tmp { other };
        std::swap(pimpl_, tmp.pimpl_);
        return *this;
    }

    /**
     * @brief Default move assignment.
     */
    basic_update& operator=(basic_update&& other) noexcept = default;

    /**
     * @brief Default destructor.
     */
    ~basic_update() = default;

    /**
     * @brief Propose a change to the simulation state.
     *
     * @return Acceptance probability of the proposed change.
     */
    double attempt() { return pimpl_->attempt(); }

    /**
     * @brief Commit the proposed change.
     */
    void accept() { pimpl_->accept(); }

    /**
     * @brief Roll back any speculative state set up by attempt().
     *
     * @details If the wrapped user type defines a `%reject()` member it is invoked; otherwise this
     * is a no-op.
     */
    void reject() { pimpl_->reject(); }

    /**
     * @brief Recover a pointer to the wrapped user value, checked by type.
     *
     * @details Returns a pointer to the wrapped value if the dynamic type matches `T`, otherwise
     * `nullptr`. Lets the user pull the concrete object back out after wrapping — useful for
     * inspecting post-run state without external storage. Requires RTTI to be enabled.
     *
     * @tparam T Expected concrete type of the wrapped user value.
     * @return Pointer to the wrapped value, or `nullptr` on type mismatch.
     */
    template <class T>
    [[nodiscard]] T* get() noexcept {
        return typeid(T) == pimpl_->type_id() ? static_cast<T*>(pimpl_->address()) : nullptr;
    }

    /**
     * @brief Const overload of get().
     */
    template <class T>
    [[nodiscard]] const T* get() const noexcept {
        return typeid(T) == pimpl_->type_id() ? static_cast<const T*>(pimpl_->address()) : nullptr;
    }

    /**
     * @brief Serialize the wrapped user type.
     *
     * @details Forwards to the serializer's `save_at()` method. If the wrapped user type is not
     * serializable through cp_serializer_type, the call is a silent no-op.
     *
     * @param s Serializer handle.
     * @param key Sub-key to write into.
     */
    void save_at(cp_serializer_type& s, std::string_view key) const { pimpl_->save_at(s, key); }

    /**
     * @brief Deserialize the wrapped user type.
     *
     * @details Forwards to the serializer's `load_at()` method. If the wrapped user type is not
     * deserializable through cp_serializer_type, the call is a silent no-op.
     *
     * @param s Serializer handle.
     * @param key Sub-key to read from.
     */
    void load_at(const cp_serializer_type& s, std::string_view key) { pimpl_->load_at(s, key); }

    /**
     * @brief Serialize the wrapped user type as input config.
     *
     * @details Forwards to the wrapped user type's ADL hook. If the wrapped user type does not
     * provide such a hook, the call is a silent no-op — the entry simply contributes nothing to the
     * input-config output.
     *
     * @param s Serializer handle.
     * @param key Sub-key to write into.
     */
    void save_input_config_at(ic_serializer_type& s, std::string_view key) const {
        pimpl_->save_input_config_at(s, key);
    }

    /**
     * @brief Deserialize the wrapped user type from input config.
     *
     * @details Forwards to the wrapped user type's ADL hook. If the wrapped user type does not
     * provide such a hook, the call is a silent no-op — the entry simply contributes nothing to the
     * input-config output.
     *
     * @param s Serializer handle.
     * @param key Sub-key to read from.
     */
    void load_input_config_at(const ic_serializer_type& s, std::string_view key) {
        pimpl_->load_input_config_at(s, key);
    }

private:
    // Hidden polymorphic base — the "concept" half of the concept-model idiom.
    struct interface {
        virtual ~interface() = default;
        virtual double attempt() = 0;
        virtual void accept() = 0;
        virtual void reject() = 0;
        [[nodiscard]] virtual std::unique_ptr<interface> clone() const = 0;
        [[nodiscard]] virtual const std::type_info& type_id() const noexcept = 0;
        [[nodiscard]] virtual void* address() noexcept = 0;
        [[nodiscard]] virtual const void* address() const noexcept = 0;
        virtual void save_at(cp_serializer_type&, std::string_view) const = 0;
        virtual void load_at(const cp_serializer_type&, std::string_view) = 0;
        virtual void save_input_config_at(ic_serializer_type&, std::string_view) const = 0;
        virtual void load_input_config_at(const ic_serializer_type&, std::string_view) = 0;
    };

    // Hidden template derived - the "model" half of the concept-model idiom.
    // Note: Each wrapped type U gets its own model type.
    template <class U>
    struct model final : interface {
        U concrete;
        explicit model(U u) : concrete { std::move(u) } {}
        double attempt() override { return concrete.attempt(); }
        void accept() override { concrete.accept(); }
        void reject() override {
            if constexpr (requires { concrete.reject(); }) {
                concrete.reject();
            }
        }
        [[nodiscard]] std::unique_ptr<interface> clone() const override { return std::make_unique<model>(*this); }
        [[nodiscard]] const std::type_info& type_id() const noexcept override { return typeid(U); }
        [[nodiscard]] void* address() noexcept override { return &concrete; }
        [[nodiscard]] const void* address() const noexcept override { return &concrete; }
        void save_at(cp_serializer_type& s, std::string_view key) const override {
            if constexpr (requires(cp_serializer_type& p, std::string_view k, const U& v) { p.save_at(k, v); }) {
                s.save_at(key, concrete);
            }
        }
        void load_at(const cp_serializer_type& s, std::string_view key) override {
            if constexpr (requires(const cp_serializer_type& p, std::string_view k, U& v) { p.load_at(k, v); }) {
                s.load_at(key, concrete);
            }
        }
        void save_input_config_at(ic_serializer_type& s, std::string_view key) const override {
            if constexpr (has_simplemc_save_input_config<U, ic_serializer_type>) {
                auto sub = s[key];
                simplemc_save_input_config(sub, concrete);
            }
        }
        void load_input_config_at(const ic_serializer_type& s, std::string_view key) override {
            if constexpr (has_simplemc_load_input_config<U, ic_serializer_type>) {
                const auto sub = s[key];
                simplemc_load_input_config(sub, concrete);
            }
        }
    };

private:
    std::unique_ptr<interface> pimpl_;
};

// Deduction guide: basic_update u { my_update{} } deduces to basic_update<json_serializer>.
template <class U>
    requires mc_update<std::remove_cvref_t<U>> && std::copy_constructible<std::remove_cvref_t<U>>
basic_update(U&&) -> basic_update<>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_BASIC_UPDATE_HPP
