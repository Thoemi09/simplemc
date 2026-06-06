/**
 * @file
 * @brief Polymorphic value type holding a Monte Carlo update.
 */

#ifndef SIMPLEMC_MC_UPDATE_HPP
#define SIMPLEMC_MC_UPDATE_HPP

#include <simplemc/mc/concepts.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>

#include <concepts>
#include <memory>
#include <string_view>
#include <type_traits>
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
 * simplemc::update u{ my_update{} };
 * @endcode
 *
 * simplemc::update is a regular value type: copies are deep, moves are cheap, and it can be stored
 * directly in containers such as `std::vector<%update>` to hold a heterogeneous collection of
 * updates. Because copies are deep, the wrapped user type must be copy-constructible; this is checked
 * at construction time.
 *
 * The wrapper is templated on the serializer type `S` used for state persistence. If the wrapped user
 * type provides a serialization path through `S`, the wrapper's save_at() / load_at() forward to it.
 * Otherwise they are silent no-ops, which lets stateless updates be added without writing trivial
 * empty overloads.
 *
 * See @ref simplemc-mc for a description of the implementation strategy.
 *
 * @tparam S Serializer type.
 */
template <serializer S = json_serializer>
class update {
public:
    /**
     * @brief Wrap a user-defined update.
     *
     * @details The user type is passed by forwarding reference, so lvalues are copied and rvalues
     * are moved into the wrapper.
     *
     * The wrapped type must also be copy-constructible: simplemc::update is a value type with
     * deep-copy semantics and the internal model holds an instance by value. This is a storage
     * requirement of the wrapper, not part of the simplemc::mc_update role.
     *
     * @tparam U Type satisfying simplemc::mc_update and `std::copy_constructible`.
     * @param u Update to wrap.
     */
    template <class U>
        requires(!std::same_as<std::remove_cvref_t<U>, update>) && mc_update<std::remove_cvref_t<U>> &&
        std::copy_constructible<std::remove_cvref_t<U>>
    update(U&& u) : pimpl_ { std::make_unique<model<std::remove_cvref_t<U>>>(std::forward<U>(u)) } {}

    /**
     * @brief Copy constructor.
     *
     * @details Performs a deep copy: the new wrapper owns an independent copy of `other`'s wrapped
     * value, produced by the wrapped type's copy constructor. The two wrappers do not share state.
     *
     * @param other Wrapper to copy from.
     */
    update(const update& other) : pimpl_ { other.pimpl_->clone() } {}

    /**
     * @brief Default move constructor.
     */
    update(update&& other) noexcept = default;

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
    update& operator=(const update& other) {
        update tmp { other };
        std::swap(pimpl_, tmp.pimpl_);
        return *this;
    }

    /**
     * @brief Default move assignment.
     */
    update& operator=(update&& other) noexcept = default;

    /**
     * @brief Default destructor.
     */
    ~update() = default;

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
     * @brief Serialize the wrapped user state.
     *
     * @details Forwards to the serializer's `save_at()` method. If the wrapped user type is not
     * serializable through `S`, the call is a silent no-op.
     *
     * @param s Serializer handle one level above where the user state should be written.
     * @param key Sub-key to write into.
     */
    void save_at(S& s, std::string_view key) const { pimpl_->save_at(s, key); }

    /**
     * @brief Deserialize the wrapped user state.
     *
     * @details Forwards to the serializer's `load_at()` method. If the wrapped user type is not
     * deserializable through `S`, the call is a silent no-op.
     *
     * @param s Serializer handle one level above where the user state should be read from.
     * @param key Sub-key to read from.
     */
    void load_at(const S& s, std::string_view key) { pimpl_->load_at(s, key); }

private:
    // Hidden polymorphic base — the "concept" half of the concept-model idiom.
    struct interface {
        virtual ~interface() = default;
        virtual double attempt() = 0;
        virtual void accept() = 0;
        virtual void reject() = 0;
        [[nodiscard]] virtual std::unique_ptr<interface> clone() const = 0;
        virtual void save_at(S&, std::string_view) const = 0;
        virtual void load_at(const S&, std::string_view) = 0;
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
        void save_at(S& s, std::string_view key) const override {
            if constexpr (requires(S& p, std::string_view k, const U& v) { p.save_at(k, v); }) {
                s.save_at(key, concrete);
            }
            // else: no serialization path for U through S — silent no-op
        }
        void load_at(const S& s, std::string_view key) override {
            if constexpr (requires(const S& p, std::string_view k, U& v) { p.load_at(k, v); }) {
                s.load_at(key, concrete);
            }
        }
    };

private:
    std::unique_ptr<interface> pimpl_;
};

// Deduction guide: update u { my_update{} } deduces to update<json_serializer>.
template <class U>
    requires mc_update<std::remove_cvref_t<U>> && std::copy_constructible<std::remove_cvref_t<U>>
update(U&&) -> update<>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_UPDATE_HPP
