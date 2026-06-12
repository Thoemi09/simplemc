/**
 * @file
 * @brief Polymorphic value type holding a Monte Carlo update.
 */

#ifndef SIMPLEMC_MC_BASIC_UPDATE_HPP
#define SIMPLEMC_MC_BASIC_UPDATE_HPP

#include <simplemc/mc/concepts.hpp>
#include <simplemc/mc/serializer.hpp>
#include <simplemc/mpi/communicator.hpp>

#include <concepts>
#include <memory>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace simplemc::detail {

// Type-erasure plumbing for simplemc::basic_update. Internal: not documented, not part of the
// public API. The interface declares every operation the wrapper forwards; update_model<U>
// implements them over a wrapped user value U. The role-agnostic overrides (RTTI access,
// serialization, MPI) are intentionally duplicated with measurement_model rather than shared
// through a common template base.

struct update_interface {
    using serializer_type = mc_serializer;

    virtual ~update_interface() = default;

    [[nodiscard]] virtual std::unique_ptr<update_interface> clone() const = 0;
    [[nodiscard]] virtual const std::type_info& type_id() const noexcept = 0;
    [[nodiscard]] virtual void* address() noexcept = 0;
    [[nodiscard]] virtual const void* address() const noexcept = 0;

    virtual void save_at(serializer_type& parent, std::string_view key) const = 0;
    virtual void load_at(const serializer_type& parent, std::string_view key) = 0;
    virtual void save_input_config_at(serializer_type& parent, std::string_view key) const = 0;
    virtual void load_input_config_at(const serializer_type& parent, std::string_view key) = 0;
    virtual void mpi_collect(const mpi::communicator& comm) = 0;

    // update role
    virtual double attempt() = 0;
    virtual void accept() = 0;
    virtual void reject() = 0;
};

template <class U>
struct update_model final : update_interface {
    U concrete;

    explicit update_model(U v) : concrete { std::move(v) } {}

    [[nodiscard]] const std::type_info& type_id() const noexcept override { return typeid(U); }
    [[nodiscard]] void* address() noexcept override { return &concrete; }
    [[nodiscard]] const void* address() const noexcept override { return &concrete; }

    // Each hook compiles to a no-op when the wrapped type does not support it.
    void save_at(serializer_type& s, std::string_view key) const override {
        if constexpr (save_at_compatible<U, serializer_type>) {
            s.save_at(key, concrete);
        }
    }

    void load_at(const serializer_type& s, std::string_view key) override {
        if constexpr (load_at_compatible<U, serializer_type>) {
            s.load_at(key, concrete);
        }
    }

    void save_input_config_at(serializer_type& s, std::string_view key) const override {
        if constexpr (has_simplemc_save_input_config<U, serializer_type>) {
            auto sub = s[key];
            simplemc_save_input_config(sub, concrete);
        }
    }

    void load_input_config_at(const serializer_type& s, std::string_view key) override {
        if constexpr (has_simplemc_load_input_config<U, serializer_type>) {
            const auto sub = s[key];
            simplemc_load_input_config(sub, concrete);
        }
    }

    void mpi_collect(const mpi::communicator& comm) override {
        if constexpr (has_simplemc_mpi_collect<U>) {
            concrete = simplemc_mpi_collect(comm, concrete);
        }
    }

    double attempt() override { return concrete.attempt(); }

    void accept() override { concrete.accept(); }

    // reject() is optional on the wrapped type; absent it, this is a no-op.
    void reject() override {
        if constexpr (requires { concrete.reject(); }) {
            concrete.reject();
        }
    }

    [[nodiscard]] std::unique_ptr<update_interface> clone() const override {
        return std::make_unique<update_model>(*this);
    }
};

} // namespace simplemc::detail

namespace simplemc {

/**
 * @addtogroup simplemc-mc-entities-updates
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
 * Serialization runs through the library-wide simplemc::mc_serializer (a backend-erasing
 * simplemc::variant_serializer), used for both the checkpoint and the input-config channels:
 * - If the wrapped user type provides a serialization path through the serializer's
 * `%save_at()` / `%load_at()`, the wrapper's save_at() / load_at() forward to it.
 * - If the wrapped user type satisfies simplemc::has_simplemc_save_input_config with the serializer,
 * the wrapper's save_input_config_at() / load_input_config_at() forward to the user type's ADL hooks.
 *
 * Otherwise they are silent no-ops.
 *
 * @note Because simplemc::mc_serializer is a variant over the shipped backends, its capability is
 * the *intersection* of those backends: with HDF5 enabled a wrapped payload must be serializable by
 * both JSON and HDF5 for the forwards above to engage.
 *
 * The type erasure is implemented inline over an internal interface/model pair; the same pattern is
 * duplicated (not shared) by simplemc::basic_measurement. See @ref simplemc-mc for a description of
 * the implementation strategy.
 */
class basic_update {
public:
    /**
     * @brief Serializer type used for both checkpoint and input-config serialization.
     */
    using serializer_type = mc_serializer;

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
        pimpl_ { std::make_unique<detail::update_model<std::remove_cvref_t<U>>>(std::forward<U>(u)) } {}

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
     * @brief Copy assignment (deep copy via copy-and-swap).
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
     * @brief Recover a pointer to the wrapped user value, checked by type.
     *
     * @details Returns a pointer to the wrapped value if the dynamic type matches `T`, otherwise
     * `nullptr`. Requires RTTI to be enabled.
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
     * @brief Serialize the wrapped user type (silent no-op if unsupported).
     *
     * @param s Serializer handle.
     * @param key Sub-key to write into.
     */
    void save_at(serializer_type& s, std::string_view key) const { pimpl_->save_at(s, key); }

    /**
     * @brief Deserialize the wrapped user type (silent no-op if unsupported).
     *
     * @param s Serializer handle.
     * @param key Sub-key to read from.
     */
    void load_at(const serializer_type& s, std::string_view key) { pimpl_->load_at(s, key); }

    /**
     * @brief Serialize the wrapped user type as input config (silent no-op if unsupported).
     *
     * @param s Serializer handle.
     * @param key Sub-key to write into.
     */
    void save_input_config_at(serializer_type& s, std::string_view key) const {
        pimpl_->save_input_config_at(s, key);
    }

    /**
     * @brief Deserialize the wrapped user type from input config (silent no-op if unsupported).
     *
     * @param s Serializer handle.
     * @param key Sub-key to read from.
     */
    void load_input_config_at(const serializer_type& s, std::string_view key) {
        pimpl_->load_input_config_at(s, key);
    }

    /**
     * @brief All-reduce the wrapped user value across MPI ranks via ADL (silent no-op if unsupported).
     *
     * @param comm MPI communicator over which to reduce.
     */
    void mpi_collect(const mpi::communicator& comm) { pimpl_->mpi_collect(comm); }

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

private:
    /// Owning pointer to the type-erased model.
    std::unique_ptr<detail::update_interface> pimpl_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_BASIC_UPDATE_HPP
