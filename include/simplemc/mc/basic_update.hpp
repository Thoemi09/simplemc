/**
 * @file
 * @brief Type erasure for user-defined updates.
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

// Type-erasure plumbing for simplemc::basic_update.
// Internal: not documented, not part of the public API.
// It follows the concept (interface) - model pattern.
struct update_interface {
    using serializer_type = mc_serializer;

    // Destructor.
    virtual ~update_interface() = default;

    // Value semantics, type information and access.
    [[nodiscard]] virtual std::unique_ptr<update_interface> clone() const = 0;
    [[nodiscard]] virtual const std::type_info& type_id() const noexcept = 0;
    [[nodiscard]] virtual void* address() noexcept = 0;
    [[nodiscard]] virtual const void* address() const noexcept = 0;

    // Serialization and MPI.
    virtual void save_at(serializer_type& parent, std::string_view key) const = 0;
    virtual void load_at(const serializer_type& parent, std::string_view key) = 0;
    virtual void save_input_config_at(serializer_type& parent, std::string_view key) const = 0;
    virtual void load_input_config_at(const serializer_type& parent, std::string_view key) = 0;
    virtual void mpi_collect(const mpi::communicator& comm) = 0;

    // Update specific API.
    virtual double attempt() = 0;
    virtual void accept() = 0;
    virtual void reject() = 0;
};

template <class U>
struct update_model final : update_interface {
    U user_update;

    // Constructor stores the user update by value.
    explicit update_model(U u) : user_update { std::move(u) } {}

    // Value semantics, type information and access.
    [[nodiscard]] std::unique_ptr<update_interface> clone() const override {
        return std::make_unique<update_model>(*this);
    }

    [[nodiscard]] const std::type_info& type_id() const noexcept override { return typeid(U); }
    [[nodiscard]] void* address() noexcept override { return &user_update; }
    [[nodiscard]] const void* address() const noexcept override { return &user_update; }

    // Serialization and MPI (no-ops if the user type does not support the corresponding operation).
    void save_at(serializer_type& s, std::string_view key) const override {
        if constexpr (save_at_all<U, serializer_type>) {
            s.save_at(key, user_update);
        }
    }

    void load_at(const serializer_type& s, std::string_view key) override {
        if constexpr (load_at_all<U, serializer_type>) {
            s.load_at(key, user_update);
        }
    }

    void save_input_config_at(serializer_type& s, std::string_view key) const override {
        if constexpr (has_simplemc_save_input_config<U, serializer_type>) {
            auto sub = s[key];
            simplemc_save_input_config(sub, user_update);
        }
    }

    void load_input_config_at(const serializer_type& s, std::string_view key) override {
        if constexpr (has_simplemc_load_input_config<U, serializer_type>) {
            const auto sub = s[key];
            simplemc_load_input_config(sub, user_update);
        }
    }

    void mpi_collect(const mpi::communicator& comm) override {
        if constexpr (has_simplemc_mpi_collect<U>) {
            user_update = simplemc_mpi_collect(comm, user_update);
        }
    }

    // Update specific API.
    double attempt() override { return user_update.attempt(); }
    void accept() override { user_update.accept(); }

    // reject() is optional on the wrapped type; if absent, this is a no-op.
    void reject() override {
        if constexpr (requires { user_update.reject(); }) {
            user_update.reject();
        }
    }
};

} // namespace simplemc::detail

namespace simplemc {

/**
 * @addtogroup simplemc-mc-entities-updates
 * @{
 */

/**
 * @brief Type erasure class for user-defined updates.
 *
 * @details This class erases the type of any user-defined MC update satisfying simplemc::mc_update.
 * It is a regular value type with deep-copy semantics, so the wrapped user type must be
 * copy-constructible. This is a storage requirement of the wrapper, not part of the update concept.
 *
 * The original user type is accessible via the typed accessor get<T>(), which returns a pointer to
 * the wrapped object when the dynamic type matches `T`, or `nullptr` otherwise. This requires RTTI to
 * be enabled.
 *
 * Serialization and MPI support depend on the underlying user type.
 */
class basic_update {
public:
    /**
     * @brief Serializer type used for both checkpoint and input-config serialization.
     */
    using serializer_type = mc_serializer;

    /**
     * @brief Constructor erases the type of a given user-defined update.
     *
     * @details The user type is passed by forwarding reference, so lvalues are copied and rvalues
     * are moved into the wrapper.
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
     * @brief Copy constructor makes a deep copy of the other update.
     *
     * @param other Update to copy from.
     */
    basic_update(const basic_update& other) : pimpl_ { other.pimpl_->clone() } {}

    /**
     * @brief Default move constructor.
     */
    basic_update(basic_update&&) noexcept = default;

    /**
     * @brief Copy assignment makes a deep copy of the other update via copy-and-swap.
     *
     * @param other Update to copy from.
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
    basic_update& operator=(basic_update&&) noexcept = default;

    /**
     * @brief Default destructor.
     */
    ~basic_update() = default;

    /**
     * @brief Recover a pointer to the wrapped user update.
     *
     * @details It returns a pointer to the original update object if its dynamic type matches `T`,
     * otherwise `nullptr`. Requires RTTI to be enabled.
     *
     * @tparam T Expected type of the wrapped user update.
     * @return Pointer to the wrapped update, or `nullptr` on type mismatch.
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
     * @brief Serialize the wrapped update.
     *
     * @details If the user type does not support serialization through `serializer_type`, it does
     * nothing.
     *
     * @param s Serializer handle.
     * @param key Sub-key to write into.
     */
    void save_at(serializer_type& s, std::string_view key) const { pimpl_->save_at(s, key); }

    /**
     * @brief Deserialize the wrapped update.
     *
     * @details If the user type does not support deserialization through `serializer_type`, it does
     * nothing.
     *
     * @param s Serializer handle.
     * @param key Sub-key to read from.
     */
    void load_at(const serializer_type& s, std::string_view key) { pimpl_->load_at(s, key); }

    /**
     * @brief Serialize the user-input config of the wrapped update.
     *
     * @details If the user type does not support input-config serialization through
     * `serializer_type`, it does nothing.
     *
     * @param s Serializer handle.
     * @param key Sub-key to write into.
     */
    void save_input_config_at(serializer_type& s, std::string_view key) const { pimpl_->save_input_config_at(s, key); }

    /**
     * @brief Deserialize the user-input config of the wrapped update.
     *
     * @details If the user type does not support input-config deserialization through
     * `serializer_type`, it does nothing.
     *
     * @param s Serializer handle.
     * @param key Sub-key to read from.
     */
    void load_input_config_at(const serializer_type& s, std::string_view key) { pimpl_->load_input_config_at(s, key); }

    /**
     * @brief Collect updates from different MPI processes.
     *
     * @details If the user type does not support cross-rank reduction via a call to
     * `%simplemc_mpi_collect`, it does nothing.
     *
     * @param comm simplemc::mpi::communicator object.
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
     * @details If the wrapped user type defines a `%reject()` member it is invoked. Otherwise this
     * is a no-op.
     */
    void reject() { pimpl_->reject(); }

private:
    std::unique_ptr<detail::update_interface> pimpl_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_BASIC_UPDATE_HPP
