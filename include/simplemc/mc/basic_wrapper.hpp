/**
 * @file
 * @brief Reusable concept-model value-semantics machinery for the simplemc-mc type-erased wrappers.
 */

#ifndef SIMPLEMC_MC_BASIC_WRAPPER_HPP
#define SIMPLEMC_MC_BASIC_WRAPPER_HPP

#include <simplemc/mc/concepts.hpp>
#include <simplemc/mpi/communicator.hpp>

#include <memory>
#include <string_view>
#include <typeinfo>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-entities
 * @{
 */

/**
 * @brief Common pure-virtual half of the concept-model idiom shared by the simplemc-mc wrappers.
 *
 * @details This is the role-agnostic interface of the concept-model type-erasure idiom (see @ref
 * simplemc-mc). It declares every virtual that does not depend on the wrapped type's role: deep
 * cloning, RTTI access for the typed `get<T>()` accessor, checkpoint / input-config serialization
 * forwarding, and MPI reduction. A concrete wrapper defines a role interface that derives from this
 * and adds the role's own pure virtuals (e.g. the update interface adds `attempt()` / `accept()` /
 * `reject()`).
 *
 * The `Final` template parameter is the most derived (role) interface type, threaded through so
 * clone() returns a `std::unique_ptr<Final>` rather than a pointer to this base — letting the owning
 * simplemc::basic_wrapper hold the role interface directly.
 *
 * @tparam Traits Traits bundle satisfying simplemc::mc_traits_like.
 * @tparam Final  The most derived (role) interface type, used as the clone() return type.
 */
template <mc_traits_like Traits, class Final>
struct basic_interface {
    /**
     * @brief Type used for checkpoint serialization.
     */
    using checkpoint_serializer_type = typename Traits::checkpoint_serializer_type;

    /**
     * @brief Type used for input-config serialization.
     */
    using input_config_serializer_type = typename Traits::input_config_serializer_type;

    /**
     * @brief Virtual destructor.
     */
    virtual ~basic_interface() = default;

    /**
     * @brief Deep-copy the held value into a new model.
     *
     * @return Owning pointer to an independent clone of the most derived interface type.
     */
    [[nodiscard]] virtual std::unique_ptr<Final> clone() const = 0;

    /**
     * @brief RTTI of the wrapped concrete type, used by the typed `get<T>()` accessor.
     *
     * @return `typeid` of the wrapped value.
     */
    [[nodiscard]] virtual const std::type_info& type_id() const noexcept = 0;

    /**
     * @brief Address of the wrapped value, for the typed `get<T>()` accessor.
     *
     * @return Pointer to the wrapped value, type-erased as `void*`.
     */
    [[nodiscard]] virtual void* address() noexcept = 0;

    /**
     * @brief Const overload of address().
     *
     * @return Const pointer to the wrapped value, type-erased as `const void*`.
     */
    [[nodiscard]] virtual const void* address() const noexcept = 0;

    /**
     * @brief Serialize the wrapped value through the checkpoint serializer (no-op if unsupported).
     *
     * @param parent Serializer handle.
     * @param key Sub-key to write into.
     */
    virtual void save_at(checkpoint_serializer_type& parent, std::string_view key) const = 0;

    /**
     * @brief Deserialize the wrapped value through the checkpoint serializer (no-op if unsupported).
     *
     * @param parent Serializer handle.
     * @param key Sub-key to read from.
     */
    virtual void load_at(const checkpoint_serializer_type& parent, std::string_view key) = 0;

    /**
     * @brief Serialize the wrapped value as input config via its ADL hook (no-op if unsupported).
     *
     * @param parent Serializer handle.
     * @param key Sub-key to write into.
     */
    virtual void save_input_config_at(input_config_serializer_type& parent, std::string_view key) const = 0;

    /**
     * @brief Deserialize the wrapped value from input config via its ADL hook (no-op if unsupported).
     *
     * @param parent Serializer handle.
     * @param key Sub-key to read from.
     */
    virtual void load_input_config_at(const input_config_serializer_type& parent, std::string_view key) = 0;

    /**
     * @brief All-reduce the wrapped value across MPI ranks via its ADL hook (no-op if unsupported).
     *
     * @param comm MPI communicator over which to reduce.
     */
    virtual void mpi_collect(const mpi::communicator& comm) = 0;
};

/**
 * @brief Common model half of the concept-model idiom: implements every role-agnostic virtual.
 *
 * @details Holds the wrapped user value by value and implements the role-agnostic virtuals declared
 * by simplemc::basic_interface with compile-time opt-in dispatch: when the wrapped type does not
 * provide the corresponding hook (checkpoint save/load, input-config save/load, or
 * `simplemc_mpi_collect`), the override is a silent no-op. A concrete wrapper's role model derives
 * from this and adds only the role overrides plus clone() (which must construct the most derived
 * model type so the dynamic type is preserved across copies).
 *
 * @tparam Traits Traits bundle satisfying simplemc::mc_traits_like.
 * @tparam Iface  Role interface type (derived from simplemc::basic_interface).
 * @tparam T      Wrapped user type.
 */
template <mc_traits_like Traits, class Iface, class T>
struct basic_model : Iface {
    /**
     * @brief Type used for checkpoint serialization.
     */
    using checkpoint_serializer_type = typename Traits::checkpoint_serializer_type;

    /**
     * @brief Type used for input-config serialization.
     */
    using input_config_serializer_type = typename Traits::input_config_serializer_type;

    /**
     * @brief The wrapped user value.
     */
    T concrete;

    /**
     * @brief Construct by taking ownership of a user value.
     *
     * @param v User value to store (moved in).
     */
    explicit basic_model(T v) : concrete { std::move(v) } {}

    /**
     * @brief RTTI of the wrapped type.
     */
    [[nodiscard]] const std::type_info& type_id() const noexcept override { return typeid(T); }

    /**
     * @brief Address of the wrapped value.
     */
    [[nodiscard]] void* address() noexcept override { return &concrete; }

    /**
     * @brief Const overload of address().
     */
    [[nodiscard]] const void* address() const noexcept override { return &concrete; }

    /**
     * @brief Serialize the wrapped value if `checkpoint_serializer_type` supports it; else no-op.
     */
    void save_at(checkpoint_serializer_type& s, std::string_view key) const override {
        if constexpr (save_at_compatible<checkpoint_serializer_type, T>) {
            s.save_at(key, concrete);
        }
    }

    /**
     * @brief Deserialize the wrapped value if `checkpoint_serializer_type` supports it; else no-op.
     */
    void load_at(const checkpoint_serializer_type& s, std::string_view key) override {
        if constexpr (load_at_compatible<checkpoint_serializer_type, T>) {
            s.load_at(key, concrete);
        }
    }

    /**
     * @brief Serialize the wrapped value as input config via ADL if supported; else no-op.
     */
    void save_input_config_at(input_config_serializer_type& s, std::string_view key) const override {
        if constexpr (has_simplemc_save_input_config<T, input_config_serializer_type>) {
            auto sub = s[key];
            simplemc_save_input_config(sub, concrete);
        }
    }

    /**
     * @brief Deserialize the wrapped value from input config via ADL if supported; else no-op.
     */
    void load_input_config_at(const input_config_serializer_type& s, std::string_view key) override {
        if constexpr (has_simplemc_load_input_config<T, input_config_serializer_type>) {
            const auto sub = s[key];
            simplemc_load_input_config(sub, concrete);
        }
    }

    /**
     * @brief All-reduce the wrapped value via ADL `simplemc_mpi_collect` if supported; else no-op.
     */
    void mpi_collect(const mpi::communicator& comm) override {
        if constexpr (has_simplemc_mpi_collect<T>) {
            concrete = simplemc_mpi_collect(comm, concrete);
        }
    }
};

/**
 * @brief Value-semantics shell shared by the simplemc-mc type-erased wrappers.
 *
 * @details Owns the type-erased model (a `std::unique_ptr<Iface>`) and implements the parts of a
 * regular value type that do not depend on the wrapped type's role: deep-copy / cheap-move
 * semantics, the typed `get<T>()` accessor, and the role-agnostic forwarders (checkpoint /
 * input-config serialization, MPI reduction). A concrete wrapper (simplemc::basic_update,
 * simplemc::basic_measurement) derives from this and adds only its role forwarders plus the
 * constrained wrapping constructor, which builds a role model and hands it to the protected
 * constructor here.
 *
 * To build a custom wrapper on top of this machinery: (1) define a role interface deriving from
 * simplemc::basic_interface with the role's pure virtuals; (2) define a role model deriving from
 * simplemc::basic_model implementing them plus clone(); (3) derive a wrapper from
 * `basic_wrapper<Traits, RoleInterface>` exposing a wrapping constructor and the role forwarders.
 * See simplemc::basic_update for a worked example.
 *
 * @tparam Traits Traits bundle satisfying simplemc::mc_traits_like.
 * @tparam Iface  Role interface type (derived from simplemc::basic_interface).
 */
template <mc_traits_like Traits, class Iface>
class basic_wrapper {
public:
    /**
     * @brief Type used for checkpoint serialization.
     */
    using checkpoint_serializer_type = typename Traits::checkpoint_serializer_type;

    /**
     * @brief Type used for input-config serialization.
     */
    using input_config_serializer_type = typename Traits::input_config_serializer_type;

    /**
     * @brief The traits bundle this wrapper was parameterized on.
     */
    using traits_type = Traits;

    /**
     * @brief Copy constructor.
     *
     * @details Performs a deep copy: the new wrapper owns an independent copy of `other`'s wrapped
     * value, produced by the wrapped type's copy constructor. The two wrappers do not share state.
     *
     * @param other Wrapper to copy from.
     */
    basic_wrapper(const basic_wrapper& other) : pimpl_ { other.pimpl_->clone() } {}

    /**
     * @brief Default move constructor.
     */
    basic_wrapper(basic_wrapper&& other) noexcept = default;

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
    basic_wrapper& operator=(const basic_wrapper& other) {
        basic_wrapper tmp { other };
        std::swap(pimpl_, tmp.pimpl_);
        return *this;
    }

    /**
     * @brief Default move assignment.
     */
    basic_wrapper& operator=(basic_wrapper&& other) noexcept = default;

    /**
     * @brief Default destructor.
     */
    ~basic_wrapper() = default;

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
     * serializable through checkpoint_serializer_type, the call is a silent no-op.
     *
     * @param s Serializer handle.
     * @param key Sub-key to write into.
     */
    void save_at(checkpoint_serializer_type& s, std::string_view key) const { pimpl_->save_at(s, key); }

    /**
     * @brief Deserialize the wrapped user type.
     *
     * @details Forwards to the serializer's `load_at()` method. If the wrapped user type is not
     * deserializable through checkpoint_serializer_type, the call is a silent no-op.
     *
     * @param s Serializer handle.
     * @param key Sub-key to read from.
     */
    void load_at(const checkpoint_serializer_type& s, std::string_view key) { pimpl_->load_at(s, key); }

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
    void save_input_config_at(input_config_serializer_type& s, std::string_view key) const {
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
    void load_input_config_at(const input_config_serializer_type& s, std::string_view key) {
        pimpl_->load_input_config_at(s, key);
    }

    /**
     * @brief All-reduce the wrapped user value across MPI ranks via ADL.
     *
     * @details Forwards to the wrapped user type's free
     * `simplemc_mpi_collect(const mpi::communicator&, T&)` picked up by ADL, replacing the held
     * value with the reduced result. If the wrapped type does not provide such an overload, the
     * call is a silent no-op — mirroring the behavior of save_at() / load_at() for non-serializable
     * types.
     *
     * @param comm MPI communicator over which to reduce.
     */
    void mpi_collect(const mpi::communicator& comm) { pimpl_->mpi_collect(comm); }

protected:
    /**
     * @brief Adopt a freshly built role model. Called by the role wrapper's wrapping constructor.
     *
     * @param p Owning pointer to the role model to take ownership of.
     */
    explicit basic_wrapper(std::unique_ptr<Iface> p) noexcept : pimpl_ { std::move(p) } {}

    /**
     * @brief Access the role interface, for the role wrapper's forwarders.
     *
     * @return Reference to the owned role interface.
     */
    [[nodiscard]] Iface& impl() noexcept { return *pimpl_; }

    /**
     * @brief Const overload of impl().
     *
     * @return Const reference to the owned role interface.
     */
    [[nodiscard]] const Iface& impl() const noexcept { return *pimpl_; }

private:
    /// Owning pointer to the type-erased model.
    std::unique_ptr<Iface> pimpl_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_BASIC_WRAPPER_HPP
