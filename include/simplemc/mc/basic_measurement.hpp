/**
 * @file
 * @brief Polymorphic value type holding a Monte Carlo measurement.
 */

#ifndef SIMPLEMC_MC_BASIC_MEASUREMENT_HPP
#define SIMPLEMC_MC_BASIC_MEASUREMENT_HPP

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

// Type-erasure plumbing for simplemc::basic_measurement. Internal: not documented, not part of the
// public API. Mirrors the update interface/model pair (the role-agnostic overrides are duplicated
// rather than shared through a common template base); the only role operation is measure().

struct measurement_interface {
    using serializer_type = mc_serializer;

    virtual ~measurement_interface() = default;

    [[nodiscard]] virtual std::unique_ptr<measurement_interface> clone() const = 0;
    [[nodiscard]] virtual const std::type_info& type_id() const noexcept = 0;
    [[nodiscard]] virtual void* address() noexcept = 0;
    [[nodiscard]] virtual const void* address() const noexcept = 0;

    virtual void save_at(serializer_type& parent, std::string_view key) const = 0;
    virtual void load_at(const serializer_type& parent, std::string_view key) = 0;
    virtual void save_input_config_at(serializer_type& parent, std::string_view key) const = 0;
    virtual void load_input_config_at(const serializer_type& parent, std::string_view key) = 0;
    virtual void mpi_collect(const mpi::communicator& comm) = 0;

    // measurement role
    virtual void measure() = 0;
};

template <class M>
struct measurement_model final : measurement_interface {
    M concrete;

    explicit measurement_model(M v) : concrete { std::move(v) } {}

    [[nodiscard]] const std::type_info& type_id() const noexcept override { return typeid(M); }
    [[nodiscard]] void* address() noexcept override { return &concrete; }
    [[nodiscard]] const void* address() const noexcept override { return &concrete; }

    // Each hook compiles to a no-op when the wrapped type does not support it.
    void save_at(serializer_type& s, std::string_view key) const override {
        if constexpr (save_at_compatible<M, serializer_type>) {
            s.save_at(key, concrete);
        }
    }

    void load_at(const serializer_type& s, std::string_view key) override {
        if constexpr (load_at_compatible<M, serializer_type>) {
            s.load_at(key, concrete);
        }
    }

    void save_input_config_at(serializer_type& s, std::string_view key) const override {
        if constexpr (has_simplemc_save_input_config<M, serializer_type>) {
            auto sub = s[key];
            simplemc_save_input_config(sub, concrete);
        }
    }

    void load_input_config_at(const serializer_type& s, std::string_view key) override {
        if constexpr (has_simplemc_load_input_config<M, serializer_type>) {
            const auto sub = s[key];
            simplemc_load_input_config(sub, concrete);
        }
    }

    void mpi_collect(const mpi::communicator& comm) override {
        if constexpr (has_simplemc_mpi_collect<M>) {
            concrete = simplemc_mpi_collect(comm, concrete);
        }
    }

    void measure() override { concrete.measure(); }

    [[nodiscard]] std::unique_ptr<measurement_interface> clone() const override {
        return std::make_unique<measurement_model>(*this);
    }
};

} // namespace simplemc::detail

namespace simplemc {

/**
 * @addtogroup simplemc-mc-entities-measurements
 * @{
 */

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
 * The type erasure is implemented inline over an internal interface/model pair; the same pattern is
 * duplicated (not shared) by simplemc::basic_update. See @ref simplemc-mc for a description of the
 * implementation strategy.
 *
 * @note The wrapper takes the user measurement by value. To access the concrete value (e.g. an
 * accumulator) after a run, use the typed accessor get<T>() — it returns a pointer to the wrapped
 * value when the dynamic type matches `T`, or `nullptr` otherwise.
 */
class basic_measurement {
public:
    /**
     * @brief Serializer type used for both checkpoint and input-config serialization.
     */
    using serializer_type = mc_serializer;

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
        pimpl_ { std::make_unique<detail::measurement_model<std::remove_cvref_t<M>>>(std::forward<M>(m)) } {}

    /**
     * @brief Copy constructor (deep copy of the wrapped value).
     *
     * @param other Wrapper to copy from.
     */
    basic_measurement(const basic_measurement& other) : pimpl_ { other.pimpl_->clone() } {}

    /**
     * @brief Default move constructor.
     */
    basic_measurement(basic_measurement&& other) noexcept = default;

    /**
     * @brief Copy assignment (deep copy via copy-and-swap).
     *
     * @param other Wrapper to copy from.
     * @return Reference to `*this`.
     */
    basic_measurement& operator=(const basic_measurement& other) {
        basic_measurement tmp { other };
        std::swap(pimpl_, tmp.pimpl_);
        return *this;
    }

    /**
     * @brief Default move assignment.
     */
    basic_measurement& operator=(basic_measurement&& other) noexcept = default;

    /**
     * @brief Default destructor.
     */
    ~basic_measurement() = default;

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
     * @brief Take a measurement.
     */
    void measure() { pimpl_->measure(); }

private:
    /// Owning pointer to the type-erased model.
    std::unique_ptr<detail::measurement_interface> pimpl_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_BASIC_MEASUREMENT_HPP
