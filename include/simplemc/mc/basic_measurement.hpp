/**
 * @file
 * @brief Type erasure for user-defined measurements.
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

// Type-erasure plumbing for simplemc::basic_measurement.
// Internal: not documented, not part of the public API.
// It follows the concept (interface) - model pattern.
struct measurement_interface {
    using serializer_type = mc_serializer;

    // Destructor.
    virtual ~measurement_interface() = default;

    // Value semantics, type information and access.
    [[nodiscard]] virtual std::unique_ptr<measurement_interface> clone() const = 0;
    [[nodiscard]] virtual const std::type_info& type_id() const noexcept = 0;
    [[nodiscard]] virtual void* address() noexcept = 0;
    [[nodiscard]] virtual const void* address() const noexcept = 0;

    // Serialization and MPI.
    virtual void save_at(serializer_type& parent, std::string_view key) const = 0;
    virtual void load_at(const serializer_type& parent, std::string_view key) = 0;
    virtual void save_input_config_at(serializer_type& parent, std::string_view key) const = 0;
    virtual void load_input_config_at(const serializer_type& parent, std::string_view key) = 0;
    virtual void mpi_collect(const mpi::communicator& comm) = 0;

    // Measurement specific API.
    virtual void measure() = 0;
};

template <class M>
struct measurement_model final : measurement_interface {
    M user_measurement;

    // Constructor stores the user measurement by value.
    explicit measurement_model(M m) : user_measurement { std::move(m) } {}

    // Value semantics, type information and access.
    [[nodiscard]] std::unique_ptr<measurement_interface> clone() const override {
        return std::make_unique<measurement_model>(*this);
    }

    [[nodiscard]] const std::type_info& type_id() const noexcept override { return typeid(M); }
    [[nodiscard]] void* address() noexcept override { return &user_measurement; }
    [[nodiscard]] const void* address() const noexcept override { return &user_measurement; }

    // Serialization and MPI (no-ops if the user type does not support the corresponding operation).
    void save_at(serializer_type& s, std::string_view key) const override {
        if constexpr (save_at_all<M, serializer_type>) {
            s.save_at(key, user_measurement);
        }
    }

    void load_at(const serializer_type& s, std::string_view key) override {
        if constexpr (load_at_all<M, serializer_type>) {
            s.load_at(key, user_measurement);
        }
    }

    void save_input_config_at(serializer_type& s, std::string_view key) const override {
        if constexpr (has_simplemc_save_input_config<M, serializer_type>) {
            auto sub = s[key];
            simplemc_save_input_config(sub, user_measurement);
        }
    }

    void load_input_config_at(const serializer_type& s, std::string_view key) override {
        if constexpr (has_simplemc_load_input_config<M, serializer_type>) {
            const auto sub = s[key];
            simplemc_load_input_config(sub, user_measurement);
        }
    }

    void mpi_collect(const mpi::communicator& comm) override {
        if constexpr (has_simplemc_mpi_collect<M>) {
            user_measurement = simplemc_mpi_collect(comm, user_measurement);
        }
    }

    // Measurement specific API.
    void measure() override { user_measurement.measure(); }
};

} // namespace simplemc::detail

namespace simplemc {

/**
 * @addtogroup simplemc-mc-entities-measurements
 * @{
 */

/**
 * @brief Type erasure class for user-defined measurements.
 *
 * @details This class erases the type of any user-defined MC measurement satisfying
 * simplemc::mc_measurement. It is a regular value type with deep-copy semantics, so the wrapped user
 * type must be copy-constructible. This is a storage requirement of the wrapper, not part of the
 * measurement concept.
 *
 * The original user type is accessible via the typed accessor get<T>(), which returns a pointer to
 * the wrapped object when the dynamic type matches `T`, or `nullptr` otherwise. This requires RTTI to
 * be enabled.
 *
 * Serialization and MPI support depend on the underlying user type.
 */
class basic_measurement {
public:
    /**
     * @brief Serializer type used for both checkpoint and input-config serialization.
     */
    using serializer_type = mc_serializer;

    /**
     * @brief Constructor erases the type of a given user-defined measurement.
     *
     * @details The user type is passed by forwarding reference, so lvalues are copied and rvalues
     * are moved into the wrapper.
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
     * @brief Copy constructor makes a deep copy of the other measurement.
     *
     * @param other Measurement to copy from.
     */
    basic_measurement(const basic_measurement& other) : pimpl_ { other.pimpl_->clone() } {}

    /**
     * @brief Default move constructor.
     */
    basic_measurement(basic_measurement&&) noexcept = default;

    /**
     * @brief Copy assignment makes a deep copy of the other measurement via copy-and-swap.
     *
     * @param other Measurement to copy from.
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
    basic_measurement& operator=(basic_measurement&&) noexcept = default;

    /**
     * @brief Default destructor.
     */
    ~basic_measurement() = default;

    /**
     * @brief Recover a pointer to the wrapped user measurement.
     *
     * @details It returns a pointer to the original measurement object if its dynamic type matches
     * `T`, otherwise `nullptr`. Requires RTTI to be enabled.
     *
     * @tparam T Expected type of the wrapped user measurement.
     * @return Pointer to the wrapped measurement, or `nullptr` on type mismatch.
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
     * @brief Serialize the wrapped measurement.
     *
     * @details If the user type does not support serialization through `serializer_type`, it does
     * nothing.
     *
     * @param s Serializer handle.
     * @param key Sub-key to write into.
     */
    void save_at(serializer_type& s, std::string_view key) const { pimpl_->save_at(s, key); }

    /**
     * @brief Deserialize the wrapped measurement.
     *
     * @details If the user type does not support deserialization through `serializer_type`, it does
     * nothing.
     *
     * @param s Serializer handle.
     * @param key Sub-key to read from.
     */
    void load_at(const serializer_type& s, std::string_view key) { pimpl_->load_at(s, key); }

    /**
     * @brief Serialize the user-input config of the wrapped measurement.
     *
     * @details If the user type does not support input-config serialization through
     * `serializer_type`, it does nothing.
     *
     * @param s Serializer handle.
     * @param key Sub-key to write into.
     */
    void save_input_config_at(serializer_type& s, std::string_view key) const { pimpl_->save_input_config_at(s, key); }

    /**
     * @brief Deserialize the user-input config of the wrapped measurement.
     *
     * @details If the user type does not support input-config deserialization through
     * `serializer_type`, it does nothing.
     *
     * @param s Serializer handle.
     * @param key Sub-key to read from.
     */
    void load_input_config_at(const serializer_type& s, std::string_view key) { pimpl_->load_input_config_at(s, key); }

    /**
     * @brief Collect measurements from different MPI processes.
     *
     * @details If the user type does not support cross-rank reduction via a call to
     * `%simplemc_mpi_collect`, it does nothing.
     *
     * @param comm simplemc::mpi::communicator object.
     */
    void mpi_collect(const mpi::communicator& comm) { pimpl_->mpi_collect(comm); }

    /**
     * @brief Perform the measurement by calling the `%measure()` member of the wrapped user type.
     */
    void measure() { pimpl_->measure(); }

private:
    std::unique_ptr<detail::measurement_interface> pimpl_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_BASIC_MEASUREMENT_HPP
