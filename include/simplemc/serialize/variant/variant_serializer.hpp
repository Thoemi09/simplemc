/**
 * @file
 * @brief Backend-erasing serializer that dispatches to one of several serializer backends at runtime.
 */

#ifndef SIMPLEMC_SERIALIZE_VARIANT_VARIANT_SERIALIZER_HPP
#define SIMPLEMC_SERIALIZE_VARIANT_VARIANT_SERIALIZER_HPP

#include <simplemc/serialize/concepts.hpp>

#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

namespace simplemc {

/**
 * @addtogroup simplemc-serialize-variant
 * @{
 */

/**
 * @brief Serializer that erases its backend behind a `std::variant`.
 *
 * @details A simplemc::variant_serializer holds exactly one of the backend handles and forwards every
 * serializer operation to the active one via `std::visit`. It is itself a concrete,
 * non-template-on-use type that models the @ref simplemc::serializer concept, which lets callers
 * choose the storage backend (e.g. JSON, HDF5, etc.) at **runtime** while passing a single fixed
 * serializer type around.
 *
 * **Capability is the intersection of the backends.** Because `std::visit` instantiates every arm,
 * `save_at` / `load_at` / `try_load_at` are constrained by simplemc::save_at_compatible /
 * simplemc::load_at_compatible folded over the whole backend pack: a value type must be serializable
 * by *every* backend, not just the active one. In practice the shipped backends cover overlapping
 * type sets.
 *
 * The backend currently held can be inspected via backend(), e.g. with `std::get_if` or
 * `std::visit`.
 *
 * @tparam Bs Backend serializer types.
 */
template <serializer... Bs>
class variant_serializer {
public:
    /**
     * @brief Variant type.
     */
    using variant_type = std::variant<Bs...>;

    /**
     * @brief Default-construct, holding a default-constructed first backend.
     *
     * @details Available only if the first backend is default-constructible.
     */
    variant_serializer() = default;

    /**
     * @brief Construct from a concrete backend handle.
     *
     * @details Constrained to the backend alternatives so it never shadows the copy/move
     * constructors. The variant adopts a copy/move of the given backend as its active alternative.
     *
     * @tparam B Backend type.
     * @param backend Backend handle to adopt.
     */
    template <class B>
        requires(!std::is_same_v<std::remove_cvref_t<B>, variant_serializer> &&
            (std::is_same_v<std::remove_cvref_t<B>, Bs> || ...))
    variant_serializer(B&& backend) : backend_ { std::forward<B>(backend) } {}

    /**
     * @brief Serialize a value at sub-location `key` through the active backend.
     *
     * @tparam T Value type; must be writable by every backend (simplemc::save_at_compatible).
     * @param key Sub-key relative to the current location.
     * @param value Value to write.
     * @return A handle wrapping the active backend (for chaining).
     */
    template <class T>
        requires save_at_compatible<T, Bs...>
    variant_serializer save_at(std::string_view key, const T& value) {
        return std::visit(
            [&](auto& s) -> variant_serializer { return variant_serializer { s.save_at(key, value) }; }, backend_);
    }

    /**
     * @brief Deserialize a value at sub-location `key` through the active backend.
     *
     * @details Throws (via the active backend) if the key is missing.
     *
     * @tparam T Value type; must be readable by every backend (simplemc::load_at_compatible).
     * @param key Sub-key relative to the current location.
     * @param value Value to read into.
     * @return A handle wrapping the active backend (for chaining).
     */
    template <class T>
        requires load_at_compatible<T, Bs...>
    variant_serializer load_at(std::string_view key, T& value) const {
        return std::visit(
            [&](const auto& s) -> variant_serializer { return variant_serializer { s.load_at(key, value) }; },
            backend_);
    }

    /**
     * @brief Try to deserialize a value at sub-location `key` through the active backend.
     *
     * @details Leaves `value` untouched and returns false if the key is missing.
     *
     * @tparam T Value type; must be readable by every backend (simplemc::load_at_compatible).
     * @param key Sub-key relative to the current location.
     * @param value Value to read into.
     * @return True if the key was present and the read occurred, false otherwise.
     */
    template <class T>
        requires load_at_compatible<T, Bs...>
    bool try_load_at(std::string_view key, T& value) const {
        return std::visit([&](const auto& s) -> bool { return s.try_load_at(key, value); }, backend_);
    }

    /**
     * @brief Return a sub-handle positioned at `<current location>/<key>` (save side).
     *
     * @param key Sub-key relative to the current location.
     * @return New variant handle wrapping the active backend's sub-handle.
     */
    variant_serializer operator[](std::string_view key) {
        return std::visit([&](auto& s) -> variant_serializer { return variant_serializer { s[key] }; }, backend_);
    }

    /**
     * @brief Return a sub-handle positioned at `<current location>/<key>` (load side).
     *
     * @details Throws (via the active backend) if the key is missing.
     *
     * @param key Sub-key relative to the current location.
     * @return New variant handle wrapping the active backend's sub-handle.
     */
    variant_serializer operator[](std::string_view key) const {
        return std::visit([&](const auto& s) -> variant_serializer { return variant_serializer { s[key] }; }, backend_);
    }

    /**
     * @brief Test whether the active backend has a sub-location at `key`.
     *
     * @param key Sub-key relative to the current location.
     * @return True if `<current location>/<key>` is present.
     */
    [[nodiscard]] bool has(std::string_view key) const {
        return std::visit([&](const auto& s) -> bool { return s.has(key); }, backend_);
    }

    /**
     * @brief Mutable access to the underlying variant of backends.
     *
     * @return Reference to the held `std::variant`.
     */
    [[nodiscard]] variant_type& backend() noexcept { return backend_; }

    /**
     * @brief Read-only access to the underlying variant of backends.
     *
     * @return Const reference to the held `std::variant`.
     */
    [[nodiscard]] const variant_type& backend() const noexcept { return backend_; }

private:
    variant_type backend_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_VARIANT_VARIANT_SERIALIZER_HPP
