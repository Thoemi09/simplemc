/**
 * @file
 * @brief Backend-erasing serializer that dispatches to one of several serializer backends at runtime.
 */

#ifndef SIMPLEMC_SERIALIZE_VARIANT_VARIANT_SERIALIZER_HPP
#define SIMPLEMC_SERIALIZE_VARIANT_VARIANT_SERIALIZER_HPP

#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

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
 * **Dispatch.** For a value type with an ADL `%simplemc_save` / `%simplemc_load` written against the
 * variant handle itself, `save_at` / `load_at` / `try_load_at` descend into a sub-handle and dispatch
 * to that overload (the **simplemc-mc** value types rely on this). Otherwise they delegate to the
 * active backend's same-named member.
 *
 * **Capability is the union of the backends.** For the backend-delegating path, the operations are
 * constrained by simplemc::save_at_any / simplemc::load_at_any folded over the whole backend pack: a
 * value type need only be serializable by *at least one* backend. A type serializable by no backend
 * is a compile error; a type unsupported by the *currently-active* backend throws a
 * simplemc::simplemc_exception at runtime (each `std::visit` arm is guarded so incompatible arms
 * throw rather than fail to compile).
 *
 * The backend currently held can be inspected via backend(), e.g. with `std::get_if` or `std::visit`.
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
    template <typename B>
        requires(!std::is_same_v<std::remove_cvref_t<B>, variant_serializer> &&
            (std::is_same_v<std::remove_cvref_t<B>, Bs> || ...))
    variant_serializer(B&& backend) : backend_ { std::forward<B>(backend) } {}

    /**
     * @brief Serialize a value at sub-location `key`.
     *
     * @details If an ADL `%simplemc_save(variant_serializer, const T&)` is reachable for `T` (i.e.
     * the type's serialization is written against this variant handle itself), descend into a
     * sub-handle and dispatch to it.
     *
     * Otherwise delegate to the active backend's `save_at`, which requires at least one backend to
     * handle `T` (simplemc::save_at_any). If the active backend cannot serialize `T`,it throws a
     * simplemc::simplemc_exception.
     *
     * @tparam T Value type.
     * @param key Sub-key relative to the current location.
     * @param value Value to write.
     * @return A copy of *this.
     */
    template <typename T>
        requires(save_at_any<T, Bs...> || has_simplemc_save<T, variant_serializer>)
    variant_serializer save_at(std::string_view key, const T& value) {
        if constexpr (has_simplemc_save<T, variant_serializer>) {
            auto sub = (*this)[key];
            simplemc_save(sub, value);
        } else {
            std::visit(
                [&](auto& s) {
                    if constexpr (save_at_all<T, std::remove_cvref_t<decltype(s)>>) {
                        s.save_at(key, value);
                    } else {
                        throw simplemc_exception(fmt::format("serializer backend cannot serialize key '{}'", key));
                    }
                },
                backend_);
        }
        return *this;
    }

    /**
     * @brief Deserialize a value at sub-location `key`.
     *
     * @details If an ADL `%simplemc_load(const variant_serializer&, T&)` is reachable for `T` (i.e.
     * the type's deserialization is written against this variant handle itself), descend into a
     * sub-handle and dispatch to it.
     *
     * Otherwise delegate to the active backend's `load_at`, which requires at least one backend to
     * handle `T` (simplemc::load_at_any). If the active backend cannot deserialize `T`, it throws a
     * simplemc::simplemc_exception.
     *
     * @tparam T Value type.
     * @param key Sub-key relative to the current location.
     * @param value Value to read into.
     * @return A copy of *this.
     */
    template <typename T>
        requires(load_at_any<T, Bs...> || has_simplemc_load<T, variant_serializer>)
    variant_serializer load_at(std::string_view key, T& value) const {
        if constexpr (has_simplemc_load<T, variant_serializer>) {
            const auto sub = (*this)[key];
            simplemc_load(sub, value);
        } else {
            std::visit(
                [&](const auto& s) {
                    if constexpr (load_at_all<T, std::remove_cvref_t<decltype(s)>>) {
                        s.load_at(key, value);
                    } else {
                        throw simplemc_exception(fmt::format("serializer backend cannot deserialize key '{}'", key));
                    }
                },
                backend_);
        }
        return *this;
    }

    /**
     * @brief Try to deserialize a value at sub-location `key` through the active backend.
     *
     * @details If an ADL `%simplemc_load(const variant_serializer&, T&)` is reachable for `T` (i.e.
     * the type's deserialization is written against this variant handle itself), descend into a
     * sub-handle and dispatch to it.
     *
     * Otherwise delegate to the active backend's `try_load_at`, which requires at least one backend to
     * handle `T` (simplemc::load_at_any). If the active backend cannot deserialize `T`, it throws a
     * simplemc::simplemc_exception.
     *
     * It leaves `value` untouched and returns false if the key is missing.
     *
     * @tparam T Value type.
     * @param key Sub-key relative to the current location.
     * @param value Value to read into.
     * @return True if the key was present and the read occurred, false otherwise.
     */
    template <typename T>
        requires(load_at_any<T, Bs...> || has_simplemc_load<T, variant_serializer>)
    bool try_load_at(std::string_view key, T& value) const {
        if constexpr (has_simplemc_load<T, variant_serializer>) {
            if (!has(key)) {
                return false;
            }
            const auto sub = (*this)[key];
            simplemc_load(sub, value);
            return true;
        } else {
            return std::visit(
                [&](const auto& s) -> bool {
                    if constexpr (load_at_all<T, std::remove_cvref_t<decltype(s)>>) {
                        return s.try_load_at(key, value);
                    } else {
                        throw simplemc_exception(fmt::format("serializer backend cannot deserialize key '{}'", key));
                    }
                },
                backend_);
        }
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
