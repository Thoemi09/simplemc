// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

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
 * @details A simplemc::variant_serializer stores a variant of given backend serializers. It is itself
 * a concrete type that models the @ref simplemc::serializer concept, which lets callers choose the
 * specific storage backend (e.g. JSON, HDF5, etc.) at runtime.
 *
 * **Dispatch.** `save_at` / `load_at` / `try_load_at` first try to find a suitable `%simplemc_save` /
 * `%simplemc_load` ADL hook and dispatch to it. If none is found, they delegate to the active
 * backend's same-named member.
 *
 * **Capability is the union of the backends.** The backend-delegating dispatches are constrained by
 * simplemc::save_at_any / simplemc::load_at_any folded over the whole backend pack. That means a
 * value type need only be (de)serializable by the *active* backend. Unsupported types will trigger
 * either a compile-time error (if no backend can handle it) or a simplemc::simplemc_exception at
 * runtime (if the active backend cannot handle it).
 *
 * The backend currently held can be inspected via backend() and `std::get_if` or `std::visit`.
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
     * @brief Default construct a variant serializer by default-constructing the first backend in the 
     * pack.
     *
     * @details Available only if the first backend is default-constructible.
     */
    variant_serializer() = default;

    /**
     * @brief Construct a variant serializer from a concrete backend handle.
     *
     * @details It is constrained to the backend alternatives so it never shadows the copy/move
     * constructors.
     *
     * @tparam B Backend type.
     * @param backend Backend handle to adopt.
     */
    template <typename B>
        requires(!std::is_same_v<std::remove_cvref_t<B>, variant_serializer> &&
            (std::is_same_v<std::remove_cvref_t<B>, Bs> || ...))
    variant_serializer(B&& backend) : backend_ { std::forward<B>(backend) } {}

    /**
     * @brief Serialize a given value at `<current location>/<key>`.
     *
     * @details If simplemc::has_simplemc_save<T, variant_serializer> is satisfied, descend into a
     * sub-handle and delegate to `%simplemc_save`. Otherwise dispatch to the active backend's 
     * `%save_at`.
     *
     * It throws a simplemc::simplemc_exception if the active backend does not support the
     * serialization of `T`.
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
     * @brief Deserialize from `<current location>/<key>` into a given value.
     *
     * @details If simplemc::has_simplemc_load<T, variant_serializer> is satisfied, descend into a
     * sub-handle and delegate to `%simplemc_load`. Otherwise dispatch to the active backend's 
     * `%load_at`.
     *
     * It throws a simplemc::simplemc_exception if the active backend does not support the
     * deserialization of `T`.
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
     * @brief Try to deserialize from `<current location>/<key>` into a given value.
     *
     * @details Same as load_at() except that it silently returns false if `key` is missing, leaving
     * `value` unchanged.
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
     * @details It simply delegates to the active backend's `operator[]` and wraps the returned handle
     * in a new variant_serializer.
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
     * @details It simply delegates to the active backend's `operator[]` and wraps the returned handle
     * in a new variant_serializer.
     *
     * @param key Sub-key relative to the current location.
     * @return New variant handle wrapping the active backend's sub-handle.
     */
    variant_serializer operator[](std::string_view key) const {
        return std::visit([&](const auto& s) -> variant_serializer { return variant_serializer { s[key] }; }, backend_);
    }

    /**
     * @brief Test whether the active backend has a sub-location at `<current location>/<key>`.
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
     * @return Reference to the stored `std::variant`.
     */
    [[nodiscard]] variant_type& backend() noexcept { return backend_; }

    /**
     * @brief Read-only access to the underlying variant of backends.
     *
     * @return Const reference to the stored `std::variant`.
     */
    [[nodiscard]] const variant_type& backend() const noexcept { return backend_; }

private:
    variant_type backend_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_VARIANT_VARIANT_SERIALIZER_HPP
