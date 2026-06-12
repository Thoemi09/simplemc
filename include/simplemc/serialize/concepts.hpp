/**
 * @file
 * @brief Concept for serializer backends and ADL customization-point detection.
 */

#ifndef SIMPLEMC_SERIALIZE_CONCEPTS_HPP
#define SIMPLEMC_SERIALIZE_CONCEPTS_HPP

#include <concepts>
#include <string_view>

namespace simplemc {

/**
 * @addtogroup simplemc-serialize
 * @{
 */

/**
 * @brief Concept that specifies the requirements for a serializer backend.
 *
 * @details A serializer is able to read and write named values/objects in some hierarchical store,
 * navigate the store (downward) to sub-locations, and report the existence of sub-locations. See
 * @ref simplemc-serialize-model for details.
 *
 * Let `s` be an instance of `S`, `cs` an instance of `const S`, `key` a `std::string_view`, `value`
 * an instance of type `const T&`, and `mvalue` an lvalue reference of type `T`. The requirements
 * for `S` to be a serializer are the following:
 *
 * - `cs[key]` returns a new serializer object (by value) pointing at sub-location `key` relative to
 * the current location.
 * - `cs.has(key)` returns a boolean reporting whether `key` is a sub-location of the current
 * location.
 * - `s.save_at(key, value)` writes `value` at sub-location `key` of the current location and returns
 * `s` for chaining.
 * - `cs.load_at(key, mvalue)` reads from sub-location `key` of the current location into `mvalue`
 * and returns `s` for chaining.
 * - `cs.try_load_at(key, mvalue)` reads from sub-location `key` of the current location into
 * `mvalue` if the sub-location exists, and returns a boolean reporting whether the read occurred.
 * If the sub-location is missing, `mvalue` is left untouched.
 *
 * @tparam S Type to check.
 * @tparam T Type used in the smoke test (defaults to `int`).
 */
template <class S, class T = int>
concept serializer = requires(S& s, const S& cs, std::string_view key, const T& value, T& mvalue) {
    { s[key] } -> std::same_as<S>;
    { cs[key] } -> std::same_as<S>;
    { cs.has(key) } -> std::convertible_to<bool>;
    { s.save_at(key, value) } -> std::same_as<S>;
    { cs.load_at(key, mvalue) } -> std::same_as<S>;
    { cs.try_load_at(key, mvalue) } -> std::convertible_to<bool>;
};

/**
 * @brief Check if type `T` is serializable by a serializer of type `S` via a call to `simplemc_save`.
 *
 * @tparam T Type being serialized.
 * @tparam S Serializer type.
 */
template <class T, class S>
concept has_simplemc_save = requires(const T& t, S& s) { simplemc_save(s, t); };

/**
 * @brief Check if type `T` is deserializable by a serializer of type `S` via a call to
 * `simplemc_load`.
 *
 * @tparam T Type being deserialized into.
 * @tparam S Serializer type.
 */
template <class T, class S>
concept has_simplemc_load = requires(T& t, const S& s) { simplemc_load(s, t); };

/**
 * @brief Check that every serializer in a pack can write a value of type `T` via `save_at`.
 *
 * @tparam T Type being serialized.
 * @tparam Bs Serializer backend types.
 */
template <class T, class... Bs>
concept save_at_compatible = (requires(Bs& s, std::string_view key, const T& v) { s.save_at(key, v); } && ...);

/**
 * @brief Check that every serializer in a pack can read a value of type `T` via `load_at`.
 *
 * @tparam T Type being deserialized into.
 * @tparam Bs Serializer backend types.
 */
template <class T, class... Bs>
concept load_at_compatible = (requires(const Bs& s, std::string_view key, T& v) { s.load_at(key, v); } && ...);

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_CONCEPTS_HPP
