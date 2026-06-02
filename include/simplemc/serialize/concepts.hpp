/**
 * @file
 * @brief Concepts for serializer/deserializer backends and ADL customization-point detection.
 */

#ifndef SIMPLEMC_SERIALIZE_CONCEPTS_HPP
#define SIMPLEMC_SERIALIZE_CONCEPTS_HPP

#include <concepts>
#include <string_view>

namespace simplemc {

/**
 * @addtogroup simplemc-serialize-core
 * @{
 */

/**
 * @brief Concept that specifies the requirements for a serializer backend.
 *
 * @details A serializer is able to write named values/objects into some hierarchical store, navigate
 * the store (downward) to sub-locations, and report the existence of sub-locations. See
 * @ref simplemc-serialize-core-model for details.
 *
 * Let `s` be an instance of `S`, `key` a `std::string_view`, and `value` an instane of type
 * `const V&`. The requirements for `S` to be a serializer are the following:
 *
 * - `s[key]` returns a new serializer object (by value) pointing at sub-location `key` relative to
 * the current location.
 * - `s.has(key)` returns a boolean reporting whether `key` is a sub-location of the current location.
 * - `s.save_at(key, value)` writes `value` at sub-location `key` of the current location and returns
 * `s` for chaining.
 *
 * @tparam S Type to check.
 * @tparam V Value type used in the smoke test (defaults to `int`).
 */
template <class S, class V = int>
concept serializer = requires(S& s, const S& cs, const V& value) {
    { s[std::string_view {}] } -> std::same_as<S>;
    { cs.has(std::string_view {}) } -> std::convertible_to<bool>;
    { s.save_at(std::string_view {}, value) } -> std::same_as<S>;
};

/**
 * @brief Concept that specifies the requirements for a deserializer backend.
 *
 * @details A deserializer is able to read named values/objets from some hierarchical store, navigate
 * the store (downward) to sub-locations, and report the existence of sub-locations. See
 * @ref simplemc-serialize-core-model for details. All operations must be callable on a `const`
 * instance — a deserializer never mutates its own state.
 *
 * Let `s` be an instance of `const S`, `key` a `std::string_view`, and `value` an lvalue reference of
 * type `V`. The requirements for the pair `S` to be a deserializer are the following:
 *
 * - `s[key]` returns a new deserializer object (by value) pointing at sub-location `key` relative to
 * the current location.
 * - `s.has(key)` returns a boolean reporting whether `key` is a sub-location of the current location.
 * - `s.load_at(key, value)` reads from sub-location `key` relative to the current location into
 * `value` and returns `s` for chaining
 *
 * @tparam S Type to check.
 * @tparam V Value type used in the smoke test (defaults to `int`).
 */
template <class S, class V = int>
concept deserializer = requires(const S& cs, V& v) {
    { cs[std::string_view {}] } -> std::same_as<S>;
    { cs.has(std::string_view {}) } -> std::convertible_to<bool>;
    { cs.load_at(std::string_view {}, v) } -> std::same_as<S>;
};

/**
 * @brief Check if type `T` is serializable by a serializer of type `S` via a call to `simplemc_save`.
 *
 * @tparam T Value type being serialized.
 * @tparam S Serializer type.
 */
template <class T, class S>
concept has_simplemc_save = requires(const T& t, S& s) { simplemc_save(s, t); };

/**
 * @brief Check if type `T` is deserializable by a deserializer of type `S` via a call to
 * `simplemc_load`.
 *
 * @tparam T Value type being deserialized into.
 * @tparam S Deserializer type.
 */
template <class T, class S>
concept has_simplemc_load = requires(T& t, const S& s) { simplemc_load(s, t); };

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_CONCEPTS_HPP
