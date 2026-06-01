/**
 * @file
 * @brief Concepts for serializer/deserializer backends and ADL customization-point detection.
 */

#ifndef SIMPLEMC_SERIALIZE_CONCEPTS_HPP
#define SIMPLEMC_SERIALIZE_CONCEPTS_HPP

#include <concepts>
#include <cstddef>
#include <string_view>

namespace simplemc {

/**
 * @addtogroup simplemc-serialize-concepts
 * @{
 */

/**
 * @brief Output backend for keyed state serialization.
 *
 * @details A type satisfies simplemc::serializer if instances of `S` can write keyed values into
 * the current position and navigate to sub-positions.
 *
 * Let `s` be a serializer instance of type `S` and let `key` be a `std::string_view`. The
 * requirements for a type `S` to be a serializer are the following:
 *
 * - `s.save_at(key, value)` writes `value` at sub-key `key` of the current position
 * - `s[key]` returns another `S` pointing at the sub-position `key`
 *
 * @tparam S Type to check.
 */
template <class S>
concept serializer = requires(S& s, int probe) {
    { s.save_at(std::string_view {}, probe) };
    { s[std::string_view {}] } -> std::same_as<S>;
};

/**
 * @brief Input backend for keyed state deserialization.
 *
 * @details A type satisfies simplemc::deserializer if instances of `S` can read keyed values from
 * the current position, navigate to sub-positions, and answer schema-tolerance queries about the
 * stored data.
 *
 * Let `s` be a deserializer instance of type `S` and let `key` be a `std::string_view`. The
 * requirements for a type `S` to be a deserializer are the following:
 *
 * - `s.load_at(key, value_ref)` reads from sub-key `key` of the current position into `value_ref`
 * - `s[key]` returns another `S` pointing at the sub-position `key`
 * - `s.has(key)` returns a boolean indicating whether `key` is present (convertible to `bool`)
 * - `s.array_size(key)` returns the number of elements in the array at `key` (convertible to
 * `std::size_t`)
 *
 * @tparam S Type to check.
 */
template <class S>
concept deserializer = requires(S& s, int& probe) {
    { s.load_at(std::string_view {}, probe) };
    { s[std::string_view {}] } -> std::same_as<S>;
    { s.has(std::string_view {}) } -> std::convertible_to<bool>;
    { s.array_size(std::string_view {}) } -> std::convertible_to<std::size_t>;
};

/** @} */

namespace detail {

/**
 * @brief True when `simplemc_save(s, t)` is callable via ADL.
 *
 * @tparam T Value type being serialized.
 * @tparam S Serializer type.
 */
template <class T, class S>
concept has_simplemc_save = requires(const T& t, S& s) { simplemc_save(s, t); };

/**
 * @brief True when `simplemc_load(s, t)` is callable via ADL.
 *
 * @tparam T Value type being deserialized into.
 * @tparam S Deserializer type.
 */
template <class T, class S>
concept has_simplemc_load = requires(T& t, S& s) { simplemc_load(s, t); };

/**
 * @brief Helper for `static_assert(detail::always_false<T>, "...")` inside templated branches.
 *
 * @tparam T Unused — the trait is `false` regardless.
 */
template <class T>
inline constexpr bool always_false = false;

} // namespace detail

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_CONCEPTS_HPP
