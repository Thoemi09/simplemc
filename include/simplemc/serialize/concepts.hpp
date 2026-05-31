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
 * @brief Concept satisfied by serializer types that write keyed values and navigate sub-positions.
 *
 * @details A type `S` models @ref output_serializer if instances of `S` expose:
 * - `s.save_at(key, value)` for keyed writes,
 * - `s[key]` returning another `S` pointing at the sub-position.
 *
 * @tparam S Type to check.
 */
template <class S>
concept output_serializer = requires(S& s, int probe) {
    { s.save_at(std::string_view {}, probe) };
    { s[std::string_view {}] } -> std::same_as<S>;
};

/**
 * @brief Concept satisfied by deserializer types that read keyed values, navigate sub-positions, and
 * support schema-tolerance queries.
 *
 * @details A type `S` models @ref input_serializer if instances of `S` expose:
 * - `s.load_at(key, value_ref)` for keyed reads,
 * - `s[key]` returning another `S` pointing at the sub-position,
 * - `s.has(key)` returning a boolean (key presence check),
 * - `s.array_size(key)` returning the length of an array at `key`.
 *
 * @tparam S Type to check.
 */
template <class S>
concept input_serializer = requires(S& s, int& probe) {
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
