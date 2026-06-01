/**
 * @file
 * @brief Explicit named helpers for serializing/deserializing ranges and tuples.
 */

#ifndef SIMPLEMC_SERIALIZE_HELPERS_HPP
#define SIMPLEMC_SERIALIZE_HELPERS_HPP

#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/ranges.hpp>

#include <cstddef>
#include <cstdio>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-serialize-core
 * @{
 */

namespace detail {

/// Render a small integer index as a string key for `save_at` / `load_at`.
[[nodiscard]] inline std::string index_key(std::size_t i) {
    char buf[32];
    auto n = std::snprintf(buf, sizeof(buf), "%zu", i);
    return std::string { buf, static_cast<std::size_t>(n) };
}

} // namespace detail

/**
 * @brief Save a sized range element-wise into the current serializer position.
 *
 * @details Writes `{"size": N, "0": elem_0, "1": elem_1, ...}` to the current position of @p s.
 * Each element dispatches through `s.save_at`, which finds the element's `simplemc_save` (via ADL on
 * the element type) or falls back to the backend's native idiom (e.g., nlohmann's `to_json` for JSON).
 *
 * @param s Output serializer (forwarding reference).
 * @param r Sized range to serialize.
 */
template <class S, ranges::sized_range R>
    requires serializer<std::remove_cvref_t<S>>
void save_range(S&& s, const R& r) {
    s.save_at("size", static_cast<std::size_t>(ranges::size(r)));
    std::size_t i = 0;
    for (const auto& elem : r) {
        s.save_at(detail::index_key(i++), elem);
    }
}

/**
 * @brief Load a sized range element-wise from the current deserializer position.
 *
 * @details Reads the size key, clears @p out, reserves space, and reads each element via
 * `s.load_at`. Requires the element type to be default-constructible.
 *
 * @param s Input serializer (forwarding reference).
 * @param out Range to populate (must support `clear`, `reserve`, `push_back`, expose `value_type`).
 */
template <class S, class R>
    requires deserializer<std::remove_cvref_t<S>>
void load_range(S&& s, R& out) {
    std::size_t size = 0;
    s.load_at("size", size);
    out.clear();
    out.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
        typename R::value_type elem;
        s.load_at(detail::index_key(i), elem);
        out.push_back(std::move(elem));
    }
}

/**
 * @brief Save a `std::tuple` element-wise into the current serializer position.
 *
 * @details Writes `{"0": elem_0, "1": elem_1, ...}`. The number of elements is implicit in the
 * tuple's type; we don't write a separate "size".
 *
 * @param s Output serializer (forwarding reference).
 * @param t Tuple to serialize.
 */
template <class S, class... Ts>
    requires serializer<std::remove_cvref_t<S>>
void save_tuple(S&& s, const std::tuple<Ts...>& t) {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        ((s.save_at(detail::index_key(I), std::get<I>(t))), ...);
    }(std::index_sequence_for<Ts...> {});
}

/**
 * @brief Load a `std::tuple` element-wise from the current deserializer position.
 *
 * @param s Input serializer (forwarding reference).
 * @param t Tuple to populate.
 */
template <class S, class... Ts>
    requires deserializer<std::remove_cvref_t<S>>
void load_tuple(S&& s, std::tuple<Ts...>& t) {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        ((s.load_at(detail::index_key(I), std::get<I>(t))), ...);
    }(std::index_sequence_for<Ts...> {});
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_HELPERS_HPP
