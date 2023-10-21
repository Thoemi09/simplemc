/**
 * @file indexing.hpp
 * @brief Convert flat indices to multi-dimensional indices and vice versa.
 */

#ifndef SIMPLEMC_UTILS_INDEXING_HPP
#define SIMPLEMC_UTILS_INDEXING_HPP

#include "range/v3/range/traits.hpp"
#include <simplemc/utils/concepts.hpp>

#include <range/v3/range/concepts.hpp>
#include <range/v3/range/primitives.hpp>
#include <range/v3/numeric/accumulate.hpp>

#include <numeric>

namespace simplemc {

/**
 * @brief Indicating column-major order.
 */
struct column_major {};

/**
 * @brief Indicating row-major order.
 */
struct row_major {};

/**
 * @brief Concept describing the order of a multi-dimensional array.
 */
template <typename T>
concept nd_order = is_any_of<T, column_major, row_major>;

/**
 * @brief Calculate size of an N-dimensional array given its shape as a range.
 *
 * @tparam R Input range.
 * @param shape Shape of N-dim array.
 * @return Number of elements of the array.
 */
auto size_from_shape(ranges::input_range auto&& shape) {
    using return_type = ranges::range_value_t<decltype(shape)>;
    if (ranges::empty(shape)) {
        return return_type { 0 };
    }
    return ranges::accumulate(shape, return_type { 1 }, ranges::multiplies {});
}

/**
 * @brief Calculate size of an N-dimensional array given its shape as a std::array.
 *
 * @tparam T Integral type.
 * @tparam N Number of dimensions.
 * @param shape Shape of N-dim array.
 * @return Number of elements of the array.
 */
template <std::integral T, std::size_t N>
constexpr auto size_from_shape(const std::array<T, N>& shape) {
    using return_type = T;
    if constexpr (N == 0) {
        return return_type { 0 };
    }
    return std::accumulate(shape.begin(), shape.end(), return_type { 1 }, std::multiplies<T> {});
}

/**
 * @brief Convert a flat index to a multi-dimensional index w.r.t. to a given shape.
 *
 * @tparam T Integer type.
 * @tparam N Number of dimensions.
 * @tparam Order Order of the multi-dimensional array.
 * @param flat_idx Flat index.
 * @param shape Shape of underlying N-dim array.
 * @param order Order of the multi-dimensional array.
 * @return Multi-dimensional index as a std::array.
 */
template <std::integral T, std::size_t N, nd_order Order = column_major>
constexpr std::array<T, N> multi_index(
    std::integral auto flat_idx, const std::array<T, N>& shape, [[maybe_unused]] Order order = Order {}) {
    if constexpr (N == 0) {
        return std::array<T, 0> {};
    }
    auto idxs = std::array<T, N> {};
    auto fac = size_from_shape(shape);
    auto rem = flat_idx;
    auto size = shape.size();
    if constexpr (std::same_as<Order, column_major>) {
        for (std::size_t i = 1; i <= N; ++i) {
            fac /= shape[size - i];
            idxs[size - i] = static_cast<T>(rem / fac);
            rem -= idxs[size - i] * fac;
        }
    } else {
        for (std::size_t i = 0; i < N; ++i) {
            fac /= shape[i];
            idxs[i] = static_cast<T>(rem / fac);
            rem -= idxs[i] * fac;
        }
    }
    return idxs;
}

} // namespace simplemc

#endif // SIMPLEMC_UTILS_INDEXING_HPP
