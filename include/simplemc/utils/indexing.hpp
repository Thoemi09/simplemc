/**
 * @file
 * @brief Convert flat indices to multi-dimensional indices and vice versa.
 */

#ifndef SIMPLEMC_UTILS_INDEXING_HPP
#define SIMPLEMC_UTILS_INDEXING_HPP

#include <simplemc/utils/concepts.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

namespace simplemc {

/**
 * @addtogroup simplemc-utils
 * @{
 */

/**
 * @brief Calculate the size of a multi-dimensional array given its shape as a std::vector.
 *
 * @note Empty shapes are not supported.
 *
 * @tparam T Integral type.
 * @param shape Shape of the multi-dimensional array.
 * @return Number of elements in the array.
 */
template <std::integral T>
[[nodiscard]] constexpr auto size_from_shape(const std::vector<T>& shape) {
    assert(!shape.empty());
    return std::accumulate(shape.begin(), shape.end(), T { 1 }, std::multiplies {});
}

/**
 * @brief Calculate the size of a multi-dimensional array given its shape as a std::array.
 *
 * @note Empty shapes are not supported.
 *
 * @tparam T Integral type.
 * @tparam N Number of dimensions.
 * @param shape Shape of the multi-dimensional array.
 * @return Number of elements in the array.
 */
template <std::integral T, std::size_t N>
[[nodiscard]] constexpr auto size_from_shape(const std::array<T, N>& shape) {
    static_assert(N != 0, "Empty shapes are not supported");
    return std::accumulate(shape.begin(), shape.end(), T { 1 }, std::multiplies<T> {});
}

/**
 * @brief Convert a flat index to a multi-dimensional index w.r.t. to a given shape.
 *
 * @note Empty shapes are not supported.
 *
 * @tparam T Integer type.
 * @tparam Order Storage order of the multi-dimensional array.
 * @param flat_idx Flat index.
 * @param shape Shape of the underlying multi-dimensional array.
 * @param order Storage order of the multi-dimensional array (used only for type deduction).
 * @return Multi-dimensional index as a std::vector.
 */
template <std::integral T, nd_order Order = column_major>
[[nodiscard]] constexpr std::vector<T> multi_index(
    std::integral auto flat_idx, const std::vector<T>& shape, [[maybe_unused]] Order order = Order {}) {
    assert(flat_idx >= 0 && flat_idx < size_from_shape(shape));
    auto idxs = std::vector<T>(shape.size());
    auto fac = size_from_shape(shape);
    auto rem = flat_idx;
    const auto size = shape.size();
    if constexpr (std::same_as<Order, column_major>) {
        for (std::size_t i = 1; i <= size; ++i) {
            fac /= shape[size - i];
            idxs[size - i] = static_cast<T>(rem / fac);
            rem -= idxs[size - i] * fac;
        }
    } else {
        for (std::size_t i = 0; i < size; ++i) {
            fac /= shape[i];
            idxs[i] = static_cast<T>(rem / fac);
            rem -= idxs[i] * fac;
        }
    }
    return idxs;
}

/**
 * @brief Convert a flat index to a multi-dimensional index w.r.t. to a given shape.
 *
 * @note Empty shapes are not supported.
 *
 * @tparam T Integer type.
 * @tparam N Number of dimensions.
 * @tparam Order Storage order of the multi-dimensional array.
 * @param flat_idx Flat index.
 * @param shape Shape of the underlying multi-dimensional array.
 * @param order Storage order of the multi-dimensional array (used only for type deduction).
 * @return Multi-dimensional index as a std::array.
 */
template <std::integral T, std::size_t N, nd_order Order = column_major>
[[nodiscard]] constexpr std::array<T, N> multi_index(
    std::integral auto flat_idx, const std::array<T, N>& shape, [[maybe_unused]] Order order = Order {}) {
    static_assert(N != 0, "Empty shapes are not supported");
    assert(flat_idx >= 0 && flat_idx < size_from_shape(shape));
    auto idxs = std::array<T, N> {};
    auto fac = size_from_shape(shape);
    auto rem = flat_idx;
    if constexpr (std::same_as<Order, column_major>) {
        for (std::size_t i = 1; i <= N; ++i) {
            fac /= shape[N - i];
            idxs[N - i] = static_cast<T>(rem / fac);
            rem -= idxs[N - i] * fac;
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

/**
 * @brief Convert a multi-dimensional index to a flat index w.r.t. to a given shape.
 *
 * @note Empty shapes or empty multi-dimensional indices are not supported.
 *
 * @tparam T1 Integer type.
 * @tparam T2 Integer type.
 * @tparam Order Storage order of the multi-dimensional array.
 * @param idxs Multi-dimensional index.
 * @param shape Shape of the underlying multi-dimensional array.
 * @param order Storage order of the multi-dimensional array (used only for type deduction).
 * @return Flat index.
 */
template <std::integral T1, std::integral T2, nd_order Order = column_major>
[[nodiscard]] constexpr auto flat_index(
    const std::vector<T1>& idxs, const std::vector<T2>& shape, [[maybe_unused]] Order order = Order {}) {
    assert(!shape.empty() && !idx.empty());
    assert(idxs.size() == shape.size());
    const auto size = shape.size();
    if constexpr (std::same_as<Order, column_major>) {
        auto flat_idx = idxs.back();
        for (std::size_t i = 2; i <= size; ++i) {
            flat_idx = flat_idx * shape[size - i] + idxs[size - i];
        }
        return flat_idx;
    } else {
        auto flat_idx = idxs.front();
        for (std::size_t i = 1; i < size; ++i) {
            flat_idx = flat_idx * shape[i] + idxs[i];
        }
        return flat_idx;
    }
}

/**
 * @brief Convert a multi-dimensional index to a flat index w.r.t. to a given shape.
 *
 * @note Empty shapes or empty multi-dimensional indices are not supported.
 *
 * @tparam T1 Integer type.
 * @tparam T2 Integer type.
 * @tparam N Number of dimensions.
 * @tparam Order Storage order of the multi-dimensional array.
 * @param idxs Multi-dimensional index.
 * @param shape Shape of the underlying multi-dimensional array.
 * @param order Storage order of the multi-dimensional array (used only for type deduction).
 * @return Flat index.
 */
template <std::integral T1, std::integral T2, std::size_t N, nd_order Order = column_major>
[[nodiscard]] constexpr auto flat_index(
    const std::array<T1, N>& idxs, const std::array<T2, N>& shape, [[maybe_unused]] Order order = Order {}) {
    static_assert(N != 0, "Empty shapes or empty multi-dimensional indices are not supported.");
    if constexpr (std::same_as<Order, column_major>) {
        auto flat_idx = idxs.back();
        for (std::size_t i = 2; i <= N; ++i) {
            flat_idx = flat_idx * shape[N - i] + idxs[N - i];
        }
        return flat_idx;
    } else {
        auto flat_idx = idxs.front();
        for (std::size_t i = 1; i < N; ++i) {
            flat_idx = flat_idx * shape[i] + idxs[i];
        }
        return flat_idx;
    }
}

/**
 * @brief Find the integer range of a given size such that a given integer lies in the center
 * restricted by the boundaries of a larger integer range.
 *
 * @details Let `R = (0, 1, ..., n-1)` be the larger integer range of size `n`. Given the integer `l`,
 * we want to find the integer range `(j, j+1, ..., j+m-1)` such that `l` lies in the center of it
 * (insofar as this is possible considering the boundaries, i.e. `j >= 0` and `j <= n-1`). `m` is the
 * size of the wanted subrange. If `m` is even, the larger possible `j` value is chosen, e.g. if m = 2,
 * then `j = l` if possible.
 *
 * @tparam T Integral type.
 * @param l Center of the wanted subrange.
 * @param n Size of the larger range.
 * @param m Size of the wanted subrange.
 * @return Index `j` such that `l` lies in the middle of `(j, j+1, ..., j+m-1)`.
 */
template <std::integral T>
[[nodiscard]] inline constexpr auto integer_subrange(T l, T n, T m) {
    assert(l >= 0 && l < n);
    assert(m > 0 && m <= n);
    return std::clamp(static_cast<T>(m % 2 == 0 ? l + 1 - m / 2 : l - m / 2), T { 0 }, n - m);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_UTILS_INDEXING_HPP
