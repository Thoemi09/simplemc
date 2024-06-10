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
 * @addtogroup simplemc-utils-indexing
 * @{
 */

/**
 * @brief Tag indicating column-major order.
 *
 * @details Column-major order is the default order in Fortran. The first index varies the fastest and
 * the last index varies the slowest.
 *
 * Let \f$ \mathbf{i} = (i_1, i_2, \ldots, i_d) \f$ be a d-dimensional index and let \f$ \mathbf{N} =
 * (N_1, N_2, \ldots, N_d) \f$ be the shape of a d-dimensional array. The function \f$ f \f$ that maps
 * the multi-dimensional index to the linear/flat index is
 * \f[
 *   f(\mathbf{i}) = i_1 + i_2 N_1 + i_3 N_1 N_2 + \ldots + i_d N_1 N_2 \ldots N_{d-1}
 *   = \sum_{j=1}^{d} i_j \left( \prod_{k=1}^{j-1} N_k \right) \; .
 * \f]
 * To convert a flat index to its multi-dimensional index, we use the fact that the contribution of
 * the first \f$ m \f$ indices is smaller than the product of their extents, i.e.
 * \f[
 *   i_1 + i_2 N_1 + i_3 N_1 N_2 + \ldots + i_m N_1 N_2 \ldots N_{m-1} < N_1 N_2 \ldots N_m \; .
 * \f]
 * This leads us to the following algorithm:
 * -# set \f$ t = N_1 N_2 \ldots N_{d-1} \f$ and \f$ m = d \f$
 * -# \f$ i_m = \lfloor \frac{f}{t} \rfloor \f$
 * -# \f$ f \to f - i_m t \f$
 * -# \f$ t \to \frac{t}{N_m} \f$
 * -# \f$ m \to m - 1 \f$
 * -# go to step 2 and repeat until \f$ m = 1 \f$
 */
struct column_major {};

/**
 * @brief Tag indicating row-major order.
 *
 * @details Row-major order is the default order in C and C++. The last index varies the fastest and
 * the first index varies the slowest.
 *
 * Let \f$ \mathbf{i} = (i_1, i_2, \ldots, i_d) \f$ be a d-dimensional index and let \f$ \mathbf{N} =
 * (N_1, N_2, \ldots, N_d) \f$ be the shape of a d-dimensional array. The function \f$ f \f$ that maps
 * the multi-dimensional index to the linear/flat index is
 * \f[
 *   f(\mathbf{i}) = i_d + i_{d-1} N_d + i_{d-2} N_d N_{d-1} + \ldots + i_1 N_2 N_3 \ldots N_d
 *   = \sum_{j=1}^{d} i_j \left( \prod_{k=j+1}^{d} N_k \right) \; .
 * \f]
 * To convert a flat index to its multi-dimensional index, we use the fact that the contribution of
 * the last \f$ m \f$ indices is smaller than the product of their extents, i.e.
 * \f[
 *   i_d + i_{d-1} N_d + i_{d-2} N_d N_{d-1} + \ldots + i_{d-m+1} N_d N_{d-1} \ldots N_{d-m+1} <
 *   N_d N_{d-1} \ldots N_{d-m} \; .
 * \f]
 * This leads us to the following algorithm:
 * -# set \f$ t = N_2 N_3 \ldots N_{d} \f$ and \f$ m = 1 \f$
 * -# \f$ i_m = \lfloor \frac{f}{t} \rfloor \f$
 * -# \f$ f \to f - i_m t \f$
 * -# \f$ t \to \frac{t}{N_{d-m+1}} \f$
 * -# \f$ m \to m + 1 \f$
 * -# go to step 2 and repeat until \f$ m = d \f$
 */
struct row_major {};

/**
 * @brief A concept that checks if a given type is either a simplemc::column_major or a
 * simplemc::row_major tag.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept nd_order = is_any_of<T, column_major, row_major>;

/**
 * @brief Calculate the size of a multi-dimensional array given its shape as a `std::vector`.
 *
 * @details The size of a multi-dimensional array with shape \f$ (N_1, N_2, \ldots, N_d) \f$ is
 * \f[
 *   s = \prod_{j=1}^{d} N_j \; .
 * \f]
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
 * @brief Calculate the size of a multi-dimensional array given its shape as a `std::array`.
 *
 * @details The size of a multi-dimensional array with shape \f$ (N_1, N_2, \ldots, N_d) \f$ is
 * \f[
 *   s = \prod_{j=1}^{d} N_j \; .
 * \f]
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
 * @details See simplemc::column_major and simplemc::row_major for more information.
 *
 * @note Empty shapes are not supported.
 *
 * @tparam T Integer type.
 * @tparam Order Storage order of the multi-dimensional array.
 * @param flat_idx Flat index.
 * @param shape Shape of the underlying multi-dimensional array.
 * @param order Storage order of the multi-dimensional array (used only for type deduction).
 * @return Multi-dimensional index as a `std::vector`.
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
 * @details See simplemc::column_major and simplemc::row_major for more information.
 *
 * @note Empty shapes are not supported.
 *
 * @tparam T Integer type.
 * @tparam N Number of dimensions.
 * @tparam Order Storage order of the multi-dimensional array.
 * @param flat_idx Flat index.
 * @param shape Shape of the underlying multi-dimensional array.
 * @param order Storage order of the multi-dimensional array (used only for type deduction).
 * @return Multi-dimensional index as a `std::array`.
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
 * @details See simplemc::column_major and simplemc::row_major for more information.
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
 * @details See simplemc::column_major and simplemc::row_major for more information.
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
 * size of the wanted subrange. If `m` is even, the larger possible `j` value is chosen, e.g. if
 * `m = 2`, then `j = l` if possible.
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
