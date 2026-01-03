/**
 * @file
 * @brief Convert flat indices to multi-dimensional indices and vice versa.
 */

#ifndef SIMPLEMC_UTILS_ND_INDEXING_HPP
#define SIMPLEMC_UTILS_ND_INDEXING_HPP

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
 * Let \f$ \mathbf{i} = (i_1, i_2, \ldots, i_N) \f$ be an N-dimensional index and let \f$ \mathbf{M} =
 * (M_1, M_2, \ldots, M_N) \f$ be the shape of an N-dimensional array. The function \f$ f \f$ that
 * maps the multi-dimensional index to the linear/flat index \f$ i_{\mathrm{lin}} \f$ is
 * \f[
 *   f(\mathbf{i}) = i_1 + i_2 M_1 + i_3 M_1 M_2 + \ldots + i_N M_1 M_2 \ldots M_{N-1}
 *   = \sum_{j=1}^{N} i_j \left( \prod_{k=1}^{j-1} M_k \right) = i_{\mathrm{lin}} \; .
 * \f]
 * To convert a flat index to its multi-dimensional index, we use the fact that the contribution of
 * the first \f$ d \f$ indices is smaller than the product of their extents, i.e.
 * \f[
 *   i_1 + i_2 M_1 + i_3 M_1 M_2 + \ldots + i_d M_1 M_2 \ldots M_{d-1} < M_1 M_2 \ldots M_d \; .
 * \f]
 * This leads us to the following algorithm:
 * -# set \f$ t = M_1 M_2 \ldots M_{N-1} \f$, \f$ f = i_{\mathrm{lin}} \f$ and \f$ d = N \f$
 * -# \f$ i_d = \lfloor \frac{f}{t} \rfloor \f$
 * -# \f$ f \to f - i_d t \f$
 * -# \f$ t \to \frac{t}{M_d} \f$
 * -# \f$ d \to d - 1 \f$
 * -# go to step 2 and repeat until \f$ d = 1 \f$
 */
struct column_major {};

/**
 * @brief Tag indicating row-major order.
 *
 * @details Row-major order is the default order in C and C++. The last index varies the fastest and
 * the first index varies the slowest.
 *
 * Let \f$ \mathbf{i} = (i_1, i_2, \ldots, i_N) \f$ be an N-dimensional index and let \f$ \mathbf{M} =
 * (M_1, M_2, \ldots, M_N) \f$ be the shape of an N-dimensional array. The function \f$ f \f$ that
 * maps the multi-dimensional index to the linear/flat index \f$ i_{\mathrm{lin}} \f$ is
 * \f[
 *   f(\mathbf{i}) = i_N + i_{N-1} M_N + i_{N-2} M_N M_{N-1} + \ldots + i_1 M_N M_{N-1} \ldots M_2
 *   = \sum_{j=1}^{N} i_j \left( \prod_{k=j+1}^{N} M_k \right) = i_{\mathrm{lin}} \; .
 * \f]
 * To convert a flat index to its multi-dimensional index, we use the fact that the contribution of
 * the last \f$ d \f$ indices is smaller than the product of their extents, i.e.
 * \f[
 *   i_N + i_{N-1} M_N + i_{N-2} M_N M_{N-1} + \ldots + i_{N-d+1} M_N M_{N-1} \ldots M_{N-d+1} <
 *   M_N M_{N-1} \ldots M_{N-d} \; .
 * \f]
 * This leads us to the following algorithm:
 * -# set \f$ t = M_N M_{N-1} \ldots M_2 \f$, \f$ f = i_{\mathrm{lin}} \f$ and \f$ d = 1 \f$
 * -# \f$ i_d = \lfloor \frac{f}{t} \rfloor \f$
 * -# \f$ f \to f - i_d t \f$
 * -# \f$ t \to \frac{t}{M_{N-d+1}} \f$
 * -# \f$ d \to d + 1 \f$
 * -# go to step 2 and repeat until \f$ d = N \f$
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
 * @details The size of a multi-dimensional array with shape \f$ \mathbf{M} = (M_1, M_2, \ldots, M_d)
 * \f$ is
 * \f[
 *   M = \prod_{j=1}^{d} M_j \; .
 * \f]
 *
 * @note The behavior is undefined if \f$ \mathbf{M} \f$ is empty.
 *
 * @tparam T Integral type.
 * @param shape Shape \f$ \mathbf{M} \f$ of the multi-dimensional array.
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
 * @details The size of a multi-dimensional array with shape \f$ \mathbf{M} = (M_1, M_2, \ldots, M_d)
 * \f$ is
 * \f[
 *   M = \prod_{j=1}^{d} M_j \; .
 * \f]
 *
 * @note The behavior is undefined if \f$ \mathbf{M} \f$ is empty.
 *
 * @tparam T Integral type.
 * @tparam N Number of dimensions.
 * @param shape Shape \f$ \mathbf{M} \f$ of the multi-dimensional array.
 * @return Number of elements in the array.
 */
template <std::integral T, std::size_t N>
    requires(N != 0)
[[nodiscard]] constexpr auto size_from_shape(const std::array<T, N>& shape) {
    return std::accumulate(shape.begin(), shape.end(), T { 1 }, std::multiplies<T> {});
}

/**
 * @brief Convert a flat index to a multi-dimensional index w.r.t. to a given shape.
 *
 * @details See simplemc::column_major and simplemc::row_major for more information.
 *
 * @note The behavior is undefined if one of the following conditions is met:
 * - \f$ M_j \leq 0 \f$ for one of the dimensions \f$ j \f$,
 * - \f$ i_{\mathrm{lin}} < 0 \f$ or \f$ i_{\mathrm{lin}} \geq M \f$,
 * - \f$ \mathbf{M} \f$ is empty.
 *
 * @tparam T Integer type.
 * @tparam Order simplemc::nd_order tag.
 * @param flat_idx Flat index \f$ i_{\mathrm{lin}} \f$.
 * @param shape Shape \f$ \mathbf{M} \f$ of the underlying multi-dimensional array.
 * @param order Storage order of the multi-dimensional array (used only for type deduction).
 * @return Multi-dimensional index \f$ \mathbf{i} \f$ as a `std::vector`.
 */
template <std::integral T, nd_order Order = column_major>
[[nodiscard]] constexpr auto nd_index(
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
 * @note The behavior is undefined if one of the following conditions is met:
 * - \f$ M_j \leq 0 \f$ for one of the dimensions \f$ j \f$,
 * - \f$ i_{\mathrm{lin}} < 0 \f$ or \f$ i_{\mathrm{lin}} \geq M \f$,
 * - \f$ \mathbf{M} \f$ is empty.
 *
 * @tparam T Integer type.
 * @tparam N Number of dimensions.
 * @tparam Order simplemc::nd_order tag.
 * @param flat_idx Flat index \f$ i_{\mathrm{lin}} \f$.
 * @param shape Shape \f$ \mathbf{M} \f$ of the underlying multi-dimensional array.
 * @param order Storage order of the multi-dimensional array (used only for type deduction).
 * @return Multi-dimensional index \f$ \mathbf{i} \f$ as a `std::array`.
 */
template <std::integral T, std::size_t N, nd_order Order = column_major>
    requires(N != 0)
[[nodiscard]] constexpr auto nd_index(
    std::integral auto flat_idx, const std::array<T, N>& shape, [[maybe_unused]] Order order = Order {}) {
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
 * @note The behavior is undefined if one of the following conditions is met:
 * - \f$ i_j < 0 \f$ or \f$ i_j \geq M_j \f$ for one of the dimensions \f$ j \f$,
 * - \f$ \mathbf{M} \f$ and \f$ \mathbf{i} \f$ have different number of dimensions or are empty.
 *
 * @tparam T1 Integer type.
 * @tparam T2 Integer type.
 * @tparam Order simplemc::nd_order tag.
 * @param idxs Multi-dimensional index \f$ \mathbf{i} \f$.
 * @param shape Shape \f$ \mathbf{M} \f$ of the underlying multi-dimensional array.
 * @param order Storage order of the multi-dimensional array (used only for type deduction).
 * @return Flat index \f$ i_{\mathrm{lin}} \f$.
 */
template <std::integral T1, std::integral T2, nd_order Order = column_major>
[[nodiscard]] constexpr auto flat_index(
    const std::vector<T1>& idxs, const std::vector<T2>& shape, [[maybe_unused]] Order order = Order {}) {
    assert(!shape.empty() && !idxs.empty());
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
 * @note The behavior is undefined if one of the following conditions is met:
 * - \f$ i_j < 0 \f$ or \f$ i_j \geq M_j \f$ for one of the dimensions \f$ j \f$,
 * - \f$ \mathbf{M} \f$ and \f$ \mathbf{i} \f$ have different number of dimensions or are empty.
 *
 * @tparam T1 Integer type.
 * @tparam T2 Integer type.
 * @tparam N Number of dimensions.
 * @tparam Order simplemc::nd_order tag.
 * @param idxs Multi-dimensional index \f$ \mathbf{i} \f$.
 * @param shape Shape \f$ \mathbf{M} \f$ of the underlying multi-dimensional array.
 * @param order Storage order of the multi-dimensional array (used only for type deduction).
 * @return Flat index \f$ i_{\mathrm{lin}} \f$.
 */
template <std::integral T1, std::integral T2, std::size_t N, nd_order Order = column_major>
    requires(N != 0)
[[nodiscard]] constexpr auto flat_index(
    const std::array<T1, N>& idxs, const std::array<T2, N>& shape, [[maybe_unused]] Order order = Order {}) {
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

#endif // SIMPLEMC_UTILS_ND_INDEXING_HPP
