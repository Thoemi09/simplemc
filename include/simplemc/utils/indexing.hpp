/**
 * @file indexing.hpp
 * @brief Convert flat indices to multi-dimensional indices and vice versa.
 */

#ifndef SIMPLEMC_UTILS_INDEXING_HPP
#define SIMPLEMC_UTILS_INDEXING_HPP

#include <simplemc/utils/concepts.hpp>

#include <array>
#include <numeric>
#include <vector>

namespace simplemc {

/**
 * @brief Calculate size of an multi-dimensional array given its shape as a std::vector.
 *
 * @tparam T Integral type.
 * @tparam N Number of dimensions.
 * @param shape Shape of multi-dimensional array.
 * @return Number of elements of the array.
 */
template <std::integral T>
constexpr auto size_from_shape(const std::vector<T>& shape) {
    using return_type = std::vector<T>::value_type;
    if (shape.empty()) {
        return return_type { 0 };
    }
    return std::accumulate(shape.begin(), shape.end(), return_type { 1 }, std::multiplies {});
}

/**
 * @brief Calculate size of an multi-dimensional array given its shape as a std::array.
 *
 * @tparam T Integral type.
 * @tparam N Number of dimensions.
 * @param shape Shape of multi-dimensional array.
 * @return Number of elements of the array.
 */
template <std::integral T, std::size_t N>
constexpr auto size_from_shape_array(const std::array<T, N>& shape) {
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
 * @tparam Order Order of the multi-dimensional array.
 * @param flat_idx Flat index.
 * @param shape Shape of underlying multi-dimensional array.
 * @param order Order of the multi-dimensional array.
 * @return Multi-dimensional index as a std::vector.
 */
template <std::integral T, nd_order Order = column_major>
constexpr std::vector<T> multi_index(
    std::integral auto flat_idx, const std::vector<T>& shape, [[maybe_unused]] Order order = Order {}) {
    assert(flat_idx >= 0 && flat_idx < size_from_shape(shape));
    if (shape.empty()) {
        return std::vector<T> {};
    }
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
 * @tparam T Integer type.
 * @tparam N Number of dimensions.
 * @tparam Order Order of the multi-dimensional array.
 * @param flat_idx Flat index.
 * @param shape Shape of underlying multi-dimensional array.
 * @param order Order of the multi-dimensional array.
 * @return Multi-dimensional index as a std::array.
 */
template <std::integral T, std::size_t N, nd_order Order = column_major>
constexpr std::array<T, N> multi_index_array(
    std::integral auto flat_idx, const std::array<T, N>& shape, [[maybe_unused]] Order order = Order {}) {
    assert(flat_idx >= 0 && flat_idx < size_from_shape_array(shape));
    if constexpr (N == 0) {
        return std::array<T, 0> {};
    }
    auto idxs = std::array<T, N> {};
    auto fac = size_from_shape_array(shape);
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
 * The multi-index as well as the shape are given as std::arrays.
 *
 * @tparam T1 Integer type.
 * @tparam T2 Integer type.
 * @tparam N Number of dimensions.
 * @tparam Order Order of the multi-dimensional array.
 * @param idxs Multi-dimensional index.
 * @param shape Shape of underlying multi-dimensional array.
 * @param order Order of the multi-dimensional array.
 * @return Flat index.
 */
template <std::integral T1, std::integral T2, nd_order Order = column_major>
constexpr auto flat_index(
    const std::vector<T1>& idxs, const std::vector<T2>& shape, [[maybe_unused]] Order order = Order {}) {
    assert(idxs.size() == shape.size());
    const auto size = shape.size();
    if constexpr (std::same_as<Order, column_major>) {
        auto idx = idxs.back();
        for (std::size_t i = 2; i <= size; ++i) {
            idx = idx * shape[size - i] + idxs[size - i];
        }
        return idx;
    } else {
        auto idx = idxs.front();
        for (std::size_t i = 1; i < size; ++i) {
            idx = idx * shape[i] + idxs[i];
        }
        return idx;
    }
}

/**
 * @brief Convert a multi-dimensional index to a flat index w.r.t. to a given shape.
 * The multi-index as well as the shape are given as std::arrays.
 *
 * @tparam T1 Integer type.
 * @tparam T2 Integer type.
 * @tparam N Number of dimensions.
 * @tparam Order Order of the multi-dimensional array.
 * @param idxs Multi-dimensional index.
 * @param shape Shape of underlying multi-dimensional array.
 * @param order Order of the multi-dimensional array.
 * @return Flat index.
 */
template <std::integral T1, std::integral T2, std::size_t N, nd_order Order = column_major>
constexpr auto flat_index_array(
    const std::array<T1, N>& idxs, const std::array<T2, N>& shape, [[maybe_unused]] Order order = Order {}) {
    if constexpr (std::same_as<Order, column_major>) {
        auto idx = idxs.back();
        for (std::size_t i = 2; i <= N; ++i) {
            idx = idx * shape[N - i] + idxs[N - i];
        }
        return idx;
    } else {
        auto idx = idxs.front();
        for (std::size_t i = 1; i < N; ++i) {
            idx = idx * shape[i] + idxs[i];
        }
        return idx;
    }
}

} // namespace simplemc

#endif // SIMPLEMC_UTILS_INDEXING_HPP
