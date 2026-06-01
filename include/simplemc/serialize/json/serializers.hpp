/**
 * @file
 * @brief nlohmann-side adapters for `std::complex` and Eigen types — JSON-fallback path.
 */

#ifndef SIMPLEMC_SERIALIZE_JSON_SERIALIZERS_HPP
#define SIMPLEMC_SERIALIZE_JSON_SERIALIZERS_HPP

#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <complex>
#include <cstddef>

namespace nlohmann {

/**
 * @ingroup simplemc-serialize-json
 * @brief Serialize / deserialize `std::complex<T>` to / from JSON as a 2-element array.
 *
 * @tparam T Value type of `std::complex<T>`.
 */
template <typename T>
struct adl_serializer<std::complex<T>> {
    /// Serialize `std::complex<T>` as `[real, imag]`.
    static void to_json(json& j, const std::complex<T>& z) {
        j = json::array({ z.real(), z.imag() });
    }
    /// Deserialize `std::complex<T>` from `[real, imag]`.
    static void from_json(const json& j, std::complex<T>& z) {
        z.real(j[0].get<T>());
        z.imag(j[1].get<T>());
    }
};

/**
 * @ingroup simplemc-serialize-json
 * @brief Serialize / deserialize `Eigen::Matrix<T, R, C>` to / from JSON.
 *
 * @details Vectors (`C == 1`) are emitted as a flat array. Matrices (`C > 1` or dynamic) are emitted
 * as `{"rows": r, "cols": c, "data": [...row-major flat...]}`. The deserializer handles dynamic
 * extents by `resize`-ing; static extents are size-checked and throw on mismatch.
 *
 * @tparam M Eigen matrix type. Must satisfy simplemc::eigen_matrix.
 */
template <typename M>
    requires simplemc::eigen_matrix<M>
struct adl_serializer<M> {
    using Scalar = typename M::Scalar;

    static void to_json(json& j, const M& m) {
        const auto rows = static_cast<std::size_t>(m.rows());
        const auto cols = static_cast<std::size_t>(m.cols());
        if constexpr (static_cast<int>(M::ColsAtCompileTime) == 1) {
            // vector: flat array
            j = json::array();
            for (long i = 0; i < m.rows(); ++i) {
                j.push_back(m(i));
            }
        } else {
            // matrix: rows + cols + row-major data
            auto data = json::array();
            for (long r = 0; r < m.rows(); ++r) {
                for (long c = 0; c < m.cols(); ++c) {
                    data.push_back(m(r, c));
                }
            }
            j = json { { "rows", rows }, { "cols", cols }, { "data", std::move(data) } };
        }
    }

    static void from_json(const json& j, M& m) {
        if constexpr (static_cast<int>(M::ColsAtCompileTime) == 1) {
            const auto n = j.size();
            if constexpr (static_cast<int>(M::RowsAtCompileTime) == Eigen::Dynamic) {
                m.resize(static_cast<long>(n));
            } else if (n != static_cast<std::size_t>(static_cast<int>(M::RowsAtCompileTime))) {
                throw simplemc::simplemc_exception(
                    fmt::format("Eigen vector size mismatch: json={}, expected={}", n, static_cast<int>(M::RowsAtCompileTime)));
            }
            for (std::size_t i = 0; i < n; ++i) {
                m(static_cast<long>(i)) = j[i].template get<Scalar>();
            }
        } else {
            const auto rows = j.at("rows").get<std::size_t>();
            const auto cols = j.at("cols").get<std::size_t>();
            if constexpr (static_cast<int>(M::RowsAtCompileTime) == Eigen::Dynamic && static_cast<int>(M::ColsAtCompileTime) == Eigen::Dynamic) {
                m.resize(static_cast<long>(rows), static_cast<long>(cols));
            } else if constexpr (static_cast<int>(M::RowsAtCompileTime) == Eigen::Dynamic) {
                if (cols != static_cast<std::size_t>(static_cast<int>(M::ColsAtCompileTime))) {
                    throw simplemc::simplemc_exception(fmt::format(
                        "Eigen matrix cols mismatch: json={}, expected={}", cols, static_cast<int>(M::ColsAtCompileTime)));
                }
                m.resize(static_cast<long>(rows), static_cast<long>(cols));
            } else if constexpr (static_cast<int>(M::ColsAtCompileTime) == Eigen::Dynamic) {
                if (rows != static_cast<std::size_t>(static_cast<int>(M::RowsAtCompileTime))) {
                    throw simplemc::simplemc_exception(fmt::format(
                        "Eigen matrix rows mismatch: json={}, expected={}", rows, static_cast<int>(M::RowsAtCompileTime)));
                }
                m.resize(static_cast<long>(rows), static_cast<long>(cols));
            } else {
                if (rows != static_cast<std::size_t>(static_cast<int>(M::RowsAtCompileTime))
                    || cols != static_cast<std::size_t>(static_cast<int>(M::ColsAtCompileTime))) {
                    throw simplemc::simplemc_exception(
                        fmt::format("Eigen matrix shape mismatch: json=({},{}), expected=({},{})", rows, cols,
                            static_cast<int>(M::RowsAtCompileTime), static_cast<int>(M::ColsAtCompileTime)));
                }
            }
            const auto& data = j.at("data");
            if (data.size() != rows * cols) {
                throw simplemc::simplemc_exception(fmt::format(
                    "Eigen matrix data length mismatch: json={}, expected={}", data.size(), rows * cols));
            }
            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t c = 0; c < cols; ++c) {
                    m(static_cast<long>(r), static_cast<long>(c))
                        = data[r * cols + c].template get<Scalar>();
                }
            }
        }
    }
};

} // namespace nlohmann

namespace simplemc {

/**
 * @addtogroup simplemc-serialize-json
 * @{
 */

/**
 * @brief Serialize a range into a JSON array node.
 *
 * @details Generalization of nlohmann's built-in container support for arbitrary input ranges
 * (e.g., range-v3 / `std::ranges` views). Each element is pushed into `j` via nlohmann's
 * machinery. The result is a JSON array.
 *
 * @param j `nlohmann::json` object to write into.
 * @param rg Range to serialize.
 */
void range_to_json(nlohmann::json& j, ranges::input_range auto&& rg) {
    j = nlohmann::json::array();
    for (auto&& x : rg) {
        j.push_back(x);
    }
}

/**
 * @brief Deserialize a JSON array node into a range.
 *
 * @details If the output range is sized, its size is checked against the JSON array length and a
 * simplemc::simplemc_exception is thrown on mismatch. Otherwise the loop simply walks until either
 * end is reached.
 *
 * @param j `nlohmann::json` object to read from.
 * @param rg Output range to deserialize into.
 */
template <ranges::range R>
    requires ranges::output_range<R, ranges::range_value_t<R>>
void range_from_json(const nlohmann::json& j, R&& rg) { // NOLINT(cppcoreguidelines-missing-std-forward)
    if constexpr (ranges::sized_range<R>) {
        if (j.size() != ranges::size(rg)) {
            throw simplemc_exception(
                fmt::format("Range size mismatch: json={}, output={}", j.size(), ranges::size(rg)));
        }
    }
    auto it = ranges::begin(rg);
    for (const auto& el : j) {
        *it = el.get<ranges::range_value_t<R>>();
        ++it;
    }
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_JSON_SERIALIZERS_HPP
