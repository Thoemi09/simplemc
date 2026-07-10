/**
 * @file
 * @brief `nlohmann::adl_serializer` specialization for types derived from `Eigen::PlainObjectBase`.
 */

#ifndef SIMPLEMC_SERIALIZE_JSON_EIGEN_HPP
#define SIMPLEMC_SERIALIZE_JSON_EIGEN_HPP

#include <simplemc/numeric/eigen.hpp>
#include <simplemc/serialize/json/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <concepts>

namespace nlohmann {

/**
 * @addtogroup simplemc-serialize-json
 * @{
 */

/**
 * @brief `nlohmann` serializer for types derived from `Eigen::PlainObjectBase`.
 *
 * @details The Eigen object is represented as a JSON object of the form
 * `{"rows": r, "cols": c, "data": [...flat...]}`, where `data` is a flat array of `r * c` elements
 * laid out in the source object's native storage order. Round-trip through this serializer is correct
 * as long as both ends use the same Eigen type.
 *
 * Static extents are checked on deserialization and a simplemc::simplemc_exception is thrown on
 * mismatch. Dynamic extents are resized to match the JSON payload.
 *
 * @tparam M Eigen type derived from `Eigen::PlainObjectBase<M>`.
 */
template <typename M>
    requires std::derived_from<M, Eigen::PlainObjectBase<M>>
struct adl_serializer<M> {
    using Scalar = typename M::Scalar;

    /**
     * @brief Serialize an Eigen object derived from `Eigen::PlainObjectBase` into a JSON object.
     *
     * @param j `nlohmann::json` object to write into.
     * @param m Eigen object to serialize.
     */
    static void to_json(json& j, const M& m) {
        json data;
        simplemc::range_to_json(data, simplemc::make_span(m));
        j = json {
            { "rows", m.rows() },
            { "cols", m.cols() },
            { "data", std::move(data) },
        };
    }

    /**
     * @brief Deserialize a JSON object into an Eigen object derived from `Eigen::PlainObjectBase`.
     *
     * @details Dynamic extents are resized to match the JSON payload. Static extents are checked
     * against the JSON shape and a simplemc::simplemc_exception is thrown on mismatch.
     *
     * Storage order is not checked as long as the dimensions match.
     *
     * @param j `nlohmann::json` object to read from.
     * @param m Eigen object to deserialize into.
     */
    static void from_json(const json& j, M& m) {
        const auto rows = j.at("rows").get<Eigen::Index>();
        const auto cols = j.at("cols").get<Eigen::Index>();
        resize_or_check(m, rows, cols);
        simplemc::range_from_json(j.at("data"), simplemc::make_span(m));
    }

private:
    // Resize `m` if any extent is dynamic; verify static extents match `(rows, cols)` otherwise.
    static void resize_or_check(M& m, Eigen::Index rows, Eigen::Index cols) {
        constexpr int Rs = M::RowsAtCompileTime;
        constexpr int Cs = M::ColsAtCompileTime;
        if constexpr (Rs != Eigen::Dynamic) {
            if (rows != Rs) {
                throw simplemc::simplemc_exception(fmt::format("Eigen rows mismatch: json={}, expected={}", rows, Rs));
            }
        }
        if constexpr (Cs != Eigen::Dynamic) {
            if (cols != Cs) {
                throw simplemc::simplemc_exception(fmt::format("Eigen cols mismatch: json={}, expected={}", cols, Cs));
            }
        }
        if constexpr (Rs == Eigen::Dynamic || Cs == Eigen::Dynamic) {
            m.resize(rows, cols);
        }
    }
};

/** @} */

} // namespace nlohmann

#endif // SIMPLEMC_SERIALIZE_JSON_EIGEN_HPP
