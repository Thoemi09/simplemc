/**
 * @file
 * @brief `nlohmann::adl_serializer` specialization for `std::complex<T>`.
 */

#ifndef SIMPLEMC_SERIALIZE_JSON_COMPLEX_HPP
#define SIMPLEMC_SERIALIZE_JSON_COMPLEX_HPP

#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <complex>

namespace nlohmann {

/**
 * @addtogroup simplemc-serialize-json
 * @{
 */

/**
 * @brief `nlohmann` serializer for `std::complex<T>`.
 *
 * @details The complex value \f$ z \f$ is represented as a JSON array \f$ [\Re(z), \Im(z)] \f$.
 *
 * @tparam T Value type.
 */
template <typename T>
struct adl_serializer<std::complex<T>> {
    /**
     * @brief Serialize `std::complex<T>` into a JSON array.
     *
     * @param j `nlohmann::json` object to write into.
     * @param z Complex value to serialize.
     */
    static void to_json(json& j, const std::complex<T>& z) { j = json::array({ z.real(), z.imag() }); }

    /**
     * @brief Deserialize a JSON array into `std::complex<T>`.
     *
     * @details It throws a simplemc::simplemc_exception if the JSON value is not an array of length
     * \f$ 2 \f$.
     *
     * @param j `nlohmann::json` array containing the real and imaginary parts.
     * @param z Complex value to deserialize into.
     */
    static void from_json(const json& j, std::complex<T>& z) {
        if (!j.is_array() || j.size() != 2) {
            throw simplemc::simplemc_exception(
                fmt::format("std::complex deserialization requires a JSON array of size 2"));
        }
        z.real(j[0].get<T>());
        z.imag(j[1].get<T>());
    }
};

/** @} */

} // namespace nlohmann

#endif // SIMPLEMC_SERIALIZE_JSON_COMPLEX_HPP
