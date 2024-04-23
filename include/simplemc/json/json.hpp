/**
 * @file
 * @brief Additional serializers for the nlohmann-json library.
 */

#ifndef SIMPLEMC_JSON_JSON_HPP
#define SIMPLEMC_JSON_JSON_HPP

#include <nlohmann/json.hpp>

#include <complex>

namespace nlohmann {

/**
 * @ingroup simplemc-json
 * @brief Serialize/Deserialize std::complex<T> to/from JSON.
 *
 * @tparam T Value type of std::complex<T>.
 */
template <typename T>
struct adl_serializer<std::complex<T>> {
    /**
     * @brief Serialize std::complex<T> to JSON as an array of size 2.
     *
     * @param j nlohmann::json object.
     * @param z std::complex<T> object.
     */
    static void to_json(json& j, const std::complex<T>& z) { j = json::array({ z.real(), z.imag() }); }

    /**
     * @brief Derialize std::complex<T> from JSON.
     *
     * @param j nlohmann::json object.
     * @param z std::complex<T> object.
     */
    static void from_json(const json& j, std::complex<T>& z) {
        z.real(j[0].get<T>());
        z.imag(j[1].get<T>());
    }
};

} // namespace nlohmann

#endif // SIMPLEMC_JSON_JSON_HPP
