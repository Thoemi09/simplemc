/**
 * @file
 * @brief Utility functions for simplemc-serialize-json.
 */

#ifndef SIMPLEMC_SERIALIZE_JSON_UTILS_HPP
#define SIMPLEMC_SERIALIZE_JSON_UTILS_HPP

#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

namespace simplemc {

/**
 * @addtogroup simplemc-serialize-json
 * @{
 */

/**
 * @brief Serialize an input range into a JSON array.
 *
 * @details Generalization of nlohmann's built-in container support for arbitrary input ranges (e.g.
 * range-v3 or `std::ranges`). Each element is pushed into a JSON array.
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
 * @brief Deserialize a JSON array into an output range.
 *
 * @details It simply iterates through the JSON array and assigns each element to the output range.
 *
 * If the output range is sized, its size is checked against the JSON array length and a
 * simplemc::simplemc_exception is thrown on mismatch.
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

#endif // SIMPLEMC_SERIALIZE_JSON_UTILS_HPP
