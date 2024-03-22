/**
 * @file
 * @brief Serialize/Deserialize ranges to/from JSON.
 */

#ifndef SIMPLEMC_JSON_RANGE_HPP
#define SIMPLEMC_JSON_RANGE_HPP

#include <simplemc/json/json.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <range/v3/range/concepts.hpp>

namespace simplemc {

/**
 * @brief Serialize a range to JSON.
 *
 * @param j nlohmann::json object.
 * @param rg input_range to serialize.
 */
void range_to_json(nlohmann::json& j, ranges::input_range auto&& rg) {
    for (auto&& x : rg) {
        j.push_back(x);
    }
}

/**
 * @brief Deserialize a range from JSON.
 *
 * @details The size of the output_range and the input JSON array are only checked
 * if the output_range is a sized_range. Throws a simplemc::simplemc_exception if the sizes
 * do not match.
 *
 * @param j nlohmann::json object.
 * @param rg output_range to deserialize into.
 */
template <ranges::range R>
    requires ranges::output_range<R, ranges::range_value_t<R>>
void range_from_json(const nlohmann::json& j, R&& rg) { // NOLINT (we don't want to forward ranges)
    if constexpr (ranges::sized_range<R>) {
        if (j.size() != ranges::size(rg)) {
            throw simplemc_exception("Range size mismatch", "range_from_json");
        }
    }
    auto it = ranges::begin(rg);
    for (const auto& el : j) {
        *it = el.get<ranges::range_value_t<R>>();
        ++it;
    }
}

} // namespace simplemc

#endif // SIMPLEMC_JSON_RANGE_HPP
