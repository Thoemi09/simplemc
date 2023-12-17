/**
 * @file basic_json.hpp
 * @brief Basic interface to support JSON IO.
 */

#ifndef SIMPLEMC_JSON_BASIC_JSON_HPP
#define SIMPLEMC_JSON_BASIC_JSON_HPP

#include <simplemc/json/json_fwd.hpp>

namespace simplemc {

/**
 * @brief Basic interface to support JSON IO.
 *
 * @details Classes can derive from this class if they want to support JSON IO.
 */
struct basic_json {
public:
    /**
     * @brief Serialize to JSON.
     *
     * @param j nlohmann::json object.
     */
    virtual void write_json([[maybe_unused]] nlohmann::json& j) const {}

    /**
     * @brief Deserialize from JSON
     *
     * @param j nlohmann::json object.
     */
    virtual void read_json([[maybe_unused]] const nlohmann::json& j) {}

    /**
     * @brief User input in JSON format. Can be used to write default input files.
     *
     * @param j nlohmann::json object.
     */
    virtual void write_input_json([[maybe_unused]] nlohmann::json& j) const {}

    /**
     * @brief User input in JSON format. Can be used to read in user input.
     *
     * @param j nlohmann::json object.
     */
    virtual void read_input_json([[maybe_unused]] const nlohmann::json& j) {}

    /**
     * @brief Virtual dtor.
     */
    virtual ~basic_json() = default;
};

/**
 * @brief Polymorphic call to basic_json::write_json.
 *
 * @param j nlohmann::json object.
 * @param bj basic_json object.
 */
inline void to_json(nlohmann::json& j, const basic_json& bj) {
    bj.write_json(j);
}

/**
 * @brief Polymorphic call to basic_json::read_json.
 *
 * @param j nlohmann::json object.
 * @param bj basic_json object.
 */
inline void from_json(const nlohmann::json& j, basic_json& bj) {
    bj.read_json(j);
}

/**
 * @brief Polymorphic call to basic_json::write_input_json.
 *
 * @param j nlohmann::json object.
 * @param bj basic_json object.
 */
inline void to_input_json(nlohmann::json& j, const basic_json& bj) {
    bj.write_input_json(j);
}

/**
 * @brief Polymorphic call to basic_json::read_input_json.
 *
 * @param j nlohmann::json object.
 * @param bj basic_json object.
 */
inline void from_input_json(const nlohmann::json& j, basic_json& bj) {
    bj.read_input_json(j);
}

} // namespace simplemc

#endif // SIMPLEMC_JSON_BASIC_JSON_HPP
