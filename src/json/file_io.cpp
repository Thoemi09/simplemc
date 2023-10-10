/**
 * @file file_io.cpp
 * @brief Implementation of JSON file IO operations.
 */

#include <simplemc/json/file_io.hpp>
#include <simplemc/json/json.hpp>

#include <fstream>
#include <iomanip>

namespace simplemc {

void write_json_file(const nlohmann::json& json, const std::string& fname, int width) {
    std::ofstream os(fname);
    os << std::setw(width) << json;
}

void write_json_file(
    const nlohmann::json& json, const std::string& fname, json_binary_mode mode) {
    using nlohmann::detail::output_adapter;
    std::ofstream os(fname, std::ios_base::binary);
    if (mode == json_binary_mode::bson) {
        nlohmann::json::to_bson(json, output_adapter<char>(os));
    } else if (mode == json_binary_mode::cbor) {
        nlohmann::json::to_cbor(json, output_adapter<char>(os));
    } else if (mode == json_binary_mode::msgpack) {
        nlohmann::json::to_msgpack(json, output_adapter<char>(os));
    } else if (mode == json_binary_mode::ubjson) {
        nlohmann::json::to_ubjson(json, output_adapter<char>(os));
    }
}

void read_json_file(nlohmann::json& json, const std::string& fname) {
    std::ifstream is(fname);
    is >> json;
}

void read_json_file(
    nlohmann::json& json, const std::string& fname, json_binary_mode mode) {
    std::ifstream is(fname, std::ios_base::binary);
    if (mode == json_binary_mode::bson) {
        json = nlohmann::json::from_bson(is);
    } else if (mode == json_binary_mode::cbor) {
        json = nlohmann::json::from_cbor(is);
    } else if (mode == json_binary_mode::msgpack) {
        json = nlohmann::json::from_msgpack(is);
    } else if (mode == json_binary_mode::ubjson) {
        json = nlohmann::json::from_ubjson(is);
    }
}

} // namespace simplemc
