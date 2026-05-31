/**
 * @file
 * @brief JSON file IO operations for the simplemc-serialize-json sublibrary.
 */

#include <simplemc/serialize/json/file_io.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <iomanip>

namespace simplemc {

void write_json_file(const nlohmann::json& json, const std::string& fname, int width) {
    std::ofstream os(fname);
    os << std::setw(width) << json;
    if (!os) {
        throw simplemc_exception(fmt::format("Writing JSON file {} in text mode failed", fname));
    }
}

void write_json_file(const nlohmann::json& json, const std::string& fname, json_binary_mode mode) {
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
    if (!os) {
        throw simplemc_exception(fmt::format("Writing JSON file {} in binary mode failed", fname));
    }
}

void read_json_file(nlohmann::json& json, const std::string& fname) {
    std::ifstream is(fname);
    if (!(is >> json)) {
        throw simplemc_exception(fmt::format("Reading JSON file {} in text mode failed", fname));
    }
}

void read_json_file(nlohmann::json& json, const std::string& fname, json_binary_mode mode) {
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
    if (!is) {
        throw simplemc_exception(fmt::format("Reading JSON file {} in binary mode failed", fname));
    }
}

} // namespace simplemc
