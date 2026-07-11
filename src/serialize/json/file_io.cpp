// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Implementation details for simplemc/serialize/json/file_io.hpp.
 */

#include <simplemc/serialize/json/file_io.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>
#include <fmt/std.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <iomanip>

namespace simplemc {

void write_json_file(const nlohmann::json& json, const std::filesystem::path& fpath, const json_io_options& opts) {
    if (opts.mode == json_file_mode::text) {
        std::ofstream os(fpath);
        os << std::setw(opts.indent) << json;
        if (!os) {
            throw simplemc_exception(fmt::format("Writing JSON file {} in text mode failed", fpath));
        }
    } else {
        using nlohmann::detail::output_adapter;
        std::ofstream os(fpath, std::ios_base::binary);
        if (opts.mode == json_file_mode::bson) {
            nlohmann::json::to_bson(json, output_adapter<char>(os));
        } else if (opts.mode == json_file_mode::cbor) {
            nlohmann::json::to_cbor(json, output_adapter<char>(os));
        } else if (opts.mode == json_file_mode::msgpack) {
            nlohmann::json::to_msgpack(json, output_adapter<char>(os));
        } else if (opts.mode == json_file_mode::ubjson) {
            nlohmann::json::to_ubjson(json, output_adapter<char>(os));
        }
        if (!os) {
            throw simplemc_exception(fmt::format("Writing JSON file {} in binary mode failed", fpath));
        }
    }
}

void read_json_file(nlohmann::json& json, const std::filesystem::path& fpath, const json_io_options& opts) {
    if (opts.mode == json_file_mode::text) {
        std::ifstream is(fpath);
        if (!is.is_open()) {
            throw simplemc_exception(fmt::format("Opening JSON file {} for reading failed", fpath));
        }
        if (!(is >> json)) {
            throw simplemc_exception(fmt::format("Reading JSON file {} in text mode failed", fpath));
        }
    } else {
        std::ifstream is(fpath, std::ios_base::binary);
        if (!is.is_open()) {
            throw simplemc_exception(fmt::format("Opening JSON file {} for reading failed", fpath));
        }
        if (opts.mode == json_file_mode::bson) {
            json = nlohmann::json::from_bson(is);
        } else if (opts.mode == json_file_mode::cbor) {
            json = nlohmann::json::from_cbor(is);
        } else if (opts.mode == json_file_mode::msgpack) {
            json = nlohmann::json::from_msgpack(is);
        } else if (opts.mode == json_file_mode::ubjson) {
            json = nlohmann::json::from_ubjson(is);
        }
        if (!is) {
            throw simplemc_exception(fmt::format("Reading JSON file {} in binary mode failed", fpath));
        }
    }
}

} // namespace simplemc
