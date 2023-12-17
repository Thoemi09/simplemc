/**
 * @file file_io.hpp
 * @brief JSON file IO operations.
 */

#ifndef SIMPLEMC_JSON_FILE_IO_HPP
#define SIMPLEMC_JSON_FILE_IO_HPP

#include <simplemc/json/json_fwd.hpp>

#include <string>

namespace simplemc {

/**
 * @brief Enumerate possible binary IO strategies for JSON files.
 *
 * @details The following binary formats are supported: BSON (bson),
 * CBOR (cbor), MessagePack (msgpack) and UBJSON (ubjson).
 */
enum class json_binary_mode { bson, cbor, msgpack, ubjson };

/**
 * @brief Write nlohmann::json object to a file in text mode.
 *
 * @details The file is opened in the standard std::ios_base::out mode, i.e.
 * any content of an existing file will be truncated before writing to it.
 *
 * Throws an exception if writing fails.
 *
 * @param json nlohmann::json object to be written.
 * @param fname Name of the file.
 * @param width Indentation width.
 */
void write_json_file(const nlohmann::json& json, const std::string& fname, int width = 0);

/**
 * @brief Write nlohmann::json object to a file in binary mode.
 *
 * @details The file is opened in the std::ios_base::binary mode, i.e. any
 * content of an existing file will be truncated before writing to it.
 *
 * Throws an exception if writing fails.
 *
 * @param json nlohmann::json object to be written.
 * @param fname Name of the file.
 * @param mode Specific JSON binary mode.
 */
void write_json_file(const nlohmann::json& json, const std::string& fname, json_binary_mode mode);

/**
 * @brief Read a JSON text file into a nlohmann::json object.
 *
 * @details Throws an exception if reading fails.
 *
 * @param json nlohmann::json object to be read into.
 * @param fname Name of the file.
 */
void read_json_file(nlohmann::json& json, const std::string& fname);

/**
 * @brief Read a JSON binary file into a nlohmann::json object.
 *
 * @details The specified binary mode has to be the same that was used to write
 * the file in the first place.
 *
 * Throws an exception if reading fails.
 *
 * @param json nlohmann::json object to be read into.
 * @param fname Name of the file.
 * @param mode Specific JSON binary mode.
 */
void read_json_file(nlohmann::json& json, const std::string& fname, json_binary_mode mode);

} // namespace simplemc

#endif // SIMPLEMC_JSON_FILE_IO_HPP
