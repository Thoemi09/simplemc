/**
 * @file
 * @brief JSON file IO operations.
 */

#ifndef SIMPLEMC_SERIALIZE_JSON_FILE_IO_HPP
#define SIMPLEMC_SERIALIZE_JSON_FILE_IO_HPP

#include <nlohmann/json_fwd.hpp>

#include <string>

namespace simplemc {

/**
 * @addtogroup simplemc-serialize-json
 * @{
 */

/**
 * @brief Enumerate the on-disk formats supported by simplemc::write_json_file and
 * simplemc::read_json_file.
 *
 * @details The supported formats are:
 * - `text`: human-readable text JSON (default)
 * - `bson`: binary JSON (BSON)
 * - `cbor`: Concise Binary Object Representation (CBOR)
 * - `msgpack`: MessagePack
 * - `ubjson`: Universal Binary JSON (UBJSON)
 */
enum class json_file_mode { text, bson, cbor, msgpack, ubjson };

/**
 * @brief Options passed to simplemc::write_json_file and simplemc::read_json_file.
 */
struct json_io_options {
    /// On-disk format. Defaults to human-readable text JSON.
    json_file_mode mode = json_file_mode::text;

    /// Indentation width for text mode (\f$ 0 \f$ for compact output). Ignored for binary modes.
    int indent = 0;
};

/**
 * @brief Write an `nlohmann::json` object to file.
 *
 * @details The file is truncated before writing. The provided simplemc::json_io_options determine the
 * format of the written file.
 *
 * It throws a simplemc::simplemc_exception on failure.
 *
 * @param json `nlohmann::json` object to be written.
 * @param fname Name of the file.
 * @param opts IO options (format and, for text, indentation).
 */
void write_json_file(const nlohmann::json& json, const std::string& fname, const json_io_options& opts = {});

/**
 * @brief Read a JSON file into an `nlohmann::json` object.
 *
 * @details The provided simplemc::json_io_options determine the expected format of the file.
 *
 * It throws a simplemc::simplemc_exception on failure.
 *
 * @param json `nlohmann::json` object to be read into.
 * @param fname Name of the file.
 * @param opts IO options (format).
 */
void read_json_file(nlohmann::json& json, const std::string& fname, const json_io_options& opts = {});

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_JSON_FILE_IO_HPP
