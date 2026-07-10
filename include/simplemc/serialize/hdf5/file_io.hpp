/**
 * @file
 * @brief HDF5 file open mode helpers for simplemc-serialize-hdf5.
 */

#ifndef SIMPLEMC_SERIALIZE_HDF5_FILE_IO_HPP
#define SIMPLEMC_SERIALIZE_HDF5_FILE_IO_HPP

#include <highfive/highfive.hpp>

namespace simplemc {

/**
 * @addtogroup simplemc-serialize-hdf5
 * @{
 */

/**
 * @brief File open mode for simplemc::hdf5_serializer.
 *
 * @details The three modes map onto HighFive's `HighFive::File` access flags:
 *
 * - `read`: Open an existing file in read-only mode. Throws if the file does not exist.
 * - `truncate`: Create a fresh file (truncate if it already exists). The default for the save side.
 * - `append`: Open an existing file for read/write, or create it if missing. Use this to add content
 * to a file that already has structure you want to keep.
 */
enum class hdf5_file_mode { read, truncate, append };

/**
 * @brief Convert a simplemc::hdf5_file_mode to the corresponding HighFive access mode flags.
 *
 * @details Mapping:
 * - `read` → `HighFive::File::ReadOnly`
 * - `truncate` → `HighFive::File::Truncate`
 * - `append` → `HighFive::File::ReadWrite | HighFive::File::Create`
 *
 * @param mode File open mode.
 * @return The corresponding HighFive access mode flags.
 */
inline HighFive::File::AccessMode to_highfive_access_mode(hdf5_file_mode mode) {
    switch (mode) {
        case hdf5_file_mode::read: return HighFive::File::ReadOnly;
        case hdf5_file_mode::truncate: return HighFive::File::Truncate;
        case hdf5_file_mode::append: return HighFive::File::ReadWrite | HighFive::File::Create;
    }
    return HighFive::File::ReadOnly; // unreachable
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_HDF5_FILE_IO_HPP
