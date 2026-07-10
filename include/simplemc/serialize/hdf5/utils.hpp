/**
 * @file
 * @brief Utility functions for simplemc-serialize-hdf5.
 */

#ifndef SIMPLEMC_SERIALIZE_HDF5_UTILS_HPP
#define SIMPLEMC_SERIALIZE_HDF5_UTILS_HPP

#include <string>
#include <string_view>

namespace simplemc {

/**
 * @addtogroup simplemc-serialize-hdf5
 * @{
 */

/**
 * @brief Join a POSIX-style HDF5 path with a sub-key.
 *
 * @details The result is always rooted (starts with `/`) and never contains a double slash. The root
 * case (`base` equal to `"/"` or empty) is handled specially so that joining with the root yields
 * `"/<key>"` rather than `"//<key>"`.
 *
 * Examples:
 * - `h5_join_path("/", "a") == "/a"`
 * - `h5_join_path("", "a") == "/a"`
 * - `h5_join_path("/a", "b") == "/a/b"`
 * - `h5_join_path("/a/b", "c") == "/a/b/c"`
 *
 * @param base Existing HDF5 path (rooted or empty). Treated as the root when empty or `"/"`.
 * @param key Sub-key to append. Should not itself contain a leading slash.
 * @return The joined HDF5 path.
 */
inline std::string h5_join_path(std::string_view base, std::string_view key) {
    if (base.empty() || base == "/") {
        return "/" + std::string { key };
    }
    return std::string { base } + "/" + std::string { key };
}

/**
 * @brief Return the parent path of an HDF5 path.
 *
 * @details The parent of a top-level path (e.g. `"/x"`) is the root `"/"`. The parent of the root
 * itself, or of a path containing no slash, is also `"/"`.
 *
 * Examples:
 * - `h5_parent_path("/a/b/c") == "/a/b"`
 * - `h5_parent_path("/x") == "/"`
 * - `h5_parent_path("/") == "/"`
 *
 * @param path POSIX-style HDF5 path.
 * @return The parent path, always rooted.
 */
inline std::string h5_parent_path(const std::string& path) {
    const auto pos = path.rfind('/');
    return (pos == std::string::npos || pos == 0) ? std::string { "/" } : path.substr(0, pos);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_HDF5_UTILS_HPP
