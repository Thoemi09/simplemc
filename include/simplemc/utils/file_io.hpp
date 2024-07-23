/**
 * @file
 * @brief Basic file input/output functions.
 */

#ifndef SIMPLEMC_UTILS_FILE_IO_HPP
#define SIMPLEMC_UTILS_FILE_IO_HPP

#include <cstdio>
#include <string>

namespace simplemc {

/**
 * @addtogroup simplemc-utils-general
 * @{
 */

/**
 * @brief Open a file and return the file pointer.
 *
 * @details Throws a simplemc::simplemc_exception if it fails.
 *
 * @param name File name.
 * @param mode File mode (see <a href="https://en.cppreference.com/w/cpp/io/c/fopen">std::fopen</a>).
 * @return File pointer.
 */
[[nodiscard]] std::FILE* open_file(const std::string& name, const std::string& mode);

/**
 * @brief Close an already opened file.
 *
 * @details Does nothing if the file pointer is equal to `nullptr`, `stdout`, `stdin` or `stderr`.
 *
 * Throws a simplemc::simplemc_exception if it fails.
 *
 * @param fp File pointer.
 */
void close_file(std::FILE* fp);

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_UTILS_FILE_IO_HPP
