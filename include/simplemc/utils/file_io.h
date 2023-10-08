/**
 * @file file_io.h
 * @brief Basic file input/output functions.
 */

#ifndef SIMPLEMC_UTILS_FILE_IO_H
#define SIMPLEMC_UTILS_FILE_IO_H

#include <cstdio>
#include <string>

namespace simplemc {

/**
 * @brief Open file. Throws an exception if it fails.
 *
 * @param name File name.
 * @param mode File mode.
 * @return File pointer.
 */
std::FILE* open_file(const std::string& name, const std::string& mode);

/**
 * @brief Close file. Does nothing if the file pointer is equal to nullptr, stdout, stdin
 * or stderr. Throws an exception if closing fails.
 *
 * @param fp File pointer.
 */
void close_file(std::FILE* fp);

} // namespace simplemc

#endif // SIMPLEMC_UTILS_FILE_IO_H