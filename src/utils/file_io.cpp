/**
 * @file file_io.cpp
 * @brief Basic file input/output functions.
 */

#include <simplemc/utils/file_io.h>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

namespace simplemc {

std::FILE* open_file(const std::string& name, const std::string& mode) {
    std::FILE* fp = std::fopen(name.c_str(), mode.c_str());
    if (!fp) {
        throw simplemc_exception(fmt::format("Cannot open file {} in mode {}", name, mode), "open_file");
    }
    return fp;
}

void close_file(std::FILE* fp) {
    if (fp && fp != stdout && fp != stdin && fp != stderr) {
        if (std::fclose(fp) != 0) {
            throw simplemc_exception("Cannot close file", "close_file");
        }
    }
}

} // namespace simplemc