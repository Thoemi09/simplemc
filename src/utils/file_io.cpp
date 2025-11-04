/**
 * @file
 * @brief Implementation details for simplemc/utils/file_io.hpp.
 */

#include <simplemc/utils/file_io.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

namespace simplemc {

namespace detail {

std::FILE* open_file_raw(std::string_view name, std::string_view mode) {
    std::FILE* fp = std::fopen(std::string(name).c_str(), std::string(mode).c_str());
    if (!fp) {
        throw simplemc_exception(
            fmt::format("Cannot open file '{}' in mode '{}': {}", name, mode, std::strerror(errno)),
            "detail::open_file_raw");
    }
    return fp;
}

void close_file_raw(std::FILE* fp, std::string_view name) {
    if (fp && fp != stdout && fp != stdin && fp != stderr) {
        if (std::fclose(fp) != 0) {
            throw simplemc_exception(
                fmt::format("Cannot close file '{}': {}", name, std::strerror(errno)), "detail::close_file_raw");
        }
    }
}

} // namespace detail

file_handle open_file(std::string_view name, std::string_view mode) {
    return { name, mode };
}

file_handle::file_handle(std::string_view name, std::string_view mode) :
    fp_(detail::open_file_raw(name, mode)),
    name_(name) {}

file_handle::file_handle(std::FILE* fp, std::string_view name) : fp_(fp), name_(name) {}

file_handle::~file_handle() noexcept {
    // close current file and catch any exception to avoid throwing from destructor
    try {
        close();
    } catch (const simplemc_exception& e) {
        fmt::print(stderr, "Warning: Failed to close file in file_handle's destructor: {}\n", e.what());
    }
}

file_handle::file_handle(file_handle&& other) noexcept :
    fp_(std::exchange(other.fp_, nullptr)),
    name_(std::move(other.name_)) {}

file_handle& file_handle::operator=(file_handle&& other) noexcept {
    if (this != &other) {
        // close current file and catch any exception to avoid throwing from noexcept function
        try {
            close();
        } catch (const simplemc_exception& e) {
            fmt::print(stderr, "Warning: Failed to close file during move assigning to a file_handle: {}\n", e.what());
        }
        // perform move
        fp_ = std::exchange(other.fp_, nullptr);
        name_ = std::move(other.name_);
    }
    return *this;
}

void file_handle::close() {
    detail::close_file_raw(fp_, name_);
    fp_ = nullptr;
}

} // namespace simplemc
