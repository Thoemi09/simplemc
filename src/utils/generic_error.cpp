/**
 * @file
 * @brief Implementation details for simplemc/utils/generic_error.hpp.
 */

#include <simplemc/utils/generic_error.hpp>

#include <fmt/format.h>

#include <filesystem>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

namespace simplemc {

generic_error::generic_error(std::string_view exc_name, std::string_view err_msg, std::string_view func_name) :
    std::runtime_error(make_msg(exc_name, err_msg, func_name)) {}

generic_error::generic_error(std::string_view exc_name, std::string_view err_msg, const std::source_location& loc) :
    std::runtime_error(make_msg(exc_name, err_msg, loc)) {}

std::string generic_error::make_msg(std::string_view exc_name, std::string_view err_msg, std::string_view func_name) {
    if (func_name.empty()) {
        return fmt::format("{}: {}", exc_name, err_msg);
    }
    return fmt::format("{} in {}: {}", exc_name, func_name, err_msg);
}

std::string generic_error::make_msg(
    std::string_view exc_name, std::string_view err_msg, const std::source_location& loc) {
    const std::filesystem::path file_path(loc.file_name());
    const std::string filename = file_path.filename().string();

    return fmt::format("{} at {}:{} in {}: {}", exc_name, filename, loc.line(), loc.function_name(), err_msg);
}

} // namespace simplemc
