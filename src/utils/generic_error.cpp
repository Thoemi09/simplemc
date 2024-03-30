#include <simplemc/utils/generic_error.hpp>

#include <fmt/format.h>

namespace simplemc {

generic_error::generic_error(std::string_view exc_name, std::string_view err_msg, std::string_view func_name) :
    std::runtime_error(make_msg(exc_name, err_msg, func_name)) {}

std::string generic_error::make_msg(
    std::string_view exc_name, std::string_view err_msg, std::string_view func_name) const {
    if (func_name.empty()) {
        return fmt::format("{}: {}.", exc_name, err_msg);
    }
    return fmt::format("{} in function {}: {}", exc_name, func_name, err_msg);
}

} // namespace simplemc
