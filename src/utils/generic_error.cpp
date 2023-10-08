/**
 * @file generic_error.cpp
 * @brief This file contains implementation details for the `generic_error` class.
 */

#include <simplemc/utils/generic_error.hpp>

#include <sstream>

namespace simplemc {

generic_error::generic_error(const std::string& exc_name, const std::string& err_msg, const std::string& func_name) :
    std::runtime_error(make_msg(exc_name, err_msg, func_name)) {}

std::string generic_error::make_msg(
    const std::string& exc_name, const std::string& err_msg, const std::string& func_name) const {
    std::stringstream ss;
    ss << exc_name << ": " << err_msg;
    if (!func_name.empty()) {
        ss << " in function " << func_name;
    }
    ss << ".";
    return ss.str();
}

} // namespace simplemc
