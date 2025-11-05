/**
 * @file
 * @brief General exception used in the **simplemc** library.
 */

#ifndef SIMPLEMC_UTILS_SIMPLEMC_EXCEPTION_HPP
#define SIMPLEMC_UTILS_SIMPLEMC_EXCEPTION_HPP

#include <simplemc/utils/generic_error.hpp>

#include <source_location>
#include <string_view>

namespace simplemc {

/**
 * @addtogroup simplemc-utils-exceptions
 * @{
 */

/**
 * @brief Generic exception class for the **simplemc** library.
 *
 * @details This exception provides two construction modes:
 *
 * 1. Providing the throwing function name manually.
 * 2. Automatic source location capture (recommended).
 *
 * See @ref simplemc-utils-exceptions for examples.
 */
class simplemc_exception : public generic_error {
public:
    /**
     * @brief Construct a new exception object with a given error message and function name.
     *
     * @details See @ref generic_error::generic_error(std::string_view, std::string_view,
     * std::string_view) "generic_error::generic_error" and @ref generic_error::make_msg(
     * std::string_view, std::string_view, std::string_view) "generic_error::make_msg" for more
     * details.
     *
     * @param err_msg Specific error message.
     * @param func_name Name of the function which throws.
     */
    simplemc_exception(std::string_view err_msg, std::string_view func_name) :
        generic_error("simplemc exception", err_msg, func_name) {}

    /**
     * @brief Construct a new exception object with a given error message and automatic source
     * location capture.
     *
     * @details See @ref generic_error::generic_error(std::string_view, std::string_view,
     * const std::source_location&) "generic_error::generic_error" and @ref generic_error::make_msg(
     * std::string_view, std::string_view, const std::source_location&) "generic_error::make_msg" for
     * more details.
     *
     * @param err_msg Specific error message.
     * @param loc Source location information (automatically captured by default).
     */
    simplemc_exception(std::string_view err_msg, const std::source_location& loc = std::source_location::current()) :
        generic_error("simplemc exception", err_msg, loc) {}
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_UTILS_SIMPLEMC_EXCEPTION_HPP
