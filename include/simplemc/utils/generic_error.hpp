/**
 * @file
 * @brief Base class for more specific exceptions.
 */

#ifndef SIMPLEMC_UTILS_GENERIC_ERROR_HPP
#define SIMPLEMC_UTILS_GENERIC_ERROR_HPP

#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

namespace simplemc {

/**
 * @addtogroup simplemc-utils-exceptions
 * @{
 */

/**
 * @brief Generic base class for exceptions.
 *
 * @details This class serves as a base class for all exceptions in the **simplemc** library but can
 * also be used in other libraries.
 *
 * It inherits from `std::runtime_error` and provides constructors for creating exceptions with
 * runtime information, including optional function names and automatic source location capture.
 */
class generic_error : public std::runtime_error {
public:
    /**
     * @brief Construct an exception with runtime information by forwarding the given arguments to
     * generic_error::make_msg.
     *
     * @param exc_name Name of the exception.
     * @param err_msg Specific error message.
     * @param func_name Name of the function which throws.
     */
    generic_error(std::string_view exc_name, std::string_view err_msg, std::string_view func_name);

    /**
     * @brief Construct an exception with automatic source location capture.
     *
     * @details It creates an exception with the given exception name and error message, automatically
     * capturing the source location (file, line, function) where the exception is thrown.
     *
     * @param exc_name Name of the exception.
     * @param err_msg Specific error message.
     * @param loc Source location information (automatically captured by default).
     */
    generic_error(std::string_view exc_name, std::string_view err_msg,
        const std::source_location& loc = std::source_location::current());

    /**
     * @brief Virtual destructor.
     */
    virtual ~generic_error() = default;

    /**
     * @brief Make an error message from exception name, error message, and function name.
     *
     * @details Depending on whether the function name argument is empty or not, the constructed error
     * message will have one of the following formats:
     *
     * - `{exc_name} in {func_name}: {err_msg}` or
     * - `{exc_name}: {err_msg}`.
     *
     * @param exc_name Name of the exception.
     * @param err_msg Specific error message.
     * @param func_name Name of the function which throws.
     * @return Formatted error message string.
     */
    [[nodiscard]] static std::string make_msg(
        std::string_view exc_name, std::string_view err_msg, std::string_view func_name);

    /**
     * @brief Make an error message with source location information.
     *
     * @details It creates a detailed error message including the source file, line number, and
     * function name from the source location. The format is:
     *
     * - `{exc_name} at {file}:{line} in {function}: {err_msg}`.
     *
     * @param exc_name Name of the exception.
     * @param err_msg Specific error message.
     * @param loc Source location information.
     * @return Formatted error message string with source location.
     */
    [[nodiscard]] static std::string make_msg(
        std::string_view exc_name, std::string_view err_msg, const std::source_location& loc);
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_UTILS_GENERIC_ERROR_HPP
