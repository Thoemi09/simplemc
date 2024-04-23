/**
 * @file
 * @brief Base class for more specific exceptions.
 */

#ifndef SIMPLEMC_UTILS_GENERIC_ERROR_HPP
#define SIMPLEMC_UTILS_GENERIC_ERROR_HPP

#include <stdexcept>
#include <string_view>

namespace simplemc {

/**
 * @addtogroup simplemc-utils
 * @{
 */

/**
 * @brief Generic base class for exceptions.
 *
 * @details This class serves as a base class for all exceptions in the simplemc library but
 * can also be used in other libraries. It inherits from std::runtime_error and provides a
 * constructor and a virtual function for constructing error messages.
 */
class generic_error : public std::runtime_error {
public:
    /**
     * @brief Construct an exception with runtime information.
     *
     * @param exc_name Name of the exception.
     * @param err_msg Specific error message.
     * @param func_name Name of the function which throws.
     */
    generic_error(std::string_view exc_name, std::string_view err_msg, std::string_view func_name = "");

    /**
     * @brief Virtual destructor.
     */
    virtual ~generic_error() = default;

    /**
     * @brief Make an error message.
     *
     * @param exc_name Name of the exception.
     * @param err_msg Specific error message.
     * @param func_name Name of the function which throws.
     */
    [[nodiscard]] virtual std::string make_msg(
        std::string_view exc_name, std::string_view err_msg, std::string_view func_name) const;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_UTILS_GENERIC_ERROR_HPP
