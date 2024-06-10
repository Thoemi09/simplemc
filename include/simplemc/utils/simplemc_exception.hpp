/**
 * @file
 * @brief General exception used in the **simplemc** library.
 */

#ifndef SIMPLEMC_UTILS_SIMPLEMC_EXCEPTION_HPP
#define SIMPLEMC_UTILS_SIMPLEMC_EXCEPTION_HPP

#include <simplemc/utils/generic_error.hpp>

#include <string_view>

namespace simplemc {

/**
 * @addtogroup simplemc-utils-exceptions
 * @{
 */

/**
 * @brief Generic exception class for the **simplemc** library.
 */
class simplemc_exception : public generic_error {
public:
    /**
     * @brief Construct a new exception object.
     *
     * @details Depending on whether the function name argument is empty or not, the constructed error
     * message will have one of the following formats:
     *
     * - `simplemc exception in function {func_name}: {err_msg}` or
     * - `simplemc exception: {err_msg}`.
     *
     * @param err_msg Specific error message.
     * @param func_name Name of the function which throws.
     */
    simplemc_exception(std::string_view err_msg, std::string_view func_name = "") :
        generic_error("simplemc exception", err_msg, func_name) {}
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_UTILS_SIMPLEMC_EXCEPTION_HPP
