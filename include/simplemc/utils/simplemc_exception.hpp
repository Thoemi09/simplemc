/**
 * @file
 * @brief General exception used in the simplemc library.
 */

#ifndef SIMPLEMC_UTILS_SIMPLEMC_EXCEPTION_HPP
#define SIMPLEMC_UTILS_SIMPLEMC_EXCEPTION_HPP

#include <simplemc/utils/generic_error.hpp>

#include <string_view>

namespace simplemc {

/**
 * @brief Exception class for the simplemc library.
 */
class simplemc_exception : public generic_error {
public:
    /**
     * @brief Construct a new exception object.
     *
     * @param err_msg Specific error message.
     * @param func_name Name of the function which throws.
     */
    simplemc_exception(std::string_view err_msg, std::string_view func_name = "") :
        generic_error("simplemc exception", err_msg, func_name) {}
};

} // namespace simplemc

#endif // SIMPLEMC_UTILS_SIMPLEMC_EXCEPTION_HPP
