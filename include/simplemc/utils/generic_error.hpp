/**
 * @file generic_error.hpp
 * @brief Base class for more specific exceptions.
 */

#ifndef SIMPLEMC_UTILS_GENERIC_ERROR_HPP
#define SIMPLEMC_UTILS_GENERIC_ERROR_HPP

#include <stdexcept>
#include <string>

namespace simplemc {

/**
 * @brief Generic base class for exceptions.
 *
 * This class serves as a base class for all exceptions in the simplemc library. 
 * It inherits from std::runtime_error and provides a constructor and a virtual 
 * function for constructing error messages.
 */
class generic_error : public std::runtime_error {
public:
    /**
     * @brief Construct exception with runtime information.
     *
     * @param exc_name Name of exception.
     * @param err_msg Specific error message.
     * @param func_name Name of the function which throws.
     */
    generic_error(const std::string& exc_name, const std::string& err_msg, const std::string& func_name = "");

    /**
     * @brief Virtual destructor.
     */
    virtual ~generic_error() = default;

    /**
     * @brief Construct error message.
     *
     * @param exc_name Name of exception.
     * @param err_msg Specific error message.
     * @param func_name Name of the function which throws.
     */
    virtual std::string make_msg(
        const std::string& exc_name, const std::string& err_msg, const std::string& func_name) const;
};

} // namespace simplemc

#endif // SIMPLEMC_UTILS_GENERIC_ERROR_HPP
