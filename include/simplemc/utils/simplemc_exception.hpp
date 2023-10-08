/**
 * @file simplemc_exception.h
 * @brief General exception used in the simplemc library.
 */

#pragma once

#include <simplemc/utils/generic_error.hpp>

namespace simplemc {

/**
 * @brief Exception class for simplemc library.
 */
class simplemc_exception : public generic_error {
public:
    /**
     * @brief Construct a new simplemc_exception object.
     * 
     * @param err_msg Specific error message.
     * @param func_name Name of the function which throws.
     */ 
    simplemc_exception(const std::string& err_msg, const std::string& func_name = "") :
        generic_error("simplemc exception", err_msg, func_name) {}
};

} // namespace simplemc
