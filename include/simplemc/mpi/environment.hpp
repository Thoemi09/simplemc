/**
 * @file
 * @brief RAII MPI environment to handle initialization and finalization of MPI programs.
 */

#ifndef SIMPLEMC_MPI_ENVIRONMENT_HPP
#define SIMPLEMC_MPI_ENVIRONMENT_HPP

#include <mpi.h>

namespace simplemc::mpi {

/**
 * @ingroup simplemc-mpi-commenv
 * @brief Initialize and finalize MPI execution environments.
 *
 * @details This is a simplified version of the `boost::mpi::environment` class. It is intended to be
 * used in the `main()` function of a program, where it calls `MPI_Init` on construction and
 * `MPI_Finalize` on destruction:
 *
 * @code{.cpp}
 * int main(int argc, char** argv) {
 *     ...
 *     // initialize MPI
 *     simplemc::mpi::environment env(argc, argv);
 *     ...
 * }
 * @endcode
 *
 * Only very basic MPI calls are supported. For more advanced tasks, e.g. like threading support, the
 * user is advised to use the MPI C library directly.
 *
 * See @ref tut_mpi_1 and @ref tut_mpi_2 for full examples.
 */
class environment {
public:
    /**
     * @brief Check if the MPI environment is initialized.
     *
     * @details Makes a call to `MPI_Initialized` and throws an exception if the call fails.
     *
     * @return True if `MPI_Init` has been called.
     */
    [[nodiscard]] static bool initialized();

    /**
     * @brief Check if the MPI environment is finalized.
     *
     * @details Makes a call to `MPI_Finalized` and throws an exception if the call fails.
     *
     * @return True if `MPI_Finalize` has been called.
     */
    [[nodiscard]] static bool is_finalized();

    /**
     * @brief Abort all MPI processes on `MPI_COMM_WORLD`.
     *
     * @details Makes a call to `MPI_Abort` and throws an exception if the call fails.
     *
     * @param errcode Error code that will be passed to the `MPI_Abort` function.
     */
    static void abort(int errcode = 0);

    /**
     * @brief Clean up all MPI processes with a call to `MPI_Finalize`.
     *
     * @details It first checks if MPI has already been finalized. If not, it calls `MPI_Finalize`.
     *
     * It throws an exception if any MPI call fails.
     */
    static void finalize();

    /**
     * @brief Constructor initializes the MPI environment.
     *
     * @details If no MPI environment has been initialized, it calls `MPI_Init` with the given
     * arguments.
     *
     * It throws an exception if any MPI call fails.
     *
     * See @ref tut_mpi_2 for an example on how `abort_on_exception` can be used.
     *
     * @param argc Number of arguments passed to `main()`.
     * @param argv Arguments passed to `main()`.
     * @param abort_on_exception If the environment is destroyed due to an uncaught exception, it will
     * call `MPI_Abort` instead of `MPI_Finalize` in its destructor.
     */
    environment(int& argc, char**& argv, bool abort_on_exception = true);

    /**
     * @brief Copy constructor is deleted.
     */
    environment(const environment&) = delete;

    /**
     * @brief Copy assignment operator is deleted.
     */
    environment& operator=(const environment&) = delete;

    /**
     * @brief Clean up the MPI environment.
     *
     * @details It either calls `MPI_Abort` due to an uncaught exception (if `abort_on_exception` is
     * set to `true`) or by calling `MPI_Finalize`.
     *
     * See @ref tut_mpi_2 for an example showing the effect of `abort_on_exception`.
     */
    ~environment();

private:
    bool init_;
    bool abort_on_exception_;
};

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_ENVIRONMENT_HPP
