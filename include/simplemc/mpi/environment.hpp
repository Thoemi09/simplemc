/**
 * @file
 * @brief RAII MPI environment to handle initialization and finalization of MPI programs.
 */

#ifndef SIMPLEMC_MPI_ENVIRONMENT_HPP
#define SIMPLEMC_MPI_ENVIRONMENT_HPP

#include <mpi.h>

namespace simplemc::mpi {

/**
 * @brief Initialize and finalize MPI execution environments.
 *
 * @details This is a short version of the boost::mpi::environment class. It is intended to be
 * used in the main() function of a program, where it calls MPI_Init on construction and MPI_Finalize
 * on destruction:
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
 * Only very basic MPI calls are supported. If more sophisticated functionality is needed,
 * e.g like threading, then the user is advised to call the MPI C library directly.
 */
class environment {
public:
    /**
     * @brief Check if the MPI environment is initialized.
     *
     * @return True if MPI_Init has been called.
     */
    [[nodiscard]] static bool initialized();

    /**
     * @brief Check if the MPI environment is finalized.
     *
     * @return True if MPI_Finalize has been called.
     */
    [[nodiscard]] static bool is_finalized();

    /**
     * @brief Abort all MPI processes on MPI_COMM_WORLD.
     *
     * @param errcode Error code that will be passed to the MPI_Abort function.
     */
    static void abort(int errcode = 0);

    /**
     * @brief Clean up all MPI processes with a call to MPI_Finalize.
     */
    static void finalize();

    /**
     * @brief Constructor to initialize the MPI environment.
     *
     * @param argc The number of arguments passed to main() through argv.
     * @param argv The arguments passed to main() as an array of strings.
     * @param abort_on_exception If the environment is destroyed due to an uncaught exception,
     * it will call MPI_Abort instead of MPI_Finalize in its destructor.
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
     * @brief Clean up the MPI environment, either by calling MPI_Abort due to an uncaught exception
     * or by calling MPI_Finalize.
     */
    ~environment();

private:
    bool init_;
    bool abort_on_exception_;
};

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_ENVIRONMENT_HPP
