// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief RAII wrapper for an MPI environment and other essential MPI functions.
 */

#ifndef SIMPLEMC_MPI_ENVIRONMENT_HPP
#define SIMPLEMC_MPI_ENVIRONMENT_HPP

#include <mpi.h>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-essentials-env
 * @{
 */

/**
 * @brief Check if the MPI environment has been initialized by calling `MPI_Initialized`.
 *
 * @return True if `MPI_Init` has been called.
 */
[[nodiscard]] bool initialized() noexcept;

/**
 * @brief Check if the MPI environment has been finalized by calling `MPI_Finalized`.
 *
 * @return True if `MPI_Finalize` has been called.
 */
[[nodiscard]] bool finalized() noexcept;

/**
 * @brief Initialize the MPI environment by calling `MPI_Init`.
 *
 * @details If the MPI environment has not been simplemc::mpi::initialized, it calls `MPI_Init` with
 * the given arguments.
 *
 * @note The caller is also responsible for finalizing the MPI environment by calling
 * simplemc::mpi::finalize or `MPI_Finalize()`.
 *
 * @param argc Number of arguments passed to `main()`.
 * @param argv Arguments passed to `main()`.
 */
void init(int& argc, char**& argv);

/**
 * @brief Initialize the MPI environment by calling `MPI_Init`.
 *
 * @details If the MPI environment has not been simplemc::mpi::initialized, it calls `MPI_Init_thread`
 * with the given arguments and the requested threading level.
 *
 * If the MPI environment is already initialized, it simply calls simplemc::mpi::query_thread.
 *
 * MPI supports four levels of thread support:
 * - `MPI_THREAD_SINGLE`: Only one thread will execute.
 * - `MPI_THREAD_FUNNELED`: The process may be multi-threaded, but only the main thread will make
 * MPI calls.
 * - `MPI_THREAD_SERIALIZED`: The process may be multi-threaded, and multiple threads may make MPI
 * calls, but only one at a time.
 * - `MPI_THREAD_MULTIPLE`: Multiple threads may call MPI, with no restrictions.
 *
 * @note The caller is also responsible for finalizing the MPI environment by calling
 * simplemc::mpi::finalize or `MPI_Finalize()`.
 *
 * @param argc Number of arguments passed to `main()`.
 * @param argv Arguments passed to `main()`.
 * @param required_thread_level Desired level of thread support.
 * @return Provided level of thread support.
 */
int init_thread(int& argc, char**& argv, int required_thread_level);

/**
 * @brief Clean up all MPI processes by calling `MPI_Finalize`.
 *
 * @details If the MPI environment has not been simplemc::mpi::finalized, it calls `MPI_Finalize`.
 */
void finalize();

/**
 * @brief Abort all MPI processes in the given MPI communicator.
 *
 * @details It simply calls `MPI_Abort` with the given communicator and error code.
 *
 * @param comm Communicator that contains all processes to abort. Default is `MPI_COMM_WORLD`.
 * @param errcode Error code to return to invoking environment.
 */
void abort(MPI_Comm comm = MPI_COMM_WORLD, int errcode = 0);

/**
 * @brief Query the level of thread support provided by the MPI implementation by calling
 * `MPI_Query_thread`.
 *
 * @details It returns one of the following levels: `MPI_THREAD_SINGLE`, `MPI_THREAD_FUNNELED`,
 * `MPI_THREAD_SERIALIZED`, or `MPI_THREAD_MULTIPLE`.
 *
 * See simplemc::mpi::init_thread for details on the different levels.
 *
 * @return Provided level of thread support.
 */
[[nodiscard]] int query_thread();

/**
 * @brief Query if this is the main thread in a multi-threaded MPI program by calling
 * `MPI_Is_thread_main`.
 *
 * @return True if the calling thread is the main thread, i.e., the thread that initialized MPI.
 */
[[nodiscard]] bool is_thread_main();

/**
 * @brief RAII wrapper for an MPI environment.
 *
 * @details This class provides a thin, non-copyable C++ wrapper that manages the initialization and
 * finalization of MPI.
 *
 * It is intended to be used in the `main()` function of a program, where it calls `MPI_Init` on
 * construction and `MPI_Finalize` on destruction:
 *
 * @code{.cpp}
 * int main(int argc, char** argv) {
 *     ...
 *     // initialize MPI
 *     simplemc::mpi::environment env(argc, argv);
 *
 *     // use MPI
 *     ...
 * }
 * @endcode
 */
class environment {
public:
    /**
     * @brief Constructor initializes the MPI environment.
     *
     * @details It calls simplemc::mpi::init with the given arguments.
     *
     * See @ref tut_mpi_2 for more information on the flag `abort_on_exception`.
     *
     * @param argc Number of arguments passed to `main()`.
     * @param argv Arguments passed to `main()`.
     * @param abort_on_exception If the environment is destroyed due to an uncaught exception, it will
     * call `MPI_Abort` instead of `MPI_Finalize` in its destructor.
     */
    environment(int& argc, char**& argv, bool abort_on_exception = true);

    /**
     * @brief Constructor initializes the MPI environment with threading support.
     *
     * @details It calls simplemc::mpi::init_thread with the given arguments and requested thread
     * level.
     *
     * See @ref tut_mpi_2 for more information on the flag `abort_on_exception`.
     *
     * @param argc Number of arguments passed to `main()`.
     * @param argv Arguments passed to `main()`.
     * @param required_thread_level Desired level of thread support.
     * @param abort_on_exception If the environment is destroyed due to an uncaught exception, it will
     * call `MPI_Abort` instead of `MPI_Finalize` in its destructor.
     */
    environment(int& argc, char**& argv, int required_thread_level, bool abort_on_exception = true);

    /**
     * @brief Copy constructor is deleted.
     */
    environment(const environment&) = delete;

    /**
     * @brief Copy assignment operator is deleted.
     */
    environment& operator=(const environment&) = delete;

    /**
     * @brief Destructor finalizes the MPI environment.
     *
     * @details If the MPI environment has not been simplemc::mpi::finalized, it either calls
     * - `MPI_Abort` due to uncaught exceptions and if `abort_on_exception` was set to `true` or
     * - `MPI_Finalize`.
     */
    ~environment();

private:
    bool abort_on_exception_;
};

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_ENVIRONMENT_HPP
