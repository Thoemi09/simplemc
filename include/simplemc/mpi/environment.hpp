/**
 * @file
 * @brief RAII MPI environment and other essential MPI functions.
 */

#ifndef SIMPLEMC_MPI_ENVIRONMENT_HPP
#define SIMPLEMC_MPI_ENVIRONMENT_HPP

#include <mpi.h>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-essentials
 */

/**
 * @brief RAII MPI execution environment.
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
 *
 *     // use MPI
 *     ...
 * }
 * @endcode
 *
 * See @ref tut_mpi_1 and @ref tut_mpi_2 for full examples.
 */
class environment {
public:
    /**
     * @brief Constructor initializes the MPI environment.
     *
     * @details It calls simplemc::mpi::init with the given arguments.
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
     * @brief Constructor initializes the MPI environment with threading support.
     *
     * @details It calls simplemc::mpi::init_thread with the given arguments and requested thread
     * level.
     *
     * See @ref tut_mpi_2 for an example on how `abort_on_exception` can be used.
     *
     * @param argc Number of arguments passed to `main()`.
     * @param argv Arguments passed to `main()`.
     * @param required_thread_level The desired level of thread support.
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
     * @brief Clean up the MPI environment.
     *
     * @details If the MPI environment has not been simplemc::mpi::finalized, it either calls
     * `MPI_Abort` due to an uncaught exception (if `abort_on_exception` is set to `true`) or
     * `MPI_Finalize`.
     * 
     * @note It only performs any action if the current instance was responsible for initializing MPI.
     */
    ~environment();

private:
    bool init_;
    bool abort_on_exception_;
};

/**
 * @brief Check if the MPI environment has been initialized by calling `MPI_Initalized`.
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
 * It checks the success of the call with simplemc::mpi::check_mpi_call.
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
 * @param argc Number of arguments passed to `main()`.
 * @param argv Arguments passed to `main()`.
 * @param required_thread_level The desired level of thread support.
 * @return The level of thread support provided.
 */
int init_thread(int& argc, char**& argv, int required_thread_level);

/**
 * @brief Clean up all MPI processes with a call to `MPI_Finalize`.
 *
 * @details It first checks if MPI has already been finalized. If not, it calls `MPI_Finalize`.
 *
 * It checks the success of the call with simplemc::mpi::check_mpi_call.
 */
void finalize();

/**
 * @brief Abort all MPI processes on `MPI_COMM_WORLD`.
 *
 * @details It makes a call to `MPI_Abort` and checks the success of the call with
 * simplemc::mpi::check_mpi_call.
 *
 * @param errcode Error code that will be passed to the `MPI_Abort` function.
 */
void abort(int errcode = 0);

/**
 * @brief Query the level of thread support provided by the MPI implementation.
 *
 * @details It makes a call to `MPI_Query_thread` and checks the success of the call with
 * simplemc::mpi::check_mpi_call.
 *
 * @return The level of thread support provided. This will be one of `MPI_THREAD_SINGLE`,
 * `MPI_THREAD_FUNNELED`, `MPI_THREAD_SERIALIZED`, or `MPI_THREAD_MULTIPLE`.
 */
[[nodiscard]] int query_thread();

/**
 * @brief Determine if this is the main thread.
 *
 * @details It makes a call to `MPI_Is_thread_main` and checks the success of the call with
 * simplemc::mpi::check_mpi_call.
 *
 * This function is useful in multi-threaded MPI programs to determine whether the calling thread
 * is the main thread (i.e., the thread that initialized MPI).
 *
 * @return True if the calling thread is the main thread.
 */
[[nodiscard]] bool is_thread_main();

/**
 * @brief Set the error handler for `MPI_COMM_WORLD`.
 *
 * @details It makes a call to `MPI_Comm_set_errhandler` to set the error handler for
 * `MPI_COMM_WORLD` checks the success of the call with simplemc::mpi::check_mpi_call.
 *
 * Common error handlers are:
 * - `MPI_ERRORS_ARE_FATAL`: MPI will abort on error (default).
 * - `MPI_ERRORS_RETURN`: MPI will return an error code instead of aborting.
 * - `MPI_ERRORS_ABORT`: MPI will abort on error (similar to `MPI_ERRORS_ARE_FATAL`).
 *
 * @param errhandler The MPI error handler to set.
 */
void set_errhandler(MPI_Errhandler errhandler);

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_ENVIRONMENT_HPP
