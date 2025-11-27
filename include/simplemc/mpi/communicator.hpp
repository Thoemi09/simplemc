/**
 * @file
 * @brief Wrapper around an `MPI_Comm` MPI communicator.
 */

#ifndef SIMPLEMC_MPI_COMMUNICATOR_HPP
#define SIMPLEMC_MPI_COMMUNICATOR_HPP

#include <mpi.h>

namespace simplemc::mpi {

/**
 * @ingroup simplemc-mpi-essentials
 * @brief A communicator that enables communication and synchronization among a set of MPI processes.
 *
 * @details It is implemented as a very thin wrapper around the actual `MPI_Comm` object. Only the
 * most basic functionality is supported:
 * - `MPI_Comm_rank` - see rank(),
 * - `MPI_Comm_size` - see size(),
 * - `MPI_Barrier` - see barrier(),
 * - `MPI_Comm_dup` - see duplicate() and
 * - `MPI_Comm_free` - see free().
 *
 * Implicit conversions to and from `MPI_Comm` objects are allowed.
 *
 * For more advanced tasks, the user is advised to use the MPI C library directly.
 *
 * @code{.cpp}
 * int main(int argc, char** argv) {
 *     ...
 *     // initialize MPI
 *     simplemc::mpi::environment env(argc, argv);
 *     simplemc::mpi::communicator comm;
 *     ...
 * }
 * @endcode
 *
 * See @ref tut_mpi_1 for a full example.
 */
class communicator {
public:
    /**
     * @brief Construct a new communicator instance.
     *
     * @details The given `MPI_Comm` is copied and stored in this communicator instance. By default,
     * i.e. if no `MPI_Comm` object is provided, `MPI_COMM_WORLD` is used.
     *
     * @note No call to `MPI_Comm_dup` is made. If this is needed, it is the responsibility of the
     * user to do so (see duplicate()).
     *
     * @param comm `MPI_Comm` object.
     */
    communicator(MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * @brief Determine the rank of the calling process.
     *
     * @details Makes a call to `MPI_Comm_rank` and throws an exception if the call fails.
     *
     * @return Rank in the current communicator.
     */
    [[nodiscard]] int rank() const;

    /**
     * @brief Determine the size of the communicator.
     *
     * @details Makes a call to `MPI_Comm_size` and throws an exception if the call fails.
     *
     * @return Number of processes in the current communicator.
     */
    [[nodiscard]] int size() const;

    /**
     * @brief Wait for all processes within the communicator to reach this barrier.
     *
     * @details Makes a call to `MPI_Barrier` and throws an exception if the call fails.
     */
    void barrier() const;

    /**
     * @brief Duplicate the communicator.
     *
     * @details Makes a call to `MPI_Comm_dup` to duplicate the communicator. See the MPI
     * documentation for more details, e.g.
     * <a href="https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/MPI_Comm_dup.3.html">
     * open-mpi docs</a>.
     *
     * Throws an exception if the call fails
     *
     * @warning This allocates a new communicator object. Make sure to call free() on the returned
     * communicator when it is no longer needed.
     *
     * @return Duplicated `MPI_Comm` object wrapped in a new simplemc::mpi::communicator.
     */
    [[nodiscard]] communicator duplicate() const;

    /**
     * @brief Free the communicator.
     *
     * @details Makes a call to `MPI_Comm_free` to mark the communicator for deallocation. See the MPI
     * documentation for more details, e.g.
     * <a href="https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/MPI_Comm_free.3.html">
     * open-mpi docs</a>.
     *
     * Throws an exception if the call fails.
     */
    void free();

    /**
     * @brief Implicit conversion function to the underlying `MPI_Comm` object.
     */
    operator MPI_Comm() const { return comm_; }

private:
    MPI_Comm comm_;
};

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_COMMUNICATOR_HPP
