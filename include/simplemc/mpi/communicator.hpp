/**
 * @file
 * @brief Wrapper for an MPI_Comm MPI communicator.
 */

#ifndef SIMPLEMC_MPI_COMMUNICATOR_HPP
#define SIMPLEMC_MPI_COMMUNICATOR_HPP

#include <mpi.h>

namespace simplemc::mpi {

/**
 * @ingroup simplemc-mpi-commenv
 * @brief A communicator that enables communication and synchronization among a set of MPI processes.
 *
 * @details It is implemented as a very thin wrapper around the actual `MPI_Comm` object. Only the
 * most basic functionality is supported (`MPI_Comm_rank`, `MPI_Comm_size`, `MPI_Barrier`).
 *
 * For more advanced tasks, the user is advised to use the MPI C library directly.
 */
class communicator {
public:
    /**
     * @brief Construct a new communicator instance.
     *
     * @details The given `MPI_Comm` is copied and stored in this communicator instance. By default,
     * i.e. if no `MPI_Comm` object is provided, `MPI_COMM_WORLD` is used. Implicit conversions from
     * `MPI_Comm` objects to communicator objects are allowed.
     *
     * @note No call to `MPI_Comm_dup` is made. If this is needed, it is the responsibility of the
     * user to do so.
     *
     * @param comm MPI_Comm object.
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
     * @brief Implicit conversion to the underlying `MPI_Comm` object.
     */
    operator MPI_Comm() const { return comm_; }

private:
    MPI_Comm comm_;
};

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_COMMUNICATOR_HPP
