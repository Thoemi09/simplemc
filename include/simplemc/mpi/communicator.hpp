/**
 * @file
 * @brief Wrapper for an MPI_Comm MPI communicator.
 */

#ifndef SIMPLEMC_MPI_COMMUNICATOR_HPP
#define SIMPLEMC_MPI_COMMUNICATOR_HPP

#include <mpi.h>

namespace simplemc::mpi {

/**
 * @brief A communicator that enables communication and synchronization among a set of
 * MPI processes.
 *
 * @details It is implemented as a very thin wrapper around the actual MPI_Comm object.
 * Only the most basic functionality is supported (MPI_Comm_rank, MPI_Comm_size, MPI_Barrier).
 */
class communicator {
public:
    /**
     * @brief Construct a new communicator instance.
     *
     * @details The given MPI_Comm is copied and stored in this communicator instance. Note that
     * no call to MPI_Comm_dup is made. If this is needed, it is the responsibility of the user
     * to do so. The default constructor creates a copy of MPI_COMM_WORLD. Implicit conversions
     * from MPI_Comm objects to communicator objects are allowed.
     *
     * @param comm MPI_Comm object.
     */
    communicator(MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * @brief Determine the rank of the calling process.
     *
     * @return Rank in the current communicator.
     */
    [[nodiscard]] int rank() const;

    /**
     * @brief Determine the size of the communicator.
     *
     * @return Number of processes in the current communicator.
     */
    [[nodiscard]] int size() const;

    /**
     * @brief Wait for all processes within the communicator to reach this barrier.
     */
    void barrier() const;

    /**
     * @brief Implicit conversion to the underlying MPI_Comm object.
     */
    operator MPI_Comm() const { return comm_; }

private:
    MPI_Comm comm_;
};

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_COMMUNICATOR_HPP
