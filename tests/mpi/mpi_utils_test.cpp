#include <gtest/gtest.h>
#include <mpi.h>
#include <simplemc/utils/simplemc_exception.hpp>
#include <simplemc/mpi/utils.hpp>

// Test simplemc::mpi::check_mpi_call with MPI_SUCCESS.
TEST(SimplemcMPI, CheckMPICallSuccess) {
    EXPECT_NO_THROW(simplemc::mpi::check_mpi_call(MPI_SUCCESS, "CheckMPICallSuccess"));
}

// Test simplemc::mpi::check_mpi_call with no MPI_SUCCESS.
TEST(SimplemcMPI, CheckMPICallNoSuccess) {
    EXPECT_THROW(simplemc::mpi::check_mpi_call(MPI_SUCCESS - 1, "CheckMPICallNoSuccess"), simplemc::simplemc_exception);
}
