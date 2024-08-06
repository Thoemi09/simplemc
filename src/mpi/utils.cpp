/**
 * @file
 * @brief Implementation details for simplemc/mpi/utils.hpp.
 */

#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>
#include <mpi.h>

#include <string_view>

namespace simplemc::mpi {

void check_mpi_call(int errcode, std::string_view mpi_routine) {
    if (errcode != MPI_SUCCESS) {
        throw simplemc_exception(fmt::format("MPI error code: {}", errcode), mpi_routine);
    }
}

} // namespace simplemc::mpi
