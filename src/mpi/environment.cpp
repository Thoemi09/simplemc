/**
 * @file environment.cpp
 * @brief RAII MPI environment to handle initialization and finalization of MPI programs.
 */

#include <simplemc/mpi/environment.hpp>
#include <simplemc/mpi/utils.hpp>

namespace simplemc::mpi {

bool environment::initialized() {
    int ini {};
    check_mpi_call(MPI_Initialized(&ini), "MPI_Intialized");
    return ini != 0;
}

bool environment::is_finalized() {
    int fin {};
    check_mpi_call(MPI_Finalized(&fin), "MPI_Finalized");
    return fin != 0;
}

void environment::abort(int errcode) {
    check_mpi_call(MPI_Abort(MPI_COMM_WORLD, errcode), "MPI_Abort");
}

void environment::finalize() {
    if (!is_finalized()) {
        check_mpi_call(MPI_Finalize(), "MPI_Finalize");
    }
}

environment::environment(int& argc, char**& argv, bool abort_on_exception) :
    init_(false),
    abort_on_exception_(abort_on_exception) {
    if (!initialized()) {
        check_mpi_call(MPI_Init(&argc, &argv), "MPI_Init");
        init_ = true;
    }
}

environment::~environment() {
    // should we also check the MPI calls in the destructor and potentially
    // throw an exception?
    if (init_) {
        if (abort_on_exception_ && std::uncaught_exceptions()) {
            MPI_Abort(MPI_COMM_WORLD, -1);
        } else if (!is_finalized()) {
            MPI_Finalize();
        }
    }
}

} // namespace simplemc::mpi
