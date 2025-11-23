/**
 * @file
 * @brief Implementation details for simplemc/mpi/environment.hpp.
 */

#include <simplemc/mpi/environment.hpp>
#include <simplemc/mpi/utils.hpp>

#include <mpi.h>

#include <exception>

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

int environment::thread_support() {
    int provided {};
    check_mpi_call(MPI_Query_thread(&provided), "MPI_Query_thread");
    return provided;
}

bool environment::is_main_thread() {
    int flag {};
    check_mpi_call(MPI_Is_thread_main(&flag), "MPI_Is_thread_main");
    return flag != 0;
}

environment::environment(int& argc, char**& argv, bool abort_on_exception) :
    init_(false),
    abort_on_exception_(abort_on_exception) {
    if (!initialized()) {
        check_mpi_call(MPI_Init(&argc, &argv), "MPI_Init");
        init_ = true;
    }
}

environment::environment(int& argc, char**& argv, int required_thread_level, bool abort_on_exception) :
    init_(false),
    abort_on_exception_(abort_on_exception) {
    if (!initialized()) {
        int provided {};
        check_mpi_call(MPI_Init_thread(&argc, &argv, required_thread_level, &provided), "MPI_Init_thread");
        init_ = true;
    }
}

environment::~environment() {
    // should we also check the MPI calls in the destructor and potentially throw an exception?
    if (init_) {
        if (abort_on_exception_ && std::uncaught_exceptions()) {
            MPI_Abort(MPI_COMM_WORLD, -1);
        } else if (!is_finalized()) {
            MPI_Finalize();
        }
    }
}

} // namespace simplemc::mpi
