/**
 * @file
 * @brief Implementation details for simplemc/mpi/environment.hpp.
 */

#include <simplemc/mpi/environment.hpp>
#include <simplemc/mpi/utils.hpp>

#include <mpi.h>

#include <exception>

namespace simplemc::mpi {

bool initialized() noexcept {
    int ini {};
    MPI_Initialized(&ini);
    return ini != 0;
}

bool finalized() noexcept {
    int fin {};
    MPI_Finalized(&fin);
    return fin != 0;
}

void init(int& argc, char**& argv) {
    if (!initialized()) {
        check_mpi_call(MPI_Init(&argc, &argv), "MPI_Init");
    }
}

int init_thread(int& argc, char**& argv, int required_thread_level) {
    if (!initialized()) {
        int provided {};
        check_mpi_call(MPI_Init_thread(&argc, &argv, required_thread_level, &provided), "MPI_Init_thread");
        return provided;
    }
    return query_thread();
}

void finalize() {
    if (!finalized()) {
        check_mpi_call(MPI_Finalize(), "MPI_Finalize");
    }
}

void abort(MPI_Comm comm, int errcode) {
    check_mpi_call(MPI_Abort(comm, errcode), "MPI_Abort");
}

int query_thread() {
    int provided {};
    check_mpi_call(MPI_Query_thread(&provided), "MPI_Query_thread");
    return provided;
}

bool is_thread_main() {
    int flag {};
    check_mpi_call(MPI_Is_thread_main(&flag), "MPI_Is_thread_main");
    return flag != 0;
}

environment::environment(int& argc, char**& argv, bool abort_on_exception) : abort_on_exception_(abort_on_exception) {
    init(argc, argv);
}

environment::environment(int& argc, char**& argv, int required_thread_level, bool abort_on_exception) :
    abort_on_exception_(abort_on_exception) {
    init_thread(argc, argv, required_thread_level);
}

environment::~environment() {
    if (!finalized()) {
        if (abort_on_exception_ && std::uncaught_exceptions()) {
            MPI_Abort(MPI_COMM_WORLD, -1);
        } else {
            MPI_Finalize();
        }
    }
}

} // namespace simplemc::mpi
