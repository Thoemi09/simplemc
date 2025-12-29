#include <fmt/base.h>
#include <simplemc/mpi.hpp>

int main(int argc, char** argv) {
    // initialize the MPI environment and the communicator
    simplemc::mpi::environment env { argc, argv };
    simplemc::mpi::communicator comm {};
    const int root = -1;

    // set the error handler to return errors instead of aborting
    simplemc::mpi::comm_set_errhandler(comm, MPI_ERRORS_RETURN);

    // broadcast a message
    int msg = (comm.rank() == 0) ? 42 : 0;
    simplemc::mpi::broadcast(msg, root, comm);

    // print the message on each process
    fmt::println("Process {} received message: {}", comm.rank(), msg);
}
