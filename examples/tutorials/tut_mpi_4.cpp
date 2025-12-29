#include <fmt/base.h>
#include <mpi.h>
#include <simplemc/mpi.hpp>

#include <array>

// Custom type.
struct foo {
    int n { 0 };
    double x { 0.0 };
};

int main(int argc, char** argv) {
    // initialize MPI environment and communicator
    simplemc::mpi::environment env { argc, argv };
    simplemc::mpi::communicator comm {};
    const int root = 0;

    // create a foo object on all processes and initialize it on process 0
    foo f {};
    if (comm.rank() == root) {
        f = foo { .n = 42, .x = 3.14 };
    }

    // print the foo object before broadcasting
    fmt::println("Rank {}: foo.n = {}, foo.x = {}", comm.rank(), f.n, f.x);

    // block lengths, displacements and MPI types for each member of foo
    using simplemc::mpi::get_displacement;
    std::array<int, 2> block_lengths { 1, 1 };
    std::array<MPI_Aint, 2> displs { get_displacement(f.n, f), get_displacement(f.x, f) };
    std::array<MPI_Datatype, 2> types { MPI_INT, MPI_DOUBLE };

    // create the MPI datatype for foo
    auto foo_type = simplemc::mpi::datatype { block_lengths, displs, types };

    // broadcast the foo object
    simplemc::mpi::broadcast(&f, 1, foo_type, root, comm);

    // print the foo object after broadcasting
    fmt::println("Rank {}: foo.n = {}, foo.x = {}", comm.rank(), f.n, f.x);
}
