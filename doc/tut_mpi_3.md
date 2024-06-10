@page tut_mpi_3 MPI Tutorial 3: Collective communication

[TOC]

In this tutorial, we show how to use some of the collective communications from the **simplemc-mpi**
library.

Note that the provided routines are implemented such that simple things can be done easily. If more
advanced functionality is needed or if performance of the MPI calls is important, it is recommended
to fall back to the MPI C library.

```cpp
#include <simplemc/mpi.hpp>

#include <fmt/ranges.h>

#include <vector>

// Print message only on rank 0.
void print(int rank, const std::string& msg) {
    if (rank == 0) {
        fmt::print("{}", msg);
    }
}

int main(int argc, char** argv) {
    // initialize MPI environment and communicator
    simplemc::mpi::environment env(argc, argv, /* abort_on_exception */ true);
    simplemc::mpi::communicator comm;
    const int root = 0;

    print(comm.rank(), fmt::format("Running on {} processes:\n\n", comm.size()));

    // reduce a single value on all processes
    int sum_of_ranks = 0;
    simplemc::mpi::all_reduce(comm, comm.rank(), sum_of_ranks, MPI_SUM);
    print(comm.rank(), fmt::format("Sum of ranks: {}\n", sum_of_ranks));

    int max_rank = 0;
    simplemc::mpi::all_reduce(comm, comm.rank(), max_rank, MPI_MAX);
    print(comm.rank(), fmt::format("Max. rank: {}\n", max_rank));

    // gather a single value from all processes
    std::vector<int> vec(comm.size());
    simplemc::mpi::all_gather(comm, comm.rank(), vec);
    print(comm.rank(), fmt::format("Gathered vector: {}\n", vec));

    // reduce the gathered vector on root
    std::vector<int> reduced_vec(comm.size());
    simplemc::mpi::reduce(comm, vec, reduced_vec, MPI_PROD, root);
    print(comm.rank(), fmt::format("Reduced vector: {}\n", reduced_vec));

    // scatter the reduced vector back to all processes
    int scattered_value = 0;
    simplemc::mpi::scatter(comm, reduced_vec, scattered_value, root);
    fmt::print("Scattered value on rank {}: {}\n", comm.rank(), scattered_value);
}

```

Output using 4 processess:

```
Running on 4 processes:

Sum of ranks: 6
Max. rank: 3
Gathered vector: [0, 1, 2, 3]
Reduced vector: [0, 1, 16, 81]
Scattered value on rank 0: 0
Scattered value on rank 1: 1
Scattered value on rank 3: 81
Scattered value on rank 2: 16
```
