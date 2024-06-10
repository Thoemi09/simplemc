@page tut_mpi_1 MPI Tutorial 1: Hello world!

[TOC]

In this tutorial, we show how to implement the standard `Hello world` program using the
**simplemc-mpi** library.

Just like the environment in [Boost MPI](https://www.boost.org/doc/libs/1_85_0/doc/html/mpi.html),
simplemc::mpi::environment automatically initializes MPI in its constructor and finializes it when
it goes out of scope.
There is usually no need for the user to do these basic operations manually.

```cpp
#include <simplemc/mpi.hpp>

#include <fmt/core.h>

int main(int argc, char** argv) {
    // initialize MPI environment and communicator
    simplemc::mpi::environment env(argc, argv);
    simplemc::mpi::communicator comm;

    // print message on each process
    fmt::print("Hello world, from process {} out of {} processes.\n", comm.rank(), comm.size());
}
```

Output (depends on the number of processes and the order is arbitrary):

```
Hello world, from process 3 out of 4 processes.
Hello world, from process 2 out of 4 processes.
Hello world, from process 1 out of 4 processes.
Hello world, from process 0 out of 4 processes.
```
