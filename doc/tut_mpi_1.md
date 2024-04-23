@page tut_mpi_1 MPI Tutorial 1: Hello world!

[TOC]

In this tutorial, we show how to implement the standard `Hello world` program using the MPI library from **simplemc**.

```cpp
#include <simplemc/mpi.hpp>
#include <fmt/core.h>

int main(int argc, char** argv) {
  // initialize MPI environment and communicator
  simplemc::mpi::environment env(argc, argv);
  simplemc::mpi::communicator comm;

  // print message on each process
  fmt::print("Hello world, from {} of {} processes.\n", comm.rank(), comm.size());
}
```

Output (depends on the number of processes and the order is arbitrary):

```
Hello world, from processor 2 of 4 processes.
Hello world, from processor 3 of 4 processes.
Hello world, from processor 0 of 4 processes.
Hello world, from processor 1 of 4 processes.
```
