@page tut_mpi_1 MPI Tutorial 1: Hello world!

[TOC]

In this tutorial, we show how to implement the standard `Hello world` program using the
**simplemc-mpi** library.

@section tut_mpi_1_details Step-by-step guide

The following code snippets are all part of the same `main` function:

```cpp
#include <fmt/base.h>
#include <simplemc/mpi.hpp>

int main(int argc, char** argv) {
    // code snippets go here
}
```

The first thing one usually does in MPI programs is to initialize the MPI runtime environment and
construct an MPI communicator object:

```cpp
// initialize the MPI environment and the communicator
simplemc::mpi::environment env { argc, argv };
simplemc::mpi::communicator comm {};
```

Just like the environment in [Boost MPI](https://www.boost.org/doc/libs/1_85_0/doc/html/mpi.html),
simplemc::mpi::environment automatically initializes MPI in its constructor and finalizes it when
it goes out of scope.
There is usually no need for the user to do these basic operations manually.

The simplemc::mpi::communicator object is by default a wrapper around `MPI_COMM_WORLD`.
In order to wrap a different MPI communicator, the user can simply pass the `MPI_Comm` object to the
constructor of simplemc::mpi::communicator.
We will show in a later tutorial how this works in practice.

Now, let us check with how many processes we started the program:

```cpp
// print the number of processes on process 0
if (comm.rank() == 0) {
    fmt::println("Number of processes: {}", comm.size());
}
```

Output:

```
Number of processes: 4
```

Here, we are running with 4 processes.
The `if` condition makes sure that the line is printed only once on the root process.

Finally, we want to output the usual `Hello world`:

```cpp
// print message on each process
fmt::println("Hello world, from process {} out of {} processes.", comm.rank(), comm.size());
```

Output:

```
Hello world, from process 0 out of 4 processes.
Hello world, from process 3 out of 4 processes.
Hello world, from process 1 out of 4 processes.
Hello world, from process 2 out of 4 processes.
```

Since we are running on 4 processes, we get 4 messages.
Note that the output is arbitrary and changes from run to run.

@section tut_mpi_1_code Full code

@include tutorials/tut_mpi_1.cpp
