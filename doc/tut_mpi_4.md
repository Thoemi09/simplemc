@page tut_mpi_4 MPI Tutorial 4: Custom MPI datatype

[TOC]

In this tutorial, we show how to make your custom types compatible with the **simplemc-mpi** library.

@section tut_mpi_4_details Step-by-step guide

The following code snippets are all part of the same `main` function:

```cpp
#include <fmt/base.h>
#include <mpi.h>
#include <simplemc/mpi.hpp>

#include <array>
#include <type_traits>

// code snippets go here

int main(int argc, char** argv) {
    // initialize MPI environment and communicator
    simplemc::mpi::environment env(argc, argv);
    simplemc::mpi::communicator comm;

    // code snippets go here
}
```

Let us first define the custom type we want to make MPI compatible:

```cpp
// Custom struct.
struct foo {
    int n;
    double x;
};
```

It is a simple struct called `foo` with two members, an integer and a double.

In principle, everytime we want to communicate a `foo` object across processes, we could simply make
two MPI calls, one for the integer `n` and one for the double `x`. This can become tedious and quite
ineffective if we have a large range of `foo` objects.

Instead, we will show how to register a new MPI dataype and make it compatible with the
**simplemc-mpi** library.

To do this, we have to provide a template specialization for the simplemc::mpi::mpi_type struct that
maps a C++ type to its MPI datatype:

```cpp
// Specialize the simplemc::mpi::mpi_type for foo.
template <>
struct simplemc::mpi::mpi_type<foo> {
    static MPI_Datatype get() {
        // block lengths
        std::array<int, 2> block_lengths { 1, 1 };

        // displacements
        foo dummy {};
        std::array<MPI_Aint, 2> displ {};
        MPI_Aint base { 0 };
        MPI_Get_address(&dummy, &base);
        MPI_Get_address(&dummy.n, &displ[0]);
        MPI_Get_address(&dummy.x, &displ[1]);
        displ[0] = MPI_Aint_diff(displ[0], base);
        displ[1] = MPI_Aint_diff(displ[1], base);

        // MPI types
        std::array<MPI_Datatype, 2> types { MPI_INT, MPI_DOUBLE };

        // create a new MPI datatype for foo
        MPI_Datatype foo_type {};
        MPI_Type_create_struct(2, block_lengths.data(), displ.data(), types.data(), &foo_type);
        MPI_Type_commit(&foo_type);
        return foo_type;
    }
};
```

Here, we create and commit a new MPI datatype, which is then returned by the static `get()` function.
For more information on user-defined MPI types, we refer the interested reader to
[this link](https://rookiehpc.org/mpi/docs/mpi_type_create_struct/index.html).

Before the custom type can be send and received across processes, we also have to specialize the type
trait simplemc::mpi::is_mpi_datatype:

```cpp
// Specialize simplemc::mpi::is_mpi_datatype for foo.
template <>
struct simplemc::mpi::is_mpi_datatype<foo> : std::true_type {};
```

Once this is done, `foo` objects can be passed to various @ref simplemc-mpi-coll.

In the following, the process with rank 0 initializes an object and calls simplemc::mpi::broadcast to
send it to all other processes.

```cpp
// create a foo object on all processes and initialize it on process 0
foo f {};
if (comm.rank() == 0) {
    f = foo { .n = 42, .x = 3.14 };
}

// broadcast the foo object from process 0 to all other processes
simplemc::mpi::broadcast(comm, f, 0);
```

To confirm that everything worked as expected, we print the results:

```cpp
// print the foo object on all processes
fmt::println("Rank {}: foo.n = {}, foo.x = {}", comm.rank(), f.n, f.x);
```

Output:

```
Rank 0: foo.n = 42, foo.x = 3.14
Rank 3: foo.n = 42, foo.x = 3.14
Rank 1: foo.n = 42, foo.x = 3.14
Rank 2: foo.n = 42, foo.x = 3.14
```

@section tut_mpi_4_code Full code

```cpp
#include <fmt/base.h>
#include <mpi.h>
#include <simplemc/mpi.hpp>

#include <array>
#include <type_traits>

// Custom struct.
struct foo {
    int n;
    double x;
};

// Specialize the simplemc::mpi::mpi_type for foo.
template <>
struct simplemc::mpi::mpi_type<foo> {
    static MPI_Datatype get() {
        // block lengths
        std::array<int, 2> block_lengths { 1, 1 };

        // displacements
        foo dummy {};
        std::array<MPI_Aint, 2> displ {};
        MPI_Aint base { 0 };
        MPI_Get_address(&dummy, &base);
        MPI_Get_address(&dummy.n, &displ[0]);
        MPI_Get_address(&dummy.x, &displ[1]);
        displ[0] = MPI_Aint_diff(displ[0], base);
        displ[1] = MPI_Aint_diff(displ[1], base);

        // MPI types
        std::array<MPI_Datatype, 2> types { MPI_INT, MPI_DOUBLE };

        // create a new MPI datatype for foo
        MPI_Datatype foo_type {};
        MPI_Type_create_struct(2, block_lengths.data(), displ.data(), types.data(), &foo_type);
        MPI_Type_commit(&foo_type);
        return foo_type;
    }
};

// Specialize simplemc::mpi::is_mpi_datatype for foo.
template <>
struct simplemc::mpi::is_mpi_datatype<foo> : std::true_type {};

int main(int argc, char** argv) {
    // initialize MPI environment and communicator
    simplemc::mpi::environment env(argc, argv);
    simplemc::mpi::communicator comm;

    // create a foo object on all processes and initialize it on process 0
    foo f {};
    if (comm.rank() == 0) {
        f = foo { .n = 42, .x = 3.14 };
    }

    // broadcast the foo object from process 0 to all other processes
    simplemc::mpi::broadcast(comm, f, 0);

    // print the foo object on all processes
    fmt::println("Rank {}: foo.n = {}, foo.x = {}", comm.rank(), f.n, f.x);
}
```
