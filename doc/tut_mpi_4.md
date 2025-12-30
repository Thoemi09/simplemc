@page tut_mpi_4 MPI Tutorial 4: Custom MPI datatype

[TOC]

In this tutorial, we show how to create custom MPI types with the **simplemc-mpi** library.

@section tut_mpi_4_details Step-by-step guide

The following code snippets are all part of the same `main` function:

```cpp
#include <fmt/base.h>
#include <mpi.h>
#include <simplemc/mpi.hpp>

#include <array>

// custom type definition goes here

int main(int argc, char** argv) {
    // initialize MPI environment and communicator
    simplemc::mpi::environment env { argc, argv };
    simplemc::mpi::communicator comm {};
    const int root = 0;

    // code snippets go here
}
```

@subsection tut_mpi_4_cpp_type Custom C++ type

Let us first define the custom C++ type:

```cpp
// Custom struct.
struct foo {
    int n { 0 };
    double x { 0.0 };
};
```

It is a simple struct called `foo` with two members, an integer `n` and a double `x`.

We can create a foo object on all processes and initialize it on the root process:

```cpp
// create a foo object on all processes and initialize it on process 0
foo f {};
if (comm.rank() == root) {
    f = foo { .n = 42, .x = 3.14 };
}

// print the foo object before broadcasting
fmt::println("Rank {}: foo.n = {}, foo.x = {}", comm.rank(), f.n, f.x);
```

Output:

```
Rank 2: foo.n = 0, foo.x = 0
Rank 1: foo.n = 0, foo.x = 0
Rank 3: foo.n = 0, foo.x = 0
Rank 0: foo.n = 42, foo.x = 3.14
```

Now suppose that we want to broadcast the `foo` object from the root process to all other processes.

In principle, we could simply make two MPI calls, one for the integer `n` and one for the double `x`.
This can become tedious and quite ineffective if we have a large range of `foo` objects or if we have
to use other MPI collective operations.

Instead, we will show how to create and use a new MPI datatype with the **simplemc-mpi** library.

@subsection tut_mpi_4_mpi_type Custom MPI datatype

Creating MPI datatypes for custom C++ types requires a little bit of information about the memory
layout of the C++ type:

```cpp
// block lengths, displacements and MPI types for each member of foo
using simplemc::mpi::get_displacement;
std::array<int, 2> block_lengths { 1, 1 };
std::array<MPI_Aint, 2> displs { get_displacement(f.n, f), get_displacement(f.x, f) };
std::array<MPI_Datatype, 2> types { MPI_INT, MPI_DOUBLE };
```

Here, we define three arrays:
- `block_lengths`: The number of elements for each member of the struct. `foo` has 1 integer and 1
double member.
- `displs`: The displacements (in bytes) of each member from the start of the struct. We use the
simplemc::mpi::get_displacement utility function to compute these displacements.
- `types`: The MPI datatypes corresponding to each member of the struct. Again, there is one `MPI_INT`
and one `MPI_DOUBLE`.

Once we have this information, we can create the MPI datatype:

```cpp
// create the MPI datatype for foo
auto foo_type = simplemc::mpi::datatype { block_lengths, displs, types };
```

simplemc::mpi::datatype is a RAII wrapper around `MPI_Datatype` that handles the creation and
destruction of the MPI datatype.

Under the hood, it calls simplemc::mpi::type_create_struct and simplemc::mpi::type_commit in its
constructor, and simplemc::mpi::type_free in its destructor.

@subsection tut_mpi_4_coll Collective communication with custom MPI datatype

Now that we have an MPI datatype for `foo`, we can use the type in collective communication
operations.

For example, let's broadcast our previously constructed `foo` object from the root to all other
processes:

```cpp
// broadcast the foo object
simplemc::mpi::broadcast(&f, 1, foo_type, root, comm);

// print the foo object after broadcasting
fmt::println("Rank {}: foo.n = {}, foo.x = {}", comm.rank(), f.n, f.x);
```

Output:

```
Rank 0: foo.n = 42, foo.x = 3.14
Rank 2: foo.n = 42, foo.x = 3.14
Rank 1: foo.n = 42, foo.x = 3.14
Rank 3: foo.n = 42, foo.x = 3.14
```

Here, we have to use the most general form of simplemc::mpi::broadcast that simply forwards the
arguments to `MPI_Bcast`.
The reason for this is that all higher-level overloads of simplemc::mpi::broadcast (as well as other
collective operations) only work for simplemc::mpi::mpi_compatible and simplemc::mpi::mpi_range types.

> **Note:** The user is allowed to make their C++ types simplemc::mpi::mpi_compatible by providing a
> template specialization of simplemc::mpi::mpi_type. We do not show the details here, but it may
> look something like this:
> ```cpp
> template <>
> struct simplemc::mpi::mpi_type<foo> {
>     static MPI_Datatype get() {
>         static MPI_Datatype foo_type = make_foo_type();
>         return foo_type;;
>     }
> };
> ```
> Here, `make_foo_type` is a function that creates, commits and returns the MPI datatype for `foo`.

@section tut_mpi_4_code Full code

@include tutorials/tut_mpi_4.cpp
