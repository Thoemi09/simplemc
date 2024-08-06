@page tut_mpi_3 MPI Tutorial 3: Collective communication

[TOC]

In this tutorial, we show how to use some of the collective communications from the **simplemc-mpi**
library.

Note that the provided routines are implemented such that simple things can be done easily. If more
advanced functionality is needed or if performance of the MPI calls is important, it is recommended
to fall back to the MPI C library.

The following code snippets are all part of the same `main` function:

```cpp
#include <simplemc/mpi.hpp>

#include <fmt/ranges.h>

#include <string>
#include <vector>

// Print message only on rank 0.
void print(int rank, const std::string& msg) {
    if (rank == 0) {
        fmt::print("{}", msg);
    }
}

int main(int argc, char** argv) {
    // initialize MPI environment and communicator
    simplemc::mpi::environment env(argc, argv);
    simplemc::mpi::communicator comm;
    const int root = 0;

    // code snippets go here
}
```

The `print` function is simply for our convenience and prints a given message only on the process with
rank 0.
We further initialize the MPI environment and communicator and define the root process to be the one
with rank 0.

Let us first check how many processes we are using:

```cpp
// print the number of processes
print(comm.rank(), fmt::format("Number of processes: {}\n", comm.size()));
```

Output:

```
Number of processes: 4
```

In this case, we are running on 4 processes.

To demonstrate some of the functionality offered by the @ref simplemc-mpi-coll, we first calculate
the sum of all the MPI ranks using simplemc::mpi::reduce.

Since the number of processes is 4, the expected result is \f$ 3 + 2 + 1 + 0 = 6 \f$.
And in fact we are getting the correct answer:

```cpp
// get the sum of all ranks using all_reduce
int sum_of_ranks = 0;
simplemc::mpi::all_reduce(comm, comm.rank(), sum_of_ranks, MPI_SUM);
print(comm.rank(), fmt::format("Sum of ranks: {}\n", sum_of_ranks));
```

Output:

```
Sum of ranks: 6
```

Here, we called simplemc::mpi::all_reduce with the `MPI_SUM` operation.
There are many other operations that can be used for reductions.
For example, to find a maximum value among all process, one can pass `MPI_MAX` to the MPI call.

Let's use it to find the maximum rank of all process:

```cpp
// get the max. rank using all_reduce
int max_rank = 0;
simplemc::mpi::all_reduce(comm, comm.rank(), max_rank, MPI_MAX);
print(comm.rank(), fmt::format("Max. rank: {}\n", max_rank));
```

Output:

```
Max. rank: 3
```

Since we are running on 4 processes and ranks start at 0, this is exactly what we expect.

We now turn to the simplemc::mpi::all_gather function to gather values from all processes into a
`std::vector`.

First, we create a vector to use as a receive buffer into which the values are gathered.
Since we want to gather only a single value from each process, the vector has the size `comm.size()`.

Let's gather the rank of each process into the vector:

```cpp
// gather a single value from all processes into a vector
std::vector<int> vec(comm.size());
simplemc::mpi::all_gather(comm, comm.rank(), vec);
print(comm.rank(), fmt::format("Gathered vector: {}\n", vec));
```

Output:

```
Gathered vector: [0, 1, 2, 3]
```

> **Note**: The buffer into which we are gathering the values is required to have the correct size. If
> it is too big or too small an exception will be thrown.

Above it was already shown, how to reduce single values across all processes.
Now, we want to do the same with multiple values by reducing the just gathered vector using the
`MPI_PROD` operation:

```cpp
// reduce the gathered vector on root
std::vector<int> reduced_vec(comm.size());
simplemc::mpi::reduce(comm, vec, reduced_vec, MPI_PROD, root);
print(comm.rank(), fmt::format("Reduced vector: {}\n", reduced_vec));
```

Output:

```
Reduced vector: [0, 1, 16, 81]
```

Here, we used simplemc::mpi::reduce and not simplemc::mpi::all_reduce.
That means that the result is only available on the root process:

```cpp
// print the reduced vector on all processes except root
if (comm.rank() != root) {
    fmt::print("Reduced vector on rank {}: {}\n", comm.rank(), reduced_vec);
}
```

Output:

```
Reduced vector on rank 3: [0, 0, 0, 0]
Reduced vector on rank 1: [0, 0, 0, 0]
Reduced vector on rank 2: [0, 0, 0, 0]
```

There are 2 possibilities to distribute the results to the other processes:
- If only part of the vector (in this case a single value) is needed on the other processes, we can
use simplemc::mpi::scatter.
- If the full vector is needed on every process, we can use simplemc::mpi::broadcast.

Let us first show how the scattering works:

```cpp
// scatter the reduced vector back to all processes
int scattered_value = 0;
simplemc::mpi::scatter(comm, reduced_vec, scattered_value, root);
fmt::print("Scattered value on rank {}: {}\n", comm.rank(), scattered_value);
```

Output:

```
Scattered value on rank 0: 0
Scattered value on rank 1: 1
Scattered value on rank 2: 16
Scattered value on rank 3: 81
```

As expected, every rank receives exactly one value.

What about broadcasting? Now every rank gets the full vector:

```cpp
// broadcast the reduced vector from root to all processes
simplemc::mpi::broadcast(comm, reduced_vec, root);
fmt::print("Broadcasted vector on rank {}: {}\n", comm.rank(), reduced_vec);
```

Output:

```
Broadcasted vector on rank 1: [0, 1, 16, 81]
Broadcasted vector on rank 0: [0, 1, 16, 81]
Broadcasted vector on rank 2: [0, 1, 16, 81]
Broadcasted vector on rank 3: [0, 1, 16, 81]
```

> **Note**: When running the code, the output might appear scrambled. One can try to use
> `comm.barrier()` after the print statements to disentangle them.
