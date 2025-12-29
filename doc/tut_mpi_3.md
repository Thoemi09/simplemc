@page tut_mpi_3 MPI Tutorial 3: Collective communication

[TOC]

In this tutorial, we show how to use some of the @ref simplemc-mpi-coll from the **simplemc-mpi**
library.

The provided routines are implemented such that simple things can be done easily. 
If more advanced functionality is needed or if performance of the MPI calls is important, it is 
recommended to fall back to more low-level routines (see @ref simplemc-mpi-coll) or the MPI C library.

> **Note**: When running the code, the output might appear scrambled and has been rearranged to make
> it easier to read and follow along.

@section tut_mpi_3_details Step-by-step guide

The following code snippets are all part of the same `main` function:

```cpp
#include <fmt/ranges.h>
#include <simplemc/mpi.hpp>

#include <vector>

int main(int argc, char** argv) {
    // initialize MPI environment and communicator
    simplemc::mpi::environment env { argc, argv };
    simplemc::mpi::communicator comm {};
    const int root = 0;

    // code snippets go here
}
```

We start by initializing the MPI environment and communicator and by defining the root process to be
the one with rank 0.

Let us first check how many processes we are using:

```cpp
// print the number of processes
if (comm.rank() == root) {
    fmt::println("Number of processes: {}", comm.size());
}
```

Output:

```
Number of processes: 4
```

In this case, we are running on 4 processes.

To demonstrate some of the functionality offered by the @ref simplemc-mpi-coll, we first calculate
the sum of all the MPI ranks using simplemc::mpi::all_reduce.

Since the number of processes is 4, the expected result is \f$ 3 + 2 + 1 + 0 = 6 \f$.
And in fact we are getting the correct answer:

```cpp
// get the sum of all ranks using all_reduce
int sum_of_ranks = 0;
simplemc::mpi::all_reduce(comm.rank(), sum_of_ranks, MPI_SUM, comm);
fmt::println("Rank {}: Sum of ranks: {}", comm.rank(), sum_of_ranks);
```

Output:

```
Rank 0: Sum of ranks: 6
Rank 2: Sum of ranks: 6
Rank 3: Sum of ranks: 6
Rank 1: Sum of ranks: 6
```

Here, we called simplemc::mpi::all_reduce with the `MPI_SUM` operation.
There are many other operations that can be used for reductions.
For example, to find a maximum value among all process, one can pass `MPI_MAX` to the MPI call.

Let's use it to find the maximum rank of all process:

```cpp
// get the max. rank using all_reduce_in_place
int max_rank = comm.rank();
simplemc::mpi::all_reduce_in_place(max_rank, MPI_MAX, comm);
fmt::println("Rank {}: Max. rank: {}", comm.rank(), max_rank);
```

Output:

```
Rank 3: Max. rank: 3
Rank 0: Max. rank: 3
Rank 2: Max. rank: 3
Rank 1: Max. rank: 3
```

Since we are running on 4 processes and ranks start at 0, this is exactly what we expect.
In contrast to the last MPI call, we have used the in place routine
simplemc::mpi::all_reduce_in_place.
This writes the result of the reduction directly into the provided variable (`max_rank` in our case).

We now turn to the simplemc::mpi::gather function to gather values from all processes into a
`std::vector` on the root process.

First, we create a vector to use as a receive buffer into which the values are gathered.
Since we want to gather only a single value from each process, the vector has the size `comm.size()`.

> **Note**: The buffer into which we are gathering the values is required to be big enough. In 
> general, **simplemc-mpi** does not perform any checks and relies on the user and the MPI C library.

Let's gather the rank of each process into the vector:

```cpp
// gather a single value from all processes into a vector on root
std::vector<int> vec(comm.size());
simplemc::mpi::gather(comm.rank(), vec, root, comm);
fmt::println("Rank {}: Gathered vector: {}", comm.rank(), vec);
```

Output:

```
Rank 3: Gathered vector: [0, 0, 0, 0]
Rank 2: Gathered vector: [0, 0, 0, 0]
Rank 1: Gathered vector: [0, 0, 0, 0]
Rank 0: Gathered vector: [0, 1, 2, 3]
```

Only the root process with rank 0 gets the result.
To make it available to all other processes, we can broadcast it:

```cpp
// broadcast the vector to all other processes
simplemc::mpi::broadcast(vec, root, comm);
fmt::println("Rank {}: Broadcasted vector: {}", comm.rank(), vec);
```

Output:

```
Rank 1: Broadcasted vector: [0, 1, 2, 3]
Rank 2: Broadcasted vector: [0, 1, 2, 3]
Rank 0: Broadcasted vector: [0, 1, 2, 3]
Rank 3: Broadcasted vector: [0, 1, 2, 3]
```

Now all processes have the same content in the vector.

> **Note**: In practice, one would simply use simplemc::mpi::all_gather instead of
> simplemc::mpi::gather + simplemc::mpi::broadcast.

Above it was already shown, how to reduce single values across all processes.
Now, we want to do the same with multiple values by reducing the just broadcasted vector using the
`MPI_PROD` operation:

```cpp
// reduce the broadcasted vector on root
std::vector<int> reduced_vec(comm.size());
simplemc::mpi::reduce(vec, reduced_vec, MPI_PROD, root, comm);
fmt::println("Rank {}: Reduced vector: {}", comm.rank(), reduced_vec);
```

Output:

```
Rank 2: Reduced vector: [0, 0, 0, 0]
Rank 1: Reduced vector: [0, 0, 0, 0]
Rank 0: Reduced vector: [0, 1, 16, 81]
Rank 3: Reduced vector: [0, 0, 0, 0]
```

Taking the element-wise product of the four vectors gives the expected result \f$ (0, 1, 16, 81) \f$.
Here, we used simplemc::mpi::reduce and not simplemc::mpi::all_reduce, so the result is only available
on the root process.

If we want to distribute the results to the other processes, we have 2 possibilities:
- If the full vector is needed on every process, we can use simplemc::mpi::broadcast.
- If only part of the vector (in this case a single value) is needed on the other processes, we can
use simplemc::mpi::scatter.

Since we have already demonstrated the first option, let us show how scattering works:

```cpp
// scatter the reduced vector back to all processes
int scattered_value = 0;
simplemc::mpi::scatter(reduced_vec, scattered_value, root, comm);
fmt::println("Rank {}: Scattered value: {}", comm.rank(), scattered_value);
```

Output:

```
Rank 1: Scattered value: 1
Rank 0: Scattered value: 0
Rank 3: Scattered value: 81
Rank 2: Scattered value: 16
```

As expected, every rank receives exactly one value.

> **Note**: These are only some of the collective communication routines provided by **simplemc-mpi**.
> For more details, please refer to the @ref simplemc-mpi-coll documentation.

@section tut_mpi_3_code Full code

@include tutorials/tut_mpi_3.cpp
