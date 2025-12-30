@page tut_mpi_5 MPI Tutorial 5: Groups and communicators

[TOC]

In this tutorial, we show how to work with MPI groups and communicators using the **simplemc-mpi**
library.

Groups represent ordered sets of processes, while communicators provide communication contexts for
those processes.
Understanding how to create and manipulate groups and communicators is essential for organizing
parallel computations.

> **Note**: When running the code, the output might appear scrambled and has been rearranged to make
> it easier to read and follow along. The code is explicitly written for 4 MPI processes.

@section tut_mpi_5_details Step-by-step guide

The following code snippets are all part of the same `main` function:

```cpp
#include <fmt/ranges.h>
#include <simplemc/mpi.hpp>

#include <array>
#include <string>

// helper function definitions go here

int main(int argc, char** argv) {
    // initialize MPI environment and communicator
    simplemc::mpi::environment env { argc, argv };
    simplemc::mpi::communicator comm {};
    const int rank = comm.rank();

    // code snippets go here
}
```

We start by initializing the MPI environment and the default communicator which wraps
`MPI_COMM_WORLD`.

To simplify the output, we define a few helper functions that print the group or communicator
information and compare two groups or communicators:

```cpp
// Helper function to print the original rank, group size and group rank.
void print_group_info(int rank, const simplemc::mpi::group& grp) {
    auto rank_str = (grp.rank() == MPI_UNDEFINED ? "UNDEFINED" : std::to_string(grp.rank()));
    fmt::println("Rank {}: Group size = {}, Group rank = {}", rank, grp.size(), rank_str);
}

// Helper function to print the original rank, communicator size and communicator rank.
void print_comm_info(int rank, const simplemc::mpi::communicator& comm) {
    if (comm) {
        fmt::println("Rank {}: Communicator size = {}, Communicator rank = {}",
                     rank, comm.size(), comm.rank());
    } else {
        fmt::println("Rank {}: Communicator is MPI_COMM_NULL", rank);
    }
}

// Helper function to compare two groups or communicators.
auto print_comparison(const auto& obj1, const auto& obj2) {
    const int cmp = obj1.compare(obj2);
    switch (cmp) {
        case MPI_IDENT: return "IDENTICAL";
        case MPI_CONGRUENT: return "CONGRUENT";
        case MPI_SIMILAR: return "SIMILAR";
        case MPI_UNEQUAL: return "UNEQUAL";
        default: return "UNKNOWN";
    }
}
```

The simplemc::mpi::group and simplemc::mpi::communicator classes provide `size()` and `rank()`
methods.
When a process is not a member of the group, simplemc::mpi::group::rank returns `MPI_UNDEFINED`.
Similarly, when a communicator is `MPI_COMM_NULL`, it converts to `false` when checked.

The `compare()` methods compare two groups or communicators:
- `MPI_IDENT`: Identical (same members in same order, and for communicators, same context).
- `MPI_CONGRUENT`: Same group but different context (communicators only).
- `MPI_SIMILAR`: Same members but in different order.
- `MPI_UNEQUAL`: Different members.

@subsection tut_mpi_5_get_group Getting the group from a communicator

Every communicator has an associated group of processes. We can retrieve it using the
simplemc::mpi::communicator::get_group method:

```cpp
// get the group associated with MPI_COMM_WORLD
auto grp = comm.get_group();
print_group_info(rank, grp);
```

Output:

```
Rank 0: Group size = 4, Group rank = 0
Rank 1: Group size = 4, Group rank = 1
Rank 2: Group size = 4, Group rank = 2
Rank 3: Group size = 4, Group rank = 3
```

The group associated with a communicator contains the same processes and the ranks in the group match
the ranks in the communicator.

@subsection tut_mpi_5_subgroups Creating subgroups

We can create subgroups containing only specific ranks using the simplemc::mpi::group::include method.

Let's create a group with only the even-ranked processes:

```cpp
// include: create a subgroup with only ranks 0 and 2 (even ranks)
std::array<int, 2> even_ranks { 0, 2 };
auto even_grp = grp.include(even_ranks);
print_group_info(rank, even_grp);
```

Output:

```
Rank 0: Group size = 2, Group rank = 0
Rank 1: Group size = 2, Group rank = UNDEFINED
Rank 2: Group size = 2, Group rank = 1
Rank 3: Group size = 2, Group rank = UNDEFINED
```

Processes 0 and 2 are members of the new group (with new ranks 0 and 1), while processes 1 and 3
are not members (`MPI_UNDEFINED`).

Similarly, we can use simplemc::mpi::group::exclude to create a group by excluding specific ranks:

```cpp
// exclude: create a subgroup by excluding ranks 0 and 2 (odd ranks)
auto odd_grp = grp.exclude(even_ranks);
print_group_info(rank, odd_grp);
```

Output:

```
Rank 0: Group size = 2, Group rank = UNDEFINED
Rank 1: Group size = 2, Group rank = 0
Rank 2: Group size = 2, Group rank = UNDEFINED
Rank 3: Group size = 2, Group rank = 1
```

Here, we have excluded all even ranks from the original group, so we are left with only the odd ranks
1 and 3.

@subsection tut_mpi_5_set_operations Group set operations

Groups support set operations: union, intersection, and difference.

The simplemc::mpi::group::union_with method combines two groups by including all unique ranks from
both.
Let's combine the even and odd groups we created earlier:

```cpp
// union: combine even and odd groups (similar to original group: same ranks but different order)
auto union_grp = even_grp.union_with(odd_grp);
print_group_info(rank, union_grp);
if (rank == 0) {
    fmt::println("Union group vs original group: {}", print_comparison(union_grp, grp));
}
```

Output:

```
Rank 0: Group size = 4, Group rank = 0
Rank 1: Group size = 4, Group rank = 2
Rank 2: Group size = 4, Group rank = 1
Rank 3: Group size = 4, Group rank = 3
Union group vs original group: SIMILAR
```

Here, the union of even and odd groups contains all 4 processes. However, the result is only
`MPI_SIMILAR` to the original group (same members but different order) since the union places even
ranks first, then odd ranks.

The simplemc::mpi::group::difference_with method creates a group with processes that are in the first
group but not in the second:

```cpp
// difference: processes in original but not in even group (identical to odd group)
auto diff_grp = grp.difference_with(even_grp);
print_group_info(rank, diff_grp);
if (rank == 0) {
    fmt::println("Difference group vs odd group: {}", print_comparison(diff_grp, odd_grp));
}
```

Output:

```
Rank 0: Group size = 2, Group rank = UNDEFINED
Rank 1: Group size = 2, Group rank = 0
Rank 2: Group size = 2, Group rank = UNDEFINED
Rank 3: Group size = 2, Group rank = 1
Difference group vs odd group: IDENTICAL
```

Here, the difference between the original group and the even group gives us the odd ranks, which is
identical to `odd_grp`.

The simplemc::mpi::group::intersect_with method creates a group with processes that are in both
groups. For example, we can take the intersection of the even and odd groups:

```cpp
// intersection: common processes in even and odd group (indentical to MPI_GROUP_EMPTY)
auto intersect_grp = even_grp.intersect_with(odd_grp);
print_group_info(rank, intersect_grp);
if (rank == 0) {
    auto empty_grp = simplemc::mpi::group { MPI_GROUP_EMPTY };
    fmt::println("Intersection group vs MPI_GROUP_EMPTY: {}", print_comparison(intersect_grp, empty_grp));
}
```

Output:

```
Rank 0: Group size = 0, Group rank = UNDEFINED
Rank 1: Group size = 0, Group rank = UNDEFINED
Rank 2: Group size = 0, Group rank = UNDEFINED
Rank 3: Group size = 0, Group rank = UNDEFINED
Intersection group vs MPI_GROUP_EMPTY: IDENTICAL
```

Since no process is in both even and odd groups, the intersection is empty and identical to
`MPI_GROUP_EMPTY`.

@subsection tut_mpi_5_duplicate Duplicating communicators

Sometimes we need an independent copy of a communicator, for example to avoid interference between
different libraries or communication patterns.
The simplemc::mpi::communicator::duplicate method creates a new communicator with the same group but a
different communication context:

```cpp
// duplicate the communicator (congruent to original communicator)
auto dup_comm = comm.duplicate();
print_comm_info(rank, dup_comm);
if (rank == 0) {
    fmt::println("Duplicated communicator vs original communicator: {}",
                 print_comparison(dup_comm, comm));
}
```

Output:

```
Rank 0: Communicator size = 4, Communicator rank = 0
Rank 1: Communicator size = 4, Communicator rank = 1
Rank 2: Communicator size = 4, Communicator rank = 2
Rank 3: Communicator size = 4, Communicator rank = 3
Duplicated communicator vs original communicator: CONGRUENT
```

As expected, the duplicated communicator has the same size and ranks as the original.
The comparison returns `MPI_CONGRUENT`, meaning the communicators have the same group members but
different contexts (not `MPI_IDENT` which would mean they are the exact same communicator).

@subsection tut_mpi_5_split Splitting communicators

A common way to create new communicators is by splitting an existing one based on a color value.
Processes with the same color end up in the same new communicator.
For example, a common pattern is to split processes into even and odd ranks:

```cpp
// split communicator by even/odd rank
const int color = comm.rank() % 2;
auto split_comm = comm.split(color);
print_comm_info(rank, split_comm);
```

Output:

```
Rank 0: Communicator size = 2, Communicator rank = 0
Rank 1: Communicator size = 2, Communicator rank = 0
Rank 2: Communicator size = 2, Communicator rank = 1
Rank 3: Communicator size = 2, Communicator rank = 1
```

Each process now belongs to a smaller communicator of size 2, with new ranks starting from 0.
Even-ranked processes (0 and 2) are in one communicator, odd-ranked processes (1 and 3) in another.

@subsection tut_mpi_5_create Creating communicators from groups

Every communicator is based on a group of processes.
We can therefore also create a communicator directly from a group using the
simplemc::mpi::communicator::create method.
This is useful when we have already constructed the exact group we want.
Processes not in the group receive `MPI_COMM_NULL`:

```cpp
// create a communicator from the even group (congruent to split communicator for even ranks)
auto even_comm = comm.create(even_grp);
print_comm_info(rank, even_comm);
if (rank == 0) {
    fmt::println("Even group communicator vs split communicator: {}",
                 print_comparison(even_comm, split_comm));
}
```

Output:

```
Rank 0: Communicator size = 2, Communicator rank = 0
Rank 1: Communicator is MPI_COMM_NULL
Rank 2: Communicator size = 2, Communicator rank = 1
Rank 3: Communicator is MPI_COMM_NULL
Even group communicator vs split communicator: CONGRUENT
```

For even-ranked processes (0 and 2), the communicator created from the even group is congruent to
the split communicator (same group, different context).

@section tut_mpi_5_code Full code

@include tutorials/tut_mpi_5.cpp
