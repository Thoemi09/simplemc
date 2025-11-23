@page tut_grids_2 Grids Tutorial 2: Introducing N-dimensional grids

[TOC]

In this tutorial, we take a closer look at @ref simplemc-grids-nd.

@section tut_grids_2_details Step-by-step guide

The following code snippets are all part of the same `main` function:

```cpp
#include <fmt/ranges.h>
#include <simplemc/grids.hpp>
#include <simplemc/random.hpp>
#include <simplemc/utils/ranges.hpp>

#include <array>

// Print a view in matrix form with the given shape.
void print_2d_view(auto&& view, long rows, long cols) {
    using namespace simplemc::ranges::views;
    for (long i = 0; i < rows; ++i) {
        fmt::println("{:n}", view | drop(i * cols) | take(cols));
    }
    fmt::println("");
}

int main() {
    // code snippets go here
}
```

Here, we have already implemented a helper function that prints a range or a view as a 2-dimensional
array with the given number of rows and columns.

In the following, we only consider a single 2-dimensional \f$ 3 \times 3 \f$ grid, so that when we
print grid points and bins, the output is still clear and easy to follow.
Higher dimensional grids or grids with more points behave exactly the same.
It just becomes more difficult to visualize them.

Let's start with defining two 1-dimensional simplemc::custom_grid objects which will serve as our basis
for the 2-dimensional grid:

```cpp
// 1D custom grid of size 3 with points 2, 4 and 8
simplemc::custom_grid grid1_1d { { 2, 4, 8 } };

// 1D custom grid of size 3 with points 1, 3 and 9
simplemc::custom_grid grid2_1d { { 1, 3, 9 } };
```

As you can see, grid #1 has size \f$ M_1 = 3 \f$ and contains the grid points \f$ \mathbf{x}_1 = (2,
4, 8) \f$, whereas grid #2 has size \f$ M_2 = 3 \f$ and contains the grid points \f$ \mathbf{x}_2 =
(1, 3, 9) \f$.

We can now combine them to construct our 2-dimensional grid using the generic simplemc::nd_grid class:

```cpp
// construct the 2D grid
simplemc::nd_grid grid_2d { grid1_1d, grid2_1d };
```

And that's it.
Grid #1 lies along the first dimension, since we passed it as the first argument to the constructor,
and grid #2 lies along the second dimension.

First, we print some basic information about our grid:

```cpp
// print basic information
const auto shape = grid_2d.shape();
fmt::println("{}D grid of size {} and shape {}\n", grid_2d.dim(), grid_2d.size(), shape);
```

Output:

```
2D grid of size 9 and shape [3, 3]
```

As expected, the grid is 2-dimensional and of size \f$ M = M_1 \cdot M_2 = 9 \f$ with shape \f$ M_1
\times M_2 = 3 \times 3 \f$.

The grid points of N-dimensional grids are given by the cartesian product of the grid points of the
underlying 1-dimensional grids.
To better see this, let's print the grid points of our 2D grid in matrix form:

```cpp
// print grid points
fmt::println("Grid points:");
print_2d_view(grid_2d, shape[0], shape[1]);
```

Output:

```
Grid points:
[2, 1], [2, 3], [2, 9]
[4, 1], [4, 3], [4, 9]
[8, 1], [8, 3], [8, 9]
```

The rows correspond to the first dimension or to grid #1 and the columns correspond to the second
dimension or to grid #2.

N-dimensional grids are always traversed in C order (see simplemc::row_major), i.e. the last dimension
is iterated over first, then the next to last dimension, and so on.
For our 2D grid, this means that grid #2 is iterated over first.
The linear order of the grid points is therefore given by
\f{eqnarray*}{
  g(0,0) &=& [2, 1] \to g(0,1) = [2, 3] \to g(0,2) = [2, 9] \to g(1,0) = [4, 1] \to g(1,1) = [4, 3]
  \to \\
  g(1,2) &=& [4, 9] \to g(2,0) = [8, 1] \to g(2,1) = [8, 3] \to g(2,2) = [8, 9] \; .
\f}

The views on bin centers and bin volumes are traversed in the same way, except that there are only \f$
(M_1 - 1) \cdot (M_2 - 1) = 4 \f$ bins:

```cpp
// print bin centers
fmt::println("Bin centers: {}\n", simplemc::bin_center_view(grid_2d));

// print bin volumes
fmt::println("Bin volumes: {}\n", simplemc::bin_volume_view(grid_2d));
```

Output:

```
Bin centers: [[3, 2], [3, 6], [6, 2], [6, 6]]

Bin volumes: [4, 12, 8, 24]
```

Here,
- the center of bin \f$ b_{0,0} \f$ is at \f$ (3, 2) \f$ and has volume \f$ 4 \f$,
- the center of bin \f$ b_{0,1} \f$ is at \f$ (3, 6) \f$ and has volume \f$ 12 \f$,
- the center of bin \f$ b_{1,0} \f$ is at \f$ (6, 2) \f$ and has volume \f$ 8 \f$ and
- the center of bin \f$ b_{1,1} \f$ is at \f$ (6, 6) \f$ and has volume \f$ 24 \f$.

This can easily be seen by drawing the grid points on 2 perpendicular axis and by calculating the
areas of the corresponding bins.

At last, we want to check how to find the bin \f$ b_{i_1, i_2} \f$ such that a given point \f$
\mathbf{x} \f$ from the domain \f$ \mathrm{R} \f$ of the grid falls into the bin, i.e. \f$ \mathbf{x}
\in b_{i_1, i_2} \f$:

```cpp
// find the bin b_i to which a value x belongs
simplemc::xoshiro256pp rng {};
simplemc::uniform_real_distribution dist1 { grid_2d.first()[0], grid_2d.last()[0] };
simplemc::uniform_real_distribution dist2 { grid_2d.first()[1], grid_2d.last()[1] };
fmt::println("Find the bin:");
for (int i = 0; i < 10; ++i) {
    const auto x = std::array<double, 2> { dist1(rng), dist2(rng) };
    const auto idx = grid_2d.index(x);
    const auto gidx = grid_2d.at(idx);
    const auto gidxp1 = grid_2d.at(idx[0] + 1, idx[1] + 1);
    fmt::println(
        "x = {::.4f} lies in bin [{},{}) x [{},{}) with index {}", x, gidx[0], gidxp1[0], gidx[1], gidxp1[1], idx);
}
```

Output:

```
Find the bin:
x = [2.6746, 7.5241] lies in bin [2,4) x [3,9) with index [0, 1]
x = [4.3605, 3.2202] lies in bin [4,8) x [3,9) with index [1, 1]
x = [3.6794, 1.3069] lies in bin [2,4) x [1,3) with index [0, 0]
x = [7.7030, 4.3149] lies in bin [4,8) x [3,9) with index [1, 1]
x = [3.9752, 4.2047] lies in bin [2,4) x [3,9) with index [0, 1]
x = [2.6444, 2.5291] lies in bin [2,4) x [1,3) with index [0, 0]
x = [7.2657, 3.8465] lies in bin [4,8) x [3,9) with index [1, 1]
x = [7.6045, 3.5419] lies in bin [4,8) x [3,9) with index [1, 1]
x = [4.1670, 6.4172] lies in bin [4,8) x [3,9) with index [1, 1]
x = [5.5762, 6.5394] lies in bin [4,8) x [3,9) with index [1, 1]
```

The code generates 10 random points in \f$ \mathrm{R} \f$ and finds the corresponding bin with the
simplemc::nd_grid::index function.

Note that the simplemc::nd_grid::index function returns the 2-dimensional index \f$ \mathbf{i} = (i_1,
i_2) \f$ of the grid point at the lower left corner of the bin which contains the given point \f$
\mathbf{x} \f$.
The wanted bin therefore spans the area \f$ [g_1(i_1), g_1(i_1 + 1)) \times [g_2(i_2), g_2(i_2 + 1))
\f$.

@section tut_grids_2_code Full code

@include tutorials/tut_grids_2.cpp
