@page tut_grids_1 Grids Tutorial 1: Introducing various 1-dimensional grids

[TOC]

In this tutorial, we introduce various @ref simplemc-grids-1d.

@section tut_grids_1_details Step-by-step guide

The following code snippets are all part of the same `main` function:

```cpp
#include <fmt/ranges.h>
#include <simplemc/grids.hpp>
#include <simplemc/random.hpp>
#include <simplemc/utils/ranges.hpp>

#include <string_view>
#include <vector>

// function definiton goes here

int main() {
    // code snippets go here
}
```

We start by writing a generic function that will test the different grids for us:

```cpp
// Test a given grid.
void test_grid(const auto& grid, std::string_view name) {
    // print basic information
    fmt::println("{} grid of size {} on the interval [{},{}]\n", name, grid.size(), grid.first(), grid.last());

    // print the grid points, bin centers and bin volumes (sizes)
    fmt::println("Grid points: {::.4g}", grid);
    fmt::println("Bin centers: {::.4g}", simplemc::bin_center_view(grid));
    fmt::println("Bin volumes: {::.4g}\n", simplemc::bin_volume_view(grid));

    // find the bin b_i to which a value x belongs
    std::mt19937 rng {};
    std::uniform_real_distribution dist { grid.first(), grid.last() };
    fmt::println("Find the bin:");
    for (int i = 0; i < 10; ++i) {
        const auto x = dist(rng);
        const auto idx = grid.index(x);
        fmt::println("x = {:.4f} lies in bin [{:.4g},{:.4g}) with index {}", x, grid.at(idx), grid.at(idx + 1), idx);
    }
    fmt::println("");
}
```

The function is fairly straightfoward.
It takes the grid that we want to test as its first argument and a given name for the grid as its
second argument.
As seen from its implementation, we have split the function into 3 blocks:
- The first code block prints the name of the grid together with its size \f$ M \f$ and its range
\f$ [a, b] \f$.
- The second code block prints all grid points \f$ g(i) \f$ and all centers and sizes of the bins
\f$ b_i = [g(i), g(i+1)) \f$.
- The last block generates random values in the range \f$ [a, b] \f$ and finds the bin \f$ b_i \f$
that contains that value.

In the following, all grids have size \f$ M = 11 \f$ and are defined on the interval \f$ [0, 10] \f$.

@subsection tut_grids_1_linear Linear grid

We test the simplemc::linear_grid by calling

```cpp
// linear grid
test_grid(simplemc::linear_grid { 0, 10, 11 }, "linear");
```

The first line gives us some basic information on the grid (what kind of grid it is, its size and its
range):

```
linear grid of size 11 on the interval [0,10]
```

Then we get more detailed information in form of the grid points, the bin centers and the bin sizes.

```
Grid points: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Bin centers: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
Bin volumes: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

Here we can see that the 11 grid points are evenly distributed on the range \f$ [0, 10] \f$ as we
expect from a simplemc::linear_grid.
This implies that all 10 bins are of the same size (\f$ = 1 \f$) and that the bin centers are the
same distant apart.

The last output block shows how for a given value \f$ x \in [0,10] \f$, we can find the bin \f$ b_i
\f$ such that \f$ x \in b_i \f$:

```
Find the bin:
x = 1.1244 lies in bin [1,2) with index 1
x = 8.1552 lies in bin [8,9) with index 8
x = 3.9342 lies in bin [3,4) with index 3
x = 2.7753 lies in bin [2,3) with index 2
x = 2.7989 lies in bin [2,3) with index 2
x = 0.3836 lies in bin [0,1) with index 0
x = 9.5050 lies in bin [9,10) with index 9
x = 4.1437 lies in bin [4,5) with index 4
x = 3.2921 lies in bin [3,4) with index 3
x = 4.0059 lies in bin [4,5) with index 4
```

Since we generate the numbers uniformly in the range of the grid and the bins are all of the same
size, we expect each bin to show up the same number of times.
This is of course only true, if we would generate a lot of \f$ x \f$ values.

@subsection tut_grids_1_quadratic Quadratic grid

We test the quadratic grid by constructing a simplemc::power_grid with a power parameter \f$ p = 2
\f$:

```cpp
// quadratic grid
test_grid(simplemc::power_grid { 0, 10, 11, 2 }, "quadratic");
```

Output:

```
quadratic grid of size 11 on the interval [0,10]

Grid points: [0, 0.1, 0.4, 0.9, 1.6, 2.5, 3.6, 4.9, 6.4, 8.1, 10]
Bin centers: [0.05, 0.25, 0.65, 1.25, 2.05, 3.05, 4.25, 5.65, 7.25, 9.05]
Bin volumes: [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]

Find the bin:
x = 1.1244 lies in bin [0.9,1.6) with index 3
x = 8.1552 lies in bin [8.1,10) with index 9
x = 3.9342 lies in bin [3.6,4.9) with index 6
x = 2.7753 lies in bin [2.5,3.6) with index 5
x = 2.7989 lies in bin [2.5,3.6) with index 5
x = 0.3836 lies in bin [0.1,0.4) with index 1
x = 9.5050 lies in bin [8.1,10) with index 9
x = 4.1437 lies in bin [3.6,4.9) with index 6
x = 3.2921 lies in bin [2.5,3.6) with index 5
x = 4.0059 lies in bin [3.6,4.9) with index 6
```

The output shows us that the grid points of the quadratic grid are not uniformly spaced.
The bin sizes are smaller for lower indices, i.e. when the grid points are closer to the lower limit
\f$ a = 0 \f$ of the interval, and larger for higher indices, i.e. when the grid points are near the
upper limit \f$ b = 10 \f$ of the interval.

We can also see that the bin indices in the last output block tend to be larger than average.
This is expected since the bins get bigger and therefore more generated \f$ x \f$ values will fall
into the bins with larger indices.

@subsection tut_grids_1_cubic Cubic grid

We test the cubic grid by constructing a simplemc::power_grid with a power parameter \f$ p = 3 \f$:

```cpp
// cubic grid
test_grid(simplemc::power_grid { 0, 10, 11, 3 }, "cubic");
```

Output:

```
cubic grid of size 11 on the interval [0,10]

Grid points: [0, 0.01, 0.08, 0.27, 0.64, 1.25, 2.16, 3.43, 5.12, 7.29, 10]
Bin centers: [0.005, 0.045, 0.175, 0.455, 0.945, 1.705, 2.795, 4.275, 6.205, 8.645]
Bin volumes: [0.01, 0.07, 0.19, 0.37, 0.61, 0.91, 1.27, 1.69, 2.17, 2.71]

Find the bin:
x = 1.1244 lies in bin [0.64,1.25) with index 4
x = 8.1552 lies in bin [7.29,10) with index 9
x = 3.9342 lies in bin [3.43,5.12) with index 7
x = 2.7753 lies in bin [2.16,3.43) with index 6
x = 2.7989 lies in bin [2.16,3.43) with index 6
x = 0.3836 lies in bin [0.27,0.64) with index 3
x = 9.5050 lies in bin [7.29,10) with index 9
x = 4.1437 lies in bin [3.43,5.12) with index 7
x = 3.2921 lies in bin [2.16,3.43) with index 6
x = 4.0059 lies in bin [3.43,5.12) with index 7
```

The cubic grid is very similar to the quadratic grid, except that the difference in bin sizes between
the beginning of the range close to \f$ a = 0 \f$ and the end of the range close to \f$ b = 10 \f$ is
even more pronounced.

@subsection tut_grids_1_sym Symmetric quadratic grid

We test the symmetric quadratic grid by constructing a simplemc::symmetric_power_grid with power
parameter \f$ p = 2 \f$:

```cpp
// symmetric quadratic grid
test_grid(simplemc::symmetric_power_grid { 0, 10, 11, 2 }, "symmetric quadratic");
```

Output:

```
symmetric quadratic grid of size 11 on the interval [0,10]

Grid points: [0, 0.2, 0.8, 1.8, 3.2, 5, 6.8, 8.2, 9.2, 9.8, 10]
Bin centers: [0.1, 0.5, 1.3, 2.5, 4.1, 5.9, 7.5, 8.7, 9.5, 9.9]
Bin volumes: [0.2, 0.6, 1, 1.4, 1.8, 1.8, 1.4, 1, 0.6, 0.2]

Find the bin:
x = 1.1244 lies in bin [0.8,1.8) with index 2
x = 8.1552 lies in bin [6.8,8.2) with index 6
x = 3.9342 lies in bin [3.2,5) with index 4
x = 2.7753 lies in bin [1.8,3.2) with index 3
x = 2.7989 lies in bin [1.8,3.2) with index 3
x = 0.3836 lies in bin [0.2,0.8) with index 1
x = 9.5050 lies in bin [9.2,9.8) with index 8
x = 4.1437 lies in bin [3.2,5) with index 4
x = 3.2921 lies in bin [3.2,5) with index 4
x = 4.0059 lies in bin [3.2,5) with index 4
```

The symmetric quadratic grid consists of two quadratic grids,
- one increasing grid with range \f$ [0, 5] \f$ and
- one decreasing grid with range \f$ [5, 10] \f$.

That means on the interval \f$ [0, 5] \f$, the bin sizes increase, while on the interval \f$ [5, 10]
\f$, the bin sizes decrease.

The symmetry is with respect to the center of the grid, i.e. \f$ c = (a + b) / 2 = 5 \f$.

@note Symmetric grids make only sense for odd sizes.

@subsection tut_grids_1_custom Custom grid

We test the simplemc::custom_grid by first creating a `std::vector` of arbitrary but increasing grid
points and then passing the grid points to the custom grid constructor:

```cpp
// custom grid
auto grid_pts = std::vector<double> { 0, 0.2, 0.8, 3.2, 4.9, 4.99, 6.1, 8.3, 8.7, 9, 10 };
test_grid(simplemc::custom_grid { grid_pts }, "custom");
```

Output:

```
custom grid of size 11 on the interval [0,10]

Grid points: [0, 0.2, 0.8, 3.2, 4.9, 4.99, 6.1, 8.3, 8.7, 9, 10]
Bin centers: [0.1, 0.5, 2, 4.05, 4.945, 5.545, 7.2, 8.5, 8.85, 9.5]
Bin volumes: [0.2, 0.6, 2.4, 1.7, 0.09, 1.11, 2.2, 0.4, 0.3, 1]

Find the bin:
x = 1.1244 lies in bin [0.8,3.2) with index 2
x = 8.1552 lies in bin [6.1,8.3) with index 6
x = 3.9342 lies in bin [3.2,4.9) with index 3
x = 2.7753 lies in bin [0.8,3.2) with index 2
x = 2.7989 lies in bin [0.8,3.2) with index 2
x = 0.3836 lies in bin [0.2,0.8) with index 1
x = 9.5050 lies in bin [9,10) with index 9
x = 4.1437 lies in bin [3.2,4.9) with index 3
x = 3.2921 lies in bin [3.2,4.9) with index 3
x = 4.0059 lies in bin [3.2,4.9) with index 3
```

Custom grids are the most flexible but also the most expensive in terms of memory and performance.
They can be constructed from any ordered array of grid points.

Their drop in performance might become noticeable for large grids when searching for the bin \f$ b_i
\f$ which contains a given value \f$ x \in [a, b] \f$.
Since the arrangement of the grid points do not have to follow any specific rule, the best we can do
is a binary search which has complexity \f$ \mathcal{O}(log M) \f$.

@note The given array to construct a simplemc::custom_grid has to be ordered w.r.t. to \f$ < \f$
(increasing grid) or \f$ > \f$ (decreasing grid). Otherwise, the behavior is undefined.

@section tut_grids_1_code Full code

@include tutorials/tut_grids_1.cpp
