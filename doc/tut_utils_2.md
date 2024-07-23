@page tut_utils_2 Utilities Tutorial 2: Multi-dimensional indexing

[TOC]

In this tutorial, we show how to convert multi-dimensional indices to flat indices using the
simplemc::flat_index function from the @ref simplemc-utils library.

We further make use of the simplemc::row_major and simplemc::column_major tags to apply a certain
memory layout.

The following code snippets are all part of the same `main` function:

```cpp
#include <simplemc/utils.hpp>

#include <fmt/ranges.h>

#include <array>
#include <numeric>
#include <vector>

int main() {
    // code snippets go here
}
```

Under the hood, a multi-dimensional array is just some contiguous block of memory that is interpreted
in a certain way.
If we interpret the same memory block in a different way, we can end up with a very different array.

Let us demonstrate this by first creating an integer vector and fill it with some numbers:

```cpp
// create an integer vector and fill it with 1, ..., 15
std::vector<int> data(15);
std::iota(data.begin(), data.end(), 1);
```

`std::vector` objects are guarenteed to be contiguous, so we can use it as our memory block.

With **fmt** it is easy to print the vector to stdout:

```cpp
// print the vector to stdout
fmt::print("{}\n", data);
```

Output:

```
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
```

Now, let us try to interpret this 1-dimensional vector of size \f$ 15 \f$ as matrices with different
shapes and memory layouts.

In order to to this, we use the following generic lambda to simply print 2-dimensional arrays:

```cpp
// generic lambda to print a matrix with a given shape and memory layout
auto print_matrix = [](const auto& data, const auto& shape, auto mem_layout) {
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            std::array<int, 2> idxs = { i, j };
            fmt::print("{:3} ", data[simplemc::flat_index(idxs, shape, mem_layout)]);
        }
        fmt::print("\n");
    }
    fmt::print("\n");
};
```

The lambda takes 3 parameters:
- `data`: The container which handles the memory block.
- `shape`: The shape of the 2-dimensional array we want to print.
- `mem_order`: The memory layout of the 2-dimensional array. This has to be either an instance of
simplemc::row_major or simplemc::column_major.

To print the array, we loop over all possible multi-dimensional indices of the given shape and access
each data element using the simplemc::flat_index function which turns a multi-dimensional index into
its corresponding flat-index for a given shape and memory layout.

Since our memory block has \f$ 15 \f$ elements, there are four possible shapes for our matrix.

The shapes `(1,15)` and `(15,1)` are rather boring, so we will skip those.

Let's start with the shape \f$ 5 \times 3 \f$:

```cpp
// shape of the 5x3 matrix
std::array<int, 2> shape1 {5, 3};
```

As mentioned above, for a given shape we have two possible memory layouts:
- simplemc::row_major also called C-order varies the last index the fastest, i.e. the first elements
in memory correspond to the first row of the matrix, then the second row and so on.
- simplemc::column_major also called Fortran-order varies the first index the fastest, i.e. the first
elements in memory correspond to the first column of the matrix, then the second column and so on.

The meaning of row-major and column-major should become clear, when we actually print the matrices:

```cpp
// print the data as a 5x3 matrix in row-major order
fmt::print("5x3 row-major order:\n");
print_matrix(data, shape1, simplemc::row_major {});

// print the data as a 5x3 matrix in column-major order
fmt::print("5x3 column-major order:\n");
print_matrix(data, shape1, simplemc::column_major {});
```

Output:

```
5x3 row-major order:
  1   2   3
  4   5   6
  7   8   9
 10  11  12
 13  14  15

5x3 column-major order:
  1   6  11
  2   7  12
  3   8  13
  4   9  14
  5  10  15
```

As you can see, the number increase horizontally in row-major ordering and vertically in column-major
ordering.

We could have also interpreted the same memory as a \f$ 3 \times 5 \f$ matrix:

```cpp
// shape of the 3x5 matrix
std::array<int, 2> shape2 {3, 5};

// print the data as a 3x5 matrix in row-major order
fmt::print("5x3 row-major order:\n");
print_matrix(data, shape2, simplemc::row_major {});

// print the data as a 3x5 matrix in column-major order
fmt::print("5x3 column-major order:\n");
print_matrix(data, shape2, simplemc::column_major {});
```

Output:

```
5x3 row-major order:
  1   2   3   4   5
  6   7   8   9  10
 11  12  13  14  15

5x3 column-major order:
  1   4   7  10  13
  2   5   8  11  14
  3   6   9  12  15
```
