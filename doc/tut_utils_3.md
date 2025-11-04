@page tut_utils_3 Utilities Tutorial 3: Printing data with fmt

[TOC]

In this tutorial, we show how to use the [fmt](https://github.com/fmtlib/fmt) library to output
formatted data.

For further examples, please look at the official [fmt documentation](https://fmt.dev/latest/index.html).

@section tut_utils_3_details Step-by-step guide

The following code snippets are all part of the same `main` function:

```cpp
#include <fmt/base.h>
#include <simplemc/utils.hpp>
#include <simplemc/utils/fmt_complex.hpp>

#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>
#include <tuple>
#include <vector>

// Simple custom point class.
struct point {
    double x;
    double y;
    friend std::ostream& operator<<(std::ostream& os, const point& p) { return os << "(" << p.x << ", " << p.y << ")"; }
};

int main() {
    // code snippets go here
}
```

Here, we simply define a structure called `point` that should represent a point in 2-dimensional
space.
It has implemented the usual `operator<<` to write its members to some `std::ostream` object.
For example,

```cpp
// output a point object
point p { 1.0, 2.0 };
std::cout << "Point: " << p << std::endl;
```

Output:

```
Point: (1, 2)
```

Let us now turn to **fmt**.
To print a string to `stdout` is as easy as

```cpp
// print "Hello, World!"
fmt::print("Hello, World!\n");
```

Output:

```
Hello, World!
```

If we want to print the same point as before but this time with **fmt**, we can use
simplemc::to_string which returns the string representation of an object given that it has an
overloaded streaming operator.

Since we have seen how to print a string with **fmt**, we can print any object that provides an
overloaded `operator<<` (as long as it works as expected and writes a string representation to the
`std::ostream` object):

```cpp
// print a point object using fmt
fmt::print("Point: {}\n\n", simplemc::to_string(p));
```

Output:

```
Point: (1, 2)
```

Another possibility to use **fmt** with custom types is to implement a specialized formatter.
This is much more flexible than the above method but also more complicated to do.
We refer the interested reader to @ref "fmt::formatter< std::complex< T >, Char >" for an example.

So far **fmt** doesn't seem to have a lot of advantages over the traditional `std::cout << ...`.
This changes when we try to output formatted data, e.g. in a table.

As a simple example, let us try to output some complex numbers on the unit circle together with their
corresponding arguments.

We first define a lambda that generates the data:

```cpp
// lambda to generate n data points on the unit circle in the complex plane
auto unit_circle_data = [](int n) {
    std::vector<double> angles(n);
    std::vector<std::complex<double>> data(n);
    for (int i = 0; i < n; ++i) {
        angles[i] = i * 2.0 * std::numbers::pi / n;
        data[i] = std::exp(std::complex<double> { 0.0, angles[i] });
    }
    return std::make_tuple(angles, data);
};
```

It produces `n` equally spaced data points on the unit circle and returns a tuple with
- a vector containing the angles or arguments of the complex numbers and
- a vector containing the complex numbers.

Let us generate 10 such data points:

```cpp
// generate 10 complex numbers
int n = 10;
auto [angles, data] = unit_circle_data(n);
```

To print the data in a nicely formatted table, we start with the header:

```cpp
// print the header of the table
fmt::print("{:<15}{:<20}{:<20}{}\n", "Phi(z)", "Re[z]", "Im[z]", "z");
```

Then we loop over all the complex numbers and print their arguments, real part, imaginary part and the
number itself:

```cpp
// print the data of the table
fmt::print("{:<15}{:<20}{:<20}{}\n", "Phi(z)", "Re[z]", "Im[z]", "z");
for (int i = 0; i < n; ++i) {
    fmt::print("{:<15.4f}{:<20.10f}{:<20.10f}{:.4f}\n", angles[i], data[i].real(), data[i].imag(), data[i]);
}
```

Output:

```
Phi(z)         Re[z]               Im[z]               z
0.0000         1.0000000000        0.0000000000        (1.0000,0.0000)
0.6283         0.8090169944        0.5877852523        (0.8090,0.5878)
1.2566         0.3090169944        0.9510565163        (0.3090,0.9511)
1.8850         -0.3090169944       0.9510565163        (-0.3090,0.9511)
2.5133         -0.8090169944       0.5877852523        (-0.8090,0.5878)
3.1416         -1.0000000000       0.0000000000        (-1.0000,0.0000)
3.7699         -0.8090169944       -0.5877852523       (-0.8090,-0.5878)
4.3982         -0.3090169944       -0.9510565163       (-0.3090,-0.9511)
5.0265         0.3090169944        -0.9510565163       (0.3090,-0.9511)
5.6549         0.8090169944        -0.5877852523       (0.8090,-0.5878)
```

If you are not familiar with the format string syntax, e.g. with the meaning of `{:<20.10f}`, please
look at the official [fmt documentation](https://fmt.dev/latest/index.html).

@section tut_utils_3_code Full code

@include tutorials/tut_utils_3.cpp
