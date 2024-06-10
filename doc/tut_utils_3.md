@page tut_utils_3 Utilities Tutorial 3: Printing data with fmt

[TOC]

In this tutorial, we show how to use the [fmt](https://github.com/fmtlib/fmt) library to output
formatted data.

For further examples, please look at the official [fmt documentation](https://fmt.dev/latest/index.html).

```cpp
#include <simplemc/utils.hpp>

#include <fmt/core.h>

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

// Create n data point on the unit circle in the complex plane.
auto unit_circle_data(int n) {
    std::vector<double> angles(10);
    std::vector<std::complex<double>> data(10);
    for (int i = 0; i < n; ++i) {
        angles[i] = i * 2.0 * std::numbers::pi / n;
        data[i] = std::exp(std::complex<double> { 0.0, angles[i] });
    }
    return std::make_tuple(angles, data);
}

int main() {
    // print "Hello, World!"
    fmt::print("Hello, World!\n\n");

    // create a point object and print it
    point p { 1.0, 2.0 };
    fmt::print("Point: {}\n\n", simplemc::to_string(p));

    // print some complex data with a certain precision
    int n = 10;
    auto [angles, data] = unit_circle_data(n);
    fmt::print("{:<15}{:<20}{:<20}{}\n", "Phi", "Re[z]", "Im[z]", "z");
    for (int i = 0; i < n; ++i) {
        fmt::print("{:<15.4f}{:<20.10f}{:<20.10f}{:.4f}\n", angles[i], data[i].real(), data[i].imag(), data[i]);
    }
}
```

Output:

```
Hello, World!

Point: (1, 2)

Phi            Re[z]               Im[z]               z
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
