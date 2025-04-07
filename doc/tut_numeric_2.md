@page tut_numeric_2 Numeric Tutorial 2: Generalized Fourer series

[TOC]

In this tutorial, we show how to approximate a function using a generalized Fourier series.
We will make use of Legendre polynomials (simplemc::legendre_polynomial) and numerical quadrature
routines (simplemc::simpson_quadrature).

@section tut_numeric_2_details Step-by-step guide

The following code snippets are all part of the same `main` function:

```cpp
#include <fmt/base.h>
#include <simplemc/grids.hpp>
#include <simplemc/numeric.hpp>

#include <array>
#include <cmath>
#include <vector>

int main() {
    // code snippets go here
}
```

@subsection tut_numeric_2_details_intro Introduction

A generalized Fourier series expands a function \f$ f(x) \f$ in terms of an orthogonal set of basis
functions \f$ \{ p_l(x) \} \f$:
\f[
  f(x) = \sum_{l=0}^\infty f_l p_l(x) \; .
\f]
Here, \f$ f_l \f$ are the Fourier coefficients.

**simplemc** provides some common basis functions in @ref simplemc-numeric-poly "Orthogonal polynomials":
- simplemc::legendre_polynomial: Legendre polynomials \f$ P_l(x) \f$,
- simplemc::chebyshev_polynomial_first: Chebyshev polynomials of the first kind \f$ T_l(x) \f$,
- simplemc::chebyshev_polynomial_second: Chebyshev polynomials of the second kind \f$ U_l(x) \f$,
- simplemc::hermite_polynomial: Hermite polynomials \f$ H_l(x) \f$ and
- simplemc::laguerre_polynomial: Laguerre polynomials \f$ L_l(x) \f$.

In the following, we will use Legendre polynomials, \f$ P_l(x) \f$, to approximate the function \f$
f(x) = e^x \f$ on the interval \f$ [c, d] = [2, 5] \f$.

@subsection tut_numeric_2_details_coeffs Fourier coefficients

In order to do that, we need to calculate the Fourier coefficiencts \f$ f_l \f$.
Using the orthogonality property of our basis functions
\f[
  \langle p_l, p_k \rangle = \int_a^b p_l(x) p_k(x) W(x) dx = N_l \delta_{lk} \; ,
\f]
we can multiply the generalized Fourier series by \f$ p_k(x) W(x) \f$ and integrate over \f$ [a, b]
\f$ to get the coefficient \f$ f_k \f$:
\f[
  \int_a^b p_k(x) W(x) f(x) dx = \sum_{l=0}^\infty f_l \int_a^b p_k(x) W(x) p_l(x) dx =
  \sum_{l=0}^\infty f_l N_l \delta_{lk} = f_k N_k \; .
\f]

Here we have assumed that the domain of the function \f$ f \f$ coincides with the domain of \f$ p_l
\f$, i.e. that \f$ [a, b] = [c, d] \f$.
In general, this is not true.
We therefore have to modify the orthogonality property by introducing the linear map \f$ y(x) = \alpha
x + \beta \f$ that maps an \f$ x \in [a, b] \f$ to \f$ y(x) \in [c, d] \f$:
\f[
  \langle p_l, p_k \rangle = \int_a^b p_l(x) p_k(x) W(x) dx = \int_{y(a)=c}^{y(b)=d} \frac{1}{\alpha}
  p_l(x(y)) p_k(x(y)) W(x(y)) dy = N_l \delta_{lk} \; ,
\f]
where \f$ x(y) \f$ denotes the inverse map \f$ y^{-1}(x) \f$.

For Legendre polynomials, the orthogonality property looks like
\f[
  \langle P_l, P_k \rangle = \int_{-1}^1 P_l(x) P_k(x) dx = \frac{2}{2l + 1} \delta_{lk} \; ,
\f]
i.e. they are defined on the interval \f$ [a, b] = [-1, 1] \f$, the weight function is \f$ W(x) = 1
\f$ and their norm is \f$ N_l = 2 / (2l + 1) \f$.

Applying the general considerations from above to our function \f$ f(x) = e^x \f$ and to Legendre
polynomials, we can finally write the Fourier coefficients as
\f[
  f_k = \frac{1}{\alpha N_l} \int_c^d P_k(x(y)) f(y) dy = \frac{2l + 1}{2 \alpha} \int_2^5 P_k \left(
  \frac{y - \beta}{\alpha} \right) e^y dy = \frac{2l + 1}{3} \int_2^5 P_k \left( \frac{2y - 7}{e}
  \right) e^y dy \; ,
\f]
where the linear map \f$ y : [a, b] = [-1, 1] \to [c, d] = [2, 5] \f$ is given by \f$ y(x) = \alpha x
+ \beta = (3 / 2) (x + 2) \f$.

@subsection tut_numeric_2_details_impl Implementation

Now let's do the actual calculation using some routines from **simplemc-numeric**.

We start by defining the function \f$ f(x) \f$ and its domain:

```cpp
// define the function f(y) = e^y
auto f = [](double y) { return std::exp(y); };

// define the domain [c, d] = [2, 5]
constexpr auto domain_f = std::array<double, 2> { 2.0, 5.0 };
```

Then we construct the linear map \f$ y : [-1, 1] \to [2, 5] \f$:

```cpp
// linear map y : [-1, 1] -> [2, 5]
auto y_x = simplemc::linear_map(simplemc::legendre_polynomial::domain(), domain_f);
```

Furthermore, to perform the integral specified above, we need a function that evaluates Legendre
polynomials \f$ P_l(x) \f$ for arbitrary orders \f$ l \f$ and for arbitrary \f$ x \in [-1, 1] \f$
arguments.
The standard library contains `std::legendre`.
However, at the time of writing, it seems that not all major compilers have implemented the function.
We therefore emulate the behaviour of `std::legendre` with a suboptimal implementation based on
simplemc::legendre_polynomial (it is good enough for our simple tutorial):

```cpp
// emulate the std::legendre function
auto legendre = [](int l, double x) {
    auto p_l = simplemc::legendre_polynomial(x);
    for (int i = 0; i < l; ++i) {
        p_l.next();
    }
    return p_l.current();
};
```

We are now ready to compute the Fourier coefficients \f$ f_k \f$:

```cpp
// calculate 10 Fourier coefficients
std::vector<double> coeffs(10);
for (int k = 0; k < static_cast<int>(coeffs.size()); ++k) {
    // define the integrand g_k(y) = f(y) p_k(x(y)) w(x(y))
    auto g = [&](double y) {
        const auto x_y = y_x.inverse_map(y);
        return f(y) * legendre(k, x_y) * simplemc::legendre_polynomial::weight(x_y);
    };

    // perform the integral I_k = \int_2^5 g_k(y) dy
    auto [i_k, err, n] = simplemc::simpson_quadrature(g, domain_f[0], domain_f[1]);

    // get the coefficient f_k = I_k / (N_k \alpha)
    coeffs[k] = i_k / simplemc::legendre_polynomial::norm(k) / y_x.params().first;
}
```

First, we construct a vector that holds the coefficients.
As we will see below, 10 coefficients are enough for our case.
We then use a `for`-loop to calculate each \f$ f_k \f$ separately with the following procedure:
- Define a lambda function that represents the integrand \f$ g_k(y) = P_k(x(y)) W(y) f(y) = P_k(x(y))
f(y) \f$ (we could remove simplemc::legendre_polynomial::weight since it always returns \f$ 1 \f$).
- Perform the integral \f$ I_k = \int_2^5 g_k(y) dy = \int_2^5 P_k(x(y)) f(y) dy \f$ using
simplemc::simpson_quadrature.
- Compute the Fourier coefficient \f$ f_k = I_k / (N_k \alpha) \f$.

As a final step, we define a lambda that performs the actual Fourier sum \f$ S_n(y) = \sum_{l=0}^n
f_l P_l(x(y)) \f$ for a given \f$ y \in [2, 5] \f$ argument and up to a given maximal order \f$ n \f$:

```cpp
// lambda to sum the truncated Fourier series S_n(y) = \sum_{l=0}^{n} f_l p_l(x(y))
auto fourier_sum = [&coeffs, &y_x](int n, double y) {
    auto p_l = simplemc::legendre_polynomial(y_x.inverse_map(y));
    double sum = 0.0;
    for (int l = 0; l <= n; ++l) {
        sum += coeffs[l] * p_l.next();
    }
    return sum;
};
```

@subsection tut_numeric_2_details_test Testing

Let's test our approximation.

```cpp
// create the grid at which we want to test our function approximation
simplemc::linear_grid test_grid { domain_f[0], domain_f[1], 11 };

// test the function approximation f(y) \approx S_n(y) for n = 3, 5, 7, 10
fmt::println("{:<10}{:<20}{:<20}{:<20}{:<20}{:<20}", "y", "S_3(y)", "S_5(y)", "S_7(y)", "S_10(y)", "f(y)");
for (auto y : simplemc::grid_view(test_grid)) {
    fmt::println("{:<10.1f}{:<20.8g}{:<20.8g}{:<20.8g}{:<20.8g}{:<20.8g}", y, fourier_sum(3, y), fourier_sum(5, y),
        fourier_sum(7, y), fourier_sum(10, y), f(y));
}
fmt::println("");
```

We construct a simplemc::linear_grid on the interval \f$ [2, 5] \f$ with 11 equally spaced grid points
\f$ y_i \f$ at which we will evaluate the truncated Fourier sums \f$ S_n(y_i) \f$.
Then we loop over the grid points and output the results for \f$ S_3(y_i) \f$, \f$ S_5(y_i) \f$,
\f$ S_7(y_i) \f$ and \f$ S_{10}(y_i) \f$ as well as the exact \f$ f(y_i) = e^{y_i} \f$.

Output:

```
y         S_3(y)              S_5(y)              S_7(y)              S_10(y)             f(y)
2.0       5.876862            7.3540199           7.3886481           7.3890533           7.3890561
2.3       10.284328           9.988444            9.9741964           9.9741816           9.9741825
2.6       14.135096           13.458356           13.463643           13.463739           13.463738
2.9       18.44093            18.162749           18.174256           18.174145           18.174145
3.2       24.213596           24.534395           24.532557           24.53253            24.53253
3.5       32.464859           33.12755            33.115331           33.115453           33.115452
3.8       44.206485           44.705656           44.701192           44.701184           44.701184
4.1       60.450237           60.329036           60.340413           60.340287           60.340288
4.4       82.207881           81.442605           81.450777           81.45087            81.450869
4.7       110.49118           109.96356           109.94717           109.94717           109.94717
5.0       146.31191           148.3691            148.41267           148.41316           148.41316
```

As expected, the approximation gets better the more Fourier coefficients we include in the sum.

@section tut_numeric_2_code Full code

```cpp
#include <fmt/base.h>
#include <simplemc/grids.hpp>
#include <simplemc/numeric.hpp>

#include <array>
#include <cmath>
#include <vector>

int main() {
    // define the function f(y) = e^y
    auto f = [](double y) { return std::exp(y); };

    // define the domain [c, d] = [2, 5]
    constexpr auto domain_f = std::array<double, 2> { 2.0, 5.0 };

    // linear map y : [-1, 1] -> [2, 5]
    auto y_x = simplemc::linear_map(simplemc::legendre_polynomial::domain(), domain_f);

    // emulate the std::legendre function
    auto legendre = [](int l, double x) {
        auto p_l = simplemc::legendre_polynomial(x);
        for (int i = 0; i < l; ++i) {
            p_l.next();
        }
        return p_l.current();
    };

    // calculate 10 Fourier coefficients
    std::vector<double> coeffs(10);
    for (int k = 0; k < static_cast<int>(coeffs.size()); ++k) {
        // define the integrand g_k(y) = f(y) p_k(x(y)) w(x(y))
        auto g = [&](double y) {
            const auto x_y = y_x.inverse_map(y);
            return f(y) * legendre(k, x_y) * simplemc::legendre_polynomial::weight(x_y);
        };

        // perform the integral I_k = \int_2^5 g_k(y) dy
        auto [i_k, err, n] = simplemc::simpson_quadrature(g, domain_f[0], domain_f[1]);

        // get the coefficient f_k = I_k / (N_k \alpha)
        coeffs[k] = i_k / simplemc::legendre_polynomial::norm(k) / y_x.params().first;
    }

    // lambda to sum the truncated Fourier series S_n(y) = \sum_{l=0}^{n} f_l p_l(x(y))
    auto fourier_sum = [&coeffs, &y_x](int n, double y) {
        auto p_l = simplemc::legendre_polynomial(y_x.inverse_map(y));
        double sum = 0.0;
        for (int l = 0; l <= n; ++l) {
            sum += coeffs[l] * p_l.next();
        }
        return sum;
    };

    // create the grid at which we want to test our function approximation
    simplemc::linear_grid test_grid { domain_f[0], domain_f[1], 11 };

    // test the function approximation f(y) \approx S_n(y) for n = 3, 5, 7, 10
    fmt::println("{:<10}{:<20}{:<20}{:<20}{:<20}{:<20}", "y", "S_3(y)", "S_5(y)", "S_7(y)", "S_10(y)", "f(y)");
    for (auto y : simplemc::grid_view(test_grid)) {
        fmt::println("{:<10.1f}{:<20.8g}{:<20.8g}{:<20.8g}{:<20.8g}{:<20.8g}", y, fourier_sum(3, y), fourier_sum(5, y),
            fourier_sum(7, y), fourier_sum(10, y), f(y));
    }
    fmt::println("");
}
```
