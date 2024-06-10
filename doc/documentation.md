@page documentation API Documentation

[TOC]

**simplemc** is a C++ library that provides various tools to help users develop their own Monte Carlo
codes.
It is implemented as several (partially independent) sublibraries, each with its own specific purpose.
The following provides a detailed reference documentation.

If you are looking for a specific function, class, etc., try using the search bar in the top left
corner.

## Utilities library

@ref simplemc-utils forms the backbone of **simplemc**. All other sublibraries link to it.

Besides offering some general functionality for the other libraries, like concepts, type traits, or
exceptions, it also contains a useful toolset for the user.

It provides

- Generic @ref simplemc-utils-exceptions that are used by the other **simplemc** libraries and can
also be used in user code.
- @ref simplemc-utils-general including some general concepts, formatted output for complex numbers
and other convenient tools.
- @ref simplemc-utils-indexing to map multi-dimensional indices to a flat index and vice versa.
- @ref simplemc-utils-timer for easy performance and runtime measurments.

## MPI library

@ref simplemc-mpi is a minimal C++ wrapper around MPI C libraries that provides a small subset of
the functionality defined in the MPI standard.

It is not intended to be a full replacement of the C implementations.
Instead, it tries to simplify some common tasks like initializing/finalizing MPI execution
environments or performing non-blocking collective communications.

It provides

- @ref simplemc-mpi-coll to perform MPI broadcast, gather, scatter and reduce operations across
multiple processes.
- @ref simplemc-mpi-commenv classes as highlevel C++ wrappers around MPI communicator objects and MPI
execution enivronments.
- @ref simplemc-mpi-types to map supported C++ types to their corresponding MPI datatypes.
- @ref simplemc-mpi-utils for the **simplemc-mpi** library.

## Random library

@ref simplemc-random implements tools for generating random numbers and sampling various probability
distributions.

It provides

- @ref simplemc-random-rngs for fast random number generation that satisfy the C++ named requirements
[RandomNumberEngine](https://en.cppreference.com/w/cpp/named_req/RandomNumberEngine).
- Some @ref simplemc-random-dists that satisy the C++ named requirements
[RandomNumberDistribution](https://en.cppreference.com/w/cpp/named_req/RandomNumberDistribution).
- @ref simplemc-random-samples from often used probability density functions.

## Grid library

@ref simplemc-grids implements various grids that can be used in histograms.

A grid is a monotonous map \f$ g \f$ from the integer set \f$ \{0, 1, \ldots, N-1\} \f$ to a closed,
real interval.

The mapping defines different grids:

- @ref simplemc::linear_grid uses a linear map with equally spaced grid points.
- @ref simplemc::nd_grid is a generic extension to multi dimensions.
- @ref simplemc::power_grid uses a mapping which takes a non-negative power \f$ p \f$ of the integer
variable. It reduces to the simplemc::linear_grid in case that \f$ p = 1.0 \f$.
- @ref simplemc::symmetric_power_grid consists of two simplemc::power_grid which are symmetric with
respect to the center of the interval.

User can define their own grids by simply inheriting from simplemc::grid_base and implementing the
purely virtual methods.

## Numerics library

@ref simplemc-numeric contains some basic tools and functionality useful in many numerical and physics related simulations:

- @ref simplemc::bravais_lattice defines all the Bravais lattice types and functions to easily construct them.
- @ref simplemc::cubic_spline_interpolation takes a grid and function values as input and performs cubic spline interpolation.
- @ref simplemc::linear_interpolation and @ref simplemc::linear_interpolation_nd performs linear interpolation in 1- and N-dimensional
space, respectively.
- @ref simplemc::polynomial_interpolation and @ref simplemc::polynomial_interpolation_nd performs polynomial interpolation to an arbitrary
order in 1- and N-dimensional space, respectively.
- Numerical quadrature can be done with @ref simplemc::trapezs_quadrature and @ref simplemc::simpson_quadrature.
- Various orthogonal polynomials are implemented and can be used in general Fourier series:
  - @ref simplemc::legendre_polynomial
  - @ref simplemc::chebyshev_polynomial_first and @ref simplemc::chebyshev_polynomial_second
  - @ref simplemc::hermite_polynomial
  - @ref simplemc::laguerre_polynomial
  - @ref simplemc::cosine and @ref simpelmc::sine

## Accumulator library

@ref simplemc-accs provides various accumulators and other tools to collect and analyze data:

- @ref simplemc::mean_acc
- @ref simplemc::var_acc
- @ref simplemc::block_acc
- @ref simplemc::covar_acc
- @ref simplemc::autocorr_acc

## JSON library

@ref simplemc-json adds some additional functionality to the [nlohmann_json](https://github.com/nlohmann/json) library:

- @ref simplemc::basic_json is an abstract base class that makes it easy for deriving types to be serialized/deserialized to JSON.
- Easy to use functions for writing/reading JSON files in text as well as binary mode.
- Specialized serializers for `std::complex` and general `std::range` types.

<!-- ## Monte Carlo library -->
