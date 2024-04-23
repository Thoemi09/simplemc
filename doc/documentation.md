@page documentation API Documentation

[TOC]

**simplemc** is a C++ library that provides various tools to help users develop their own Monte Carlo codes.
It is implemented as several (partially independent) sublibraries, each with its own specific purpose.
The following provides a detailed reference documentation.

If you are looking for a specific function, class, etc., try using the search bar in the top left corner.

## Utilities library

@ref simplemc-utils forms the backbone of **simplemc**. All other sublibraries link to it.

Besides offering some general functionality for the other libraries, like concepts, type traits, or exceptions,
it also provides a useful toolset for the user.

For example,

- simplemc::timer: A timer class that makes it easy to measure basic performance or runtime.
- simplemc::multi_index and simplemc::flat_index: Functions to map multi-dimensional indices to a flat index and vice versa.
- simplemc::open_file and simplemc::close_file: Open and close files.
- A specialized formatter for the [fmt](https://github.com/fmtlib/fmt) library to format and print std::complex numbers.

## MPI library

@ref simplemc-mpi is a minimal C++ wrapper around the MPI C library and provides only a small subset of the functionality defined
in the MPI standard.

The main purpose of the library is to simplify the most common tasks like initializing/finalizing the MPI execution environment or
performing non-blocking collective communications.

For more advanced tasks, the user can always resort to the underlying MPI C-implementation.

It provides

- @ref simplemc-mpi-coll to perform MPI broadcast, gather, scatter and reduce operations across multiple processes.
- A @ref simplemc::mpi::communicator class as a highlevel C++ wrapper around an MPI communicator object.
- A @ref simplemc::mpi::environment class to simplify the initialization and finalization of MPI execution enivronments.
- @ref simplemc-mpi-types to map supported C++ types to their corresponding MPI datatypes.
- @ref simplemc-mpi-utils for the **simplemc-mpi** library.

## JSON library

@ref simplemc-json adds some additional functionality to the [nlohmann_json](https://github.com/nlohmann/json) library:

- @ref simplemc::basic_json is an abstract base class that makes it easy for deriving types to be serialized/deserialized to JSON.
- Easy to use functions for writing/reading JSON files in text as well as binary mode.
- Specialized serializers for `std::complex` and general `std::range` types.

## HDF5 library

## Random library

@ref simplemc-random provides tools for generating random numbers and sampling various probability distributions:

- @ref simplemc::splitmix64 is a very fast and lightweight RNG. It is intended to be used for simple tasks like seeding other more
sophisticated RNGs.
- @ref simplemc::xoshiro256 is another fast RNG that has better statistics than the @ref simplemc::splitmix64 and can be used in Monte Carlo simulations.
- Various probability distributions:
  - @ref simplemc::uniform_real_distribution is an alternative to `std::uniform_real_distribution` and intended to be used with
  @ref simplemc::xoshiro256.
  - @ref simplemc::discrete_distribution is an alternative to `std::discrete_distribution` using a linear search algorithm.
  - @ref simplemc::discrete_alias_distribution is an alternative to `std::discrete_distribution` using the Walkers's Alias method.
- Functions to sample from and to calculate various PDFs.

## Grid library

@ref simplemc-grids implements various grids that can be used in histograms.
A grid is a strictly monotonous map from the integers `[0, N-1]` to the real numbers, where `N` is the size of the grid.
The mapping defines the different grids:

- @ref simplemc::linear_grid uses a linear map with equally spaced grid points.
- @ref simplemc::nd_grid is a generic extension to multi dimensions.
- @ref simplemc::power_grid uses a power map.
- @ref simplemc::symmetric_power_grid is a specialization of the simplemc::power_grid.

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

## Monte Carlo library
