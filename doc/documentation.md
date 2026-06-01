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

- @ref simplemc-utils-concepts that are not specific to any library.
- Generic @ref simplemc-utils-exceptions that are used by the other **simplemc** libraries and can
also be used in user code.
- Some @ref simplemc-utils-fileio functionalities.
- @ref simplemc-utils-indexing to map multi-dimensional indices to flat indices and vice versa.
- @ref simplemc-utils-other including formatted output for complex numbers and conversion to strings
for types with an overloaded `operator<<`.
- A @ref simplemc-utils-timer for easy performance and runtime measurments.

## MPI library

@ref simplemc-mpi is a minimal C++ wrapper around MPI C libraries that exposes a small subset of
the functionality defined in the MPI standard.

It provides

- @ref simplemc-mpi-coll to perform @ref simplemc-mpi-coll-bcast, @ref simplemc-mpi-coll-reduce,
@ref simplemc-mpi-coll-gather and @ref simplemc-mpi-coll-scatter operations across multiple processes.
- @ref simplemc-mpi-essentials that wrap the most important MPI objects and functions. They are
further grouped into @ref simplemc-mpi-essentials-env, @ref simplemc-mpi-essentials-comms,
@ref simplemc-mpi-essentials-groups and @ref simplemc-mpi-essentials-types.
- @ref simplemc-mpi-utils that contain some helper functions.

## Random library

@ref simplemc-random implements tools for generating random numbers and sampling various probability
distributions.

It provides

- @ref simplemc-random-rngs (RNGs) for fast random number generation that satisfy the C++ named
requirements [RandomNumberEngine](https://en.cppreference.com/w/cpp/named_req/RandomNumberEngine).
- Functions that draw @ref simplemc-random-samples from often used probability distributions and that
evaluate their corresponding probability density functions (PDFs).

## Grid library

@ref simplemc-grids implements various @ref simplemc-grids-1d on the real line. They can be further
combined to form @ref simplemc-grids-nd.

A 1-dimensional grid is a strictly monotone map \f$ g : \mathrm{I} \to \mathbb{R} \f$ from the integer
set \f$ \mathrm{I} = \{0, 1, \ldots, M-1\} \f$ to a closed interval on the real line (see
@ref simplemc-grids-1d).

Depending on the map, one can define different grids:

- @ref simplemc::linear_grid uses a linear map with equally spaced grid points.
- @ref simplemc::power_grid uses a mapping which takes a non-negative power \f$ p \f$ of the integer
variable. It reduces to the simplemc::linear_grid in case that \f$ p = 1.0 \f$.
- @ref simplemc::symmetric_power_grid consists of two simplemc::power_grid which are symmetric with
respect to the center of the interval.
- @ref simplemc::custom_grid uses any strictly ordered array (increasing or decreasing) as its grid
points.

Users can define their own grids by simply inheriting from the abstract simplemc::grid_base class and
implementing the purely virtual methods.

Furthermore, the simplemc::nd_grid class lets the user combine an arbitrary number of 1-dimensional
grids to form @ref simplemc-grids-nd.

## Numerics library

@ref simplemc-numeric contains some basic tools and functionality useful in many numerical
calculations and physics related simulations.

It provides

- Various @ref simplemc-numeric-lattices in 1D, 2D and 3D.
- @ref simplemc-numeric-interpolation to perform linear and polynomial interpolation of arbitrary
order (in N-dimensions) as well as cubic spline interpolation (only in 1D).
- General @ref simplemc-numeric-utils that include the Eigen library and other convenient features to
simplify numerical calculations.
- @ref simplemc-numeric-quadrature to do basic numerical integration in 1D.
- Some implementations of @ref simplemc-numeric-poly, like Legendre polynomials or Chebyshev
polynomials, that are especially useful for general Fourier series.

## Accumulator library

@ref simplemc-accs implements various accumulators and other tools to collect and analyze data.

It provides

- @ref simplemc-accs-concepts that define accumulator specific requirements.
- @ref simplemc-accs-utils which are mostly used internally in the **simplemc-accs** library.
- Several @ref simplemc-accs-accs to calculate the mean, variance and covariance of a collection of
data values.
- Various @ref simplemc-accs-wrappers that can be used with specific accumulator types to extend or
modify their functionality.
- Tools and techniques for @ref simplemc-accs-resampling.

## Serialization library

@ref simplemc-serialize is a generic serialization framework for multiple backends. It is split into
a backend-agnostic core and one backend per shipped format.

It provides

- @ref simplemc-serialize-core — header-only core with the @ref simplemc::serializer and
@ref simplemc::deserializer concepts, the ADL customization point
(`simplemc_save` / `simplemc_load`), and explicit element-wise helpers for ranges and tuples.
Library types (accumulators, grids, lattices, RNG state) ship with `simplemc_save` / `simplemc_load`
overloads next to each class so they round-trip out of the box.
- @ref simplemc-serialize-json — JSON backend on top of [nlohmann_json](https://github.com/nlohmann/
json), with @ref simplemc::json_serializer / @ref simplemc::json_deserializer, text- and binary-mode
file I/O helpers, and nlohmann-side adapters for `std::complex` and Eigen vector/matrix types.

<!-- ## Monte Carlo library -->
