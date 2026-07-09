@page documentation API Documentation

[TOC]

**simplemc** is a C++ library that provides various tools to help users develop their own Monte Carlo
codes.
It is implemented as several (partially independent) sublibraries, each with its own specific purpose.
The following pages provide detailed reference documentation.

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
- A @ref simplemc-utils-timer for easy performance and runtime measurements.

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
- @ref simplemc::power_grid uses a mapping which takes a positive power \f$ p \f$ of the integer
variable. It reduces to the simplemc::linear_grid in case that \f$ p = 1.0 \f$.
- @ref simplemc::symmetric_power_grid consists of two simplemc::power_grid which are symmetric with
respect to the center of the interval.
- @ref simplemc::custom_grid uses any strictly ordered array (increasing or decreasing) as its grid
points.

Users can define their own grids by simply inheriting from simplemc::grid_base (a CRTP base class) and
implementing the `at()` and `index()` member functions it expects.

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

@ref simplemc-serialize is a generic, extensible serialization framework that supports multiple
backends. It defines the @ref simplemc::serializer concept and the ADL customization point
(`simplemc_save` / `simplemc_load`) that lets each type carry its own state-I/O logic without
coupling to a specific backend.

It currently ships with the following backends:

- @ref simplemc-serialize-json, implemented on top of [nlohmann_json](https://github.com/nlohmann/
json) and providing @ref simplemc::json_serializer, text- and binary-mode file I/O helpers, and
various custom nlohmann adapters. Always available.
- @ref simplemc-serialize-hdf5, implemented on top of [HighFive](https://github.com/BlueBrain/
HighFive). It requires a system HDF5 install and provides @ref simplemc::hdf5_serializer with
POSIX-style path navigation and lazy group materialization. Header-only and opt-in via the CMake
option `SIMPLEMC_USE_HDF5=ON`.
- @ref simplemc-serialize-variant, erases a set of backends behind a `std::variant`, giving a single
concrete serializer type whose active backend is chosen at runtime.

Additional backends can be added by satisfying the @ref simplemc::serializer concept.

## Monte Carlo library

@ref simplemc-mc is the Monte Carlo–specific layer of **simplemc**. It hosts the types and drivers 
needed to set up, run, and persist a Markov-chain Monte Carlo simulation, while leaving the physics 
entirely to the user.

It provides

- @ref simplemc-mc-entities, containing the user-extensible @ref simplemc-mc-entities-updates and
@ref simplemc-mc-entities-measurements that propose moves and sample observables, together with the
value-semantic wrappers and tuple-backed collections that manage them.
- @ref simplemc-mc-kernels, representing the underlying MC algorithm (defaults to 
Metropolis-Hastings).
- @ref simplemc-mc-sim, including the free simplemc::run function that performs a simulation.
- @ref simplemc-mc-callbacks, optional per-run hooks that observe a simulation.
- @ref simplemc-mc-utils, defining the serializer-generic save/load helpers and various other helpers
to support serialization and MPI communication.
