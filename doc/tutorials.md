@page tutorials Tutorials

[TOC]

| Tutorial | Description |
|---------|-------------|
| @ref tut_utils_1 | Measure the runtime of your program using simplemc::timer |
| @ref tut_utils_2 | Convert multi-dimensional to flat indices using simplemc::flat_index |
| @ref tut_utils_3 | Formatted output with **fmt** |
| @ref tut_mpi_1 | Introduction to simplemc::mpi::environment and simplemc::mpi::communicator |
| @ref tut_mpi_2 | How to handle errors in MPI programs |
| @ref tut_mpi_3 | How to use various @ref simplemc-mpi-coll |
| @ref tut_mpi_4 | Make your custom type compatible with @ref simplemc-mpi |
| @ref tut_mpi_5 | Working with MPI groups and communicators |
| @ref tut_random_1 | Approximating PDFs with histograms from @ref simplemc-random-samples |
| @ref tut_grids_1 | Showcase the different @ref simplemc-grids-1d |
| @ref tut_grids_2 | Showcase @ref simplemc-grids-nd |
| @ref tut_numeric_1 | Working with @ref simplemc-numeric-lattices |
| @ref tut_numeric_2 | Approximate a function using @ref simplemc-numeric-poly |
| @ref tut_accs_1 | Estimating errors with @ref simplemc-accs |
| @ref tut_mc_1 | A minimal MC integration with @ref simplemc-mc |
| @ref tut_mc_2 | MC integration with checkpointing and input-config files |
| @ref tut_mc_3 | Parallelizing MC simulations with @ref simplemc-mpi |

@section tut_compilation Compiling the tutorials

All tutorials have been compiled on a MacBook Pro with an Apple M2 Max chip and *clang++* v22.1.8
together with *CMake* v4.3.4.

Furthermore, the following dependencies have been used:
- [open-mpi](https://www.open-mpi.org/) 5.0.9 (installed with brew)
- [hdf5](https://www.hdfgroup.org/solutions/hdf5/) 2.1.1 (installed with brew)

Assuming that the actual tutorial code is in a file `main.cpp`, the following generic `CMakeLists.txt`
should work for all tutorials:

```cmake
cmake_minimum_required(VERSION 3.28)
project(example CXX)

# fetch simplemc from github
include(FetchContent)
FetchContent_Declare(
  simplemc
  GIT_REPOSITORY git@github.com:Thoemi09/simplemc.git
  GIT_TAG main
)
FetchContent_MakeAvailable(simplemc)

# build the example
add_executable(ex main.cpp)
target_link_libraries(ex simplemc::simplemc)
```
