# simplemc

**simplemc** is a C++ library that provides various tools to simplify the implementation of Monte
Carlo (MC) simulations, with a special focus on applications in computational physics. Its
functionality is similar to other established codes like [ALPSCore](https://alpscore.org/) or the MC
part of [TRIQS](https://triqs.github.io/triqs/latest/).

It helps with the common, error-prone tasks encountered in nearly all MC implementations — data
accumulation, statistical analysis and error propagation, MPI parallelization, serialization of the
simulation state, parsing of user input and configuration files, and more — so that developers and
researchers can focus on the actual (physics) problem.

📖 **Full documentation, tutorials and API reference:
[thoemi09.github.io/simplemc](https://thoemi09.github.io/simplemc/)**

```cpp
#include <simplemc/mpi.hpp>
#include <fmt/core.h>

int main(int argc, char** argv) {
  // initialize MPI environment and communicator
  simplemc::mpi::environment env(argc, argv);
  simplemc::mpi::communicator comm;

  // print message on each process
  fmt::print("Hello world, from {} of {} processes.\n", comm.rank(), comm.size());
}
```

## Sublibraries

**simplemc** is implemented as several (partially independent) sublibraries, each with its own
specific purpose. The top-level `simplemc::simplemc` target aggregates all of them.

| Sublibrary | Description |
|------------|-------------|
| **utils**     | The foundation all other sublibraries link to.                                   |
| **mpi**       | A minimal C++ wrapper around the MPI C libraries.                                |
| **random**    | Tools for random number generation.                                              |
| **grids**     | 1- and N-dimensional grids.                                                      |
| **numeric**   | Bravais lattices, interpolation, quadrature, and orthogonal polynomials.         |
| **accs**      | Statistical accumulators and tools for mean, variance and covariance estimation. |
| **serialize** | A generic, extensible serialization framework.                                   |
| **mc**        | Everything needed for setting up a MC simulation.                                |

## Quick start

**simplemc** uses the usual CMake build procedure; out-of-source builds are mandatory. All external
dependencies are fetched automatically via CMake's `FetchContent` by default.

```console
$ git clone https://github.com/Thoemi09/simplemc simplemc.src
$ mkdir simplemc.build && cd simplemc.build
$ cmake ../simplemc.src
$ make -j N && make test && make install
```

For the list of dependencies, the full set of CMake options, and how to integrate **simplemc** into
your own project (`FetchContent`, `find_package`, `add_subdirectory`), see:

* [Installation](https://thoemi09.github.io/simplemc/installation.html)
* [Integration in C++ projects](https://thoemi09.github.io/simplemc/integration.html)

The [Tutorials](https://thoemi09.github.io/simplemc/tutorials.html) on the documentation page are a
good starting point to get familiar with the library, its philosophy and its features.

For a more detailed outline of the library's functionality, see the 
[API reference](https://thoemi09.github.io/simplemc/api.html).

## Contributors

This project has been in development for some years and has been refactored from the ground up a
couple of times. 

Contributions along the way came from 

- [samox73](https://github.com/samox73), 
- [st3r4g](https://github.com/st3r4g), and
- [sda-express-one](https://github.com/sda-express-one).

## Questions, issues and contributions

If you have any questions, issues or suggestions, please feel free to open an issue or a pull request
on [GitHub](https://github.com/Thoemi09/simplemc). 

Every bug report, feature request and contribution is **highly appreciated** and will be reviewed as 
soon as possible.
