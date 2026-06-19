# simplemc

Tools for creating Monte Carlo simulations.

## Dependencies

| Dependency | Required | Notes |
|---|---|---|
| C++23 compiler (GCC ≥ 14 or Clang ≥ 19) | yes | |
| CMake ≥ 3.24 | yes | |
| MPI (e.g. OpenMPI) | yes | always resolved via `find_package` |
| fmt | yes | fetched from GitHub by default |
| range-v3 | yes | fetched from GitHub by default (skipped when using GCC with `SIMPLEMC_USE_STD_RANGES=ON`) |
| Eigen3 | yes | fetched from GitLab by default |
| nlohmann_json | yes | fetched from GitHub by default |
| HDF5 C library | optional | only needed when `SIMPLEMC_USE_HDF5=ON` |
| HighFive | optional | only needed when `SIMPLEMC_USE_HDF5=ON`; fetched by default |
| googletest | optional | only needed when `SIMPLEMC_BUILD_TESTS=ON`; always fetched |

MPI and (optionally) HDF5 must be installed on your system. All other dependencies are fetched
automatically by CMake at configure time unless you disable the corresponding `SIMPLEMC_FETCH_*` flag.

### NixOS

Add to `environment.systemPackages` in your NixOS configuration:

```nix
environment.systemPackages = with pkgs; [
  cmake
  openmpi
  # hdf5  # optional, only needed if SIMPLEMC_USE_HDF5=ON
];
```

`git` must also be available (needed by CMake's FetchContent); it is covered by
`programs.git.enable = true`.

## Building

```sh
# configure
cmake -B build

# compile
cmake --build build -j$(nproc)

# run tests
ctest --test-dir build --output-on-failure

# install (default prefix: build/install)
cmake --install build
```

### Common CMake options

| Option | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | — | `Debug`, `Release`, `RelWithDebInfo` |
| `CMAKE_INSTALL_PREFIX` | `build/install` | Installation prefix |
| `SIMPLEMC_BUILD_TESTS` | `ON` | Build unit tests with googletest |
| `SIMPLEMC_BUILD_EXAMPLES` | `ON` | Build example/tutorial executables |
| `SIMPLEMC_BUILD_DOC` | `OFF` | Build Doxygen documentation |
| `SIMPLEMC_USE_HDF5` | `OFF` | Enable HDF5 serialization backend |
| `SIMPLEMC_USE_STD_RANGES` | `ON` (GCC only) | Use `std::ranges` instead of range-v3 |
| `SIMPLEMC_FETCH_FMT` | `ON` | Fetch fmt from GitHub |
| `SIMPLEMC_FETCH_RANGEV3` | `ON` | Fetch range-v3 from GitHub |
| `SIMPLEMC_FETCH_EIGEN3` | `ON` | Fetch Eigen3 from GitLab |
| `SIMPLEMC_FETCH_NLOHMANN_JSON` | `ON` | Fetch nlohmann_json from GitHub |
| `SIMPLEMC_FETCH_HIGHFIVE` | `ON` | Fetch HighFive from GitHub |
| `SIMPLEMC_DISABLE_WARNINGS` | `ON` | Disable stricter compiler warnings |

Example — release build with HDF5:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSIMPLEMC_USE_HDF5=ON
cmake --build build -j$(nproc)
```

Example — use system-installed libraries instead of fetching:

```sh
cmake -B build \
  -DSIMPLEMC_FETCH_FMT=OFF \
  -DSIMPLEMC_FETCH_RANGEV3=OFF \
  -DSIMPLEMC_FETCH_EIGEN3=OFF \
  -DSIMPLEMC_FETCH_NLOHMANN_JSON=OFF
```

## Running examples

After building with `SIMPLEMC_BUILD_EXAMPLES=ON`, executables land in `build/examples/`:

```sh
# random distributions
./build/examples/tutorials/tut_random_1

# grids
./build/examples/tutorials/tut_grids_1

# accumulators
./build/examples/tutorials/tut_accs_1

# numeric / lattice
./build/examples/tutorials/tut_numeric_1

# MPI hello world (4 processes)
mpirun -n 4 ./build/examples/mpi/doc_mpi_hello_world
```

## Running tests

```sh
ctest --test-dir build --output-on-failure

# filter by name
ctest --test-dir build -R grids --output-on-failure
```

## Using simplemc in your own project

### Via FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(simplemc
  GIT_REPOSITORY https://github.com/Thoemi09/simplemc.git
  GIT_TAG        main)
FetchContent_MakeAvailable(simplemc)

target_link_libraries(my_target PRIVATE simplemc::simplemc)
```

### Via find_package (after installing)

```cmake
find_package(simplemc REQUIRED CONFIG)
target_link_libraries(my_target PRIVATE simplemc::simplemc)
```

Set `simplemc_DIR` to `<install_prefix>/lib/cmake/simplemc` if CMake cannot find the package
automatically.

### Via add_subdirectory

```cmake
add_subdirectory(deps/simplemc)
target_link_libraries(my_target PRIVATE simplemc::simplemc)
```

### Including headers

```cpp
#include <simplemc/mpi.hpp>
#include <simplemc/random.hpp>
#include <simplemc/grids.hpp>
#include <simplemc/numeric.hpp>
#include <simplemc/accs.hpp>
#include <simplemc/serialize.hpp>
#include <simplemc/utils.hpp>
```
