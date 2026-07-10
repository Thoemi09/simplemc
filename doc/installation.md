@page installation Installation

[TOC]

**simplemc** supports the usual installation procedure using CMake.

If you want to skip the installation step, you can go directly to @ref integration to see how you can
integrate **simplemc** into your own C++ project by using CMake's @ref fetch.

@section dependencies Dependencies

The minimum dependencies of **simplemc** are as follows:

* a recent g++ OR clang++ compiler (tested with gcc v16.1.0 and clang v22.1.8)
* a recent CMake version (tested with v4.3.4)
* a working MPI implementation (tested with open-mpi v5.0.9)

Furthermore, **simplemc** depends on some external libraries. They can either be fetched directly by
**simplemc** or provided by the user.

The following libraries are required for **simplemc** to work:

* [fmt](https://github.com/fmtlib/fmt): Formatting library (as of now `std::format` and `std::print`
are not fully supported by most compilers).
* [range-v3](https://github.com/ericniebler/range-v3): Range library (as of now the ranges library is
not complete in the standard library).
* [Eigen3](https://gitlab.com/libeigen/eigen): Linear algebra libray.
* [nlohmann_json](https://github.com/nlohmann/json): JSON backend for the simplemc-serialize library.

The following libraries are optional and only required for certain features:

* [googletest](https://github.com/google/googletest): Unit testing framework (only required when 
`SIMPLEMC_BUILD_TESTS=ON`).
* [HDF5](https://www.hdfgroup.org/solutions/hdf5/): HDF5 library for serialization (only required when 
`SIMPLEMC_USE_HDF5=ON`)
* [HighFive](https://github.com/highfive-devs/highfive): C++ HDF5 wrapper for serialization (only 
required when `SIMPLEMC_USE_HDF5=ON`).

@section install_steps Installation steps

1. Download the source code by cloning [simplemc](https://github.com/Thoemi09/simplemc) from GitHub:

    ```console
    $ git clone https://github.com/Thoemi09/simplemc simplemc.src
    ```

2. Create and change to a build directory:

    ```console
    $ mkdir simplemc.build && cd simplemc.build
    ```

3. In the build directory call CMake (including any additional @ref cmake_options):

    ```console
    $ cmake ../simplemc.src
    ```

4. Compile the code, run the tests and install the application:

    ```console
    $ make -j N
    $ make test
    $ make install
    ```

    Replace \f$ N \f$ with the number of cores you want to use to build the library.

@section cmake_options Custom CMake options

The compilation of **simplemc** can be configured using various CMake options:

```console
$ cmake ../simplemc.src -DOPTION1=value1 -DOPTION2=value2 ...
```

| Options                                 | Syntax                                            |
|-----------------------------------------|---------------------------------------------------|
| Specify an installation path            | ``-DCMAKE_INSTALL_PREFIX=path_to_install_dir``    |
| Build type                              | ``-DCMAKE_BUILD_TYPE=type``                       |
| Enable install target                   | ``-DSIMPLEMC_ENABLE_INSTALL=ON``                  |
| Build tests                             | ``-DSIMPLEMC_BUILD_TESTS=ON``                     |
| Build examples                          | ``-DSIMPLEMC_BUILD_EXAMPLES=ON``                  |
| Build stochastic process tests          | ``-DSIMPLEMC_BUILD_STOCHASTIC_PROCESS_TESTS=ON``  |
| Build documentation with Doxygen        | ``-DSIMPLEMC_BUILD_DOC=ON``                       |
| Use ranges from standard library        | ``-DSIMPLEMC_USE_STD_RANGES=ON``                  |
| Enable HDF5 backend for serialization   | ``-DSIMPLEMC_USE_HDF5=ON``                        |
| Fetch fmt from github                   | ``-DSIMPLEMC_FETCH_FMT=ON``                       |
| Fetch range-v3 from github              | ``-DSIMPLEMC_FETCH_RANGEV3=ON``                   |
| Fetch Eigen3 from gitlab                | ``-DSIMPLEMC_FETCH_EIGEN3=ON``                    |
| Fetch nlohmann_json library from github | ``-DSIMPLEMC_FETCH_NLOHMANN_JSON=ON``             |
| Fetch HighFive library from github      | ``-DSIMPLEMC_FETCH_HIGHFIVE=ON``                  |
| Disable stricter compilation warnings   | ``-DSIMPLEMC_DISABLE_WARNINGS=ON``                |

Please check out the toplevel [`CMakeLists.txt`](https://github.com/Thoemi09/simplemc/blob/main/CMakeLists.txt)
to see what the default options are.
