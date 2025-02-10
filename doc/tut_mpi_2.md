@page tut_mpi_2 MPI Tutorial 2: Exception handling

[TOC]

In this tutorial, we show how the variable `abort_on_exception` influences the behavior of a
simplemc::mpi::environment.

@section tut_mpi_2_details Step-by-step guide

A simplemc::mpi::environment takes 3 arguments in its constructor.
The first two are the usual arguments (`argc` and `argv`) passed to the `main` function, which are
simply forwarded to the `MPI_Init` call.
The third one, called `abort_on_exception`, is a boolean value that determines how an environment
object should behave when an exception is encountered.
More specifically, it determines the behavior of its destructor.

By default, `abort_on_exception` is set to `true`.
That means the environment object will use `std::uncaught_exceptions()` in its destructor to check if
an exception has been thrown somewhere and has not been caught.
If this is the case, it will call `MPI_Abort`.

On the other hand, if there are no uncaught exceptions or if `abort_on_exception` is set to `false`,
the destructor will call `MPI_Finialize`.

The following code demonstrates the default behaviour:

```cpp
#include <fmt/ostream.h>
#include <simplemc/mpi.hpp>

#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
    try {
        // initialize MPI environment and communicator
        simplemc::mpi::environment env(argc, argv, /* abort_on_exception */ true);
        simplemc::mpi::communicator comm;

        // throw an exception on process 0
        if (comm.rank() == 0) {
            throw simplemc::simplemc_exception("This is a test exception on rank 0");
        }

        // on all other processes, sleep for 3 seconds and print a message
        std::this_thread::sleep_for(std::chrono::seconds(3));
        fmt::println("Hello from rank {}", comm.rank());
    } catch(const simplemc::simplemc_exception& e) {
        // catch any exceptions (no exception should be caught)
        fmt::println(std::cerr, "{}", e.what());
    }
}
```

Here, we wrap our simplemc::mpi::environment object in a `try-catch` block.
Rank 0 throws an exception, while the other ranks use `std::this_thread::sleep_for` to sleep for 3
seconds before they print a message to `stdout`.

Let's run this code with 4 processes:

```
--------------------------------------------------------------------------
MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD
  Proc: [[34368,1],0]
  Errorcode: -1

NOTE: invoking MPI_ABORT causes Open MPI to kill all MPI processes.
You may or may not see output from other processes, depending on
exactly when Open MPI kills them.
--------------------------------------------------------------------------
```

The program ends immediately after starting it and prints the message shown above.
This is expected as rank 0 throws an exception and due to stack unwinding the
simplemc::mpi::environment object gets destroyed.
It sees an uncaught exception in its destructor and therefore invokes `MPI_Abort`.

Now, let's see what happens when we set `abort_on_exception` to false, i.e. simply change the line

```cpp
simplemc::mpi::environment env(argc, argv);
```

to

```cpp
simplemc::mpi::environment env(argc, argv, /* abort_on_exception */ false);
```

If we run the program again, we get

```
Hello from rank 2
Hello from rank 1
Hello from rank 3
simplemc exception: This is a test exception on rank 0
```

Although rank 0 still throws an exception, the program finishes without any error messages from MPI.
The environment object still gets destroyed due to stack unwinding on rank 0 and it still sees an
uncaught exception in its destructor but since we constructed the environment with
`abort_on_exception = false`, it simply calls `MPI_Finalize` and waits for the other ranks to finish
their work.

> **Note**: Although the second approach might seem nicer since it ends the program more gracefully,
> it is recommended to use the default behaviour unless one has full controll over all possible MPI
> calls in a program.
>
> In the example above, if there would be another (collective) MPI call after the line where rank 0
> throws the exception, all other ranks would be deadlocked in this call waiting for rank 0 which
> itself would be deadlocked in the destructor of the environment object.
