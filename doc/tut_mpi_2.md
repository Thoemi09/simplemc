@page tut_mpi_2 MPI Tutorial 2: Error handling

[TOC]

In this tutorial, we give an overview of error handling in MPI programs in general and some more
**simplemc-mpi** specific features.

@section tut_mpi_2_details Step-by-step guide

Error handling in MPI is a difficult subject.
Often it is not obvious what's the correct way to proceed after an error has been encountered.
In most cases, there is no way to recover and aborting the program is the right thing to do.

@subsection tut_mpi_2_ex Working example

Let's start with a simple working example that should not run into any errors:

@include tutorials/tut_mpi_2_1.cpp

Output (using 2 processes):

```
Process 0 received message: 42
Process 1 received message: 42
```

The example simply broadcasts an integer message (\f$ = 42 \f$) from the root process with rank 0 to
the other process with rank 1.
At the end, we print the message to check that everything worked as expected.

@subsection tut_mpi_2_default Default error handling

MPI comes with its own error handlers and the default is called `MPI_ERRORS_ARE_FATAL`.
As the name suggests, it will cause the program to abort all connected MPI processes.

To see this in action, let us change the working example from above to use an invalid root process
with rank -1:

@include tutorials/tut_mpi_2_2.cpp

Output:

```
[XL19Q7FHQT:00000] *** An error occurred in MPI_Bcast
[XL19Q7FHQT:00000] *** reported by process [4053467137,1]
[XL19Q7FHQT:00000] *** on communicator MPI_COMM_WORLD
[XL19Q7FHQT:00000] *** MPI_ERR_ROOT: invalid root
[XL19Q7FHQT:00000] *** MPI_ERRORS_ARE_FATAL (processes in this communicator will now abort,
[XL19Q7FHQT:00000] ***    and MPI will try to terminate your MPI job as well)
```

The program aborts as soon as the error is encountered.
We can see from the error message that the error occured in `MPI_Bcast` and that the cause was an
*invalid root* rank.

@subsection tut_mpi_2_ret Return error codes/Throw exceptions

We can change the default behvaiour by setting the error handler manually.

In the following example, we set the error handler of `MPI_COMM_WORLD` to `MPI_ERRORS_RETURN` which
causes MPI functions to return error codes instead of aborting the program:

@include tutorials/tut_mpi_2_3.cpp

Output:

```
libc++abi: terminating due to uncaught exception of type simplemc::simplemc_exception: simplemc exception in MPI_Bcast: MPI error code: 8
libc++abi: terminating due to uncaught exception of type simplemc::simplemc_exception: simplemc exception in MPI_Bcast: MPI error code: 8
[XL19Q7FHQT:31902] *** Process received signal ***
[XL19Q7FHQT:31902] Signal: Abort trap: 6 (6)
[XL19Q7FHQT:31902] Signal code:  (0)

// ... more output ...

[XL19Q7FHQT:31902] *** End of error message ***
--------------------------------------------------------------------------
prterun noticed that process rank 1 with PID 31903 on node XL19Q7FHQT exited on
signal 6 (Abort trap: 6).
--------------------------------------------------------------------------
```

However, running the program still results in an abort.

**simplemc-mpi** automatically checks the return codes of MPI functions and throws an exception if an
error code is returned (see simplemc::mpi::check_mpi_call).
The reason why we choose to do so is that some functions in **simplemc-mpi** return values of their
own (e.g. simplemc::mpi::comm_rank) or make multiple MPI calls internally.
In such cases, it is not clear how and which error code(s) to return.
Since exceptions are the standard way of handling errors in C++, we decided to simply throw whenever
an MPI call fails.

Coming back to our example, the invalid root rank causes simplemc::mpi::broadcast to throw a
simplemc::simplemc_exception.
Since we are not catching this exception anywhere, the program terminates and MPI aborts all
processes.

@subsection tut_mpi_2_loc Catching exceptions locally

The easiest way to catch exceptions is right at the call site:

@include tutorials/tut_mpi_2_4.cpp

Output (using 2 processes):

```
Caught exception during broadcast on process 0: simplemc exception in MPI_Bcast: MPI error code: 8
Caught exception during broadcast on process 1: simplemc exception in MPI_Bcast: MPI error code: 8
Process 0 received message: 42
Process 1 received message: 0
```

Here, we wrap the call to simplemc::mpi::broadcast in a `try-catch` block to catch any exceptions
right away.
We then print the error message and proceed with the program.

However, note that the broadcasting did not succeed, so the message variable on rank 1 still has its
initial value of \f$ 0 \f$.
In many cases, it might not make much sense to continue the program after an MPI error has been
encountered.

@subsection tut_mpi_2_glob Catching exceptions globally

We can also try to catch exceptions globally in the `main` function and wrap the whole
simplemc::mpi::environment in a `try-catch` block:

@include tutorials/tut_mpi_2_5.cpp

Output:

```
--------------------------------------------------------------------------
MPI_ABORT was invoked on rank 1 in communicator MPI_COMM_WORLD
  Proc: [[1813,1],1]
  Errorcode: -1

NOTE: invoking MPI_ABORT causes Open MPI to kill all MPI processes.
You may or may not see output from other processes, depending on
exactly when Open MPI kills them.
--------------------------------------------------------------------------
```

This result again in an `MPI_Abort`.
The reason is that by default, the destructor of simplemc::mpi::environment checks if there are any
uncaught exceptions using `std::uncaught_exceptions()` and calls `MPI_Abort` if this is the case.

To change this behaviour, we can set `abort_on_exception` to `false` when constructing the
simplemc::mpi::environment object.
This is a boolean flag that determines how an environment object should be destroyed when an exception
is encountered:
- `abort_on_exception == true` (default): Call `MPI_Abort`.
- `abort_on_exception == false`: Call `MPI_Finalize`.

Let's see how this works in practice:

@include tutorials/tut_mpi_2_6.cpp

Output:

```
Caught exception: simplemc exception in MPI_Bcast: MPI error code: 8
Caught exception: simplemc exception in MPI_Bcast: MPI error code: 8
```

In this case, the program finishes without any error messages from MPI.
The environment object still gets destroyed due to stack unwinding and it still sees an uncaught 
exception in its destructor but since we constructed the environment with 
`abort_on_exception = false`, it simply calls `MPI_Finalize`.
After the MPI environment has been destroyed, the catch block is executed as expected.

@section tut_mpi_2_conclusion Conclusion

We have demonstrated different ways of handling errors in MPI programs using **simplemc-mpi**.
However, there is no one-fits-all solution and the right choice depends on the specific use case.

In general, we would recommend to stick to the default behaviour of aborting the program on errors 
unless one has a good reason not to do so.
Handling errors in an MPI program is difficult and can become its own source of bugs and issues if not 
done carefully.
