@page tut_mc_1 MC Tutorial 1: A minimal MC integration

[TOC]

In this tutorial, we use the modular Monte Carlo framework from @ref simplemc-mc to estimate a
simple 1-dimensional integral. 

We keep everything as minimal as possible: 
- a single update with a uniform proposal that is always accepted, 
- a single measurement that averages the integrand, and 
- no callbacks, no checkpointing, no user input, no MPI parallelization.

Follow-up tutorials will extend this example to show how to implement more complex MC simulations.

@section tut_mc_1_problem The problem

We want to compute the integral

\f[
  I \;=\; \int_a^b f(x) \, \mathrm{d} x \;, \qquad f(x) = \cos(x) \;, \quad a = 0 \;, \quad b = 2 \;,
\f]

whose exact value is \f$ I = \sin(2) - \sin(0) \approx 0.9093 \f$. 

Writing the integral as an expectation value over the uniform distribution \f$ U[a, b] \f$,

\f[
  I \;=\; (b - a) \int_a^b f(x) \, \frac{\mathrm{d} x}{b - a} \;=\; (b - a) \, \langle f
  \rangle_{U[a,b]} \approx (b - a) \, \bar{f} = I_{MC} \;,
\f]

lets us approximate \f$ I \f$ using Monte Carlo integration:

- draw \f$ N \f$ samples \f$ x_i \sim U[a, b] \f$, 
- calculate the sample mean \f$ \bar{f} = 1/N \sum_{i=1}^N f(x_i) \f$, and 
- rescale by the volume \f$ (b - a) \f$ to obtain \f$ I_{MC} \f$.

@section tut_mc_1_details Step-by-step guide

The user-defined types and other helper functions live in an anonymous namespace above `main`:

```cpp
#include <fmt/base.h>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>

#include <cmath>
#include <random>

namespace {
    // user types go here
} // namespace

int main() {
    // code snippets go here
}
```

@subsection tut_mc_1_integrand The integrand

We first define the integrand \f$ f(x) = \cos(x) \f$ as a free function:

```cpp
// The integrand.
double f(double x) {
    return std::cos(x);
}
```

@subsection tut_mc_1_config The MC configuration

The state of the Markov chain is the current sample \f$ x \f$. We bundle it with the integration
domain \f$ [a, b] \f$ into a plain struct that is shared between updates and measurements:

```cpp
// MC configuration shared between the update and the measurement:
// - x: the current sample,
// - [a, b]: the integration domain.
struct mc_config {
    double x = 1.0;
    static constexpr double a = 0.0;
    static constexpr double b = 2.0;
};
```

@subsection tut_mc_1_update The update

An MC update is any type that satisfies the simplemc::mc_update concept. It must provide

- an `attempt()` method that proposes a new configuration and returns its acceptance probability, and 
- an `accept()` method that commits the proposal. 

An optional `reject()` is called when a proposal is turned down — we do not need one here, because a 
uniform proposal for a uniform target distribution is always accepted:

```cpp
// Uniform-resampling update:
// - attempt(): proposes a new x uniformly in [a, b] and always returns 1.0.
// - accept(): accepts the proposal by updating the current sample x.
struct uniform_update {
    mc_config* cfg;
    simplemc::xoshiro256ss* rng;
    std::uniform_real_distribution<double> uni01 { 0.0, 1.0 };
    double new_x = 0.0;

    double attempt() {
        new_x = cfg->a + (cfg->b - cfg->a) * uni01(*rng);
        return 1.0;
    }
    void accept() { cfg->x = new_x; }
};
```

Note that the update stores a pointer to the MC configuration and the RNG.

The `attempt()` method draws a new sample \f$ x' \sim U[a, b] \f$ and returns the acceptance 
probability \f$ 1.0 \f$ to indicate that the proposed state should always be accepted.

The `accept()` method commits the proposed state by updating the current sample \f$ x \leftarrow x' 
\f$.

@subsection tut_mc_1_measurement The measurement

A measurement is any type that satisfies the simplemc::mc_measurement concept. It must provide

- a `measure()` method that performs some specific measurement(s):

```cpp
// Integral measurement: I = \int_0^2 f(x) dx
// - acc: accumulator for the sample mean of f(x), i.e. <f> \approx 1/N \sum_{i=1}^N f(x_i).
// - measure(): evaluates f(x) and accumulates the result.
struct integral {
    const mc_config* cfg;
    simplemc::mean_acc<double> acc {};

    void measure() { acc << f(cfg->x); }
};
```

Note that the measurement stores a (const) pointer to the MC configuration.

In our case, the `measure()` function streams \f$ f(x_i) \f$ into a simplemc::mean_acc. We use the 
mean accumulator here since we are only interested in the sample mean \f$ \bar{f} \f$ and don't care 
about error estimation yet.

@subsection tut_mc_1_assembly Assembling the components

In **simplemc-mc**, a MC simulation consists of a set of loose components that are passed to 
simplemc::run which drives the simulation.

The driver expects a random number generator, (e.g. from @ref simplemc-random), a simplemc::mc_kernel 
that implements the actual MC algorithm, a simplemc::measurement_set, and simplemc::simulation_params 
to control the simulation.

The MC configuration is an implementation detail that is usually shared between the updates and
measurements. simplemc::run does not need to know about it.

Also note that the simplemc::update_set is not passed to simplemc::run directly, but is instead given
to the kernel which is responsible for selecting and executing updates.

We can now assemble the components for our MC simulation in `main()`:

```cpp
// random number generator
simplemc::xoshiro256ss rng;

// MC configuration
mc_config cfg;

// construct the update set with our uniform update
simplemc::update_set us { simplemc::update { uniform_update { .cfg = &cfg, .rng = &rng }, "uniform", 1.0 } };

// construct the measurement set with our integral measurement
simplemc::measurement_set ms { simplemc::measurement { integral { .cfg = &cfg }, "integral" } };

// construct the Metropolis kernel and give it access to the update set
simplemc::metropolis_kernel kernel { us };
```

Updates and measurements are specified at construction, each with a unique name. Updates additionally 
carry a selection weight \f$ \geq 0 \f$ (which is only relevant when there are multiple updates).

Here, we use the provided simplemc::metropolis_kernel which implements the standard Metropolis 
algorithm:

- select an update with probability proportional to its weight, 
- call its `attempt()` to propose a new state, and 
- accept or reject the proposal based on the returned acceptance probability.

Since we only have a single update and it is always accepted, the kernel does not do much in this
tutorial.

@subsection tut_mc_1_par Controlling the simulation

As already mentioned above, MC simulations are driven by the free simplemc::run function. 

It performs a nested loop controlled by simplemc::simulation_params: each *MC cycle* consists of 
`steps_per_cycle` kernel steps followed by one measurement of all active measurements. The stopping 
criteria (`max_steps`, `max_time`) are checked every `cycles_per_check` cycles:

```cpp
// set the simulation parameters
simplemc::simulation_params params {
    .max_steps = 1'000'000, .max_time = 60.0, .steps_per_cycle = 1, .cycles_per_check = 10'000
};
```

For our simulation, we set

- `max_steps = 10^6` and `max_time = 60.0` to either stop after one million MC steps or after 60 
seconds of wall-clock time (whichever comes first),
- `steps_per_cycle = 1` to take one measurement after every MC step, and
- `cycles_per_check = 10^4` to check the stopping criteria only once every ten thousand cycles.

@subsection tut_mc_1_run Running the simulation

To run the simulation, we simply call simplemc::run with the assembled components:

```cpp
// run the simulation
fmt::println("Running the simulation with the following parameters:");
simplemc::print(params);

simplemc::simulation_stats stats;
stats += simplemc::run(rng, kernel, ms, params);

fmt::println("\nSimulation finished. Final statistics:");
simplemc::print(stats);
```

simplemc::run returns a simplemc::simulation_ctx with the number of steps done and the runtime of 
this run. We fold it into the cumulative simplemc::simulation_stats with `operator+=()` — with a 
single run this makes no difference, but it is the idiom that keeps the total step count and runtime 
correct across multiple runs (e.g. when restarting from a checkpoint, see @ref tut_mc_2).

The free `print` overloads render the parameters and the statistics as human-readable blocks:

```
Running the simulation with the following parameters:
Max. steps              = 1000000
Max. time               = 60 sec
Steps per cycle         = 1
Cycles per check        = 10000
Skip measurements       = false
Checkpoint after steps  = 18446744073709551615
Checkpoint after time   = 1.7976931348623157e+308 sec

Simulation finished. Final statistics:
Runtime        = 0.268473125 sec
MC steps done  = 1000000
Steps per sec  = 3.72477e+06
```

On our machine, the simulation took about 0.27 seconds to perform one million MC steps, i.e. it
stopped after reaching simplemc::simulation_params::max_steps.

The simplemc::simulation_params::max_time limit was not reached.

@subsection tut_mc_1_result Reading out the result

Finally, we fetch the integral measurement back from the measurement set by its concrete type and 
name, and rescale the sample mean \f$ \bar{f} \f$ by the volume of the integration domain \f$ (b - a) 
\f$ to estimate the integral \f$ I \f$:

```cpp
// fetch the integral measurement by name and rescale \bar{f} by (b - a) to estimate I_MC
const auto* m = ms.get<integral>("integral");
const double I_MC = m->acc.mean() * (cfg.b - cfg.a);

// print results and compare with the exact value
fmt::println("\nResults:");
fmt::println("I_MC = {}", I_MC);
fmt::println("I    = {}", std::sin(cfg.b) - std::sin(cfg.a));
```

Output:

```
Results:
I_MC = 0.9092568281756135
I    = 0.9092974268256817
```

We can see that the calculated integral agrees with the exact result to the expected \f$ 
\mathcal{O}(1/\sqrt{N}) \sim 10^{-3} \f$ accuracy. 

The next tutorial (@ref tut_mc_2) will show you how to implement a more flexible MC simulation by
writing and reading checkpoint and user input files.

@section tut_mc_1_code Full code

@include tutorials/tut_mc_1.cpp
