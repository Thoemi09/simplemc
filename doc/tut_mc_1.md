@page tut_mc_1 MC Tutorial 1: A minimal MC integration

[TOC]

In this tutorial, we use the modular Monte Carlo framework from @ref simplemc-mc to estimate a
simple 1-dimensional integral. 

We keep everything as minimal as possible: 
- a single update with a uniform proposal that is always accepted, 
- a single measurement that averages the integrand, and 
- the free simplemc::run function driving the sampling loop. 

Error estimation, non-trivial Metropolis sampling, and checkpointing are the topics of the follow-up 
tutorials.

@section tut_mc_1_problem The problem

We want to compute the integral

\f[
  I \;=\; \int_a^b f(x) \, \mathrm{d} x \;, \qquad f(x) = \cos(x) \;, \quad a = 0 \;, \quad b = 2
  \;,
\f]

whose exact value is \f$ I = \sin(2) - \sin(0) \approx 0.9093 \f$. 

Writing the integral as an expectation value over the uniform distribution \f$ U[a, b] \f$,

\f[
  I \;=\; (b - a) \int_a^b f(x) \, \frac{\mathrm{d} x}{b - a} \;=\; (b - a) \, \langle f
  \rangle_{U[a,b]} \approx (b - a) \, \bar{f}  \;,
\f]

turns it into a Monte Carlo problem: draw \f$ N \f$ samples \f$ x_i \sim U[a, b] \f$, calculate the
sample mean \f$ \bar{f} = 1/N \sum_{i=1}^N f(x_i) \f$, and rescale by the volume \f$ (b - a) \f$.

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

An update is any type that satisfies the simplemc::mc_update concept: 
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

A measurement is any type that satisfies the simplemc::mc_measurement concept: a single `measure()`
method:

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

The `measure()` function streams \f$ f(x) \f$ into a simplemc::mean_acc. We use the mean accumulator
here since we are only interested in the sample mean \f$ \bar{f} \f$ and don't care about error
estimation yet.

@subsection tut_mc_1_assembly Assembling the components

A MC simulation is just a set of loose components assembled in the `main` function: the MC 
configuration, a random number generator (e.g. from @ref simplemc-random), a simplemc::update_set, a 
simplemc::measurement_set, and a kernel.

Updates and measurements are registered at construction, each with a unique name. Updates additionally 
carry a selection weight \f$ \geq 0 \f$ (which is only relevant when there are multiple updates):

```cpp
// the MC configuration and the random number generator driving the simulation
mc_config cfg;
simplemc::xoshiro256ss rng { 0xc0ffee };

// register the update with a unique name and a selection weight
simplemc::update_set us { simplemc::update { uniform_update { .cfg = &cfg, .rng = &rng }, "uniform", 1.0 } };

// register the measurement with a unique name
simplemc::measurement_set ms { simplemc::measurement { integral { .cfg = &cfg }, "integral" } };

// the kernel implements the Metropolis algorithm on top of the update set
simplemc::metropolis_kernel kernel { us };
```

Here, we use the provided simplemc::metropolis_kernel which implements the standard Metropolis step: 
- select an update with probability proportional to its weight, 
- call its `attempt()` to propose a new state, and 
- accept or reject the proposal based on the returned acceptance probability.

Since we only have a single update and it is always accepted, the kernel does not do much in this
tutorial.

@subsection tut_mc_1_run Running the simulation

MC simulations are driven by the free simplemc::run function. It performs a nested loop controlled by 
simplemc::simulation_params: each *MC cycle* consists of `steps_per_cycle` kernel steps followed by 
one measurement of all active measurements, and the stopping criteria (`max_steps`, `max_time`) are 
checked every `cycles_per_check` cycles:

```cpp
// run the simulation: measurements are taken once per cycle, so steps_per_cycle = 1 measures
// after every step
const auto ctx = simplemc::run(rng, kernel, ms,
    { .max_steps = 1'000'000, .max_time = 60.0, .steps_per_cycle = 1, .cycles_per_check = 10'000 });
```

Here, we set
- `max_steps = 10^6` and `max_time = 60.0` to either stop after one million MC steps or after 60 
seconds of wall-clock time (whichever comes first),
- `steps_per_cycle = 1` to take one measurement after every MC step, and
- `cycles_per_check = 10^4` to check the stopping criteria only once every ten thousand cycles.

simplemc::run returns a simplemc::simulation_ctx with the number of steps done and the runtime.

@subsection tut_mc_1_result Reading out the result

Finally, we fetch the integral measurement back from the measurement set by its concrete type and 
name, and rescale the sample mean \f$ \bar{f} \f$ by the volume of the integration domain \f$ (b - a) 
\f$ to estimate the integral \f$ I \f$:

```cpp
// fetch the integral measurement by name and rescale the sample mean by the volume to estimate 
// the integral
const auto* m = ms.get<integral>("integral");
const double result = m->acc.mean() * (cfg.b - cfg.a);
const double exact = std::sin(cfg.b) - std::sin(cfg.a);

// print results
fmt::println("runtime: {}", ctx.runtime);
fmt::println("steps:   {}", ctx.steps_done);
fmt::println("result:  {}", result);
fmt::println("exact:   {}", exact);
```

Output:

```
runtime: 0.248824083
steps:   1000000
result:  0.9078870389280093
exact:   0.9092974268256817
```

We can see that the MC simulation took about 0.25 seconds to perform one million MC steps and that
the calculated integral agrees with the exact result to the expected \f$ \mathcal{O}(1/\sqrt{N}) \sim
10^{-3} \f$ accuracy. 

The next tutorial (@ref tut_mc_2) will show you how to implement a more flexible MC simulation by
writing and reading checkpoint files and user input files.

@section tut_mc_1_code Full code

@include tutorials/tut_mc_1.cpp
