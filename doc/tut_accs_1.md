@page tut_accs_1 Accumulator Tutorial 1: Estimating errors

[TOC]

In this tutorial, we use the accumulators provided by @ref simplemc-accs to estimate the mean of
various data **together with its statistical error**.

We start with the easy case of independent and identically distributed (i.i.d.) samples, where
simplemc::mean_acc and simplemc::var_acc are all we need. We then switch to a correlated time series,
where the naive variance estimator dramatically underestimates the true error of the mean, and
demonstrate how the @ref simplemc-accs-wrappers — simplemc::block_acc and simplemc::autocorr_acc —
together with simplemc::batch_acc recover the correct error bar.

@section tut_accs_1_details Step-by-step guide

The following code snippets are all part of the same `main` function:

```cpp
#include <fmt/base.h>
#include <simplemc/accs.hpp>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

int main() {
    // code snippets go here
}
```

@subsection tut_accs_1_iid The i.i.d. baseline

Before we look at the accumulators, let's pin down what they are *supposed* to compute. Given \f$ N
\f$ i.i.d. samples \f$ S_X = \{ x_i : i = 1, \dots, N \} \f$ from some distribution with unknown mean
\f$ \mu_X \f$ and unknown variance \f$ \sigma^2_X \f$, the textbook estimators are

\f[
  \bar{x} \;=\; \frac{1}{N} \sum_{i=1}^{N} x_i
  \qquad \text{(sample mean)} \;,
\f]
\f[
  s_X^2 \;=\; \frac{1}{N - 1} \sum_{i=1}^{N} \left( x_i - \bar{x} \right)^2
  \qquad \text{(unbiased sample variance)} \;,
\f]
\f[
  s_{\bar{X}} \;=\; \sqrt{ \frac{s_X^2}{N} } \;=\; \frac{s_X}{\sqrt{N}}
  \qquad \text{(standard error of the mean)} \;.
\f]

The divisor \f$ N - 1 \f$ in \f$ s_X^2 \f$ — Bessel's correction — is what makes the estimator
*unbiased*: the data is used twice (once to fix \f$ \bar{x} \f$, once to measure deviations from it),
which removes one degree of freedom. If \f$ \sigma^2_X \f$ were known, the standard error of the mean
\f$ \sigma_{\bar{X}} \f$ would simplify to the famous data-independent expression \f$ \sigma_{\bar{X}}
= \sigma_X / \sqrt{N} \f$. In the following, we verify that the estimators from @ref simplemc-accs 
reproduce these known results.

Let's start by generating the data for \f$ S_X \f$. We first draw \f$ N = 10^5 \f$ samples from the 
normal distribution \f$ \mathcal{N}(\mu_X = 5, \sigma^2_X = 1) \f$:

```cpp
// Part 1: i.i.d. samples from a normal distribution N(mu_x, sigma_x^2)
const double mu_x = 5.0;
const double sigma2_x = 1.0;
const long n_x = 100000;
const auto n_xd = static_cast<double>(n_x);
std::mt19937_64 rng;
std::normal_distribution<double> iid_dist(mu_x, sigma2_x);

// draw samples from the distribution
std::vector<double> samples_x(n_x);
for (long i = 0; i < n_x; ++i) {
    samples_x[i] = iid_dist(rng);
}
```

Then we compute the three estimators directly with a plain two-pass loop. The first pass computes the 
sum \f$ \sum_{i=1}^{N} x_i \f$ to obtain \f$ \bar{x} \f$, the second pass computes \f$ \sum_{i=1}^{N} 
\left( x_i - \bar{x} \right)^2 \f$ to get \f$ s_X^2 \f$ and \f$ s_{\bar{X}} \f$:

```cpp
// manual two-pass reference: first compute the sample mean
double sum = 0.0;
for (double x : samples_x) {
    sum += x;
}
const double x_bar_ref = sum / n_xd;

// then compute the standard error of the mean
double sum_sq = 0.0;
for (double x : samples_x) {
    const double d = x - x_bar_ref;
    sum_sq += d * d;
}
const double s_x_bar_ref = std::sqrt(sum_sq / (n_xd - 1) / n_xd);
```

> **Note**: this textbook two-pass algorithm is exactly what the equations above say. It is
> numerically fine for moderate \f$ N \f$ and well-conditioned data, but the difference
> \f$ x_i - \bar x \f$ can suffer from catastrophic cancellation when the samples cluster far
> from the origin relative to their spread. To handle that case, simplemc::var_acc uses a
> Welford-style update internally; the algorithm is selected by the simplemc::varalg
> template parameter (see @ref simplemc-accs-utils), with `simplemc::varalg::welford` as the
> default.

Now let's use the accumulators from @ref simplemc-accs. The simplest and fastest accumulator is
simplemc::mean_acc. It only tracks the running mean of some data without any variance/error 
estimation:

```cpp
// accumulate the same data with a mean accumulator
simplemc::mean_acc<double> macc;
for (double x : samples_x) {
    macc << x;
}
```

Data is accumulated by streaming it into the simplemc::mean_acc object with `operator<<`. This is true
for all accumulators in @ref simplemc-accs.

To also get a variance estimate and not just the mean, we have to use simplemc::var_acc:

```cpp
// accumulate the same data with a variance accumulator
simplemc::var_acc<double> vacc;
for (double x : samples_x) {
    vacc << x;
}
```

The variance accumulator exposes two related variance quantities, and it pays to keep them apart:

- @ref simplemc::var_acc< X, A >::variance_of_data returns the sample variance \f$ s_X^2 \f$.
- @ref simplemc::var_acc< X, A >::variance returns the sample variance of the mean
\f$ s_{\bar{X}}^2 = s_X^2 / N \f$. To get the standard error \f$ s_{\bar{X}} \f$, we can either take
the square root manually or make use of simplemc::stderror.

Printing the analytic known results together with the manual reference and both accumulators side by 
side gives:

```cpp
// print results side by side: mean and standard error of the mean
fmt::println("analytic:   mu_x  = {:<8.1f}, sigma_x_bar = {:.6f}", mu_x, sigma2_x / std::sqrt(n_xd));
fmt::println("reference:  x_bar = {:<8.6f}, s_x_bar     = {:.6f}", x_bar_ref, s_x_bar_ref);
fmt::println("mean_acc:   x_bar = {:<8.6f}, s_x_bar     = N/A", macc.mean());
fmt::println("var_acc:    x_bar = {:<8.6f}, s_x_bar     = {:.6f}", vacc.mean(), simplemc::stderror(vacc));
fmt::println("");
```

Output:

```
analytic:   mu_x  = 5.0     , sigma_x_bar = 0.003162
reference:  x_bar = 5.002072, s_x_bar     = 0.003168
mean_acc:   x_bar = 5.002072, s_x_bar     = N/A
var_acc:    x_bar = 5.002072, s_x_bar     = 0.003168
```

The accumulator results agree on \f$ \bar{x} \f$ and \f$ s_{\bar{X}} \f$ with the reference data to
high accuracy. The standard error of the mean \f$ s_{\bar{X}} \approx 0.003168 \f$ sits within \f$ 
0.2\% \f$ of the data-independent analytically known value \f$ \sigma_{\bar{X}} = \sigma_X / \sqrt{N}
\approx 0.0031623 \f$ — the small residual is just the random fluctuation of the sample variance
around \f$ \sigma^2_X \f$.

So the takeaway from the i.i.d. case is that simplemc::var_acc gives the textbook answer in a single
online pass, without storing the samples and without numerical loss. The next sections look at the
regime where the textbook formula and simplemc::var_acc no longer agree with the truth: correlated
samples.

@subsection tut_accs_1_ar1 A correlated time series: the AR(1) process

Monte Carlo simulations rarely produce i.i.d. samples — the output of a Markov chain is correlated
by construction. To study this regime in a controlled way, we use a first-order autoregressive
process - AR(1) - defined by the recursion,
\f[
  y_{t+1} = \phi \, y_t + \varepsilon_t \;, \qquad \varepsilon_t \sim \mathcal{N}(0,
  \sigma_\varepsilon^2) \;,
\f]
with \f$ \phi = 0.9 \f$, \f$ \sigma_\varepsilon = 1 \f$, and \f$ N = 10^6 \f$ steps. The AR(1) is
useful because its stationary statistics are known in closed form,
\f[
  \langle y \rangle = \mu_y 0 \;, \qquad
  \sigma^2_Y = \frac{\sigma_\varepsilon^2}{1 - \phi^2} \;, \qquad
  \tau_{\mathrm{int}} = \frac{1 + \phi}{1 - \phi} = 19 \;,
\f]
so the true standard error of the sample mean is
\f[
  \sigma_{\bar{Y}} = \sqrt{\sigma^2_Y \cdot \tau_{\mathrm{int}} / N} \approx 0.01
\f]

Before we start, let's set the process parameters and print the analytic reference results:

```cpp
// parameters of the AR(1) process
const double mu_y = 0.0;
const double phi = 0.9;
const double sigma2_eps = 1.0;
const long n_y = 1000000;
const auto n_yd = static_cast<double>(n_y);
std::normal_distribution<double> noise(mu_y, sigma2_eps);

// analytic reference results for the AR(1) process
const double sigma2_y = sigma2_eps * sigma2_eps / (1.0 - phi * phi);
const double tau_int = (1.0 + phi) / (1.0 - phi);
const double sigma_y_bar = std::sqrt(sigma2_y * tau_int / n_yd);
fmt::println("AR(1) analytic:");
fmt::println("  y_bar       = {:.1f}", mu_y);
fmt::println("  tau_int     = {:.1f}", tau_int);
fmt::println("  sigma_x_bar = {:.2f}", sigma_y_bar);
fmt::println("");
```

Output:

```
AR(1) analytic:
  y_bar       = 0.0
  tau_int     = 19.0
  sigma_x_bar = 0.01
```

Then we generate the samples from the AR(1) process:

```cpp
// generate samples from the AR(1) process
std::vector<double> samples_y(n_y);
double y_t = 0.0;
for (long i = 0; i < n_y; ++i) {
    y_t = phi * y_t + noise(rng);
    samples_y[i] = y_t;
}
```

@subsection tut_accs_1_naive Why naive var_acc fails on correlated data

Let's see what simplemc::var_acc thinks \f$ \sigma_{\bar{Y}} \f$ is when fed the AR(1) chain:

```cpp
// 1) naive var_acc with correlated samples
simplemc::var_acc<double> vacc_y;
for (double y : samples_y) {
    vacc_y << y;

}
fmt::println("AR(1) var_acc:");
fmt::println("  y_bar   = {:.6f}", vacc_y.mean());
fmt::println("  s_y_bar = {:.6f}", simplemc::stderror(vacc_y));
fmt::println("");
```

Output:

```
AR(1) var_acc:
  y_bar   = 0.000961
  s_y_bar = 0.002290
```

The mean is close to the expected value 0, but the reported \f$ \sigma_{\bar{Y}} \approx 0.0023 \f$ is
more than a factor of four smaller than the analytic \f$ 0.01 \f$. The reason is that
simplemc::var_acc assumes i.i.d. samples: it divides the sample variance of the data by \f$ N \f$,
which is the effective sample size only when consecutive samples are independent. Correlated samples
carry less information, so the effective sample size is \f$ N_{\mathrm{eff}} = N / \tau_{\mathrm{int}}
\f$, and the naive estimate underestimates \f$ \sigma_{\bar{Y}} \f$ by a factor of \f$
\sqrt{\tau_{\mathrm{int}}} \approx \sqrt{19} \approx 4.4 \f$ — exactly what we see.

@subsection tut_accs_1_block Fix 1: simplemc::block_acc

The first remedy is to group consecutive samples into blocks of size \f$ B \f$. If
\f$ B \gg \tau_{\mathrm{int}} \f$, the block means are approximately independent, and the wrapped
simplemc::var_acc sees \f$ N_{\mathrm{eff}} = N / B \f$ effective samples. simplemc::block_acc
does this with no extra storage:

```cpp
// 2) block_acc with a fixed block size B = 100
simplemc::block_acc<simplemc::var_acc<double>> blkacc_y { 100 };
for (double y : samples_y) {
    blkacc_y << y;
}

fmt::println("AR(1) block_acc:");
fmt::println("  y_bar   = {:.6f}", blkacc_y.mean());
fmt::println("  s_y_bar = {:.6f}", simplemc::stderror(blkacc_y));
fmt::println("");
```

Output:

```
AR(1) block_acc:
  y_bar   = 0.000961
  s_y_bar = 0.009558
```

The mean is unchanged, but \f$ \sigma_{\bar{X}} \f$ jumped from \f$ 0.0023 \f$ to \f$ 0.0096 \f$ —
within a few percent of the analytic \f$ 0.01 \f$. Picking a good \f$ B \f$ requires knowing
\f$ \tau_{\mathrm{int}} \f$ in advance, which is exactly the problem the next accumulator solves.

@subsection tut_accs_1_autocorr Fix 2: simplemc::autocorr_acc

simplemc::autocorr_acc maintains a hierarchy of block accumulators with geometrically increasing
block sizes \f$ B_l = 2^l \f$. It lets us inspect \f$ \sigma_{\bar{Y}} \f$ as a function of the block
size and estimate the autocorrelation time \f$ \tau \f$ at every level via simplemc::accs::tau,
defined as
\f[
  \tau = \frac{1}{2} \! \left( B_l \, \frac{s_{B_l}^2}{s_0^2} - 1 \right) \; .
\f]
Here, \f$ s_{B_l}^2 \f$ is the variance of the block means at level \f$ l \f$ and \f$ s_0^2 \f$ is the
naive variance of the data.

The \f$ \tau \f$ returned by simplemc::accs::tau is related to \f$ \tau_{\mathrm{int}} \f$ via \f$ 
\tau_{\mathrm{int}} = 1 + 2 \tau \f$. For the AR(1) example, \f$ \tau_{\mathrm{int}} = 19 \f$ 
corresponds to \f$ \tau = 9 \f$.

Let's use simplemc::autocorr_acc to scan different block sizes and see the plateau emerge:

```cpp
// 3) autocorr_acc to scan a geometric hierarchy of block sizes
simplemc::autocorr_acc<simplemc::var_acc<double>> aacc_y;
for (double y : samples_y) {
    aacc_y << y;
}

// get the naive variance estimate and the highest level with at least 256 effective samples
const auto s0 = aacc_y.accumulators()[0].variance_of_data();
const auto max_level = aacc_y.find_level(256);

// loop over levels and print the block size, effective sample size, standard error of the mean,
// and tau estimate
fmt::println("AR(1) autocorr_acc:");
fmt::println("{:<8}{:<14}{:<14}{:<16}{:<14}", "l", "B_l", "N_eff", "s_y_bar", "tau");
for (std::size_t l = 0; l <= max_level; ++l) {
    const auto& va = aacc_y.accumulators()[l];
    fmt::println("{:<8}{:<14}{:<14}{:<16.8f}{:<14.4f}", l, aacc_y.block_size(l), va.count(),simplemc::stderror(va),
        simplemc::accs::tau(s0, va.variance_of_data(), aacc_y.block_size(l)));
}
fmt::println("");
```

Here, we call simplemc::autocorr_acc::find_level to cap the table at the largest block size that still
has at least 256 effective samples — beyond that, the estimates become too noisy.

Output:

```
AR(1) autocorr_acc:
l       B_l           N_eff         s_y_bar         tau           
0       1             1000000       0.00229008      0.0000        
1       2             500000        0.00315622      0.4497        
2       4             250000        0.00429782      1.2610        
3       8             125000        0.00569536      2.5925        
4       16            62500         0.00718380      4.4201        
5       32            31250         0.00844036      6.2919        
6       64            15625         0.00920803      7.5835        
7       128           7812          0.00964894      8.3756        
8       256           3906          0.00988409      8.8135        
9       512           1953          0.01012303      9.2692        
10      1024          976           0.01024417      9.4993        
11      2048          488           0.01019317      9.4000
```

Reading the table:
- Level 0 reproduces the naive estimate (no blocking): \f$ s_{\bar{Y}} = 0.00229 \f$ and \f$ \tau = 0 
\f$ by construction.
- As the block size grows, both \f$ s_{\bar{Y}} \f$ and \f$ \tau \f$ climb and **plateau** once \f$ 
B_l \gtrsim \tau_{\mathrm{int}} = 19 \f$. The plateau of the \f$ s_{\bar{Y}} \f$ column near \f$ 
0.010 \f$ is in good agreement with the analytic value.
- The \f$ \tau \f$ plateau near \f$ 9.5 \approx (\tau_{\mathrm{int}} - 1)/2 \f$ is the expected value 
of \f$ \tau \f$ for an AR(1) with \f$ \phi = 0.9 \f$.

The plateau is the signal that further blocking is no longer needed and that the blocks have become 
effectively independent.

@subsection tut_accs_1_batch Fix 3: simplemc::batch_acc

simplemc::block_acc and simplemc::autocorr_acc both require choosing or scanning a block size.
simplemc::batch_acc instead groups samples into a fixed number of batches. The batches can then be 
used as effective samples to perform further statistical analysis like @ref simplemc-accs-resampling. 
This makes it a good compromise between the fast and simple simplemc::block_acc and the more robust 
but more expensive simplemc::autocorr_acc. 

Let's see what it gives for the AR(1) chain:

```cpp
// 4) batch_acc to group samples into a fixed number of batches (default: min 256, max 512)
simplemc::batch_acc<double> bacc_y {};
for (double y : samples_y) {
    bacc_y << y;
}

fmt::println("AR(1) batch_acc:");
fmt::println("  y_bar   = {:.6f}", bacc_y.mean());
fmt::println("  s_y_bar = {:.6f}", simplemc::stderror(bacc_y));
fmt::println("  samples/batch = {}, N_eff = {}", bacc_y.batch_count(), bacc_y.num_full_batches());
```

Output:

```
AR(1) batch_acc:
  y_bar   = 0.000961
  s_y_bar = 0.010193
  samples/batch = 2048, N_eff = 488
```

simplemc::batch_acc settled on a batch size of \f$ 2048 \f$ samples (488 full batches) and reports 
\f$ \sigma_X \approx 0.0102 \f$, again consistent with the analytic value.

@subsection tut_accs_1_summary Comparison

Putting it together, the same AR(1) chain produces:

| Estimator                            | \f$ s_{\bar{Y}}/\sigma_{\bar{Y}} \f$ |
|--------------------------------------|------------------|
| Naive simplemc::var_acc              | 0.00229          |
| simplemc::block_acc                  | 0.00956          |
| simplemc::autocorr_acc               | ≈ 0.01           |
| simplemc::batch_acc                  | 0.01019          |
| Analytic \f$ \sqrt{\sigma^2_Y \, \tau_{\mathrm{int}} / N} \f$ | 0.01000 |

The takeaway: when the samples come from a correlated process — i.e. essentially always in a
real Monte Carlo simulation — pair simplemc::var_acc with a wrapper that handles the
correlations. simplemc::block_acc is the cheapest option when a safe block size is known;
simplemc::autocorr_acc is the right tool when it is not; simplemc::batch_acc is a convenient
single-pass alternative and in most cases the right balance between the two.

@section tut_accs_1_code Full code

@include tutorials/tut_accs_1.cpp
