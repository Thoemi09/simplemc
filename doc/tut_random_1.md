@page tut_random_1 Random Tutorial 1: Generating random samples

[TOC]

In this tutorial, we show how to generate @ref simplemc-random-samples using the tools provided by
@ref simplemc-random.

@section tut_random_1_details Step-by-step guide

The following code snippets are all part of the same `main` function:

```cpp
#include <fmt/ranges.h>
#include <simplemc/random.hpp>

#include <cmath>
#include <random>
#include <vector>

// histogram definiton goes here

int main() {
    // code snippets go here
}
```

@subsection tut_random_1_hist Histogram

Let's start with defining a simple histogram class that will be used in the following tutorial:

```cpp
// Histogram on the interval [a, b] with uniform bin sizes.
struct histogram {
    // Constructor that takes the interval [a, b] and the number of bins.
    histogram(double a, double b, int nbins) : a(a), b(b), nbins(nbins), step((b - a) / nbins), data(nbins, 0.0) {}

    // Add a value to the histogram.
    void add(double value) {
        nsamples += 1;
        if (value < a || value >= b) {
            return;
        }
        int idx = static_cast<int>((value - a) / step);
        data[idx] += 1.0;
    }

    // Print the histogram + a reference function.
    void print(auto f) const {
        const auto norm = static_cast<double>(nsamples) * step;
        fmt::println("{:<15}{:<15}{:<15}", "x", "h(x)", "f(x)");
        for (int i = 0; i < nbins; ++i) {
            const auto x = a + step * (i + 0.5);
            fmt::println("{:<15.8f}{:<15.8f}{:<15.8f}", x, data[i] / norm, f(x));
        }
    }

    double a;
    double b;
    int nbins;
    double step;
    std::vector<double> data;
    long nsamples { 0 };
};
```

Suppose our histogram is the defined on the domain \f$ [a, b] \f$.
We construct the histogram with the limits \f$ a \f$ and \f$ b \f$ and the number of bins \f$ n \f$
that we want.
Each bin has the same size \f$ \delta = (b - a) / n \f$ and the bin with index \f$ i \f$ is defined as
the interval \f$ \Delta_i = [x_i, x_{i+1}) \f$ where \f$ x_i = a + i \cdot \delta \f$.

In the following, we use histograms to approximate functions \f$ f(x) \f$ at the center of the bins
\f$ y_i = (x_i + x_{i+1}) / 2 \f$:
\f[
  f(y_i) \approx h(i) = \frac{1}{\delta} \int_{\Delta_i} f(x) dx \; ,
\f]
i.e. we take the average of the functions over a bin.

The functions that we want to approximate are either probability density functions (PDF) or discrete
probabilities associated with certain probability distributions.
To do this, we simply generate a large number of random samples from a distribution and put them into
a histogram.
After normalizing the histogram, we expect it to resemble the real PDF of the distribution.

@subsection tut_random_1_setup Basic setup

Producing random samples requires a random number generator (RNG) and a uniform distribution on the
interval \f$ [0, 1) \f$:

```cpp
// construct a random number generator and a uniform distribution on [0, 1)
simplemc::xoshiro256pp rng {};
std::uniform_real_distribution uni01 { 0.0, 1.0 };
```

Here, we use an RNG provided by @ref simplemc-random and the standard library's uniform distribution.
This is just our choice, we could also use an RNG from the standard library instead.

Next we define the number of samples to be generated from each distribution as well as the default
number of bins we use in our histograms:

```cpp
// number of random samples to generate for each distribution and number of bins for the histograms
const int nsamples = 1000000;
int nbins = 10;
```

@subsection tut_random_1_uni25 Uniform distribution

Let's start with approximating our first function, a uniform distribution on the interval \f$ [2, 5)
\f$:

```cpp
// sample uniform distribution on [2, 5) and print histogram
double a = 2;
double b = 5;
histogram hist_uni25 { a, b, nbins };
for (int i = 0; i < nsamples; ++i) {
    hist_uni25.add(simplemc::uniform_sample(uni01(rng), a, b));
}
hist_uni25.print([a, b](auto) { return simplemc::uniform_pdf(a, b); });
```

We first set the limits \f$ a = 2 \f$ and \f$ b = 5 \f$ and pass them to the constructor of the
histogram class.
Then we generate the random samples using simplemc::uniform_sample and add them to the histogram with
its `add` method.
Finally, we print the (normalized) histogram together with a reference function, the exact PDF of the
uniform distribution \f$ f(y_i) = \mathcal{U}(y_i; 2, 5) \f$:

```
y_i            h(i)           f(y_i)         
2.15000000     0.33236000     0.33333333     
2.45000000     0.33268000     0.33333333     
2.75000000     0.33352667     0.33333333     
3.05000000     0.33274000     0.33333333     
3.35000000     0.33335000     0.33333333     
3.65000000     0.33335667     0.33333333     
3.95000000     0.33425667     0.33333333     
4.25000000     0.33286000     0.33333333     
4.55000000     0.33180333     0.33333333     
4.85000000     0.33640000     0.33333333
```

The first column shows the center of the bins at which we evaluate the reference function, the second
column contains the histogram results and the third column the values of the exact PDF.
As you can see, the error of our approximation is \f$ \mathcal{O}(10^3) \f$ and should decrease when
we increase the number of samples.

@subsection tut_random_1_exp25 Exponential distribution

Now that we understand how to generate random samples from a distribution and approximate its PDF with
a histogram, let's try another one, e.g. an exponential distribution restricted to the interval
\f$ [2, 5) \f$:

```cpp
// sample exponential distribution on [2, 5) with lambda = 2.5 and print histogram
double lambda = 2.5;
histogram hist_exp { a, b, nbins };
for (int i = 0; i < nsamples; ++i) {
    hist_exp.add(simplemc::exponential_sample(uni01(rng), lambda, a, b));
}
hist_exp.print([a, b, lambda](auto x) { return simplemc::exponential_pdf(x, lambda, a, b); });
```

Output:

```
y_i            h(i)           f(y_i)         
2.15000000     1.75846667     1.71917405     
2.45000000     0.83121333     0.81208032     
2.75000000     0.39328667     0.38359958     
3.05000000     0.18639667     0.18119961     
3.35000000     0.08716000     0.08559264     
3.65000000     0.04193667     0.04043110     
3.95000000     0.01926333     0.01909830     
4.25000000     0.00903667     0.00902140     
4.55000000     0.00463333     0.00426141     
4.85000000     0.00194000     0.00201295
```

The procedure is exactly the same as before, except that we use
@ref simplemc::exponential_sample(double, double, double, double) "simplemc::exponential_sample" and
@ref simplemc::exponential_pdf(double, double, double, double) "simplemc::exponential_pdf".

Again, our histogram resembles the exact PDF, although the error is bigger than before.
This makes sense since the histogram converges to the average value of the PDF for each bin.
The exponential function is convex so that the average value of a bin will always be larger than the
PDF at its center.
For example, the average value of our first bin is \f$ \approx 1.75975 \f$ which is close to the value
of the histogram there.

@subsection tut_random_1_cauchy Cauchy distribution

We can also approximate functions which are defined on the real line \f$ (-\infty, \infty) \f$, e.g.
the Cauchy distribution:

```cpp
// sample Cauchy distribution with x0 = -2 and gamma = 2.5 and print histogram on [-5, 1]
double x0 = -2;
double gamma = 2.5;
a = -5;
b = 1;
histogram hist_cauchy { a, b, nbins };
for (int i = 0; i < nsamples; ++i) {
    hist_cauchy.add(simplemc::cauchy_sample(uni01(rng), x0, gamma));
}
hist_cauchy.print([x0, gamma](auto x) { return simplemc::cauchy_pdf(x, x0, gamma); });
```

Output:

```
-4.70000000    0.05890667     0.05877214     
-4.10000000    0.07500000     0.07465054     
-3.50000000    0.09381500     0.09362055     
-2.90000000    0.11248833     0.11271597     
-2.30000000    0.12443500     0.12551652     
-1.70000000    0.12505833     0.12551652     
-1.10000000    0.11205500     0.11271597     
-0.50000000    0.09376500     0.09362055     
0.10000000     0.07508667     0.07465054     
0.70000000     0.05875833     0.05877214
```

Of course we need to restrict the histogram to some finite domain, in this case we choose \f$ [-5, 1]
\f$.
The samples are generated with simplemc::cauchy_sample and the exact PDF is given by
simplemc::cauchy_pdf.
The peak is located at \f$ x_0 = -2 \f$ and its width is determined by the parameter \f$ \gamma = 2.5
\f$.

> **Note**: It is crucial that also random samples which fall outside of the domain of our histogram
> are counted (see the definition of the `add` method). Otherwise, the normalization would be off.

@subsection tut_random_1_unidisc Uniform discrete distribution

As a last example, we use a uniform discrete distribution defined on the integer set \f$ D = \{ 2, 3,
4, 6, 7 \} \f$:

```cpp
// sample exclusive uniform integer distribution on {2, ..., 7} excluding y = 5 and print histogram
int a_i = 2;
int b_i = 7;
int y = 5;
nbins = 6;
histogram hist_ex { a_i - 0.5, b_i + 0.5, nbins };
for (int i = 0; i < nsamples; ++i) {
    hist_ex.add(simplemc::exclusive_uniform_int_sample(rng, a_i, b_i, y));
}
hist_ex.print([a_i, b_i, y](auto x) {
    return (std::abs(x - y) < 0.1 ? 0.0 : simplemc::exclusive_uniform_int_pdf(a_i, b_i));
});
```

Output:

```
y_i            h(i)           f(y_i)         
2.00000000     0.20035200     0.20000000     
3.00000000     0.19944000     0.20000000     
4.00000000     0.20004000     0.20000000     
5.00000000     0.00000000     0.00000000     
6.00000000     0.19954700     0.20000000     
7.00000000     0.20062100     0.20000000
```

The number of bins in the histogram is set to 6 and shifted by 0.5 such that the center of each bin
lies on one of the integers \f$ i = 2, \dots, 7 \f$.
The samples are generated with simplemc::exclusive_uniform_int_sample and the exact probabilities are
given by simplemc::exclusive_uniform_int_pdf.
Note that we have to take care of the fact that the distribution is not defined for the excluded
integer \f$ y = 5 \f$ when passing the reference function to the `print` method.

@section tut_random_1_code Full code

@include tutorials/tut_random_1.cpp
