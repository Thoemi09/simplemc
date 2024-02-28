#include "./accs/stochastic_process.hpp"

#include <simplemc/utils/format.hpp>
#include <simplemc/utils/to_string.hpp>
#include <fmt/ranges.h>

#include <vector>

int main() {
    using proc_type = stochastic_process<double, 3>;
    // using proc_type = stochastic_process<std::complex<double>, 3>;
    using state_type = typename proc_type::state_type;

    std::vector<double> probs = { 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };

    state_type s1 = { 1.0, 0.0, 0.0 };
    state_type s2 = { 0.0, 1.0, 0.0 };
    state_type s3 = { 0.0, 0.0, 1.0 };

    // state_type s1 = { std::complex<double>{ 1.0, -0.5 }, std::complex<double>{ 0.0, 0.0 }, std::complex<double>{ 0.0, 0.0 } }; 
    // state_type s2 = { std::complex<double>{ 0.0, 0.0 }, std::complex<double>{ -0.3, 0.8 }, std::complex<double>{ 0.0, 0.0 } }; 
    // state_type s3 = { std::complex<double>{ 0.0, 0.0 }, std::complex<double>{ 0.0, 0.0 }, std::complex<double>{ 0.9, 3.4 } };

    std::vector<state_type> states = { s1, s2, s3 };
    auto sp = proc_type(states, probs);

    sp.run(10000);
    fmt::print("Analytic mean: {}\n", analytic_mean(sp));
    fmt::print("Sample mean: {}\n", sample_mean(sp));
    fmt::print("Analytical variance: {}\n", analytic_variance(sp));
    fmt::print("Sample variance: {}\n", sample_variance(sp));
    fmt::print("Analytic covariance:\n{}\n", simplemc::to_string(analytic_covariance(sp)));
    fmt::print("Sample covariance:\n{}\n", simplemc::to_string(sample_covariance(sp)));
}