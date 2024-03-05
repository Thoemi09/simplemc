#include "./test_utils.hpp"
#include "./accs/stochastic_process.hpp"

#include <simplemc/accs/utils.hpp>
#include <simplemc/utils/format.hpp>
#include <simplemc/utils/to_string.hpp>
#include <fmt/ranges.h>
#include <range/v3/all.hpp>

int main() {
    using sp_type = stochastic_process<double, 1>;
    using state_type = sp_type::state_type;
    using mat_type = Eigen::MatrixXd;
    std::vector<state_type> states{ state_type{ 1.0 }, state_type{ 2.0 } };
    std::vector<double> weights{ 0.5, 0.5 };
    mat_type mcmc_props(2, 2);
    mcmc_props << 0.9, 0.4, 0.1, 0.3;
    sp_type sp;
    sp.set_states(states);
    sp.set_weights(weights);
    sp.set_mcmc_proposal(mcmc_props);
    sp.mcmc_warmup(10000);
    sp.mcmc_run(10000);
    fmt::print("Analytic mean: {}\n", analytic_mean(sp));
    fmt::print("Analytic variance: {}\n", analytic_variance(sp));
    fmt::print("Sample mean: {}\n", sample_mean(sp));
    fmt::print("Sample variance: {}\n", sample_variance(sp));
}