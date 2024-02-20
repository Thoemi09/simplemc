// #include "./telegraph_process.hpp"

// #include <simplemc/utils/format.hpp>

// #include <fmt/ranges.h>

// int main() {
//     telegraph_process<std::complex<double>> tp;
//     tp.set_states({ -1.0, 2.3 }, { 1.0, -4.7 });
//     tp.set_probabilities(0.8, 0.1);
//     tp.warm_up(10000);
//     tp.simulate(10000);
//     const auto [re_v, im_v, re_im_cov] = tp.variance();
//     const auto [re_v_est, im_v_est, re_im_cov_est] = sample_variance(tp.samples);
//     fmt::print("mean = {}, estimated mean = {}\n", tp.mean(), sample_mean(tp.samples));
//     fmt::print("real var = {}, estimated real var = {}\n", re_v, re_v_est);
//     fmt::print("imag var = {}, estimated imag var = {}\n", im_v, im_v_est);
//     fmt::print("covar = {}, estimated covar = {}\n", re_im_cov, re_im_cov_est);
//     return 0;
// }