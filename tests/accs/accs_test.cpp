/**
 * @file accs_test.cpp
 * @brief Unit tests for simplemc-accs.
 */

#include "../test_utils.hpp"

#include <simplemc/accs.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

// Test traits class.
TEST(SimplemcAccs, Traits) {
    using cd_type = typename simplemc::acc_traits<std::complex<double>>::storage_type;
    cd_type data = simplemc::acc_traits<std::complex<double>>::storage_type::Constant(5, std::complex<double> { 1.0, 1.0 });
    for (auto i = 0; i < data.size(); ++i) {
        data[i] = std::complex<double> { i * 1.0, i * 1.0 };
    }
    auto mean = simplemc::acc_traits<std::complex<double>>::mean(data, 5);
    auto mean_nan = simplemc::acc_traits<std::complex<double>>::mean(data, 0);
    for (auto i = 0; i < mean.size(); ++i) {
        check_complex_near(mean[i], std::complex<double> { i / 5.0, i / 5.0 });
        ASSERT_TRUE(std::isnan(mean_nan[i].real()));
        ASSERT_TRUE(std::isnan(mean_nan[i].imag()));
    }
}
