#include "./test_utils.hpp"
#include "./accs/stochastic_process.hpp"

#include <simplemc/accs/utils.hpp>
#include <simplemc/utils/format.hpp>
#include <simplemc/utils/to_string.hpp>
#include <fmt/ranges.h>
#include <range/v3/all.hpp>

int main() {
    Eigen::MatrixXd m(2, 2);
    m << 1, 2, 3, 4;
    std::cout << m << std::endl;
    Eigen::MatrixXd t = m.triangularView<Eigen::Lower>();
    Eigen::MatrixXd t2 = (m + m).triangularView<Eigen::Lower>();
    std::cout << t2 << std::endl;
}