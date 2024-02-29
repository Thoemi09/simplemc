#include "./test_utils.hpp"
#include "./accs/stochastic_process.hpp"

#include <simplemc/accs/utils.hpp>
#include <simplemc/utils/format.hpp>
#include <simplemc/utils/to_string.hpp>
#include <fmt/ranges.h>

#include <vector>

int main() {
    Eigen::Matrix3d m;
    m << 1, 2, 3,
         0, 5, 6,
         0, 0, 9;
    std::cout << m << std::endl;
    Eigen::Matrix3d m2 = m.selfadjointView<Eigen::Upper>();
    std::cout << m2 << std::endl;
}