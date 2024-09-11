#include <Eigen/Dense>
#include <iostream>

int main() {
    auto tmp = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(3);
    std::cout << tmp.RowsAtCompileTime << " " << tmp.ColsAtCompileTime << std::endl;
}
