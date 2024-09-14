#include <Eigen/Dense>
#include <iostream>

int main() {
    auto arr = Eigen::Array2<double>{1.0, 2.0};
    auto mat = Eigen::Vector2d{3.0, 4.0};
    auto res = (arr.array() + mat.array()).eval();
    std::cout << res << std::endl;
}
