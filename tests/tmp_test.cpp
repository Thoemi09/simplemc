#include <Eigen/Dense>
#include <iostream>

int main() {
    auto vec = Eigen::VectorXd(5);
    vec << 1.0, 2.0, 3.0, 4.0, 5.0;
    vec += vec.matrix();
    std::cout << vec << std::endl;
}
