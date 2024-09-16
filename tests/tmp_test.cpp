#include <Eigen/Dense>
#include <complex>
#include <iostream>

int main() {
    using cplx = std::complex<double>;
    auto vec_c = Eigen::Vector<std::complex<double>, 2>{ cplx{1, 2}, cplx{3, 4} };
    auto mat_d = Eigen::Matrix2d{};
    mat_d << 1, 2, 3, 4;
    std::cout << mat_d << std::endl;
    auto prod = vec_c.real() * vec_c.real().transpose();
    std::cout << prod << std::endl;
    mat_d += (mat_d + vec_c.real() * vec_c.real().transpose()).template triangularView<Eigen::Lower>();
    std::cout << mat_d << std::endl;
}
