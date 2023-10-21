#include <fmt/ranges.h>
#include <Eigen/Dense>

int main() {
    Eigen::Vector3d v { 1, 2, 3 };
    fmt::print("v = {}\n", v);
}
