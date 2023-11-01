#include <fmt/ranges.h>
#include <simplemc/json.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <span>

int main() {
    Eigen::MatrixXd v = Eigen::MatrixXd::Random(3, 3);
    auto sp = std::span<double> { v.data(), static_cast<std::size_t>(v.size()) };
    fmt::print("v = {}\n", sp);
    nlohmann::json j;
    simplemc::range_to_json(j, sp);
    fmt::print("j = {}\n", j.dump());
}
