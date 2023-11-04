#include <fmt/ranges.h>
#include <simplemc/json.hpp>
#include <simplemc/numeric/eigen.hpp>
#include <range/v3/all.hpp>

int main() {
    Eigen::Vector3d v = Eigen::Vector3d::Random();
    //auto sp = std::span<const double> { &v[0], static_cast<std::size_t>(v.size()) };
    auto sp = simplemc::make_span(v);
    fmt::print("v = {}\n", sp);
    nlohmann::json j;
    simplemc::range_to_json(j, simplemc::make_span(v));
    fmt::print("j = {}\n", j.dump());
}
