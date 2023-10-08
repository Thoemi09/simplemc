#include "nlohmann/json_fwd.hpp"
#include <fmt/ranges.h>
#include <range/v3/range/concepts.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/linear_distribute.hpp>
#include <nlohmann/json.hpp>
#include <span>

struct linspace {
    double start, stop, step;
    int num;
    linspace(double start, double stop, int num) : start(start), stop(stop), num(num) {
        step = (stop - start) / (num - 1);
    }
    double operator()(int i) const { return start + i * step; }
};

void to_json(nlohmann::json& j, ranges::input_range auto&& r) {
    j = nlohmann::json::array();
    for (auto&& x : r) {
        j.push_back(x);
    }
}

int main() {
    auto view = ranges::views::iota(0, 10);
    auto transformed = view | ranges::views::transform(linspace(0, 1, 11));
    fmt::print("{}\n", view);
    fmt::print("{}\n", transformed);
    fmt::print("{}\n", ranges::views::linear_distribute(0., 1., 11));
    nlohmann::json j = view;
    fmt::print("{}\n", j.dump());
    auto vec = ranges::to<std::vector>(transformed);
    j = std::span(vec);
    fmt::print("{}\n", j.dump());
}
