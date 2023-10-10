#include <simplemc/json/range.hpp>

#include <fmt/ranges.h>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/reverse.hpp>

#include <span>

int main() {
    auto view = ranges::views::iota(0, 10);
    nlohmann::json j;
    simplemc::range_to_json(j["view"], view);
    auto vec = ranges::to<std::vector<int>>(view);
    simplemc::range_to_json(j["vec"], vec);
    fmt::print("{}\n", j.dump());
    auto span = std::span(vec);
    simplemc::range_from_json(j["view"], span | ranges::views::reverse);
    fmt::print("{}\n", span);
}
