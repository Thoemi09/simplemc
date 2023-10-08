#include <fmt/ranges.h>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>

struct linspace {
    double start, stop, step;
    int num;
    linspace(double start, double stop, int num) : start(start), stop(stop), num(num) {
        step = (stop - start) / (num - 1);
    }
    double operator()(int i) const { return start + i * step; }
};

int main() {
    auto view = ranges::views::iota(0, 10);
    auto transformed = view | ranges::views::transform(linspace(0, 9, 10));
    fmt::print("{}\n", view);
    fmt::print("{}\n", transformed);
}
