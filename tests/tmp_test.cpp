#include <iostream>
#include <vector>
#include <simplemc/grids.hpp>

struct foo {
    foo(int x = 2) : x(x) { std::cout << "Default constructor\n"; }
    foo(const foo& f) : x(f.x) { std::cout << "Copy constructor\n"; }
    foo(foo&& f) noexcept : x(f.x) {
        f.x = 0;
        std::cout << "Move constructor\n";
    }
    foo& operator=(const foo& f) {
        x = f.x;
        std::cout << "Copy assignment\n";
        return *this;
    }
    foo& operator=(foo&& f) noexcept {
        x = f.x;
        f.x = 0;
        std::cout << "Move assignment\n";
        return *this;
    }
    int x { 2 };
};

constexpr auto bar() {
    auto cg = simplemc::custom_grid { std::vector<double>{ 1.0, 2.0, 3.0 } };
    return std::accumulate(cg.begin(), cg.end(), 0.0);
}

int main() { // NOLINT
    constexpr auto sum = bar();
    std::cout << "Sum of grid points: " << sum << "\n";
}
