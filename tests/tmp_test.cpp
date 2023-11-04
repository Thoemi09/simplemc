#include <fmt/ranges.h>
#include <simplemc/numeric.hpp>
#include <concepts>

template <typename T, std::convertible_to<T>... Ts>
constexpr std::array<T, sizeof...(Ts) + 1> make_array(T t, Ts... ts) {
    return { t, static_cast<T>(ts)... };
}

struct foo {
    int x { 10 };
    operator int() const { return x; }
};

int main() {
    auto arr = make_array(1, 2.0f, 3l, foo());
    fmt::print("v = {}\n", arr);
}
