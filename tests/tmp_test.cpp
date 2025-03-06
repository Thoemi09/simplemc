#include <iostream>

struct foo {
    foo(int x = 2) : x(x) { std::cout << "Default constructor\n"; }
    foo(const foo& f) : x(f.x) { std::cout << "Copy constructor\n"; }
    foo(foo&& f) noexcept : x(f.x) { f.x = 0; std::cout << "Move constructor\n"; }
    foo& operator=(const foo& f) { x = f.x; std::cout << "Copy assignment\n"; return *this; }
    foo& operator=(foo&& f) noexcept { x = f.x; f.x = 0; std::cout << "Move assignment\n"; return *this; }
    int x { 2 };
};

template <typename T>
auto make_lambda(T&& t) {
    return [f = std::forward<T>(t)]() { std::cout << f.x << std::endl; };
}

int main() {
    // foo f2{10};
    // auto l = [f = std::move(f2)]() { std::cout << f.x << std::endl; };
    auto l = make_lambda(foo{10});
    l();
    // std::cout << f2.x << std::endl;
}
