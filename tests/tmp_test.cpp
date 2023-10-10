#include <fmt/core.h>

#include <utility>

struct foo {
    int x = 5;
    foo() { fmt::print("default constructor\n"); };
    foo(const foo& f) : x(f.x) { fmt::print("copy constructor\n"); };
    foo(foo&& f) : x(f.x) { fmt::print("move constructor\n"); };
    foo& operator=(const foo& f) { x = f.x; fmt::print("copy assignment\n"); return *this; };
    foo& operator=(foo&& f) { x = f.x; fmt::print("move assignment\n"); return *this; };
    ~foo() { fmt::print("destructor\n"); };

    void set(const foo& f) { this->operator=(std::move(f)); }
};

foo foo_factory(int x) {
    foo f;
    f.x = x;
    return f;
}

int main() {
    auto f = foo_factory(10);
    foo f2;
    f2.set(std::move(f));
}
