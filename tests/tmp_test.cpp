#include <exception>
#include <iostream>
#include <stdexcept>

struct Foo {
    char id { '?' };
    int count = std::uncaught_exceptions();

    ~Foo() {
        std::cout << std::uncaught_exceptions() << ", " << count << std::endl;
        count == std::uncaught_exceptions() ? std::cout << id << ".~Foo() called normally\n" :
                                              std::cout << id << ".~Foo() called during stack unwinding\n";
    }
};

int main() {
    Foo f { 'f' };

    try {
        Foo g { 'g' };
        std::cout << "Exception thrown\n";
        throw std::runtime_error("test exception");
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << '\n';
    }
}