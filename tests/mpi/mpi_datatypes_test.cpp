#include <gtest/gtest.h>

#include <simplemc/mpi.hpp>

#include <array>
#include <list>
#include <span>
#include <vector>

// Custom type for testing MPI datatypes.
struct foo {
    int x;
};

// Test concepts and traits involving MPI datatypes.
TEST(SimplemcMPI, ConceptsAndTraits) {
    static_assert(simplemc::mpi::mpi_compatible<char>);
    static_assert(simplemc::mpi::mpi_compatible<signed char>);
    static_assert(simplemc::mpi::mpi_compatible<unsigned char>);
    static_assert(simplemc::mpi::mpi_compatible<short>);
    static_assert(simplemc::mpi::mpi_compatible<unsigned short>);
    static_assert(simplemc::mpi::mpi_compatible<int>);
    static_assert(simplemc::mpi::mpi_compatible<unsigned int>);
    static_assert(simplemc::mpi::mpi_compatible<long>);
    static_assert(simplemc::mpi::mpi_compatible<unsigned long>);
    static_assert(simplemc::mpi::mpi_compatible<long long>);
    static_assert(simplemc::mpi::mpi_compatible<unsigned long long>);
    static_assert(simplemc::mpi::mpi_compatible<float>);
    static_assert(simplemc::mpi::mpi_compatible<double>);
    static_assert(simplemc::mpi::mpi_compatible<bool>);
    static_assert(simplemc::mpi::mpi_compatible<std::complex<double>>);
    static_assert(!simplemc::mpi::mpi_compatible<std::string>);
    static_assert(!simplemc::mpi::mpi_compatible<foo>);

    static_assert(simplemc::mpi::mpi_range<std::vector<double>>);
    static_assert(!simplemc::mpi::mpi_range<std::vector<foo>>);
    static_assert(simplemc::mpi::mpi_range<std::array<double, 10>>);
    static_assert(!simplemc::mpi::mpi_range<std::array<foo, 10>>);
    static_assert(!simplemc::mpi::mpi_range<std::list<double>>);
    static_assert(simplemc::mpi::mpi_range<std::span<double>>);
    static_assert(!simplemc::mpi::mpi_range<std::span<foo>>);
    static_assert(simplemc::mpi::mpi_range<std::string>);
}
