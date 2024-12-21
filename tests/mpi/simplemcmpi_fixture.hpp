#pragma once

#include <gtest/gtest.h>

#include <simplemc/mpi.hpp>

#include <complex>
#include <string>
#include <vector>

// Fixture class for testing the simplemc-mpi library.
class SimplemcMPI : public ::testing::Test {
protected:
    [[nodiscard]] char char_val() const { return 'x'; }
    [[nodiscard]] signed char schar_val() const { return 'u'; }
    [[nodiscard]] unsigned char uchar_val() const { return 'P'; }
    [[nodiscard]] short short_val() const { return -1234; }
    [[nodiscard]] unsigned short ushort_val() const { return 1234; }
    [[nodiscard]] int int_val() const { return -123456; }
    [[nodiscard]] unsigned int uint_val() const { return 123456; }
    [[nodiscard]] long long_val() const { return -123456789012; }
    [[nodiscard]] unsigned long ulong_val() const { return 123456789012; }
    [[nodiscard]] long long llong_val() const { return -123456789012; }
    [[nodiscard]] unsigned long long ullong_val() const { return 123456789012; }
    [[nodiscard]] float float_val() const { return static_cast<float>(12.3456789); }
    [[nodiscard]] double double_val() const { return 12.3456789012; }
    [[nodiscard]] bool bool_val() const { return true; }
    [[nodiscard]] std::complex<double> cdouble_val() const { return { 1.2345, 6.7890 }; }
    [[nodiscard]] std::string str_val() const { return "my test string"; }

    template <typename T>
    [[nodiscard]] std::vector<T> make_vec(int size, T t) {
        return std::vector<T>(size, t);
    }

    simplemc::mpi::communicator comm;
};
