/**
 * @file datatypes.hpp
 * @brief Mappings of MPI datatypes to C++ types.
 */

#ifndef SIMPLEMC_MPI_DATATYPES_HPP
#define SIMPLEMC_MPI_DATATYPES_HPP

#include <mpi.h>
#include <range/v3/range/concepts.hpp>

#include <complex>
#include <cstddef>
#include <type_traits>

namespace simplemc {

namespace mpi {

/**
 * @brief Determine if there is a mapping between the C++ type T and a MPI datatype.
 *
 * @details Types which have a corresponding MPI datatype specialize this struct and 
 * inherit from std::true_type. All other type inherit from std::false_type.
 *
 * @tparam T C++ type.
 */
template <typename T>
struct is_mpi_datatype : public std::false_type {};

/**
 * @brief Alias template for is_mpi_datatype<T>::value.
 *
 * @tparam T C++ type.
 */
template <typename T>
inline constexpr bool is_mpi_datatype_v = is_mpi_datatype<T>::value;

/**
 * @brief Map C++ type T to MPI datatypes.
 *
 * @details For not supported C++ types, this struct will be empty. All others provide a 
 * specialization with a static method get(), which return the corresponding MPI datatype.
 *
 * @tparam T C++ type
 */
template <typename T>
struct mpi_type {};

// Define specializations for is_mpi_datatype and mpi_type.
#define MAKE_MPI_DATATYPE(__cxx_type__, __mpi_type__) \
    template <> \
    struct mpi_type<__cxx_type__> { \
        static MPI_Datatype get() { return __mpi_type__; } \
    }; \
    template <> \
    struct is_mpi_datatype<__cxx_type__> : public std::true_type {};

// character type
MAKE_MPI_DATATYPE(char, MPI_CHAR)

// byte type
MAKE_MPI_DATATYPE(std::byte, MPI_BYTE)

// signed integer types
MAKE_MPI_DATATYPE(short, MPI_SHORT)
MAKE_MPI_DATATYPE(int, MPI_INT)
MAKE_MPI_DATATYPE(long, MPI_LONG)
MAKE_MPI_DATATYPE(long long, MPI_LONG_LONG)
MAKE_MPI_DATATYPE(signed char, MPI_SIGNED_CHAR)

// unsigned integer types
MAKE_MPI_DATATYPE(unsigned char, MPI_UNSIGNED_CHAR)
MAKE_MPI_DATATYPE(unsigned short, MPI_UNSIGNED_SHORT)
MAKE_MPI_DATATYPE(unsigned int, MPI_UNSIGNED)
MAKE_MPI_DATATYPE(unsigned long, MPI_UNSIGNED_LONG)
MAKE_MPI_DATATYPE(unsigned long long, MPI_UNSIGNED_LONG_LONG)
MAKE_MPI_DATATYPE(float, MPI_FLOAT)
MAKE_MPI_DATATYPE(double, MPI_DOUBLE)
MAKE_MPI_DATATYPE(long double, MPI_LONG_DOUBLE)

// bool type
MAKE_MPI_DATATYPE(bool, MPI_CXX_BOOL)

// complex types
MAKE_MPI_DATATYPE(std::complex<float>, MPI_CXX_FLOAT_COMPLEX)
MAKE_MPI_DATATYPE(std::complex<double>, MPI_CXX_DOUBLE_COMPLEX)
MAKE_MPI_DATATYPE(std::complex<long double>, MPI_CXX_LONG_DOUBLE_COMPLEX)

#undef MAKE_MPI_DATATYPE

/**
 * @brief Concept for types which have a corresponding MPI datatype.
 *
 * @tparam T C++ type.
 */
template <typename T>
concept mpi_compatible = is_mpi_datatype_v<T>;

/**
 * @brief Concept for ranges which can be used in MPI communication.
 *
 * @tparam R Range type.
 */
template <typename R>
concept mpi_range = ranges::contiguous_range<R> && mpi_compatible<ranges::range_value_t<R>> && ranges::sized_range<R>;

} // namespace mpi

} // namespace simplemc

#endif // SIMPLEMC_MPI_DATATYPES_HPP
