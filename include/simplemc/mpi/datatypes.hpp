/**
 * @file
 * @brief Mappings of MPI datatypes to C++ types.
 */

#ifndef SIMPLEMC_MPI_DATATYPES_HPP
#define SIMPLEMC_MPI_DATATYPES_HPP

#include <mpi.h>
#include <range/v3/range/concepts.hpp>

#include <complex>
#include <cstddef>
#include <type_traits>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-types
 * @{
 */

/**
 * @brief Determine if there is a mapping between a given C++ type and an MPI datatype.
 *
 * @details This struct is intended to be specizalized for C++ types that have a corresponding MPI
 * datatype by inheriting from `std::true_type`.
 *
 * For all other types, it inherits from `std::false_type`.
 *
 * @tparam T C++ type.
 */
template <typename T>
struct is_mpi_datatype : public std::false_type {};

/**
 * @brief Alias template for the boolean value in simplemc::mpi::is_mpi_datatype.
 *
 * @tparam T C++ type.
 */
template <typename T>
inline constexpr bool is_mpi_datatype_v = is_mpi_datatype<T>::value;

/**
 * @brief Map a C++ type to an MPI datatype.
 *
 * @details This struct is intended to be specialized for C++ types that have a corresponding MPI
 * datatype by providing a static method `get()`, which returns the corresponding MPI datatype.
 *
 * For all other types, this struct will be empty.
 *
 * @tparam T C++ type.
 */
template <typename T>
struct mpi_type {};

// Define specializations for is_mpi_datatype and mpi_type.
#define MAKE_MPI_DATATYPE(__cxx_type__, __mpi_type__) \
    /** @brief Specialization of simplemc::mpi::mpi_type for `__cxx_type__`. */ \
    template <> \
    struct mpi_type<__cxx_type__> { \
        /** @brief Returns the MPI datatype `__mpi_type__`. */ \
        static MPI_Datatype get() { \
            return __mpi_type__; \
        } \
    }; \
    /** @brief Specialization of simplemc::mpi::is_mpi_datatype for `__cxx_type__`. */ \
    template <> \
    struct is_mpi_datatype<__cxx_type__> : public std::true_type {};

// Character type.
MAKE_MPI_DATATYPE(char, MPI_CHAR)

// Byte type.
MAKE_MPI_DATATYPE(std::byte, MPI_BYTE)

// Signed integer types.
MAKE_MPI_DATATYPE(short, MPI_SHORT)
MAKE_MPI_DATATYPE(int, MPI_INT)
MAKE_MPI_DATATYPE(long, MPI_LONG)
MAKE_MPI_DATATYPE(long long, MPI_LONG_LONG)
MAKE_MPI_DATATYPE(signed char, MPI_SIGNED_CHAR)

// Unsigned integer types.
MAKE_MPI_DATATYPE(unsigned char, MPI_UNSIGNED_CHAR)
MAKE_MPI_DATATYPE(unsigned short, MPI_UNSIGNED_SHORT)
MAKE_MPI_DATATYPE(unsigned int, MPI_UNSIGNED)
MAKE_MPI_DATATYPE(unsigned long, MPI_UNSIGNED_LONG)
MAKE_MPI_DATATYPE(unsigned long long, MPI_UNSIGNED_LONG_LONG)

// Floating point types.
MAKE_MPI_DATATYPE(float, MPI_FLOAT)
MAKE_MPI_DATATYPE(double, MPI_DOUBLE)
MAKE_MPI_DATATYPE(long double, MPI_LONG_DOUBLE)

// Bool type.
MAKE_MPI_DATATYPE(bool, MPI_CXX_BOOL)

// Complex types.
MAKE_MPI_DATATYPE(std::complex<float>, MPI_CXX_FLOAT_COMPLEX)
MAKE_MPI_DATATYPE(std::complex<double>, MPI_CXX_DOUBLE_COMPLEX)
MAKE_MPI_DATATYPE(std::complex<long double>, MPI_CXX_LONG_DOUBLE_COMPLEX)

#undef MAKE_MPI_DATATYPE

/**
 * @brief A concept that checks if a C++ type has a corresponding MPI datatype and can therefore be
 * used in @ref simplemc-mpi-coll.
 *
 * @tparam T C++ type.
 */
template <typename T>
concept mpi_compatible = is_mpi_datatype_v<T>;

/**
 * @brief A concept that checks if a range type can be used in @ref simplemc-mpi-coll.
 *
 * @details The range has to be contiguous and sized and the value type has to be
 * simplemc::mpi::mpi_compatible.
 *
 * @tparam R Range type.
 */
template <typename R>
concept mpi_range = ranges::contiguous_range<R> && mpi_compatible<ranges::range_value_t<R>> && ranges::sized_range<R>;

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_DATATYPES_HPP
