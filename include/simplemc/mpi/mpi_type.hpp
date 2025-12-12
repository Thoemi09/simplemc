/**
 * @file
 * @brief Mappings of MPI datatypes to C++ types.
 */

#ifndef SIMPLEMC_MPI_MPI_TYPE_HPP
#define SIMPLEMC_MPI_MPI_TYPE_HPP

#include <simplemc/utils/ranges.hpp>

#include <mpi.h>

#include <complex>
#include <cstddef>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-essentials-types
 * @{
 */

/**
 * @brief Map a C++ type to an MPI datatype.
 *
 * @details This struct is intended to be specialized for C++ types that have a corresponding MPI
 * datatype by providing a static method `get()`, which returns the corresponding MPI datatype.
 *
 * For all other types, this struct will be empty.
 *
 * Supported predefined types with their corresponding MPI datatypes are:
 * - Character types:
 *   - `char` \f$ \rightarrow \f$ `MPI_CHAR`
 * - Byte type:
 *   - `std::byte` \f$ \rightarrow \f$ `MPI_BYTE`
 * - Signed integer types:
 *   - `signed char` \f$ \rightarrow \f$ `MPI_SIGNED_CHAR`
 *   - `short` \f$ \rightarrow \f$ `MPI_SHORT`
 *   - `int` \f$ \rightarrow \f$ `MPI_INT`
 *   - `long` \f$ \rightarrow \f$ `MPI_LONG`
 *   - `long long` \f$ \rightarrow \f$ `MPI_LONG_LONG`
 * - Unsigned integer types:
 *   - `unsigned char` \f$ \rightarrow \f$ `MPI_UNSIGNED_CHAR`
 *   - `unsigned short` \f$ \rightarrow \f$ `MPI_UNSIGNED_SHORT`
 *   - `unsigned int` \f$ \rightarrow \f$ `MPI_UNSIGNED`
 *   - `unsigned long` \f$ \rightarrow \f$ `MPI_UNSIGNED_LONG`
 *   - `unsigned long long` \f$ \rightarrow \f$ `MPI_UNSIGNED_LONG_LONG`
 * - Floating point types:
 *   - `float` \f$ \rightarrow \f$ `MPI_FLOAT`
 *   - `double` \f$ \rightarrow \f$ `MPI_DOUBLE`
 *   - `long double` \f$ \rightarrow \f$ `MPI_LONG_DOUBLE`
 * - Boolean type:
 *   - `bool` \f$ \rightarrow \f$ `MPI_CXX_BOOL`
 * - Complex types:
 *   - `std::complex<float>` \f$ \rightarrow \f$ `MPI_CXX_FLOAT_COMPLEX`
 *   - `std::complex<double>` \f$ \rightarrow \f$ `MPI_CXX_DOUBLE_COMPLEX`
 *   - `std::complex<long double>` \f$ \rightarrow \f$ `MPI_CXX_LONG_DOUBLE_COMPLEX`
 *
 * @note Users can provide specializations for their own types or for types not covered here. See 
 * @ref tut_mpi_4.
 *
 * @tparam T C++ type.
 */
template <typename T>
struct mpi_type {};

// Define specializations for mpi_type.
#define MAKE_MPI_DATATYPE(__cxx_type__, __mpi_type__) \
    template <> \
    struct mpi_type<__cxx_type__> { \
        [[nodiscard]] static constexpr MPI_Datatype get() noexcept { return __mpi_type__; } \
    };

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
concept mpi_compatible = requires {
    { mpi_type<T>::get() } -> std::same_as<MPI_Datatype>;
};

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

#endif // SIMPLEMC_MPI_MPI_TYPE_HPP
