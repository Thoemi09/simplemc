// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief RAII wrapper for `MPI_Datatype` objects and other essential MPI functions involving
 * datatypes.
 */

#ifndef SIMPLEMC_MPI_DATATYPE_HPP
#define SIMPLEMC_MPI_DATATYPE_HPP

#include <simplemc/mpi/utils.hpp>

#include <mpi.h>

#include <memory>
#include <span>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-essentials-types
 * @{
 */

/**
 * @brief Information about how an MPI datatype was constructed.
 *
 * @details It holds the envelope information returned by simplemc::mpi::type_get_envelope, which
 * describes how a datatype was constructed.
 */
struct type_envelope {
    /// Number of input integers used in the datatype constructor.
    int num_integers;

    /// Number of input addresses used in the datatype constructor.
    int num_addresses;

    /// Number of input datatypes used in the datatype constructor.
    int num_datatypes;

    /// Combiner indicating how the datatype was constructed.
    int combiner;
};

/**
 * @brief Information about the extent of an MPI datatype.
 *
 * @details It holds the lower bound and extent returned by simplemc::mpi::type_get_extent.
 */
struct type_extent {
    /// Lower bound of the datatype.
    MPI_Aint lb;

    /// Extent of the datatype.
    MPI_Aint extent;
};

/**
 * @brief Free an `MPI_Datatype` object.
 *
 * @details It makes a call to `MPI_Type_free` to free the datatype, unless it is `MPI_DATATYPE_NULL`
 * or a predefined type (see simplemc::mpi::type_is_predefined), in which case it does nothing.
 *
 * @param dt Reference to the datatype to free.
 */
void type_free(MPI_Datatype& dt);

/**
 * @brief Commit an MPI datatype by calling `MPI_Type_commit`.
 *
 * @details It commits the datatype, which is required before using a derived datatype in
 * communication operations.
 *
 * Committing an already committed datatype is a no-op in MPI and `MPI_DATATYPE_NULL` is ignored.
 *
 * @param dt Datatype to commit.
 */
void type_commit(MPI_Datatype& dt);

/**
 * @brief Get the size of an MPI datatype in bytes by calling `MPI_Type_size`.
 *
 * @param dt Datatype to query.
 * @return Size of the datatype in bytes.
 */
[[nodiscard]] int type_size(MPI_Datatype dt);

/**
 * @brief Get the lower bound and extent of an MPI datatype by calling `MPI_Type_get_extent`.
 *
 * @details The lower bound is the smallest displacement used in the datatype's definition (may be
 * negative or user-defined).
 *
 * The extent is the span from the first (lower bound) to the last byte occupied by entries in the
 * datatype, which includes any padding. This is used to determine the stride for array elements.
 *
 * @param dt Datatype to query.
 * @return A simplemc::mpi::type_extent struct containing the lower bound and the extent.
 */
[[nodiscard]] type_extent type_get_extent(MPI_Datatype dt);

/**
 * @brief Get the envelope of an MPI datatype by calling `MPI_Type_get_envelope`.
 *
 * @details It retrieves information about how the datatype was constructed, including the combiner
 * and the number of integers, addresses, and datatypes used in its construction.
 *
 * @param dt Datatype to query.
 * @return A simplemc::mpi::type_envelope struct containing the envelope information.
 */
[[nodiscard]] type_envelope type_get_envelope(MPI_Datatype dt);

/**
 * @brief Check if an MPI datatype is a predefined (built-in) type.
 *
 * @details Predefined types include `MPI_INT`, `MPI_DOUBLE`, `MPI_CHAR`, etc. These types should
 * not be freed by the user. A type is predefined if its combiner is `MPI_COMBINER_NAMED`.
 *
 * @note The function returns false for `MPI_DATATYPE_NULL`.
 *
 * @param dt Datatype to check.
 * @return True if the datatype is predefined, false otherwise.
 */
[[nodiscard]] bool type_is_predefined(MPI_Datatype dt);

/**
 * @brief Create a contiguous MPI datatype by calling `MPI_Type_contiguous`.
 *
 * @details It creates a new datatype representing `count` contiguous elements of the old datatype.
 *
 * @note The caller is responsible for freeing the returned `MPI_Datatype` using
 * simplemc::mpi::type_free or `MPI_Type_free()`.
 *
 * @param count Number of elements in the new datatype.
 * @param oldtype Base datatype.
 * @return New contiguous datatype.
 */
[[nodiscard]] MPI_Datatype type_contiguous(int count, MPI_Datatype oldtype);

/**
 * @brief Create a vector MPI datatype by calling `MPI_Type_vector`.
 *
 * @details It creates a new datatype representing a sequence of `count` blocks, each containing
 * `blocklength` elements of the old datatype, with `stride` elements between the start of each
 * block.
 *
 * @note The caller is responsible for freeing the returned `MPI_Datatype` using
 * simplemc::mpi::type_free or `MPI_Type_free()`.
 *
 * @param count Number of blocks.
 * @param blocklength Number of elements in each block.
 * @param stride Number of elements between the start of each block.
 * @param oldtype Base datatype.
 * @return New vector datatype.
 */
[[nodiscard]] MPI_Datatype type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype);

/**
 * @brief Create a structured MPI datatype by calling `MPI_Type_create_struct`.
 *
 * @details It creates a new datatype with arbitrary component types, displacements, and block
 * lengths.
 *
 * @note The caller is responsible for freeing the returned `MPI_Datatype` using
 * simplemc::mpi::type_free or `MPI_Type_free()`.
 *
 * @param blocklengths Array of block lengths (number of elements of each type).
 * @param displacements Array of byte displacements of each block.
 * @param oldtypes Array of datatypes for each block.
 * @return New structured datatype.
 */
[[nodiscard]] MPI_Datatype type_create_struct(
    std::span<const int> blocklengths, std::span<const MPI_Aint> displacements, std::span<const MPI_Datatype> oldtypes);

/**
 * @brief RAII wrapper for `MPI_Datatype` objects.
 *
 * @details This class provides a thin C++ wrapper around `MPI_Datatype` objects. MPI datatypes
 * describe the layout of data in memory and are used in communication operations.
 *
 * The datatype is managed using a `std::shared_ptr` with a custom deleter providing automatic
 * resource management similar to <a href="https://www.boost.org/doc/libs/latest/doc/html/mpi.html">
 * Boost.MPI</a>. The behavior of the deleter depends on the simplemc::mpi::resource_policy
 * specified during construction:
 * - `take_ownership`: The wrapper is responsible for managing the lifetime of the MPI datatype, i.e.
 * it calls `MPI_Type_free` when the last reference is destroyed (unless it is a predefined type like
 * `MPI_INT` or `MPI_DATATYPE_NULL`).
 * - `attach`: The lifetime of the MPI datatype is managed somewhere else and the wrapper just uses
 * it.
 *
 * Implicit conversions to `MPI_Datatype` are allowed for interoperability with MPI C functions.
 *
 * See @ref tut_mpi_4 for an example.
 */
class datatype {
public:
    /**
     * @brief Default construct a null datatype.
     *
     * @details It creates a datatype that wraps `MPI_DATATYPE_NULL`. This represents an invalid or
     * uninitialized datatype.
     */
    datatype() : dt_(new MPI_Datatype(MPI_DATATYPE_NULL)) {}

    /**
     * @brief Construct a datatype from an existing `MPI_Datatype` with a given
     * simplemc::mpi::resource_policy.
     *
     * @details If the resource policy is set to `take_ownership` (the default), the datatype takes
     * ownership of the provided `MPI_Datatype` and will free it when the last reference is destroyed
     * (unless it is a predefined type).
     *
     * If the policy is `attach`, the datatype simply attaches to the provided `MPI_Datatype` without
     * taking ownership, and the caller remains responsible for freeing it.
     *
     * @note The datatype is automatically committed (committing an already committed type is a no-op
     * in MPI).
     *
     * @param dt `MPI_Datatype` object to wrap.
     * @param pol simplemc::mpi::resource_policy indicating ownership semantics.
     */
    explicit datatype(MPI_Datatype dt, resource_policy pol = resource_policy::take_ownership);

    /**
     * @brief Construct a contiguous datatype by calling simplemc::mpi::type_contiguous.
     *
     * @details It calls simplemc::mpi::type_contiguous to create a new MPI datatype and passes it to
     * datatype(MPI_Datatype, resource_policy).
     *
     * @param count Number of elements in the new datatype.
     * @param oldtype Base datatype.
     * @param pol simplemc::mpi::resource_policy indicating ownership semantics.
     */
    datatype(int count, MPI_Datatype oldtype, resource_policy pol = resource_policy::take_ownership) :
        datatype { type_contiguous(count, oldtype), pol } {}

    /**
     * @brief Construct a vector datatype by calling simplemc::mpi::type_vector.
     *
     * @details It calls simplemc::mpi::type_vector to create a new MPI datatype and passes it to
     * datatype(MPI_Datatype, resource_policy).
     *
     * @param count Number of blocks.
     * @param blocklength Number of elements in each block.
     * @param stride Number of elements between the start of each block.
     * @param oldtype Base datatype.
     * @param pol simplemc::mpi::resource_policy indicating ownership semantics.
     */
    datatype(int count, int blocklength, int stride, MPI_Datatype oldtype,
        resource_policy pol = resource_policy::take_ownership) :
        datatype { type_vector(count, blocklength, stride, oldtype), pol } {}

    /**
     * @brief Construct a structured datatype by calling simplemc::mpi::type_create_struct.
     *
     * @details It calls simplemc::mpi::type_create_struct to create a new MPI datatype and passes it
     * to datatype(MPI_Datatype, resource_policy).
     *
     * @param blocklengths Array of block lengths (number of elements of each type).
     * @param displacements Array of byte displacements of each block.
     * @param oldtypes Array of datatypes for each block.
     * @param pol simplemc::mpi::resource_policy indicating ownership semantics.
     */
    datatype(std::span<const int> blocklengths, std::span<const MPI_Aint> displacements,
        std::span<const MPI_Datatype> oldtypes, resource_policy pol = resource_policy::take_ownership) :
        datatype { type_create_struct(blocklengths, displacements, oldtypes), pol } {}

    /**
     * @brief Get the size of the datatype in bytes by calling simplemc::mpi::type_size.
     *
     * @return Size of the datatype in bytes.
     */
    [[nodiscard]] int size() const { return type_size(*dt_); }

    /**
     * @brief Check if the datatype is predefined.
     *
     * @details Predefined types include `MPI_INT`, `MPI_DOUBLE`, `MPI_CHAR`, etc. These types
     * should not be freed by the user.
     *
     * @return True if the datatype is predefined, false otherwise.
     */
    [[nodiscard]] bool is_predefined() const noexcept { return is_predefined_; }

    /**
     * @brief Get the lower bound and extent of the datatype by calling
     * simplemc::mpi::type_get_extent.
     *
     * @return A simplemc::mpi::type_extent containing the lower bound and the extent.
     */
    [[nodiscard]] type_extent extent() const { return type_get_extent(*dt_); }

    /**
     * @brief Get the envelope of the datatype by calling simplemc::mpi::type_get_envelope.
     *
     * @return A simplemc::mpi::type_envelope struct containing the envelope information.
     */
    [[nodiscard]] type_envelope envelope() const { return type_get_envelope(*dt_); }

    /**
     * @brief Implicit conversion to the underlying `MPI_Datatype` object.
     *
     * @return The underlying `MPI_Datatype` or `MPI_DATATYPE_NULL` if no datatype is wrapped.
     */
    operator MPI_Datatype() const { return dt_ ? *dt_ : MPI_DATATYPE_NULL; }

    /**
     * @brief Explicit conversion to bool to check if the datatype is valid.
     *
     * @return True if the datatype is not `MPI_DATATYPE_NULL`.
     */
    [[nodiscard]] explicit operator bool() const { return dt_ && *dt_ != MPI_DATATYPE_NULL; }

private:
    // Custom deleter that makes a protected call to MPI_Type_free.
    struct datatype_deleter {
        bool is_predefined;
        void operator()(MPI_Datatype* dt) const noexcept;
    };

private:
    std::shared_ptr<MPI_Datatype> dt_;
    bool is_predefined_ { false };
};

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_DATATYPE_HPP
