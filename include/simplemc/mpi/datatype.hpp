/**
 * @file
 * @brief Wrapper around an `MPI_Datatype` object.
 */

#ifndef SIMPLEMC_MPI_DATATYPE_HPP
#define SIMPLEMC_MPI_DATATYPE_HPP

#include <simplemc/mpi/utils.hpp>

#include <mpi.h>

#include <memory>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-types
 * @{
 */

/**
 * @brief A wrapper around an MPI datatype.
 *
 * @details This class provides a thin C++ wrapper around `MPI_Datatype` objects. MPI datatypes
 * describe the layout of data in memory and are used in communication operations.
 *
 * The datatype is managed using a `std::shared_ptr` with a custom deleter providing automatic
 * resource management similar to <a href="https://www.boost.org/doc/libs/latest/doc/html/mpi.html">
 * Boost.MPI</a>. The behaviour of the deleter depends on the simplemc::mpi::resource_policy
 * specified during construction:
 * - `take_ownership`: The wrapper is responsible for managing the lifetime of the MPI datatype, i.e.
 * it calls `MPI_Type_free` when the last reference is destroyed (unless it is a predefined type like
 * `MPI_INT` or `MPI_DATATYPE_NULL`).
 * - `attach`: The lifetime of the MPI datatype is managed somewhere else and the wrapper just uses
 * it.
 *
 * Implicit conversions to `MPI_Datatype` are allowed for interoperability with MPI C functions.
 *
 * @note Derived MPI datatypes are committed automatically in the constructor and factory functions
 * (see e.g. simplemc::mpi::type_create_contiguous).
 */
class datatype {
public:
    /**
     * @brief Default construct a null datatype.
     *
     * @details It creates a datatype that wraps `MPI_DATATYPE_NULL`. This represents an invalid or
     * uninitialized datatype.
     */
    datatype();

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
     * @brief Get the size of the datatype in bytes.
     *
     * @details It makes a call to `MPI_Type_size` and throws an exception if the call fails.
     *
     * @return Size of the datatype in bytes.
     */
    [[nodiscard]] int size() const;

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
        void operator()(MPI_Datatype* dt) const;
    };

private:
    // Shared pointer to the underlying MPI_Datatype.
    std::shared_ptr<MPI_Datatype> dt_;
};

/**
 * @brief Free an `MPI_Datatype` object.
 *
 * @details It makes a call to `MPI_Type_free` to free the MPI datatype. After calling this
 * function, the datatype will be set to `MPI_DATATYPE_NULL`.
 *
 * It throws an exception if the call fails. Does nothing if the datatype is `MPI_DATATYPE_NULL` or
 * a predefined type.
 *
 * @param dt Reference to the `MPI_Datatype` to free.
 */
void type_free(MPI_Datatype& dt);

/**
 * @brief Get the combiner of an MPI datatype.
 *
 * @details It makes a call to `MPI_Type_get_envelope` to retrieve the combiner. The combiner
 * indicates how the datatype was constructed (e.g., `MPI_COMBINER_NAMED` for predefined types,
 * `MPI_COMBINER_CONTIGUOUS` for contiguous types, etc.).
 *
 * It throws an exception if the call fails.
 *
 * @param dt The `MPI_Datatype` to query.
 * @return The combiner value (e.g., `MPI_COMBINER_NAMED`, `MPI_COMBINER_CONTIGUOUS`, etc.).
 */
[[nodiscard]] int type_combiner(MPI_Datatype dt);

/**
 * @brief Check if an MPI datatype is a predefined (built-in) type.
 *
 * @details Predefined types include `MPI_INT`, `MPI_DOUBLE`, `MPI_CHAR`, etc. These types should
 * not be freed by the user. A type is predefined if its combiner is `MPI_COMBINER_NAMED`.
 *
 * @param dt The `MPI_Datatype` to check.
 * @return True if the datatype is predefined, false otherwise.
 */
[[nodiscard]] bool type_is_predefined(MPI_Datatype dt);

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_DATATYPE_HPP
