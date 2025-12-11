/**
 * @file
 * @brief Implementation details for simplemc/mpi/datatype.hpp.
 */

#include <simplemc/mpi/datatype.hpp>
#include <simplemc/mpi/environment.hpp>
#include <simplemc/mpi/utils.hpp>

#include <mpi.h>

#include <span>

namespace simplemc::mpi {

void type_free(MPI_Datatype& dt) {
    if (dt != MPI_DATATYPE_NULL && !type_is_predefined(dt)) {
        check_mpi_call(MPI_Type_free(&dt), "MPI_Type_free");
    }
}

void type_commit(MPI_Datatype& dt) {
    if (dt != MPI_DATATYPE_NULL) {
        check_mpi_call(MPI_Type_commit(&dt), "MPI_Type_commit");
    }
}

int type_size(MPI_Datatype dt) {
    int sz {};
    check_mpi_call(MPI_Type_size(dt, &sz), "MPI_Type_size");
    return sz;
}

type_extent type_get_extent(MPI_Datatype dt) {
    MPI_Aint lb {};
    MPI_Aint extent {};
    check_mpi_call(MPI_Type_get_extent(dt, &lb, &extent), "MPI_Type_get_extent");
    return { .lb = lb, .extent = extent };
}

type_envelope type_get_envelope(MPI_Datatype dt) {
    int num_integers {};
    int num_addresses {};
    int num_datatypes {};
    int combiner {};
    check_mpi_call(
        MPI_Type_get_envelope(dt, &num_integers, &num_addresses, &num_datatypes, &combiner), "MPI_Type_get_envelope");
    return { .num_integers = num_integers,
        .num_addresses = num_addresses,
        .num_datatypes = num_datatypes,
        .combiner = combiner };
}

bool type_is_predefined(MPI_Datatype dt) {
    if (dt == MPI_DATATYPE_NULL) {
        return false;
    }
    auto envelope = type_get_envelope(dt);
    return envelope.combiner == MPI_COMBINER_NAMED;
}

MPI_Datatype type_contiguous(int count, MPI_Datatype oldtype) {
    MPI_Datatype newtype {};
    check_mpi_call(MPI_Type_contiguous(count, oldtype, &newtype), "MPI_Type_contiguous");
    return newtype;
}

MPI_Datatype type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype) {
    MPI_Datatype newtype {};
    check_mpi_call(MPI_Type_vector(count, blocklength, stride, oldtype, &newtype), "MPI_Type_vector");
    return newtype;
}

MPI_Datatype type_create_struct(std::span<const int> blocklengths, std::span<const MPI_Aint> displacements,
    std::span<const MPI_Datatype> oldtypes) {
    MPI_Datatype newtype {};
    check_mpi_call(MPI_Type_create_struct(static_cast<int>(blocklengths.size()), blocklengths.data(),
                       displacements.data(), oldtypes.data(), &newtype),
        "MPI_Type_create_struct");
    return newtype;
}

datatype::datatype(MPI_Datatype dt, resource_policy pol) : is_predefined_(type_is_predefined(dt)) {
    if (pol == resource_policy::take_ownership) {
        dt_ = std::shared_ptr<MPI_Datatype> { new MPI_Datatype(dt), datatype_deleter { is_predefined_ } };
    } else {
        dt_ = std::make_shared<MPI_Datatype>(dt);
    }

    // automatically commit the datatype (no-op if already committed or predefined)
    if (dt_ && *dt_ != MPI_DATATYPE_NULL && !is_predefined_) {
        type_commit(*dt_);
    }
}

void datatype::datatype_deleter::operator()(MPI_Datatype* dt) const noexcept {
    if (!finalized() && dt && *dt != MPI_DATATYPE_NULL && !is_predefined) {
        MPI_Type_free(dt);
    }
    delete dt;
}

} // namespace simplemc::mpi
