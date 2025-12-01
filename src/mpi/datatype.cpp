/**
 * @file
 * @brief Implementation details for simplemc/mpi/datatype.hpp.
 */

#include <simplemc/mpi/datatype.hpp>
#include <simplemc/mpi/environment.hpp>
#include <simplemc/mpi/utils.hpp>

#include <mpi.h>

#include <memory>

namespace simplemc::mpi {

datatype::datatype() : dt_(new MPI_Datatype(MPI_DATATYPE_NULL)) {}

datatype::datatype(MPI_Datatype dt, resource_policy pol) {
    if (pol == resource_policy::take_ownership) {
        dt_ = std::shared_ptr<MPI_Datatype> { new MPI_Datatype(dt), datatype_deleter {} };
    } else {
        dt_ = std::make_shared<MPI_Datatype>(dt);
    }

    // automatically commit the datatype (no-op if already committed or predefined)
    if (dt_ && *dt_ != MPI_DATATYPE_NULL && !type_is_predefined(*dt_)) {
        check_mpi_call(MPI_Type_commit(&(*dt_)), "MPI_Type_commit");
    }
}

int datatype::size() const {
    int type_size = 0;
    check_mpi_call(MPI_Type_size(*dt_, &type_size), "MPI_Type_size");
    return type_size;
}

void datatype::datatype_deleter::operator()(MPI_Datatype* dt) const {
    // only free if valid and not a predefined datatype
    if (!environment::is_finalized() && dt && *dt != MPI_DATATYPE_NULL && !type_is_predefined(*dt)) {
        MPI_Type_free(dt);
    }
    delete dt;
}

void type_free(MPI_Datatype& dt) {
    if (dt != MPI_DATATYPE_NULL && !type_is_predefined(dt)) {
        check_mpi_call(MPI_Type_free(&dt), "MPI_Type_free");
    }
}

int type_combiner(MPI_Datatype dt) {
    int num_integers = 0;
    int num_addresses = 0;
    int num_datatypes = 0;
    int combiner = 0;
    check_mpi_call(MPI_Type_get_envelope(dt, &num_integers, &num_addresses, &num_datatypes, &combiner),
        "MPI_Type_get_envelope");
    return combiner;
}

bool type_is_predefined(MPI_Datatype dt) {
    if (dt == MPI_DATATYPE_NULL) {
        return false;
    }
    return type_combiner(dt) == MPI_COMBINER_NAMED;
}

} // namespace simplemc::mpi
