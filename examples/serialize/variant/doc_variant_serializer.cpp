// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/base.h>
#include <simplemc/serialize/hdf5.hpp>
#include <simplemc/serialize/json.hpp>
#include <simplemc/serialize/variant.hpp>

namespace {

using backend_variant = simplemc::variant_serializer<simplemc::json_serializer, simplemc::hdf5_serializer>;

// Generic code that works for all backends.
void roundtrip(backend_variant& s) {
    s["physics"].save_at("temperature", 3.14);
    double temperature = 0.0;
    s["physics"].load_at("temperature", temperature);
    fmt::println("temperature: {}", temperature);
}

} // namespace

int main() {
    // JSON backend
    backend_variant json_backend { simplemc::json_serializer {} };
    roundtrip(json_backend);

    // HDF5 backend
    backend_variant hdf5_backend { simplemc::hdf5_serializer { "doc_variant_serializer.h5" } };
    roundtrip(hdf5_backend);
}
