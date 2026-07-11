// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/base.h>
#include <simplemc/serialize/hdf5.hpp>

int main() {
    const auto path = "doc_hdf5_serializer.h5";

    // save a scalar at /physics/temperature
    {
        simplemc::hdf5_serializer s { path, simplemc::hdf5_file_mode::truncate };
        s["physics"].save_at("temperature", 3.14);
    }

    // load a scalar from /physics/temperature and print
    double temperature = 0.0;
    {
        const simplemc::hdf5_serializer d { path, simplemc::hdf5_file_mode::read };
        d["physics"].load_at("temperature", temperature);
    }
    fmt::println("temperature: {}", temperature);
}
