// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/base.h>
#include <simplemc/serialize/json.hpp>

int main() {
    const auto path = "doc_json_serializer.json";

    // save a scalar at /physics/temperature
    {
        simplemc::json_serializer s;
        s["physics"].save_at("temperature", 3.14);
        simplemc::write_json_file(s.root(), path, { .indent = 4 });
    }

    // load a scalar from /physics/temperature and print
    double temperature = 0.0;
    {
        simplemc::json_serializer d;
        simplemc::read_json_file(d.root(), path);
        d["physics"].load_at("temperature", temperature);
    }
    fmt::println("temperature: {}", temperature);
}
