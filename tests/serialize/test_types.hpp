// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#ifndef SIMPLEMC_TESTS_SERIALIZE_TEST_TYPES_HPP
#define SIMPLEMC_TESTS_SERIALIZE_TEST_TYPES_HPP

#include <simplemc/serialize/concepts.hpp>

#include <vector>

namespace test_types {

// Type with an intrusive friend simplemc_save / simplemc_load (private member access).
struct intrusive_point {
private:
    double x_ = 0;
    double y_ = 0;

public:
    intrusive_point() = default;
    intrusive_point(double x, double y) : x_ { x }, y_ { y } {}

    [[nodiscard]] double x() const { return x_; }
    [[nodiscard]] double y() const { return y_; }

    bool operator==(const intrusive_point&) const = default;

    template <simplemc::serializer S>
    friend void simplemc_save(S s, const intrusive_point& p) {
        s.save_at("x", p.x_);
        s.save_at("y", p.y_);
    }
    template <simplemc::serializer S>
    friend void simplemc_load(const S& s, intrusive_point& p) {
        s.load_at("x", p.x_);
        s.load_at("y", p.y_);
    }
};

// Type with public fields and namespace-scope simplemc_save / simplemc_load (non-intrusive).
struct nonintrusive_box {
    int width = 0;
    int height = 0;

    bool operator==(const nonintrusive_box&) const = default;
};

template <simplemc::serializer S>
void simplemc_save(S s, const nonintrusive_box& b) {
    s.save_at("width", b.width);
    s.save_at("height", b.height);
}
template <simplemc::serializer S>
void simplemc_load(const S& s, nonintrusive_box& b) {
    s.load_at("width", b.width);
    s.load_at("height", b.height);
}

// Composite type that uses both customization mechanisms recursively.
struct composite {
    intrusive_point pt;
    nonintrusive_box bx;
    std::vector<double> data;

    bool operator==(const composite&) const = default;
};

template <simplemc::serializer S>
void simplemc_save(S s, const composite& c) {
    s.save_at("pt", c.pt);
    s.save_at("bx", c.bx);
    s.save_at("data", c.data);
}
template <simplemc::serializer S>
void simplemc_load(const S& s, composite& c) {
    s.load_at("pt", c.pt);
    s.load_at("bx", c.bx);
    s.load_at("data", c.data);
}

} // namespace test_types

#endif // SIMPLEMC_TESTS_SERIALIZE_TEST_TYPES_HPP
