/**
 * @file
 * @brief ADL `simplemc_save` / `simplemc_load` overloads for simplemc-owned types.
 *
 * @details Non-intrusive free functions in `namespace simplemc` that round-trip the library's
 * accumulators, grids, lattices and RNG state. ADL finds them because the serializer's namespace
 * (`simplemc`) overlaps with each type's namespace (`simplemc`).
 *
 * Each entry uses only public getters + reconstructing constructors of the underlying type, so no
 * changes to the type headers are required. Generic adapters for `std::vector<T>` and `std::tuple<…>`
 * recurse element-wise so containers of library types work through the same machinery (and don't
 * need separate per-element `nlohmann::adl_serializer` shims).
 */

#ifndef SIMPLEMC_SERIALIZE_LIBRARY_SERIALIZERS_HPP
#define SIMPLEMC_SERIALIZE_LIBRARY_SERIALIZERS_HPP

#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/covar_acc.hpp>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/accs/var_acc.hpp>
#include <simplemc/accs/varalg.hpp>
#include <simplemc/grids/linear_grid.hpp>
#include <simplemc/grids/nd_grid.hpp>
#include <simplemc/grids/power_grid.hpp>
#include <simplemc/grids/symmetric_power_grid.hpp>
#include <simplemc/numeric/bravais_lattice.hpp>
#include <simplemc/random/splitmix64.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json/serializers.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace simplemc {

// ============================================================================================
// Generic adapters for std containers used by library types
// ============================================================================================

/// Render a small integer index as a string key for `save_at` / `load_at`.
///
/// Used by the generic `std::vector` / `std::tuple` adapters below.
namespace detail {
[[nodiscard]] inline std::string index_key(std::size_t i) {
    char buf[32];
    auto n = std::snprintf(buf, sizeof(buf), "%zu", i);
    return std::string { buf, static_cast<std::size_t>(n) };
}
} // namespace detail

/**
 * @brief Serialize `std::vector<T>` element-wise into the current serializer position.
 *
 * @details Only enabled when `T` has its own ADL `simplemc_save` — for types nlohmann already knows
 * how to serialize (`std::vector<int>`, `std::vector<double>`, etc.) the serializer's nlohmann
 * fallback handles it directly. This overload exists for vectors of library types
 * (e.g., `std::vector<mean_acc<…>>`) which nlohmann doesn't know about.
 */
template <output_serializer S, class T, class Alloc>
    requires detail::has_simplemc_save<T, S>
void simplemc_save(S& s, const std::vector<T, Alloc>& v) {
    s.save_at("size", v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        s.save_at(detail::index_key(i), v[i]);
    }
}

/**
 * @brief Deserialize `std::vector<T>` element-wise. Requires `T` to be default-constructible.
 */
template <input_serializer S, class T, class Alloc>
    requires detail::has_simplemc_load<T, S> && std::is_default_constructible_v<T>
void simplemc_load(S& s, std::vector<T, Alloc>& v) {
    std::size_t size = 0;
    s.load_at("size", size);
    v.clear();
    v.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
        T elem;
        s.load_at(detail::index_key(i), elem);
        v.push_back(std::move(elem));
    }
}

/**
 * @brief Serialize `std::tuple<Ts...>` element-wise.
 *
 * @details Each element is written under a numeric key ("0", "1", ...). The number of elements is
 * implicit in the type, so we don't write a separate "size".
 */
template <output_serializer S, class... Ts>
void simplemc_save(S& s, const std::tuple<Ts...>& t) {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        ((s.save_at(detail::index_key(I), std::get<I>(t))), ...);
    }(std::index_sequence_for<Ts...> {});
}

template <input_serializer S, class... Ts>
void simplemc_load(S& s, std::tuple<Ts...>& t) {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        ((s.load_at(detail::index_key(I), std::get<I>(t))), ...);
    }(std::index_sequence_for<Ts...> {});
}

// ============================================================================================
// Accumulators (simplemc-accs)
// ============================================================================================

/**
 * @brief Serialize a `mean_acc` as `{"count": N, "mdata": [...]}`.
 *
 * @details Uses the public `count()` / `mdata()` getters and the `(mdata, count)` constructor for
 * round-trip. The sticky streaming index `idx_` is intentionally not serialized — it resets to 0 on
 * reconstruction, which is the correct invariant.
 */
template <output_serializer S, sample_type T, varalg A>
void simplemc_save(S& s, const mean_acc<T, A>& a) {
    s.save_at("count", a.count());
    s.save_at("mdata", a.mdata());
}

template <input_serializer S, sample_type T, varalg A>
void simplemc_load(S& s, mean_acc<T, A>& a) {
    using ma = mean_acc<T, A>;
    typename ma::count_type c {};
    auto md = a.mdata();
    s.load_at("count", c);
    s.load_at("mdata", md);
    a = ma { md, c };
}

/**
 * @brief Serialize a `var_acc` as `{"count": N, "mdata": [...], "cdata": [...]}`.
 *
 * @details Same approach as `mean_acc`: public getters + `(md, cd, n)` constructor.
 */
template <output_serializer S, sample_type T, varalg A>
void simplemc_save(S& s, const var_acc<T, A>& a) {
    s.save_at("count", a.count());
    s.save_at("mdata", a.mdata());
    s.save_at("cdata", a.cdata());
}

template <input_serializer S, sample_type T, varalg A>
void simplemc_load(S& s, var_acc<T, A>& a) {
    using va = var_acc<T, A>;
    typename va::count_type c {};
    auto md = a.mdata();
    auto cd = a.cdata();
    s.load_at("count", c);
    s.load_at("mdata", md);
    s.load_at("cdata", cd);
    a = va { md, cd, c };
}

/**
 * @brief Serialize a `covar_acc` as `{"count": N, "mdata": [...], "cdata": [[...]]}`.
 *
 * @details `cdata` is a matrix for covar_acc; the Eigen `adl_serializer` in `json/serializers.hpp`
 * round-trips it.
 */
template <output_serializer S, sample_type T, varalg A>
void simplemc_save(S& s, const covar_acc<T, A>& a) {
    s.save_at("count", a.count());
    s.save_at("mdata", a.mdata());
    s.save_at("cdata", a.cdata());
}

template <input_serializer S, sample_type T, varalg A>
void simplemc_load(S& s, covar_acc<T, A>& a) {
    using ca = covar_acc<T, A>;
    typename ca::count_type c {};
    auto md = a.mdata();
    auto cd = a.cdata();
    s.load_at("count", c);
    s.load_at("mdata", md);
    s.load_at("cdata", cd);
    a = ca { md, cd, c };
}

/**
 * @brief Serialize a `batch_acc` as `{"full_batches": [...], "acc_batches": [...]}`.
 *
 * @details The (full, acc) constructor reconstructs `bcount_` and `bidx_` from the batch counts;
 * sticky index inside each mean_acc resets to 0.
 */
template <output_serializer S, sample_type T, varalg A>
void simplemc_save(S& s, const batch_acc<T, A>& a) {
    s.save_at("full_batches", a.batch_vector_full());
    s.save_at("acc_batches", a.batch_vector_accumulating());
}

template <input_serializer S, sample_type T, varalg A>
void simplemc_load(S& s, batch_acc<T, A>& a) {
    using ba = batch_acc<T, A>;
    using mac = typename ba::mean_acc_type;
    std::vector<mac> full;
    std::vector<mac> acc;
    s.load_at("full_batches", full);
    s.load_at("acc_batches", acc);
    a = ba { std::move(full), std::move(acc) };
}

// ============================================================================================
// 1-D grids (simplemc-grids)
// ============================================================================================

/// `linear_grid` is fully described by `(first, last, size)`.
template <output_serializer S>
void simplemc_save(S& s, const linear_grid& g) {
    s.save_at("first", g.first());
    s.save_at("last", g.last());
    s.save_at("size", g.size());
}

template <input_serializer S>
void simplemc_load(S& s, linear_grid& g) {
    double first = 0;
    double last = 0;
    long size = 2;
    s.load_at("first", first);
    s.load_at("last", last);
    s.load_at("size", size);
    g.reset(first, last, size);
}

/// `power_grid` adds the power exponent.
template <output_serializer S>
void simplemc_save(S& s, const power_grid& g) {
    s.save_at("first", g.first());
    s.save_at("last", g.last());
    s.save_at("size", g.size());
    s.save_at("power", g.power());
}

template <input_serializer S>
void simplemc_load(S& s, power_grid& g) {
    double first = 0;
    double last = 0;
    long size = 2;
    double power = 1.0;
    s.load_at("first", first);
    s.load_at("last", last);
    s.load_at("size", size);
    s.load_at("power", power);
    g.reset(first, last, size, power);
}

/// `symmetric_power_grid` reads the power exponent from its underlying half-grid.
template <output_serializer S>
void simplemc_save(S& s, const symmetric_power_grid& g) {
    s.save_at("first", g.first());
    s.save_at("last", g.last());
    s.save_at("size", g.size());
    s.save_at("power", g.grid1().power());
}

template <input_serializer S>
void simplemc_load(S& s, symmetric_power_grid& g) {
    double first = 0;
    double last = 0;
    long size = 3;
    double power = 1.0;
    s.load_at("first", first);
    s.load_at("last", last);
    s.load_at("size", size);
    s.load_at("power", power);
    g.reset(first, last, size, power);
}

// ============================================================================================
// N-D grid (simplemc-grids)
// ============================================================================================

/// `nd_grid` is just a tuple of 1-D grids; the tuple adapter does the work.
template <output_serializer S, grid_1d... Grids>
void simplemc_save(S& s, const nd_grid<Grids...>& g) {
    s.save_at("grids", g.grids());
}

template <input_serializer S, grid_1d... Grids>
void simplemc_load(S& s, nd_grid<Grids...>& g) {
    s.load_at("grids", g.grids());
}

// ============================================================================================
// Bravais lattice (simplemc-numeric)
// ============================================================================================

/// `bravais_lattice<N>` is reconstructed from its real-space basis matrix; B/volumes are derived.
template <output_serializer S, int N>
void simplemc_save(S& s, const bravais_lattice<N>& l) {
    s.save_at("A", l.real_lattice_vectors());
}

template <input_serializer S, int N>
void simplemc_load(S& s, bravais_lattice<N>& l) {
    typename bravais_lattice<N>::matrix_type A;
    s.load_at("A", A);
    l.reset(A);
}

// ============================================================================================
// RNG state (simplemc-random)
// ============================================================================================

/// `xoshiro256<X>` round-trips through its 256-bit state.
template <output_serializer S, xoshiro256_type X>
void simplemc_save(S& s, const xoshiro256<X>& r) {
    const auto& st = r.internal_state();
    s.save_at("s0", st[0]);
    s.save_at("s1", st[1]);
    s.save_at("s2", st[2]);
    s.save_at("s3", st[3]);
}

template <input_serializer S, xoshiro256_type X>
void simplemc_load(S& s, xoshiro256<X>& r) {
    std::uint64_t s0 = 0;
    std::uint64_t s1 = 0;
    std::uint64_t s2 = 0;
    std::uint64_t s3 = 0;
    s.load_at("s0", s0);
    s.load_at("s1", s1);
    s.load_at("s2", s2);
    s.load_at("s3", s3);
    r.seed(s0, s1, s2, s3);
}

/// `splitmix64` round-trips through its single 64-bit state value.
template <output_serializer S>
void simplemc_save(S& s, const splitmix64& r) {
    s.save_at("state", r.internal_state());
}

template <input_serializer S>
void simplemc_load(S& s, splitmix64& r) {
    std::uint64_t st = 0;
    s.load_at("state", st);
    r.seed(st);
}

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_LIBRARY_SERIALIZERS_HPP
