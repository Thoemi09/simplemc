/**
 * @file
 * @brief Base class of simplemc::update_set / simplemc::measurement_set.
 */

#ifndef SIMPLEMC_MC_TUPLE_SET_HPP
#define SIMPLEMC_MC_TUPLE_SET_HPP

#include <simplemc/mpi/communicator.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <optional>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-entities
 * @{
 */

/**
 * @brief Statically-typed, ordered collection of named entries.
 *
 * @details It defines the storage, traversal, lookup and typed-access API common to 
 * simplemc::update_set and simplemc::measurement_set. The entries are stored in a `std::tuple`, so 
 * the set of entry *types* is fixed at construction. Their metadata (name, weight, active flag, ...) 
 * stays mutable at runtime.
 *
 * Each entry `e` is required to expose a unique name via `e.name`, the wrapped user value via
 * `e.payload`, and its type via the nested alias `e::payload_type`. The name is used as the key for
 * lookup and serialization.
 *
 * Two traversal primitives carry the whole design:
 *
 * - for_each() applies a callable to *every* entry (a compile-time loop over the tuple), and
 * - visit_at() applies a callable to the *single* entry at a runtime index.
 *
 * Both accept a generic callable (e.g. `[](auto& e){ ... }`) since every entry is a different type.
 *
 * @tparam Entries Entry types (simplemc::update or simplemc::measurement instantiations).
 */
template <typename... Entries>
class tuple_set {
public:
    /**
     * @brief Construct the set from its entries.
     *
     * @details The entries are moved into the underlying tuple. With no arguments the set is empty.
     *
     * @param es Entries to store, in order.
     */
    explicit tuple_set(Entries... es) : entries_ { std::move(es)... } {}

    /**
     * @brief Number of entries in the set.
     *
     * @return Entry count (a compile-time constant).
     */
    [[nodiscard]] static constexpr std::size_t size() noexcept { return sizeof...(Entries); }

    /**
     * @brief Whether the set has no entries.
     *
     * @return True if the set is empty.
     */
    [[nodiscard]] static constexpr bool empty() noexcept { return sizeof...(Entries) == 0; }

    /**
     * @brief Access the entry at a compile-time index.
     *
     * @tparam I Entry index.
     * @return Reference to the i-th entry (a concrete entry type).
     */
    template <std::size_t I>
    [[nodiscard]] constexpr auto& at() noexcept {
        return std::get<I>(entries_);
    }

    /**
     * @brief Const overload of at().
     */
    template <std::size_t I>
    [[nodiscard]] constexpr const auto& at() const noexcept {
        return std::get<I>(entries_);
    }

    /**
     * @brief Apply a callable to every entry, in order.
     *
     * @tparam F Callable type invocable as `f(entry&)` for every entry type.
     * @param f Callable to apply.
     */
    template <typename F>
    constexpr void for_each(F&& f) { // NOLINT (callable invoked in place, not forwarded)
        std::apply([&](auto&... es) { (f(es), ...); }, entries_);
    }

    /**
     * @brief Const overload of for_each().
     */
    template <typename F>
    constexpr void for_each(F&& f) const { // NOLINT (callable invoked in place, not forwarded)
        std::apply([&](const auto&... es) { (f(es), ...); }, entries_);
    }

    /**
     * @brief Apply a callable to the single entry at a runtime index.
     *
     * @details Only the matching entry is invoked; the dispatch is O(1) through a per-instantiation
     * function-pointer table. Out-of-range indices invoke nothing. The callable must return `void`.
     *
     * @tparam F Callable type invocable as `f(entry&)` for every entry type.
     * @param i Entry index.
     * @param f Callable to apply to the i-th entry.
     */
    template <typename F>
    constexpr void visit_at(std::size_t i, F&& f) { // NOLINT (callable invoked in place, not forwarded)
        using fn_t = void (*)(std::tuple<Entries...>&, F&);
        static constexpr auto table = []<std::size_t... I>(std::index_sequence<I...>) {
            return std::array<fn_t, sizeof...(Entries)> { [](std::tuple<Entries...>& es, F& g) {
                g(std::get<I>(es));
            }... };
        }(std::index_sequence_for<Entries...> {});
        if (i < size()) {
            table[i](entries_, f);
        }
    }

    /**
     * @brief Const overload of visit_at().
     */
    template <typename F>
    constexpr void visit_at(std::size_t i, F&& f) const { // NOLINT (callable invoked in place, not forwarded)
        using fn_t = void (*)(const std::tuple<Entries...>&, F&);
        static constexpr auto table = []<std::size_t... I>(std::index_sequence<I...>) {
            return std::array<fn_t, sizeof...(Entries)> { [](const std::tuple<Entries...>& es, F& g) {
                g(std::get<I>(es));
            }... };
        }(std::index_sequence_for<Entries...> {});
        if (i < size()) {
            table[i](entries_, f);
        }
    }

    /**
     * @brief Look up an entry by name.
     *
     * @param name Name to search for.
     * @return Index of the first entry with that name, or `std::nullopt` if none matches.
     */
    [[nodiscard]] std::optional<std::size_t> find(std::string_view name) const noexcept {
        return [&]<std::size_t... I>(std::index_sequence<I...>) -> std::optional<std::size_t> {
            std::optional<std::size_t> idx;
            (void)(((std::get<I>(entries_).name == name) ? (idx = I, true) : false) || ...);
            return idx;
        }(std::index_sequence_for<Entries...> {});
    }

    /**
     * @brief Recover a typed pointer to the single entry whose payload has type `T`.
     *
     * @details A compile-time error is raised unless exactly one entry has payload type `T`.
     *
     * @tparam T Concrete user type.
     * @return Pointer to the wrapped user value.
     */
    template <typename T>
    [[nodiscard]] T* get() noexcept {
        static_assert(count_payload<T>() == 1, "get<T>() requires exactly one entry with payload type T");
        T* out = nullptr;
        for_each([&](auto& e) {
            if constexpr (std::same_as<typename std::remove_cvref_t<decltype(e)>::payload_type, T>) {
                out = &e.payload;
            }
        });
        return out;
    }

    /**
     * @brief Const overload of get().
     */
    template <typename T>
    [[nodiscard]] const T* get() const noexcept {
        static_assert(count_payload<T>() == 1, "get<T>() requires exactly one entry with payload type T");
        const T* out = nullptr;
        for_each([&](const auto& e) {
            if constexpr (std::same_as<typename std::remove_cvref_t<decltype(e)>::payload_type, T>) {
                out = &e.payload;
            }
        });
        return out;
    }

    /**
     * @brief Recover a typed pointer to a named entry's wrapped user value.
     *
     * @tparam T Concrete user type.
     * @param name Entry name.
     * @return Pointer to the wrapped user value, or `nullptr` if the name is unknown or the type does
     * not match.
     */
    template <typename T>
    [[nodiscard]] T* get(std::string_view name) noexcept {
        T* out = nullptr;
        if (const auto idx = find(name)) {
            visit_at(*idx, [&](auto& e) {
                if constexpr (std::same_as<typename std::remove_cvref_t<decltype(e)>::payload_type, T>) {
                    out = &e.payload;
                }
            });
        }
        return out;
    }

    /**
     * @brief Const overload of get(std::string_view).
     */
    template <typename T>
    [[nodiscard]] const T* get(std::string_view name) const noexcept {
        const T* out = nullptr;
        if (const auto idx = find(name)) {
            visit_at(*idx, [&](const auto& e) {
                if constexpr (std::same_as<typename std::remove_cvref_t<decltype(e)>::payload_type, T>) {
                    out = &e.payload;
                }
            });
        }
        return out;
    }

protected:
    /// Number of entries whose payload has type `T`.
    template <typename T>
    static constexpr std::size_t count_payload() noexcept {
        return ((std::same_as<typename Entries::payload_type, T> ? std::size_t { 1 } : std::size_t { 0 }) + ...
            + std::size_t { 0 });
    }

    /**
     * @brief Serialize all entries, keyed by name, via the ADL hook `%simplemc_save`.
     *
     * @tparam S Serializer type.
     * @param s Serializer handle to write into.
     */
    template <typename S>
    void save_entries(S& s) const {
        for_each([&](const auto& e) {
            auto sub = s[e.name];
            simplemc_save(sub, e);
        });
    }

    /**
     * @brief Deserialize all entries via the ADL hook `%simplemc_load`.
     *
     * @details It throws a simplemc::simplemc_exception when an entry's name is missing from the
     * serialized data.
     *
     * @tparam S Serializer type.
     * @param s Serializer handle to read from.
     * @param kind Entity word used in the error message (e.g. "update").
     */
    template <typename S>
    void load_entries(const S& s, std::string_view kind) {
        for_each([&](auto& e) {
            if (!s.has(e.name)) {
                throw simplemc_exception(fmt::format("{} '{}' not found in serialized data", kind, e.name));
            }
            const auto sub = s[e.name];
            simplemc_load(sub, e);
        });
    }

    /**
     * @brief Serialize the user-input config of all entries via `%simplemc_save_input_config`.
     *
     * @tparam S Serializer type.
     * @param s Serializer handle to write into.
     */
    template <typename S>
    void save_input_config_entries(S& s) const {
        for_each([&](const auto& e) {
            auto sub = s[e.name];
            simplemc_save_input_config(sub, e);
        });
    }

    /**
     * @brief Deserialize the user-input config of all entries via `%simplemc_load_input_config`.
     *
     * @details It throws a simplemc::simplemc_exception when an entry's name is missing from the
     * serialized data.
     *
     * @tparam S Serializer type.
     * @param s Serializer handle to read from.
     * @param kind Entity word used in the error message (e.g. "update").
     */
    template <typename S>
    void load_input_config_entries(const S& s, std::string_view kind) {
        for_each([&](auto& e) {
            if (!s.has(e.name)) {
                throw simplemc_exception(fmt::format("{} '{}' not found in input config", kind, e.name));
            }
            const auto sub = s[e.name];
            simplemc_load_input_config(sub, e);
        });
    }

    /**
     * @brief Collect all entries from different MPI processes via the ADL hook `%simplemc_mpi_collect`.
     *
     * @param comm simplemc::mpi::communicator object.
     */
    void mpi_collect_entries(const mpi::communicator& comm) {
        for_each([&](auto& e) { simplemc_mpi_collect(comm, e); });
    }

protected:
    /// Tuple holding the stored entries, in order.
    std::tuple<Entries...> entries_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_TUPLE_SET_HPP
