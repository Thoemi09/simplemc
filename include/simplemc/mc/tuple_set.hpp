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
#include <unordered_set>
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
 * the set of entry *types* is fixed at construction.
 *
 * Each entry `e` is required to expose a unique name via `e.name()`, the wrapped user value via
 * `e.value()`, and its type via the nested alias `e::value_type`. The name is used as the key for
 * lookup and serialization. The constructor enforces its uniqueness across the set.
 *
 * Two traversal primitives carry the whole design:
 *
 * - for_each() applies a callable to *every* entry (a compile-time loop over the tuple), and
 * - visit_at() applies a callable to the *single* entry at a runtime index or to the entry with a
 * given registered name.
 *
 * Both accept a generic callable (e.g. `[](auto& e){ ... }`) since every entry is a different type.
 *
 * @tparam Entries Entry types.
 */
template <typename... Entries>
class tuple_set {
public:
    /**
     * @brief Tuple type holding the stored entries, in order.
     */
    using tuple_type = std::tuple<Entries...>;

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
    [[nodiscard]] static constexpr bool empty() noexcept { return size() == 0; }

    /**
     * @brief Construct the set from its entries.
     *
     * @details The entries are moved into the underlying tuple. With no arguments the set is empty.
     *
     * It throws a simplemc::simplemc_exception when two entries share a name.
     *
     * @param es Entries to store, in order.
     */
    explicit tuple_set(Entries... es) : entries_ { std::move(es)... } { ensure_unique_names(); }

    /**
     * @brief Apply a callable to every entry, in order.
     *
     * @tparam F Callable type.
     * @param f Callable to apply (invocable as `f(entry&)` for every entry type).
     */
    template <typename F>
    void for_each(F&& f) { // NOLINT (callable invoked in place, not forwarded)
        std::apply([&](auto&... es) { (f(es), ...); }, entries_);
    }

    /**
     * @brief Apply a callable to every entry, in order.
     *
     * @tparam F Callable type.
     * @param f Callable to apply (invocable as `f(const entry&)` for every entry type).
     */
    template <typename F>
    void for_each(F&& f) const { // NOLINT (callable invoked in place, not forwarded)
        std::apply([&](const auto&... es) { (f(es), ...); }, entries_);
    }

    /**
     * @brief Apply a callable to the single entry at a runtime index.
     *
     * @details Only the matching entry is invoked. The dispatch is O(1) through a per-instantiation
     * function-pointer table.
     *
     * The index must be in range, i.e. `i < size()`. It is the caller's responsibility to validate
     * it. The callable must return `void`.
     *
     * @tparam F Callable type.
     * @param i Entry index (must be `< size()`).
     * @param f Callable to apply to the i-th entry (invocable as `f(entry&)` for every entry type).
     */
    template <typename F>
    void visit_at(std::size_t i, F&& f) { // NOLINT (callable invoked in place, not forwarded)
        using fn_t = void (*)(std::tuple<Entries...>&, F&);
        static constexpr auto table = []<std::size_t... I>(std::index_sequence<I...>) {
            return std::array<fn_t, sizeof...(Entries)> { [](std::tuple<Entries...>& es, F& g) {
                g(std::get<I>(es));
            }... };
        }(std::index_sequence_for<Entries...> {});
        table[i](entries_, f);
    }

    /**
     * @brief Apply a callable to the single entry at a runtime index.
     *
     * @details See the non-const visit_at() overload for details.
     *
     * @tparam F Callable type.
     * @param i Entry index (must be `< size()`).
     * @param f Callable to apply to the i-th entry (invocable as `f(const entry&)` for every entry
     * type).
     */
    template <typename F>
    void visit_at(std::size_t i, F&& f) const { // NOLINT (callable invoked in place, not forwarded)
        using fn_t = void (*)(const std::tuple<Entries...>&, F&);
        static constexpr auto table = []<std::size_t... I>(std::index_sequence<I...>) {
            return std::array<fn_t, sizeof...(Entries)> { [](const std::tuple<Entries...>& es, F& g) {
                g(std::get<I>(es));
            }... };
        }(std::index_sequence_for<Entries...> {});
        table[i](entries_, f);
    }

    /**
     * @brief Apply a callable to the single entry with a given registered name.
     *
     * @details It looks up the name via find() and dispatches to the index-based visit_at() overload.
     * A failed lookup throws a simplemc::simplemc_exception when no entry has the given name.
     *
     * @tparam F Callable type.
     * @param name Entry name.
     * @param f Callable to apply to the named entry.
     */
    template <typename F>
    void visit_at(std::string_view name, F&& f) { // NOLINT (callable invoked in place, not forwarded)
        const auto idx = find(name);
        if (!idx) {
            throw simplemc_exception(fmt::format("no entry named '{}' in the set", name));
        }
        visit_at(*idx, f);
    }

    /**
     * @brief Apply a callable to the single entry with a registered name.
     *
     * @details See the non-const visit_at(std::string_view, F&&) overload for details.
     *
     * @tparam F Callable type.
     * @param name Entry name.
     * @param f Callable to apply to the named entry.
     */
    template <typename F>
    void visit_at(std::string_view name, F&& f) const { // NOLINT (callable invoked in place, not forwarded)
        const auto idx = find(name);
        if (!idx) {
            throw simplemc_exception(fmt::format("no entry named '{}' in the set", name));
        }
        visit_at(*idx, f);
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
            (void)(((std::get<I>(entries_).name() == name) ? (idx = I, true) : false) || ...);
            return idx;
        }(std::index_sequence_for<Entries...> {});
    }

    /**
     * @brief Access the entry at a compile-time index.
     *
     * @tparam I Entry index.
     * @return Reference to the i-th entry (a concrete entry type).
     */
    template <std::size_t I>
    [[nodiscard]] auto& get() noexcept {
        return std::get<I>(entries_);
    }

    /**
     * @brief Access the entry at a compile-time index.
     *
     * @tparam I Entry index.
     * @return Const reference to the i-th entry (a concrete entry type).
     */
    template <std::size_t I>
    [[nodiscard]] const auto& get() const noexcept {
        return std::get<I>(entries_);
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
                if constexpr (std::same_as<typename std::remove_cvref_t<decltype(e)>::value_type, T>) {
                    out = &e.value();
                }
            });
        }
        return out;
    }

    /**
     * @brief Recover a typed pointer to a named entry's wrapped user value.
     *
     * @tparam T Concrete user type.
     * @param name Entry name.
     * @return Const pointer to the wrapped user value, or `nullptr` if the name is unknown or the
     * type does not match.
     */
    template <typename T>
    [[nodiscard]] const T* get(std::string_view name) const noexcept {
        const T* out = nullptr;
        if (const auto idx = find(name)) {
            visit_at(*idx, [&](const auto& e) {
                if constexpr (std::same_as<typename std::remove_cvref_t<decltype(e)>::value_type, T>) {
                    out = &e.value();
                }
            });
        }
        return out;
    }

private:
    // It throws a simplemc::simplemc_exception when two entries share a name.
    void ensure_unique_names() const {
        std::unordered_set<std::string_view> names {};
        names.reserve(size());
        for_each([&](const auto& e) {
            if (auto [it, inserted] = names.insert(e.name()); !inserted) {
                throw simplemc_exception(fmt::format("duplicate entry name '{}' in set", e.name()));
            }
        });
    }

protected:
    /**
     * @brief Serialize all entries, keyed by name, via the ADL hook `%simplemc_save`.
     *
     * @tparam S Serializer type.
     * @param s Serializer handle to write into.
     */
    template <typename S>
    void save_entries(S& s) const {
        for_each([&](const auto& e) { simplemc_save(s[e.name()], e); });
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
            if (!s.has(e.name())) {
                throw simplemc_exception(fmt::format("{} '{}' not found in serialized data", kind, e.name()));
            }
            const auto sub = s[e.name()];
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
        for_each([&](const auto& e) { simplemc_save_input_config(s[e.name()], e); });
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
            if (!s.has(e.name())) {
                throw simplemc_exception(fmt::format("{} '{}' not found in input config", kind, e.name()));
            }
            const auto sub = s[e.name()];
            simplemc_load_input_config(sub, e);
        });
    }

    /**
     * @brief Collect all entries from different MPI processes via the ADL hook
     * `%simplemc_mpi_collect`.
     *
     * @param comm simplemc::mpi::communicator object.
     */
    void mpi_collect_entries(const mpi::communicator& comm) {
        for_each([&](auto& e) { simplemc_mpi_collect(comm, e); });
    }

protected:
    /// Tuple holding the stored entries, in order.
    tuple_type entries_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_TUPLE_SET_HPP
