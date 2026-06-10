/**
 * @file
 * @brief Reusable named-collection base for simplemc::update_set / simplemc::measurement_set.
 */

#ifndef SIMPLEMC_MC_NAMED_SET_HPP
#define SIMPLEMC_MC_NAMED_SET_HPP

#include <simplemc/mpi/communicator.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace simplemc {

/**
 * @addtogroup simplemc-mc
 * @{
 */

/**
 * @brief Ordered collection of named entries shared by the simplemc-mc sets.
 *
 * @details Provides the registration / lookup / typed-access API common to simplemc::update_set and
 * simplemc::measurement_set, plus protected helpers that implement the per-entry serialization and
 * MPI-reduction loops. A derived set inherits the public read API directly and calls the protected
 * helpers from its own `simplemc_save` / `simplemc_load` / `simplemc_mpi_collect` overloads.
 *
 * `Entry` must expose a public `std::string name` member, a typed `get<T>()` accessor, and the
 * per-entity ADL hooks (`simplemc_save`, `simplemc_load`, `simplemc_save_input_config`,
 * `simplemc_load_input_config`, `simplemc_mpi_collect`) used by the loop helpers that the derived
 * set opts into.
 *
 * @tparam Entry Per-entity aggregate type (simplemc::update or simplemc::measurement).
 */
template <class Entry>
class named_set {
public:
    /**
     * @brief Look up an entry by name.
     *
     * @param name Entry name to search for.
     * @return Index of the entry, or `std::nullopt` if no entry has that name.
     */
    [[nodiscard]] std::optional<std::size_t> find(std::string_view name) const noexcept {
        const auto it = std::ranges::find_if(entries_, [name](const Entry& e) { return e.name == name; });
        if (it == entries_.end()) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(std::distance(entries_.begin(), it));
    }

    /**
     * @brief Number of registered entries.
     *
     * @return Entry count.
     */
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    /**
     * @brief Whether the set has no registered entries.
     *
     * @return `true` if empty.
     */
    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }

    /**
     * @brief Mutable access to the i-th entry.
     *
     * @param i Entry index (unchecked).
     * @return Reference to the entry.
     */
    [[nodiscard]] Entry& at(std::size_t i) noexcept { return entries_[i]; }

    /**
     * @brief Const access to the i-th entry.
     *
     * @param i Entry index (unchecked).
     * @return Const reference to the entry.
     */
    [[nodiscard]] const Entry& at(std::size_t i) const noexcept { return entries_[i]; }

    /**
     * @brief Read-only view over all registered entries.
     *
     * @return Span over the entries, in registration order.
     */
    [[nodiscard]] std::span<const Entry> data() const noexcept { return entries_; }

    /**
     * @brief Recover a typed pointer to a registered entry's wrapped user value.
     *
     * @tparam T Concrete user type.
     * @param name Entry name.
     * @return Pointer to the wrapped user value, or `nullptr` if the name is unknown or the type
     *         does not match.
     */
    template <class T>
    [[nodiscard]] T* get(std::string_view name) noexcept {
        const auto idx = find(name);
        return idx ? entries_[*idx].template get<T>() : nullptr;
    }

    /**
     * @brief Const overload of get().
     */
    template <class T>
    [[nodiscard]] const T* get(std::string_view name) const noexcept {
        const auto idx = find(name);
        return idx ? entries_[*idx].template get<T>() : nullptr;
    }

protected:
    /**
     * @brief Register an entry, enforcing name uniqueness within the set.
     *
     * @details Throws simplemc::simplemc_exception if an entry with the same name is already
     * registered; otherwise moves the entry into the collection.
     *
     * @param e Entry to register (moved in).
     * @param kind Entity word used in the error message (e.g. "update").
     */
    void add_checked(Entry e, std::string_view kind) {
        if (find(e.name).has_value()) {
            throw simplemc_exception(fmt::format("{} '{}' is already registered", kind, e.name));
        }
        entries_.push_back(std::move(e));
    }

    /**
     * @brief Serialize all entries into the passed serializer, keyed by name.
     *
     * @tparam S Serializer type.
     * @param s Serializer handle to write into.
     */
    template <class S>
    void save_entries(S& s) const {
        for (const auto& e : entries_) {
            auto entry = s[e.name];
            simplemc_save(entry, e);
        }
    }

    /**
     * @brief Restore all entries from the serializer; throws when a registered name is missing.
     *
     * @tparam S Serializer type.
     * @param s Serializer handle to read from.
     * @param kind Entity word used in the error message (e.g. "update").
     */
    template <class S>
    void load_entries(const S& s, std::string_view kind) {
        for (auto& e : entries_) {
            if (!s.has(e.name)) {
                throw simplemc_exception(fmt::format("{} '{}' not found in serialized data", kind, e.name));
            }
            const auto entry = s[e.name];
            simplemc_load(entry, e);
        }
    }

    /**
     * @brief Serialize the user-input config of all entries, keyed by name.
     *
     * @tparam S Serializer type.
     * @param s Serializer handle to write into.
     */
    template <class S>
    void save_input_config_entries(S& s) const {
        for (const auto& e : entries_) {
            auto entry = s[e.name];
            simplemc_save_input_config(entry, e);
        }
    }

    /**
     * @brief Restore the user-input config of all entries; throws when a registered name is missing.
     *
     * @tparam S Serializer type.
     * @param s Serializer handle to read from.
     * @param kind Entity word used in the error message (e.g. "update").
     */
    template <class S>
    void load_input_config_entries(const S& s, std::string_view kind) {
        for (auto& e : entries_) {
            if (!s.has(e.name)) {
                throw simplemc_exception(fmt::format("{} '{}' not found in input config", kind, e.name));
            }
            const auto entry = s[e.name];
            simplemc_load_input_config(entry, e);
        }
    }

    /**
     * @brief All-reduce every entry across MPI ranks via the per-entity ADL hook.
     *
     * @param comm MPI communicator over which to reduce.
     */
    void mpi_collect_entries(const mpi::communicator& comm) {
        for (auto& e : entries_) {
            simplemc_mpi_collect(comm, e);
        }
    }

    /// The registered entries, in registration order.
    std::vector<Entry> entries_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_NAMED_SET_HPP
