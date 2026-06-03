/**
 * @file
 * @brief JSON serializer for simplemc-serialize-json.
 */

#ifndef SIMPLEMC_SERIALIZE_JSON_JSON_SERIALIZER_HPP
#define SIMPLEMC_SERIALIZE_JSON_JSON_SERIALIZER_HPP

#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json/file_io.hpp>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-serialize-json
 * @{
 */

/**
 * @brief JSON serializer.
 *
 * @details The JSON serializer satisfies the simplemc::serializer concept. It owns the underlying
 * `nlohmann::json` document via a `std::shared_ptr` and tracks the current position as a JSON
 * pointer (path from root, RFC 6901). Navigation via `operator[]` returns a new `json_serializer`
 * instance sharing the document but pointing at the sub-node — there is no stack and no RAII scope.
 *
 * Serialization is done via save_at() which performs the following dispatch:
 * - If ADL finds an overload of `simplemc_save(json_serializer&, const T&)` for `T`, it is invoked
 *   on a sub-serializer at the given key (created if needed).
 * - Otherwise, the value is written directly via nlohmann assignment, triggering `to_json` /
 *   `adl_serializer<T>` if defined.
 *
 * Two raw-access points are provided: root() returns the document root regardless of the current
 * position, while get() returns the node this (sub-)serializer is positioned at. For a top-level
 * serializer the two are the same.
 *
 * @note Reference semantics on copy are being used, i.e. a copy will refer to the same underlying
 * `nlohmann::json` document. Users who need a fresh document should default-construct a new
 * serializer object.
 */
class json_serializer {
public:
    /// File path type.
    using file_handle = std::filesystem::path;
    /// Configuration options type (see @ref json_io_options).
    using options = json_io_options;

    /**
     * @brief Construct a serializer with an empty JSON object as the document root.
     *
     * @param opts JSON IO options to use when writing the document to file (default: text mode).
     */
    explicit json_serializer(json_io_options opts = {}) :
        opts_ { std::move(opts) },
        root_ { std::make_shared<nlohmann::json>() },
        current_ {} {}

    /**
     * @brief Write `v` at sub-key `key` of the current position; return `*this` for chaining.
     *
     * @details If `T` has an ADL `simplemc_save`, this descends into the sub-position (creating it
     * if needed) and dispatches. Otherwise the value is written directly via nlohmann assignment,
     * triggering `to_json` / `adl_serializer<T>` if defined.
     *
     * @tparam T Value type being written.
     * @param key Sub-key (relative to the current position) to write at.
     * @param v Value to write.
     * @return `*this`, to satisfy the simplemc::serializer concept.
     */
    template <class T>
    json_serializer save_at(std::string_view key, const T& v) {
        if constexpr (has_simplemc_save<T, json_serializer>) {
            auto sub = (*this)[key];
            simplemc_save(sub, v);
        } else {
            (*root_)[current_ / std::string { key }] = v;
        }
        return *this;
    }

    /**
     * @brief Return a new serializer pointing at sub-key `key`, sharing the document.
     *
     * @details The sub-node is not materialised eagerly; it will be auto-created (as an object) by
     * nlohmann on the first write through the returned serializer. A sub-serializer that is never
     * written into leaves the underlying node as JSON `null`. The returned serializer carries the
     * same options.
     *
     * @param key Sub-key (relative to the current position) to descend into.
     * @return Sub-serializer pointing at `key`.
     */
    json_serializer operator[](std::string_view key) {
        return json_serializer { root_, current_ / std::string { key }, opts_ };
    }

    /**
     * @brief Test whether the current position contains sub-key `key`.
     *
     * @param key Sub-key to check.
     * @return `true` if `key` is present at the current position.
     */
    [[nodiscard]] bool has(std::string_view key) const {
        return root_->contains(current_ / std::string { key });
    }

    /**
     * @brief Write the full document to a file.
     *
     * @details Always writes from the root (regardless of which sub-serializer this is called on),
     * since the document is shared. Uses the configured `options::mode` and `options::indent`.
     *
     * @param path Destination file path.
     */
    void write_to_file(const std::filesystem::path& path) const {
        simplemc::write_json_file(*root_, path, opts_);
    }

    /**
     * @brief Mutable access to the document root, regardless of current position.
     *
     * @return Reference to the underlying `nlohmann::json` document root.
     */
    nlohmann::json& root() noexcept { return *root_; }

    /**
     * @brief Read-only access to the document root, regardless of current position.
     *
     * @return Const reference to the underlying `nlohmann::json` document root.
     */
    [[nodiscard]] const nlohmann::json& root() const noexcept { return *root_; }

    /**
     * @brief Mutable access to the node this (sub-)serializer is positioned at.
     *
     * @details For a top-level serializer this is the same as root(); for a sub-serializer
     * returned by `operator[]` it is the sub-node.
     *
     * @return Reference to the `nlohmann::json` node at the current position.
     */
    nlohmann::json& get() { return (*root_)[current_]; }

    /**
     * @brief Read-only access to the node this (sub-)serializer is positioned at.
     *
     * @return Const reference to the `nlohmann::json` node at the current position.
     */
    [[nodiscard]] const nlohmann::json& get() const { return root_->at(current_); }

    /**
     * @brief Logical position of this (sub-)serializer as a JSON pointer (RFC 6901).
     *
     * @details For a top-level serializer this is the empty pointer `""` (the document root);
     * for a sub-serializer returned by `operator[]` it is the path from the root to the sub-node.
     *
     * @return Const reference to the underlying `nlohmann::json::json_pointer`.
     */
    [[nodiscard]] const nlohmann::json::json_pointer& json_ptr() const noexcept { return current_; }

private:
    /// Sub-serializer constructor — shares `root`, points at `current`.
    json_serializer(std::shared_ptr<nlohmann::json> root, nlohmann::json::json_pointer current,
                    json_io_options opts) :
        opts_ { std::move(opts) },
        root_ { std::move(root) },
        current_ { std::move(current) } {}

    json_io_options opts_ {};
    std::shared_ptr<nlohmann::json> root_;
    nlohmann::json::json_pointer current_;
};

// Check serializer concept.
static_assert(serializer<json_serializer>);

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_JSON_JSON_SERIALIZER_HPP
