/**
 * @file
 * @brief JSON read-side deserializer for simplemc-serialize-json.
 */

#ifndef SIMPLEMC_SERIALIZE_JSON_JSON_DESERIALIZER_HPP
#define SIMPLEMC_SERIALIZE_JSON_JSON_DESERIALIZER_HPP

#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json/file_io.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>
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
 * @brief JSON read-side deserializer.
 *
 * @details Mirror of @ref json_serializer for the read direction. Owns a `std::shared_ptr` to the
 * loaded `nlohmann::json` document and tracks the current position as a JSON pointer (path from
 * root, RFC 6901). Navigation via `operator[]` returns a new instance sharing the document;
 * missing keys throw.
 *
 * Per-type customization is via ADL `simplemc_load(const json_deserializer&, T&)`. Types without
 * such a function fall through to `current.get_to(v)`, which dispatches to nlohmann's `from_json`
 * / `adl_serializer<T>` machinery.
 *
 * Two raw-access points are provided: root() returns the document root regardless of the current
 * position, while get() returns the node this (sub-)deserializer is positioned at. For a top-level
 * deserializer the two are the same.
 */
class json_deserializer {
public:
    /// File path type.
    using file_handle = std::filesystem::path;
    /// Configuration options type (see @ref json_io_options).
    using options = json_io_options;

    /**
     * @brief Construct by reading a JSON file from `path`.
     *
     * @param path Source file path.
     * @param opts JSON IO options (defaults to text mode).
     */
    explicit json_deserializer(const file_handle& path, options opts = {}) :
        root_ { std::make_shared<nlohmann::json>() },
        current_ {} {
        simplemc::read_json_file(*root_, path, opts);
    }

    /**
     * @brief Construct from an in-memory `nlohmann::json` document.
     *
     * @param doc JSON document to take ownership of (moved in).
     * @return A deserializer pointing at the root of `doc`.
     */
    [[nodiscard]] static json_deserializer from_json(nlohmann::json doc) {
        auto shared = std::make_shared<nlohmann::json>(std::move(doc));
        return json_deserializer { std::move(shared), nlohmann::json::json_pointer {} };
    }

    /**
     * @brief Read sub-key `key` of the current position into `v`; return `*this` for chaining.
     *
     * @details If `T` has an ADL `simplemc_load`, this descends into the sub-position and
     * dispatches. Otherwise the value is read via `nlohmann::json::get_to`, triggering
     * `from_json` / `adl_serializer<T>` if defined.
     *
     * Throws simplemc::simplemc_exception via nlohmann's `at()` if the key is missing.
     *
     * @tparam T Value type being read into.
     * @param key Sub-key (relative to the current position) to read from.
     * @param v Output value to read into.
     * @return `*this`, to satisfy the simplemc::deserializer concept.
     */
    template <class T>
    json_deserializer load_at(std::string_view key, T& v) const {
        if constexpr (has_simplemc_load<T, json_deserializer>) {
            const auto sub = (*this)[key];
            simplemc_load(sub, v);
        } else {
            root_->at(current_ / std::string { key }).get_to(v);
        }
        return *this;
    }

    /**
     * @brief Return a new deserializer pointing at sub-key `key`, sharing the document.
     *
     * @details Throws simplemc::simplemc_exception if `key` is absent. Use @ref has to check
     * presence first when the key is optional.
     *
     * @param key Sub-key (relative to the current position) to descend into.
     * @return Sub-deserializer pointing at `key`.
     */
    json_deserializer operator[](std::string_view key) const {
        auto sub_ptr = current_ / std::string { key };
        if (!root_->contains(sub_ptr)) {
            throw simplemc_exception(fmt::format("missing key: '{}'", key));
        }
        return json_deserializer { root_, std::move(sub_ptr) };
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
     * @brief Read-only access to the document root, regardless of current position.
     *
     * @return Const reference to the underlying `nlohmann::json` document root.
     */
    [[nodiscard]] const nlohmann::json& root() const noexcept { return *root_; }

    /**
     * @brief Read-only access to the node this (sub-)deserializer is positioned at.
     *
     * @details For a top-level deserializer this is the same as root(); for a sub-deserializer
     * returned by `operator[]` it is the sub-node.
     *
     * @return Const reference to the `nlohmann::json` node at the current position.
     */
    [[nodiscard]] const nlohmann::json& get() const { return root_->at(current_); }

    /**
     * @brief Logical position of this (sub-)deserializer as a JSON pointer (RFC 6901).
     *
     * @details For a top-level deserializer this is the empty pointer `""` (the document root);
     * for a sub-deserializer returned by `operator[]` it is the path from the root to the sub-node.
     *
     * @return Const reference to the underlying `nlohmann::json::json_pointer`.
     */
    [[nodiscard]] const nlohmann::json::json_pointer& json_ptr() const noexcept { return current_; }

private:
    /// Sub-deserializer constructor — shares `root`, points at `current`.
    json_deserializer(std::shared_ptr<nlohmann::json> root, nlohmann::json::json_pointer current) :
        root_ { std::move(root) },
        current_ { std::move(current) } {}

    std::shared_ptr<nlohmann::json> root_;
    nlohmann::json::json_pointer current_;
};

// Check deserializer concept.
static_assert(deserializer<json_deserializer>);

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_JSON_JSON_DESERIALIZER_HPP
