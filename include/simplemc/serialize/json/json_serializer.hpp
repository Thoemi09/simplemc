/**
 * @file
 * @brief JSON write-side serializer. Owns a shared `nlohmann::json` tree; `operator[]` returns a new
 * `json_serializer` sharing the tree but pointing at a sub-node.
 */

#ifndef SIMPLEMC_SERIALIZE_JSON_JSON_SERIALIZER_HPP
#define SIMPLEMC_SERIALIZE_JSON_JSON_SERIALIZER_HPP

#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json/file_io.hpp>
#include <simplemc/serialize/json/serializers.hpp>

#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace simplemc {

/**
 * @addtogroup simplemc-serialize-json
 * @{
 */

/**
 * @brief Configuration options for @ref json_serializer.
 */
struct json_serializer_options {
    /// Indentation width when writing as text (0 for compact output).
    int indent = 0;
    /// Binary mode used when @c text_mode is false.
    json_binary_mode binary_mode = json_binary_mode::bson;
    /// When true, `write_to_file` emits text JSON; otherwise binary.
    bool text_mode = true;
};

/**
 * @brief JSON write-side serializer.
 *
 * @details Owns the underlying `nlohmann::json` tree via a `std::shared_ptr` and tracks the current
 * position via a raw pointer into the tree. Navigation via `operator[]` returns a new
 * `json_serializer` instance sharing the tree but pointing at the sub-node — there is no stack and no
 * RAII scope.
 *
 * Per-type customization is via ADL `simplemc_save(json_serializer&, const T&)`. Types without such a
 * function fall through to `(*current)[key] = v`, which dispatches to nlohmann's `to_json` /
 * `adl_serializer<T>` machinery.
 *
 * @note Reference semantics on copy: `auto s2 = s;` aliases the same underlying tree. Users who need
 * a fresh tree default-construct a new `json_serializer`.
 */
class json_serializer {
public:
    /// File path type.
    using file_handle = std::string;
    /// Configuration options type (see @ref json_serializer_options).
    using options = json_serializer_options;

    /// Default-construct a fresh root serializer with default options.
    json_serializer() : tree_ { std::make_shared<nlohmann::json>(nlohmann::json::object()) }, current_ { tree_.get() } {}

    /// Construct a fresh root serializer with the given options.
    explicit json_serializer(options opts) :
        opts_ { opts }, tree_ { std::make_shared<nlohmann::json>(nlohmann::json::object()) },
        current_ { tree_.get() } {}

    /**
     * @brief Write `v` at sub-key `key` of the current position.
     *
     * @details If `T` has an ADL `simplemc_save`, this descends into the sub-position (creating it if
     * needed) and dispatches. Otherwise the value is written directly via nlohmann assignment,
     * triggering `to_json` / `adl_serializer<T>` if defined.
     */
    template <class T>
    void save_at(std::string_view key, const T& v) {
        if constexpr (detail::has_simplemc_save<T, json_serializer>) {
            json_serializer sub = (*this)[key];
            simplemc_save(sub, v);
        } else {
            (*current_)[std::string { key }] = v;
        }
    }

    /**
     * @brief Return a new `json_serializer` pointing at the sub-position `key`, sharing the tree.
     *
     * @details If the sub-node does not exist or is null, it is created as an empty JSON object. The
     * returned serializer carries the same options.
     */
    json_serializer operator[](std::string_view key) {
        auto& sub = (*current_)[std::string { key }];
        if (sub.is_null()) {
            sub = nlohmann::json::object();
        }
        return json_serializer { tree_, &sub, opts_ };
    }

    /**
     * @brief Write the full tree to a file.
     *
     * @details Always writes from the root (regardless of which sub-serializer this is called on),
     * since the tree is shared. Uses the configured `options::text_mode` and `options::indent` /
     * `options::binary_mode`.
     */
    void write_to_file(const file_handle& path) const {
        if (opts_.text_mode) {
            simplemc::write_json_file(*tree_, path, opts_.indent);
        } else {
            simplemc::write_json_file(*tree_, path, opts_.binary_mode);
        }
    }

    /**
     * @brief One-shot helper: serialize `v` into a fresh JSON tree and write it to `path`.
     *
     * @details If `T` has an ADL `simplemc_save`, the customization is invoked on a fresh serializer
     * (writing at the root level). Otherwise the value is assigned to the root via nlohmann, which
     * dispatches to `to_json` / `adl_serializer<T>`.
     */
    template <class T>
    static void save_to_file(const file_handle& path, const T& v, options opts = {}) {
        json_serializer s { opts };
        if constexpr (detail::has_simplemc_save<T, json_serializer>) {
            simplemc_save(s, v);
        } else {
            *s.tree_ = v;
        }
        s.write_to_file(path);
    }

    /// Mutable access to the underlying tree (rarely needed).
    nlohmann::json& tree() { return *tree_; }
    /// Read-only access to the underlying tree (rarely needed).
    const nlohmann::json& tree() const { return *tree_; }

private:
    /// Sub-serializer constructor — shares `tree`, points at `current`.
    json_serializer(std::shared_ptr<nlohmann::json> tree, nlohmann::json* current, options opts) :
        opts_ { opts }, tree_ { std::move(tree) }, current_ { current } {}

    options opts_ {};
    std::shared_ptr<nlohmann::json> tree_;
    nlohmann::json* current_;
};

static_assert(output_serializer<json_serializer>);

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_JSON_JSON_SERIALIZER_HPP
