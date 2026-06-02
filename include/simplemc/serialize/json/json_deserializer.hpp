/**
 * @file
 * @brief JSON read-side deserializer for simplemc-serialize-json.
 */

#ifndef SIMPLEMC_SERIALIZE_JSON_JSON_DESERIALIZER_HPP
#define SIMPLEMC_SERIALIZE_JSON_JSON_DESERIALIZER_HPP

#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json/file_io.hpp>
#include <simplemc/serialize/json/serializers.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <cstddef>
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
 * @brief JSON read-side deserializer.
 *
 * @details Mirror of @ref json_serializer for the read direction. Owns a `std::shared_ptr` to the
 * loaded `nlohmann::json` tree and tracks the current position via a `const nlohmann::json*`.
 * Navigation via `operator[]` returns a new instance sharing the tree; missing keys throw.
 *
 * Per-type customization is via ADL `simplemc_load(json_deserializer&, T&)`. Types without such a
 * function fall through to `current.get_to(v)`, which dispatches to nlohmann's `from_json` /
 * `adl_serializer<T>` machinery.
 */
class json_deserializer {
public:
    /// File path type.
    using file_handle = std::string;
    /// Configuration options type (see @ref json_io_options).
    using options = json_io_options;

    /// Construct by reading a JSON file from `path` using `opts` (text mode by default).
    explicit json_deserializer(const file_handle& path, options opts = {}) :
        tree_ { std::make_shared<nlohmann::json>() }, current_ { tree_.get() } {
        simplemc::read_json_file(*tree_, path, opts);
    }

    /// Factory: construct from an in-memory JSON tree (moved in).
    static json_deserializer from_tree(nlohmann::json tree) {
        auto shared = std::make_shared<nlohmann::json>(std::move(tree));
        return json_deserializer { shared, shared.get() };
    }

    /**
     * @brief Read sub-key `key` of the current position into `v`; return `*this` for chaining.
     *
     * @details If `T` has an ADL `simplemc_load`, this descends into the sub-position and dispatches.
     * Otherwise the value is read via `nlohmann::json::get_to`, triggering `from_json` /
     * `adl_serializer<T>` if defined.
     *
     * Throws simplemc::simplemc_exception via nlohmann's `at()` if the key is missing.
     */
    template <class T>
    json_deserializer load_at(std::string_view key, T& v) const {
        if constexpr (has_simplemc_load<T, json_deserializer>) {
            const auto sub = (*this)[key];
            simplemc_load(sub, v);
        } else {
            current_->at(std::string { key }).get_to(v);
        }
        return *this;
    }

    /**
     * @brief Return a new `json_deserializer` pointing at the sub-position `key`, sharing the tree.
     *
     * @details Throws simplemc::simplemc_exception if `key` is absent. Use @ref has to check
     * presence first when the key is optional.
     */
    json_deserializer operator[](std::string_view key) const {
        if (!current_->contains(std::string { key })) {
            throw simplemc_exception(fmt::format("missing key: '{}'", key));
        }
        return json_deserializer { tree_, &current_->at(std::string { key }) };
    }

    /// Test whether the current position contains `key`.
    [[nodiscard]] bool has(std::string_view key) const { return current_->contains(std::string { key }); }

    /// Number of elements in the array at `key`. Throws if the key is missing or not an array.
    [[nodiscard]] std::size_t array_size(std::string_view key) const {
        return current_->at(std::string { key }).size();
    }

    /// Shape of a nested-array structure at `key`. Returns the dimensions in row-major order.
    [[nodiscard]] std::vector<std::size_t> array_shape(std::string_view key) const {
        std::vector<std::size_t> shape;
        const nlohmann::json* p = &current_->at(std::string { key });
        while (p->is_array()) {
            shape.push_back(p->size());
            if (p->empty()) {
                break;
            }
            p = &p->front();
        }
        return shape;
    }

    /**
     * @brief One-shot helper: read `v` from `path`.
     *
     * @details If `T` has an ADL `simplemc_load`, the customization is invoked on a fresh
     * deserializer (reading from the root level). Otherwise the value is read from the root via
     * nlohmann, which dispatches to `from_json` / `adl_serializer<T>`.
     */
    template <class T>
    static void load_from_file(const file_handle& path, T& v, options opts = {}) {
        json_deserializer d { path, opts };
        if constexpr (has_simplemc_load<T, json_deserializer>) {
            simplemc_load(d, v);
        } else {
            d.tree().get_to(v);
        }
    }

    /// Read-only access to the underlying tree.
    [[nodiscard]] const nlohmann::json& tree() const { return *tree_; }

private:
    /// Sub-deserializer constructor — shares `tree`, points at `current`.
    json_deserializer(std::shared_ptr<nlohmann::json> tree, const nlohmann::json* current) :
        tree_ { std::move(tree) }, current_ { current } {}

    std::shared_ptr<nlohmann::json> tree_;
    const nlohmann::json* current_;
};

static_assert(deserializer<json_deserializer>);

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_JSON_JSON_DESERIALIZER_HPP
