/**
 * @file
 * @brief Unified JSON serializer/deserializer for simplemc-serialize-json.
 */

#ifndef SIMPLEMC_SERIALIZE_JSON_JSON_SERIALIZER_HPP
#define SIMPLEMC_SERIALIZE_JSON_JSON_SERIALIZER_HPP

#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

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
 * @brief Serializer/deserializer for the JSON backend.
 *
 * @details A simplemc::json_serializer is a lightweight **handle** into a shared `nlohmann::json`
 * document. The handle carries two pieces of state, which together define everything it does:
 *
 * - The **root document** — a `std::shared_ptr<nlohmann::json>` to the top of the JSON tree. The
 * document itself is never duplicated. Every copy of a serializer, and every sub-serializer produced
 * by operator[](), points at the same underlying document. Mutations through one handle are visible
 * through all of them. The document is destroyed only when the last handle to it goes out of scope.
 * - The **current location** — a `nlohmann::json::json_pointer` (RFC 6901) recording where inside
 * the root document this particular handle is positioned. A freshly constructed serializer is
 * positioned at the document root (the empty pointer `""`); a sub-serializer returned by operator[]()
 * is positioned one level deeper.
 *
 * The class satisfies both the @ref simplemc::serializer and @ref simplemc::deserializer concepts:
 *
 * - **Save direction**: `save_at(key, value)` (non-const) writes into the shared document via ADL
 * `%simplemc_save(json_serializer&, const T&)` if such an overload is reachable, falling back to
 * nlohmann's `to_json` / `adl_serializer<T>` machinery otherwise.
 * - **Load direction**: `load_at(key, value)` (const) reads from the shared document via ADL
 * `%simplemc_load(const json_serializer&, T&)` if such an overload is reachable, falling back to
 * `nlohmann::json::get_to` otherwise.
 *
 * Every operation — save_at(), load_at(), has(), operator[](), get() — is interpreted **relative to
 * the current location**, not relative to the root. For example, given the document
 *
 * ```
 * { "physics": { "temperature": 1.5 } }
 * ```
 *
 * the call `s["physics"].save_at("pressure", 2.0)` proceeds as follows:
 *
 * 1. `s["physics"]` returns a new handle still pointing at the same document, but with its current
 * location advanced from `""` (root) to `"/physics"`. No copying, no materialization of any sub-node;
 * just a pointer-advance operation.
 * 2. `.save_at("pressure", 2.0)` writes `2.0` into the document at location `<current>/<key>` =
 * `"/physics/pressure"`.
 *
 * After the call, the document reads `{ "physics": { "temperature": 1.5, "pressure": 2.0 } }`,
 * visible through both `s` and the temporary sub-handle (had we kept it).
 */
class json_serializer {
public:
    /**
     * @brief Construct a top-level handle that wraps a JSON document.
     *
     * @details The new handle is positioned at the document root (i.e. its current location is the
     * empty JSON pointer `""`). The document is moved into a `std::shared_ptr` that is shared with
     * every copy of this handle and every sub-serializer produced by operator[]().
     *
     * @param doc JSON document to wrap.
     */
    explicit json_serializer(nlohmann::json doc = {}) :
        root_ { std::make_shared<nlohmann::json>(std::move(doc)) },
        current_ {} {}

    /**
     * @brief Serialize a given value into the document at `<current location>/<key>`.
     *
     * @details The write target is always *relative to this handle's current location*, never
     * relative to the root. On a top-level handle, the target is `/<key>` (immediate child of root);
     * on a sub-serializer at `/a/b`, the target is `/a/b/<key>`.
     *
     * Dispatch rule:
     *
     * - If an ADL `%simplemc_save(json_serializer&, const T&)` is reachable for `T`, descend into
     * a sub-handle and delegate to it.
     * - Otherwise, assign directly via `nlohmann::json::operator=`, which triggers nlohmann's
     * `to_json` / `adl_serializer<T>` machinery.
     *
     * Both paths will write the value to `<current location>/<key>`.
     *
     * @tparam T Value type.
     * @param key Sub-key relative to the current location.
     * @param value Value to write.
     * @return A copy of `*this`.
     */
    template <class T>
    json_serializer save_at(std::string_view key, const T& value) {
        if constexpr (has_simplemc_save<T, json_serializer>) {
            auto sub = (*this)[key];
            simplemc_save(sub, value);
        } else {
            (*root_)[current_ / std::string { key }] = value;
        }
        return *this;
    }

    /**
     * @brief Deserialize the document at `<current location>/<key>` into a given value.
     *
     * @details The read source is always *relative to this handle's current location*, never
     * relative to the root. On a top-level handle, the source is `/<key>` (immediate child of root);
     * on a sub-serializer at `/a/b`, the source is `/a/b/<key>`.
     *
     * Dispatch rule:
     *
     * - If an ADL `%simplemc_load(const json_serializer&, T&)` is reachable for `T`, descend into
     * a sub-handle and delegate to it.
     * - Otherwise, read directly via `nlohmann::json::get_to`, which triggers nlohmann's `from_json`
     * / `adl_serializer<T>` machinery.
     *
     * Both paths will try to read from `<current location>/<key>`.
     *
     * It throws an exception if the key is missing in the document.
     *
     * @tparam T Value type.
     * @param key Sub-key relative to the current location.
     * @param value Value to read into.
     * @return A copy of `*this`.
     */
    template <class T>
    json_serializer load_at(std::string_view key, T& value) const {
        if constexpr (has_simplemc_load<T, json_serializer>) {
            const auto sub = (*this)[key];
            simplemc_load(sub, value);
        } else {
            root_->at(current_ / std::string { key }).get_to(value);
        }
        return *this;
    }

    /**
     * @brief Return a sub-handle positioned at `<current location>/<key>` (save side).
     *
     * @details Conceptually, this is a "pointer-advance" operation: the returned handle shares
     * the same root document as `*this`, but its current location is one level deeper. The
     * document is not modified, and the underlying JSON node at the new location is not
     * materialized eagerly — nlohmann auto-creates intermediate objects only when something is
     * actually written through the returned handle.
     *
     * If you never write through the returned handle, the document remains unchanged.
     *
     * @param key Sub-key relative to the current location.
     * @return New sub-handle pointing at `<current location>/<key>`.
     */
    json_serializer operator[](std::string_view key) {
        return json_serializer { root_, current_ / std::string { key } };
    }

    /**
     * @brief Return a sub-handle positioned at `<current location>/<key>` (load side).
     *
     * @details Same conceptual operation as the non-const overload operator[]() — share the document,
     * advance the current location and return a new handle.
     *
     * If `<current location>/<key>` does not exist in the document, it throws a
     * simplemc::simplemc_exception. This is the safe default for the load side, where a missing key
     * should fail loudly.
     *
     * Use has() to test presence beforehand when the key is optional.
     *
     * @param key Sub-key relative to the current location.
     * @return New sub-handle pointing at `<current location>/<key>`.
     */
    json_serializer operator[](std::string_view key) const {
        auto sub_ptr = current_ / std::string { key };
        if (!root_->contains(sub_ptr)) {
            throw simplemc_exception(fmt::format("missing key: '{}'", key));
        }
        return json_serializer { root_, std::move(sub_ptr) };
    }

    /**
     * @brief Test whether the document contains a node at `<current location>/<key>`.
     *
     * @param key Sub-key relative to the current location.
     * @return True if `<current location>/<key>` is present in the document.
     */
    [[nodiscard]] bool has(std::string_view key) const { return root_->contains(current_ / std::string { key }); }

    /**
     * @brief Mutable access to the root document.
     *
     * @return Reference to the underlying `nlohmann::json` document root.
     */
    nlohmann::json& root() noexcept { return *root_; }

    /**
     * @brief Read-only access to the root document.
     *
     * @return Const reference to the underlying `nlohmann::json` document root.
     */
    [[nodiscard]] const nlohmann::json& root() const noexcept { return *root_; }

    /**
     * @brief Mutable access to the JSON node at `<current location>`.
     *
     * @details For a top-level handle (current location = `""`) this is the same as root(). For a
     * sub-serializer returned by operator[](), it is the node at the deeper path.
     *
     * If the current location does not exist in the document, it is auto-created as an empty object.
     *
     * @return Reference to the `nlohmann::json` node at the current location.
     */
    nlohmann::json& get() { return (*root_)[current_]; }

    /**
     * @brief Read-only access to the JSON node at `<current location>`.
     *
     * @details It uses `nlohmann::json::at` to access the JSON node. If the current location does not
     * exist in the document, it throws an exception.
     *
     * @return Const reference to the `nlohmann::json` node at the current location.
     */
    [[nodiscard]] const nlohmann::json& get() const { return root_->at(current_); }

    /**
     * @brief The current location of this handle, as an RFC 6901 JSON pointer.
     *
     * @return Const reference to the underlying `nlohmann::json::json_pointer`.
     */
    [[nodiscard]] const nlohmann::json::json_pointer& json_ptr() const noexcept { return current_; }

private:
    /// Sub-serializer constructor — shares root, points at current.
    json_serializer(std::shared_ptr<nlohmann::json> root, nlohmann::json::json_pointer current) :
        root_ { std::move(root) },
        current_ { std::move(current) } {}

private:
    std::shared_ptr<nlohmann::json> root_;
    nlohmann::json::json_pointer current_;
};

// Check serializer and deserializer concepts.
static_assert(serializer<json_serializer>);
static_assert(deserializer<json_serializer>);

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_JSON_JSON_SERIALIZER_HPP
