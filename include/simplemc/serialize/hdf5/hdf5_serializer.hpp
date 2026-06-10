/**
 * @file
 * @brief HDF5 serializer for simplemc-serialize-hdf5.
 */

#ifndef SIMPLEMC_SERIALIZE_HDF5_HDF5_SERIALIZER_HPP
#define SIMPLEMC_SERIALIZE_HDF5_HDF5_SERIALIZER_HPP

#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>
#include <fmt/std.h>
#include <highfive/highfive.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-serialize-hdf5
 * @{
 */

// Forward declaration.
class hdf5_serializer;

/**
 * @brief Concept describing value types that can be written with simplemc::hdf5_serializer::save_at.
 *
 * @details This is satisfied when either
 *
 * - (a) type `T` opts into ADL serialization (simplemc::has_simplemc_save with a
 * simplemc::hdf5_serializer, or
 * - (b) type `T` is supported by HighFive's `createDataSet`.
 *
 * The second clause inherits HighFive's own SFINAE constraints, so this concept tracks exactly what
 * hdf5_serializer::save_at accepts at instantiation.
 *
 * @tparam T Value type.
 */
template <class T>
concept hdf5_savable = requires(HighFive::File& f, const T& v) { f.createDataSet(std::string {}, v); } ||
    has_simplemc_save<T, hdf5_serializer>;

/**
 * @brief Concept describing value types that can be read with simplemc::hdf5_serializer::load_at.
 *
 * @details This is satisfied when either
 *
 * - (a) type `T` opts into ADL serialization (simplemc::has_simplemc_load with a
 * simplemc::hdf5_serializer, or
 * - (b) type `T` is supported by HighFive's `HighFive::DataSet::read`.
 *
 * The second clause inherits HighFive's own SFINAE constraints, so this concept tracks exactly what
 * hdf5_serializer::save_at accepts at instantiation.
 *
 * @tparam T Value type.
 */
template <class T>
concept hdf5_loadable = requires(HighFive::DataSet& ds, T& v) { ds.read(v); } || has_simplemc_load<T, hdf5_serializer>;

/**
 * @brief File open mode for simplemc::hdf5_serializer.
 *
 * @details The three modes map onto HighFive's `HighFive::File` access flags:
 *
 * - `read`: Open an existing file in read-only mode. Throws if the file does not exist.
 * - `truncate`: Create a fresh file (truncate if it already exists). The default for the save side.
 * - `append`: Open an existing file for read/write, or create it if missing. Use this to add content
 * to a file that already has structure you want to keep.
 */
enum class hdf5_file_mode { read, truncate, append };

/**
 * @brief Convert a simplemc::hdf5_file_mode to the corresponding HighFive access mode flags.
 *
 * @details Mapping:
 * - `read` → `HighFive::File::ReadOnly`
 * - `truncate` → `HighFive::File::Truncate`
 * - `append` → `HighFive::File::ReadWrite | HighFive::File::Create`
 *
 * @param mode File open mode.
 * @return The corresponding HighFive access mode flags.
 */
inline HighFive::File::AccessMode to_highfive_access_mode(hdf5_file_mode mode) {
    switch (mode) {
        case hdf5_file_mode::read: return HighFive::File::ReadOnly;
        case hdf5_file_mode::truncate: return HighFive::File::Truncate;
        case hdf5_file_mode::append: return HighFive::File::ReadWrite | HighFive::File::Create;
    }
    return HighFive::File::ReadOnly; // unreachable
}

/**
 * @brief Join a POSIX-style HDF5 path with a sub-key.
 *
 * @details The result is always rooted (starts with `/`) and never contains a double slash. The root
 * case (`base` equal to `"/"` or empty) is handled specially so that joining with the root yields
 * `"/<key>"` rather than `"//<key>"`.
 *
 * Examples:
 * - `h5_join_path("/", "a") == "/a"`
 * - `h5_join_path("", "a") == "/a"`
 * - `h5_join_path("/a", "b") == "/a/b"`
 * - `h5_join_path("/a/b", "c") == "/a/b/c"`
 *
 * @param base Existing HDF5 path (rooted or empty). Treated as the root when empty or `"/"`.
 * @param key Sub-key to append. Should not itself contain a leading slash.
 * @return The joined HDF5 path.
 */
inline std::string h5_join_path(std::string_view base, std::string_view key) {
    if (base.empty() || base == "/") {
        return "/" + std::string { key };
    }
    return std::string { base } + "/" + std::string { key };
}

/**
 * @brief Return the parent path of an HDF5 path.
 *
 * @details The parent of a top-level path (e.g. `"/x"`) is the root `"/"`. The parent of the root
 * itself, or of a path containing no slash, is also `"/"`.
 *
 * Examples:
 * - `h5_parent_path("/a/b/c") == "/a/b"`
 * - `h5_parent_path("/x") == "/"`
 * - `h5_parent_path("/") == "/"`
 *
 * @param path POSIX-style HDF5 path.
 * @return The parent path, always rooted.
 */
inline std::string h5_parent_path(const std::string& path) {
    const auto pos = path.rfind('/');
    return (pos == std::string::npos || pos == 0) ? std::string { "/" } : path.substr(0, pos);
}

/**
 * @brief Serializer for the HDF5 backend.
 *
 * @details A simplemc::hdf5_serializer is a lightweight **handle** into a shared `HighFive::File`.
 * The handle carries two pieces of state, which together define everything it does:
 *
 * - The **file** — a `std::shared_ptr<HighFive::File>` to the open HDF5 file. The file itself is
 * never duplicated. Every copy of a serializer, and every sub-serializer produced by operator[](),
 * points at the same underlying file. Writes through one handle are visible through all of them.
 * The file is flushed and closed when the last handle to it goes out of scope.
 * - The **current location** — a POSIX-style HDF5 path (e.g. `/physics/temperature`) recording where
 * inside the file this particular handle is positioned. A freshly constructed serializer is
 * positioned at the root (`/`); a sub-serializer returned by operator[]() is positioned one level
 * deeper.
 *
 * The class satisfies the @ref simplemc::serializer concept:
 *
 * - **Save direction**: `save_at(key, value)` writes into the file via ADL
 * `%simplemc_save(hdf5_serializer&, const T&)` if such an overload is reachable, falling back to
 * `HighFive::File::createDataSet` otherwise. The fallback creates an HDF5 **dataset** at
 * `<current location>/<key>`; the ADL path delegates to the user's overload, which typically descends via
 * operator[]() into an HDF5 **group**.
 * - **Load direction**: `load_at(key, value)` and `try_load_at(key, value)` read from the file via
 * ADL `%simplemc_load(const hdf5_serializer&, T&)` if such an overload is reachable, falling back to
 * `HighFive::DataSet::read` otherwise. The two differ only in how they handle a missing key —
 * `load_at` throws, `try_load_at` returns `false`.
 *
 * Group materialization is lazy: operator[]() never touches the file, and `save_at` only creates the
 * parent group when a leaf dataset is actually written. This mirrors the auto-create-on-write
 * semantics of the JSON backend.
 *
 * `HighFive::Exception` is caught at every public boundary and rethrown as a
 * simplemc::simplemc_exception so callers see a consistent error type.
 */
class hdf5_serializer {
public:
    /**
     * @brief Construct a top-level handle that opens an HDF5 file.
     *
     * @details The new handle is positioned at the file root (`/`). The file is stored in a
     * `std::shared_ptr` shared with every copy of this handle and every sub-serializer produced by
     * operator[](). It is closed when the last handle goes out of scope.
     *
     * Throws simplemc::simplemc_exception if the file cannot be opened in the requested mode.
     *
     * @param fpath Filesystem path of the HDF5 file.
     * @param mode File open mode; defaults to `hdf5_file_mode::truncate` (fresh write).
     */
    explicit hdf5_serializer(const std::filesystem::path& fpath, hdf5_file_mode mode = hdf5_file_mode::truncate) {
        try {
            file_ = std::make_shared<HighFive::File>(fpath.string(), to_highfive_access_mode(mode));
        } catch (const HighFive::Exception& e) {
            throw simplemc_exception(fmt::format("Opening HDF5 file {} failed: {}", fpath, e.what()));
        }
        current_ = "/";
    }

    /**
     * @brief Serialize a given value into the file at `<current location>/<key>`.
     *
     * @details The write target is always *relative to this handle's current location*, never
     * relative to the root.
     *
     * Dispatch rule:
     *
     * - If an ADL `%simplemc_save(hdf5_serializer&, const T&)` is reachable for `T`, descend into a
     * sub-handle and delegate to it. The user overload typically writes named sub-keys, which
     * materialize as datasets/groups under `<current location>/<key>`.
     * - Otherwise, create an HDF5 dataset at `<current location>/<key>` containing `value` via
     * `HighFive::File::createDataSet`. The parent group is auto-created if missing.
     *
     * @tparam T Value type.
     * @param key Sub-key relative to the current location.
     * @param value Value to write.
     * @return A copy of `*this`.
     */
    template <class T>
        requires hdf5_savable<T>
    hdf5_serializer save_at(std::string_view key, const T& value) {
        if constexpr (has_simplemc_save<T, hdf5_serializer>) {
            auto sub = (*this)[key];
            simplemc_save(sub, value);
        } else {
            const auto full_path = h5_join_path(current_, key);
            try {
                const auto parent_path = h5_parent_path(full_path);
                if (parent_path != "/" && !file_->exist(parent_path)) {
                    file_->createGroup(parent_path);
                }
                file_->createDataSet(full_path, value);
            } catch (const HighFive::Exception& e) {
                throw simplemc_exception(fmt::format("save_at(key='{}') failed: {}", key, e.what()));
            }
        }
        return *this;
    }

    /**
     * @brief Deserialize the file at `<current location>/<key>` into a given value.
     *
     * @details The read source is always *relative to this handle's current location*, never relative
     * to the root.
     *
     * Dispatch rule:
     *
     * - If an ADL `%simplemc_load(const hdf5_serializer&, T&)` is reachable for `T`, descend into a
     * sub-handle and delegate to it.
     * - Otherwise, read the HDF5 dataset at `<urrent location>/<key>` into `value` via
     * `HighFive::DataSet::read`.
     *
     * Throws simplemc::simplemc_exception if the key is missing in the file.
     *
     * @tparam T Value type.
     * @param key Sub-key relative to the current location.
     * @param value Value to read into.
     * @return A copy of `*this`.
     */
    template <class T>
        requires hdf5_loadable<T>
    hdf5_serializer load_at(std::string_view key, T& value) const {
        if constexpr (has_simplemc_load<T, hdf5_serializer>) {
            const auto sub = (*this)[key];
            simplemc_load(sub, value);
        } else {
            const auto full_path = h5_join_path(current_, key);
            try {
                file_->getDataSet(full_path).read(value);
            } catch (const HighFive::Exception& e) {
                throw simplemc_exception(fmt::format("load_at(key='{}') failed: {}", key, e.what()));
            }
        }
        return *this;
    }

    /**
     * @brief Try to deserialize the file at `<current location>/<key>` into a given value.
     *
     * @details Same as load_at() except that it silently returns false if `key` is missing,
     * leaving `value` unchanged.
     *
     * @tparam T Value type.
     * @param key Sub-key relative to the current location.
     * @param value Value to read into.
     * @return True if the key was present and the read occurred, false otherwise.
     */
    template <class T>
        requires hdf5_loadable<T>
    bool try_load_at(std::string_view key, T& value) const {
        if (!has(key)) {
            return false;
        }
        load_at(key, value);
        return true;
    }

    /**
     * @brief Return a sub-handle positioned at `<current location>/<key>` (save side).
     *
     * @details Conceptually, this is a "pointer-advance" operation: the returned handle shares
     * the same file as `*this`, but its current location is one level deeper. The file is not
     * modified, and the underlying group at the new location is not materialized eagerly — HDF5
     * groups are created only when something is actually written through the returned handle.
     *
     * @param key Sub-key relative to the current location.
     * @return New sub-handle pointing at `<current location>/<key>`.
     */
    hdf5_serializer operator[](std::string_view key) { return hdf5_serializer { file_, h5_join_path(current_, key) }; }

    /**
     * @brief Return a sub-handle positioned at `<current location>/<key>` (load side).
     *
     * @details Same conceptual operation as the non-const overload operator[]() — share the file,
     * advance the current location and return a new handle.
     *
     * If `<current location>/<key>` does not exist in the file, it throws a
     * simplemc::simplemc_exception. This is the safe default for the load side, where a missing key
     * should fail loudly.
     *
     * Use has() to test presence beforehand when the key is optional.
     *
     * @param key Sub-key relative to the current location.
     * @return New sub-handle pointing at `<current location>/<key>`.
     */
    hdf5_serializer operator[](std::string_view key) const {
        auto sub = h5_join_path(current_, key);
        if (!file_->exist(sub)) {
            throw simplemc_exception(fmt::format("missing key: '{}'", key));
        }
        return hdf5_serializer { file_, std::move(sub) };
    }

    /**
     * @brief Test whether the file contains a group or dataset at `<current location>/<key>`.
     *
     * @param key Sub-key relative to the current location.
     * @return True if `<current location>/<key>` is present in the file.
     */
    [[nodiscard]] bool has(std::string_view key) const { return file_->exist(h5_join_path(current_, key)); }

    /**
     * @brief Mutable access to the underlying HighFive file.
     *
     * @return Reference to the `HighFive::File`.
     */
    HighFive::File& file() noexcept { return *file_; }

    /**
     * @brief Read-only access to the underlying HighFive file.
     *
     * @return Const reference to the `HighFive::File`.
     */
    [[nodiscard]] const HighFive::File& file() const noexcept { return *file_; }

    /**
     * @brief The current location of this handle as a POSIX-style HDF5 path.
     *
     * @return View of the current location string.
     */
    [[nodiscard]] std::string_view path() const noexcept { return current_; }

private:
    /// Sub-serializer constructor — shares file, points at current.
    hdf5_serializer(std::shared_ptr<HighFive::File> file, std::string current) :
        file_ { std::move(file) },
        current_ { std::move(current) } {}

private:
    std::shared_ptr<HighFive::File> file_;
    std::string current_;
};

// Check serializer concept.
static_assert(serializer<hdf5_serializer>);

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_HDF5_HDF5_SERIALIZER_HPP
