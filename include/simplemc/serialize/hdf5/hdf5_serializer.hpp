/**
 * @file
 * @brief HDF5 serializer for simplemc-serialize-hdf5.
 */

#ifndef SIMPLEMC_SERIALIZE_HDF5_HDF5_SERIALIZER_HPP
#define SIMPLEMC_SERIALIZE_HDF5_HDF5_SERIALIZER_HPP

#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/hdf5/concepts.hpp>
#include <simplemc/serialize/hdf5/file_io.hpp>
#include <simplemc/serialize/hdf5/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>
#include <fmt/std.h>
#include <highfive/eigen.hpp>
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
 * `%simplemc_save(hdf5_serializer, const T&)` if such an overload is reachable, falling back to
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
     * - If an ADL `%simplemc_save(hdf5_serializer, const T&)` is reachable for `T`, descend into a
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
    template <typename T>
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
    template <typename T>
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
    template <typename T>
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
