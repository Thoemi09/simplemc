/**
 * @file
 * @brief Basic file input/output functions.
 */

#ifndef SIMPLEMC_UTILS_FILE_IO_HPP
#define SIMPLEMC_UTILS_FILE_IO_HPP

#include <cstdio>
#include <string>
#include <string_view>

namespace simplemc {

/**
 * @addtogroup simplemc-utils-fileio
 * @{
 */

namespace detail {

// Open a file and return a raw file pointer.
[[nodiscard]] std::FILE* open_file_raw(std::string_view name, std::string_view mode);

// Close an already opened file given its file pointer.
void close_file_raw(std::FILE* fp, std::string_view name = "");

} // namespace detail

// Forward declaration.
class file_handle;

/**
 * @brief Open a file and return a RAII file handle.
 *
 * @details It simply calls the file_handle::file_handle constructor.
 *
 * See @ref simplemc-utils-fileio for an example.
 *
 * @param name File name.
 * @param mode File mode (see <a href="https://en.cppreference.com/w/cpp/io/c/fopen">std::fopen</a>).
 * @return RAII file handle.
 */
[[nodiscard]] file_handle open_file(std::string_view name, std::string_view mode);

/**
 * @brief RAII wrapper for file handles.
 *
 * @details Automatically closes the file when the object goes out of scope, ensuring exception-safe
 * file handling.
 *
 * Typically created via simplemc::open_file() or constructed directly.
 *
 * See @ref simplemc-utils-fileio for an example.
 */
class file_handle {
public:
    /**
     * @brief Construct a file handle by opening a file.
     *
     * @details Throws a simplemc::simplemc_exception if opening fails.
     *
     * @param name File name.
     * @param mode File mode (see <a href="https://en.cppreference.com/w/cpp/io/c/fopen">std::fopen
     * </a>).
     */
    file_handle(std::string_view name, std::string_view mode);

    /**
     * @brief Construct a file handle from an existing file pointer.
     *
     * @details Takes ownership of the file pointer. The file will be closed when the handle is
     * destroyed (unless it is `nullptr`, `stdout`, `stdin`, or `stderr`).
     *
     * @param fp File pointer to take ownership of.
     * @param name Optional file name for error messages.
     */
    explicit file_handle(std::FILE* fp, std::string_view name = "");

    /**
     * @brief Destructor closes the file.
     *
     * @details It does not throw exceptions. Errors during close are silently ignored to maintain
     * `noexcept` guarantee in destructor.
     */
    ~file_handle() noexcept;

    /// Deleted copy constructor.
    file_handle(const file_handle&) = delete;

    /// Deleted copy assignment operator.
    file_handle& operator=(const file_handle&) = delete;

    /**
     * @brief Move constructor leaves `other` in a valid but unspecified state.
     *
     * @param other File handle to move from.
     */
    file_handle(file_handle&& other) noexcept;

    /**
     * @brief Move assignment operator leaves `other` in a valid but unspecified state.
     *
     * @details The `other` file is closed by calling close().
     *
     * @param other File handle to move from.
     * @return Reference to this object.
     */
    file_handle& operator=(file_handle&& other) noexcept;

    /**
     * @brief Get the underlying file pointer.
     *
     * @return File pointer.
     */
    [[nodiscard]] std::FILE* get() const noexcept { return fp_; }

    /**
     * @brief Get the file name.
     *
     * @return File name.
     */
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    /**
     * @brief Implicit conversion to file pointer.
     *
     * @return File pointer.
     */
    [[nodiscard]] operator std::FILE*() const noexcept { return fp_; }

    /**
     * @brief Explicitly close the file.
     *
     * @details It closes the file associated with the file pointer if it is not equal to `nullptr`,
     * `stdout`, `stdin` or `stderr` and sets the internal pointer to `nullptr`.
     *
     * It throws a simplemc::simplemc_exception if closing the file fails.
     */
    void close();

private:
    std::FILE* fp_ { nullptr };
    std::string name_ {};
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_UTILS_FILE_IO_HPP
