// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_EXCEPTIONS_HPP
#define FOUR_C_UTILS_EXCEPTIONS_HPP

#include "4C_config.hpp"

#include <exception>
#include <format>
#include <memory>
#include <source_location>
#include <string>

FOUR_C_NAMESPACE_OPEN

namespace Core
{
  namespace Internal
  {
    class ExceptionImplementation;

    [[noreturn]] void throw_error(
        const std::source_location& loc, const std::string& formatted_message);

    template <typename... Args>
    [[noreturn]] void constexpr format_and_throw_error(
        const std::source_location& loc, std::format_string<Args...> fmt, Args&&... args)
    {
      throw_error(loc, std::format(std::move(fmt), std::forward<Args>(args)...));
    }
  }  // namespace Internal

  /**
   * @brief Base class for all 4C exceptions.
   *
   * Any exceptions generated directly by 4C will have this or a derived type. This allows to
   * catch 4C exceptions specifically by using this type in the `catch` clause.
   */
  class Exception : public std::exception
  {
   public:
    /**
     * Generate an Exception with the given message. A stacktrace is automatically attached to this
     * Exception.
     */
    explicit Exception(std::string message);

    /**
     * Destructor.
     */
    ~Exception() override;

    /**
     * Return a message that describes what happened.
     */
    [[nodiscard]] const char* what() const noexcept override;

    /**
     * Return a message that describes what happened and includes a stack trace.
     *
     * @note Calling this function can be a lot more expensive than the what() function because the
     * stacktrace needs to be symbolized.
     */
    [[nodiscard]] std::string what_with_stacktrace() const noexcept;

   private:
    /**
     * Pointer to implementation. This technique is used to minimize the footprint of the exception
     * class that is put on the stack.
     */
    std::unique_ptr<Internal::ExceptionImplementation> pimpl_;
  };
}  // namespace Core


/**
 * Throw an error in the form of a Core::Exception.
 *
 * @note Consider using the more expressive FOUR_C_ASSERT and FOUR_C_ASSERT_ALWAYS macros which
 * take a violated assertion as an argument and print it in the error message.
 *
 * This macro takes an error message, which may contain replacement fields for formatting.
 * The format arguments are passed as additional arguments. For example:
 *
 * @code
 *    FOUR_C_THROW("An error occurred in iteration {}.", iter);
 * @endcode
 */
#define FOUR_C_THROW(fmt, ...)            \
  Core::Internal::format_and_throw_error( \
      std::source_location::current(), (fmt)__VA_OPT__(, ) __VA_ARGS__)

/**
 * Assert that @p test is `true`. If not issue an error in the form of a Core::Exception. In
 * contrast to FOUR_C_ASSERT, this macro is *always* active and should therefore be used to
 * check for user-fixable errors. It is not intended to be used for internal consistency checks;
 * these should be done with FOUR_C_ASSERT.
 *
 * This macro takes an error message, which may contain replacement fields for formatting.
 * The format arguments are passed as additional arguments. For example:
 *
 * @code
 *    FOUR_C_ASSERT_ALWAYS(vector.size() == dim, "Vector size {} does not equal dimension {}.",
 *      vector.size(), dim);
 * @endcode
 */
#define FOUR_C_ASSERT_ALWAYS(test, fmt, ...)                                 \
  do                                                                         \
  {                                                                          \
    if (!static_cast<bool>(test))                                            \
    {                                                                        \
      Core::Internal::format_and_throw_error(                                \
          std::source_location::current(), (fmt)__VA_OPT__(, ) __VA_ARGS__); \
    }                                                                        \
  } while (0)


#ifdef FOUR_C_ENABLE_ASSERTIONS

/**
 * Assert that @p test is `true`. If not issue an error in the form of a Core::Exception.
 * This macro is only active if FOUR_C_ENABLE_ASSERTIONS is set and may therefore be used for
 * expensive checks. Use FOUR_C_ASSERT_ALWAYS if you want to evaluate the test in any case.
 *
 * This macro takes an error message, which may contain replacement fields for formatting.
 * The format arguments are passed as additional arguments. For example:
 *
 * @code
 *   FOUR_C_ASSERT(vector.size() == dim, "Vector size {} does not equal dimension {}.",
 *     vector.size(), dim);
 * @endcode
 */
#define FOUR_C_ASSERT FOUR_C_ASSERT_ALWAYS

#else

/**
 * This macro would assert that @p test is true, but only if FOUR_C_ENABLE_ASSERTIONS is set.
 */
#define FOUR_C_ASSERT(test, args...)  \
  do                                  \
  {                                   \
    /* Assertions are not enabled. */ \
  } while (0)

#endif


FOUR_C_NAMESPACE_CLOSE

#endif
