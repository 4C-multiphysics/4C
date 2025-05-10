// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#ifdef FOUR_C_WITH_BACKTRACE
#define BOOST_STACKTRACE_USE_BACKTRACE
#endif

#include <boost/stacktrace.hpp>
#include <mpi.h>

#include <array>
#include <cstdarg>
#include <iostream>
#include <sstream>

FOUR_C_NAMESPACE_OPEN

namespace Core
{
  namespace Internal
  {
    void throw_error(const std::source_location& loc, const std::string& formatted_message)
    {
      int initialized;
      MPI_Initialized(&initialized);
      int myrank = 0;
      if (initialized > 0)
      {
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
      }

      std::stringstream compound_message;
      compound_message << "PROC " << myrank << " ERROR in " << loc.file_name() << ", line "
                       << loc.line() << ":\n";

      compound_message << formatted_message;
      compound_message << "\n------------------\n";

      throw Core::Exception(compound_message.str());
    }

    class ExceptionImplementation
    {
     public:
      /**
       * A user-defined message that explains what happened.
       */
      std::string message;

      /**
       * The generated stacktrace which is used to construct a nice error message when calling
       * what().
       */
      boost::stacktrace::stacktrace stacktrace;

      /**
       * The full message that is returned by what. This message is composed of all the other
       * information stored in this class.
       *
       * @note This needs to be stored here since we only return a `const char*` from what().
       */
      mutable std::string what_message_{};
    };
  }  // namespace Internal

  const char* Exception::what() const noexcept
  {
    pimpl_->what_message_ = pimpl_->message;
    return pimpl_->what_message_.c_str();
  }

  std::string Exception::what_with_stacktrace() const noexcept
  {
    return pimpl_->message + to_string(pimpl_->stacktrace);
  }

  // This number tells the stack trace class to skip a certain number of frames that are introduced
  // by our exception framework itself. Users are not interested in these calls, and providing them
  // exposes unnecessary details.
  constexpr std::size_t skip_frames = 2;

  Exception::Exception(std::string message)
      : pimpl_(new Internal::ExceptionImplementation{
            std::move(message), boost::stacktrace::stacktrace(skip_frames, /*max_depth=*/-1)})
  {
  }

  // Default the destructor here to facilitate an incomplete implementation type as member variable.
  Exception::~Exception() = default;
}  // namespace Core

FOUR_C_NAMESPACE_CLOSE
