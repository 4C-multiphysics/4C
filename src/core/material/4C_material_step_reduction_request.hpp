// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MATERIAL_STEP_REDUCTION_REQUEST_HPP
#define FOUR_C_MATERIAL_STEP_REDUCTION_REQUEST_HPP

#include "4C_config.hpp"

#include <mpi.h>

#include <functional>
#include <optional>
#include <string>

FOUR_C_NAMESPACE_OPEN

namespace Core::Mat::StepReduction
{

  /*!
   * \brief Request a retry of the current step with a smaller step size.
   *
   * Materials call this when the current state cannot be evaluated reliably, but a smaller
   * step may make the next attempt valid. This function always throws and does not return.
   *
   * A valid request is only made inside a \c run_and_synchronize_request() call nested in an
   * enabled \c run_and_detect_synchronized_request() call. If called outside such a context, this
   * function throws a regular \c Core::Exception instead, including the provided \p reason.
   *
   * @param reason Short diagnostic text explaining why the material requests a smaller step.
   * @param detailed_reason Optionally, a longer diagnostic error message that is printed instead of
   * \p reason in case the time-step request is invalid. If not given, \p reason is used.
   */
  [[noreturn]] void request(
      const std::string& reason, const std::optional<std::string>& detailed_reason = std::nullopt);

  /*!
   * \brief Run \p evaluation and synchronize rank-local material step-reduction requests.
   *
   * If called inside an enabled \c run_and_detect_synchronized_request(), this function allows
   * material code to make valid \c request() calls inside \p evaluation. If at least one rank does
   * so, the request is synchronized over \p comm and signaled to the enclosing
   * \c run_and_detect_synchronized_request() call.
   *
   * Calling this function outside an enabled \c run_and_detect_synchronized_request() call is valid
   * but does not enable valid \c request() calls and will not enter a collective.
   *
   * @note Use this around the smallest evaluation region that may call material code and is
   * entered by every rank in \p comm.
   *
   * @note Only material step-reduction requests are synchronized here. Other exceptions
   * propagate unchanged.
   *
   * @param comm Communicator over which rank-local requests are synchronized.
   * @param evaluation Evaluation function that may call rank-local material code.
   */
  void run_and_synchronize_request(MPI_Comm comm, const std::function<void()>& evaluation);

  /*!
   * \brief Run \p operation and detect synchronized material step-reduction requests.
   *
   * The \p operation may contain calls to \c run_and_synchronize_request(). If material code
   * requests step reduction in any such call, the synchronized request is detected here and
   * this function returns \c true. If no material requests step reduction, the operation
   * completes normally and this function returns \c false. If \p enabled is \c false, material
   * requests produce a diagnostic error instead. Exceptions unrelated to material step
   * reduction propagate unchanged.
   *
   * @note Use this function around algorithmic steps that may reach material code, at a level that
   * can decide that the current step is reduced and retried.
   *
   * @param enabled Whether the calling algorithm is configured to allow material step-reduction
   * requests. This should be an input parameter of that algorithm.
   * @param operation Algorithmic operation in which synchronized requests should be detected.
   * @return \c true if step reduction was requested, otherwise \c false.
   */
  [[nodiscard]] bool run_and_detect_synchronized_request(
      bool enabled, const std::function<void()>& operation);

}  // namespace Core::Mat::StepReduction

FOUR_C_NAMESPACE_CLOSE

#endif
