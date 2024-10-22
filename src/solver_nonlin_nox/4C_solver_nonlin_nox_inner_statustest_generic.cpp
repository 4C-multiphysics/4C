// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_inner_statustest_generic.hpp"  // class definition

#include <iomanip>

FOUR_C_NAMESPACE_OPEN

std::ostream& NOX::Nln::Inner::StatusTest::operator<<(std::ostream& os, StatusType type)
{
  os << std::setiosflags(std::ios::left) << std::setw(13) << std::setfill('.');
  switch (type)
  {
    case status_failed:
      os << "Failed";
      break;
    case status_converged:
      os << "Converged";
      break;
    case status_unevaluated:
      os << "??";
      break;
    case status_no_descent_direction:
    case status_step_too_short:
    case status_step_too_long:
    default:
      os << "**";
      break;
  }
  os << std::resetiosflags(std::ios::adjustfield) << std::setfill(' ');
  return os;
}

FOUR_C_NAMESPACE_CLOSE
