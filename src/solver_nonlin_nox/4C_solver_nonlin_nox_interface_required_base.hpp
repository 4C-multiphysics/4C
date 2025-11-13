// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_INTERFACE_REQUIRED_BASE_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_INTERFACE_REQUIRED_BASE_HPP

#include "4C_config.hpp"

#include <NOX_Epetra_Interface_Required.H>  // base class

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    namespace Interface
    {
      class RequiredBase : public ::NOX::Epetra::Interface::Required
      {
      };
    }  // namespace Interface
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
