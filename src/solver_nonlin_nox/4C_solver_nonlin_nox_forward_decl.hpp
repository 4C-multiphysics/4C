// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_FORWARD_DECL_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_FORWARD_DECL_HPP

#include "4C_config.hpp"

// Do not try to lint the following names. They are all defined by Trilinos. Including this header
// and not using all identifiers will lead to clang-tidy thinking these are our own names.
// NOLINTBEGIN(readability-identifier-naming)


class Epetra_Operator;
class Epetra_RowMatrix;

namespace NOX
{
  class GlobalData;
  class Observer;
  class Utils;
  namespace Abstract
  {
    class Group;
    class Vector;
  }  // namespace Abstract
  namespace Direction
  {
    class Generic;
    class UserDefinedFactory;
  }  // namespace Direction
  namespace Epetra
  {
    class Vector;
    class LinearSystem;
    namespace Interface
    {
      class Jacobian;
      class Preconditioner;
      class Required;
    }  // namespace Interface
  }  // namespace Epetra
  namespace LineSearch
  {
    class Generic;
  }  // namespace LineSearch
  namespace MeritFunction
  {
    class Generic;
  }  // namespace MeritFunction
  namespace Solver
  {
    class Generic;
  }  // namespace Solver
  namespace StatusTest
  {
    class Factory;
    class Generic;
  }  // namespace StatusTest
}  // namespace NOX

// NOLINTEND(readability-identifier-naming)

#endif
