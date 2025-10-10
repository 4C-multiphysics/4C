// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_LINEARSYSTEM_FACTORY_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_LINEARSYSTEM_FACTORY_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_solver_nonlin_nox_forward_decl.hpp"
#include "4C_solver_nonlin_nox_linearsystem_base.hpp"

#include <Teuchos_RCP.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::LinAlg
{
  class Solver;
  class SparseOperator;
}  // namespace Core::LinAlg

namespace NOX
{
  namespace Nln
  {
    class GlobalData;
    class Scaling;

    namespace LinSystem
    {
      class Factory
      {
       public:
        //! Constructor.
        Factory();


        Teuchos::RCP<NOX::Nln::LinearSystemBase> build_linear_system(
            const NOX::Nln::LinSystem::LinearSystemType& linsystype,
            NOX::Nln::GlobalData& noxNlnGlobalData,
            const Teuchos::RCP<Core::LinAlg::SparseOperator>& jac,
            ::NOX::Epetra::Vector& cloneVector,
            const Teuchos::RCP<Core::LinAlg::SparseOperator>& precMat,
            const std::shared_ptr<NOX::Nln::Scaling>& scalingObject) const;
      };

      /*! \brief Nonmember helper function for the NOX::Nln::LinearSystem::Factory.

      \relates NOX::Nln::LinearSystem::Factory

      */
      Teuchos::RCP<NOX::Nln::LinearSystemBase> build_linear_system(
          const NOX::Nln::LinSystem::LinearSystemType& linsystype,
          NOX::Nln::GlobalData& noxNlnGlobalData,
          const Teuchos::RCP<Core::LinAlg::SparseOperator>& jac, ::NOX::Epetra::Vector& cloneVector,
          const Teuchos::RCP<Core::LinAlg::SparseOperator>& precMat,
          const std::shared_ptr<NOX::Nln::Scaling>& scalingObject);
    }  // namespace LinSystem
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
