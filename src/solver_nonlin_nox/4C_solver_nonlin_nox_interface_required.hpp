// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_INTERFACE_REQUIRED_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_INTERFACE_REQUIRED_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_solver_nonlin_nox_interface_required_base.hpp"
#include "4C_utils_exceptions.hpp"

#include <NOX_Abstract_Vector.H>

#include <set>
#include <vector>

namespace NOX
{
  namespace Abstract
  {
    class Group;
  }  // namespace Abstract
}  // namespace NOX

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    namespace Interface
    {
      class Required : public RequiredBase
      {
       public:
        //! Constructor
        Required() {};
        //! Return the desired norm of the residual of the primary DoF fields
        virtual double get_primary_rhs_norms(const Core::LinAlg::Vector<double>& F,
            const NOX::Nln::StatusTest::QuantityType& checkQuantity,
            const ::NOX::Abstract::Vector::NormType& type = ::NOX::Abstract::Vector::TwoNorm,
            const bool& isScaled = false) const = 0;

        //! Returns the Root Mean Squares (abbr.: RMS) of the primary solution updates
        virtual double get_primary_solution_update_rms(const Core::LinAlg::Vector<double>& xNew,
            const Core::LinAlg::Vector<double>& xOld, const double& aTol, const double& rTol,
            const NOX::Nln::StatusTest::QuantityType& checkQuantity,
            const bool& disable_implicit_weighting = false) const = 0;

        //! Return the desired norm of the increment of the primary DoF fields
        virtual double get_primary_solution_update_norms(const Core::LinAlg::Vector<double>& xNew,
            const Core::LinAlg::Vector<double>& xOld,
            const NOX::Nln::StatusTest::QuantityType& checkQuantity,
            const ::NOX::Abstract::Vector::NormType& type = ::NOX::Abstract::Vector::TwoNorm,
            const bool& isScaled = false) const = 0;

        //! Return the desired norm of the solution of primary DoF fields from previous iteration
        virtual double get_previous_primary_solution_norms(const Core::LinAlg::Vector<double>& xOld,
            const NOX::Nln::StatusTest::QuantityType& checkQuantity,
            const ::NOX::Abstract::Vector::NormType& type = ::NOX::Abstract::Vector::TwoNorm,
            const bool& isScaled = false) const = 0;

        //! calculate characteristic/reference norms for forces
        virtual double calc_ref_norm_force() = 0;

        //! create a backup state (optional)
        virtual void create_backup_state(const Core::LinAlg::Vector<double>& dir)
        {
          FOUR_C_THROW("There is no meaningful implementation for this method!");
        }

        //! recover from a backup state (optional)
        virtual void recover_from_backup_state()
        {
          FOUR_C_THROW("There is no meaningful implementation for this method!");
        }

        //! access dofs of specific elements (optional)
        virtual void get_dofs_from_elements(
            const std::vector<int>& my_ele_gids, std::set<int>& my_ele_dofs) const
        {
          FOUR_C_THROW("There is no meaningful implementation for this method!");
        };
      };
    }  // namespace Interface
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
