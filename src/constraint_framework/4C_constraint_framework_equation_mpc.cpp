// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_constraint_framework_equation_mpc.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_io_pstream.hpp"

FOUR_C_NAMESPACE_OPEN
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Constraints::SUBMODELEVALUATOR::LinearCoupledEquation::evaluate_equation(
    Core::LinAlg::SparseMatrix& Q_dd, Core::LinAlg::SparseMatrix& Q_dL,
    Core::LinAlg::SparseMatrix& Q_Ld, Core::LinAlg::Vector<double>& constraint_vector,
    const Core::LinAlg::Vector<double>& D_np1)
{
  double constraintViolation = 0.;

  // Iterate over the elements (coefficient, rowId, dofId) in equationData.
  // Each element of equation data represents one term of the defined multipoint constraints
  // The rowId is equivalent to the Number of the equation
  for (const auto& [coefficient, rowId, dofId] : equation_data_)
  {
    // stiffness contribution
    Q_dL.assemble(coefficient, dofId, rowId);
    Q_Ld.assemble(coefficient, rowId, dofId);

    // force contribution
    constraintViolation = D_np1.get_values()[dofId] * coefficient;
    constraint_vector.sum_into_global_values(1, &constraintViolation, &rowId);
  }
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Constraints::SUBMODELEVALUATOR::MultiPointConstraintEquationBase::get_number_of_mp_cs() const
{
  return n_dof_coupled_;
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Constraints::SUBMODELEVALUATOR::MultiPointConstraintEquationBase::get_first_row_id() const
{
  return first_row_id_;
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Constraints::SUBMODELEVALUATOR::MultiPointConstraintEquationBase::set_first_row_id(
    int global_row_id)
{
  first_row_id_ = global_row_id;
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Constraints::SUBMODELEVALUATOR::LinearCoupledEquation::LinearCoupledEquation(
    int id, const std::vector<int>& dofs, std::vector<double> coefficients)
{
  Core::IO::cout(Core::IO::debug) << "\nLinear coupled equation saved (ID: " << id << ")\n ";
  Core::IO::cout(Core::IO::debug) << " 0 = ";  // #Todo

  set_first_row_id(id);

  for (std::vector<double>::size_type i = 0; i < coefficients.size(); ++i)
  {
    TermData term = {coefficients[i], id, dofs[i]};
    equation_data_.emplace_back(term);

    Core::IO::cout(Core::IO::debug) << " + " << coefficients[i] << " * d" << dofs[i];
  }
  Core::IO::cout(Core::IO::debug) << Core::IO::endl;
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE