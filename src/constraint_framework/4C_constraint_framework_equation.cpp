// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_constraint_framework_equation.hpp"

#include "4C_fem_discretization.hpp"

FOUR_C_NAMESPACE_OPEN
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Constraints::SubmodelEvaluator::LinearCoupledEquation::evaluate_equation(
    Core::LinAlg::SparseMatrix& Q_dd, Core::LinAlg::SparseMatrix& Q_dL,
    Core::LinAlg::SparseMatrix& Q_Ld, Core::LinAlg::Vector<double>& constraint_vector,
    const Core::LinAlg::Vector<double>& D_np1)
{
  // loop over the terms of this equation; each term is one coefficient times one dof,
  // stored as (coefficient, row id, dof id) with row id = id of the equation
  for (const auto& [coefficient, row_id, dof_id] : equation_data_)
  {
    FOUR_C_ASSERT(Q_Ld.row_map().my_gid(row_id),
        "Constraint equation row {} is not owned by this rank.", row_id);
    const int dof_lid = D_np1.get_map().lid(dof_id);
    FOUR_C_ASSERT(dof_lid >= 0, "Displacement of dof {} is not available on this rank.", dof_id);

    // stiffness contribution; the dof row of Q_dL may belong to another rank (FE assembly)
    Q_Ld.assemble(coefficient, row_id, dof_id);
    Q_dL.fe_assemble(coefficient, dof_id, row_id);

    // force contribution: violation of the equation = sum of coefficient * displacement
    const double constraint_violation = D_np1.get_values()[dof_lid] * coefficient;
    constraint_vector.sum_into_global_values(1, &constraint_violation, &row_id);
  }
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::vector<int> Constraints::SubmodelEvaluator::LinearCoupledEquation::get_dof_gids() const
{
  std::vector<int> dof_gids;
  dof_gids.reserve(equation_data_.size());
  for (const auto& term : equation_data_) dof_gids.push_back(term.DofId);
  return dof_gids;
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Constraints::SubmodelEvaluator::ConstraintEquationBase::
    get_number_of_constraint_equation_objects() const
{
  return n_dof_coupled_;
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Constraints::SubmodelEvaluator::ConstraintEquationBase::get_first_row_id() const
{
  return first_row_id_;
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Constraints::SubmodelEvaluator::ConstraintEquationBase::set_first_row_id(int global_row_id)
{
  first_row_id_ = global_row_id;
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Constraints::SubmodelEvaluator::LinearCoupledEquation::LinearCoupledEquation(
    int id, const std::vector<int>& dofs, std::vector<double> coefficients)
{
  set_first_row_id(id);

  for (std::vector<double>::size_type i = 0; i < coefficients.size(); ++i)
  {
    TermData term = {coefficients[i], id, dofs[i]};
    equation_data_.emplace_back(term);
  }
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE
