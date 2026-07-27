// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_constraint_framework_equation.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_io_pstream.hpp"

FOUR_C_NAMESPACE_OPEN
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Constraints::SubmodelEvaluator::LinearCoupledEquation::evaluate_equation(
    Core::LinAlg::SparseMatrix& Q_Ld)
{
  // assemble the rows owned by this rank
  for (const auto& [coefficient, row_id, dof_id] : equation_data_)
    if (Q_Ld.row_map().my_gid(row_id)) Q_Ld.assemble(coefficient, row_id, dof_id);
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
