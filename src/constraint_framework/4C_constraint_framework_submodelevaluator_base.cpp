// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_constraint_framework_submodelevaluator_base.hpp"

#include "4C_global_data.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_structure_new_timint_base.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

Constraints::SubmodelEvaluator::ConstraintBase::ConstraintBase()
{
  auto constraint_parameter_list = Global::Problem::instance()->constraint_params();

  strategy_ = Teuchos::getIntegralValue<EnforcementStrategy>(
      constraint_parameter_list, "CONSTRAINT_ENFORCEMENT");
}

void Constraints::SubmodelEvaluator::ConstraintBase::set_owned_constraint_row_ids(
    std::vector<int> row_ids)
{
  owned_constraint_row_ids_ = std::move(row_ids);
  use_explicit_constraint_row_ids_ = true;
}

bool Constraints::SubmodelEvaluator::ConstraintBase::evaluate_force_stiff(
    const Core::LinAlg::Vector<double>& displacement_vector,
    std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr,
    std::shared_ptr<Core::LinAlg::SparseMatrix> me_stiff_ptr,
    std::shared_ptr<Core::LinAlg::Vector<double>> me_force_ptr)
{
  if (me_stiff_ptr == nullptr && me_force_ptr == nullptr)
    FOUR_C_THROW("Both stiffness and force point are null");

  if (me_stiff_ptr != nullptr)
  {
    if (!(Q_Ld_->filled() && Q_dd_->filled() && Q_dL_->filled()))
      FOUR_C_THROW("Call evaluate_coupling_terms() first.");

    // evaluate the stiffness contribution of this some:
    auto some_stiff_ptr = Core::LinAlg::matrix_multiply(*Q_dL_, false, *Q_Ld_, false, false);
    some_stiff_ptr->scale(penalty_parameter_);
    Core::LinAlg::matrix_add(*Q_dd_, false, 1.0, *some_stiff_ptr, 1.0);
    some_stiff_ptr->complete();

    // add it to the modelevaluator stiffness
    Core::LinAlg::matrix_add(*some_stiff_ptr, false, 1., *me_stiff_ptr, 1.);
  }

  if (me_force_ptr != nullptr)
  {
    //  Calculate force contribution
    Core::LinAlg::Vector<double> r_pen(stiff_ptr_->row_map(), true);
    Q_Ld_->multiply(true, *constraint_residual_, r_pen);
    Core::LinAlg::assemble_my_vector(1.0, *me_force_ptr, penalty_parameter_, r_pen);
  }
  return true;
}

void Constraints::SubmodelEvaluator::ConstraintBase::evaluate_coupling_terms(
    Solid::TimeInt::BaseDataGlobalState& gstate)
{
  // Get the number of multipoint equations
  int ncon_ = 0;
  for (const auto& mpc : constraint_equations_)
    ncon_ += mpc->get_number_of_constraint_equation_objects();

  if (!use_explicit_constraint_row_ids_)
  {
    n_condition_map_ = std::make_shared<Core::LinAlg::Map>(ncon_, 0, stiff_ptr_->get_comm());
  }
  else
  {
    n_condition_map_ =
        std::make_shared<Core::LinAlg::Map>(-1, static_cast<int>(owned_constraint_row_ids_.size()),
            owned_constraint_row_ids_.data(), 0, stiff_ptr_->get_comm());
  }

  // initialise all global coupling objects
  constraint_residual_ = std::make_shared<Core::LinAlg::Vector<double>>(*n_condition_map_, true);
  Q_Ld_ = std::make_shared<Core::LinAlg::SparseMatrix>(*n_condition_map_, 4);
  Q_dd_ = std::make_shared<Core::LinAlg::SparseMatrix>(stiff_ptr_->row_map(), 0);
  Q_dd_->zero();

  std::shared_ptr<const Core::LinAlg::Vector<double>> dis_np = gstate.get_dis_np();
  for (const auto& obj : constraint_equations_) obj->evaluate_equation(*Q_Ld_);

  Q_dd_->complete();
  Q_Ld_->complete(stiff_ptr_->domain_map(), *n_condition_map_);

  // Q_dL = Q_Ld^T
  Q_dL_ = Core::LinAlg::matrix_transpose(*Q_Ld_);
  Q_dL_->complete(*n_condition_map_, stiff_ptr_->domain_map());

  // constraint residual r_L = Q_Ld * d
  Q_Ld_->multiply(false, *dis_np, *constraint_residual_);
}
FOUR_C_NAMESPACE_CLOSE
