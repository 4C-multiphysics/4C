// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_constraint_framework_submodelevaluator_base.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_structure_new_timint_base.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <set>

FOUR_C_NAMESPACE_OPEN

Constraints::SubmodelEvaluator::ConstraintBase::ConstraintBase()
{
  auto constraint_parameter_list = Global::Problem::instance()->constraint_params();

  strategy_ = Teuchos::getIntegralValue<EnforcementStrategy>(
      constraint_parameter_list, "CONSTRAINT_ENFORCEMENT");
}

void Constraints::SubmodelEvaluator::ConstraintBase::evaluate_force_stiff(
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
}

void Constraints::SubmodelEvaluator::ConstraintBase::setup_constraint_equation_map(
    const Core::LinAlg::Map& displacement_map)
{
  const MPI_Comm comm = stiff_ptr_->get_comm();

  // start counting where the lower ranks stop
  int num_my_equations = 0;
  for (const auto& mpc : constraint_equations_)
    num_my_equations += mpc->get_number_of_constraint_equation_objects();
  std::vector<int> num_equations_per_rank(Core::Communication::num_mpi_ranks(comm), 0);
  Core::Communication::gather_all(&num_my_equations, num_equations_per_rank.data(), 1, comm);
  int equation_id = 0;
  for (int rank = 0; rank < Core::Communication::my_mpi_rank(comm); ++rank)
    equation_id += num_equations_per_rank[rank];

  // ToDo: Add an offset to the constraint equation ids so they do not overlap with the dof gids
  // (needed for a Lagrange multiplier enforcement).
  std::vector<int> my_equation_ids;
  my_equation_ids.reserve(num_my_equations);
  for (const auto& mpc : constraint_equations_)
  {
    mpc->set_first_row_id(equation_id);
    for (int i = 0; i < mpc->get_number_of_constraint_equation_objects(); ++i)
      my_equation_ids.push_back(equation_id++);
  }

  // each rank owns the rows of the equations it created
  constraint_equation_map_ =
      std::make_shared<Core::LinAlg::Map>(-1, num_my_equations, my_equation_ids.data(), 0, comm);

  // the equations need the displacements of dofs that may be owned by other ranks:
  // collect those dofs in an overlapping map and prepare the import into it
  std::set<int> unique_dof_gids;
  for (const auto& mpc : constraint_equations_)
  {
    const std::vector<int> dof_gids = mpc->get_dof_gids();
    unique_dof_gids.insert(dof_gids.begin(), dof_gids.end());
  }
  const std::vector<int> equation_dof_gids(unique_dof_gids.begin(), unique_dof_gids.end());
  const Core::LinAlg::Map equation_dof_map(
      -1, static_cast<int>(equation_dof_gids.size()), equation_dof_gids.data(), 0, comm);
  equation_dof_displacements_ = std::make_shared<Core::LinAlg::Vector<double>>(equation_dof_map);
  equation_dof_importer_ =
      std::make_shared<Core::LinAlg::Import>(equation_dof_map, displacement_map);
}

void Constraints::SubmodelEvaluator::ConstraintBase::evaluate_coupling_terms(
    Solid::TimeInt::BaseDataGlobalState& gstate)
{
  if (!constraint_equation_map_) setup_constraint_equation_map(gstate.get_dis_np()->get_map());

#ifdef FOUR_C_ENABLE_ASSERTIONS
  int num_my_equations = 0;
  for (const auto& mpc : constraint_equations_)
    num_my_equations += mpc->get_number_of_constraint_equation_objects();
  FOUR_C_ASSERT(num_my_equations == constraint_equation_map_->num_my_elements(),
      "{} constraint equations, but {} rows: equations were added after the first evaluation.",
      num_my_equations, constraint_equation_map_->num_my_elements());
#endif

  // initialise all global coupling objects
  constraint_residual_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*constraint_equation_map_, true);
  Q_Ld_ = std::make_shared<Core::LinAlg::SparseMatrix>(*constraint_equation_map_, 4);
  // FE matrices (FE_MATRIX), since the dofs of an equation (and thus the rows of Q_dL and Q_dd)
  // may be owned by other ranks
  Q_dL_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      stiff_ptr_->row_map(), 4, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Q_dd_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      stiff_ptr_->row_map(), 0, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Q_dd_->zero();

  // fetch the current displacements of the dofs the equations need
  equation_dof_displacements_->import(
      *gstate.get_dis_np(), *equation_dof_importer_, Core::LinAlg::CombineMode::insert);

  // Evaluate the Constraint Pairs / equations objects
  for (const auto& obj : constraint_equations_)
  {
    obj->evaluate_equation(
        *Q_dd_, *Q_dL_, *Q_Ld_, *constraint_residual_, *equation_dof_displacements_);
  }
  Core::IO::cout(Core::IO::debug) << "Evaluated all constraint objects" << Core::IO::endl;

  // Complete
  Q_dd_->complete();
  Q_Ld_->complete(stiff_ptr_->domain_map(), *constraint_equation_map_);
  Q_dL_->complete(*constraint_equation_map_, stiff_ptr_->domain_map());
}
FOUR_C_NAMESPACE_CLOSE
