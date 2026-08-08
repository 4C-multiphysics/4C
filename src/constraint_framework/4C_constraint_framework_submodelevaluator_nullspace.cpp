// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_framework_submodelevaluator_nullspace.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_condition_selector.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization_nullspace.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Constraints::SubmodelEvaluator::NullspaceConstraintManager::NullspaceConstraintManager(
    std::shared_ptr<Core::FE::Discretization> discret_ptr)
{
  discret_ptr_ = discret_ptr;

  std::vector<const Core::Conditions::Condition*> nullspace_constraint_conditions;
  discret_ptr_->get_condition("KrylovSpaceProjection", nullspace_constraint_conditions);
  int numcond = nullspace_constraint_conditions.size();
  int numsolid = 0;

  const Core::Conditions::Condition* structure_nullspace_constraint_condition = nullptr;
  // check if for solid Krylov projection is required
  for (int icond = 0; icond < numcond; icond++)
  {
    const std::string& name =
        nullspace_constraint_conditions[icond]->parameters().get<std::string>("DIS");
    if (name == "structure")
    {
      numsolid++;
      structure_nullspace_constraint_condition = nullspace_constraint_conditions[icond];
    }
  }

  if (numsolid != 1)
  {
    FOUR_C_THROW(
        "For the nullspace constraint enforcement exactly one KrylovSpaceCondition for "
        "the structure field is necessary.");
  }

  const auto type = structure_nullspace_constraint_condition->parameters().get<std::string>("TYPE");

  if (type != "constraint")
  {
    FOUR_C_THROW(
        "The KrylovSpaceCondition needs to be of type 'constraint' to be properly enforced.");
  }

  nullspace_dimension_ =
      structure_nullspace_constraint_condition->parameters().get<int>("NUMMODES");

  // just grab the block information on the first element that appears and check with given number
  Core::Elements::Element* dwele = discret_ptr->l_row_element(0);
  int nullspace_dimension;
  int number_of_dofs;
  dwele->element_type().nodal_block_information(dwele, number_of_dofs, nullspace_dimension);

  if (nullspace_dimension_ != nullspace_dimension)
  {
    FOUR_C_THROW(
        "The number of constraint modes needs to be equivalent to the number of nullspace "
        "dimensions. The number of modes given is {}, while the nullspace has a dimension "
        "of {}.",
        nullspace_dimension_, nullspace_dimension);
  }

  const auto mode_flags =
      structure_nullspace_constraint_condition->parameters().get<std::vector<int>>("ONOFF");

  for (int rr = 0; rr < nullspace_dimension_; ++rr)
  {
    if (mode_flags[rr] != 0) active_mode_ids_.push_back(rr);
  }

  const int max_gid_structure = discret_ptr_->dof_row_map()->max_all_gid();
  constraint_map_ = std::make_shared<Core::LinAlg::Map>(
      active_mode_ids_.size(), max_gid_structure + 1, discret_ptr_->get_comm());

  Core::IO::cout(Core::IO::verbose)
      << "Constructing nullspace constraint with " << active_mode_ids_.size()
      << " active components out of " << nullspace_dimension_ << " given.\n";

  const auto node_cond_map =
      Core::Conditions::condition_node_row_map(*discret_ptr_, "KrylovSpaceProjection");

  std::vector<int> cond_dof_gids;

  for (int i = 0; i < discret_ptr_->num_my_row_nodes(); i++)
  {
    const Core::Nodes::Node* node = discret_ptr_->l_row_node(i);
    if (node_cond_map->my_gid(node->id())) discret_ptr_->dof(node, cond_dof_gids);
  }

  dof_condition_map_ = std::make_shared<Core::LinAlg::Map>(
      -1, std::span<const int>(cond_dof_gids), 0, discret_ptr_->get_comm());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Constraints::SubmodelEvaluator::NullspaceConstraintManager::evaluate_force_stiff(
    const Core::LinAlg::Vector<double>& displacement_vector,
    std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr,
    std::shared_ptr<Core::LinAlg::SparseMatrix> me_stiff_ptr,
    std::shared_ptr<Core::LinAlg::Vector<double>> me_force_ptr)
{
  auto jacobian = global_state_ptr->get_jacobian();

  global_state_ptr->assign_model_block(
      *jacobian, *Q_dL_, Solid::model_constraints, Solid::MatBlockType::displ_lm);
  global_state_ptr->assign_model_block(
      *jacobian, *Q_Ld_, Solid::model_constraints, Solid::MatBlockType::lm_displ);

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Constraints::SubmodelEvaluator::NullspaceConstraintManager::evaluate_coupling_terms(
    Solid::TimeInt::BaseDataGlobalState& gstate)
{
  auto dof_map = discret_ptr_->dof_row_map();
  auto nullspace =
      Core::FE::compute_null_space(*discret_ptr_, nullspace_dimension_, *dof_condition_map_);

  Core::LinAlg::MultiVector<double> constraint_space(*dof_condition_map_, active_mode_ids_.size());

  for (int mode = 0; mode < static_cast<int>(active_mode_ids_.size()); mode++)
  {
    auto& constraint_space_column = constraint_space.get_vector(mode);
    auto& nullspace_column = nullspace->get_vector(active_mode_ids_[mode]);

    const size_t my_length = constraint_space_column.local_length();
    for (size_t j = 0; j < my_length; j++)
    {
      constraint_space_column.get_values()[j] = nullspace_column.get_values()[j];
    }
  }

  auto orthonormal_constraint_space = Core::LinAlg::orthonormalize_multi_vector(constraint_space);

  auto constraint_matrix = std::make_shared<Core::LinAlg::SparseMatrix>(
      *dof_condition_map_, orthonormal_constraint_space.num_vectors());
  Core::LinAlg::multi_vector_to_linalg_sparse_matrix(
      orthonormal_constraint_space, *dof_condition_map_, *constraint_map_, *constraint_matrix);

  Q_dL_ = Core::LinAlg::matrix_row_transform(*constraint_matrix, *dof_map);
  Q_Ld_ = Core::LinAlg::matrix_transpose(*Q_dL_);
  Q_Ld_->complete(*dof_map, *constraint_map_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::map<Solid::EnergyType, double>
Constraints::SubmodelEvaluator::NullspaceConstraintManager::get_energy() const
{
  FOUR_C_THROW("This function is not implemented for the NullspaceConstraintManager.");
}

FOUR_C_NAMESPACE_CLOSE
