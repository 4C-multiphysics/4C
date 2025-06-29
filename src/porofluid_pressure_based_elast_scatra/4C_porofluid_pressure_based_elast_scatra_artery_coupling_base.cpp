// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_base.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PorofluidElastScatraArteryCouplingBaseAlgorithm::
    PorofluidElastScatraArteryCouplingBaseAlgorithm(
        const std::shared_ptr<Core::FE::Discretization> artery_dis,
        const std::shared_ptr<Core::FE::Discretization> homogenized_dis,
        const Teuchos::ParameterList& coupling_params)
    : artery_dis_(artery_dis),
      homogenized_dis_(homogenized_dis),
      my_mpi_rank_(Core::Communication::my_mpi_rank(artery_dis->get_comm())),
      evaluate_in_ref_config_(Global::Problem::instance()
              ->porofluid_pressure_based_dynamic_params()
              .sublist("artery_coupling")
              .get<bool>("evaluate_in_reference_configuration")),
      comm_(artery_dis->get_comm())
{
  // safety check
  if (artery_dis_->num_global_nodes() == 0)
    FOUR_C_THROW("artery discretization does not seem to have any nodes");

  // get the actual coupled DOFs
  // 1) 1D artery discretization
  int dof_value;
  size_t index = 0;
  std::istringstream coupled_artery_dof_stream(
      Teuchos::getNumericStringParameter(coupling_params.sublist("coupled_dofs"), "artery"));
  while (coupled_artery_dof_stream >> dof_value)
  {
    // check ascending order
    if (index > 0)
      if ((dof_value - 1) <= coupled_dofs_artery_[index - 1])
        FOUR_C_THROW("DOFs have to be ordered in ascending order");
    coupled_dofs_artery_.push_back((dof_value - 1));
    index++;
  }

  // 2) 2D, 3D continuous field discretization
  index = 0;
  std::istringstream coupled_homogenized_dof_stream(
      Teuchos::getNumericStringParameter(coupling_params.sublist("coupled_dofs"), "homogenized"));
  while (coupled_homogenized_dof_stream >> dof_value)
  {
    // check ascending order
    if (index > 0)
      if ((dof_value - 1) <= coupled_dofs_homogenized_[index - 1])
        FOUR_C_THROW("DOFs have to be ordered in ascending order");
    coupled_dofs_homogenized_.push_back((dof_value - 1));
    index++;
  }

  // no coupling selected by user
  if (coupled_dofs_homogenized_.size() == 1 and coupled_dofs_artery_.size() == 1 and
      coupled_dofs_homogenized_[0] < 0 and coupled_dofs_artery_[0] < 0)
  {
    coupled_dofs_homogenized_.resize(0);
    coupled_dofs_artery_.resize(0);
  }

  if (coupled_dofs_homogenized_.size() != coupled_dofs_artery_.size())
    FOUR_C_THROW("size mismatch between the coupled DOFs in the artery and the homogenized domain");

  num_coupled_dofs_ = coupled_dofs_homogenized_.size();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingBaseAlgorithm::
    recompute_coupled_dofs_for_node_to_point_coupling(
        std::vector<const Core::Conditions::Condition*> coupling_condition,
        unsigned int coupling_node_idx)
{
  // decrease the value of all elements by 1
  // because we start counting from 0 here and in the input file we start from 1
  coupled_dofs_artery_ = (coupling_condition[coupling_node_idx]->parameters().get<std::vector<int>>(
      "COUPLEDDOF_REDUCED"));
  for (int& value : coupled_dofs_artery_)
  {
    value--;
  }

  coupled_dofs_homogenized_ =
      (coupling_condition[coupling_node_idx]->parameters().get<std::vector<int>>(
          "COUPLEDDOF_PORO"));
  for (int& value : coupled_dofs_homogenized_)
  {
    value--;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const std::shared_ptr<const Core::LinAlg::Map>&
PoroPressureBased::PorofluidElastScatraArteryCouplingBaseAlgorithm::full_map() const
{
  return global_extractor_->full_map();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const std::shared_ptr<Core::LinAlg::MultiMapExtractor>&
PoroPressureBased::PorofluidElastScatraArteryCouplingBaseAlgorithm::global_extractor() const
{
  return global_extractor_;
}

FOUR_C_NAMESPACE_CLOSE
