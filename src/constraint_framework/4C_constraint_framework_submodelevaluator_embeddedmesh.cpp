// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_framework_submodelevaluator_embeddedmesh.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_constraint_framework_embeddedmesh_solid_to_solid_mortar_manager.hpp"
#include "4C_constraint_framework_embeddedmesh_solid_to_solid_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_io_visualization_manager.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

Constraints::SubmodelEvaluator::EmbeddedMeshConstraintManager::EmbeddedMeshConstraintManager(
    std::shared_ptr<Core::FE::Discretization> discret_ptr,
    const Core::LinAlg::Vector<double>& dispnp)
{
  // Get the parameter lists and get information from them
  auto constraint_parameter_list = Global::Problem::instance()->constraint_params();
  auto xfem_parameter_list = Global::Problem::instance()->xfem_general_params();
  auto cut_parameter_list = Global::Problem::instance()->cut_general_params();

  auto embedded_mesh_constraint_enforcement =
      Teuchos::getIntegralValue<Constraints::EnforcementStrategy>(
          constraint_parameter_list, "CONSTRAINT_ENFORCEMENT");

  auto embedded_mesh_constraint_penalty_parameter =
      constraint_parameter_list.get<double>("PENALTY_PARAM");

  auto embedded_mesh_parameter_list = constraint_parameter_list.sublist("EMBEDDED MESH COUPLING");

  auto embedded_mesh_coupling_strategy =
      Teuchos::getIntegralValue<Constraints::EmbeddedMesh::CouplingStrategy>(
          embedded_mesh_parameter_list, "COUPLING_STRATEGY");

  auto embedded_mesh_mortar_shape_function =
      Teuchos::getIntegralValue<Constraints::EmbeddedMesh::SolidToSolidMortarShapefunctions>(
          embedded_mesh_parameter_list, "MORTAR_SHAPE_FUNCTION");

  auto nodal_dofset_strategy = Teuchos::getIntegralValue<Cut::NodalDofSetStrategy>(
      xfem_parameter_list, "NODAL_DOFSET_STRATEGY");
  auto volume_cell_gauss_point_by =
      Teuchos::getIntegralValue<Cut::VCellGaussPts>(xfem_parameter_list, "VOLUME_GAUSS_POINTS_BY");
  auto bound_cell_gauss_point_by = Teuchos::getIntegralValue<Cut::BCellGaussPts>(
      xfem_parameter_list, "BOUNDARY_GAUSS_POINTS_BY");

  bool gmsh_cut_out = xfem_parameter_list.get<bool>("GMSH_CUT_OUT");
  bool cut_screen_output = xfem_parameter_list.get<bool>("PRINT_OUTPUT");

  // Initialize embedded mesh coupling parameters
  Constraints::EmbeddedMesh::EmbeddedMeshParams embedded_mesh_coupling_params = {
      .coupling_strategy_ = embedded_mesh_coupling_strategy,
      .constraint_enforcement_ = embedded_mesh_constraint_enforcement,
      .constraint_penalty_parameter_ = embedded_mesh_constraint_penalty_parameter,
      .mortar_shape_function_ = embedded_mesh_mortar_shape_function,
      .xfem_nodal_dof_set_strategy_ = nodal_dofset_strategy,
      .xfem_volume_cell_gauss_point_by_ = volume_cell_gauss_point_by,
      .xfem_bcell_gauss_point_by_ = bound_cell_gauss_point_by,
      .gmsh_cut_out_ = gmsh_cut_out,
      .cut_screen_output_ = cut_screen_output,
      .cut_params_ = cut_parameter_list};

  // Initialize visualization manager
  auto visualization_manager = std::make_shared<Core::IO::VisualizationManager>(
      Core::IO::visualization_parameters_factory(
          Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
          *Global::Problem::instance()->output_control_file(), 0.0),  // Fix time
      discret_ptr->get_comm(), "embedded_mesh");

  mortar_manager_ = std::make_shared<Constraints::EmbeddedMesh::SolidToSolidMortarManager>(
      discret_ptr, dispnp, embedded_mesh_coupling_params, visualization_manager,
      discret_ptr->dof_row_map()->max_all_gid() + 1);
}

bool Constraints::SubmodelEvaluator::EmbeddedMeshConstraintManager::evaluate_force_stiff(
    const Core::LinAlg::Vector<double>& displacement_vector,
    std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr,
    std::shared_ptr<Core::LinAlg::SparseMatrix> me_stiff_ptr,
    std::shared_ptr<Core::LinAlg::Vector<double>> me_force_ptr)
{
  // Evaluate the global mortar matrices
  mortar_manager_->evaluate_global_coupling_contributions(displacement_vector);
  mortar_manager_->add_global_force_stiffness_penalty_contributions(
      *global_state_ptr, me_stiff_ptr, me_force_ptr);

  return true;
}

void Constraints::SubmodelEvaluator::EmbeddedMeshConstraintManager::runtime_output_step_state(
    std::pair<double, int> output_time_and_step)
{
  // Write runtime output for the embedded mesh method
  mortar_manager_->write_output(output_time_and_step.first, output_time_and_step.second);
}

std::map<Solid::EnergyType, double>
Constraints::SubmodelEvaluator::EmbeddedMeshConstraintManager::get_energy() const
{
  std::map<Solid::EnergyType, double> embedded_mesh_energy;
  double mortar_manager_energy = mortar_manager_->get_energy();

  // The value we returned here is summed up over all processors. Since we already have the global
  // energy here, we only return it on rank 0.
  if (Core::Communication::my_mpi_rank(mortar_manager_->get_my_comm()) == 0)
  {
    embedded_mesh_energy[Solid::embedded_mesh_penalty_potential] = mortar_manager_energy;
  }

  return embedded_mesh_energy;
}

FOUR_C_NAMESPACE_CLOSE
