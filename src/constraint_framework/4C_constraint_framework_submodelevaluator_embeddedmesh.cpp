// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_framework_submodelevaluator_embeddedmesh.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_constraint_framework_embeddedmesh_solid_to_solid_coupling_manager.hpp"
#include "4C_constraint_framework_embeddedmesh_solid_to_solid_mortar_manager.hpp"
#include "4C_constraint_framework_embeddedmesh_solid_to_solid_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_io_visualization_manager.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

void get_cut_xfem_parameters(
    Constraints::EmbeddedMesh::EmbeddedMeshParams& embedded_mesh_coupling_params,
    Teuchos::ParameterList& cut_parameter_list, Teuchos::ParameterList& xfem_parameter_list)
{
  auto nodal_dofset_strategy = Teuchos::getIntegralValue<Cut::NodalDofSetStrategy>(
      xfem_parameter_list, "NODAL_DOFSET_STRATEGY");
  auto volume_cell_gauss_point_by =
      Teuchos::getIntegralValue<Cut::VCellGaussPts>(xfem_parameter_list, "VOLUME_GAUSS_POINTS_BY");
  auto bound_cell_gauss_point_by = Teuchos::getIntegralValue<Cut::BCellGaussPts>(
      xfem_parameter_list, "BOUNDARY_GAUSS_POINTS_BY");

  bool gmsh_cut_out = xfem_parameter_list.get<bool>("GMSH_CUT_OUT");
  bool cut_screen_output = xfem_parameter_list.get<bool>("PRINT_OUTPUT");

  embedded_mesh_coupling_params.xfem_nodal_dof_set_strategy_ = nodal_dofset_strategy;
  embedded_mesh_coupling_params.xfem_volume_cell_gauss_point_by_ = volume_cell_gauss_point_by;
  embedded_mesh_coupling_params.xfem_bcell_gauss_point_by_ = bound_cell_gauss_point_by;
  embedded_mesh_coupling_params.gmsh_cut_out_ = gmsh_cut_out;
  embedded_mesh_coupling_params.cut_screen_output_ = cut_screen_output;
  embedded_mesh_coupling_params.cut_params_ = cut_parameter_list;
}

void get_mortar_parameters(
    Constraints::EmbeddedMesh::EmbeddedMeshParams& embedded_mesh_coupling_params,
    Teuchos::ParameterList& constraint_parameter_list,
    const Teuchos::ParameterList& embedded_mesh_parameter_list)
{
  auto embedded_mesh_coupling_strategy =
      Teuchos::getIntegralValue<Constraints::EmbeddedMesh::CouplingStrategy>(
          embedded_mesh_parameter_list, "COUPLING_STRATEGY");

  auto embedded_mesh_constraint_enforcement =
      Teuchos::getIntegralValue<Constraints::EnforcementStrategy>(
          constraint_parameter_list, "CONSTRAINT_ENFORCEMENT");

  auto embedded_mesh_constraint_penalty_parameter =
      constraint_parameter_list.get<double>("PENALTY_PARAM");

  auto embedded_mesh_mortar_shape_function =
      Teuchos::getIntegralValue<Constraints::EmbeddedMesh::SolidToSolidMortarShapefunctions>(
          embedded_mesh_parameter_list, "MORTAR_SHAPE_FUNCTION");

  embedded_mesh_coupling_params.coupling_strategy_ = embedded_mesh_coupling_strategy;
  embedded_mesh_coupling_params.constraint_enforcement_ = embedded_mesh_constraint_enforcement;
  embedded_mesh_coupling_params.constraint_penalty_parameter_ =
      embedded_mesh_constraint_penalty_parameter;
  embedded_mesh_coupling_params.mortar_shape_function_ = embedded_mesh_mortar_shape_function;
}

void get_coupling_parameters(
    Constraints::EmbeddedMesh::EmbeddedMeshParams& embedded_mesh_coupling_params,
    Teuchos::ParameterList& constraint_parameter_list,
    const Teuchos::ParameterList& embedded_mesh_parameter_list,
    const Constraints::EmbeddedMesh::CouplingStrategy& coupling_strategy)
{
  switch (coupling_strategy)
  {
    case Constraints::EmbeddedMesh::CouplingStrategy::mortar:
    {
      get_mortar_parameters(
          embedded_mesh_coupling_params, constraint_parameter_list, embedded_mesh_parameter_list);
      break;
    }
    default:
      FOUR_C_THROW("Parameter call not implemented for the coupling strategy.");
  }
}

Constraints::SubmodelEvaluator::EmbeddedMeshConstraintManager::EmbeddedMeshConstraintManager(
    std::shared_ptr<Core::FE::Discretization> discret_ptr,
    const Core::LinAlg::Vector<double>& displacement_np)
{
  // Initialize visualization manager
  auto visualization_manager = std::make_shared<Core::IO::VisualizationManager>(
      Core::IO::visualization_parameters_factory(
          Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
          *Global::Problem::instance()->output_control_file(), 0.0),  // Fix time
      discret_ptr->get_comm(), "embedded_mesh");

  // Get the parameter lists
  auto constraint_parameter_list = Global::Problem::instance()->constraint_params();
  auto xfem_parameter_list = Global::Problem::instance()->xfem_general_params();
  auto cut_parameter_list = Global::Problem::instance()->cut_general_params();
  auto embedded_mesh_parameter_list = constraint_parameter_list.sublist("EMBEDDED MESH COUPLING");

  // Get the coupling strategy
  coupling_strategy_ = Teuchos::getIntegralValue<Constraints::EmbeddedMesh::CouplingStrategy>(
      embedded_mesh_parameter_list, "COUPLING_STRATEGY");

  // Make instance of embedded mesh coupling parameters
  Constraints::EmbeddedMesh::EmbeddedMeshParams embedded_mesh_coupling_params;

  // Assign parameters of cut and xfem
  get_cut_xfem_parameters(embedded_mesh_coupling_params, cut_parameter_list, xfem_parameter_list);

  // Define parameters depending on the coupling strategy
  get_coupling_parameters(embedded_mesh_coupling_params, constraint_parameter_list,
      embedded_mesh_parameter_list, coupling_strategy_);

  // Get the coupling manager depending on the coupling strategy
  coupling_manager_ = get_coupling_manager(
      discret_ptr, displacement_np, embedded_mesh_coupling_params, visualization_manager);
}

std::shared_ptr<Constraints::EmbeddedMesh::SolidToSolidCouplingManager>
Constraints::SubmodelEvaluator::EmbeddedMeshConstraintManager::get_coupling_manager(
    std::shared_ptr<Core::FE::Discretization> discret_ptr,
    const Core::LinAlg::Vector<double>& displacement_np,
    Constraints::EmbeddedMesh::EmbeddedMeshParams embedded_mesh_coupling_params,
    std::shared_ptr<Core::IO::VisualizationManager> visualization_manager)
{
  switch (coupling_strategy_)
  {
    case Constraints::EmbeddedMesh::CouplingStrategy::mortar:
    {
      return std::make_shared<Constraints::EmbeddedMesh::SolidToSolidMortarManager>(discret_ptr,
          displacement_np, embedded_mesh_coupling_params, visualization_manager,
          discret_ptr->dof_row_map()->max_all_gid() + 1);
      break;
    }
    default:
      FOUR_C_THROW("Coupling manager not implemented for this coupling strategy");
  }
}

bool Constraints::SubmodelEvaluator::EmbeddedMeshConstraintManager::evaluate_force_stiff(
    const Core::LinAlg::Vector<double>& displacement_vector,
    std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr,
    std::shared_ptr<Core::LinAlg::SparseMatrix> me_stiff_ptr,
    std::shared_ptr<Core::LinAlg::Vector<double>> me_force_ptr)
{
  // Evaluate the global matrices
  coupling_manager_->evaluate_global_coupling_contributions(displacement_vector);
  coupling_manager_->add_global_force_stiffness_contributions(
      *global_state_ptr, me_stiff_ptr, me_force_ptr);

  return true;
}

void Constraints::SubmodelEvaluator::EmbeddedMeshConstraintManager::runtime_output_step_state(
    std::pair<double, int> output_time_and_step)
{
  // Write runtime output for the embedded mesh method
  coupling_manager_->write_output(output_time_and_step.first, output_time_and_step.second);
}

std::map<Solid::EnergyType, double>
Constraints::SubmodelEvaluator::EmbeddedMeshConstraintManager::get_energy() const
{
  if (coupling_strategy_ == Constraints::EmbeddedMesh::CouplingStrategy::mortar)
  {
    std::map<Solid::EnergyType, double> embedded_mesh_energy;
    double coupling_manager_energy = coupling_manager_->get_energy();

    // The value we returned here is summed up over all processors. Since we already have the global
    // energy here, we only return it on rank 0.
    if (Core::Communication::my_mpi_rank(coupling_manager_->get_my_comm()) == 0)
    {
      embedded_mesh_energy[Solid::embedded_mesh_penalty_potential] = coupling_manager_energy;
    }

    return embedded_mesh_energy;
  }
  else
  {
    FOUR_C_THROW("Energy not implemented for this constraint strategy.");
  }
}

FOUR_C_NAMESPACE_CLOSE
