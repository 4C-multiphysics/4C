// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_utils.hpp"

#include "4C_art_net_input.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_poroelast_utils.hpp"
#include "4C_porofluid_pressure_based_elast_artery_coupling.hpp"
#include "4C_porofluid_pressure_based_elast_base.hpp"
#include "4C_porofluid_pressure_based_elast_clonestrategy.hpp"
#include "4C_porofluid_pressure_based_elast_monolithic.hpp"
#include "4C_porofluid_pressure_based_elast_partitioned.hpp"
#include "4C_porofluid_pressure_based_utils.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::setup_discretizations_and_field_coupling_porofluid_elast(
    const PorofluidElastAlgorithmDeps& algorithm_deps, const std::string& struct_disname,
    const std::string& fluid_disname, int& nds_disp, int& nds_vel, int& nds_solidpressure)
{
  // Creates a poro-fluid discretization by cloning the structural discretization read from the
  // input file.

  FOUR_C_ASSERT_ALWAYS(
      algorithm_deps.discretization_by_name, "Discretization callback is not initialized.");
  FOUR_C_ASSERT_ALWAYS(
      algorithm_deps.cloning_material_map != nullptr, "Cloning material map is not initialized.");
  FOUR_C_ASSERT_ALWAYS(algorithm_deps.validate_porofluid_material_id,
      "Material validation callback is not initialized.");

  const std::shared_ptr<Core::FE::Discretization> structure_discretization =
      algorithm_deps.discretization_by_name(struct_disname);
  const std::shared_ptr<Core::FE::Discretization> porofluid_discretization =
      algorithm_deps.discretization_by_name(fluid_disname);

  if (!structure_discretization->filled()) structure_discretization->fill_complete();
  if (!porofluid_discretization->filled()) porofluid_discretization->fill_complete();

  if (porofluid_discretization->num_global_nodes() == 0)
  {
    PorofluidCloneStrategy::set_material_validation_callback(
        algorithm_deps.validate_porofluid_material_id);

    // fill poro fluid discretization by cloning structure discretization
    Core::FE::clone_discretization<PorofluidCloneStrategy>(
        *structure_discretization, *porofluid_discretization, *algorithm_deps.cloning_material_map);
  }
  else
  {
    FOUR_C_THROW("Fluid discretization given in input file. This is not supported!");
  }

  structure_discretization->fill_complete();
  porofluid_discretization->fill_complete();

  // build a proxy of the structure discretization for the scatra field
  const std::shared_ptr<Core::DOFSets::DofSetInterface> structure_dofset =
      structure_discretization->get_dof_set_proxy();
  // build a proxy of the scatra discretization for the structure field
  const std::shared_ptr<Core::DOFSets::DofSetInterface> porofluid_dofset =
      porofluid_discretization->get_dof_set_proxy();

  // assign the structure dofset to the porofluid and save the dofset number
  nds_disp = porofluid_discretization->add_dof_set(structure_dofset);
  if (nds_disp != 1) FOUR_C_THROW("Unexpected dof sets in porofluid field.");
  // velocities have the same dofs as displacements
  nds_vel = nds_disp;

  if (structure_discretization->add_dof_set(porofluid_dofset) != 1)
    FOUR_C_THROW("Unexpected dof sets in structure field.");

  // build auxiliary dofset for postprocessing solid pressures
  const std::shared_ptr<Core::DOFSets::DofSetInterface> solid_pressure_dofset =
      std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(1, 0, 0, false);
  nds_solidpressure = porofluid_discretization->add_dof_set(solid_pressure_dofset);
  // add it also to the solid field
  structure_discretization->add_dof_set(
      porofluid_discretization->get_dof_set_proxy(nds_solidpressure));

  structure_discretization->fill_complete();
  porofluid_discretization->fill_complete();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::map<int, std::set<int>> PoroPressureBased::setup_discretizations_and_field_coupling_artery(
    const PorofluidElastAlgorithmDeps& algorithm_deps, const std::string& struct_disname)
{
  FOUR_C_ASSERT_ALWAYS(
      algorithm_deps.discretization_by_name, "Discretization callback is not initialized.");
  FOUR_C_ASSERT_ALWAYS(algorithm_deps.porofluid_pressure_based_dynamic_parameters != nullptr,
      "Porofluid pressure based dynamic parameters are not initialized.");

  const Teuchos::ParameterList& porofluid_pressure_based_dynamic_parameters =
      *algorithm_deps.porofluid_pressure_based_dynamic_parameters;

  const std::shared_ptr<Core::FE::Discretization> structure_discretization =
      algorithm_deps.discretization_by_name(struct_disname);

  std::shared_ptr<Core::FE::Discretization> artery_discretization = nullptr;
  artery_discretization = algorithm_deps.discretization_by_name("artery");

  const auto artery_coupling_method =
      Teuchos::getIntegralValue<ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod>(
          porofluid_pressure_based_dynamic_parameters.sublist("artery_coupling"),
          "coupling_method");

  const bool evaluate_on_lateral_surface =
      porofluid_pressure_based_dynamic_parameters.sublist("artery_coupling")
          .get<bool>("lateral_surface_coupling");

  const int maximum_number_of_segments_per_artery_element =
      porofluid_pressure_based_dynamic_parameters.sublist("artery_coupling")
          .get<int>("maximum_number_of_segments_per_artery_element");

  // curr_seg_lengths: defined as element-wise quantity
  const std::shared_ptr<Core::DOFSets::DofSetInterface> segment_dofset =
      std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(
          0, maximum_number_of_segments_per_artery_element, 0, false);
  // add it to artery discretization
  artery_discretization->add_dof_set(segment_dofset);

  // possible interaction partners [artelegid; contelegid_1, ... contelegid_n]
  std::map<int, std::set<int>> nearby_ele_pairs;

  switch (artery_coupling_method)
  {
    case ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::gauss_point_to_segment:
    case ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::mortar_penalty:
    case ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::node_to_point:
    {
      // perform extended ghosting on artery discretization
      nearby_ele_pairs = extended_ghosting_artery_discretization(*structure_discretization,
          artery_discretization, evaluate_on_lateral_surface, artery_coupling_method);
      break;
    }
    default:
    {
      break;
    }
  }

  if (!artery_discretization->filled()) artery_discretization->fill_complete();

  return nearby_ele_pairs;
}

/*----------------------------------------------------------------------*
 | exchange material pointers of both discretizations       vuong 08/16 |
 *----------------------------------------------------------------------*/
void PoroPressureBased::assign_material_pointers_porofluid_elast(
    const PorofluidElastAlgorithmDeps& algorithm_deps, const std::string& struct_disname,
    const std::string& fluid_disname)
{
  FOUR_C_ASSERT_ALWAYS(
      algorithm_deps.discretization_by_name, "Discretization callback is not initialized.");

  std::shared_ptr<Core::FE::Discretization> structdis =
      algorithm_deps.discretization_by_name(struct_disname);
  std::shared_ptr<Core::FE::Discretization> fluiddis =
      algorithm_deps.discretization_by_name(fluid_disname);

  PoroElast::Utils::set_material_pointers_matching_grid(*structdis, *fluiddis);
}

/*----------------------------------------------------------------------*
 | create algorithm                                                      |
 *----------------------------------------------------------------------*/
std::shared_ptr<PoroPressureBased::PorofluidElastAlgorithm>
PoroPressureBased::create_algorithm_porofluid_elast(
    PoroPressureBased::SolutionSchemePorofluidElast solscheme,
    const Teuchos::ParameterList& timeparams, MPI_Comm comm,
    PorofluidElastAlgorithmDeps algorithm_deps)
{
  // Creation of Coupled Problem algorithm.
  std::shared_ptr<PoroPressureBased::PorofluidElastAlgorithm> algo;

  // Translate updated porofluid input format to old adapter format
  Teuchos::ParameterList adapter_global_time_params;
  adapter_global_time_params.set<double>(
      "TIMESTEP", timeparams.sublist("time_integration").get<double>("time_step_size"));
  adapter_global_time_params.set<int>(
      "NUMSTEP", timeparams.sublist("time_integration").get<int>("number_of_time_steps"));
  adapter_global_time_params.set<double>(
      "MAXTIME", timeparams.get<double>("total_simulation_time"));

  switch (solscheme)
  {
    case SolutionSchemePorofluidElast::twoway_partitioned:
    {
      // call constructor
      algo = std::make_shared<PoroPressureBased::PorofluidElastPartitionedAlgorithm>(
          comm, adapter_global_time_params);
      break;
    }
    case SolutionSchemePorofluidElast::twoway_monolithic:
    {
      const bool artery_coupl = timeparams.get<bool>("artery_coupling_active");
      if (!artery_coupl)
      {
        // call constructor
        algo = std::make_shared<PoroPressureBased::PorofluidElastMonolithicAlgorithm>(
            comm, adapter_global_time_params);
      }
      else
      {
        // call constructor
        algo = std::make_shared<PoroPressureBased::PorofluidElastArteryCouplingAlgorithm>(
            comm, adapter_global_time_params);
      }
      break;
    }
    default:
      FOUR_C_THROW("Unknown time-integration scheme for multiphase poro fluid problem");
      break;
  }

  FOUR_C_ASSERT_ALWAYS(algo != nullptr, "Porofluid-elast algorithm creation failed.");
  algo->set_algorithm_deps(std::move(algorithm_deps));

  return algo;
}


FOUR_C_NAMESPACE_CLOSE
