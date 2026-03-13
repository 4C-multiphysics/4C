// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_utils.hpp"

#include "4C_art_net_utils.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_data.hpp"
#include "4C_poroelast_scatra_utils_clonestrategy.hpp"
#include "4C_poroelast_utils.hpp"
#include "4C_poroelast_utils_clonestrategy.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_linebased.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_nodebased.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_nodetopoint.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_surfbased.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_monolithic.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_partitioned.hpp"
#include "4C_porofluid_pressure_based_elast_utils.hpp"
#include "4C_rebalance_binning_based.hpp"
#include "4C_scatra_ele.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<PoroPressureBased::PorofluidElastScatraBaseAlgorithm>
PoroPressureBased::create_algorithm_porofluid_elast_scatra(
    PoroPressureBased::SolutionSchemePorofluidElastScatra solscheme,
    const Teuchos::ParameterList& timeparams, MPI_Comm comm)
{
  // Creation of Coupled Problem algorithm.
  std::shared_ptr<PoroPressureBased::PorofluidElastScatraBaseAlgorithm> algo;

  // Translate updated porofluid input format to old adapter format
  Teuchos::ParameterList adapter_global_time_params;
  adapter_global_time_params.set<double>(
      "TIMESTEP", timeparams.sublist("time_integration").get<double>("time_step_size"));
  adapter_global_time_params.set<int>(
      "NUMSTEP", timeparams.sublist("time_integration").get<int>("number_of_time_steps"));
  adapter_global_time_params.set<double>(
      "MAXTIME", timeparams.get<double>("total_simulation_time"));
  adapter_global_time_params.set<bool>(
      "artery_coupling_active", timeparams.get<bool>("artery_coupling_active"));
  adapter_global_time_params.set<DivergenceAction>(
      "divergence_action", timeparams.get<DivergenceAction>("divergence_action"));

  switch (solscheme)
  {
    case SolutionSchemePorofluidElastScatra::twoway_partitioned_nested:
    {
      // call constructor
      algo = std::make_shared<PoroPressureBased::PorofluidElastScatraNestedPartitionedAlgorithm>(
          comm, adapter_global_time_params);
      break;
    }
    case SolutionSchemePorofluidElastScatra::twoway_partitioned_sequential:
    {
      // call constructor
      algo =
          std::make_shared<PoroPressureBased::PorofluidElastScatraSequentialPartitionedAlgorithm>(
              comm, adapter_global_time_params);
      break;
    }
    case SolutionSchemePorofluidElastScatra::twoway_monolithic:
    {
      const bool artery_coupl = timeparams.get<bool>("artery_coupling_active");
      if (!artery_coupl)
      {
        // call constructor
        algo = std::make_shared<PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm>(
            comm, adapter_global_time_params);
      }
      else
      {
        // call constructor
        algo = std::make_shared<
            PoroPressureBased::PorofluidElastScatraMonolithicArteryCouplingAlgorithm>(

            comm, adapter_global_time_params);
      }
      break;
    }
    default:
      FOUR_C_THROW("Unknown time-integration scheme for multiphase poro fluid problem");
      break;
  }

  return algo;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<PoroPressureBased::PorofluidElastScatraArteryCouplingBaseAlgorithm>
PoroPressureBased::create_and_init_artery_coupling_strategy(
    std::shared_ptr<Core::FE::Discretization> arterydis,
    std::shared_ptr<Core::FE::Discretization> contdis,
    const PoroPressureBased::PorofluidElastScatraArteryCouplingDeps& artery_coupling_deps,
    const Teuchos::ParameterList& meshtyingparams, const std::string& condname,
    const bool evaluate_on_lateral_surface)
{
  // Creation of coupling strategy.
  std::shared_ptr<PoroPressureBased::PorofluidElastScatraArteryCouplingBaseAlgorithm> strategy;

  auto arterycoupl =
      Teuchos::getIntegralValue<ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod>(
          meshtyingparams, "coupling_method");

  switch (arterycoupl)
  {
    case ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::gauss_point_to_segment:
    case ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::mortar_penalty:
    {
      if (evaluate_on_lateral_surface)
        strategy = std::make_shared<
            PoroPressureBased::PorofluidElastScatraArteryCouplingSurfaceBasedAlgorithm>(
            arterydis, contdis, meshtyingparams, condname, artery_coupling_deps);
      else
        strategy = std::make_shared<
            PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm>(
            arterydis, contdis, meshtyingparams, condname, artery_coupling_deps);
      break;
    }
    case ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::nodal:
    {
      strategy =
          std::make_shared<PoroPressureBased::PorofluidElastScatraArteryCouplingNodeBasedAlgorithm>(
              arterydis, contdis, meshtyingparams, condname, artery_coupling_deps);
      break;
    }
    case ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::node_to_point:
    {
      strategy = std::make_shared<
          PoroPressureBased::PorofluidElastScatraArteryCouplingNodeToPointAlgorithm>(
          arterydis, contdis, meshtyingparams, condname, artery_coupling_deps);
      break;
    }
    default:
    {
      FOUR_C_THROW("Wrong type of artery-coupling strategy");
      break;
    }
  }

  strategy->init();

  return strategy;
}

std::shared_ptr<PoroPressureBased::PorofluidElastScatraArteryCouplingBaseAlgorithm>
PoroPressureBased::create_and_init_artery_coupling_strategy(
    std::shared_ptr<Core::FE::Discretization> arterydis,
    std::shared_ptr<Core::FE::Discretization> contdis,
    const Teuchos::ParameterList& meshtyingparams, const std::string& condname,
    const bool evaluate_on_lateral_surface)
{
  Global::Problem* problem = Global::Problem::instance();
  const PoroPressureBased::PorofluidElastScatraArteryCouplingDeps artery_coupling_deps =
      PoroPressureBased::make_artery_coupling_deps_from_problem(*problem);

  return create_and_init_artery_coupling_strategy(std::move(arterydis), std::move(contdis),
      artery_coupling_deps, meshtyingparams, condname, evaluate_on_lateral_surface);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::map<int, std::set<int>>
PoroPressureBased::setup_discretizations_and_field_coupling_porofluid_elast_scatra(
    const PoroPressureBased::PorofluidElastAlgorithmDeps& porofluid_elast_algorithm_deps,
    MPI_Comm comm, const std::string& struct_disname, const std::string& fluid_disname,
    const std::string& scatra_disname, int& ndsporo_disp, int& ndsporo_vel,
    int& ndsporo_solidpressure, int& ndsporofluid_scatra, const bool artery_coupl)
{
  // Discretization initialization scheme:
  // 1. Structural discretization - provided as input
  // 2. Poro-fluid discretization - cloned from structural discretization
  // 3. Scatra discretization - cloned from structural discretization
  //
  // For artery-coupled simulations:
  // 4. Artery discretization - provided as input
  // 5. Artery-scatra discretization - cloned from artery discretization

  setup_discretizations_and_field_coupling_porofluid_elast(porofluid_elast_algorithm_deps,
      struct_disname, fluid_disname, ndsporo_disp, ndsporo_vel, ndsporo_solidpressure);

  std::shared_ptr<Core::FE::Discretization> structdis =
      porofluid_elast_algorithm_deps.discretization_by_name(struct_disname);
  std::shared_ptr<Core::FE::Discretization> fluiddis =
      porofluid_elast_algorithm_deps.discretization_by_name(fluid_disname);
  std::shared_ptr<Core::FE::Discretization> scatradis =
      porofluid_elast_algorithm_deps.discretization_by_name(scatra_disname);

  // fill scatra discretization by cloning structure discretization
  Core::FE::clone_discretization<PoroElastScaTra::Utils::PoroScatraCloneStrategy>(
      *structdis, *scatradis, *porofluid_elast_algorithm_deps.cloning_material_map);
  scatradis->fill_complete();

  // the problem is two way coupled, thus each discretization must know the other discretization

  // build a proxy of the structure discretization for the scatra field
  std::shared_ptr<Core::DOFSets::DofSetInterface> structdofset = structdis->get_dof_set_proxy();
  // build a proxy of the fluid discretization for the scatra field
  std::shared_ptr<Core::DOFSets::DofSetInterface> fluiddofset = fluiddis->get_dof_set_proxy();
  // build a proxy of the fluid discretization for the structure/fluid field
  std::shared_ptr<Core::DOFSets::DofSetInterface> scatradofset = scatradis->get_dof_set_proxy();

  // check if ScatraField has 2 discretizations, so that coupling is possible
  if (scatradis->add_dof_set(structdofset) != 1)
    FOUR_C_THROW("unexpected dof sets in scatra field");
  if (scatradis->add_dof_set(fluiddofset) != 2) FOUR_C_THROW("unexpected dof sets in scatra field");
  if (scatradis->add_dof_set(fluiddis->get_dof_set_proxy(ndsporo_solidpressure)) != 3)
    FOUR_C_THROW("unexpected dof sets in scatra field");
  if (structdis->add_dof_set(scatradofset) != 3)
    FOUR_C_THROW("unexpected dof sets in structure field");

  ndsporofluid_scatra = fluiddis->add_dof_set(scatradofset);
  if (ndsporofluid_scatra != 3) FOUR_C_THROW("unexpected dof sets in fluid field");

  structdis->fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });
  fluiddis->fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });
  scatradis->fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });

  std::map<int, std::set<int>> nearby_ele_pairs;
  if (artery_coupl)
  {
    nearby_ele_pairs = setup_discretizations_and_field_coupling_artery(
        porofluid_elast_algorithm_deps, struct_disname);

    std::shared_ptr<Core::FE::Discretization> artdis =
        porofluid_elast_algorithm_deps.discretization_by_name("artery");
    std::shared_ptr<Core::FE::Discretization> artscatradis =
        porofluid_elast_algorithm_deps.discretization_by_name("artery_scatra");

    if (!artdis->filled()) FOUR_C_THROW("artery discretization should be filled at this point");

    // fill artery scatra discretization by cloning artery discretization
    Core::FE::clone_discretization<Arteries::ArteryScatraCloneStrategy>(
        *artdis, *artscatradis, *porofluid_elast_algorithm_deps.cloning_material_map);
    artscatradis->fill_complete();

    std::shared_ptr<Core::DOFSets::DofSetInterface> arterydofset = artdis->get_dof_set_proxy();
    std::shared_ptr<Core::DOFSets::DofSetInterface> artscatradofset =
        artscatradis->get_dof_set_proxy();

    // get MAXNUMSEGPERARTELE
    const int maxnumsegperele =
        porofluid_elast_algorithm_deps.porofluid_pressure_based_dynamic_parameters
            ->sublist("artery_coupling")
            .get<int>("maximum_number_of_segments_per_artery_element");

    // curr_seg_lengths: defined as element-wise quantity
    std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetaux;
    dofsetaux =
        std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(0, maxnumsegperele, 0, false);
    // add it to artery-scatra discretization
    artscatradis->add_dof_set(dofsetaux);

    // check if ScatraField has 2 discretizations, so that coupling is possible
    if (artscatradis->add_dof_set(arterydofset) != 2)
      FOUR_C_THROW("unexpected dof sets in artscatra field");

    // check if ArteryField has 2 discretizations, so that coupling is possible
    if (artdis->add_dof_set(artscatradofset) != 2)
      FOUR_C_THROW("unexpected dof sets in artery field");

    artscatradis->fill_complete({
        .assign_degrees_of_freedom = true,
        .init_elements = false,
        .do_boundary_conditions = false,
    });
  }

  return nearby_ele_pairs;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::assign_material_pointers_porofluid_elast_scatra(
    const PoroPressureBased::PorofluidElastAlgorithmDeps& porofluid_elast_algorithm_deps,
    const std::string& struct_disname, const std::string& fluid_disname,
    const std::string& scatra_disname, const bool artery_coupl)
{
  PoroPressureBased::assign_material_pointers_porofluid_elast(
      porofluid_elast_algorithm_deps, struct_disname, fluid_disname);

  std::shared_ptr<Core::FE::Discretization> structdis =
      porofluid_elast_algorithm_deps.discretization_by_name(struct_disname);
  std::shared_ptr<Core::FE::Discretization> fluiddis =
      porofluid_elast_algorithm_deps.discretization_by_name(fluid_disname);
  std::shared_ptr<Core::FE::Discretization> scatradis =
      porofluid_elast_algorithm_deps.discretization_by_name(scatra_disname);

  PoroElast::Utils::set_material_pointers_matching_grid(*structdis, *scatradis);
  PoroElast::Utils::set_material_pointers_matching_grid(*fluiddis, *scatradis);

  if (artery_coupl)
  {
    std::shared_ptr<Core::FE::Discretization> arterydis =
        porofluid_elast_algorithm_deps.discretization_by_name("artery");
    std::shared_ptr<Core::FE::Discretization> artscatradis =
        porofluid_elast_algorithm_deps.discretization_by_name("artery_scatra");

    Arteries::Utils::set_material_pointers_matching_grid(*arterydis, *artscatradis);
  }
}

FOUR_C_NAMESPACE_CLOSE
