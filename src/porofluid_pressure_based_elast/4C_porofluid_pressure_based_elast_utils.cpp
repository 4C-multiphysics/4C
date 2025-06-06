// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_utils.hpp"

#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_bio.hpp"
#include "4C_poroelast_utils.hpp"
#include "4C_porofluid_pressure_based_elast_artery_coupling.hpp"
#include "4C_porofluid_pressure_based_elast_base.hpp"
#include "4C_porofluid_pressure_based_elast_clonestrategy.hpp"
#include "4C_porofluid_pressure_based_elast_monolithic.hpp"
#include "4C_porofluid_pressure_based_elast_partitioned.hpp"
#include "4C_porofluid_pressure_based_ele.hpp"
#include "4C_porofluid_pressure_based_utils.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | setup discretizations and dofsets                         vuong 08/16 |
 *----------------------------------------------------------------------*/
std::map<int, std::set<int>>
PoroPressureBased::setup_discretizations_and_field_coupling_porofluid_elast(MPI_Comm comm,
    const std::string& struct_disname, const std::string& fluid_disname, int& nds_disp,
    int& nds_vel, int& nds_solidpressure)
{
  // Scheme   : the structure discretization is received from the input.
  //            Then, a poro fluid disc. is cloned.
  //            If an artery discretization with non-matching coupling is present, we first
  //            redistribute

  Global::Problem* problem = Global::Problem::instance();

  // 1.-Initialization.
  std::shared_ptr<Core::FE::Discretization> structdis = problem->get_dis(struct_disname);

  // possible interaction partners [artelegid; contelegid_1, ... contelegid_n]
  std::map<int, std::set<int>> nearby_ele_pairs;

  if (Global::Problem::instance()->does_exist_dis("artery"))
  {
    std::shared_ptr<Core::FE::Discretization> arterydis = nullptr;
    arterydis = Global::Problem::instance()->get_dis("artery");

    // get coupling method
    auto arterycoupl =
        Teuchos::getIntegralValue<Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod>(
            problem->poro_fluid_multi_phase_dynamic_params().sublist("ARTERY COUPLING"),
            "ARTERY_COUPLING_METHOD");

    // lateral surface coupling active?
    const bool evaluate_on_lateral_surface = problem->poro_fluid_multi_phase_dynamic_params()
                                                 .sublist("ARTERY COUPLING")
                                                 .get<bool>("LATERAL_SURFACE_COUPLING");

    // get MAXNUMSEGPERARTELE
    const int maxnumsegperele = problem->poro_fluid_multi_phase_dynamic_params()
                                    .sublist("ARTERY COUPLING")
                                    .get<int>("MAXNUMSEGPERARTELE");

    // curr_seg_lengths: defined as element-wise quantity
    std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetaux;
    dofsetaux =
        std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(0, maxnumsegperele, 0, false);
    // add it to artery discretization
    arterydis->add_dof_set(dofsetaux);

    switch (arterycoupl)
    {
      case Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::gpts:
      case Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::mp:
      case Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::ntp:
      {
        // perform extended ghosting on artery discretization
        nearby_ele_pairs = PoroPressureBased::extended_ghosting_artery_discretization(
            *structdis, arterydis, evaluate_on_lateral_surface, arterycoupl);
        break;
      }
      default:
      {
        break;
      }
    }
    if (!arterydis->filled()) arterydis->fill_complete();
  }

  std::shared_ptr<Core::FE::Discretization> fluiddis = problem->get_dis(fluid_disname);
  if (!structdis->filled()) structdis->fill_complete();
  if (!fluiddis->filled()) fluiddis->fill_complete();

  if (fluiddis->num_global_nodes() == 0)
  {
    // fill poro fluid discretization by cloning structure discretization
    Core::FE::clone_discretization<PorofluidCloneStrategy>(
        *structdis, *fluiddis, Global::Problem::instance()->cloning_material_map());
  }
  else
  {
    FOUR_C_THROW("Fluid discretization given in input file. This is not supported!");
  }

  structdis->fill_complete();
  fluiddis->fill_complete();

  // build a proxy of the structure discretization for the scatra field
  std::shared_ptr<Core::DOFSets::DofSetInterface> structdofset = structdis->get_dof_set_proxy();
  // build a proxy of the scatra discretization for the structure field
  std::shared_ptr<Core::DOFSets::DofSetInterface> fluiddofset = fluiddis->get_dof_set_proxy();

  // assign structure dof set to fluid and save the dofset number
  nds_disp = fluiddis->add_dof_set(structdofset);
  if (nds_disp != 1) FOUR_C_THROW("unexpected dof sets in porofluid field");
  // velocities live on same dofs as displacements
  nds_vel = nds_disp;

  if (structdis->add_dof_set(fluiddofset) != 1)
    FOUR_C_THROW("unexpected dof sets in structure field");

  // build auxiliary dofset for postprocessing solid pressures
  std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetaux =
      std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(1, 0, 0, false);
  nds_solidpressure = fluiddis->add_dof_set(dofsetaux);
  // add it also to the solid field
  structdis->add_dof_set(fluiddis->get_dof_set_proxy(nds_solidpressure));

  structdis->fill_complete();
  fluiddis->fill_complete();

  return nearby_ele_pairs;
}

/*----------------------------------------------------------------------*
 | exchange material pointers of both discretizations       vuong 08/16 |
 *----------------------------------------------------------------------*/
void PoroPressureBased::assign_material_pointers_porofluid_elast(
    const std::string& struct_disname, const std::string& fluid_disname)
{
  Global::Problem* problem = Global::Problem::instance();

  std::shared_ptr<Core::FE::Discretization> structdis = problem->get_dis(struct_disname);
  std::shared_ptr<Core::FE::Discretization> fluiddis = problem->get_dis(fluid_disname);

  PoroElast::Utils::set_material_pointers_matching_grid(*structdis, *fluiddis);
}

/*----------------------------------------------------------------------*
 | create algorithm                                                      |
 *----------------------------------------------------------------------*/
std::shared_ptr<PoroPressureBased::PorofluidElastAlgorithm>
PoroPressureBased::create_algorithm_porofluid_elast(
    PoroPressureBased::SolutionSchemePorofluidElast solscheme,
    const Teuchos::ParameterList& timeparams, MPI_Comm comm)
{
  // Creation of Coupled Problem algorithm.
  std::shared_ptr<PoroPressureBased::PorofluidElastAlgorithm> algo = nullptr;

  switch (solscheme)
  {
    case SolutionSchemePorofluidElast::twoway_partitioned:
    {
      // call constructor
      algo =
          std::make_shared<PoroPressureBased::PorofluidElastPartitionedAlgorithm>(comm, timeparams);
      break;
    }
    case SolutionSchemePorofluidElast::twoway_monolithic:
    {
      const bool artery_coupl = timeparams.get<bool>("ARTERY_COUPLING");
      if (!artery_coupl)
      {
        // call constructor
        algo = std::make_shared<PoroPressureBased::PorofluidElastMonolithicAlgorithm>(
            comm, timeparams);
      }
      else
      {
        // call constructor
        algo = std::make_shared<PoroPressureBased::PorofluidElastArteryCouplingAlgorithm>(
            comm, timeparams);
      }
      break;
    }
    default:
      FOUR_C_THROW("Unknown time-integration scheme for multiphase poro fluid problem");
      break;
  }

  return algo;
}


FOUR_C_NAMESPACE_CLOSE
