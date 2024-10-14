/*----------------------------------------------------------------------*/
/*! \file
 \brief utils methods for for porous multiphase flow through elastic medium problems

   \level 3

 *----------------------------------------------------------------------*/

#include "4C_poromultiphase_utils.hpp"

#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_bio.hpp"
#include "4C_poroelast_utils.hpp"
#include "4C_porofluidmultiphase_ele.hpp"
#include "4C_porofluidmultiphase_utils.hpp"
#include "4C_poromultiphase_adapter.hpp"
#include "4C_poromultiphase_monolithic_twoway.hpp"
#include "4C_poromultiphase_partitioned_twoway.hpp"
#include "4C_poromultiphase_utils_clonestrategy.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | setup discretizations and dofsets                         vuong 08/16 |
 *----------------------------------------------------------------------*/
std::map<int, std::set<int>> POROMULTIPHASE::Utils::setup_discretizations_and_field_coupling(
    const Epetra_Comm& comm, const std::string& struct_disname, const std::string& fluid_disname,
    int& nds_disp, int& nds_vel, int& nds_solidpressure)
{
  // Scheme   : the structure discretization is received from the input.
  //            Then, a poro fluid disc. is cloned.
  //            If an artery discretization with non-matching coupling is present, we first
  //            redistribute

  Global::Problem* problem = Global::Problem::instance();

  // 1.-Initialization.
  Teuchos::RCP<Core::FE::Discretization> structdis = problem->get_dis(struct_disname);

  // possible interaction partners [artelegid; contelegid_1, ... contelegid_n]
  std::map<int, std::set<int>> nearbyelepairs;

  if (Global::Problem::instance()->does_exist_dis("artery"))
  {
    Teuchos::RCP<Core::FE::Discretization> arterydis = Teuchos::null;
    arterydis = Global::Problem::instance()->get_dis("artery");

    // get coupling method
    auto arterycoupl =
        Teuchos::getIntegralValue<Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod>(
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
    Teuchos::RCP<Core::DOFSets::DofSetInterface> dofsetaux;
    dofsetaux =
        Teuchos::make_rcp<Core::DOFSets::DofSetPredefinedDoFNumber>(0, maxnumsegperele, 0, false);
    // add it to artery discretization
    arterydis->add_dof_set(dofsetaux);

    switch (arterycoupl)
    {
      case Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::gpts:
      case Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::mp:
      case Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::ntp:
      {
        // perform extended ghosting on artery discretization
        nearbyelepairs = POROFLUIDMULTIPHASE::Utils::extended_ghosting_artery_discretization(
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

  Teuchos::RCP<Core::FE::Discretization> fluiddis = problem->get_dis(fluid_disname);
  if (!structdis->filled()) structdis->fill_complete();
  if (!fluiddis->filled()) fluiddis->fill_complete();

  if (fluiddis->num_global_nodes() == 0)
  {
    // fill poro fluid discretization by cloning structure discretization
    Core::FE::clone_discretization<POROMULTIPHASE::Utils::PoroFluidMultiPhaseCloneStrategy>(
        *structdis, *fluiddis, Global::Problem::instance()->cloning_material_map());
  }
  else
  {
    FOUR_C_THROW("Fluid discretization given in input file. This is not supported!");
  }

  structdis->fill_complete();
  fluiddis->fill_complete();

  // build a proxy of the structure discretization for the scatra field
  Teuchos::RCP<Core::DOFSets::DofSetInterface> structdofset = structdis->get_dof_set_proxy();
  // build a proxy of the scatra discretization for the structure field
  Teuchos::RCP<Core::DOFSets::DofSetInterface> fluiddofset = fluiddis->get_dof_set_proxy();

  // assign structure dof set to fluid and save the dofset number
  nds_disp = fluiddis->add_dof_set(structdofset);
  if (nds_disp != 1) FOUR_C_THROW("unexpected dof sets in porofluid field");
  // velocities live on same dofs as displacements
  nds_vel = nds_disp;

  if (structdis->add_dof_set(fluiddofset) != 1)
    FOUR_C_THROW("unexpected dof sets in structure field");

  // build auxiliary dofset for postprocessing solid pressures
  Teuchos::RCP<Core::DOFSets::DofSetInterface> dofsetaux =
      Teuchos::make_rcp<Core::DOFSets::DofSetPredefinedDoFNumber>(1, 0, 0, false);
  nds_solidpressure = fluiddis->add_dof_set(dofsetaux);
  // add it also to the solid field
  structdis->add_dof_set(fluiddis->get_dof_set_proxy(nds_solidpressure));

  structdis->fill_complete();
  fluiddis->fill_complete();

  return nearbyelepairs;
}

/*----------------------------------------------------------------------*
 | exchange material pointers of both discretizations       vuong 08/16 |
 *----------------------------------------------------------------------*/
void POROMULTIPHASE::Utils::assign_material_pointers(
    const std::string& struct_disname, const std::string& fluid_disname)
{
  Global::Problem* problem = Global::Problem::instance();

  Teuchos::RCP<Core::FE::Discretization> structdis = problem->get_dis(struct_disname);
  Teuchos::RCP<Core::FE::Discretization> fluiddis = problem->get_dis(fluid_disname);

  PoroElast::Utils::set_material_pointers_matching_grid(*structdis, *fluiddis);
}

/*----------------------------------------------------------------------*
 | create algorithm                                                      |
 *----------------------------------------------------------------------*/
Teuchos::RCP<POROMULTIPHASE::PoroMultiPhase>
POROMULTIPHASE::Utils::create_poro_multi_phase_algorithm(
    Inpar::POROMULTIPHASE::SolutionSchemeOverFields solscheme,
    const Teuchos::ParameterList& timeparams, const Epetra_Comm& comm)
{
  // Creation of Coupled Problem algorithm.
  Teuchos::RCP<POROMULTIPHASE::PoroMultiPhase> algo = Teuchos::null;

  switch (solscheme)
  {
    case Inpar::POROMULTIPHASE::solscheme_twoway_partitioned:
    {
      // call constructor
      algo = Teuchos::make_rcp<POROMULTIPHASE::PoroMultiPhasePartitionedTwoWay>(comm, timeparams);
      break;
    }
    case Inpar::POROMULTIPHASE::solscheme_twoway_monolithic:
    {
      const bool artery_coupl = timeparams.get<bool>("ARTERY_COUPLING");
      if (!artery_coupl)
      {
        // call constructor
        algo = Teuchos::make_rcp<POROMULTIPHASE::PoroMultiPhaseMonolithicTwoWay>(comm, timeparams);
      }
      else
      {
        // call constructor
        algo = Teuchos::make_rcp<POROMULTIPHASE::PoroMultiPhaseMonolithicTwoWayArteryCoupling>(
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

/*----------------------------------------------------------------------*
 | calculate vector norm                             kremheller 07/17   |
 *----------------------------------------------------------------------*/
double POROMULTIPHASE::Utils::calculate_vector_norm(
    const enum Inpar::POROMULTIPHASE::VectorNorm norm, const Core::LinAlg::Vector<double>& vect)
{
  // L1 norm
  // norm = sum_0^i vect[i]
  if (norm == Inpar::POROMULTIPHASE::norm_l1)
  {
    double vectnorm;
    vect.Norm1(&vectnorm);
    return vectnorm;
  }
  // L2/Euclidian norm
  // norm = sqrt{sum_0^i vect[i]^2 }
  else if (norm == Inpar::POROMULTIPHASE::norm_l2)
  {
    double vectnorm;
    vect.Norm2(&vectnorm);
    return vectnorm;
  }
  // RMS norm
  // norm = sqrt{sum_0^i vect[i]^2 }/ sqrt{length_vect}
  else if (norm == Inpar::POROMULTIPHASE::norm_rms)
  {
    double vectnorm;
    vect.Norm2(&vectnorm);
    return vectnorm / sqrt((double)vect.GlobalLength());
  }
  // infinity/maximum norm
  // norm = max( vect[i] )
  else if (norm == Inpar::POROMULTIPHASE::norm_inf)
  {
    double vectnorm;
    vect.NormInf(&vectnorm);
    return vectnorm;
  }
  // norm = sum_0^i vect[i]/length_vect
  else if (norm == Inpar::POROMULTIPHASE::norm_l1_scaled)
  {
    double vectnorm;
    vect.Norm1(&vectnorm);
    return vectnorm / ((double)vect.GlobalLength());
  }
  else
  {
    FOUR_C_THROW("Cannot handle vector norm");
    return 0;
  }
}  // calculate_vector_norm()

/*----------------------------------------------------------------------*
 |                                                    kremheller 03/17  |
 *----------------------------------------------------------------------*/
void POROMULTIPHASE::print_logo()
{
  std::cout << "This is a Porous Media problem with multiphase flow and deformation" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "              +----------+" << std::endl;
  std::cout << "              |  Krebs-  |" << std::endl;
  std::cout << "              |  Modell  |" << std::endl;
  std::cout << "              +----------+" << std::endl;
  std::cout << "              |          |" << std::endl;
  std::cout << "              |          |" << std::endl;
  std::cout << " /\\           |          /\\" << std::endl;
  std::cout << "( /   @ @    (|)        ( /   @ @    ()" << std::endl;
  std::cout << " \\  __| |__  /           \\  __| |__  /" << std::endl;
  std::cout << "  \\/   \"   \\/             \\/   \"   \\/" << std::endl;
  std::cout << " /-|       |-\\           /-|       |-\\" << std::endl;
  std::cout << "/ /-\\     /-\\ \\         / /-\\     /-\\ \\" << std::endl;
  std::cout << " / /-`---'-\\ \\           / /-`---'-\\ \\" << std::endl;
  std::cout << "  /         \\             /         \\" << std::endl;

  return;
}

FOUR_C_NAMESPACE_CLOSE
