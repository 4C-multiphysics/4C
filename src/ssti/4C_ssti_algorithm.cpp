// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssti_algorithm.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_adapter_str_factory.hpp"
#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_adapter_str_structure_new.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_ssti.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"
#include "4C_scatra_utils.hpp"
#include "4C_ssi_utils.hpp"
#include "4C_ssti_monolithic.hpp"
#include "4C_ssti_resulttest.hpp"
#include "4C_ssti_utils.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
SSTI::SSTIAlgorithm::SSTIAlgorithm(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : AlgorithmBase(comm, globaltimeparams),
      iter_(0),
      scatra_(nullptr),
      structure_(nullptr),
      struct_adapterbase_ptr_(nullptr),
      thermo_(nullptr),
      meshtying_strategy_scatra_(nullptr),
      meshtying_strategy_thermo_(nullptr),
      ssti_structure_meshtying_(nullptr),
      interfacemeshtying_(Global::Problem::instance()
                              ->get_dis("structure")
                              ->get_condition("SSTIInterfaceMeshtying") != nullptr),
      isinit_(false),
      issetup_(false)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::SSTIAlgorithm::init(MPI_Comm comm, const Teuchos::ParameterList& sstitimeparams,
    const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& thermoparams,
    const Teuchos::ParameterList& structparams)
{
  // reset the setup flag
  issetup_ = false;

  // get the global problem
  Global::Problem* problem = Global::Problem::instance();

  problem->get_dis("structure")->fill_complete(true, true, true);
  problem->get_dis("scatra")->fill_complete(true, true, true);
  problem->get_dis("thermo")->fill_complete(true, true, true);

  // clone scatra discretization from structure discretization first. Afterwards, clone thermo
  // discretization from scatra discretization
  clone_discretizations(comm);

  std::shared_ptr<Core::FE::Discretization> structuredis = problem->get_dis("structure");
  std::shared_ptr<Core::FE::Discretization> scatradis = problem->get_dis("scatra");
  std::shared_ptr<Core::FE::Discretization> thermodis = problem->get_dis("thermo");

  // safety check
  if (Teuchos::getIntegralValue<Inpar::Solid::IntegrationStrategy>(structparams, "INT_STRATEGY") ==
      Inpar::Solid::IntegrationStrategy::int_old)
    FOUR_C_THROW("Old structural time integration is not supported");

  struct_adapterbase_ptr_ = Adapter::build_structure_algorithm(structparams);

  // initialize structure base algorithm
  struct_adapterbase_ptr_->init(
      sstitimeparams, const_cast<Teuchos::ParameterList&>(structparams), structuredis);

  // create and initialize scatra problem and thermo problem
  scatra_ = std::make_shared<Adapter::ScaTraBaseAlgorithm>(sstitimeparams,
      SSI::Utils::modify_sca_tra_params(scatraparams),
      problem->solver_params(scatraparams.get<int>("LINEAR_SOLVER")), "scatra", true);
  scatra_->init();
  scatra_->scatra_field()->set_number_of_dof_set_displacement(1);
  scatra_->scatra_field()->set_number_of_dof_set_velocity(1);
  scatra_->scatra_field()->set_number_of_dof_set_thermo(2);
  thermo_ = std::make_shared<Adapter::ScaTraBaseAlgorithm>(sstitimeparams,
      clone_thermo_params(scatraparams, thermoparams),
      problem->solver_params(thermoparams.get<int>("LINEAR_SOLVER")), "thermo", true);
  thermo_->init();
  thermo_->scatra_field()->set_number_of_dof_set_displacement(1);
  thermo_->scatra_field()->set_number_of_dof_set_velocity(1);
  thermo_->scatra_field()->set_number_of_dof_set_scatra(2);
  thermo_->scatra_field()->set_number_of_dof_set_thermo(3);

  // distribute dofsets among subproblems
  std::shared_ptr<Core::DOFSets::DofSetInterface> scatradofset = scatradis->get_dof_set_proxy();
  std::shared_ptr<Core::DOFSets::DofSetInterface> structdofset = structuredis->get_dof_set_proxy();
  std::shared_ptr<Core::DOFSets::DofSetInterface> thermodofset = thermodis->get_dof_set_proxy();
  if (scatradis->add_dof_set(structdofset) != 1)
    FOUR_C_THROW("unexpected dof sets in scatra field");
  if (scatradis->add_dof_set(thermodofset) != 2)
    FOUR_C_THROW("unexpected dof sets in scatra field");
  if (structuredis->add_dof_set(scatradofset) != 1)
    FOUR_C_THROW("unexpected dof sets in structure field");
  if (structuredis->add_dof_set(thermodofset) != 2)
    FOUR_C_THROW("unexpected dof sets in structure field");
  if (thermodis->add_dof_set(structdofset) != 1)
    FOUR_C_THROW("unexpected dof sets in thermo field");
  if (thermodis->add_dof_set(scatradofset) != 2)
    FOUR_C_THROW("unexpected dof sets in thermo field");
  if (thermodis->add_dof_set(thermodofset) != 3)
    FOUR_C_THROW("unexpected dof sets in thermo field");

  // is adaptive time stepping activated?
  if (sstitimeparams.get<bool>("ADAPTIVE_TIMESTEPPING"))
  {
    // safety check: adaptive time stepping in one of the subproblems?
    if (!scatraparams.get<bool>("ADAPTIVE_TIMESTEPPING"))
      FOUR_C_THROW(
          "Must provide adaptive time stepping in one of the subproblems. (Currently just ScaTra)");
    if (Teuchos::getIntegralValue<Inpar::Solid::TimAdaKind>(
            structparams.sublist("TIMEADAPTIVITY"), "KIND") != Inpar::Solid::timada_kind_none)
      FOUR_C_THROW("Adaptive time stepping in SSI currently just from ScaTra");
    if (Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(structparams, "DYNAMICTYPE") ==
        Inpar::Solid::DynamicType::AdamsBashforth2)
      FOUR_C_THROW("Currently, only one step methods are allowed for adaptive time stepping");
  }

  // now we can finally fill our discretizations
  // reinitialization of the structural elements is
  // vital for parallelization here!
  problem->get_dis("structure")->fill_complete(true, true, true);
  problem->get_dis("scatra")->fill_complete(true, false, true);
  problem->get_dis("thermo")->fill_complete(true, false, true);

  isinit_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::SSTIAlgorithm::setup()
{
  // get the global problem
  Global::Problem* problem = Global::Problem::instance();

  // check initialization
  check_is_init();

  // set up scatra and thermo problem
  scatra_->scatra_field()->setup();
  thermo_->scatra_field()->setup();

  // pass initial scalar field to structural discretization to correctly compute initial
  // accelerations
  problem->get_dis("structure")->set_state(1, "scalarfield", *scatra_->scatra_field()->phinp());
  problem->get_dis("structure")->set_state(2, "temperature", *thermo_->scatra_field()->phinp());

  // set up structural base algorithm
  struct_adapterbase_ptr_->setup();

  // get wrapper and cast it to specific type
  if (structure_ == nullptr)
    structure_ = std::dynamic_pointer_cast<Adapter::SSIStructureWrapper>(
        struct_adapterbase_ptr_->structure_field());
  if (structure_ == nullptr) FOUR_C_THROW("No valid pointer to Adapter::SSIStructureWrapper !");

  // check maps from subproblems
  if (scatra_->scatra_field()->dof_row_map()->NumGlobalElements() == 0)
    FOUR_C_THROW("Scalar transport discretization does not have any degrees of freedom!");
  if (thermo_->scatra_field()->dof_row_map()->NumGlobalElements() == 0)
    FOUR_C_THROW("Scalar transport discretization does not have any degrees of freedom!");
  if (structure_->dof_row_map()->NumGlobalElements() == 0)
    FOUR_C_THROW("Structure discretization does not have any degrees of freedom!");

  // set up materials
  assign_material_pointers();

  // set up scatra-scatra interface coupling
  if (interface_meshtying())
  {
    // check for consistent parameterization of these conditions
    ScaTra::ScaTraUtils::check_consistency_with_s2_i_kinetics_condition(
        "SSTIInterfaceMeshtying", structure_field()->discretization());

    // extract meshtying strategy for scatra-scatra interface coupling on scatra discretization
    meshtying_strategy_scatra_ = std::dynamic_pointer_cast<const ScaTra::MeshtyingStrategyS2I>(
        scatra_->scatra_field()->strategy());

    // safety checks
    if (meshtying_strategy_scatra_ == nullptr)
      FOUR_C_THROW("Invalid scatra-scatra interface coupling strategy!");
    if (meshtying_strategy_scatra_->coupling_type() != Inpar::S2I::coupling_matching_nodes)
      FOUR_C_THROW("SSTI only implemented for interface coupling with matching interface nodes!");

    // extract meshtying strategy for scatra-scatra interface coupling on thermo discretization
    meshtying_strategy_thermo_ = std::dynamic_pointer_cast<const ScaTra::MeshtyingStrategyS2I>(
        thermo_->scatra_field()->strategy());
    if (meshtying_strategy_thermo_ == nullptr)
      FOUR_C_THROW("Invalid scatra-scatra interface coupling strategy!");
    if (meshtying_strategy_thermo_->coupling_type() != Inpar::S2I::coupling_matching_nodes)
      FOUR_C_THROW("SSTI only implemented for interface coupling with matching interface nodes!");

    // setup everything for SSTI structure meshtying
    ssti_structure_meshtying_ = std::make_shared<SSI::Utils::SSIMeshTying>(
        "SSTIInterfaceMeshtying", structure_->discretization(), true, true);
  }

  issetup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::SSTIAlgorithm::post_setup()
{
  // call the post_setup routine of the structure field
  structure_->post_setup();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::SSTIAlgorithm::clone_discretizations(MPI_Comm comm)
{
  // The structure discretization is received from the input.
  // Then, the scatra discretization is cloned.
  // Then, the thermo discretization is cloned.

  Global::Problem* problem = Global::Problem::instance();

  std::shared_ptr<Core::FE::Discretization> structdis = problem->get_dis("structure");
  std::shared_ptr<Core::FE::Discretization> scatradis = problem->get_dis("scatra");
  std::shared_ptr<Core::FE::Discretization> thermodis = problem->get_dis("thermo");

  if (scatradis->num_global_nodes() == 0)
  {
    Core::FE::clone_discretization<SSTI::SSTIScatraStructureCloneStrategy>(
        *structdis, *scatradis, Global::Problem::instance()->cloning_material_map());
    scatradis->fill_complete();
    Core::FE::clone_discretization<SSTI::SSTIScatraThermoCloneStrategy>(
        *scatradis, *thermodis, Global::Problem::instance()->cloning_material_map());
    thermodis->fill_complete();
  }
  else
    FOUR_C_THROW("Only matching nodes in SSTI");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::SSTIAlgorithm::read_restart(int restart)
{
  structure_field()->read_restart(restart);
  scatra_field()->read_restart(restart);
  thermo_field()->read_restart(restart);

  set_time_step(structure_->time_old(), restart);

  // Material pointers to other field were deleted during read_restart(). They need to be reset.
  assign_material_pointers();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSTI::SSTIAlgorithm::test_results(MPI_Comm comm) const
{
  Global::Problem* problem = Global::Problem::instance();

  problem->add_field_test(structure_->create_field_test());
  problem->add_field_test(scatra_->create_scatra_field_test());
  problem->add_field_test(thermo_->create_scatra_field_test());
  problem->add_field_test(std::make_shared<SSTI::SSTIResultTest>(*this));
  problem->test_all(comm);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSTI::SSTIAlgorithm::distribute_structure_solution()
{
  scatra_field()->apply_mesh_movement(structure_->dispnp());
  thermo_field()->apply_mesh_movement(structure_->dispnp());

  // convective velocity is set to zero
  const auto convective_velocity = Core::LinAlg::create_vector(*structure_->dof_row_map());

  scatra_field()->set_velocity_field(convective_velocity, nullptr, structure_->velnp(), nullptr);
  thermo_field()->set_velocity_field(convective_velocity, nullptr, structure_->velnp(), nullptr);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSTI::SSTIAlgorithm::distribute_scatra_solution()
{
  structure_field()->discretization()->set_state(
      1, "scalarfield", *scatra_->scatra_field()->phinp());
  thermo_field()->discretization()->set_state(2, "scatra", *scatra_->scatra_field()->phinp());

  if (interfacemeshtying_)
  {
    // pass master-side scatra degrees of freedom to thermo discretization
    const std::shared_ptr<Core::LinAlg::Vector<double>> imasterphinp =
        Core::LinAlg::create_vector(*scatra_field()->discretization()->dof_row_map(), true);
    meshtying_strategy_scatra_->interface_maps()->insert_vector(
        *meshtying_strategy_scatra_->coupling_adapter()->master_to_slave(
            *meshtying_strategy_scatra_->interface_maps()->extract_vector(
                *scatra_field()->phinp(), 2)),
        1, *imasterphinp);
    thermo_field()->discretization()->set_state(2, "imasterscatra", *imasterphinp);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSTI::SSTIAlgorithm::distribute_thermo_solution()
{
  structure_field()->discretization()->set_state(
      2, "temperature", *thermo_->scatra_field()->phinp());

  scatra_field()->discretization()->set_state(2, "thermo", *thermo_->scatra_field()->phinp());

  if (interfacemeshtying_)
  {
    // extract master side temperatures and copy to slave side dof map
    const std::shared_ptr<Core::LinAlg::Vector<double>> imastertempnp =
        Core::LinAlg::create_vector(*thermo_field()->discretization()->dof_row_map(), true);
    meshtying_strategy_thermo_->interface_maps()->insert_vector(
        *meshtying_strategy_thermo_->coupling_adapter()->master_to_slave(
            *meshtying_strategy_thermo_->interface_maps()->extract_vector(
                *thermo_field()->phinp(), 2)),
        1, *imastertempnp);

    // extract slave side temperatures
    const std::shared_ptr<Core::LinAlg::Vector<double>> islavetempnp =
        Core::LinAlg::create_vector(*thermo_field()->discretization()->dof_row_map(), true);
    meshtying_strategy_thermo_->interface_maps()->insert_vector(
        *meshtying_strategy_thermo_->interface_maps()->extract_vector(*thermo_field()->phinp(), 1),
        1, *islavetempnp);

    // set master side temperature to thermo discretization
    thermo_field()->discretization()->set_state(3, "imastertemp", *imastertempnp);

    // set master and slave side temperature to scatra discretization
    scatra_field()->discretization()->set_state(2, "islavetemp", *islavetempnp);
    scatra_field()->discretization()->set_state(2, "imastertemp", *imastertempnp);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSTI::SSTIAlgorithm::distribute_solution_all_fields()
{
  distribute_scatra_solution();
  distribute_structure_solution();
  distribute_thermo_solution();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSTI::SSTIAlgorithm::distribute_dt_from_scatra()
{
  // get adapted time, timestep, and step (incremented)
  const double newtime = scatra_field()->time();
  const double newtimestep = scatra_field()->dt();
  const int newstep = step();

  // change time step size of thermo according to ScaTra
  thermo_field()->set_dt(newtimestep);
  // change current time and time step number of thermo field according to ScaTra
  // The incremental reduction is is needed, because time and step are incremented in
  // prepare_time_step() of thermo field
  thermo_field()->set_time_step(newtime - newtimestep, newstep - 1);

  // change current time and time step number of structure according to ScaTra
  structure_field()->set_dt(newtimestep);
  structure_field()->set_timen(newtime);
  structure_field()->post_update();

  // change current time and time step of this algorithm according to ScaTra
  set_time_step(newtime, newstep);
  set_dt(newtimestep);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSTI::SSTIAlgorithm::assign_material_pointers()
{
  // scatra - structure
  const int numscatraelements = scatra_field()->discretization()->num_my_col_elements();
  for (int i = 0; i < numscatraelements; ++i)
  {
    Core::Elements::Element* scatratele = scatra_field()->discretization()->l_col_element(i);
    const int gid = scatratele->id();

    Core::Elements::Element* structele = structure_field()->discretization()->g_element(gid);

    // for coupling we add the source material to the target element and vice versa
    scatratele->add_material(structele->material());
    structele->add_material(scatratele->material());
  }

  // thermo - scatra
  const int numthermoelements = thermo_field()->discretization()->num_my_col_elements();
  for (int i = 0; i < numthermoelements; ++i)
  {
    Core::Elements::Element* thermotele = thermo_field()->discretization()->l_col_element(i);
    const int gid = thermotele->id();

    Core::Elements::Element* scatraele = scatra_field()->discretization()->g_element(gid);

    // for coupling we add the source material to the target element and vice versa
    thermotele->add_material(scatraele->material());
    scatraele->add_material(thermotele->material());
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<ScaTra::ScaTraTimIntImpl> SSTI::SSTIAlgorithm::scatra_field() const
{
  return scatra_->scatra_field();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<ScaTra::ScaTraTimIntImpl> SSTI::SSTIAlgorithm::thermo_field() const
{
  return thermo_->scatra_field();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSTI::SSTIAlgorithm::check_is_init()
{
  if (not isinit_) FOUR_C_THROW("init(...) was not called.");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::ParameterList SSTI::SSTIAlgorithm::clone_thermo_params(
    const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& thermoparams)
{
  auto thermoparams_copy = Teuchos::ParameterList(scatraparams);

  auto initial_field =
      Teuchos::getIntegralValue<Inpar::ScaTra::InitialField>(thermoparams, "INITIALFIELD");
  switch (initial_field)
  {
    case Inpar::ScaTra::initfield_field_by_function:
    case Inpar::ScaTra::initfield_field_by_condition:
    {
      thermoparams_copy.set("INITIALFIELD", initial_field);
      break;
    }
    default:
      FOUR_C_THROW("Initial field type for thermo not supported");
  }

  thermoparams_copy.set<int>("INITFUNCNO", thermoparams.get<int>("INITTHERMOFUNCT"));
  thermoparams_copy.sublist("S2I COUPLING").set<bool>("SLAVEONLY", false);

  if (Teuchos::getIntegralValue<Inpar::ScaTra::OutputScalarType>(scatraparams, "OUTPUTSCALARS") !=
      Inpar::ScaTra::outputscalars_none)
    thermoparams_copy.set<bool>("output_file_name_discretization", true);

  // adaptive time stepping only from scatra
  thermoparams_copy.set<bool>("ADAPTIVE_TIMESTEPPING", false);

  return thermoparams_copy;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<SSTI::SSTIAlgorithm> SSTI::build_ssti(
    Inpar::SSTI::SolutionScheme coupling, MPI_Comm comm, const Teuchos::ParameterList& sstiparams)
{
  std::shared_ptr<SSTI::SSTIAlgorithm> ssti = nullptr;
  switch (coupling)
  {
    case Inpar::SSTI::SolutionScheme::monolithic:
    {
      ssti = std::make_shared<SSTI::SSTIMono>(comm, sstiparams);
      break;
    }
    default:
      FOUR_C_THROW("unknown coupling algorithm for SSTI!");
  }
  return ssti;
}

FOUR_C_NAMESPACE_CLOSE
