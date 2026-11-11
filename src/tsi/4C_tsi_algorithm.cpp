// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_tsi_algorithm.hpp"

#include "4C_adapter_str_factory.hpp"
#include "4C_adapter_str_structure_new.hpp"
#include "4C_adapter_str_wrapper.hpp"
#include "4C_contact_lagrange_strategy_tsi.hpp"
#include "4C_contact_meshtying_contact_bridge.hpp"
#include "4C_contact_strategy_factory.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_mortar.hpp"
#include "4C_coupling_adapter_volmortar.hpp"
#include "4C_coupling_volmortar_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_mortar_multifield_coupling.hpp"
#include "4C_thermo_adapter.hpp"
#include "4C_tsi_input.hpp"
#include "4C_tsi_problem_access.hpp"
#include "4C_tsi_utils.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

//! Note: The order of calling the two BaseAlgorithm-constructors is
//! important here! In here control file entries are written. And these entries
//! define the order in which the filters handle the Discretizations, which in
//! turn defines the dof number ordering of the Discretizations.
/*----------------------------------------------------------------------*
 | constructor (public)                                      dano 12/09 |
 *----------------------------------------------------------------------*/
TSI::Algorithm::Algorithm(MPI_Comm comm)
    : AlgorithmBase(*TSI::Utils::problem_from_instance(), comm,
          TSI::Utils::tsi_dynamic_params_from_problem()),
      dispnp_(nullptr),
      tempnp_(nullptr),
      problem_(TSI::Utils::problem_from_instance()),
      matchinggrid_(TSI::Utils::tsi_dynamic_params_from_problem().get<bool>("MATCHINGGRID")),
      volcoupl_(nullptr)
{
  // access the structural discretization
  std::shared_ptr<Core::FE::Discretization> structdis = problem_->get_dis("structure");
  // access the thermo discretization
  std::shared_ptr<Core::FE::Discretization> thermodis = problem_->get_dis("thermo");
  // get the restart step
  const int restart = problem_->restart();

  if (!matchinggrid_)
  {
    // Scheme: non matching meshes --> volumetric mortar coupling...
    volcoupl_ = std::make_shared<Coupling::Adapter::MortarVolCoupl>();

    std::shared_ptr<Coupling::VolMortar::Utils::DefaultMaterialStrategy> materialstrategy =
        std::make_shared<TSI::Utils::TSIMaterialStrategy>();
    // init coupling adapter projection matrices
    volcoupl_->init(problem_->n_dim(), structdis, thermodis, nullptr, nullptr, nullptr, nullptr,
        materialstrategy);
    // redistribute discretizations to meet needs of volmortar coupling
    Teuchos::ParameterList binning_params = problem_->binning_strategy_params();
    Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
        "spatial_approximation_type", problem_->spatial_approximation_type(), binning_params);
    volcoupl_->redistribute(binning_params, problem_->output_control_file());
    // setup projection matrices
    volcoupl_->setup(problem_->volmortar_params(), problem_->cut_general_params());
  }

  if (Teuchos::getIntegralValue<Solid::IntegrationStrategy>(
          problem_->structural_dynamic_params(), "INT_STRATEGY") == Solid::int_old)
    FOUR_C_THROW("old structural time integration no longer supported in tsi");
  else
  {
    Thermo::BaseAlgorithm thermo(problem_->tsi_dynamic_params(), thermodis);
    thermo_ = thermo.thermo_field();

    //  // access structural dynamic params list which will be possibly modified while creating the
    //  time integrator
    const Teuchos::ParameterList& sdyn = problem_->structural_dynamic_params();
    std::shared_ptr<Adapter::StructureBaseAlgorithmNew> adapterbase_ptr =
        Adapter::build_structure_algorithm(*problem_, sdyn);
    adapterbase_ptr->init(
        problem_->tsi_dynamic_params(), const_cast<Teuchos::ParameterList&>(sdyn), structdis);

    // set the temperature; Monolithic does this in it's own constructor with potentially
    // redistributed discretizations
    if (Teuchos::getIntegralValue<TSI::SolutionSchemeOverFields>(problem_->tsi_dynamic_params(),
            "COUPALGO") != TSI::SolutionSchemeOverFields::Monolithic)
    {
      if (matchinggrid_)
        structdis->set_state(1, "temperature", *thermo_field()->tempnp());
      else
        structdis->set_state(
            1, "temperature", *volcoupl_->apply_vector_mapping12(*thermo_field()->tempnp()));
    }

    adapterbase_ptr->setup();
    structure_ =
        std::dynamic_pointer_cast<Adapter::StructureWrapper>(adapterbase_ptr->structure_field());

    if (restart &&
        Teuchos::getIntegralValue<TSI::SolutionSchemeOverFields>(problem_->tsi_dynamic_params(),
            "COUPALGO") == TSI::SolutionSchemeOverFields::Monolithic)
      structure_->setup();

    structure_field()->discretization()->clear_state(true);
  }

  // initialise displacement field needed for output()
  // (get noderowmap of discretisation for creating this multivector)
  // TODO: why nds 0 and not 1????
  dispnp_ = std::make_shared<Core::LinAlg::MultiVector<double>>(
      *(thermo_field()->discretization()->node_row_map()), 3, true);
  tempnp_ = std::make_shared<Core::LinAlg::MultiVector<double>>(
      *(structure_field()->discretization()->node_row_map()), 1, true);

  // setup coupling object for matching discretization
  if (matchinggrid_)
  {
    structure_thermo_coupling_ = std::make_shared<Coupling::Adapter::Coupling>();
    structure_thermo_coupling_->setup_coupling(*structure_field()->discretization(),
        *thermo_field()->discretization(), *structure_field()->discretization()->node_row_map(),
        *thermo_field()->discretization()->node_row_map(), 1, true);
  }

  // setup mortar coupling
  if (problem_->get_problem_type() == Core::ProblemType::tsi)
  {
    if (structure_field()->discretization()->has_condition("MortarMulti"))
    {
      mortar_coupling_ = std::make_shared<Mortar::MultiFieldCoupling>();
      mortar_coupling_->push_back_coupling(structure_field()->discretization(), 0,
          std::vector<int>(3, 1), problem_->mortar_coupling_params(),
          problem_->contact_dynamic_params(), problem_->binning_strategy_params(),
          {{"structure", problem_->get_dis("structure")}, {"thermo", problem_->get_dis("thermo")}},
          problem_->function_manager(), problem_->output_control_file(),
          problem_->spatial_approximation_type(), problem_->n_dim());
      mortar_coupling_->push_back_coupling(thermo_field()->discretization(), 0,
          std::vector<int>(1, 1), problem_->mortar_coupling_params(),
          problem_->contact_dynamic_params(), problem_->binning_strategy_params(),
          {{"structure", problem_->get_dis("structure")}, {"thermo", problem_->get_dis("thermo")}},
          problem_->function_manager(), problem_->output_control_file(),
          problem_->spatial_approximation_type(), problem_->n_dim());
    }
  }

  // reset states
  structure_field()->discretization()->clear_state(true);
  thermo_field()->discretization()->clear_state(true);
}

/*----------------------------------------------------------------------*
 | output (protected)                                        dano 12/09 |
 *----------------------------------------------------------------------*/
void TSI::Algorithm::output(bool forced_writerestart)
{
  thermo_field()->output(forced_writerestart);
  structure_field()->output(forced_writerestart);
}

void TSI::Algorithm::time_loop()
{
  // time loop
  while (not_finished())
  {
    // increment step, print header and apply coupling (only monolithic scheme)
    prepare_time_step();

    // integrate time step
    solve();

    // calculate stresses, strains, energies
    prepare_output();

    // update all single field solvers
    update();

    // write output to screen and files
    output();

  }  // not_finished
}


/*----------------------------------------------------------------------*
 | calculate velocities                                      dano 12/10 |
 | like interface_velocity(disp) in FSI::DirichletNeumann                |
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> TSI::Algorithm::calc_velocity(
    const Core::LinAlg::Vector<double>& dispnp)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> vel = nullptr;
  // copy D_n onto V_n+1
  vel = std::make_shared<Core::LinAlg::Vector<double>>(*(structure_field()->dispn()));
  // calculate velocity with timestep Dt()
  //  V_n+1^k = (D_n+1^k - D_n) / Dt
  vel->update(1. / dt(), dispnp, -1. / dt());

  return vel;
}  // calc_velocity()


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void TSI::Algorithm::apply_thermo_coupling_state(
    std::shared_ptr<const Core::LinAlg::Vector<double>> temp,
    std::shared_ptr<const Core::LinAlg::Vector<double>> temp_res)
{
  if (matchinggrid_)
  {
    if (temp != nullptr) structure_field()->discretization()->set_state(1, "temperature", *temp);
    if (temp_res != nullptr)
      structure_field()->discretization()->set_state(1, "residual temperature", *temp_res);
  }
  else
  {
    if (temp != nullptr)
      structure_field()->discretization()->set_state(
          1, "temperature", *volcoupl_->apply_vector_mapping12(*temp));
  }

  // set new temperatures to contact
  if (contact_strategy_lagrange_ != nullptr)
  {
    FOUR_C_ASSERT_ALWAYS(structure_thermo_coupling_ != nullptr,
        "Invalid configuration: structure_thermo_coupling_ is required for contact with thermal "
        "coupling, but is only initialized for matching grids. Either disable TSI contact or use "
        "matching grids.");
    contact_strategy_lagrange_->set_state(Mortar::state_temperature,
        *structure_thermo_coupling_->source_to_target(*thermo_field()->tempnp()));
  }
}  // apply_thermo_coupling_state()


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void TSI::Algorithm::apply_struct_coupling_state(
    std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel)
{
  if (matchinggrid_)
  {
    if (disp != nullptr) thermo_field()->discretization()->set_state(1, "displacement", *disp);
    if (vel != nullptr) thermo_field()->discretization()->set_state(1, "velocity", *vel);
  }
  else
  {
    if (disp != nullptr)
      thermo_field()->discretization()->set_state(
          1, "displacement", *volcoupl_->apply_vector_mapping21(*disp));
    if (vel != nullptr)
      thermo_field()->discretization()->set_state(
          1, "velocity", *volcoupl_->apply_vector_mapping21(*vel));
  }
}  // apply_struct_coupling_state()


/*----------------------------------------------------------------------*/
void TSI::Algorithm::prepare_contact_strategy()
{
  auto stype = Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(
      problem_->contact_dynamic_params(), "STRATEGY");

  if (stype == CONTACT::SolvingStrategy::lagmult)
  {
    if (structure_field()->have_model(Solid::model_contact))
    {
      FOUR_C_THROW(
          "structure should not have a Lagrange strategy ... as long as condensed"
          "contact formulations are not moved to the new structural time integration");
    }

    std::vector<const Core::Conditions::Condition*> ccond;
    structure_field()->discretization()->get_condition("Contact", ccond);
    if (ccond.size() == 0) return;

    // ---------------------------------------------------------------------
    // create the contact factory
    // ---------------------------------------------------------------------
    CONTACT::STRATEGY::Factory factory;
    factory.init(structure_field()->discretization());
    factory.setup(problem_->n_dim());

    // check the problem dimension
    factory.check_dimension();

    // create some local variables (later to be stored in strategy)
    std::vector<std::shared_ptr<CONTACT::Interface>> interfaces;
    Teuchos::ParameterList cparams;

    // read and check contact input parameters
    factory.read_and_check_input(cparams);

    // ---------------------------------------------------------------------
    // build the contact interfaces
    // ---------------------------------------------------------------------
    // FixMe Would be great, if we get rid of these poro parameters...
    bool poro_source = false;
    bool poro_target = false;
    factory.build_interfaces(cparams, interfaces, poro_source, poro_target);

    // ---------------------------------------------------------------------
    // build the solver strategy object
    // ---------------------------------------------------------------------
    contact_strategy_lagrange_ = std::dynamic_pointer_cast<CONTACT::LagrangeStrategyTsi>(
        factory.build_strategy(cparams, poro_source, poro_target, 1e8, interfaces));

    // build the search tree
    factory.build_search_tree(interfaces);

    // print final screen output
    factory.print(interfaces, *contact_strategy_lagrange_, cparams);

    // ---------------------------------------------------------------------
    // final touches to the contact strategy
    // ---------------------------------------------------------------------

    contact_strategy_lagrange_->store_dirichlet_status(structure_field()->get_dbc_map_extractor());

    std::shared_ptr<Core::LinAlg::Vector<double>> zero_disp =
        std::make_shared<Core::LinAlg::Vector<double>>(*structure_field()->dof_row_map(), true);
    contact_strategy_lagrange_->set_state(Mortar::state_new_displacement, *zero_disp);
    contact_strategy_lagrange_->save_reference_state(zero_disp);
    contact_strategy_lagrange_->evaluate_reference_state();
    contact_strategy_lagrange_->inttime_init();
    contact_strategy_lagrange_->set_time_integration_info(structure_field()->tim_int_param());
    contact_strategy_lagrange_->redistribute_contact(
        structure_field()->dispn(), structure_field()->veln());

    if (contact_strategy_lagrange_ != nullptr)
    {
      FOUR_C_ASSERT_ALWAYS(structure_thermo_coupling_ != nullptr,
          "Invalid configuration: structure_thermo_coupling_ is required for contact with thermal "
          "coupling, but is only initialized for matching grids. Either disable TSI contact or use "
          "matching grids.");
      contact_strategy_lagrange_->set_alphaf_thermo(problem_->thermal_dynamic_params());
      contact_strategy_lagrange_->set_coupling(structure_thermo_coupling_);
    }
  }
}

void TSI::Algorithm::post_setup()
{
  // call the post_setup routine of the structure field
  structure_->post_setup();
}

FOUR_C_NAMESPACE_CLOSE
