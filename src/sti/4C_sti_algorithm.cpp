// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_sti_algorithm.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
STI::Algorithm::Algorithm(MPI_Comm comm, const Teuchos::ParameterList& stidyn,
    const Teuchos::ParameterList& scatradyn, const Teuchos::ParameterList& solverparams_scatra,
    const Teuchos::ParameterList& solverparams_thermo)
    :  // instantiate base class
      AlgorithmBase(comm, scatradyn),
      scatra_(nullptr),
      thermo_(nullptr),
      strategyscatra_(nullptr),
      strategythermo_(nullptr),
      fieldparameters_(std::make_shared<Teuchos::ParameterList>(scatradyn)),
      iter_(0),
      itermax_(0),
      itertol_(0.),
      stiparameters_(std::make_shared<Teuchos::ParameterList>(stidyn)),
      timer_(std::make_shared<Teuchos::Time>("STI::ALG", true))
{
  // check input parameters for scatra and thermo fields
  if (Teuchos::getIntegralValue<Inpar::ScaTra::VelocityField>(*fieldparameters_, "VELOCITYFIELD") !=
      Inpar::ScaTra::velocity_zero)
    FOUR_C_THROW("Scatra-thermo interaction with convection not yet implemented!");

  // initialize scatra time integrator
  scatra_ = std::make_shared<Adapter::ScaTraBaseAlgorithm>(
      *fieldparameters_, *fieldparameters_, solverparams_scatra);
  scatra_->init();
  scatra_->scatra_field()->set_number_of_dof_set_velocity(1);
  scatra_->scatra_field()->set_number_of_dof_set_thermo(2);
  scatra_->setup();

  // modify field parameters for thermo field
  modify_field_parameters_for_thermo_field();

  // initialize thermo time integrator
  thermo_ = std::make_shared<Adapter::ScaTraBaseAlgorithm>(
      *fieldparameters_, *fieldparameters_, solverparams_thermo, "thermo");
  thermo_->init();
  thermo_->scatra_field()->set_number_of_dof_set_velocity(1);
  thermo_->scatra_field()->set_number_of_dof_set_scatra(2);
  thermo_->setup();

  // check maps from scatra and thermo discretizations
  if (scatra_->scatra_field()->discretization()->dof_row_map()->num_global_elements() == 0)
    FOUR_C_THROW("Scatra discretization does not have any degrees of freedom!");
  if (thermo_->scatra_field()->discretization()->dof_row_map()->num_global_elements() == 0)
    FOUR_C_THROW("Thermo discretization does not have any degrees of freedom!");

  // additional safety check
  if (thermo_->scatra_field()->num_scal() != 1)
    FOUR_C_THROW("Thermo field must involve exactly one transported scalar!");

  // perform initializations associated with scatra-scatra interface mesh tying
  if (scatra_->scatra_field()->s2i_meshtying())
  {
    // safety check
    if (!thermo_->scatra_field()->s2i_meshtying())
    {
      FOUR_C_THROW(
          "Can't evaluate scatra-scatra interface mesh tying in scatra field, but not in thermo "
          "field!");
    }

    // extract meshtying strategies for scatra-scatra interface coupling from scatra and thermo time
    // integrators
    strategyscatra_ = std::dynamic_pointer_cast<ScaTra::MeshtyingStrategyS2I>(
        scatra_->scatra_field()->strategy());
    strategythermo_ = std::dynamic_pointer_cast<ScaTra::MeshtyingStrategyS2I>(
        thermo_->scatra_field()->strategy());

    // perform initializations depending on type of meshtying method
    switch (strategyscatra_->coupling_type())
    {
      case Inpar::S2I::coupling_matching_nodes:
      {
        // safety check
        if (strategythermo_->coupling_type() != Inpar::S2I::coupling_matching_nodes)
        {
          FOUR_C_THROW(
              "Must have matching nodes at scatra-scatra coupling interfaces in both the scatra "
              "and the thermo fields!");
        }

        break;
      }

      case Inpar::S2I::coupling_mortar_standard:
      {
        // safety check
        if (strategythermo_->coupling_type() != Inpar::S2I::coupling_mortar_condensed_bubnov)
          FOUR_C_THROW("Invalid type of scatra-scatra interface coupling for thermo field!");

        // extract scatra-scatra interface mesh tying conditions
        std::vector<const Core::Conditions::Condition*> conditions;
        scatra_->scatra_field()->discretization()->get_condition("S2IMeshtying", conditions);

        // loop over all conditions
        for (auto& condition : conditions)
        {
          // consider conditions for slave side only
          if (condition->parameters().get<Inpar::S2I::InterfaceSides>("INTERFACE_SIDE") ==
              Inpar::S2I::side_slave)
          {
            // extract ID of current condition
            const int condid = condition->parameters().get<int>("ConditionID");
            if (condid < 0) FOUR_C_THROW("Invalid condition ID!");

            // extract mortar discretizations associated with current condition
            Core::FE::Discretization& scatradis = strategyscatra_->mortar_discretization(condid);
            Core::FE::Discretization& thermodis = strategythermo_->mortar_discretization(condid);

            // exchange dofsets between discretizations
            scatradis.add_dof_set(thermodis.get_dof_set_proxy());
            thermodis.add_dof_set(scatradis.get_dof_set_proxy());
          }
        }

        break;
      }

      default:
      {
        FOUR_C_THROW("Invalid type of scatra-scatra interface coupling!");
      }
    }
  }
}  // STI::Algorithm::Algorithm

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::Algorithm::modify_field_parameters_for_thermo_field()
{
  // extract parameters for initial temperature field from parameter list for scatra-thermo
  // interaction and overwrite corresponding parameters in parameter list for thermo field
  if (!fieldparameters_->isParameter("INITIALFIELD") or
      !fieldparameters_->isParameter("INITFUNCNO"))
  {
    FOUR_C_THROW(
        "Initial field parameters not properly set in input file section SCALAR TRANSPORT "
        "DYNAMIC!");
  }
  if (!stiparameters_->isParameter("THERMO_INITIALFIELD") or
      !stiparameters_->isParameter("THERMO_INITFUNCNO"))
  {
    FOUR_C_THROW(
        "Initial field parameters not properly set in input file section SCALAR TRANSPORT "
        "DYNAMIC!");
  }
  fieldparameters_->set<Inpar::ScaTra::InitialField>(
      "INITIALFIELD", stiparameters_->get<Inpar::ScaTra::InitialField>("THERMO_INITIALFIELD"));
  fieldparameters_->set<int>("INITFUNCNO", stiparameters_->get<int>("THERMO_INITFUNCNO"));

  // perform additional manipulations associated with scatra-scatra interface mesh tying
  if (scatra_->scatra_field()->s2i_meshtying())
  {
    // set flag for matrix type associated with thermo field
    fieldparameters_->set<Core::LinAlg::MatrixType>("MATRIXTYPE", Core::LinAlg::MatrixType::sparse);

    // set flag in thermo meshtying strategy for evaluation of interface linearizations and
    // residuals on slave side only
    fieldparameters_->sublist("S2I COUPLING").set<bool>("SLAVEONLY", true);

    // adapt type of meshtying method for thermo field
    if (fieldparameters_->sublist("S2I COUPLING").get<Inpar::S2I::CouplingType>("COUPLINGTYPE") ==
        Inpar::S2I::CouplingType::coupling_mortar_standard)
    {
      fieldparameters_->sublist("S2I COUPLING")
          .set<Inpar::S2I::CouplingType>(
              "COUPLINGTYPE", Inpar::S2I::CouplingType::coupling_mortar_condensed_bubnov);
    }
    else if (fieldparameters_->sublist("S2I COUPLING")
                 .get<Inpar::S2I::CouplingType>("COUPLINGTYPE") !=
             Inpar::S2I::CouplingType::coupling_matching_nodes)
      FOUR_C_THROW("Invalid type of scatra-scatra interface coupling!");

    // make sure that interface side underlying Lagrange multiplier definition is slave side
    fieldparameters_->sublist("S2I COUPLING")
        .set<Inpar::S2I::InterfaceSides>("LMSIDE", Inpar::S2I::InterfaceSides::side_slave);
  }
}  // STI::Algorithm::modify_field_parameters_for_thermo_field()

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::Algorithm::output()
{
  // output scatra field
  scatra_->scatra_field()->check_and_write_output_and_restart();

  // output thermo field
  thermo_->scatra_field()->check_and_write_output_and_restart();
}

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::Algorithm::prepare_time_step()
{
  // update time and time step
  increment_time_and_step();

  // provide scatra and thermo fields with velocities
  scatra_->scatra_field()->set_velocity_field_from_function();
  thermo_->scatra_field()->set_velocity_field_from_function();

  // pass thermo degrees of freedom to scatra discretization for preparation of first time step
  // (calculation of initial time derivatives etc.)
  if (step() == 1) transfer_thermo_to_scatra(thermo_->scatra_field()->phiafnp());

  // prepare time step for scatra field
  scatra_->scatra_field()->prepare_time_step();

  // pass scatra degrees of freedom to thermo discretization for preparation of first time step
  // (calculation of initial time derivatives etc.)
  if (step() == 1) transfer_scatra_to_thermo(scatra_->scatra_field()->phiafnp());

  // prepare time step for thermo field
  thermo_->scatra_field()->prepare_time_step();
}  // STI::Algorithm::prepare_time_step()

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::Algorithm::read_restart(int step  //! time step for restart
)
{
  // read scatra and thermo restart variables
  scatra_->scatra_field()->read_restart(step);
  thermo_->scatra_field()->read_restart(step);

  // set time and time step
  set_time_step(scatra_->scatra_field()->time(), step);
}  // STI::Algorithm::read_restart

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::Algorithm::time_loop()
{
  // output initial solution to screen and files
  if (step() == 0)
  {
    transfer_thermo_to_scatra(thermo_->scatra_field()->phiafnp());
    transfer_scatra_to_thermo(scatra_->scatra_field()->phiafnp());
    scatra_field()->prepare_time_loop();
    thermo_field()->prepare_time_loop();
  }

  // time loop
  while (not_finished())
  {
    // prepare time step
    prepare_time_step();

    // store time before calling nonlinear solver
    double time = timer_->wallTime();

    // evaluate time step
    solve();

    // determine time spent by nonlinear solver and take maximum over all processors via
    // communication
    double mydtnonlinsolve(timer_->wallTime() - time), dtnonlinsolve(0.);
    Core::Communication::max_all(&mydtnonlinsolve, &dtnonlinsolve, 1, get_comm());

    // output performance statistics associated with nonlinear solver into *.csv file if applicable
    if (fieldparameters_->get<bool>("OUTPUTNONLINSOLVERSTATS"))
      scatra_->scatra_field()->output_nonlin_solver_stats(
          static_cast<int>(iter_), dtnonlinsolve, step(), get_comm());

    // update scatra and thermo fields
    update();

    // output solution to screen and files
    output();
  }  // while(not_finished())
}  // STI::Algorithm::TimeLoop()

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::Algorithm::transfer_scatra_to_thermo(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> scatra) const
{
  // pass scatra degrees of freedom to thermo discretization
  thermo_->scatra_field()->discretization()->set_state(2, "scatra", *scatra);

  // transfer state vector for evaluation of scatra-scatra interface mesh tying
  if (thermo_->scatra_field()->s2i_meshtying())
  {
    switch (strategythermo_->coupling_type())
    {
      case Inpar::S2I::coupling_matching_nodes:
      {
        // pass master-side scatra degrees of freedom to thermo discretization
        const std::shared_ptr<Core::LinAlg::Vector<double>> imasterphinp =
            Core::LinAlg::create_vector(
                *scatra_->scatra_field()->discretization()->dof_row_map(), true);
        strategyscatra_->interface_maps()->insert_vector(
            *strategyscatra_->coupling_adapter()->master_to_slave(
                *strategyscatra_->interface_maps()->extract_vector(*scatra, 2)),
            1, *imasterphinp);
        thermo_->scatra_field()->discretization()->set_state(2, "imasterscatra", *imasterphinp);

        break;
      }

      case Inpar::S2I::coupling_mortar_condensed_bubnov:
      {
        // extract scatra-scatra interface mesh tying conditions
        std::vector<const Core::Conditions::Condition*> conditions;
        thermo_->scatra_field()->discretization()->get_condition("S2IMeshtying", conditions);

        // loop over all conditions
        for (auto& condition : conditions)
        {
          // consider conditions for slave side only
          if (condition->parameters().get<Inpar::S2I::InterfaceSides>("INTERFACE_SIDE") ==
              Inpar::S2I::side_slave)
          {
            // extract ID of current condition
            const int condid = condition->parameters().get<int>("ConditionID");
            if (condid < 0) FOUR_C_THROW("Invalid condition ID!");

            // extract mortar discretization associated with current condition
            Core::FE::Discretization& thermodis = strategythermo_->mortar_discretization(condid);

            // pass interfacial scatra degrees of freedom to thermo discretization
            Core::LinAlg::Vector<double> iscatra(*thermodis.dof_row_map(1));
            Core::LinAlg::export_to(*scatra, iscatra);
            thermodis.set_state(1, "scatra", iscatra);
          }
        }

        break;
      }

      default:
      {
        FOUR_C_THROW("You must be kidding me...");
      }
    }
  }
}  // STI::Algorithm::transfer_scatra_to_thermo()

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::Algorithm::transfer_thermo_to_scatra(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> thermo) const
{
  // pass thermo degrees of freedom to scatra discretization
  scatra_->scatra_field()->discretization()->set_state(2, "thermo", *thermo);

  // transfer state vector for evaluation of scatra-scatra interface mesh tying
  if (scatra_->scatra_field()->s2i_meshtying() and
      strategyscatra_->coupling_type() == Inpar::S2I::coupling_mortar_standard)
  {
    // extract scatra-scatra interface mesh tying conditions
    std::vector<const Core::Conditions::Condition*> conditions;
    scatra_->scatra_field()->discretization()->get_condition("S2IMeshtying", conditions);

    // loop over all conditions
    for (auto& condition : conditions)
    {
      // consider conditions for slave side only
      if (condition->parameters().get<Inpar::S2I::InterfaceSides>("INTERFACE_SIDE") ==
          Inpar::S2I::side_slave)
      {
        // extract ID of current condition
        const int condid = condition->parameters().get<int>("ConditionID");
        if (condid < 0) FOUR_C_THROW("Invalid condition ID!");

        // extract mortar discretization associated with current condition
        Core::FE::Discretization& scatradis = strategyscatra_->mortar_discretization(condid);

        // pass interfacial thermo degrees of freedom to scatra discretization
        Core::LinAlg::Vector<double> ithermo(*scatradis.dof_row_map(1));
        Core::LinAlg::export_to(*thermo, ithermo);
        scatradis.set_state(1, "thermo", ithermo);
      }
    }
  }
}  // STI::Algorithm::transfer_thermo_to_scatra()

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::Algorithm::update()
{
  // update scatra field
  scatra_->scatra_field()->update();

  // compare scatra field to analytical solution if applicable
  scatra_->scatra_field()->evaluate_error_compared_to_analytical_sol();

  // update thermo field
  thermo_->scatra_field()->update();

  // compare thermo field to analytical solution if applicable
  thermo_->scatra_field()->evaluate_error_compared_to_analytical_sol();
}  // STI::Algorithm::update()

FOUR_C_NAMESPACE_CLOSE
