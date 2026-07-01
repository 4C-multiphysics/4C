// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_str_timeada_joint.hpp"

#include "4C_adapter_str_timeloop.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_structure_new_input.hpp"
#include "4C_structure_new_solver_factory.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"
#include "4C_structure_new_timint_basedataio.hpp"
#include "4C_structure_new_timint_basedatasdyn.hpp"
#include "4C_structure_new_timint_factory.hpp"

#include <Teuchos_ParameterList.hpp>

#include <tuple>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Adapter::StructureTimeAdaJoint::StructureTimeAdaJoint(
    Global::Problem& problem, std::shared_ptr<Structure> structure)
    : StructureTimeAda(problem, structure), sta_(nullptr)
{
  if (stm_->is_setup()) setup_auxiliary();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::StructureTimeAdaJoint::setup_auxiliary()
{
  auto& problem = StructureTimeAda::problem();
  const Teuchos::ParameterList& sdyn = problem.structural_dynamic_params();
  const Teuchos::ParameterList& jep = sdyn.sublist("TIMEADAPTIVITY").sublist("JOINT EXPLICIT");

  // get the parameters of the auxiliary integrator
  Teuchos::ParameterList adyn(sdyn);
  adyn.remove("TIMEADAPTIVITY");
  for (auto i = jep.begin(); i != jep.end(); ++i) adyn.setEntry(jep.name(i), jep.entry(i));

  // construct the auxiliary time integrator
  sta_ = Solid::TimeInt::build_strategy(adyn);

  ///// setup dataio
  std::shared_ptr<Teuchos::ParameterList> ioflags =
      std::make_shared<Teuchos::ParameterList>(problem.io_params());
  ioflags->set("STDOUTEVERY", 0);
  //
  std::shared_ptr<Teuchos::ParameterList> xparams = std::make_shared<Teuchos::ParameterList>();
  Teuchos::ParameterList& nox = xparams->sublist("NOX");
  nox = problem.structural_nox_params();
  //
  std::shared_ptr<Core::IO::DiscretizationWriter> output = stm_->discretization()->writer();
  //
  std::shared_ptr<Solid::TimeInt::BaseDataIO> dataio =
      std::make_shared<Solid::TimeInt::BaseDataIO>();
  dataio->init(*ioflags, adyn, *xparams, output);
  dataio->setup();

  ///// setup datasdyn
  std::shared_ptr<std::set<Solid::ModelType>> modeltypes =
      std::make_shared<std::set<enum Solid::ModelType>>();
  modeltypes->insert(Solid::model_structure);
  //
  std::shared_ptr<std::set<Solid::EleTech>> eletechs =
      std::make_shared<std::set<enum Solid::EleTech>>();
  //
  std::shared_ptr<std::map<Solid::ModelType, std::shared_ptr<Core::LinAlg::Solver>>> linsolvers =
      Solid::SOLVER::build_lin_solvers(*modeltypes, adyn, *stm_->discretization());
  //
  std::shared_ptr<Solid::TimeInt::BaseDataSDyn> datasdyn = Solid::TimeInt::build_data_sdyn(adyn);
  datasdyn->init(stm_->discretization(), adyn, *xparams, modeltypes, eletechs, linsolvers);
  datasdyn->setup();

  // setup global state
  std::shared_ptr<Solid::TimeInt::BaseDataGlobalState> dataglobalstate =
      Solid::TimeInt::build_data_global_state();
  dataglobalstate->init(stm_->discretization(), adyn, datasdyn);
  dataglobalstate->setup();

  // setup auxiliary integrator
  sta_->init(problem, dataio, datasdyn, dataglobalstate);
  sta_->setup();

  const int restart = problem.restart();
  if (restart)
  {
    sta_->post_setup();

    const Solid::TimeInt::Base& sti = *stm_;
    const auto& gstate = sti.data_global_state();
    dataglobalstate->get_dis_n()->update(1.0, *(gstate.get_dis_n()), 0.0);
    dataglobalstate->get_vel_n()->update(1.0, *(gstate.get_vel_n()), 0.0);
    dataglobalstate->get_acc_n()->update(1.0, *(gstate.get_acc_n()), 0.0);
  }

  // check explicitness
  if (sta_->is_implicit())
  {
    FOUR_C_THROW("Implicit might work, but please check carefully");
  }

  // check order
  if (sta_->method_order_of_accuracy_dis() > stm_->method_order_of_accuracy_dis())
  {
    ada_ = ada_upward;
  }
  else if (sta_->method_order_of_accuracy_dis() < stm_->method_order_of_accuracy_dis())
  {
    ada_ = ada_downward;
  }
  else if (sta_->method_name() == stm_->method_name())
  {
    ada_ = ada_ident;
  }
  else
  {
    ada_ = ada_orderequal;
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::string Adapter::StructureTimeAdaJoint::method_title() const
{
  return "JointExplicit_" + sta_->method_title();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Adapter::StructureTimeAdaJoint::method_order_of_accuracy_dis() const
{
  return sta_->method_order_of_accuracy_dis();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Adapter::StructureTimeAdaJoint::method_order_of_accuracy_vel() const
{
  return sta_->method_order_of_accuracy_vel();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Adapter::StructureTimeAdaJoint::method_lin_err_coeff_dis() const
{
  return sta_->method_lin_err_coeff_dis();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Adapter::StructureTimeAdaJoint::method_lin_err_coeff_vel() const
{
  return sta_->method_lin_err_coeff_vel();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Adapter::StructureTimeAda::AdaEnum Adapter::StructureTimeAdaJoint::method_adapt_dis() const
{
  return ada_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::StructureTimeAdaJoint::integrate_step_auxiliary()
{
  // set current step size
  sta_->set_delta_time(stepsize_);
  sta_->set_time_np(time_ + stepsize_);

  // integrate the auxiliary time integrator one step in time
  // buih: another solution is to use the wrapper, but it will do more than necessary
  const Solid::TimeInt::Base& sta = *sta_;
  const auto& gstate = sta.data_global_state();

  sta_->integrate_step();

  // copy onto target
  locerrdisn_->update(1.0, *(gstate.get_dis_np()), 0.0);

  // reset
  sta_->reset_step();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::StructureTimeAdaJoint::update_auxiliary()
{
  // copy the data from main integrator to the auxiliary one
  // for reference: the vector map of the global state vectors may need to be checked to ensure they
  // are the same
  const Solid::TimeInt::Base& stm = *stm_;
  const Solid::TimeInt::BaseDataGlobalState& gstate_i = stm.data_global_state();

  const Solid::TimeInt::Base& sta = *sta_;
  const Solid::TimeInt::BaseDataGlobalState& gstate_a_const = sta.data_global_state();
  Solid::TimeInt::BaseDataGlobalState& gstate_a =
      const_cast<Solid::TimeInt::BaseDataGlobalState&>(gstate_a_const);

  gstate_a.get_dis_np()->update(1.0, (*gstate_i.get_dis_n()), 0.0);
  gstate_a.get_vel_np()->update(1.0, (*gstate_i.get_vel_n()), 0.0);
  gstate_a.get_acc_np()->update(1.0, (*gstate_i.get_acc_n()), 0.0);
  gstate_a.get_multi_dis().update_steps((*gstate_i.get_dis_n()));
  gstate_a.get_multi_vel().update_steps((*gstate_i.get_vel_n()));
  gstate_a.get_multi_acc().update_steps((*gstate_i.get_acc_n()));

  gstate_a.get_time_np() = gstate_i.get_time_np();
  gstate_a.get_delta_time().update_steps((gstate_i.get_delta_time())[0]);
  gstate_a.get_fvisco_np()->update(1.0, (*gstate_i.get_fvisco_n()), 0.0);
  gstate_a.get_fvisco_n()->update(1.0, (*gstate_i.get_fvisco_n()), 0.0);
  gstate_a.get_finertial_np()->update(1.0, (*gstate_i.get_finertial_n()), 0.0);
  gstate_a.get_finertial_n()->update(1.0, (*gstate_i.get_finertial_n()), 0.0);
  gstate_a.get_fint_np()->update(1.0, (*gstate_i.get_fint_n()), 0.0);
  gstate_a.get_fint_n()->update(1.0, (*gstate_i.get_fint_n()), 0.0);
  gstate_a.get_fext_np()->update(1.0, (*gstate_i.get_fext_n()), 0.0);
  gstate_a.get_fext_n()->update(1.0, (*gstate_i.get_fext_n()), 0.0);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::StructureTimeAdaJoint::reset_step()
{
  // base reset
  Adapter::StructureTimeAda::reset_step();
  // set current step size
  sta_->set_delta_time(stepsize_);
  sta_->set_time_np(time_ + stepsize_);
  // reset the integrator
  sta_->reset_step();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::StructureTimeAdaJoint::post_setup()
{
  // base post setup
  Adapter::StructureTimeAda::post_setup();

  // post setup the auxiliary time integrator
  sta_->post_setup();
}

FOUR_C_NAMESPACE_CLOSE
