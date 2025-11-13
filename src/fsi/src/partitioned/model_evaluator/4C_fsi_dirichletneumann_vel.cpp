// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_dirichletneumann_vel.hpp"

#include "4C_adapter_fld_fbi_movingboundary.hpp"
#include "4C_adapter_str_fbiwrapper.hpp"
#include "4C_binstrategy.hpp"
#include "4C_fbi_adapter_constraintbridge.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_output_writer.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_params.hpp"
#include "4C_fbi_constraintenforcer.hpp"
#include "4C_fbi_constraintenforcer_factory.hpp"
#include "4C_fbi_input.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_io_control.hpp"
#include "4C_io_visualization_parameters.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <iostream>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::DirichletNeumannVel::DirichletNeumannVel(MPI_Comm comm)
    : DirichletNeumann(comm),
      constraint_manager_(Adapter::ConstraintEnforcerFactory::create_enforcer(
          Global::Problem::instance()->fsi_dynamic_params(),
          Global::Problem::instance()->fbi_params()))
{
  // empty constructor
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::DirichletNeumannVel::setup()
{
  // call setup of base class
  FSI::DirichletNeumann::setup();
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  const Teuchos::ParameterList& fsipart = fsidyn.sublist("PARTITIONED SOLVER");
  if (Teuchos::getIntegralValue<Inpar::FSI::CoupVarPart>(fsipart, "COUPVARIABLE") ==
      Inpar::FSI::CoupVarPart::disp)
    FOUR_C_THROW("Please set the fsi coupling variable to Velocity or Force!\n");
  set_kinematic_coupling(Teuchos::getIntegralValue<Inpar::FSI::CoupVarPart>(
                             fsipart, "COUPVARIABLE") == Inpar::FSI::CoupVarPart::vel);
  if (std::dynamic_pointer_cast<Adapter::FBIStructureWrapper>(structure_field()) == nullptr)
  {
    FOUR_C_THROW("Something went very wrong here! You should have a FBIStructureWrapper!\n");
  }
  if (std::dynamic_pointer_cast<Adapter::FBIFluidMB>(mb_fluid_field()) == nullptr)
  {
    FOUR_C_THROW("Something went very wrong here! You should have a FBIFluidMB adapter!\n");
  }
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::DirichletNeumannVel::fluid_op(
    std::shared_ptr<Core::LinAlg::Vector<double>> ivel, const FillType fillFlag)
{
  const Teuchos::ParameterList& fbi = Global::Problem::instance()->fbi_params();

  FSI::Partitioned::fluid_op(ivel, fillFlag);

  if (fillFlag == User)
  {
    FOUR_C_THROW("Not implemented!\n");
    return fluid_to_struct(mb_fluid_field()->relaxation_solve(nullptr, dt()));
  }
  else
  {
    // A rather simple hack. We need something better!
    const int itemax = mb_fluid_field()->itemax();

    if (fillFlag == MF_Res and mfresitemax_ > 0) mb_fluid_field()->set_itemax(mfresitemax_ + 1);

    mb_fluid_field()->nonlinear_solve(nullptr, nullptr);

    mb_fluid_field()->set_itemax(itemax);

    if (Teuchos::getIntegralValue<FBI::BeamToFluidCoupling>(fbi, "COUPLING") !=
            FBI::BeamToFluidCoupling::fluid &&
        fbi.get<int>("STARTSTEP") < step())
    {
      constraint_manager_->recompute_coupling_without_pair_creation();
    }

    return fluid_to_struct(ivel);
  }
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::DirichletNeumannVel::struct_op(
    std::shared_ptr<Core::LinAlg::Vector<double>> iforce, const FillType fillFlag)
{
  FSI::Partitioned::struct_op(iforce, fillFlag);

  const Teuchos::ParameterList& fbi = Global::Problem::instance()->fbi_params();
  if (!(Teuchos::getIntegralValue<FBI::BeamToFluidCoupling>(fbi, "COUPLING") ==
          FBI::BeamToFluidCoupling::fluid) &&
      fbi.get<int>("STARTSTEP") < step())
  {
    if (not use_old_structure_)
      structure_field()->apply_interface_forces(iforce);
    else
      FOUR_C_THROW(
          "Fluid beam interaction is not tested with the old structure time. You should not be "
          "here! Change the IntStrategy in your Input file to Standard.\n");
  }

  structure_field()->solve();
  structure_field()->write_gmsh_struct_output_step();

  if (fbi.get<int>("STARTSTEP") < step())
  {
    constraint_manager_->prepare_fluid_solve();
    constraint_manager_->evaluate();
    if (!(Teuchos::getIntegralValue<FBI::BeamToFluidCoupling>(fbi, "COUPLING") ==
            FBI::BeamToFluidCoupling::solid))
      std::dynamic_pointer_cast<Adapter::FBIFluidMB>(mb_fluid_field())->reset_external_forces();
  }

  return struct_to_fluid(iforce);
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::DirichletNeumannVel::initial_guess()
{
  if (get_kinematic_coupling())
  {
    // predict velocity
    // For now, we want to use no predictor. This function returns the current interface velocity.
    if (std::dynamic_pointer_cast<Adapter::FBIStructureWrapper>(structure_field()) != nullptr)
      return std::dynamic_pointer_cast<Adapter::FBIStructureWrapper>(structure_field())
          ->predict_interface_velnp();
  }
  else
  {
    const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
    const Teuchos::ParameterList& fsipart = fsidyn.sublist("PARTITIONED SOLVER");
    if (fsipart.get<std::string>("PREDICTOR") != "d(n)")
    {
      FOUR_C_THROW("unknown interface force predictor '{}'", fsipart.get<std::string>("PREDICTOR"));
    }
    return interface_force();
  }
  FOUR_C_THROW("Something went very wrong here!\n");
  return nullptr;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

std::shared_ptr<Core::LinAlg::Vector<double>> FSI::DirichletNeumannVel::fluid_to_struct(
    std::shared_ptr<Core::LinAlg::Vector<double>> iv)
{
  return constraint_manager_->fluid_to_structure();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

std::shared_ptr<Core::LinAlg::Vector<double>> FSI::DirichletNeumannVel::struct_to_fluid(
    std::shared_ptr<Core::LinAlg::Vector<double>> iv)
{
  return constraint_manager_->structure_to_fluid(step());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

void FSI::DirichletNeumannVel::output()
{
  FSI::DirichletNeumann::output();
  constraint_manager_->output(time(), step());
  visualization_output_writer_->write_output_runtime(*constraint_manager_, step(), time());
  structure_field()->discretization()->writer()->clear_map_cache();
  mb_fluid_field()->discretization()->writer()->clear_map_cache();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

void FSI::DirichletNeumannVel::timeloop(
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> interface)
{
  constraint_manager_->setup(structure_field(), mb_fluid_field());
  if (get_kinematic_coupling()) constraint_manager_->prepare_fluid_solve();
  visualization_output_writer_ =
      std::make_shared<BeamInteraction::BeamToFluidMeshtyingVtkOutputWriter>(
          Core::IO::visualization_parameters_factory(
              Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
              *Global::Problem::instance()->output_control_file(), time()),
          constraint_manager_->get_bridge()->get_params()->get_visualization_output_params_ptr());
  constraint_manager_->evaluate();
  if (get_kinematic_coupling()) struct_to_fluid(nullptr);

  FSI::Partitioned::timeloop(interface);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

void FSI::DirichletNeumannVel::set_binning(
    std::shared_ptr<Core::Binstrategy::BinningStrategy> binning)
{
  constraint_manager_->set_binning(binning);
};

FOUR_C_NAMESPACE_CLOSE
