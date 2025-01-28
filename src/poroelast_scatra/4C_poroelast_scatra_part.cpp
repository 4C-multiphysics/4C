// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_poroelast_scatra_part.hpp"

#include "4C_adapter_fld_poro.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_contact_nitsche_strategy_ssi.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_structure_new_model_evaluator_contact.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |                                                         vuong 08/13  |
 *----------------------------------------------------------------------*/
PoroElastScaTra::PoroScatraPart::PoroScatraPart(
    MPI_Comm comm, const Teuchos::ParameterList& timeparams)
    : PoroScatraBase(comm, timeparams)
{
  poro_field()->setup_solver();
}

void PoroElastScaTra::PoroScatraPart::setup_system()
{
  PoroScatraBase::setup_system();
  // if (ssi_interface_contact() && !PoroScatraBase::IsRestart())
  if (ssi_interface_contact())
  {
    setup_contact_strategy();
    scatra_->scatra_field()->set_nitsche_contact_strategy(contact_strategy_nitsche_);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraPart::setup_contact_strategy()
{
  // get the contact solution strategy
  auto contact_solution_type =
      Global::Problem::instance()->contact_dynamic_params().get<Inpar::CONTACT::SolvingStrategy>(
          "STRATEGY");

  if (contact_solution_type == Inpar::CONTACT::solution_nitsche)
  {
    if (Global::Problem::instance()
            ->structural_dynamic_params()
            .get<Inpar::Solid::IntegrationStrategy>("INT_STRATEGY") != Inpar::Solid::int_standard)
      FOUR_C_THROW("ssi contact only with new structural time integration");

    // get the contact model evaluator and store a pointer to the strategy
    auto& model_evaluator_contact = dynamic_cast<Solid::ModelEvaluator::Contact&>(
        structure_field()->model_evaluator(Inpar::Solid::model_contact));

    contact_strategy_nitsche_ = std::dynamic_pointer_cast<CONTACT::NitscheStrategySsi>(
        model_evaluator_contact.strategy_ptr());
  }
  else
    FOUR_C_THROW("Only Nitsche contact implemented for SSI problems at the moment!");
}

void PoroElastScaTra::PoroScatraPart::set_poro_solution()
{
  PoroScatraBase::set_poro_solution();

  // set poro solution for contact
  if (ssi_interface_contact())
  {
    // poro pressure and velocity
    std::shared_ptr<const Core::LinAlg::Vector<double>> velnp = nullptr;

    if (matchinggrid_)
    {
      velnp = poro_->fluid_field()->velnp();
    }
    else
    {
      velnp = volcoupl_fluidscatra_->apply_vector_mapping21(*poro_->fluid_field()->velnp());
    }

    contact_strategy_nitsche_->set_state(Mortar::state_fvelocity, *velnp);

    // structure velocity
    std::shared_ptr<const Core::LinAlg::Vector<double>> svel = poro_->structure_field()->velnp();
    contact_strategy_nitsche_->set_state(Mortar::state_svelocity, *svel);
  }
}

void PoroElastScaTra::PoroScatraPart::set_scatra_solution()
{
  PoroScatraBase::set_scatra_solution();

  std::shared_ptr<const Core::LinAlg::Vector<double>> phinp_s = nullptr;

  if (matchinggrid_)
  {
    phinp_s = scatra_->scatra_field()->phinp();
  }
  else
  {
    phinp_s = volcoupl_structurescatra_->apply_vector_mapping12(*scatra_->scatra_field()->phinp());
  }

  scatra_->scatra_field()->discretization()->set_state("phinp", phinp_s);

  // set scatra solution for contact
  if (ssi_interface_contact())
  {
    contact_strategy_nitsche_->set_state(Mortar::state_scalar, *phinp_s);
  }
}

FOUR_C_NAMESPACE_CLOSE
