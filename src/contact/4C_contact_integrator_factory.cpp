// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_integrator_factory.hpp"

#include "4C_contact_ehl_integrator.hpp"
#include "4C_contact_nitsche_integrator.hpp"
#include "4C_contact_nitsche_integrator_fpi.hpp"
#include "4C_contact_nitsche_integrator_fsi.hpp"
#include "4C_contact_nitsche_integrator_poro.hpp"
#include "4C_contact_nitsche_integrator_ssi.hpp"
#include "4C_contact_nitsche_integrator_ssi_elch.hpp"
#include "4C_contact_nitsche_integrator_tsi.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<CONTACT::Integrator> CONTACT::INTEGRATOR::Factory::build_integrator(
    const Inpar::CONTACT::SolvingStrategy& sol_type, Teuchos::ParameterList& mortar_params,
    const Core::FE::CellType& slave_type, const Epetra_Comm& comm) const
{
  Teuchos::RCP<CONTACT::Integrator> integrator = Teuchos::null;
  switch (sol_type)
  {
    case Inpar::CONTACT::solution_nitsche:
    {
      if (mortar_params.get<int>("PROBTYPE") == Inpar::CONTACT::tsi)
      {
        integrator =
            Teuchos::make_rcp<CONTACT::IntegratorNitscheTsi>(mortar_params, slave_type, comm);
      }
      else if (mortar_params.get<int>("PROBTYPE") == Inpar::CONTACT::ssi)
      {
        integrator =
            Teuchos::make_rcp<CONTACT::IntegratorNitscheSsi>(mortar_params, slave_type, comm);
      }
      else if (mortar_params.get<int>("PROBTYPE") == Inpar::CONTACT::ssi_elch)
      {
        integrator =
            Teuchos::make_rcp<CONTACT::IntegratorNitscheSsiElch>(mortar_params, slave_type, comm);
      }
      else if (mortar_params.get<int>("PROBTYPE") == Inpar::CONTACT::poroelast ||
               mortar_params.get<int>("PROBTYPE") == Inpar::CONTACT::poroscatra)
      {
        integrator =
            Teuchos::make_rcp<CONTACT::IntegratorNitschePoro>(mortar_params, slave_type, comm);
      }
      else if (mortar_params.get<int>("PROBTYPE") == Inpar::CONTACT::fsi)
      {
        integrator =
            Teuchos::make_rcp<CONTACT::IntegratorNitscheFsi>(mortar_params, slave_type, comm);
      }
      else if (mortar_params.get<int>("PROBTYPE") == Inpar::CONTACT::fpi)
      {
        integrator =
            Teuchos::make_rcp<CONTACT::IntegratorNitscheFpi>(mortar_params, slave_type, comm);
      }
      else
      {
        integrator = Teuchos::make_rcp<CONTACT::IntegratorNitsche>(mortar_params, slave_type, comm);
      }
      break;
    }
    case Inpar::CONTACT::solution_penalty:
    case Inpar::CONTACT::solution_multiscale:
    {
      if (Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(mortar_params, "ALGORITHM") ==
          Inpar::Mortar::algorithm_gpts)
        integrator = Teuchos::make_rcp<CONTACT::IntegratorNitsche>(mortar_params, slave_type, comm);
      else
        integrator = Teuchos::make_rcp<CONTACT::Integrator>(mortar_params, slave_type, comm);
      break;
    }
    case Inpar::CONTACT::solution_lagmult:
    case Inpar::CONTACT::solution_uzawa:
    {
      integrator = Teuchos::make_rcp<CONTACT::Integrator>(mortar_params, slave_type, comm);
      break;
    }
    case Inpar::CONTACT::solution_ehl:
    {
      integrator = Teuchos::make_rcp<CONTACT::IntegratorEhl>(mortar_params, slave_type, comm);

      break;
    }
    default:
    {
      FOUR_C_THROW("Unsupported solving strategy! (stype = %s | %d)",
          Inpar::CONTACT::solving_strategy_to_string(sol_type).c_str(), sol_type);
      exit(EXIT_FAILURE);
    }
  }  // end switch

  return integrator;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<CONTACT::Integrator> CONTACT::INTEGRATOR::build_integrator(
    const Inpar::CONTACT::SolvingStrategy& sol_type, Teuchos::ParameterList& mortar_params,
    const Core::FE::CellType& slave_type, const Epetra_Comm& comm)
{
  Factory factory;
  return factory.build_integrator(sol_type, mortar_params, slave_type, comm);
}

FOUR_C_NAMESPACE_CLOSE
