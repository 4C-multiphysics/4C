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
#include "4C_contact_nitsche_integrator_poro_scatra.hpp"
#include "4C_contact_nitsche_integrator_ssi.hpp"
#include "4C_contact_nitsche_integrator_ssi_elch.hpp"
#include "4C_global_data.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<CONTACT::Integrator> CONTACT::INTEGRATOR::Factory::build_integrator(
    const Inpar::CONTACT::SolvingStrategy& sol_type, Teuchos::ParameterList& mortar_params,
    const Core::FE::CellType& slave_type, MPI_Comm comm) const
{
  std::shared_ptr<CONTACT::Integrator> integrator = nullptr;
  switch (sol_type)
  {
    case Inpar::CONTACT::solution_nitsche:
    {
      if (mortar_params.get<int>("PROBTYPE") == Inpar::CONTACT::ssi)
      {
        integrator =
            std::make_shared<CONTACT::IntegratorNitscheSsi>(mortar_params, slave_type, comm);
      }
      else if (mortar_params.get<int>("PROBTYPE") == Inpar::CONTACT::ssi_elch)
      {
        integrator =
            std::make_shared<CONTACT::IntegratorNitscheSsiElch>(mortar_params, slave_type, comm);
      }
      else if (mortar_params.get<int>("PROBTYPE") == Inpar::CONTACT::poroelast)
      {
        integrator =
            std::make_shared<CONTACT::IntegratorNitschePoro>(mortar_params, slave_type, comm);
      }
      else if (mortar_params.get<int>("PROBTYPE") == Inpar::CONTACT::poroscatra)
      {
        const Teuchos::ParameterList& stru =
            Global::Problem::instance()->structural_dynamic_params();
        const Teuchos::ParameterList& porodyn =
            Global::Problem::instance()->poroelast_dynamic_params();
        // porotimefac = 1/(theta*dt) --- required for derivation of structural displacements!
        double porotimefac =
            1 / (stru.sublist("ONESTEPTHETA").get<double>("THETA") * stru.get<double>("TIMESTEP"));
        mortar_params.set<double>("porotimefac", porotimefac);
        mortar_params.set<bool>(
            "CONTACTNOPEN", porodyn.get<bool>("CONTACTNOPEN"));  // used in the integrator
        integrator =
            std::make_shared<CONTACT::IntegratorNitschePoroScatra>(mortar_params, slave_type, comm);
      }
      else if (mortar_params.get<int>("PROBTYPE") == Inpar::CONTACT::fsi)
      {
        integrator =
            std::make_shared<CONTACT::IntegratorNitscheFsi>(mortar_params, slave_type, comm);
      }
      else if (mortar_params.get<int>("PROBTYPE") == Inpar::CONTACT::fpi)
      {
        integrator =
            std::make_shared<CONTACT::IntegratorNitscheFpi>(mortar_params, slave_type, comm);
      }
      else
      {
        integrator = std::make_shared<CONTACT::IntegratorNitsche>(mortar_params, slave_type, comm);
      }
      break;
    }
    case Inpar::CONTACT::solution_penalty:
    case Inpar::CONTACT::solution_multiscale:
    {
      if (Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(mortar_params, "ALGORITHM") ==
          Inpar::Mortar::algorithm_gpts)
        integrator = std::make_shared<CONTACT::IntegratorNitsche>(mortar_params, slave_type, comm);
      else
        integrator = std::make_shared<CONTACT::Integrator>(mortar_params, slave_type, comm);
      break;
    }
    case Inpar::CONTACT::solution_lagmult:
    case Inpar::CONTACT::solution_uzawa:
    {
      integrator = std::make_shared<CONTACT::Integrator>(mortar_params, slave_type, comm);
      break;
    }
    case Inpar::CONTACT::solution_ehl:
    {
      integrator = std::make_shared<CONTACT::IntegratorEhl>(mortar_params, slave_type, comm);

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
std::shared_ptr<CONTACT::Integrator> CONTACT::INTEGRATOR::build_integrator(
    const Inpar::CONTACT::SolvingStrategy& sol_type, Teuchos::ParameterList& mortar_params,
    const Core::FE::CellType& slave_type, MPI_Comm comm)
{
  Factory factory;
  return factory.build_integrator(sol_type, mortar_params, slave_type, comm);
}

FOUR_C_NAMESPACE_CLOSE
