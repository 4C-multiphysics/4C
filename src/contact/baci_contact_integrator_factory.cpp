/*---------------------------------------------------------------------*/
/*! \file
\brief Factory to create the desired integrator object.

\level 2


*/
/*---------------------------------------------------------------------*/

#include "baci_contact_integrator_factory.H"

// supported contact integrators
#include "baci_contact_aug_integrator.H"
#include "baci_contact_ehl_integrator.H"
#include "baci_contact_nitsche_integrator.H"
#include "baci_contact_nitsche_integrator_fpi.H"
#include "baci_contact_nitsche_integrator_fsi.H"
#include "baci_contact_nitsche_integrator_poro.H"
#include "baci_contact_nitsche_integrator_ssi.H"
#include "baci_contact_nitsche_integrator_ssi_elch.H"
#include "baci_contact_nitsche_integrator_tsi.H"

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<CONTACT::CoIntegrator> CONTACT::INTEGRATOR::Factory::BuildIntegrator(
    const INPAR::CONTACT::SolvingStrategy& sol_type, Teuchos::ParameterList& p_mortar,
    const DRT::Element::DiscretizationType& slave_type, const Epetra_Comm& comm) const
{
  Teuchos::RCP<CONTACT::CoIntegrator> integrator = Teuchos::null;
  switch (sol_type)
  {
    case INPAR::CONTACT::solution_augmented:
    case INPAR::CONTACT::solution_std_lagrange:
    case INPAR::CONTACT::solution_steepest_ascent:
    case INPAR::CONTACT::solution_steepest_ascent_sp:
    case INPAR::CONTACT::solution_combo:
    {
      integrator = Teuchos::rcp<CONTACT::CoIntegrator>(
          new CONTACT::AUG::IntegrationWrapper(p_mortar, slave_type, comm));
      break;
    }
    case INPAR::CONTACT::solution_nitsche:
    {
      if (p_mortar.get<int>("PROBTYPE") == INPAR::CONTACT::tsi)
      {
        integrator = Teuchos::rcp(new CONTACT::CoIntegratorNitscheTsi(p_mortar, slave_type, comm));
      }
      else if (p_mortar.get<int>("PROBTYPE") == INPAR::CONTACT::ssi)
      {
        integrator = Teuchos::rcp(new CONTACT::CoIntegratorNitscheSsi(p_mortar, slave_type, comm));
      }
      else if (p_mortar.get<int>("PROBTYPE") == INPAR::CONTACT::ssi_elch)
      {
        integrator =
            Teuchos::rcp(new CONTACT::CoIntegratorNitscheSsiElch(p_mortar, slave_type, comm));
      }
      else if (p_mortar.get<int>("PROBTYPE") == INPAR::CONTACT::poro)
      {
        integrator = Teuchos::rcp(new CONTACT::CoIntegratorNitschePoro(p_mortar, slave_type, comm));
      }
      else if (p_mortar.get<int>("PROBTYPE") == INPAR::CONTACT::fsi)
      {
        integrator = Teuchos::rcp(new CONTACT::CoIntegratorNitscheFsi(p_mortar, slave_type, comm));
      }
      else if (p_mortar.get<int>("PROBTYPE") == INPAR::CONTACT::fpi)
      {
        integrator = Teuchos::rcp(new CONTACT::CoIntegratorNitscheFpi(p_mortar, slave_type, comm));
      }
      else
      {
        integrator = Teuchos::rcp(new CONTACT::CoIntegratorNitsche(p_mortar, slave_type, comm));
      }
      break;
    }
    case INPAR::CONTACT::solution_penalty:
    case INPAR::CONTACT::solution_multiscale:
    {
      if (DRT::INPUT::IntegralValue<INPAR::MORTAR::AlgorithmType>(p_mortar, "ALGORITHM") ==
          INPAR::MORTAR::algorithm_gpts)
        integrator = Teuchos::rcp(new CONTACT::CoIntegratorNitsche(p_mortar, slave_type, comm));
      else
        integrator = Teuchos::rcp(new CONTACT::CoIntegrator(p_mortar, slave_type, comm));
      break;
    }
    case INPAR::CONTACT::solution_lagmult:
    case INPAR::CONTACT::solution_uzawa:
    {
      integrator = Teuchos::rcp(new CONTACT::CoIntegrator(p_mortar, slave_type, comm));
      break;
    }
    case INPAR::CONTACT::solution_ehl:
    {
      integrator = Teuchos::rcp(new CONTACT::CoIntegratorEhl(p_mortar, slave_type, comm));

      break;
    }
    default:
    {
      dserror("Unsupported solving strategy! (stype = %s | %d)",
          INPAR::CONTACT::SolvingStrategy2String(sol_type).c_str(), sol_type);
      exit(EXIT_FAILURE);
    }
  }  // end switch

  return integrator;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<CONTACT::CoIntegrator> CONTACT::INTEGRATOR::BuildIntegrator(
    const INPAR::CONTACT::SolvingStrategy& sol_type, Teuchos::ParameterList& p_mortar,
    const DRT::Element::DiscretizationType& slave_type, const Epetra_Comm& comm)
{
  Factory factory;
  return factory.BuildIntegrator(sol_type, p_mortar, slave_type, comm);
}
