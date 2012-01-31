/*----------------------------------------------------------------------*/
/*!
\file strtimint_create.cpp
\brief Creation of structural time integrators in accordance with user's wishes

<pre>
Maintainer: Thomas Klöppel
            kloeppel@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15257
</pre>
*/

/*----------------------------------------------------------------------*/
/* macros */
#ifdef CCADISCRET

/*----------------------------------------------------------------------*/
/* headers */
#include <ctime>
#include <cstdlib>
#include <iostream>

#include <Teuchos_StandardParameterEntryValidators.hpp>

#ifdef PARALLEL
#include <mpi.h>
#endif

#include "strtimint_create.H"
#include "strtimint_statics.H"
#include "strtimint_genalpha.H"
#include "strtimint_ost.H"
#include "strtimint_gemm.H"
#include "strtimint_expleuler.H"
#include "strtimint_centrdiff.H"
#include "strtimint_ab2.H"

#include "../drt_io/io.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_inpar/drt_validparameters.H"
#include "../linalg/linalg_utils.H"


/*======================================================================*/
/* create marching time integrator */
Teuchos::RCP<STR::TimInt> STR::TimIntCreate
(
  const Teuchos::ParameterList& ioflags,
  const Teuchos::ParameterList& sdyn,
  const Teuchos::ParameterList& xparams,
  Teuchos::RCP<DRT::Discretization>& actdis,
  Teuchos::RCP<LINALG::Solver>& solver,
  Teuchos::RCP<LINALG::Solver>& contactsolver,
  Teuchos::RCP<IO::DiscretizationWriter>& output
)
{
  // set default output
  Teuchos::RCP<STR::TimInt> sti = Teuchos::null;

  // exclude old names
  switch (DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(sdyn, "DYNAMICTYP"))
  {
    // old style time integrators
    case INPAR::STR::dyna_gen_alfa :
    case INPAR::STR::dyna_gen_alfa_statics :
    {
      dserror("You should not turn up here.");
      break;
    }

    // new style
    default :
    {
      // try implicit integrators
      sti = TimIntImplCreate(ioflags, sdyn, xparams, actdis, solver, contactsolver, output);
      // if nothing found try explicit integrators
      if (sti == Teuchos::null)
      {
        sti = TimIntExplCreate(ioflags, sdyn, xparams, actdis, solver, contactsolver, output);
      }
    }
  }

  // deliver
  return sti;
}

/*======================================================================*/
/* create implicit marching time integrator */
Teuchos::RCP<STR::TimIntImpl> STR::TimIntImplCreate
(
  const Teuchos::ParameterList& ioflags,
  const Teuchos::ParameterList& sdyn,
  const Teuchos::ParameterList& xparams,
  Teuchos::RCP<DRT::Discretization>& actdis,
  Teuchos::RCP<LINALG::Solver>& solver,
  Teuchos::RCP<LINALG::Solver>& contactsolver,
  Teuchos::RCP<IO::DiscretizationWriter>& output
)
{
  Teuchos::RCP<STR::TimIntImpl> sti = Teuchos::null;

  // TODO: add contact solver...

  // create specific time integrator
  switch (DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(sdyn, "DYNAMICTYP"))
  {
    // Static analysis
    case INPAR::STR::dyna_statics :
    {
      sti = Teuchos::rcp(new STR::TimIntStatics(ioflags, sdyn, xparams,
                                                actdis, solver, contactsolver, output));
      break;
    }

    // Generalised-alpha time integration
    case INPAR::STR::dyna_genalpha :
    {
      sti = Teuchos::rcp(new STR::TimIntGenAlpha(ioflags, sdyn, xparams,
                                                 actdis, solver, contactsolver, output));
      break;
    }

    // One-step-theta (OST) time integration
    case INPAR::STR::dyna_onesteptheta :
    {
      sti = Teuchos::rcp(new STR::TimIntOneStepTheta(ioflags, sdyn, xparams,
                                                     actdis, solver, contactsolver, output));
      break;
    }

    // Generalised energy-momentum method (GEMM)
    case INPAR::STR::dyna_gemm :
    {
      sti = Teuchos::rcp(new STR::TimIntGEMM(ioflags, sdyn, xparams,
                                             actdis, solver, contactsolver, output));
      break;
    }

    // Everything else
    default :
    {
      // do nothing
      break;
    }
  } // end of switch(sdyn->Typ)

  // return the integrator
  return sti;
}

/*======================================================================*/
/* create explicit marching time integrator */
Teuchos::RCP<STR::TimIntExpl> STR::TimIntExplCreate
(
  const Teuchos::ParameterList& ioflags,
  const Teuchos::ParameterList& sdyn,
  const Teuchos::ParameterList& xparams,
  Teuchos::RCP<DRT::Discretization>& actdis,
  Teuchos::RCP<LINALG::Solver>& solver,
  Teuchos::RCP<LINALG::Solver>& contactsolver,
  Teuchos::RCP<IO::DiscretizationWriter>& output
)
{
  Teuchos::RCP<STR::TimIntExpl> sti = Teuchos::null;

  // create specific time integrator
  switch (DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(sdyn, "DYNAMICTYP"))
  {
      // forward Euler time integration
    case INPAR::STR::dyna_expleuler :
    {
      sti = Teuchos::rcp(new STR::TimIntExplEuler(ioflags, sdyn, xparams,
                                            actdis, solver, contactsolver, output));
      break;
    }
    // central differences time integration
    case INPAR::STR::dyna_centrdiff:
    {
      sti = Teuchos::rcp(new STR::TimIntCentrDiff(ioflags, sdyn, xparams,
                                            actdis, solver, contactsolver, output));
      break;
    }
    // Adams-Bashforth 2nd order (AB2) time integration
    case INPAR::STR::dyna_ab2 :
    {
      sti = Teuchos::rcp(new STR::TimIntAB2(ioflags, sdyn, xparams,
                                            actdis, solver, contactsolver, output));
      break;
    }

    // Everything else
    default :
    {
      // do nothing
      break;
    }
  } // end of switch(sdyn->Typ)

  // return the integrator
  return sti;
}

/*----------------------------------------------------------------------*/
#endif  // #ifdef CCADISCRET
