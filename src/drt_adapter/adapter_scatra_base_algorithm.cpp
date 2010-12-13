/*----------------------------------------------------------------------*/
/*!
\file adapter_scatra_base_algorithm.cpp

\brief scalar transport field base algorithm

<pre>
Maintainer: Georg Bauer
            bauer@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15252
</pre>
*/
/*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "../drt_io/io_control.H"
#include "../drt_io/io.H"
#include "adapter_scatra_base_algorithm.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../linalg/linalg_solver.H"
#include "../drt_inpar/drt_validparameters.H"
#include "../drt_inpar/inpar_scatra.H"
#include "../drt_inpar/inpar_elch.H"
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include "../drt_scatra/scatra_timint_implicit.H"
#include "../drt_scatra/scatra_timint_stat.H"
#include "../drt_scatra/scatra_timint_ost.H"
#include "../drt_scatra/scatra_timint_bdf2.H"
#include "../drt_scatra/scatra_timint_genalpha.H"
#include "../drt_scatra/scatra_resulttest.H"

/*----------------------------------------------------------------------*
 |                                                       m.gee 06/01    |
 | general problem data                                                 |
 | global variable GENPROB genprob is defined in global_control.c       |
 *----------------------------------------------------------------------*/
extern struct _GENPROB     genprob;

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ADAPTER::ScaTraBaseAlgorithm::ScaTraBaseAlgorithm(
    const Teuchos::ParameterList& prbdyn,
    bool isale
)
{
  // setup scalar transport algorithm (overriding some dynamic parameters
  // with values specified in given problem-dependent ParameterList prbdyn)

  // -------------------------------------------------------------------
  // access the discretization
  // -------------------------------------------------------------------
  RCP<DRT::Discretization> actdis = null;
  actdis = DRT::Problem::Instance()->Dis(genprob.numscatra,0);

  // -------------------------------------------------------------------
  // set degrees of freedom in the discretization
  // -------------------------------------------------------------------
  if (!actdis->Filled()) actdis->FillComplete();

  // -------------------------------------------------------------------
  // context for output and restart
  // -------------------------------------------------------------------
  RCP<IO::DiscretizationWriter> output =
    rcp(new IO::DiscretizationWriter(actdis));
  output->WriteMesh(0,0.0);

  // -------------------------------------------------------------------
  // set some pointers and variables
  // -------------------------------------------------------------------

  const Teuchos::ParameterList& scatradyn =
    DRT::Problem::Instance()->ScalarTransportDynamicParams();

  // print out default parameters of scalar transport parameter list
  if (actdis->Comm().MyPID()==0)
  {
    DRT::INPUT::PrintDefaultParameters(std::cout, scatradyn);
    DRT::INPUT::PrintDefaultParameters(std::cout, scatradyn.sublist("STABILIZATION"));
    DRT::INPUT::PrintDefaultParameters(std::cout, scatradyn.sublist("NONLINEAR"));
//    DRT::INPUT::PrintDefaultParameters(std::cout, scatradyn.sublist("LEVELSET"));
  }
  // -------------------------------------------------------------------
  // create a solver
  // -------------------------------------------------------------------
  RCP<LINALG::Solver> solver =
    rcp(new LINALG::Solver(DRT::Problem::Instance()->ScalarTransportSolverParams(),
                           actdis->Comm(),
                           DRT::Problem::Instance()->ErrorFile()->Handle()));
  actdis->ComputeNullSpaceIfNecessary(solver->Params());

  // -------------------------------------------------------------------
  // set parameters in list required for all schemes
  // -------------------------------------------------------------------
  // make a copy (inside an rcp) containing also all sublists
  RCP<ParameterList> scatratimeparams= rcp(new ParameterList(scatradyn));

  // -------------------------------------------------------------------
  // overrule certain parameters for coupled problems
  // -------------------------------------------------------------------
  // the default time step size
  scatratimeparams->set<double>   ("TIMESTEP"    ,prbdyn.get<double>("TIMESTEP"));
  // maximum simulation time
  scatratimeparams->set<double>   ("MAXTIME"     ,prbdyn.get<double>("MAXTIME"));
  // maximum number of timesteps
  scatratimeparams->set<int>      ("NUMSTEP"     ,prbdyn.get<int>("NUMSTEP"));
  // restart
  scatratimeparams->set           ("RESTARTEVRY" ,prbdyn.get<int>("RESTARTEVRY"));
  // solution output
  scatratimeparams->set           ("UPRES"       ,prbdyn.get<int>("UPRES"));

  // -------------------------------------------------------------------
  // list for extra parameters
  // (put here everything that is not available in scatradyn or its sublists)
  // -------------------------------------------------------------------
  Teuchos::RCP<Teuchos::ParameterList> extraparams
    = Teuchos::rcp(new Teuchos::ParameterList());

  // ----problem type (type of scalar transport problem we want to solve)
  extraparams->set<string>("problem type",DRT::Problem::Instance()->ProblemType());

  // ------------------------------pointer to the error file (for output)
  extraparams->set<FILE*>("err file",DRT::Problem::Instance()->ErrorFile()->Handle());

  // ----------------Eulerian or ALE formulation of transport equation(s)
  extraparams->set<bool>("isale",isale);

  // --------------sublist for combustion-specific gfunction parameters
  /* This sublist COMBUSTION DYNAMIC/GFUNCTION contains parameters for the gfunction field
   * which are only relevant for a combustion problem.                         07/08 henke */
  if (genprob.probtyp == prb_combust)
  {
    extraparams->sublist("COMBUSTION GFUNCTION")=prbdyn.sublist("COMBUSTION GFUNCTION");
  }

  // -------------------sublist for electrochemistry-specific parameters
  if (genprob.probtyp == prb_elch)
  {
    // temperature of electrolyte solution
    extraparams->set<double>("TEMPERATURE",prbdyn.get<double>("TEMPERATURE"));

    // we provide all available electrochemistry-related parameters
    extraparams->sublist("ELCH CONTROL")=prbdyn;

    // create a 2nd solver for block-preconditioning if chosen from input
    if (Teuchos::getIntegralValue<int>(scatradyn,"BLOCKPRECOND"))
    {
      // switch to the SIMPLE(R) algorithms
      solver->PutSolverParamsToSubParams("SIMPLER",
         DRT::Problem::Instance()->ScalarTransportElectricPotentialSolverParams());
    }
  }

  // ------------------------------------get also fluid turbulence sublist
  const Teuchos::ParameterList& fdyn = DRT::Problem::Instance()->FluidDynamicParams();
  extraparams->sublist("TURBULENCE PARAMETERS")=fdyn.sublist("TURBULENCE MODEL");

  // -------------------------------------------------------------------
  // algorithm construction depending on
  // respective time-integration (or stationary) scheme
  // -------------------------------------------------------------------
   INPAR::SCATRA::TimeIntegrationScheme timintscheme =
     Teuchos::getIntegralValue<INPAR::SCATRA::TimeIntegrationScheme>(scatradyn,"TIMEINTEGR");

   switch(timintscheme)
   {
   case INPAR::SCATRA::timeint_stationary:
   {
     // create instance of time integration class (call the constructor)
     scatra_ = rcp(new SCATRA::TimIntStationary(actdis, solver, scatratimeparams, extraparams, output));
     break;
   }
   case INPAR::SCATRA::timeint_one_step_theta:
   {
     // create instance of time integration class (call the constructor)
     scatra_ = rcp(new SCATRA::TimIntOneStepTheta(actdis, solver, scatratimeparams, extraparams,output));
     break;
   }
   case INPAR::SCATRA::timeint_bdf2:
   {
     // create instance of time integration class (call the constructor)
     scatra_ = rcp(new SCATRA::TimIntBDF2(actdis, solver, scatratimeparams,extraparams, output));
     break;
   }
   case INPAR::SCATRA::timeint_gen_alpha:
   {
     // create instance of time integration class (call the constructor)
     scatra_ = rcp(new SCATRA::TimIntGenAlpha(actdis, solver, scatratimeparams,extraparams, output));
     break;
   }
   default:
     dserror("Unknown time-integration scheme for scalar tranport problem");
   }// switch(timintscheme)

  return;

}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
SCATRA::ScaTraTimIntImpl& ADAPTER::ScaTraBaseAlgorithm::ScaTraField()
{
  return *scatra_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<DRT::ResultTest> ADAPTER::ScaTraBaseAlgorithm::CreateScaTraFieldTest()
{
  return Teuchos::rcp(new SCATRA::ScaTraResultTest(*scatra_));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ADAPTER::ScaTraBaseAlgorithm::~ScaTraBaseAlgorithm()
{
}


#endif  // #ifdef CCADISCRET
