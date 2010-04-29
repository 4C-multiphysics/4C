/*!----------------------------------------------------------------------
\file
\brief

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#ifdef PARALLEL
#include <mpi.h>
#endif

#include "stru_dyn_nln_drt.H"
#include "strugenalpha.H"
#include "strudyn_direct.H"
#include "../drt_beamcontact/beam3contactstrugenalpha.H"
#include "../drt_contact/strugenalpha_cmt.H"
#include "../drt_io/io.H"
#include "../drt_io/io_control.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_inpar/inpar_contact.H"
#include "../drt_inpar/inpar_statmech.H"
#include "../drt_inpar/inpar_structure.H"
#include "../drt_inpar/inpar_invanalysis.H"
#include "stru_resulttest.H"

#include "str_invanalysis.H"
#include "../drt_inv_analysis/inv_analysis.H"

#include "../drt_statmech/statmech_time.H"

/*----------------------------------------------------------------------*
 |                                                       m.gee 06/01    |
 | general problem data                                                 |
 | global variable GENPROB genprob is defined in global_control.c       |
 *----------------------------------------------------------------------*/
extern struct _GENPROB     genprob;

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
extern "C"
void caldyn_drt()
{
  // get input lists
  const Teuchos::ParameterList& iap = DRT::Problem::Instance()->InverseAnalysisParams();

  // do we want to do inverse analysis?
  if (Teuchos::getIntegralValue<INPAR::STR::InvAnalysisType>(iap,"INV_ANALYSIS")
      != INPAR::STR::inv_none)
  {
    STR::invanalysis();
  }
  else
  {
    // get input lists
    const Teuchos::ParameterList& sdyn = DRT::Problem::Instance()->StructuralDynamicParams();

    // major switch to different time integrators
    switch (Teuchos::getIntegralValue<INPAR::STR::DynamicType>(sdyn,"DYNAMICTYP"))
    {
    case INPAR::STR::dyna_centr_diff:
      dserror("no central differences in DRT");
      break;
    case INPAR::STR::dyna_gen_alfa:
    case INPAR::STR::dyna_gen_alfa_statics:
    case INPAR::STR::dyna_statics:
    case INPAR::STR::dyna_genalpha:
    case INPAR::STR::dyna_onesteptheta:
    case INPAR::STR::dyna_gemm:
    case INPAR::STR::dyna_ab2:
    case INPAR::STR::dyna_euma :
    case INPAR::STR::dyna_euimsto :
      dyn_nlnstructural_drt();
      break;
    case INPAR::STR::dyna_Gen_EMM:
      dserror("GEMM not supported");
      break;
    default:
      dserror("unknown time integration scheme '%s'", sdyn.get<std::string>("DYNAMICTYP").c_str());
    }
  }
}


/*----------------------------------------------------------------------*
  | structural nonlinear dynamics (gen-alpha)              m.gee 12/06  |
 *----------------------------------------------------------------------*/
void dyn_nlnstructural_drt()
{

#if 1
  // the adapter expects a couple of variables that do not exist in the StructuralDynamicParams()
  // list so rename them here to the expected name
  int upres = DRT::Problem::Instance()->StructuralDynamicParams().get<int>("RESEVRYDISP");
  const_cast<Teuchos::ParameterList&>(DRT::Problem::Instance()->StructuralDynamicParams()).set<int>("UPRES",upres);

  // create an adapterbase and adapter
  ADAPTER::StructureBaseAlgorithm adapterbase(DRT::Problem::Instance()->StructuralDynamicParams());
  ADAPTER::Structure& structadaptor = const_cast<ADAPTER::Structure&>(adapterbase.StructureField());

  // do restart
  if (genprob.restart)
  {
    structadaptor.ReadRestart(genprob.restart);
  }
  
  // write output at beginnning of calc
  else
  {
    //RCP<DRT::Discretization> actdis = DRT::Problem::Instance()->Dis(genprob.numsf,0);
    //RCP<IO::DiscretizationWriter> output = rcp(new IO::DiscretizationWriter(actdis));
    //output->NewStep(0, 0.0);
    //RCP<Epetra_Vector> zeros = rcp (new Epetra_Vector(*(actdis->DofRowMap())));
    //output->WriteVector("displacement",zeros);
    //output->WriteElementData();
  }
  
  // run time integration
  structadaptor.Integrate();

  // test results
  DRT::Problem::Instance()->AddFieldTest(structadaptor.CreateFieldTest());
  DRT::Problem::Instance()->TestAll(structadaptor.DofRowMap()->Comm());

  // time to go home...
  return;
#else

  // -------------------------------------------------------------------
  // access the discretization
  // -------------------------------------------------------------------
  RCP<DRT::Discretization> actdis = null;
  actdis = DRT::Problem::Instance()->Dis(genprob.numsf,0);

  // set degrees of freedom in the discretization
  if (!actdis->Filled() || !actdis->HaveDofs()) actdis->FillComplete();

  // -------------------------------------------------------------------
  // context for output and restart
  // -------------------------------------------------------------------
  IO::DiscretizationWriter output(actdis);

  // -------------------------------------------------------------------
  // set some pointers and variables
  // -------------------------------------------------------------------
  const Teuchos::ParameterList& probtype = DRT::Problem::Instance()->ProblemTypeParams();
  const Teuchos::ParameterList& ioflags  = DRT::Problem::Instance()->IOParams();
  const Teuchos::ParameterList& sdyn     = DRT::Problem::Instance()->StructuralDynamicParams();
  const Teuchos::ParameterList& scontact = DRT::Problem::Instance()->MeshtyingAndContactParams();
  const Teuchos::ParameterList& statmech = DRT::Problem::Instance()->StatisticalMechanicsParams();

  if (actdis->Comm().MyPID()==0)
    DRT::INPUT::PrintDefaultParameters(std::cout, sdyn);

  // -------------------------------------------------------------------
  // create a solver
  // -------------------------------------------------------------------
  LINALG::Solver solver(DRT::Problem::Instance()->StructSolverParams(),
                        actdis->Comm(),
                        DRT::Problem::Instance()->ErrorFile()->Handle());
  actdis->ComputeNullSpaceIfNecessary(solver.Params());

  // -------------------------------------------------------------------
  // create a generalized alpha time integrator
  // -------------------------------------------------------------------
  switch (Teuchos::getIntegralValue<INPAR::STR::DynamicType>(sdyn,"DYNAMICTYP"))
  {
    //==================================================================
    // Generalized alpha time integration
    //==================================================================
    case INPAR::STR::dyna_gen_alfa :
    case INPAR::STR::dyna_gen_alfa_statics :
    {
      ParameterList genalphaparams;
      StruGenAlpha::SetDefaults(genalphaparams);

      genalphaparams.set<string>("DYNAMICTYP",sdyn.get<string>("DYNAMICTYP"));

      INPAR::STR::ControlType controltype = Teuchos::getIntegralValue<INPAR::STR::ControlType>(sdyn,"CONTROLTYPE");
      genalphaparams.set<INPAR::STR::ControlType>("CONTROLTYPE",controltype);
      {
        vector<int> controlnode;
        std::istringstream contnode(Teuchos::getNumericStringParameter(sdyn,"CONTROLNODE"));
        std::string word;
        while (contnode >> word)
          controlnode.push_back(std::atoi(word.c_str()));
        if ((int)controlnode.size() != 3) dserror("Give proper values for CONTROLNODE in input file");
        genalphaparams.set("CONTROLNODE",controlnode[0]);
        genalphaparams.set("CONTROLDOF",controlnode[1]);
        genalphaparams.set("CONTROLCURVE",controlnode[2]);
      }

      {
        // use linearization of follower loads in Newton
        int loadlin = Teuchos::getIntegralValue<int>(sdyn,"LOADLIN");
        genalphaparams.set<bool>("LOADLIN",loadlin!=0);
      }

      // Rayleigh damping
      genalphaparams.set<bool>  ("damping",(not (sdyn.get<std::string>("DAMPING") == "no"
                                                 or sdyn.get<std::string>("DAMPING") == "No"
                                                 or sdyn.get<std::string>("DAMPING") == "NO")));
      genalphaparams.set<double>("damping factor K",sdyn.get<double>("K_DAMP"));
      genalphaparams.set<double>("damping factor M",sdyn.get<double>("M_DAMP"));

      // Generalised-alpha coefficients
      genalphaparams.set<double>("beta",sdyn.get<double>("BETA"));
#ifdef STRUGENALPHA_BE
      genalphaparams.set<double>("delta",sdyn.get<double>("DELTA"));
#endif
      genalphaparams.set<double>("gamma",sdyn.get<double>("GAMMA"));
      genalphaparams.set<double>("alpha m",sdyn.get<double>("ALPHA_M"));
      genalphaparams.set<double>("alpha f",sdyn.get<double>("ALPHA_F"));

      genalphaparams.set<double>("total time",0.0);
      genalphaparams.set<double>("delta time",sdyn.get<double>("TIMESTEP"));
      genalphaparams.set<double>("max time",sdyn.get<double>("MAXTIME"));
      genalphaparams.set<int>   ("step",0);
      genalphaparams.set<int>   ("nstep",sdyn.get<int>("NUMSTEP"));
      genalphaparams.set<int>   ("max iterations",sdyn.get<int>("MAXITER"));
      genalphaparams.set<int>   ("num iterations",-1);

      genalphaparams.set<string>("convcheck", sdyn.get<string>("CONV_CHECK"));
      genalphaparams.set<double>("tolerance displacements",sdyn.get<double>("TOLDISP"));
      genalphaparams.set<double>("tolerance residual",sdyn.get<double>("TOLRES"));
      genalphaparams.set<double>("tolerance constraint",sdyn.get<double>("TOLCONSTR"));

      genalphaparams.set<double>("UZAWAPARAM",sdyn.get<double>("UZAWAPARAM"));
      genalphaparams.set<double>("UZAWATOL",sdyn.get<double>("UZAWATOL"));
      genalphaparams.set<int>   ("UZAWAMAXITER",sdyn.get<int>("UZAWAMAXITER"));
      genalphaparams.set<INPAR::STR::ConSolveAlgo>("UZAWAALGO",getIntegralValue<INPAR::STR::ConSolveAlgo>(sdyn,"UZAWAALGO"));
      genalphaparams.set<bool>  ("io structural disp",Teuchos::getIntegralValue<int>(ioflags,"STRUCT_DISP"));
      genalphaparams.set<int>   ("io disp every nstep",sdyn.get<int>("RESEVRYDISP"));

      genalphaparams.set<bool>  ("ADAPTCONV",getIntegralValue<int>(sdyn,"ADAPTCONV")==1);
      genalphaparams.set<double>("ADAPTCONV_BETTER",sdyn.get<double>("ADAPTCONV_BETTER"));

      INPAR::STR::StressType iostress = Teuchos::getIntegralValue<INPAR::STR::StressType>(ioflags,"STRUCT_STRESS");
      genalphaparams.set<INPAR::STR::StressType>("io structural stress", iostress);

      genalphaparams.set<int>   ("io stress every nstep",sdyn.get<int>("RESEVRYSTRS"));

      INPAR::STR::StrainType iostrain = Teuchos::getIntegralValue<INPAR::STR::StrainType>(ioflags,"STRUCT_STRAIN");
      genalphaparams.set<INPAR::STR::StrainType>("io structural strain", iostrain);

      genalphaparams.set<bool>  ("io surfactant",Teuchos::getIntegralValue<int>(ioflags,"STRUCT_SURFACTANT"));

      genalphaparams.set<int>   ("restart",probtype.get<int>("RESTART"));
      genalphaparams.set<int>   ("write restart every",sdyn.get<int>("RESTARTEVRY"));

      genalphaparams.set<bool>  ("print to screen",true);
      genalphaparams.set<bool>  ("print to err",true);
      genalphaparams.set<FILE*> ("err file",DRT::Problem::Instance()->ErrorFile()->Handle());

      // non-linear solution technique
      switch (Teuchos::getIntegralValue<INPAR::STR::NonlinSolTech>(sdyn,"NLNSOL"))
      {
        case INPAR::STR::soltech_newtonfull:
          genalphaparams.set<string>("equilibrium iteration","full newton");
        break;
        case INPAR::STR::soltech_newtonls:
          genalphaparams.set<string>("equilibrium iteration","line search newton");
        break;
        case INPAR::STR::soltech_newtonopp:
          genalphaparams.set<string>("equilibrium iteration","oppositely converging newton");
        break;
        case INPAR::STR::soltech_newtonmod:
          genalphaparams.set<string>("equilibrium iteration","modified newton");
        break;
        case INPAR::STR::soltech_nlncg:
          genalphaparams.set<string>("equilibrium iteration","nonlinear cg");
        break;
        case INPAR::STR::soltech_ptc:
          genalphaparams.set<string>("equilibrium iteration","ptc");
        break;
        case INPAR::STR::soltech_newtonuzawalin:
          genalphaparams.set<string>("equilibrium iteration","newtonlinuzawa");
        break;
        case INPAR::STR::soltech_newtonuzawanonlin:
          genalphaparams.set<string>("equilibrium iteration","augmentedlagrange");
        break;
        default:
          genalphaparams.set<string>("equilibrium iteration","full newton");
        break;
      }

      // set predictor (takes values "constant" "consistent")
      switch (Teuchos::getIntegralValue<INPAR::STR::PredEnum>(sdyn,"PREDICT"))
      {
        case INPAR::STR::pred_vague:
          dserror("You have to define the predictor");
          break;
        case INPAR::STR::pred_constdis:
          genalphaparams.set<string>("predictor","consistent");
          break;
        case INPAR::STR::pred_constdisvelacc:
          genalphaparams.set<string>("predictor","constant");
          break;
        case INPAR::STR::pred_tangdis:
          genalphaparams.set<string>("predictor","tangdis");
          break;
        default:
          dserror("Cannot cope with choice of predictor");
          break;
      }

      // detect if contact or meshtying are present
      bool mortarcontact = false;
      bool mortarmeshtying = false;
      bool beamcontact = false;
      INPAR::CONTACT::ApplicationType ctype =
        Teuchos::getIntegralValue<INPAR::CONTACT::ApplicationType>(scontact,"APPLICATION");
      switch (ctype)
      {
        case INPAR::CONTACT::app_none:
          break;
        case INPAR::CONTACT::app_mortarcontact:
          mortarcontact = true;
          break;
        case INPAR::CONTACT::app_mortarmeshtying:
          mortarmeshtying = true;
          break;
        case INPAR::CONTACT::app_beamcontact:
          beamcontact = true;
          break;
        default:
          dserror("Cannot cope with choice of contact or meshtying type");
          break;
      }

      // detect whether thermal bath is present
      bool thermalbath = false;
      switch (Teuchos::getIntegralValue<INPAR::STATMECH::ThermalBathType>(statmech,"THERMALBATH"))
      {
        case INPAR::STATMECH::thermalbath_none:
          thermalbath = false;
          break;
        case INPAR::STATMECH::thermalbath_uniform:
          thermalbath = true;
          break;
        case INPAR::STATMECH::thermalbath_shearflow:
          thermalbath = true;
          break;
        default:
          dserror("Cannot cope with choice of thermal bath");
          break;
      }

      // create the time integrator
      bool inv_analysis = genalphaparams.get("inv_analysis",false);
      RCP<StruGenAlpha> tintegrator;
      if (!mortarcontact && !mortarmeshtying && !beamcontact && !inv_analysis && !thermalbath)
        tintegrator = rcp(new StruGenAlpha(genalphaparams,*actdis,solver,output));
      else
      {
        if (mortarcontact)
          tintegrator = rcp(new CONTACT::CmtStruGenAlpha(genalphaparams,*actdis,solver,output));
        if (mortarmeshtying)
          tintegrator = rcp (new CONTACT::CmtStruGenAlpha(genalphaparams,*actdis,solver,output));
        if (beamcontact)
          tintegrator = rcp(new CONTACT::Beam3ContactStruGenAlpha(genalphaparams,*actdis,solver,output));
        if (thermalbath)
          tintegrator = rcp(new StatMechTime(genalphaparams,*actdis,solver,output));
      }

      // do restart if demanded from input file
      // note that this changes time and step in genalphaparams
      if (genprob.restart)
        tintegrator->ReadRestart(genprob.restart);

      // write mesh always at beginning of calc or restart
      {
        int    step = genalphaparams.get<int>("step",0);
        double time = genalphaparams.get<double>("total time",0.0);
        output.WriteMesh(step,time);
      }

      // integrate in time and space
      tintegrator->Integrate();

      // test results
      {
        DRT::Problem::Instance()->AddFieldTest(rcp(new StruResultTest(*tintegrator)));
        DRT::Problem::Instance()->TestAll(actdis->Comm());
      }

    }
    break;
    //==================================================================
    // Generalized Energy Momentum Method
    //==================================================================
    case INPAR::STR::dyna_Gen_EMM :
    {
      dserror("Not yet impl.");
    }
    break;
    //==================================================================
    // Everything else
    //==================================================================
    default :
    {
      dserror("Time integration scheme is not available");
    }
    break;
  } // end of switch(sdyn->Typ)

  Teuchos::TimeMonitor::summarize();

  return;
#endif
} // end of dyn_nlnstructural_drt()

#endif  // #ifdef CCADISCRET
