/*----------------------------------------------------------------------*/
/*!
\file inpar_tsi.cpp

\brief Input parameters for tsi

<pre>
Maintainer: Georg Hammerl
            hammerl@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
</pre>
*/

/*----------------------------------------------------------------------*/



#include "drt_validparameters.H"
#include "inpar_tsi.H"



void INPAR::TSI::SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list)
{
  using namespace DRT::INPUT;
  using Teuchos::tuple;
  using Teuchos::setStringToIntegralParameter;

  Teuchos::Array<std::string> yesnotuple = tuple<std::string>("Yes","No","yes","no","YES","NO");
  Teuchos::Array<int> yesnovalue = tuple<int>(true,false,true,false,true,false);

  Teuchos::ParameterList& tsidyn = list->sublist(
    "TSI DYNAMIC",false,
    "Thermo Structure Interaction\n"
    "Dynamic section for TSI solver with various coupling methods"
     );

  // coupling strategy for (partitioned and monolithic) TSI solvers
  setStringToIntegralParameter<int>("COUPALGO","tsi_monolithic",
    "Coupling strategies for TSI solvers",
    tuple<std::string>(
      "tsi_oneway",
      "tsi_sequstagg",
      "tsi_iterstagg",
      "tsi_iterstagg_aitken",
      "tsi_iterstagg_aitkenirons",
      "tsi_iterstagg_fixedrelax",
      "tsi_monolithic"),
    tuple<int>(
      OneWay,
      SequStagg,
      IterStagg,
      IterStaggAitken,
      IterStaggAitkenIrons,
      IterStaggFixedRel,
      Monolithic),
    &tsidyn
    );

  BoolParameter("MATCHINGGRID","Yes","is matching grid",&tsidyn);

  // coupling strategy for BACI-INCA coupling
  setStringToIntegralParameter<int>("TFSI_COUPALGO","tfsi",
    "Coupling strategies for BACI-INCA coupling (TFSI)",
    tuple<std::string>(
      "tfsi",
      "fsi",
      "conj_heat_transfer",
      "no_inca_fsi"),
    tuple<int>(
        TFSI,
        FSI,
        ConjHeatTransfer,
        NoIncaFSI),
    &tsidyn
    );

  // scaling factor for AeroTFSI problems when length unit other than SI [m] is used
  setStringToIntegralParameter<double>("TFSI_length_unit","m",
    "Used unit for extension in the structural model in AeroTFSI",
    tuple<std::string>(
      "m",
      "cm",
      "mm"),
    tuple<double>(
      1.0,
      100.0,
      1000.0),
    &tsidyn
    );

  // scaling factor for AeroTFSI problems when time unit other than SI [s] is used
  setStringToIntegralParameter<double>("TFSI_time_unit","s",
    "Used unit for time in the structural model in AeroTFSI",
    tuple<std::string>(
      "s",
      "ms",
      "mikros"),
    tuple<double>(
      1.0,
      1.0e3,
      1.0e6),
    &tsidyn
    );

  // output type
  IntParameter("RESTARTEVRY",1,"write restart possibility every RESTARTEVRY steps",&tsidyn);

  // time loop control
  IntParameter("NUMSTEP",200,"maximum number of Timesteps",&tsidyn);
  DoubleParameter("MAXTIME",1000.0,"total simulation time",&tsidyn);
  DoubleParameter("TIMESTEP",0.05,"time step size dt",&tsidyn);
  IntParameter("ITEMAX",10,"maximum number of iterations over fields",&tsidyn);
  IntParameter("ITEMIN",1,"minimal number of iterations over fields",&tsidyn);
  IntParameter("UPRES",1,"increment for writing solution",&tsidyn);

  setStringToIntegralParameter<int>("NORM_INC","Abs",
    "type of norm for convergence check of primary variables in TSI",
    tuple<std::string>(
      "Abs",
      "Rel",
      "Mix"),
    tuple<int>(
      convnorm_abs,
      convnorm_rel,
      convnorm_mix),
    &tsidyn
    );

  /*----------------------------------------------------------------------*/
  /* parameters for monolithic TSI */
  Teuchos::ParameterList& tsidynmono = tsidyn.sublist("MONOLITHIC",false,
    "Monolithic Thermo Structure Interaction\n"
    "Dynamic section for monolithic TSI"
    );

  // convergence tolerance of tsi residual
  DoubleParameter("CONVTOL",1e-6,"tolerance for convergence check of TSI",&tsidynmono);
  // Iterationparameters
  DoubleParameter("TOLINC",1.0e-6,"tolerance for convergence check of TSI-increment in monolithic TSI",&tsidynmono);

  setStringToIntegralParameter<int>("NORM_RESF","Abs",
    "type of norm for residual convergence check",
    tuple<std::string>(
      "Abs",
      "Rel",
      "Mix"),
    tuple<int>(
      convnorm_abs,
      convnorm_rel,
      convnorm_mix),
    &tsidynmono
    );

  setStringToIntegralParameter<int>("NORMCOMBI_RESFINC","Coupl_And_Singl",
    "binary operator to combine primary variables and residual force values",
    tuple<std::string>(
      "And",
      "Or",
      "Coupl_Or_Singl",
      "Coupl_And_Singl",
      "And_Singl",
      "Or_Singl"),
    tuple<int>(
      bop_and,
      bop_or,
      bop_coupl_or_singl,
      bop_coupl_and_singl,
      bop_and_singl,
      bop_or_singl),
    &tsidynmono
    );

  setStringToIntegralParameter<int>("ITERNORM","Rms",
    "type of norm to be applied to residuals",
    tuple<std::string>(
      "L1",
      "L1_Scaled",
      "L2",
      "Rms",
      "Inf"),
    tuple<int>(
      norm_l1,
      norm_l1_scaled,
      norm_l2,
      norm_rms,
      norm_inf),
    &tsidynmono
    );

  setStringToIntegralParameter<int>("NLNSOL","fullnewton",
    "Nonlinear solution technique",
    tuple<std::string>(
      "fullnewton",
      "ptc"),
    tuple<int>(
      soltech_newtonfull,
      soltech_ptc),
    &tsidynmono
    );

  DoubleParameter("PTCDT",0.1,
    "pseudo time step for pseudo-transient continuation (PTC) stabilised Newton procedure",
    &tsidynmono
    );

  // number of linear solver used for monolithic TSI
  IntParameter("LINEAR_SOLVER",-1,
    "number of linear solver used for monolithic TSI problems",
    &tsidynmono
    );

  // convergence criteria adaptivity of monolithic TSI solver
  setStringToIntegralParameter<int>("ADAPTCONV","No",
    "Switch on adaptive control of linear solver tolerance for nonlinear solution",
    yesnotuple,
    yesnovalue,
    &tsidynmono
    );
  DoubleParameter("ADAPTCONV_BETTER",0.1,
    "The linear solver shall be this much better than the current nonlinear residual in the nonlinear convergence limit",
    &tsidynmono
    );

  setStringToIntegralParameter<int>("INFNORMSCALING","yes",
    "Scale blocks of matrix with row infnorm?",
    yesnotuple,
    yesnovalue,
    &tsidynmono
    );

  // merge TSI block matrix to enable use of direct solver in monolithic TSI
  // default: "No", i.e. use block matrix
  BoolParameter("MERGE_TSI_BLOCK_MATRIX","No","Merge TSI block matrix",&tsidynmono);

  // in case of monolithic TSI nodal values (displacements, temperatures and
  // reaction forces) at fix points of the body can be calculated
  // default: "No", i.e. nothing is calculated
  BoolParameter("CALC_NECKING_TSI_VALUES","No",
    "Calculate nodal values for evaluation and validation of necking",
    &tsidynmono
    );

  /*----------------------------------------------------------------------*/
  /* parameters for partitioned TSI */
  Teuchos::ParameterList& tsidynpart = tsidyn.sublist(
      "PARTITIONED",false,
      "Partitioned Thermo Structure Interaction\n"
      "Dynamic section for partitioned TSI"
       );

  // decide in partitioned TSI which one-way coupling or predictor should be used
  setStringToIntegralParameter<int>("COUPVARIABLE","Displacement",
    "Coupling variable",
    tuple<std::string>(
      "Displacement",
      "Temperature"),
    tuple<int>(0,1),
    &tsidynpart
    );

  // Solver parameter for relaxation of iterative staggered partitioned TSI
  DoubleParameter("MAXOMEGA",0.0,"largest omega allowed for Aitken relaxation (0.0 means no constraint)",&tsidynpart);
  DoubleParameter("FIXEDOMEGA",1.0,"fixed relaxation parameter",&tsidynpart);

  // convergence tolerance of outer iteration loop
  DoubleParameter("CONVTOL",1e-6,"tolerance for convergence check of outer iteraiton within partitioned TSI",&tsidynpart);

  /*----------------------------------------------------------------------*/
  /* parameters for tsi contact */
  Teuchos::ParameterList& tsic = list->sublist("TSI CONTACT",false,"");

  DoubleParameter("HEATTRANSSLAVE",0.0,"Heat transfer parameter for slave side in thermal contact",&tsic);
  DoubleParameter("HEATTRANSMASTER",0.0,"Heat transfer parameter for master side in thermal contact",&tsic);
  DoubleParameter("TEMP_DAMAGE",1.0e12,"damage temperatue at contact interface: friction coefficient zero there",&tsic);
  DoubleParameter("TEMP_REF",0.0,"reference temperatue at contact interface: friction coefficient equals the given value",&tsic);

}
