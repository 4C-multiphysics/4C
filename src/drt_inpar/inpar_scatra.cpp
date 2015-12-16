/*----------------------------------------------------------------------*/
/*!
\file inpar_scatra.cpp

\brief Input parameters for scatra

<pre>
Maintainer: Anh-Tu Vuong
            vuong@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
</pre>
*/

/*----------------------------------------------------------------------*/



#include "drt_validparameters.H"
#include "inpar_scatra.H"
#include "inpar_fluid.H"
#include "inpar_thermo.H"
#include "inpar_s2i.H"
#include "../drt_lib/drt_conditiondefinition.H"



void INPAR::SCATRA::SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list)
{
  using namespace DRT::INPUT;
  using Teuchos::tuple;
  using Teuchos::setStringToIntegralParameter;

  Teuchos::ParameterList& scatradyn = list->sublist(
      "SCALAR TRANSPORT DYNAMIC",
      false,
      "control parameters for scalar transport problems\n");

  setStringToIntegralParameter<int>("SOLVERTYPE","linear_full",
                               "type of scalar transport solver",
                               tuple<std::string>(
                                 "linear_full",
                                 "linear_incremental",
                                 "nonlinear"
                                 ),
                               tuple<int>(
                                   solvertype_linear_full,
                                   solvertype_linear_incremental,
                                   solvertype_nonlinear),
                               &scatradyn);

  setStringToIntegralParameter<int>("TIMEINTEGR","One_Step_Theta",
                               "Time Integration Scheme",
                               tuple<std::string>(
                                 "Stationary",
                                 "One_Step_Theta",
                                 "BDF2",
                                 "Gen_Alpha"
                                 ),
                               tuple<int>(
                                   timeint_stationary,
                                   timeint_one_step_theta,
                                   timeint_bdf2,
                                   timeint_gen_alpha
                                 ),
                               &scatradyn);

  DoubleParameter("MAXTIME",1000.0,"Total simulation time",&scatradyn);
  IntParameter("NUMSTEP",20,"Total number of time steps",&scatradyn);
  DoubleParameter("TIMESTEP",0.1,"Time increment dt",&scatradyn);
  DoubleParameter("THETA",0.5,"One-step-theta time integration factor",&scatradyn);
  DoubleParameter("ALPHA_M",0.5,"Generalized-alpha time integration factor",&scatradyn);
  DoubleParameter("ALPHA_F",0.5,"Generalized-alpha time integration factor",&scatradyn);
  DoubleParameter("GAMMA",0.5,"Generalized-alpha time integration factor",&scatradyn);
  //IntParameter("WRITESOLEVRY",1,"Increment for writing solution",&scatradyn);
  IntParameter("UPRES",1,"Increment for writing solution",&scatradyn);
  IntParameter("RESTARTEVRY",1,"Increment for writing restart",&scatradyn);
  IntParameter("MATID",-1,"Material ID for automatic mesh generation",&scatradyn);

  setStringToIntegralParameter<int>("VELOCITYFIELD","zero",
                               "type of velocity field used for scalar transport problems",
                               tuple<std::string>(
                                 "zero",
                                 "function",
                                 "function_and_curve",
                                 "Navier_Stokes"
                                 ),
                               tuple<int>(
                                   velocity_zero,
                                   velocity_function,
                                   velocity_function_and_curve,
                                   velocity_Navier_Stokes),
                               &scatradyn);

  IntParameter("VELFUNCNO",-1,"function number for scalar transport velocity field",&scatradyn);

  IntParameter("VELCURVENO",-1,"curve number for time-dependent scalar transport velocity field",&scatradyn);

  {
    // a standard Teuchos::tuple can have at maximum 10 entries! We have to circumvent this here.
    Teuchos::Tuple<std::string,12> name;
    Teuchos::Tuple<int,12> label;
    name[ 0] = "zero_field";                   label[ 0] = initfield_zero_field;
    name[ 1] = "field_by_function";            label[ 1] = initfield_field_by_function;
    name[ 2] = "field_by_condition";           label[ 2] = initfield_field_by_condition;
    name[ 3] = "disturbed_field_by_function";  label[ 3] = initfield_disturbed_field_by_function;
    name[ 4] = "1D_DISCONTPV";                 label[ 4] = initfield_discontprogvar_1D;
    name[ 5] = "FLAME_VORTEX_INTERACTION";     label[ 5] = initfield_flame_vortex_interaction;
    name[ 6] = "RAYTAYMIXFRAC";                label[ 6] = initfield_raytaymixfrac;
    name[ 7] = "L_shaped_domain";              label[ 7] = initfield_Lshapeddomain;
    name[ 8] = "facing_flame_fronts";          label[ 8] = initfield_facing_flame_fronts;
    name[ 9] = "oracles_flame";                label[ 9] = initfield_oracles_flame;
    name[10] = "high_forced_hit";              label[10] = initialfield_forced_hit_high_Sc;
    name[11] = "low_forced_hit";               label[11] = initialfield_forced_hit_low_Sc;

    setStringToIntegralParameter<int>(
        "INITIALFIELD",
        "zero_field",
        "Initial Field for scalar transport problem",
        name,
        label,
        &scatradyn);
  }

  IntParameter("INITFUNCNO",-1,"function number for scalar transport initial field",&scatradyn);

  setStringToIntegralParameter<int>("CALCERROR","No",
                               "compute error compared to analytical solution",
                               tuple<std::string>(
                                 "No",
                                 "Kwok_Wu",
                                 "ConcentricCylinders",
                                 "Electroneutrality",
                                 "error_by_function",
                                 "SphereDiffusion"
                                 ),
                               tuple<int>(
                                   calcerror_no,
                                   calcerror_Kwok_Wu,
                                   calcerror_cylinder,
                                   calcerror_electroneutrality,
                                   calcerror_byfunction,
                                   calcerror_spherediffusion
                                   ),
                               &scatradyn);

  IntParameter("CALCERRORNO",-1,"function number for scalar transport error computation",&scatradyn);

  setStringToIntegralParameter<int>("WRITEFLUX","No","output of diffusive/total flux vectors",
                               tuple<std::string>(
                                 "No",
                                 "totalflux_domain",
                                 "diffusiveflux_domain",
                                 "totalflux_boundary",
                                 "diffusiveflux_boundary",
                                 "convectiveflux_boundary"
                                 ),
                               tuple<int>(
                                   flux_no,
                                   flux_total_domain,
                                   flux_diffusive_domain,
                                   flux_total_boundary,
                                   flux_diffusive_boundary,
                                   flux_convective_boundary),
                               &scatradyn);

  // Parameters for reaction-diffusion systems (for example cardiac electrophysiology)
  IntParameter("WRITEMAXINTSTATE",0,"number of maximal internal state variables to be postprocessed",&scatradyn);
  IntParameter("WRITEMAXIONICCURRENTS",0,"number of maximal ionic currents to be postprocessed",&scatradyn);
  DoubleParameter("ACTTHRES",1.0,"threshold for the potential for computing and postprocessing activation time ",&scatradyn);

  setNumericStringParameter("WRITEFLUX_IDS","-1",
      "Write diffusive/total flux vector fields for these scalar fields only (starting with 1)",
      &scatradyn);

  BoolParameter("OUTMEAN","No","Output of mean values for scalars and density",&scatradyn);
  BoolParameter("OUTINTEGRREAC","No","Output of integral reaction values",&scatradyn);
  BoolParameter("OUTPUT_GMSH","No","Do you want to write Gmsh postprocessing files?",&scatradyn);

  BoolParameter("MATLAB_STATE_OUTPUT","No","Do you want to write the state solution to Matlab file?",&scatradyn);

  setStringToIntegralParameter<int>("CONVFORM","convective","form of convective term",
                               tuple<std::string>(
                                 "convective",
                                 "conservative"
                                 ),
                               tuple<int>(
                                 convform_convective,
                                 convform_conservative),
                               &scatradyn);

  BoolParameter("NEUMANNINFLOW",
      "no","Flag to (de)activate potential Neumann inflow term(s)",&scatradyn);

  BoolParameter("CONV_HEAT_TRANS",
      "no","Flag to (de)activate potential convective heat transfer boundary conditions",&scatradyn);

  BoolParameter("SKIPINITDER",
      "no","Flag to skip computation of initial time derivative",&scatradyn);

  setStringToIntegralParameter<int>("FSSUGRDIFF",
                               "No",
                               "fine-scale subgrid diffusivity",
                               tuple<std::string>(
                                 "No",
                                 "artificial",
                                 "Smagorinsky_all",
                                 "Smagorinsky_small"
                                 ),
                               tuple<int>(
                                   fssugrdiff_no,
                                   fssugrdiff_artificial,
                                   fssugrdiff_smagorinsky_all,
                                   fssugrdiff_smagorinsky_small),
                               &scatradyn);

  setStringToIntegralParameter<int>("MESHTYING", "no", "Flag to (de)activate mesh tying algorithm",
                                  tuple<std::string>(
                                      "no",
                                      "Condensed_Smat",
                                      "Condensed_Bmat",
                                      "Condensed_Bmat_merged"), //use the condensed_bmat_merged strategy
                                    tuple<int>(
                                        INPAR::FLUID::no_meshtying,
                                        INPAR::FLUID::condensed_smat,
                                        INPAR::FLUID::condensed_bmat,
                                        INPAR::FLUID::condensed_bmat_merged),   //use the condensed_bmat_merged strategy
                                    &scatradyn);

  // linear solver id used for scalar transport/elch problems
  IntParameter("LINEAR_SOLVER",-1,"number of linear solver used for scalar transport/elch...",&scatradyn);
  //IntParameter("SIMPLER_SOLVER",-1,"number of linear solver used for ELCH (solved with SIMPLER)...",&scatradyn);

  // parameters for natural convection effects
  BoolParameter("NATURAL_CONVECTION","No","Include natural convection effects",&scatradyn);
  IntParameter("NATCONVITEMAX",10,"Maximum number of outer iterations for natural convection",&scatradyn);
  DoubleParameter("NATCONVCONVTOL",1e-6,"Convergence check tolerance for outer loop for natural convection",&scatradyn);

  // parameters for finite difference check
  setStringToIntegralParameter<int>("FDCHECK", "none", "flag for finite difference check: none, local, or global",
                                    tuple<std::string>(
                                      "none",
                                      "local",    // perform finite difference check on element level
                                      "global"),  // perform finite difference check on time integrator level
                                    tuple<int>(
                                        fdcheck_none,
                                        fdcheck_local,
                                        fdcheck_global),
                                    &scatradyn);
  DoubleParameter("FDCHECKEPS",1.e-6,"dof perturbation magnitude for finite difference check (1.e-6 seems to work very well, whereas smaller values don't)",&scatradyn);
  DoubleParameter("FDCHECKTOL",1.e-6,"relative tolerance for finite difference check",&scatradyn);

  // parameter for optional computation of domain and boundary integrals, i.e., of surface areas and volumes associated with specified nodesets
  setStringToIntegralParameter<int>(
      "COMPUTEINTEGRALS",
      "none",
      "flag for optional computation of domain integrals",
      tuple<std::string>(
          "none",
          "initial",
          "repeated"
          ),
      tuple<int>(
          computeintegrals_none,
          computeintegrals_initial,
          computeintegrals_repeated
          ),
      &scatradyn
      );

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& scatra_nonlin = scatradyn.sublist(
      "NONLINEAR",
      false,
      "control parameters for solving nonlinear SCATRA problems\n");

  IntParameter("ITEMAX",10,"max. number of nonlin. iterations",&scatra_nonlin);
  DoubleParameter("CONVTOL",1e-6,"Tolerance for convergence check",&scatra_nonlin);
  BoolParameter("EXPLPREDICT","no","do an explicit predictor step before starting nonlinear iteration",&scatra_nonlin);
  DoubleParameter("ABSTOLRES",1e-14,"Absolute tolerance for deciding if residual of nonlinear problem is already zero",&scatra_nonlin);

  // convergence criteria adaptivity
  BoolParameter("ADAPTCONV","yes","Switch on adaptive control of linear solver tolerance for nonlinear solution",&scatra_nonlin);
  DoubleParameter("ADAPTCONV_BETTER",0.1,"The linear solver shall be this much better than the current nonlinear residual in the nonlinear convergence limit",&scatra_nonlin);

/*----------------------------------------------------------------------*/
  Teuchos::ParameterList& scatradyn_stab = scatradyn.sublist("STABILIZATION",false,"control parameters for the stabilization of scalar transport problems");

  // this parameter governs type of stabilization
  setStringToIntegralParameter<int>("STABTYPE",
                                    "SUPG",
                                    "type of stabilization (if any)",
                               tuple<std::string>(
                                 "no_stabilization",
                                 "SUPG",
                                 "GLS",
                                 "USFEM"),
                               tuple<std::string>(
                                 "Do not use any stabilization -> only reasonable for low-Peclet-number flows",
                                 "Use SUPG",
                                 "Use GLS",
                                 "Use USFEM")  ,
                               tuple<int>(
                                   stabtype_no_stabilization,
                                   stabtype_SUPG,
                                   stabtype_GLS,
                                   stabtype_USFEM),
                               &scatradyn_stab);

  // this parameter governs whether subgrid-scale velocity is included
  BoolParameter("SUGRVEL","no","potential incorporation of subgrid-scale velocity",&scatradyn_stab);

  // this parameter governs whether all-scale subgrid diffusivity is included
  BoolParameter("ASSUGRDIFF","no",
      "potential incorporation of all-scale subgrid diffusivity (a.k.a. discontinuity-capturing) term",&scatradyn_stab);

  // this parameter selects the tau definition applied
  setStringToIntegralParameter<int>("DEFINITION_TAU",
                               "Franca_Valentin",
                               "Definition of tau",
                               tuple<std::string>(
                                 "Taylor_Hughes_Zarins",
                                 "Taylor_Hughes_Zarins_wo_dt",
                                 "Franca_Valentin",
                                 "Franca_Valentin_wo_dt",
                                 "Shakib_Hughes_Codina",
                                 "Shakib_Hughes_Codina_wo_dt",
                                 "Codina",
                                 "Codina_wo_dt",
                                 "Franca_Madureira_Valentin",
                                 "Franca_Madureira_Valentin_wo_dt",
                                 "Exact_1D",
                                 "Zero"),
                                tuple<int>(
                                    tau_taylor_hughes_zarins,
                                    tau_taylor_hughes_zarins_wo_dt,
                                    tau_franca_valentin,
                                    tau_franca_valentin_wo_dt,
                                    tau_shakib_hughes_codina,
                                    tau_shakib_hughes_codina_wo_dt,
                                    tau_codina,
                                    tau_codina_wo_dt,
                                    tau_franca_madureira_valentin,
                                    tau_franca_madureira_valentin_wo_dt,
                                    tau_exact_1d,
                                    tau_zero),
                               &scatradyn_stab);

  // this parameter selects the characteristic element length for tau for all
  // stabilization parameter definitions requiring such a length
  setStringToIntegralParameter<int>("CHARELELENGTH",
                               "streamlength",
                               "Characteristic element length for tau",
                               tuple<std::string>(
                                 "streamlength",
                                 "volume_equivalent_diameter",
                                 "root_of_volume"),
                               tuple<int>(
                                   streamlength,
                                   volume_equivalent_diameter,
                                   root_of_volume),
                               &scatradyn_stab);

  // this parameter selects the all-scale subgrid-diffusivity definition applied
  setStringToIntegralParameter<int>("DEFINITION_ASSGD",
                               "artificial_linear",
                               "Definition of (all-scale) subgrid diffusivity",
                               tuple<std::string>(
                                 "artificial_linear",
                                 "artificial_linear_reinit",
                                 "Hughes_etal_86_nonlinear",
                                 "Tezduyar_Park_86_nonlinear",
                                 "Tezduyar_Park_86_nonlinear_wo_phizero",
                                 "doCarmo_Galeao_91_nonlinear",
                                 "Almeida_Silva_97_nonlinear",
                                 "YZbeta_nonlinear",
                                 "Codina_nonlinear"),
                               tuple<std::string>(
                                 "classical linear artificial subgrid-diffusivity",
                                 "simple linear artificial subgrid-diffusivity const*h",
                                 "nonlinear isotropic according to Hughes et al. (1986)",
                                 "nonlinear isotropic according to Tezduyar and Park (1986)",
                                 "nonlinear isotropic according to Tezduyar and Park (1986) without user parameter phi_zero",
                                 "nonlinear isotropic according to doCarmo and Galeao (1991)",
                                 "nonlinear isotropic according to Almeida and Silva (1997)",
                                 "nonlinear YZ beta model",
                                 "nonlinear isotropic according to Codina")  ,
                                tuple<int>(
                                    assgd_artificial,
                                    assgd_lin_reinit,
                                    assgd_hughes,
                                    assgd_tezduyar,
                                    assgd_tezduyar_wo_phizero,
                                    assgd_docarmo,
                                    assgd_almeida,
                                    assgd_yzbeta,
                                    assgd_codina),
                               &scatradyn_stab);

  // this parameter selects the location where tau is evaluated
  setStringToIntegralParameter<int>("EVALUATION_TAU",
                               "element_center",
                               "Location where tau is evaluated",
                               tuple<std::string>(
                                 "element_center",
                                 "integration_point"),
                               tuple<std::string>(
                                 "evaluate tau at element center",
                                 "evaluate tau at integration point")  ,
                                tuple<int>(
                                  evaltau_element_center,
                                  evaltau_integration_point),
                               &scatradyn_stab);

  // this parameter selects the location where the material law is evaluated
  // (does not fit here very well, but parameter transfer is easier)
  setStringToIntegralParameter<int>("EVALUATION_MAT",
                               "element_center",
                               "Location where material law is evaluated",
                               tuple<std::string>(
                                 "element_center",
                                 "integration_point"),
                               tuple<std::string>(
                                 "evaluate material law at element center",
                                 "evaluate material law at integration point"),
                               tuple<int>(
                                 evalmat_element_center,
                                 evalmat_integration_point),
                               &scatradyn_stab);

  // this parameter selects methods for improving consistency of stabilization terms
  setStringToIntegralParameter<int>("CONSISTENCY",
                               "no",
                               "improvement of consistency for stabilization",
                               tuple<std::string>(
                                 "no",
                                 "L2_projection_lumped"),
                               tuple<std::string>(
                                 "inconsistent",
                                 "L2 projection with lumped mass matrix")  ,
                                tuple<int>(
                                  consistency_no,
                                  consistency_l2_projection_lumped),
                               &scatradyn_stab);

  /*-------------------------------------------------------------------------*/

  Teuchos::ParameterList& stidyn = list->sublist(
      "STI DYNAMIC",
      false,
      "general control parameters for scatra-thermo interaction problems"
      );

  // type of scalar transport
  setStringToIntegralParameter<int>(
      "SCATRATYPE",
      "Undefined",
      "type of scalar transport",
       tuple<std::string>(
           "Undefined",
           "ConvectionDiffusion",
           "Elch"
           ),
       tuple<int>(
           INPAR::SCATRA::impltype_undefined,
           INPAR::SCATRA::impltype_std,
           INPAR::SCATRA::impltype_elch_diffcond   // we abuse this enumeration entry here to indicate electrochemistry in general
           ),
       &stidyn
       );

  // specification of initial temperature field
  setStringToIntegralParameter<int>(
      "THERMO_INITIALFIELD",
      "zero_field",
      "initial temperature field for scatra-thermo interaction problems",
      tuple<std::string>(
          "zero_field",
          "field_by_function",
          "field_by_condition"
          ),
      tuple<int>(
          INPAR::SCATRA::initfield_zero_field,
          INPAR::SCATRA::initfield_field_by_function,
          INPAR::SCATRA::initfield_field_by_condition
          ),
      &stidyn
      );

  // function number for initial temperature field
  IntParameter("THERMO_INITFUNCNO",-1,"function number for initial temperature field for scatra-thermo interaction problems",&stidyn);

}



void INPAR::SCATRA::SetValidConditions(std::vector<Teuchos::RCP<DRT::INPUT::ConditionDefinition> >& condlist)
{
  using namespace DRT::INPUT;

  // scatra-scatra interface coupling (slave side)
  {
    // definition of scatra-scatra interface coupling line condition (slave side)
    Teuchos::RCP<ConditionDefinition> s2ilineslave =
        Teuchos::rcp(new ConditionDefinition("DESIGN S2I COUPLING LINE CONDITIONS / SLAVE",
                                             "S2ICouplingSlave",
                                             "Scatra-scatra line interface coupling (slave side)",
                                             DRT::Condition::S2ICouplingSlave,
                                             true,
                                             DRT::Condition::Line));

    // definition of scatra-scatra interface coupling surface condition (slave side)
    Teuchos::RCP<ConditionDefinition> s2isurfslave =
        Teuchos::rcp(new ConditionDefinition("DESIGN S2I COUPLING SURF CONDITIONS / SLAVE",
                                             "S2ICouplingSlave",
                                             "Scatra-scatra surface interface coupling (slave side)",
                                             DRT::Condition::S2ICouplingSlave,
                                             true,
                                             DRT::Condition::Surface));

    // equip condition definitions with input file line components
    std::vector<Teuchos::RCP<ConditionComponent> > s2icomponents;

    {
      // kinetic models for scatra-scatra interface coupling
      std::vector<Teuchos::RCP<CondCompBundle> > kineticmodels;
      {
        {
          // constant permeability
          std::vector<Teuchos::RCP<ConditionComponent> > constperm;
          constperm.push_back(Teuchos::rcp(new SeparatorConditionComponent("numscal")));                // total number of existing scalars
          std::vector<Teuchos::RCP<SeparatorConditionComponent> > intsepcomp;                           // empty vector --> no separators for integer vectors needed
          std::vector<Teuchos::RCP<IntVectorConditionComponent> > intvectcomp;                          // empty vector --> no integer vectors needed
          std::vector<Teuchos::RCP<SeparatorConditionComponent> > realsepcomp;
          realsepcomp.push_back(Teuchos::rcp(new SeparatorConditionComponent("permeabilities")));       // string separator in front of real permeability vector in input file line
          std::vector<Teuchos::RCP<RealVectorConditionComponent> > realvectcomp;
          realvectcomp.push_back(Teuchos::rcp(new RealVectorConditionComponent("permeabilities",0)));   // real vector of constant permeabilities
          constperm.push_back(Teuchos::rcp(new IntRealBundle(
              "permeabilities",
              Teuchos::rcp(new IntConditionComponent("numscal")),
              intsepcomp,
              intvectcomp,
              realsepcomp,
              realvectcomp
          )));
          kineticmodels.push_back(Teuchos::rcp(new CondCompBundle("ConstantPermeability",constperm,INPAR::S2I::kinetics_constperm)));
        }

        {
          // Butler-Volmer
          std::vector<Teuchos::RCP<ConditionComponent> > butlervolmer;
          butlervolmer.push_back(Teuchos::rcp(new SeparatorConditionComponent("numscal")));            // total number of existing scalars
          std::vector<Teuchos::RCP<SeparatorConditionComponent> > intsepcomp;
          intsepcomp.push_back(Teuchos::rcp(new SeparatorConditionComponent("stoichiometries")));
          std::vector<Teuchos::RCP<IntVectorConditionComponent> > intvectcomp;                         // string separator in front of integer stoichiometry vector in input file line
          intvectcomp.push_back(Teuchos::rcp(new IntVectorConditionComponent("stoichiometries",0)));   // integer vector of stoichiometric coefficients
          std::vector<Teuchos::RCP<SeparatorConditionComponent> > realsepcomp;                         // empty vector --> no separators for real vectors needed
          std::vector<Teuchos::RCP<RealVectorConditionComponent> > realvectcomp;                       // empty vector --> no real vectors needed
          butlervolmer.push_back(Teuchos::rcp(new IntRealBundle(
              "stoichiometries",
              Teuchos::rcp(new IntConditionComponent("numscal")),
              intsepcomp,
              intvectcomp,
              realsepcomp,
              realvectcomp
          )));
          butlervolmer.push_back(Teuchos::rcp(new SeparatorConditionComponent("e-")));
          butlervolmer.push_back(Teuchos::rcp(new IntConditionComponent("e-")));
          butlervolmer.push_back(Teuchos::rcp(new SeparatorConditionComponent("k_r")));
          butlervolmer.push_back(Teuchos::rcp(new RealConditionComponent("k_r")));
          butlervolmer.push_back(Teuchos::rcp(new SeparatorConditionComponent("alpha_a")));
          butlervolmer.push_back(Teuchos::rcp(new RealConditionComponent("alpha_a")));
          butlervolmer.push_back(Teuchos::rcp(new SeparatorConditionComponent("alpha_c")));
          butlervolmer.push_back(Teuchos::rcp(new RealConditionComponent("alpha_c")));

          kineticmodels.push_back(Teuchos::rcp(new CondCompBundle("Butler-Volmer",butlervolmer,INPAR::S2I::kinetics_butlervolmer)));
        }

        {
          // Butler-Volmer-Peltier
          std::vector<Teuchos::RCP<ConditionComponent> > butlervolmerpeltier;
          butlervolmerpeltier.push_back(Teuchos::rcp(new SeparatorConditionComponent("numscal")));            // total number of existing scalars
          std::vector<Teuchos::RCP<SeparatorConditionComponent> > intsepcomp;
          intsepcomp.push_back(Teuchos::rcp(new SeparatorConditionComponent("stoichiometries")));
          std::vector<Teuchos::RCP<IntVectorConditionComponent> > intvectcomp;                         // string separator in front of integer stoichiometry vector in input file line
          intvectcomp.push_back(Teuchos::rcp(new IntVectorConditionComponent("stoichiometries",0)));   // integer vector of stoichiometric coefficients
          std::vector<Teuchos::RCP<SeparatorConditionComponent> > realsepcomp;                         // empty vector --> no separators for real vectors needed
          std::vector<Teuchos::RCP<RealVectorConditionComponent> > realvectcomp;                       // empty vector --> no real vectors needed
          butlervolmerpeltier.push_back(Teuchos::rcp(new IntRealBundle(
              "stoichiometries",
              Teuchos::rcp(new IntConditionComponent("numscal")),
              intsepcomp,
              intvectcomp,
              realsepcomp,
              realvectcomp
          )));
          butlervolmerpeltier.push_back(Teuchos::rcp(new SeparatorConditionComponent("e-")));
          butlervolmerpeltier.push_back(Teuchos::rcp(new IntConditionComponent("e-")));
          butlervolmerpeltier.push_back(Teuchos::rcp(new SeparatorConditionComponent("k_r")));
          butlervolmerpeltier.push_back(Teuchos::rcp(new RealConditionComponent("k_r")));
          butlervolmerpeltier.push_back(Teuchos::rcp(new SeparatorConditionComponent("alpha_a")));
          butlervolmerpeltier.push_back(Teuchos::rcp(new RealConditionComponent("alpha_a")));
          butlervolmerpeltier.push_back(Teuchos::rcp(new SeparatorConditionComponent("alpha_c")));
          butlervolmerpeltier.push_back(Teuchos::rcp(new RealConditionComponent("alpha_c")));
          butlervolmerpeltier.push_back(Teuchos::rcp(new SeparatorConditionComponent("peltier")));
          butlervolmerpeltier.push_back(Teuchos::rcp(new RealConditionComponent("peltier")));

          kineticmodels.push_back(Teuchos::rcp(new CondCompBundle("Butler-Volmer-Peltier",butlervolmerpeltier,INPAR::S2I::kinetics_butlervolmerpeltier)));
        }
      } // kinetic models for scatra-scatra interface coupling

      // insert kinetic models into vector with input file line components
      s2icomponents.push_back(Teuchos::rcp(new SeparatorConditionComponent("KineticModel")));
      s2icomponents.push_back(Teuchos::rcp(new CondCompBundleSelector(
          "kinetic models for scatra-scatra interface coupling",
          Teuchos::rcp(new StringConditionComponent(
             "kinetic model",
             "ConstantPermeability",
             Teuchos::tuple<std::string>("ConstantPermeability","Butler-Volmer","Butler-Volmer-Peltier"),
             Teuchos::tuple<int>(INPAR::S2I::kinetics_constperm,INPAR::S2I::kinetics_butlervolmer,INPAR::S2I::kinetics_butlervolmerpeltier))),
          kineticmodels)));
    }

    // insert input file line components into condition definitions
    for (unsigned i=0; i<s2icomponents.size(); ++i)
    {
      s2ilineslave->AddComponent(s2icomponents[i]);
      s2isurfslave->AddComponent(s2icomponents[i]);
    }

    // insert condition definitions into global list of valid condition definitions
    condlist.push_back(s2ilineslave);
    condlist.push_back(s2isurfslave);
  }


  /*--------------------------------------------------------------------*/
  // scatra-scatra interface coupling (master side)
  {
    // definition of scatra-scatra interface coupling line condition (master side)
    Teuchos::RCP<ConditionDefinition> s2ilinemaster =
        Teuchos::rcp(new ConditionDefinition("DESIGN S2I COUPLING LINE CONDITIONS / MASTER",
                                             "S2ICouplingMaster",
                                             "Scatra-scatra line interface coupling (master side)",
                                             DRT::Condition::S2ICouplingMaster,
                                             true,
                                             DRT::Condition::Line));

    // definition of scatra-scatra interface coupling surface condition (master side)
    Teuchos::RCP<ConditionDefinition> s2isurfmaster =
        Teuchos::rcp(new ConditionDefinition("DESIGN S2I COUPLING SURF CONDITIONS / MASTER",
                                             "S2ICouplingMaster",
                                             "Scatra-scatra surface interface coupling (master side)",
                                             DRT::Condition::S2ICouplingMaster,
                                             true,
                                             DRT::Condition::Surface));

    // insert condition definitions into global list of valid condition definitions
    condlist.push_back(s2ilinemaster);
    condlist.push_back(s2isurfmaster);
  }


  /*--------------------------------------------------------------------*/
  // scatra-scatra interface coupling (domain partitioning for block preconditioning of global system matrix)
  {
    // partitioning of 2D domain into 2D subdomains
    Teuchos::RCP<ConditionDefinition> s2ilinepartitioning =
        Teuchos::rcp(new ConditionDefinition("DESIGN S2I COUPLING SURF CONDITIONS / PARTITIONING",
                                             "S2ICouplingPartitioning",
                                             "Scatra-scatra line interface coupling (domain partitioning)",
                                             DRT::Condition::S2ICouplingPartitioning,
                                             false,
                                             DRT::Condition::Surface));

    // partitioning of 3D domain into 3D subdomains
    Teuchos::RCP<ConditionDefinition> s2isurfpartitioning =
        Teuchos::rcp(new ConditionDefinition("DESIGN S2I COUPLING VOL CONDITIONS / PARTITIONING",
                                             "S2ICouplingPartitioning",
                                             "Scatra-scatra surface interface coupling (domain partitioning)",
                                             DRT::Condition::S2ICouplingPartitioning,
                                             false,
                                             DRT::Condition::Volume));

    // insert condition definitions into global list of valid condition definitions
    condlist.push_back(s2ilinepartitioning);
    condlist.push_back(s2isurfpartitioning);
  }


  /*--------------------------------------------------------------------*/
  // Boundary flux evaluation condition for scalar transport
  Teuchos::RCP<ConditionDefinition> linebndryfluxeval =
    Teuchos::rcp(new ConditionDefinition("SCATRA FLUX CALC LINE CONDITIONS",
                                         "ScaTraFluxCalc",
                                         "Scalar Transport Boundary Flux Calculation",
                                         DRT::Condition::ScaTraFluxCalc,
                                         true,
                                         DRT::Condition::Line));
  Teuchos::RCP<ConditionDefinition> surfbndryfluxeval =
    Teuchos::rcp(new ConditionDefinition("SCATRA FLUX CALC SURF CONDITIONS",
                                         "ScaTraFluxCalc",
                                         "Scalar Transport Boundary Flux Calculation",
                                         DRT::Condition::ScaTraFluxCalc,
                                         true,
                                         DRT::Condition::Surface));
  condlist.push_back(linebndryfluxeval);
  condlist.push_back(surfbndryfluxeval);

  /*--------------------------------------------------------------------*/
  // Coupling of different scalar transport fields

  std::vector<Teuchos::RCP<ConditionComponent> > scatracoupcomponents;

  std::vector<Teuchos::RCP<SeparatorConditionComponent> > KKintsepveccompstoich;
  KKintsepveccompstoich.push_back(Teuchos::rcp(new SeparatorConditionComponent("ONOFF")));
  // definition int vectors
  std::vector<Teuchos::RCP<IntVectorConditionComponent> > KKintveccompstoich;
  KKintveccompstoich.push_back(Teuchos::rcp(new IntVectorConditionComponent("onoff",2)));
  // definition separator for real vectors: length of the real vector is zero -> nothing is read
  std::vector<Teuchos::RCP<SeparatorConditionComponent> > KKrealsepveccompstoich;
  // definition real vectors: length of the real vector is zero -> nothing is read
  std::vector<Teuchos::RCP<RealVectorConditionComponent> > KKrealveccompstoich;


  scatracoupcomponents.push_back(Teuchos::rcp(new SeparatorConditionComponent("NUMSCAL")) );
  scatracoupcomponents.push_back(Teuchos::rcp(new IntRealBundle(
                                 "intreal bundle numscal",
                                Teuchos::rcp(new IntConditionComponent("numscal")),
                                KKintsepveccompstoich,
                                KKintveccompstoich,
                                KKrealsepveccompstoich,
                                KKrealveccompstoich)) );
  scatracoupcomponents.push_back(Teuchos::rcp(new SeparatorConditionComponent("COUPID")));
  scatracoupcomponents.push_back(Teuchos::rcp(new IntConditionComponent("coupling id")));
  scatracoupcomponents.push_back(Teuchos::rcp(new SeparatorConditionComponent("PERMCOEF")));
  scatracoupcomponents.push_back(Teuchos::rcp(new RealConditionComponent("permeability coefficient")));
  scatracoupcomponents.push_back(Teuchos::rcp(new SeparatorConditionComponent("CONDUCT")));
  scatracoupcomponents.push_back(Teuchos::rcp(new RealConditionComponent("hydraulic conductivity")));
  scatracoupcomponents.push_back(Teuchos::rcp(new SeparatorConditionComponent("FILTR")));
  scatracoupcomponents.push_back(Teuchos::rcp(new RealConditionComponent("filtration coefficient")));
  scatracoupcomponents.push_back(Teuchos::rcp(new SeparatorConditionComponent("WSSONOFF")));
  scatracoupcomponents.push_back(Teuchos::rcp(new IntConditionComponent("wss onoff")));
  scatracoupcomponents.push_back(Teuchos::rcp(new SeparatorConditionComponent("WSSCOEFFS")));
  scatracoupcomponents.push_back(Teuchos::rcp(new RealVectorConditionComponent("wss coeffs",2)));


  Teuchos::RCP<ConditionDefinition> surfscatracoup =
    Teuchos::rcp(new ConditionDefinition("DESIGN SCATRA COUPLING SURF CONDITIONS",
                                         "ScaTraCoupling",
                                         "ScaTra Coupling",
                                         DRT::Condition::ScaTraCoupling,
                                         true,
                                         DRT::Condition::Surface));

  for (unsigned i=0; i<scatracoupcomponents.size(); ++i)
  {
    surfscatracoup->AddComponent(scatracoupcomponents[i]);
  }

  condlist.push_back(surfscatracoup);

  /*--------------------------------------------------------------------*/
  // Robin boundary condition for scalar transport problems
  // line
  Teuchos::RCP<ConditionDefinition> scatrarobinline =
    Teuchos::rcp(new ConditionDefinition("SCATRA ROBIN LINE CONDITIONS",
                                         "ScatraRobin",
                                         "Scalar Transport Robin Boundary Condition",
                                         DRT::Condition::TransportRobin,
                                         true,
                                         DRT::Condition::Line));
  // surface
  Teuchos::RCP<ConditionDefinition> scatrarobinsurf =
    Teuchos::rcp(new ConditionDefinition("SCATRA ROBIN SURF CONDITIONS",
                                         "ScatraRobin",
                                         "Scalar Transport Robin Boundary Condition",
                                         DRT::Condition::TransportRobin,
                                         true,
                                         DRT::Condition::Surface));

  std::vector<Teuchos::RCP<ConditionComponent> > scatrarobincomponents;
  scatrarobincomponents.push_back(Teuchos::rcp(new SeparatorConditionComponent("Prefactor")));
  scatrarobincomponents.push_back(Teuchos::rcp(new RealConditionComponent("Prefactor")));
  scatrarobincomponents.push_back(Teuchos::rcp(new SeparatorConditionComponent("Refvalue")));
  scatrarobincomponents.push_back(Teuchos::rcp(new RealConditionComponent("Refvalue")));

  for(unsigned i=0; i<scatrarobincomponents.size(); ++i)
  {
    scatrarobinline->AddComponent(scatrarobincomponents[i]);
    scatrarobinsurf->AddComponent(scatrarobincomponents[i]);
  }

  condlist.push_back(scatrarobinline);
  condlist.push_back(scatrarobinsurf);

  /*--------------------------------------------------------------------*/
  // Neumann inflow for SCATRA

  Teuchos::RCP<ConditionDefinition> linetransportneumanninflow =
    Teuchos::rcp(new ConditionDefinition("TRANSPORT NEUMANN INFLOW LINE CONDITIONS",
                                         "TransportNeumannInflow",
                                         "Line Transport Neumann Inflow",
                                         DRT::Condition::TransportNeumannInflow,
                                         true,
                                         DRT::Condition::Line));
  Teuchos::RCP<ConditionDefinition> surftransportneumanninflow =
    Teuchos::rcp(new ConditionDefinition("TRANSPORT NEUMANN INFLOW SURF CONDITIONS",
                                         "TransportNeumannInflow",
                                         "Surface Transport Neumann Inflow",
                                         DRT::Condition::TransportNeumannInflow,
                                         true,
                                         DRT::Condition::Surface));

   condlist.push_back(linetransportneumanninflow);
   condlist.push_back(surftransportneumanninflow);
}
