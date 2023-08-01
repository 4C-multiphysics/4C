/*----------------------------------------------------------------------*/
/*! \file
\brief Input parameters for biomedical simulations

\level 3


*/
/*----------------------------------------------------------------------*/



#include "baci_inpar_bio.H"

#include "baci_inpar_validparameters.H"
#include "baci_lib_conditiondefinition.H"


void INPAR::ARTDYN::SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list)
{
  using namespace DRT::INPUT;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;
  Teuchos::ParameterList& andyn = list->sublist("ARTERIAL DYNAMIC", false, "");

  setStringToIntegralParameter<int>("DYNAMICTYP", "ExpTaylorGalerkin",
      "Explicit Taylor Galerkin Scheme", tuple<std::string>("ExpTaylorGalerkin", "Stationary"),
      tuple<int>(tay_gal, stationary), &andyn);

  DoubleParameter("TIMESTEP", 0.01, "Time increment dt", &andyn);
  IntParameter("NUMSTEP", 0, "Number of Time Steps", &andyn);
  DoubleParameter("MAXTIME", 1000.0, "total simulation time", &andyn);
  IntParameter("RESTARTEVRY", 1, "Increment for writing restart", &andyn);
  IntParameter("RESULTSEVRY", 1, "Increment for writing solution", &andyn);
  setStringToIntegralParameter<int>("SOLVESCATRA", "no",
      "Flag to (de)activate solving scalar transport in blood", tuple<std::string>("no", "yes"),
      tuple<std::string>("do not solve scatra", "solve scatra"), tuple<int>(0, 1), &andyn);

  // number of linear solver used for arterial dynamics
  IntParameter("LINEAR_SOLVER", -1, "number of linear solver used for arterial dynamics", &andyn);

  // initial function number
  IntParameter("INITFUNCNO", -1, "function number for artery initial field", &andyn);

  // type of initial field
  setStringToIntegralParameter<int>("INITIALFIELD", "zero_field",
      "Initial Field for artery problem",
      tuple<std::string>("zero_field", "field_by_function", "field_by_condition"),
      tuple<int>(initfield_zero_field, initfield_field_by_function, initfield_field_by_condition),
      &andyn);
}



void INPAR::ARTNET::SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list)
{
  using namespace DRT::INPUT;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& redtisdyn =
      list->sublist("COUPLED REDUCED-D AIRWAYS AND TISSUE DYNAMIC", false, "");
  DoubleParameter("CONVTOL_P", 1E-6,
      "Coupled red_airway and tissue iteration convergence for pressure", &redtisdyn);
  DoubleParameter("CONVTOL_Q", 1E-6, "Coupled red_airway and tissue iteration convergence for flux",
      &redtisdyn);
  IntParameter("MAXITER", 5, "Maximum coupling iterations", &redtisdyn);
  setStringToIntegralParameter<int>("RELAXTYPE", "norelaxation", "Dynamic Relaxation Type",
      tuple<std::string>("norelaxation", "fixedrelaxation", "Aitken", "SD"),
      tuple<int>(norelaxation, fixedrelaxation, Aitken, SD), &redtisdyn);
  DoubleParameter("TIMESTEP", 0.01, "Time increment dt", &redtisdyn);
  IntParameter("NUMSTEP", 1, "Number of Time Steps", &redtisdyn);
  DoubleParameter("MAXTIME", 4.0, "", &redtisdyn);
  DoubleParameter("NORMAL", 1.0, "", &redtisdyn);
}



void INPAR::ARTNET::SetValidConditions(
    std::vector<Teuchos::RCP<DRT::INPUT::ConditionDefinition>>& condlist)
{
  using namespace DRT::INPUT;

  /*--------------------------------------------------------------------*/
  // 1D-Artery connector condition

  Teuchos::RCP<ConditionDefinition> art_connection_bc =
      Teuchos::rcp(new ConditionDefinition("DESIGN NODE 1D ARTERY JUNCTION CONDITIONS",
          "ArtJunctionCond", "Artery junction boundary condition", DRT::Condition::ArtJunctionCond,
          true, DRT::Condition::Point));

  art_connection_bc->AddComponent(Teuchos::rcp(new IntConditionComponent("ConditionID")));
  art_connection_bc->AddComponent(Teuchos::rcp(new RealConditionComponent("Kr")));

  condlist.push_back(art_connection_bc);

  /*--------------------------------------------------------------------*/
  // Export 1D-Arterial nefrk in gnuplot format

  Teuchos::RCP<ConditionDefinition> art_write_gnuplot_c =
      Teuchos::rcp(new ConditionDefinition("DESIGN LINE EXPORT 1D-ARTERIAL NETWORK GNUPLOT FORMAT",
          "ArtWriteGnuplotCond", "Artery write gnuplot format condition",
          DRT::Condition::ArtWriteGnuplotCond, false, DRT::Condition::Line));

  art_write_gnuplot_c->AddComponent(Teuchos::rcp(new IntConditionComponent("ArteryNumber")));

  condlist.push_back(art_write_gnuplot_c);

  /*--------------------------------------------------------------------*/
  // 1D artery prescribed BC

  Teuchos::RCP<ConditionDefinition> art_in_bc =
      Teuchos::rcp(new ConditionDefinition("DESIGN NODE 1D ARTERY PRESCRIBED CONDITIONS",
          "ArtPrescribedCond", "Artery prescribed boundary condition",
          DRT::Condition::ArtPrescribedCond, true, DRT::Condition::Point));

  art_in_bc->AddComponent(Teuchos::rcp(new StringConditionComponent("boundarycond", "flow",
      Teuchos::tuple<std::string>("flow", "pressure", "velocity", "area", "characteristicWave"),
      Teuchos::tuple<std::string>("flow", "pressure", "velocity", "area", "characteristicWave"),
      true)));
  art_in_bc->AddComponent(Teuchos::rcp(new StringConditionComponent("type", "forced",
      Teuchos::tuple<std::string>("forced", "absorbing"),
      Teuchos::tuple<std::string>("forced", "absorbing"), true)));

  std::vector<Teuchos::RCP<ConditionComponent>> artinletcomponents;
  artinletcomponents.push_back(Teuchos::rcp(new RealVectorConditionComponent("val", 2)));
  artinletcomponents.push_back(
      Teuchos::rcp(new IntVectorConditionComponent("curve", 2, true, true)));
  for (unsigned i = 0; i < artinletcomponents.size(); ++i)
    art_in_bc->AddComponent(artinletcomponents[i]);

  condlist.push_back(art_in_bc);

  /*--------------------------------------------------------------------*/
  // 1D artery reflective BC
  Teuchos::RCP<ConditionDefinition> art_rf_bc = Teuchos::rcp(
      new ConditionDefinition("DESIGN NODE 1D ARTERY REFLECTIVE CONDITIONS", "ArtRfCond",
          "Artery reflection condition", DRT::Condition::ArtRfCond, true, DRT::Condition::Point));

  std::vector<Teuchos::RCP<ConditionComponent>> artrfcomponents;
  artrfcomponents.push_back(Teuchos::rcp(new RealVectorConditionComponent("val", 1)));
  artrfcomponents.push_back(Teuchos::rcp(new IntVectorConditionComponent("curve", 1, true, true)));
  for (unsigned i = 0; i < artrfcomponents.size(); ++i) art_rf_bc->AddComponent(artrfcomponents[i]);

  condlist.push_back(art_rf_bc);

  /*--------------------------------------------------------------------*/
  // 1D artery windkessel BC
  Teuchos::RCP<ConditionDefinition> art_wk_bc = Teuchos::rcp(
      new ConditionDefinition("DESIGN NODE 1D ARTERY WINDKESSEL CONDITIONS", "ArtWkCond",
          "Artery windkessel condition", DRT::Condition::ArtWkCond, true, DRT::Condition::Point));

  std::vector<Teuchos::RCP<ConditionComponent>> artwkcomponents;

  art_wk_bc->AddComponent(Teuchos::rcp(new StringConditionComponent("intigrationType",
      "ExplicitWindkessel", Teuchos::tuple<std::string>("ExplicitWindkessel", "ImpedaceWindkessel"),
      Teuchos::tuple<std::string>("ExplicitWindkessel", "ImpedaceWindkessel"), true)));

  art_wk_bc->AddComponent(Teuchos::rcp(new StringConditionComponent("windkesselType", "RCR",
      Teuchos::tuple<std::string>("R", "RC", "RCR", "RCRL", "none"),
      Teuchos::tuple<std::string>("R", "RC", "RCR", "RCRL", "none"), true)));

  artwkcomponents.push_back(Teuchos::rcp(new RealVectorConditionComponent("val", 5)));
  artwkcomponents.push_back(Teuchos::rcp(new IntVectorConditionComponent("curve", 5, true, true)));
  for (unsigned i = 0; i < artwkcomponents.size(); ++i) art_wk_bc->AddComponent(artwkcomponents[i]);

  condlist.push_back(art_wk_bc);

  /*--------------------------------------------------------------------*/
  // 1D artery in/out condition

  Teuchos::RCP<ConditionDefinition> art_in_outlet_bc =
      Teuchos::rcp(new ConditionDefinition("DESIGN NODE 1D ARTERY IN_OUTLET CONDITIONS",
          "ArtInOutCond", "Artery terminal in_outlet condition", DRT::Condition::ArtInOutletCond,
          true, DRT::Condition::Point));

  art_in_outlet_bc->AddComponent(Teuchos::rcp(new StringConditionComponent("terminaltype", "inlet",
      Teuchos::tuple<std::string>("inlet", "outlet"),
      Teuchos::tuple<std::string>("inlet", "outlet"), true)));

  condlist.push_back(art_in_outlet_bc);
  /*--------------------------------------------------------------------*/
  // 1D artery scalar transport condition
  Teuchos::RCP<ConditionDefinition> art_scatra_bc =
      Teuchos::rcp(new ConditionDefinition("DESIGN NODE 1D ARTERY SCATRA PRESCRIBED CONDITIONS",
          "ArtPrescribedScatraCond", "Artery prescribed scatra boundary condition",
          DRT::Condition::ArtPrescribedScatraCond, true, DRT::Condition::Point));

  std::vector<Teuchos::RCP<ConditionComponent>> artscatracomponents;
  artscatracomponents.push_back(Teuchos::rcp(new RealVectorConditionComponent("val", 1)));
  artscatracomponents.push_back(
      Teuchos::rcp(new IntVectorConditionComponent("curve", 1, true, true)));
  for (unsigned i = 0; i < artscatracomponents.size(); ++i)
    art_scatra_bc->AddComponent(artscatracomponents[i]);

  condlist.push_back(art_scatra_bc);

  /*--------------------------------------------------------------------*/
  // 1D artery-to-porofluid coupling BC
  std::vector<Teuchos::RCP<ConditionComponent>> artcoupcomponents;

  Teuchos::RCP<ConditionDefinition> artcoup =
      Teuchos::rcp(new ConditionDefinition("DESIGN NODE 1D ARTERY TO POROFLUID COUPLING CONDITIONS",
          "ArtPorofluidCouplConNodebased", "Artery coupling with porofluid",
          DRT::Condition::ArtPorofluidCouplingCondNodebased, true, DRT::Condition::Point));

  artcoupcomponents.push_back(Teuchos::rcp(new SeparatorConditionComponent("COUPID")));
  artcoupcomponents.push_back(Teuchos::rcp(new IntConditionComponent("coupling id")));

  for (unsigned i = 0; i < artcoupcomponents.size(); ++i)
  {
    artcoup->AddComponent(artcoupcomponents[i]);
  }
  condlist.push_back(artcoup);

  /*--------------------------------------------------------------------*/
  // 1D artery-to-scatra coupling BC
  std::vector<Teuchos::RCP<ConditionComponent>> artscatracoupcomponents;

  Teuchos::RCP<ConditionDefinition> artscatracoup =
      Teuchos::rcp(new ConditionDefinition("DESIGN NODE 1D ARTERY TO SCATRA COUPLING CONDITIONS",
          "ArtScatraCouplConNodebased", "Artery coupling with porofluid",
          DRT::Condition::ArtScatraCouplingCondNodebased, true, DRT::Condition::Point));

  artscatracoupcomponents.push_back(Teuchos::rcp(new SeparatorConditionComponent("COUPID")));
  artscatracoupcomponents.push_back(Teuchos::rcp(new IntConditionComponent("coupling id")));

  for (unsigned i = 0; i < artscatracoupcomponents.size(); ++i)
  {
    artscatracoup->AddComponent(artscatracoupcomponents[i]);
  }

  condlist.push_back(artscatracoup);

  /*--------------------------------------------------------------------*/
  // 1D artery-to-porofluid coupling BC Node-To-Point
  Teuchos::RCP<ConditionDefinition> artcoup_ntp = Teuchos::rcp(
      new ConditionDefinition("DESIGN 1D ARTERY/AIRWAY TO POROFLUID NONCONF COUPLING CONDITIONS",
          "ArtPorofluidCouplConNodeToPoint", "Artery coupling with porofluid nonconf",
          DRT::Condition::ArtPorofluidCouplingCondNodeToPoint, true, DRT::Condition::Point));

  artcoup_ntp->AddComponent(Teuchos::rcp(new StringConditionComponent("coupling_type", "ARTERY",
      Teuchos::tuple<std::string>("ARTERY", "AIRWAY"),
      Teuchos::tuple<std::string>("ARTERY", "AIRWAY"), true)));
  AddNamedInt(artcoup_ntp, "COUPLEDDOF_REDUCED");
  AddNamedInt(artcoup_ntp, "COUPLEDDOF_PORO");
  AddNamedReal(artcoup_ntp, "PENALTY");

  condlist.push_back(artcoup_ntp);

  /*--------------------------------------------------------------------*/
  // 1D artery-to-scatra coupling BC Node-To-Point
  Teuchos::RCP<ConditionDefinition> artscatracoup_ntp = Teuchos::rcp(
      new ConditionDefinition("DESIGN 1D ARTERY/AIRWAY TO SCATRA NONCONF COUPLING CONDITIONS",
          "ArtScatraCouplConNodeToPoint", "Artery coupling with scatra nonconf",
          DRT::Condition::ArtScatraCouplingCondNodeToPoint, true, DRT::Condition::Point));

  artscatracoup_ntp->AddComponent(Teuchos::rcp(new StringConditionComponent("coupling_type",
      "ARTERY", Teuchos::tuple<std::string>("ARTERY", "AIRWAY"),
      Teuchos::tuple<std::string>("ARTERY", "AIRWAY"), true)));
  AddNamedInt(artscatracoup_ntp, "COUPLEDDOF_REDUCED");
  AddNamedInt(artscatracoup_ntp, "COUPLEDDOF_PORO");
  AddNamedReal(artscatracoup_ntp, "PENALTY");

  condlist.push_back(artscatracoup_ntp);
}



void INPAR::BIOFILM::SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list)
{
  Teuchos::ParameterList& biofilmcontrol =
      list->sublist("BIOFILM CONTROL", false, "control parameters for biofilm problems\n");

  DRT::INPUT::BoolParameter(
      "BIOFILMGROWTH", "No", "Scatra algorithm for biofilm growth", &biofilmcontrol);
  DRT::INPUT::BoolParameter("AVGROWTH", "No",
      "The calculation of growth parameters is based on averaged values", &biofilmcontrol);
  DRT::INPUT::DoubleParameter(
      "FLUXCOEF", 0.0, "Coefficient for growth due to scalar flux", &biofilmcontrol);
  DRT::INPUT::DoubleParameter("NORMFORCEPOSCOEF", 0.0,
      "Coefficient for erosion due to traction normal surface forces", &biofilmcontrol);
  DRT::INPUT::DoubleParameter("NORMFORCENEGCOEF", 0.0,
      "Coefficient for erosion due to compression normal surface forces", &biofilmcontrol);
  DRT::INPUT::DoubleParameter("TANGONEFORCECOEF", 0.0,
      "Coefficient for erosion due to the first tangential surface force", &biofilmcontrol);
  DRT::INPUT::DoubleParameter("TANGTWOFORCECOEF", 0.0,
      "Coefficient for erosion due to the second tangential surface force", &biofilmcontrol);
  DRT::INPUT::DoubleParameter(
      "BIOTIMESTEP", 0.05, "Time step size for biofilm growth", &biofilmcontrol);
  DRT::INPUT::IntParameter(
      "BIONUMSTEP", 0, "Maximum number of steps for biofilm growth", &biofilmcontrol);
  DRT::INPUT::BoolParameter(
      "OUTPUT_GMSH", "No", "Do you want to write Gmsh postprocessing files?", &biofilmcontrol);
}



void INPAR::BIOFILM::SetValidConditions(
    std::vector<Teuchos::RCP<DRT::INPUT::ConditionDefinition>>& condlist)
{
  using namespace DRT::INPUT;

  /*--------------------------------------------------------------------*/
  // Additional coupling for biofilm growth

  std::vector<Teuchos::RCP<ConditionComponent>> biogrcomponents;

  biogrcomponents.push_back(Teuchos::rcp(new IntConditionComponent("coupling id")));

  Teuchos::RCP<ConditionDefinition> linebiogr = Teuchos::rcp(
      new ConditionDefinition("DESIGN BIOFILM GROWTH COUPLING LINE CONDITIONS", "BioGrCoupling",
          "BioGrCoupling", DRT::Condition::BioGrCoupling, true, DRT::Condition::Line));
  Teuchos::RCP<ConditionDefinition> surfbiogr = Teuchos::rcp(
      new ConditionDefinition("DESIGN BIOFILM GROWTH COUPLING SURF CONDITIONS", "BioGrCoupling",
          "BioGrCoupling", DRT::Condition::BioGrCoupling, true, DRT::Condition::Surface));

  for (unsigned i = 0; i < biogrcomponents.size(); ++i)
  {
    linebiogr->AddComponent(biogrcomponents[i]);
    surfbiogr->AddComponent(biogrcomponents[i]);
  }

  condlist.push_back(linebiogr);
  condlist.push_back(surfbiogr);
}


void INPAR::REDAIRWAYS::SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list)
{
  using namespace DRT::INPUT;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& redawdyn =
      list->sublist("REDUCED DIMENSIONAL AIRWAYS DYNAMIC", false, "");

  setStringToIntegralParameter<int>("DYNAMICTYP", "OneStepTheta", "OneStepTheta Scheme",
      tuple<std::string>("OneStepTheta"), tuple<int>(one_step_theta), &redawdyn);

  setStringToIntegralParameter<int>("SOLVERTYPE", "Linear", "Solver type",
      tuple<std::string>("Linear", "Nonlinear"), tuple<int>(linear, nonlinear), &redawdyn);

  DoubleParameter("TIMESTEP", 0.01, "Time increment dt", &redawdyn);
  IntParameter("NUMSTEP", 0, "Number of Time Steps", &redawdyn);
  IntParameter("RESTARTEVRY", 1, "Increment for writing restart", &redawdyn);
  IntParameter("RESULTSEVRY", 1, "Increment for writing solution", &redawdyn);
  DoubleParameter("THETA", 1.0, "One-step-theta time integration factor", &redawdyn);

  IntParameter("MAXITERATIONS", 1, "maximum iteration steps", &redawdyn);
  DoubleParameter("TOLERANCE", 1.0E-6, "tolerance", &redawdyn);

  // number of linear solver used for reduced dimensional airways dynamic
  IntParameter("LINEAR_SOLVER", -1,
      "number of linear solver used for reduced dim arterial dynamics", &redawdyn);

  // Solve scatra flag
  setStringToIntegralParameter<int>("SOLVESCATRA", "no",
      "Flag to (de)activate solving scalar transport in blood", tuple<std::string>("no", "yes"),
      tuple<std::string>("do not solve scatra", "solve scatra"), tuple<int>(0, 1), &redawdyn);
  // Compute airway-acinus interdependency flag
  setStringToIntegralParameter<int>("COMPAWACINTER", "no",
      "Flag to (de)activate computation of airway-acinus interdependency ",
      tuple<std::string>("no", "yes"),
      tuple<std::string>("do not compute interdependency", "compute interdependency"),
      tuple<int>(0, 1), &redawdyn);
  // Re-calculate initial acini volume flag
  setStringToIntegralParameter<int>("CALCV0PRESTRESS", "no",
      "Flag to (de)activate initial acini volume adjustment with pre-stress condition ",
      tuple<std::string>("no", "yes"), tuple<std::string>("do not adjust", "adjust volumes"),
      tuple<int>(0, 1), &redawdyn);
  DoubleParameter("TRANSPULMPRESS", 800.0,
      "Transpulmonary pressure needed for recalculation of acini volumes", &redawdyn);
}



void INPAR::REDAIRWAYS::SetValidConditions(
    std::vector<Teuchos::RCP<DRT::INPUT::ConditionDefinition>>& condlist)
{
  using namespace DRT::INPUT;

  /*--------------------------------------------------------------------*/
  // 3-D/reduced-D coupling boundary condition
  Teuchos::RCP<ConditionDefinition> art_red_to_3d_bc =
      Teuchos::rcp(new ConditionDefinition("DESIGN NODE REDUCED D To 3D FLOW COUPLING CONDITIONS",
          "Art_redD_3D_CouplingCond", "Artery reduced D 3D coupling condition",
          DRT::Condition::ArtRedTo3DCouplingCond, true, DRT::Condition::Point));

  art_red_to_3d_bc->AddComponent(Teuchos::rcp(new IntConditionComponent("ConditionID")));

  art_red_to_3d_bc->AddComponent(Teuchos::rcp(new StringConditionComponent("CouplingType", "forced",
      Teuchos::tuple<std::string>("forced", "absorbing"),
      Teuchos::tuple<std::string>("forced", "absorbing"), true)));

  art_red_to_3d_bc->AddComponent(Teuchos::rcp(new StringConditionComponent("ReturnedVariable",
      "pressure", Teuchos::tuple<std::string>("pressure", "flow"),
      Teuchos::tuple<std::string>("pressure", "flow"), true)));
  AddNamedReal(art_red_to_3d_bc, "Tolerance");
  AddNamedInt(art_red_to_3d_bc, "MaximumIterations");

  condlist.push_back(art_red_to_3d_bc);

  /*--------------------------------------------------------------------*/
  // 3-D/reduced-D coupling boundary condition
  Teuchos::RCP<ConditionDefinition> art_3d_to_red_bc =
      Teuchos::rcp(new ConditionDefinition("DESIGN SURF 3D To REDUCED D FLOW COUPLING CONDITIONS",
          "Art_3D_redD_CouplingCond", "Artery 3D reduced D coupling condition",
          DRT::Condition::Art3DToRedCouplingCond, true, DRT::Condition::Surface));

  art_3d_to_red_bc->AddComponent(Teuchos::rcp(new IntConditionComponent("ConditionID")));

  art_3d_to_red_bc->AddComponent(Teuchos::rcp(new StringConditionComponent("ReturnedVariable",
      "flow", Teuchos::tuple<std::string>("pressure", "flow"),
      Teuchos::tuple<std::string>("pressure", "flow"), true)));
  AddNamedReal(art_3d_to_red_bc, "Tolerance");
  AddNamedInt(art_3d_to_red_bc, "MaximumIterations");

  condlist.push_back(art_3d_to_red_bc);

  /*--------------------------------------------------------------------*/
  // Coupling of 3D tissue models and reduced-D airway tree

  std::vector<Teuchos::RCP<ConditionComponent>> redairtiscomponents;

  redairtiscomponents.push_back(Teuchos::rcp(new IntConditionComponent("coupling id")));

  Teuchos::RCP<ConditionDefinition> surfredairtis =
      Teuchos::rcp(new ConditionDefinition("DESIGN SURF TISSUE REDAIRWAY CONDITIONS",
          "SurfaceNeumann", "tissue RedAirway coupling surface condition",
          DRT::Condition::RedAirwayTissue, true, DRT::Condition::Surface));

  for (unsigned i = 0; i < redairtiscomponents.size(); ++i)
  {
    surfredairtis->AddComponent(redairtiscomponents[i]);
  }

  condlist.push_back(surfredairtis);


  /*--------------------------------------------------------------------*/
  // Prescribed BC for reduced dimensional airways

  std::vector<Teuchos::RCP<ConditionComponent>> noderedairtiscomponents;

  noderedairtiscomponents.push_back(Teuchos::rcp(new IntConditionComponent("coupling id")));

  Teuchos::RCP<ConditionDefinition> noderedairtis =
      Teuchos::rcp(new ConditionDefinition("DESIGN NODE TISSUE REDAIRWAY CONDITIONS",
          "RedAirwayPrescribedCond", "tissue RedAirway coupling node condition",
          DRT::Condition::RedAirwayNodeTissue, true, DRT::Condition::Point));


  for (unsigned i = 0; i < noderedairtiscomponents.size(); ++i)
  {
    noderedairtis->AddComponent(noderedairtiscomponents[i]);
  }

  condlist.push_back(noderedairtis);



  /*--------------------------------------------------------------------*/
  // Prescribed BC for reduced dimensional airways

  Teuchos::RCP<ConditionDefinition> raw_in_bc =
      Teuchos::rcp(new ConditionDefinition("DESIGN NODE Reduced D AIRWAYS PRESCRIBED CONDITIONS",
          "RedAirwayPrescribedCond", "Reduced d airway prescribed boundary condition",
          DRT::Condition::RedAirwayPrescribedCond, true, DRT::Condition::Point));

  raw_in_bc->AddComponent(Teuchos::rcp(new StringConditionComponent("boundarycond", "flow",
      Teuchos::tuple<std::string>(
          "flow", "pressure", "switchFlowPressure", "VolumeDependentPleuralPressure"),
      Teuchos::tuple<std::string>(
          "flow", "pressure", "switchFlowPressure", "VolumeDependentPleuralPressure"),
      true)));

  std::vector<Teuchos::RCP<ConditionComponent>> redairwayinletcomponents;
  redairwayinletcomponents.push_back(Teuchos::rcp(new RealVectorConditionComponent("val", 1)));
  redairwayinletcomponents.push_back(
      Teuchos::rcp(new IntVectorConditionComponent("curve", 2, true, true)));
  redairwayinletcomponents.push_back(
      Teuchos::rcp(new IntVectorConditionComponent("funct", 1, false, true, true)));
  for (unsigned i = 0; i < redairwayinletcomponents.size(); ++i)
    raw_in_bc->AddComponent(redairwayinletcomponents[i]);

  condlist.push_back(raw_in_bc);

  /*--------------------------------------------------------------------*/
  // Prescribed BC for reduced dimensional airways switching between different types of boundary
  // conditions

  Teuchos::RCP<ConditionDefinition> raw_in_switch_bc = Teuchos::rcp(new ConditionDefinition(
      "DESIGN NODE Reduced D AIRWAYS SWITCH FLOW PRESSURE CONDITIONS",
      "RedAirwaySwitchFlowPressureCond", "Reduced d airway switch flow pressure boundary condition",
      DRT::Condition::RedAirwayPrescribedSwitchCond, true, DRT::Condition::Point));

  AddNamedInt(raw_in_switch_bc, "FUNCT_ID_FLOW");
  AddNamedInt(raw_in_switch_bc, "FUNCT_ID_PRESSURE");
  AddNamedInt(raw_in_switch_bc, "FUNCT_ID_PRESSURE_ACTIVE");

  condlist.push_back(raw_in_switch_bc);

  /*--------------------------------------------------------------------*/
  // Prescribed BC for reduced dimensional airways external pressure

  Teuchos::RCP<ConditionDefinition> raw_pext_bc = Teuchos::rcp(new ConditionDefinition(
      "DESIGN LINE Reduced D AIRWAYS PRESCRIBED EXTERNAL PRESSURE CONDITIONS",
      "RedAirwayPrescribedExternalPressure",
      "Reduced d airway prescribed external pressure boundary condition",
      DRT::Condition::RedAirwayPrescribedExternalPressure, true, DRT::Condition::Line));

  raw_pext_bc->AddComponent(Teuchos::rcp(new StringConditionComponent("boundarycond",
      "ExternalPressure", Teuchos::tuple<std::string>("ExternalPressure"),
      Teuchos::tuple<std::string>("ExternalPressure"), true)));

  std::vector<Teuchos::RCP<ConditionComponent>> redairwaypextcomponents;
  redairwaypextcomponents.push_back(Teuchos::rcp(new RealVectorConditionComponent("val", 1)));
  redairwaypextcomponents.push_back(
      Teuchos::rcp(new IntVectorConditionComponent("curve", 2, true, true)));
  redairwaypextcomponents.push_back(
      Teuchos::rcp(new IntVectorConditionComponent("funct", 1, false, false, true)));
  for (unsigned i = 0; i < redairwaypextcomponents.size(); ++i)
    raw_pext_bc->AddComponent(redairwaypextcomponents[i]);

  condlist.push_back(raw_pext_bc);


  /*--------------------------------------------------------------------*/
  // Prescribed BC for reduced dimensional scalar transport in airways

  Teuchos::RCP<ConditionDefinition> raw_in_scatra_bc = Teuchos::rcp(
      new ConditionDefinition("DESIGN NODE Reduced D AIRWAYS PRESCRIBED SCATRA CONDITIONS",
          "RedAirwayPrescribedScatraCond", "Reduced d airway prescribed scatra boundary condition",
          DRT::Condition::RedAirwayPrescribedScatraCond, true, DRT::Condition::Point));

  std::vector<Teuchos::RCP<ConditionComponent>> redairwayinletscatracomponents;
  redairwayinletscatracomponents.push_back(
      Teuchos::rcp(new RealVectorConditionComponent("val", 1)));
  redairwayinletscatracomponents.push_back(
      Teuchos::rcp(new IntVectorConditionComponent("curve", 1, true, true)));
  redairwayinletscatracomponents.push_back(
      Teuchos::rcp(new IntVectorConditionComponent("funct", 1, false, false, true)));
  for (unsigned i = 0; i < redairwayinletscatracomponents.size(); ++i)
    raw_in_scatra_bc->AddComponent(redairwayinletscatracomponents[i]);

  condlist.push_back(raw_in_scatra_bc);

  /*--------------------------------------------------------------------*/
  // Prescribed BC for initial values of the scalar transport in reduced dimensional airways

  Teuchos::RCP<ConditionDefinition> raw_int_scatra_bc = Teuchos::rcp(
      new ConditionDefinition("DESIGN LINE Reduced D AIRWAYS INITIAL SCATRA CONDITIONS",
          "RedAirwayInitialScatraCond", "Reduced d airway initial scatra boundary condition",
          DRT::Condition::RedAirwayInitialScatraCond, true, DRT::Condition::Line));

  raw_int_scatra_bc->AddComponent(Teuchos::rcp(new StringConditionComponent("scalar", "O2",
      Teuchos::tuple<std::string>("O2", "CO2"), Teuchos::tuple<std::string>("O2", "CO2"), true)));

  AddNamedReal(raw_int_scatra_bc, "CONCENTRATION");
  condlist.push_back(raw_int_scatra_bc);

  /*--------------------------------------------------------------------*/
  // Reduced D airway Scatra condition for regions of scatra exchange
  Teuchos::RCP<ConditionDefinition> scatra_exchange_cond = Teuchos::rcp(
      new ConditionDefinition("DESIGN LINE Reduced D AIRWAYS SCATRA EXCHANGE CONDITIONS",
          "RedAirwayScatraExchangeCond", "scatra exchange condition",
          DRT::Condition::RedAirwayScatraExchangeCond, true, DRT::Condition::Line));

  scatra_exchange_cond->AddComponent(Teuchos::rcp(new IntConditionComponent("ConditionID")));

  condlist.push_back(scatra_exchange_cond);

  /*--------------------------------------------------------------------*/
  // Reduced D airway Scatra condition for regions with hemoglobin
  Teuchos::RCP<ConditionDefinition> scatra_hemoglobin_cond =
      Teuchos::rcp(new ConditionDefinition("DESIGN LINE Reduced D AIRWAYS HEMOGLOBIN CONDITIONS",
          "RedAirwayScatraHemoglobinCond", "scatra hemoglobin condition",
          DRT::Condition::RedAirwayScatraHemoglobinCond, false, DRT::Condition::Line));

  AddNamedReal(scatra_hemoglobin_cond, "INITIAL_CONCENTRATION");
  condlist.push_back(scatra_hemoglobin_cond);

  /*--------------------------------------------------------------------*/
  // Reduced D airway Scatra condition for regions with hemoglobin
  Teuchos::RCP<ConditionDefinition> scatra_air_cond = Teuchos::rcp(new ConditionDefinition(
      "DESIGN LINE Reduced D AIRWAYS AIR CONDITIONS", "RedAirwayScatraAirCond",
      "scatra air condition", DRT::Condition::RedAirwayScatraAirCond, false, DRT::Condition::Line));

  AddNamedReal(scatra_air_cond, "INITIAL_CONCENTRATION");
  condlist.push_back(scatra_air_cond);

  /*--------------------------------------------------------------------*/
  // Reduced D airway Scatra condition for regions with hemoglobin
  Teuchos::RCP<ConditionDefinition> scatra_capillary_cond =
      Teuchos::rcp(new ConditionDefinition("DESIGN LINE Reduced D AIRWAYS CAPILLARY CONDITIONS",
          "RedAirwayScatraCapillaryCond", "scatra capillary condition",
          DRT::Condition::RedAirwayScatraCapillaryCond, false, DRT::Condition::Line));

  condlist.push_back(scatra_capillary_cond);

  /*--------------------------------------------------------------------*/
  // Prescribed Ventilator BC for reduced dimensional airways

  Teuchos::RCP<ConditionDefinition> raw_vent_bc =
      Teuchos::rcp(new ConditionDefinition("DESIGN NODE Reduced D AIRWAYS VENTILATOR CONDITIONS",
          "RedAirwayVentilatorCond", "Reduced d airway prescribed ventilator condition",
          DRT::Condition::RedAirwayVentilatorCond, true, DRT::Condition::Point));

  raw_vent_bc->AddComponent(Teuchos::rcp(new StringConditionComponent("phase1", "flow",
      Teuchos::tuple<std::string>("flow", "volume", "pressure"),
      Teuchos::tuple<std::string>("flow", "volume", "pressure"), true)));

  raw_vent_bc->AddComponent(Teuchos::rcp(new StringConditionComponent("Phase1Smoothness", "smooth",
      Teuchos::tuple<std::string>("smooth", "discontinous"),
      Teuchos::tuple<std::string>("smooth", "discontinous"), true)));

  raw_vent_bc->AddComponent(Teuchos::rcp(new StringConditionComponent("phase2", "pressure",
      Teuchos::tuple<std::string>("pressure", "flow", "volume"),
      Teuchos::tuple<std::string>("pressure", "flow", "volume"), true)));

  raw_vent_bc->AddComponent(Teuchos::rcp(new StringConditionComponent("Phase2Smoothness", "smooth",
      Teuchos::tuple<std::string>("smooth", "discontinous"),
      Teuchos::tuple<std::string>("smooth", "discontinous"), true)));

  AddNamedReal(raw_vent_bc, "period");
  AddNamedReal(raw_vent_bc, "phase1_period");
  AddNamedReal(raw_vent_bc, "smoothness_period1");
  AddNamedReal(raw_vent_bc, "smoothness_period2");

  std::vector<Teuchos::RCP<ConditionComponent>> redairwayventcomponents;
  redairwayventcomponents.push_back(Teuchos::rcp(new RealVectorConditionComponent("val", 2)));
  redairwayventcomponents.push_back(
      Teuchos::rcp(new IntVectorConditionComponent("curve", 2, true, true)));
  for (unsigned i = 0; i < redairwayventcomponents.size(); ++i)
    raw_vent_bc->AddComponent(redairwayventcomponents[i]);

  condlist.push_back(raw_vent_bc);



  /*--------------------------------------------------------------------*/
  // Prescribed volume dependent pleural pressure for reduced dimensional airways

  Teuchos::RCP<ConditionDefinition> raw_volPpl_bc = Teuchos::rcp(new ConditionDefinition(
      "DESIGN LINE REDUCED D AIRWAYS VOL DEPENDENT PLEURAL PRESSURE CONDITIONS",
      "RedAirwayVolDependentPleuralPressureCond",
      "Reduced D airways volume-dependent peural pressure condition",
      DRT::Condition::RedAirwayVolDependentPleuralPressureCond, true, DRT::Condition::Line));

  raw_volPpl_bc->AddComponent(
      Teuchos::rcp(new StringConditionComponent("TYPE", "Linear_Exponential",
          Teuchos::tuple<std::string>("Linear_Polynomial", "Linear_Exponential", "Linear_Ogden",
              "Nonlinear_Polynomial", "Nonlinear_Exponential", "Nonlinear_Ogden"),
          Teuchos::tuple<std::string>("Linear_Polynomial", "Linear_Exponential", "Linear_Ogden",
              "Nonlinear_Polynomial", "Nonlinear_Exponential", "Nonlinear_Ogden"),
          true)));

  AddNamedReal(raw_volPpl_bc, "TLC");
  AddNamedReal(raw_volPpl_bc, "RV");

  AddNamedReal(raw_volPpl_bc, "P_PLEURAL_0");
  AddNamedReal(raw_volPpl_bc, "P_PLEURAL_LIN");
  AddNamedReal(raw_volPpl_bc, "P_PLEURAL_NONLIN");
  AddNamedReal(raw_volPpl_bc, "TAU");


  std::vector<Teuchos::RCP<ConditionComponent>> raw_volPpl_bc_components;
  raw_volPpl_bc_components.push_back(Teuchos::rcp(new RealVectorConditionComponent("val", 1)));
  raw_volPpl_bc_components.push_back(
      Teuchos::rcp(new IntVectorConditionComponent("curve", 1, true, true)));
  for (unsigned i = 0; i < raw_volPpl_bc_components.size(); ++i)
    raw_volPpl_bc->AddComponent(raw_volPpl_bc_components[i]);

  condlist.push_back(raw_volPpl_bc);

  /*--------------------------------------------------------------------*/
  // Evaluate lung volume condition for reduced dimensional airways

  Teuchos::RCP<ConditionDefinition> raw_eval_lungV_bc = Teuchos::rcp(
      new ConditionDefinition("DESIGN LINE REDUCED D AIRWAYS EVALUATE LUNG VOLUME CONDITIONS",
          "RedAirwayEvalLungVolCond", "Reduced D airways evaluate lung volume condition",
          DRT::Condition::RedAirwayEvalLungVolCond, true, DRT::Condition::Line));


  condlist.push_back(raw_eval_lungV_bc);


  /*--------------------------------------------------------------------*/
  // Impedance condition

  Teuchos::RCP<ConditionDefinition> impedancebc = Teuchos::rcp(new ConditionDefinition(
      "DESIGN SURF IMPEDANCE CONDITIONS", "ImpedanceCond", "Impedance boundary condition",
      DRT::Condition::ImpedanceCond, true, DRT::Condition::Surface));

  impedancebc->AddComponent(Teuchos::rcp(new IntConditionComponent("ConditionID")));

  impedancebc->AddComponent(Teuchos::rcp(new StringConditionComponent("TYPE", "windkessel",
      Teuchos::tuple<std::string>("windkessel", "resistive", "pressure_by_funct"),
      Teuchos::tuple<std::string>("windkessel", "resistive", "pressure_by_funct"), true)));
  AddNamedReal(impedancebc, "R1");
  AddNamedReal(impedancebc, "R2");
  AddNamedReal(impedancebc, "C");
  AddNamedReal(impedancebc, "TIMEPERIOD");
  AddNamedInt(impedancebc, "FUNCT");

  condlist.push_back(impedancebc);
}
