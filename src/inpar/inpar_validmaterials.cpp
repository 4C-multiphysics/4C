/*----------------------------------------------------------------------*/
/*! \file

\brief Setup of the list of valid materials for input

\level 1

*/
/*----------------------------------------------------------------------*/
#include "inpar_validmaterials.H"
#include "lib_materialdefinition.H"
#include "inpar_material.H"
#include "lib_colors.H"

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::INPUT::PrintEmptyMaterialDefinitions(std::ostream& stream,
    std::vector<Teuchos::RCP<DRT::INPUT::MaterialDefinition>>& matlist, bool color)
{
  std::string blue2light = "";
  std::string bluelight = "";
  std::string redlight = "";
  std::string yellowlight = "";
  std::string greenlight = "";
  std::string magentalight = "";
  std::string endcolor = "";

  if (color)
  {
    blue2light = BLUE2_LIGHT;
    bluelight = BLUE_LIGHT;
    redlight = RED_LIGHT;
    yellowlight = YELLOW_LIGHT;
    greenlight = GREEN_LIGHT;
    magentalight = MAGENTA_LIGHT;
    endcolor = END_COLOR;
  }

  const std::string sectionname = "MATERIALS";
  const unsigned l = sectionname.length();
  stream << redlight << "--" << std::string(std::max<int>(65 - l, 0), '-');
  stream << greenlight << sectionname << endcolor << '\n';

  for (auto& i : matlist)
  {
    i->Print(stream, nullptr, color);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void PrintMaterialDatHeader()
{
  Teuchos::RCP<std::vector<Teuchos::RCP<DRT::INPUT::MaterialDefinition>>> matlist =
      DRT::INPUT::ValidMaterials();
  DRT::INPUT::PrintEmptyMaterialDefinitions(std::cout, *matlist);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<std::vector<Teuchos::RCP<DRT::INPUT::MaterialDefinition>>> DRT::INPUT::ValidMaterials()
{
  using Teuchos::tuple;

  // a list containing all valid materials
  Teuchos::RCP<std::vector<Teuchos::RCP<DRT::INPUT::MaterialDefinition>>> vm =
      Teuchos::rcp(new std::vector<Teuchos::RCP<DRT::INPUT::MaterialDefinition>>());

  // convenience
  std::vector<Teuchos::RCP<DRT::INPUT::MaterialDefinition>>& matlist = *vm;


  /*----------------------------------------------------------------------*/
  // Newtonian fluid
  {
    auto m =
        Teuchos::rcp(new MaterialDefinition("MAT_fluid", "Newtonian fluid", INPAR::MAT::m_fluid));

    AddNamedReal(m, "DYNVISCOSITY", "dynamic viscosity");
    AddNamedReal(m, "DENSITY", "spatial mass density");
    AddNamedReal(m, "GAMMA", "surface tension coefficient", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Weakly compressible fluid according to Murnaghan-Tait
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_fluid_murnaghantait",
        "Weakly compressible fluid according to Murnaghan-Tait",
        INPAR::MAT::m_fluid_murnaghantait));

    AddNamedReal(m, "DYNVISCOSITY", "dynamic viscosity");
    AddNamedReal(m, "REFDENSITY", "reference spatial mass density");
    AddNamedReal(m, "REFPRESSURE", "reference pressure");
    AddNamedReal(m, "REFBULKMODULUS", "reference bulk modulus");
    AddNamedReal(m, "MATPARAMETER", "material parameter according to Murnaghan-Tait");
    AddNamedReal(m, "GAMMA", "surface tension coefficient", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Linear law (pressure-dependent) for the density and the viscosity
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_fluid_linear_density_viscosity",
        "Linear law (pressure-dependent) for the density and the viscosity",
        INPAR::MAT::m_fluid_linear_density_viscosity));

    AddNamedReal(m, "REFDENSITY", "reference density");
    AddNamedReal(m, "REFVISCOSITY", "reference viscosity");
    AddNamedReal(m, "REFPRESSURE", "reference pressure");
    AddNamedReal(m, "COEFFDENSITY", "density-pressure coefficient");
    AddNamedReal(m, "COEFFVISCOSITY", "viscosity-pressure coefficient");
    AddNamedReal(m, "GAMMA", "surface tension coefficient", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Weakly compressible fluid
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_fluid_weakly_compressible",
        "Weakly compressible fluid", INPAR::MAT::m_fluid_weakly_compressible));

    AddNamedReal(m, "VISCOSITY", "viscosity");
    AddNamedReal(m, "REFDENSITY", "reference density");
    AddNamedReal(m, "REFPRESSURE", "reference pressure");
    AddNamedReal(m, "COMPRCOEFF", "compressibility coefficient");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // fluid with non-linear viscosity according to Carreau-Yasuda
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_carreauyasuda",
        "fluid with non-linear viscosity according to Carreau-Yasuda",
        INPAR::MAT::m_carreauyasuda));

    AddNamedReal(m, "NU_0", "zero-shear viscosity");
    AddNamedReal(m, "NU_INF", "infinite-shear viscosity");
    AddNamedReal(m, "LAMBDA", "characteristic time");
    AddNamedReal(m, "APARAM", "constant parameter");
    AddNamedReal(m, "BPARAM", "constant parameter");
    AddNamedReal(m, "DENSITY", "density");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // fluid with nonlinear viscosity according to a modified power law
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_modpowerlaw",
        "fluid with nonlinear viscosity according to a modified power law",
        INPAR::MAT::m_modpowerlaw));

    AddNamedReal(m, "MCONS", "consistency");
    AddNamedReal(m, "DELTA", "safety factor");
    AddNamedReal(m, "AEXP", "exponent");
    AddNamedReal(m, "DENSITY", "density");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // fluid with non-linear viscosity according to Herschel-Bulkley
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_herschelbulkley",
        "fluid with non-linear viscosity according to Herschel-Bulkley",
        INPAR::MAT::m_herschelbulkley));

    AddNamedReal(m, "TAU_0", "yield stress");
    AddNamedReal(m, "KFAC", "constant factor");
    AddNamedReal(m, "NEXP", "exponent");
    AddNamedReal(m, "MEXP", "exponent");
    AddNamedReal(m, "LOLIMSHEARRATE", "lower limit of shear rate");
    AddNamedReal(m, "UPLIMSHEARRATE", "upper limit of shear rate");
    AddNamedReal(m, "DENSITY", "density");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // "yoghurt-type" fluid with nonlinear viscosity according to a power law
  // and extended by an Arrhenius-type term to account for temperature dependence
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_yoghurt", "yoghurt-type fluid with nonlinear viscosity", INPAR::MAT::m_yoghurt));

    AddNamedReal(m, "SHC", "specific heat capacity at constant pressure (J/(kg*K))");
    AddNamedReal(m, "DENSITY", "density");
    AddNamedReal(m, "THERMCOND", "thermal conductivity (J/(m*K*s))");
    AddNamedReal(m, "STRAINRATEEXP", "exponent of strain-rate term");
    AddNamedReal(m, "PREEXCON", "pre-exponential constant (1/s)");
    AddNamedReal(m, "ACTENERGY", "activation energy (J/kg)");
    AddNamedReal(m, "GASCON", "specific gas constant R (J/(kg*K))");
    AddNamedReal(m, "DELTA", "safety factor");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // fluid flow in a permeable material
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_permeable", "permeability for flow in porous media", INPAR::MAT::m_permeable_fluid));

    AddNamedString(m, "TYPE", "Problem type: Darcy, Darcy-Stokes (default)", "Darcy-Stokes");
    AddNamedReal(m, "DYNVISCOSITY", "dynamic viscosity");
    AddNamedReal(m, "DENSITY", "density");
    AddNamedReal(m, "PERMEABILITY", "permeability of medium");

    AppendMaterialDefinition(matlist, m);
  }


  /*----------------------------------------------------------------------*/
  // lubrication material
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_lubrication", "lubrication material", INPAR::MAT::m_lubrication));

    AddNamedInt(m, "LUBRICATIONLAWID", "lubrication law id");
    AddNamedReal(m, "DENSITY", "lubricant density");

    AppendMaterialDefinition(matlist, m);
  }


  /*----------------------------------------------------------------------*/
  // constant lubrication material law
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_lubrication_law_constant",
        "constant lubrication material law", INPAR::MAT::m_lubrication_law_constant));

    AddNamedReal(m, "VISCOSITY", "lubricant viscosity");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Barus viscosity lubrication material law
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_lubrication_law_barus",
        "barus lubrication material law", INPAR::MAT::m_lubrication_law_barus));

    AddNamedReal(m, "ABSViscosity", "absolute lubricant viscosity");
    AddNamedReal(m, "PreVisCoeff", "pressure viscosity coefficient");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Roeland viscosity lubrication material law
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_lubrication_law_roeland",
        "roeland lubrication material law", INPAR::MAT::m_lubrication_law_roeland));

    AddNamedReal(m, "ABSViscosity", "absolute lubricant viscosity");
    AddNamedReal(m, "PreVisCoeff", "pressure viscosity coefficient");
    AddNamedReal(m, "RefVisc", "reference viscosity");
    AddNamedReal(m, "RefPress", "reference Pressure");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport material (with potential reaction coefficient)
  {
    auto m = Teuchos::rcp(
        new MaterialDefinition("MAT_scatra", "scalar transport material", INPAR::MAT::m_scatra));

    AddNamedReal(m, "DIFFUSIVITY", "kinematic diffusivity");
    AddNamedReal(m, "REACOEFF", "reaction coefficient", 0.0, true);
    AddNamedReal(m, "SCNUM", "schmidt number", 0.0, true);
    AddNamedReal(m, "DENSIFICATION", "densification coefficient", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }


  /*----------------------------------------------------------------------*/
  // scalar transport material (with potential reaction coefficient)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_scatra_reaction_poro",
        "scalar transport material", INPAR::MAT::m_scatra_reaction_poroECM));

    AddNamedInt(m, "NUMSCAL", "number of scalars for these elements");
    AddNamedIntVector(m, "STOICH", "reaction stoichometrie list", "NUMSCAL");
    AddNamedReal(m, "REACCOEFF", "reaction coefficient");
    AddNamedReal(m, "REACSCALE", "scaling for reaction coefficient");
    // reacscale could now be done by constant distribution function
    AddNamedInt(m, "DISTRFUNCT", "spatial distribution of reaction coefficient", 0, true);
    AddNamedString(m, "COUPLING",
        "type of coupling: "
        "simple_multiplicative, power_multiplicative, constant, michaelis_menten, by_function, "
        "no_coupling (default)",
        "no_coupling", false);
    AddNamedRealVector(m, "ROLE", "role in michaelis-menten like reactions", "NUMSCAL");
    AddNamedRealVector(m, "REACSTART", "starting point of reaction", "NUMSCAL", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }
  /*----------------------------------------------------------------------*/
  // scalar transport reaction material
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_scatra_reaction", "advanced reaction material", INPAR::MAT::m_scatra_reaction));

    AddNamedInt(m, "NUMSCAL", "number of scalars for these elements");
    AddNamedIntVector(m, "STOICH", "reaction stoichometrie list", "NUMSCAL");
    AddNamedReal(m, "REACCOEFF", "reaction coefficient");
    AddNamedInt(m, "DISTRFUNCT", "spatial distribution of reaction coefficient", 0, true);
    AddNamedString(m, "COUPLING",
        "type of coupling: "
        "simple_multiplicative, power_multiplicative, constant, michaelis_menten, by_function, "
        "no_coupling (default)",
        "no_coupling", false);
    AddNamedRealVector(m, "ROLE", "role in michaelis-menten like reactions", "NUMSCAL");
    AddNamedRealVector(m, "REACSTART", "starting point of reaction", "NUMSCAL", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport reaction material (species in fluid)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_scatra_multiporo_fluid",
        "advanced reaction material for multiphase porous flow (species in fluid)",
        INPAR::MAT::m_scatra_multiporo_fluid));

    AddNamedReal(m, "DIFFUSIVITY", "kinematic diffusivity");
    AddNamedInt(m, "PHASEID", "ID of fluid phase the scalar is associated with");
    AddNamedReal(m, "REACOEFF", "reaction coefficient", 0.0, true);
    AddNamedReal(m, "SCNUM", "schmidt number", 0.0, true);
    AddNamedReal(m, "DENSIFICATION", "densification coefficient", 0.0, true);
    AddNamedReal(m, "DELTA", "delta", 0.0, true);
    AddNamedReal(m, "MIN_SAT",
        "minimum saturation under which also corresponding mass fraction is equal to zero", 1.0e-9,
        true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport reaction material (species in volume fraction)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_scatra_multiporo_volfrac",
        "advanced reaction material for multiphase porous flow (species in volfrac)",
        INPAR::MAT::m_scatra_multiporo_volfrac));

    AddNamedReal(m, "DIFFUSIVITY", "kinematic diffusivity");
    AddNamedInt(m, "PHASEID", "ID of fluid phase the scalar is associated with");
    AddNamedReal(m, "REACOEFF", "reaction coefficient", 0.0, true);
    AddNamedReal(m, "SCNUM", "schmidt number", 0.0, true);
    AddNamedReal(m, "DENSIFICATION", "densification coefficient", 0.0, true);
    AddNamedReal(m, "DELTA", "delta", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport reaction material (species in solid)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_scatra_multiporo_solid",
        "advanced reaction material for multiphase "
        "porous flow (species in solid)",
        INPAR::MAT::m_scatra_multiporo_solid));

    AddNamedReal(m, "DIFFUSIVITY", "kinematic diffusivity");
    // no phaseID because only one solid phase
    AddNamedReal(m, "REACOEFF", "reaction coefficient", 0.0, true);
    AddNamedReal(m, "SCNUM", "schmidt number", 0.0, true);
    AddNamedReal(m, "DENSIFICATION", "densification coefficient", 0.0, true);
    AddNamedReal(m, "DELTA", "delta", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport reaction material (temperature)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_scatra_multiporo_temperature",
        "advanced reaction material for multiphase porous flow (temperature)",
        INPAR::MAT::m_scatra_multiporo_temperature));

    AddNamedInt(m, "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE", "number of fluid dofs");
    AddNamedRealVector(
        m, "CP_FLUID", "heat capacity fluid phases", "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE");
    AddNamedInt(m, "NUMVOLFRAC", "number of volfrac dofs");
    AddNamedRealVector(m, "CP_VOLFRAC", "heat capacity volfrac", "NUMVOLFRAC");
    AddNamedReal(m, "CP_SOLID", "heat capacity solid");
    AddNamedRealVector(m, "KAPPA_FLUID", "thermal diffusivity fluid phases",
        "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE");
    AddNamedRealVector(m, "KAPPA_VOLFRAC", "thermal diffusivity volfrac", "NUMVOLFRAC");
    AddNamedReal(m, "KAPPA_SOLID", "heat capacity solid");
    AddNamedReal(m, "DIFFUSIVITY", "kinematic diffusivity", 1.0, true);
    AddNamedReal(m, "REACOEFF", "reaction coefficient", 0.0, true);
    AddNamedReal(m, "SCNUM", "schmidt number", 0.0, true);
    AddNamedReal(m, "DENSIFICATION", "densification coefficient", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport chemotaxis material
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_scatra_chemotaxis", "chemotaxis material", INPAR::MAT::m_scatra_chemotaxis));

    AddNamedInt(m, "NUMSCAL", "number of chemotactic pairs for these elements");
    AddNamedIntVector(m, "PAIR", "chemotaxis pairing", "NUMSCAL");
    AddNamedReal(m, "CHEMOCOEFF", "chemotaxis coefficient");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // anisotropic scalar transport material (with potential reaction coefficient)
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_scatra_aniso", "anisotropic scalar transport material", INPAR::MAT::m_scatra_aniso));

    AddNamedReal(m, "DIFF1", "kinematic diffusivity component 1");
    AddNamedReal(m, "DIFF2", "kinematic diffusivity component 2");
    AddNamedReal(m, "DIFF3", "kinematic diffusivity component 3");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport material for multi-scale approach
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_scatra_multiscale",
        "scalar transport material for multi-scale approach", INPAR::MAT::m_scatra_multiscale));

    AddNamedString(m, "MICROFILE", "input file for micro scale", "filename.dat");
    AddNamedInt(m, "MICRODIS_NUM", "number of micro-scale discretization");
    AddNamedReal(m, "POROSITY", "porosity");
    AddNamedReal(m, "TORTUOSITY", "tortuosity");
    AddNamedReal(m, "A_s", "specific micro-scale surface area");
    AddNamedReal(m, "DIFFUSIVITY", "kinematic diffusivity");
    AddNamedReal(m, "REACOEFF", "reaction coefficient", 0.0, true);
    AddNamedReal(m, "SCNUM", "Schmidt number", 0.0, true);
    AddNamedReal(m, "DENSIFICATION", "densification coefficient", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Weickenmeier muscle material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Muscle_Weickenmeier",
        "Weickenmeier muscle material", INPAR::MAT::m_muscle_weickenmeier));

    AddNamedReal(m, "ALPHA", "experimentally fitted material parameter");
    AddNamedReal(m, "BETA", "experimentally fitted material parameter");
    AddNamedReal(m, "GAMMA", "experimentally fitted material parameter");
    AddNamedReal(m, "KAPPA", "material parameter for coupled volumetric contribution");
    AddNamedReal(m, "OMEGA0", "weighting factor for isotropic tissue constituents");
    AddNamedReal(
        m, "ACTMUNUM", "number of active motor units per undeformed muscle cross-sectional area");
    AddNamedInt(m, "MUTYPESNUM", "number of motor unit types");
    AddNamedRealVector(m, "INTERSTIM", "interstimulus interval", "MUTYPESNUM");
    AddNamedRealVector(m, "FRACACTMU", "fraction of motor unit type", "MUTYPESNUM");
    AddNamedRealVector(m, "FTWITCH", "twitch force of motor unit type", "MUTYPESNUM");
    AddNamedRealVector(m, "TTWITCH", "twitch contraction time of motor unit type", "MUTYPESNUM");
    AddNamedReal(m, "LAMBDAMIN", "minimal active fiber stretch");
    AddNamedReal(
        m, "LAMBDAOPT", "optimal active fiber stretch related to active nominal stress maximum");
    AddNamedReal(m, "DOTLAMBDAMIN", "minimal stretch rate");
    AddNamedReal(m, "KE",
        "parameter controlling the curvature of the velocity dependent activation function in the "
        "eccentric case");
    AddNamedReal(m, "KC",
        "parameter controlling the curvature of the velocity dependent activation function in the "
        "concentric case");
    AddNamedReal(m, "DE",
        "parameter controlling the amplitude of the velocity dependent activation function in the "
        "eccentric case");
    AddNamedReal(m, "DC",
        "parameter controlling the amplitude of the velocity dependent activation function in the "
        "concentric case");
    AddNamedInt(m, "ACTTIMESNUM", "number of time boundaries to prescribe activation");
    AddNamedRealVector(m, "ACTTIMES", "time boundaries between intervals", "ACTTIMESNUM");
    AddNamedInt(m, "ACTINTERVALSNUM", "number of time intervals to prescribe activation");
    AddNamedRealVector(m, "ACTVALUES",
        "scaling factor in intervals (1=full activation, 0=no activation)", "ACTINTERVALSNUM");
    AddNamedReal(m, "DENS", "density");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Active strain Giantesio muscle material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Muscle_Giantesio",
        "Giantesio active strain muscle material", INPAR::MAT::m_muscle_giantesio));

    AddNamedReal(m, "ALPHA", "experimentally fitted material parameter");
    AddNamedReal(m, "BETA", "experimentally fitted material parameter");
    AddNamedReal(m, "GAMMA", "experimentally fitted material parameter");
    AddNamedReal(m, "KAPPA", "material parameter for coupled volumetric contribution");
    AddNamedReal(m, "OMEGA0", "weighting factor for isotropic tissue constituents");
    AddNamedReal(
        m, "ACTMUNUM", "number of active motor units per undeformed muscle cross-sectional area");
    AddNamedInt(m, "MUTYPESNUM", "number of motor unit types");
    AddNamedRealVector(m, "INTERSTIM", "interstimulus interval", "MUTYPESNUM");
    AddNamedRealVector(m, "FRACACTMU", "fraction of motor unit type", "MUTYPESNUM");
    AddNamedRealVector(m, "FTWITCH", "twitch force of motor unit type", "MUTYPESNUM");
    AddNamedRealVector(m, "TTWITCH", "twitch contraction time of motor unit type", "MUTYPESNUM");
    AddNamedReal(m, "LAMBDAMIN", "minimal active fiber stretch");
    AddNamedReal(
        m, "LAMBDAOPT", "optimal active fiber stretch related to active nominal stress maximum");
    AddNamedReal(m, "DOTLAMBDAMIN", "minimal stretch rate");
    AddNamedReal(m, "KE",
        "parameter controlling the curvature of the velocity dependent activation function in the "
        "eccentric case");
    AddNamedReal(m, "KC",
        "parameter controlling the curvature of the velocity dependent activation function in the "
        "concentric case");
    AddNamedReal(m, "DE",
        "parameter controlling the amplitude of the velocity dependent activation function in the "
        "eccentric case");
    AddNamedReal(m, "DC",
        "parameter controlling the amplitude of the velocity dependent activation function in the "
        "concentric case");
    AddNamedInt(m, "ACTTIMESNUM", "number of time boundaries to prescribe activation");
    AddNamedRealVector(m, "ACTTIMES", "time boundaries between intervals", "ACTTIMESNUM");
    AddNamedInt(m, "ACTINTERVALSNUM", "number of time intervals to prescribe activation");
    AddNamedRealVector(m, "ACTVALUES",
        "scaling factor in intervals (1=full activation, 0=no activation)", "ACTINTERVALSNUM");
    AddNamedReal(m, "DENS", "density");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Myocard muscle material (with complicated reaction coefficient)
  {
    auto m = Teuchos::rcp(
        new MaterialDefinition("MAT_myocard", "Myocard muscle material", INPAR::MAT::m_myocard));

    AddNamedReal(m, "DIFF1", "conductivity in fiber direction");
    AddNamedReal(m, "DIFF2", "conductivity perpendicular to fiber direction");
    AddNamedReal(m, "DIFF3", "conductivity perpendicular to fiber direction");
    AddNamedReal(
        m, "PERTUBATION_DERIV", "pertubation for calculation of reaction coefficient derivative");
    AddNamedString(m, "MODEL", "Model type: MV (default), FHN, TNNP, SAN or INADA", "MV");
    AddNamedString(m, "TISSUE", "Tissue type: M (default), ENDO, EPI, AN, N or NH", "M");
    AddNamedReal(m, "TIME_SCALE", "Scale factor for time units of Model");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material according to mixture-fraction approach
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_mixfrac", "material according to mixture-fraction approach", INPAR::MAT::m_mixfrac));

    AddNamedReal(m, "KINVISC", "kinematic viscosity");
    AddNamedReal(m, "KINDIFF", "kinematic diffusivity");
    AddNamedReal(m, "EOSFACA", "equation-of-state factor a");
    AddNamedReal(m, "EOSFACB", "equation-of-state factor b");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material according to Sutherland law
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_sutherland", "material according to Sutherland law", INPAR::MAT::m_sutherland));

    AddNamedReal(m, "REFVISC", "reference dynamic viscosity (kg/(m*s))");
    AddNamedReal(m, "REFTEMP", "reference temperature (K)");
    AddNamedReal(m, "SUTHTEMP", "Sutherland temperature (K)");
    AddNamedReal(m, "SHC", "specific heat capacity at constant pressure (J/(kg*K))");
    AddNamedReal(m, "PRANUM", "Prandtl number");
    AddNamedReal(m, "THERMPRESS", "(initial) thermodynamic pressure (J/m^3)");
    AddNamedReal(m, "GASCON", "specific gas constant R (J/(kg*K))");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material for temperature-dependent water according to VDI Waermeatlas
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_tempdepwater",
        "material for temperature-dependent water", INPAR::MAT::m_tempdepwater));

    AddNamedReal(m, "CRITDENS", "critical density (kg/m^3)");
    AddNamedReal(m, "CRITTEMP", "critical temperature (K)");
    AddNamedReal(m, "SHC", "specific heat capacity at constant pressure (J/(kg*K))");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material according to Sutherland law with Arrhenius-type chemical
  // kinetics (species)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_arrhenius_spec",
        "Arrhenius-type chemical kinetics (species)", INPAR::MAT::m_arrhenius_spec));

    AddNamedReal(m, "REFVISC", "reference dynamic viscosity (kg/(m*s))");
    AddNamedReal(m, "REFTEMP", "reference temperature (K)");
    AddNamedReal(m, "SUTHTEMP", "Sutherland temperature (K)");
    AddNamedReal(m, "SCHNUM", "Schmidt number");
    AddNamedReal(m, "PREEXCON", "pre-exponential constant (1/s)");
    AddNamedReal(m, "TEMPEXP", "exponent of temperature dependence");
    AddNamedReal(m, "ACTEMP", "activation temperature (K)");
    AddNamedReal(m, "GASCON", "specific gas constant R (J/(kg*K))");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material according to Sutherland law with Arrhenius-type chemical
  // kinetics (temperature)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_arrhenius_temp",
        "Arrhenius-type chemical kinetics (temperature)", INPAR::MAT::m_arrhenius_temp));

    AddNamedReal(m, "REFVISC", "reference dynamic viscosity (kg/(m*s))");
    AddNamedReal(m, "REFTEMP", "reference temperature (K)");
    AddNamedReal(m, "SUTHTEMP", "Sutherland temperature (K)");
    AddNamedReal(m, "SHC", "specific heat capacity at constant pressure (J/(kg*K))");
    AddNamedReal(m, "PRANUM", "Prandtl number");
    AddNamedReal(m, "REAHEAT", "heat of reaction per unit mass (J/kg)");
    AddNamedReal(m, "PREEXCON", "pre-exponential constant (1/s)");
    AddNamedReal(m, "TEMPEXP", "exponent of temperature dependence");
    AddNamedReal(m, "ACTEMP", "activation temperature (K)");
    AddNamedReal(m, "GASCON", "specific gas constant R (J/(kg*K))");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material according to Sutherland law with Arrhenius-type chemical
  // kinetics (progress variable)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_arrhenius_pv",
        "material with Arrhenius-type chemical kinetics (progress variable)",
        INPAR::MAT::m_arrhenius_pv));

    AddNamedReal(m, "REFVISC", "reference dynamic viscosity (kg/(m*s))");
    AddNamedReal(m, "REFTEMP", "reference temperature (K)");
    AddNamedReal(m, "SUTHTEMP", "Sutherland temperature (K)");
    AddNamedReal(m, "PRANUM", "Prandtl number");
    AddNamedReal(m, "PREEXCON", "pre-exponential constant (1/s)");
    AddNamedReal(m, "TEMPEXP", "exponent of temperature dependence");
    AddNamedReal(m, "ACTEMP", "activation temperature (K)");
    AddNamedReal(m, "UNBSHC", "specific heat capacity of unburnt phase (J/(kg*K))");
    AddNamedReal(m, "BURSHC", "specific heat capacity of burnt phase (J/(kg*K))");
    AddNamedReal(m, "UNBTEMP", "temperature of unburnt phase (K)");
    AddNamedReal(m, "BURTEMP", "temperature of burnt phase (K)");
    AddNamedReal(m, "UNBDENS", "density of unburnt phase (kg/m^3)");
    AddNamedReal(m, "BURDENS", "density of burnt phase (kg/m^3)");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material according to Sutherland law with simplified chemical
  // kinetics due to Ferziger and Echekki (1993) (original version and
  // modification by Poinsot and Veynante (2005)) (progress variable)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_ferech_pv",
        "material with Ferziger-Echekki (1993) chemical kinetics (progress variable)",
        INPAR::MAT::m_ferech_pv));

    AddNamedReal(m, "REFVISC", "reference dynamic viscosity (kg/(m*s))");
    AddNamedReal(m, "REFTEMP", "reference temperature (K)");
    AddNamedReal(m, "SUTHTEMP", "Sutherland temperature (K)");
    AddNamedReal(m, "PRANUM", "Prandtl number");
    AddNamedReal(m, "REACRATECON", "reaction-rate constant (1/s)");
    AddNamedReal(m, "PVCRIT", "critical value of progress variable");
    AddNamedReal(m, "UNBSHC", "specific heat capacity of unburnt phase (J/(kg*K))");
    AddNamedReal(m, "BURSHC", "specific heat capacity of burnt phase (J/(kg*K))");
    AddNamedReal(m, "UNBTEMP", "temperature of unburnt phase (K)");
    AddNamedReal(m, "BURTEMP", "temperature of burnt phase (K)");
    AddNamedReal(m, "UNBDENS", "density of unburnt phase (kg/m^3)");
    AddNamedReal(m, "BURDENS", "density of burnt phase (kg/m^3)");
    AddNamedReal(m, "MOD", "modification factor (0.0=original, 1.0=modified)");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material parameters for ion species in electrolyte solution (gjb 07/08)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_ion",
        "material parameters for ion species in electrolyte solution", INPAR::MAT::m_ion));

    AddNamedReal(m, "DIFFUSIVITY", "kinematic diffusivity");
    AddNamedReal(m, "VALENCE", "valence (= charge number)");
    AddNamedReal(m, "DENSIFICATION", "densification coefficient", 0.0, true);
    // via these two optional parameters we can bring the material parameters
    // of one eliminated ionic species into BACI if needed
    AddNamedReal(m, "ELIM_DIFFUSIVITY", "kinematic diffusivity of elim. species", 0.0, true);
    AddNamedReal(m, "ELIM_VALENCE", "valence of elim. species", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material parameters for ion species in electrolyte solution (ehrl 07/12)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_newman",
        "material parameters for ion species in electrolyte solution", INPAR::MAT::m_newman));

    AddNamedReal(m, "VALENCE", "valence (= charge number)");
    AddNamedInt(m, "DIFF_COEF_CONC_DEP_FUNCT",
        "function number of function describing concentration dependence of diffusion coefficient",
        0);
    AddNamedInt(m, "DIFF_COEF_TEMP_SCALE_FUNCT",
        "FUNCT number describing temperature scaling of diffusion coefficient", 0);
    AddNamedInt(m, "TRANSNR", "curve number for transference number");
    AddNamedInt(m, "THERMFAC", "curve number for thermodynamic factor");
    AddNamedInt(m, "COND_CONC_DEP_FUNCT",
        "function number of function describing concentration dependence of conductivity", 0);
    AddNamedInt(m, "COND_TEMP_SCALE_FUNCT",
        "FUNCT number describing temperature scaling of conductivity", 0);
    // optional parameter for implemented concentration depending function
    AddNamedInt(m, "DIFF_PARA_NUM", "number of parameters for diffusion coefficient", 0, true);
    AddNamedRealVector(
        m, "DIFF_PARA", "parameters for diffusion coefficient", "DIFF_PARA_NUM", 0.0, true);
    AddNamedInt(m, "DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM",
        "number of parameters for scaling function describing temperature dependence of diffusion "
        "coefficient",
        0, true);
    AddNamedRealVector(m, "DIFF_COEF_TEMP_SCALE_FUNCT_PARA",
        "parameters for function describing temperature dependence of diffusion coefficient",
        "DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM", 0.0, true);
    AddNamedInt(m, "TRANS_PARA_NUM", "number of parameters for transference number", 0, true);
    AddNamedRealVector(
        m, "TRANS_PARA", "parameters for transference number", "TRANS_PARA_NUM", 0.0, true);
    AddNamedInt(m, "THERM_PARA_NUM", "number of parameters for thermodynamic factor", 0, true);
    AddNamedRealVector(
        m, "THERM_PARA", "parameters for thermodynamic factor", "THERM_PARA_NUM", 0.0, true);
    AddNamedInt(m, "COND_PARA_NUM", "number of parameters for conductivity", 0, true);
    AddNamedRealVector(m, "COND_PARA", "parameters for conductivity", "COND_PARA_NUM", 0.0, true);
    AddNamedInt(m, "COND_TEMP_SCALE_FUNCT_PARA_NUM",
        "number of parameters for temperature scaling of conductivity", 0, true);
    AddNamedRealVector(m, "COND_TEMP_SCALE_FUNCT_PARA",
        "parameters for temperature scaling of conductivity", "COND_TEMP_SCALE_FUNCT_PARA_NUM", 0.0,
        true);
    AddNamedSeparator(m, "END", "indicating end of line");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material parameters for ion species in electrolyte solution for multi-scale approach (fang
  // 07/17)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_newman_multiscale",
        "material parameters for ion species in electrolyte solution for multi-scale approach",
        INPAR::MAT::m_newman_multiscale));

    AddNamedReal(m, "VALENCE", "valence (= charge number)");
    AddNamedInt(m, "DIFF_COEF_CONC_DEP_FUNCT",
        "function number of function describing concentration dependence of diffusion coefficient",
        0);
    AddNamedInt(m, "DIFF_COEF_TEMP_SCALE_FUNCT",
        "FUNCT number describing temperature scaling of diffusion coefficient", 0);
    AddNamedInt(m, "TRANSNR", "curve number for transference number");
    AddNamedInt(m, "THERMFAC", "curve number for thermodynamic factor");
    AddNamedInt(m, "COND_CONC_DEP_FUNCT",
        "function number of function describing concentration dependence of conductivity", 0);
    AddNamedInt(m, "COND_TEMP_SCALE_FUNCT",
        "FUNCT number describing temperature scaling of conductivity", 0);
    AddNamedReal(m, "SIGMA", "electronic conductivity");
    AddNamedReal(m, "A_s", "specific micro-scale surface area");
    AddNamedString(m, "MICROFILE", "input file for micro scale", "filename.dat");
    AddNamedInt(m, "MICRODIS_NUM", "number of micro-scale discretization");
    // optional parameters for implemented concentration-depending functions
    AddNamedInt(m, "DIFF_PARA_NUM", "number of parameters for diffusion coefficient", 0, true);
    AddNamedRealVector(
        m, "DIFF_PARA", "parameters for diffusion coefficient", "DIFF_PARA_NUM", 0.0, true);
    AddNamedInt(m, "DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM",
        "number of parameters for scaling function describing temperature dependence of diffusion "
        "coefficient",
        0, true);
    AddNamedRealVector(m, "DIFF_COEF_TEMP_SCALE_FUNCT_PARA",
        "parameters for function describing temperature dependence of diffusion coefficient",
        "DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM", 0.0, true);
    AddNamedInt(m, "TRANS_PARA_NUM", "number of parameters for transference number", 0, true);
    AddNamedRealVector(
        m, "TRANS_PARA", "parameters for transference number", "TRANS_PARA_NUM", 0.0, true);
    AddNamedInt(m, "THERM_PARA_NUM", "number of parameters for thermodynamic factor", 0, true);
    AddNamedRealVector(
        m, "THERM_PARA", "parameters for thermodynamic factor", "THERM_PARA_NUM", 0.0, true);
    AddNamedInt(m, "COND_PARA_NUM", "number of parameters for ionic conductivity", 0, true);
    AddNamedRealVector(
        m, "COND_PARA", "parameters for ionic conductivity", "COND_PARA_NUM", 0.0, true);
    AddNamedInt(m, "COND_TEMP_SCALE_FUNCT_PARA_NUM",
        "number of parameters for temperature scaling of conductivity", 0, true);
    AddNamedRealVector(m, "COND_TEMP_SCALE_FUNCT_PARA",
        "parameters for temperature scaling of conductivity", "COND_TEMP_SCALE_FUNCT_PARA_NUM", 0.0,
        true);
    AddNamedSeparator(m, "END", "indicating end of line");

    AppendMaterialDefinition(matlist, m);
  }
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_scl", "material parameters for space charge layers", INPAR::MAT::m_scl));

    AddNamedReal(m, "VALENCE", "valence/charge number");
    AddNamedInt(m, "DIFF_COEF_CONC_DEP_FUNCT",
        "function number of function describing concentration dependence of diffusion coefficient",
        0);
    AddNamedInt(m, "DIFF_COEF_TEMP_SCALE_FUNCT",
        "function number describing temperature scaling of diffusion coefficient", 0);
    AddNamedInt(m, "TRANSNR", "curve number for transference number");
    AddNamedInt(m, "COND_CONC_DEP_FUNCT",
        "function number of function describing concentration dependence of conductivity", 0);
    AddNamedInt(m, "COND_TEMP_SCALE_FUNCT",
        "function number describing temperature scaling of conductivity", 0);
    AddNamedInt(m, "DIFF_PARA_NUM", "number of parameters for diffusion coefficient", 0, true);
    AddNamedRealVector(
        m, "DIFF_PARA", "parameters for diffusion coefficient", "DIFF_PARA_NUM", 0.0, true);
    AddNamedInt(m, "DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM",
        "number of parameters for scaling function describing temperature dependence of diffusion "
        "coefficient",
        0, true);
    AddNamedRealVector(m, "DIFF_COEF_TEMP_SCALE_FUNCT_PARA",
        "parameters for function describing temperature dependence of diffusion coefficient",
        "DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM", 0.0, true);
    AddNamedInt(m, "TRANS_PARA_NUM", "number of parameters for transference number", 0, true);
    AddNamedRealVector(
        m, "TRANS_PARA", "parameters for transference number", "TRANS_PARA_NUM", 0.0, true);
    AddNamedInt(m, "COND_PARA_NUM", "number of parameters for conductivity", 0, true);
    AddNamedRealVector(m, "COND_PARA", "parameters for conductivity", "COND_PARA_NUM", 0.0, true);
    AddNamedInt(m, "COND_TEMP_SCALE_FUNCT_PARA_NUM",
        "number of parameters for temperature scaling of conductivity", 0, true);
    AddNamedRealVector(m, "COND_TEMP_SCALE_FUNCT_PARA",
        "parameters for temperature scaling of conductivity", "COND_TEMP_SCALE_FUNCT_PARA_NUM", 0.0,
        true);
    AddNamedReal(m, "MAX_CONC", "maximum cation concentration", 1.0);
    AddNamedInt(m, "EXTRAPOL_DIFF",
        "strategy for extrapolation of diffusion coefficient below 0 and above MAX_CONC (-1: "
        "disabled, 0: constant)",
        0);
    AddNamedReal(m, "LIM_CONC", "limiting concentration for extrapolation", 1.0, true);
    AddNamedReal(m, "BULK_CONC", "bulk ion concentration", 1.0);
    AddNamedReal(m, "SUSCEPT", "susceptibility", 1.0);
    AddNamedReal(m, "DELTA_NU", "difference of partial molar volumes (vacancy & cation)", 0.0);
    AddNamedSeparator(m, "END", "indicating end of line");

    AppendMaterialDefinition(matlist, m);
  }


  /*----------------------------------------------------------------------*/
  // electrode material (fang 02/15)
  {
    auto matelectrode = Teuchos::rcp(
        new MaterialDefinition("MAT_electrode", "electrode material", INPAR::MAT::m_electrode));

    // diffusivity and electronic conductivity
    AddNamedInt(matelectrode, "DIFF_COEF_CONC_DEP_FUNCT",
        "function number of function describing concentration dependence of diffusion coefficient",
        0);
    AddNamedInt(matelectrode, "DIFF_COEF_TEMP_SCALE_FUNCT",
        "FUNCT number describing temperature scaling of diffusion coefficient", 0);
    AddNamedInt(matelectrode, "COND_CONC_DEP_FUNCT",
        "function number of function describing concentration dependence of conductivity", 0);
    AddNamedInt(matelectrode, "COND_TEMP_SCALE_FUNCT",
        "FUNCT number describing temperature scaling of conductivity", 0);

    // optional parameters for concentration dependency of diffusivity and electronic conductivity
    AddNamedInt(
        matelectrode, "DIFF_PARA_NUM", "number of parameters for diffusion coefficient", 0, true);
    AddNamedRealVector(matelectrode, "DIFF_PARA", "parameters for diffusion coefficient",
        "DIFF_PARA_NUM", 0.0, true);
    AddNamedInt(matelectrode, "DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM",
        "number of parameters for scaling function describing temperature dependence of diffusion "
        "coefficient",
        0, true);
    AddNamedRealVector(matelectrode, "DIFF_COEF_TEMP_SCALE_FUNCT_PARA",
        "parameters for function describing temperature dependence of diffusion coefficient",
        "DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM", 0.0, true);
    AddNamedInt(
        matelectrode, "COND_PARA_NUM", "number of parameters for electronic conductivity", 0, true);
    AddNamedRealVector(matelectrode, "COND_PARA", "parameters for electronic conductivity",
        "COND_PARA_NUM", 0.0, true);
    AddNamedInt(matelectrode, "COND_TEMP_SCALE_FUNCT_PARA_NUM",
        "number of parameters for temperature scaling of conductivity", 0, true);
    AddNamedRealVector(matelectrode, "COND_TEMP_SCALE_FUNCT_PARA",
        "parameters for temperature scaling of conductivity", "COND_TEMP_SCALE_FUNCT_PARA_NUM", 0.0,
        true);
    // saturation value of intercalated Lithium concentration
    AddNamedReal(matelectrode, "C_MAX", "saturation value of intercalated Lithium concentration");

    // lithiation value corresponding to saturation value of intercalated Lithium concentration
    AddNamedReal(matelectrode, "CHI_MAX",
        "lithiation value corresponding to saturation value of intercalated Lithium concentration "
        "'C_MAX'");

    // model for half cell open circuit potential of electrode
    AddNamedString(matelectrode, "OCP_MODEL",
        "model for half cell open circuit potential of electrode: "
        "Redlich-Kister, Taralov, Polynomial, csv",
        "none");

    // lower bound of range of validity as a fraction of C_MAX for ocp calculation model
    AddNamedReal(matelectrode, "X_MIN",
        "lower bound of range of validity as a fraction of C_MAX for ocp calculation model", 2.0,
        false);

    // upper bound of range of validity as a fraction of C_MAX for ocp calculation model
    AddNamedReal(matelectrode, "X_MAX",
        "upper bound of range of validity as a fraction of C_MAX for ocp calculation model", 2.0,
        false);

    // number of parameters underlying half cell open circuit potential model
    AddNamedInt(matelectrode, "OCP_PARA_NUM",
        "number of parameters underlying half cell open circuit potential model", 0, true);

    // parameters underlying half cell open circuit potential model
    AddNamedRealVector(matelectrode, "OCP_PARA",
        "parameters underlying half cell open circuit potential model", "OCP_PARA_NUM", 0., true);

    // *.csv file with data points for half cell open circuit potential
    AddNamedString(matelectrode, "OCP_CSV",
        "\\*.csv file with data points for half cell open circuit potential", "", true);

    // end of input line
    AddNamedSeparator(matelectrode, "END", "indicating end of line");

    // add electrode material to global list of valid materials
    AppendMaterialDefinition(matlist, matelectrode);
  }

  /*----------------------------------------------------------------------*/
  // material collection (gjb 07/08)
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_matlist", "list/collection of materials, i.e. material IDs", INPAR::MAT::m_matlist));

    AddNamedBool(m, "LOCAL", "individual materials allocated per element or only at global scope");
    // AddNamedInt(m,"LOCAL","individual materials allocated per element or only at global scope");
    AddNamedInt(m, "NUMMAT", "number of materials in list");
    AddNamedIntVector(m, "MATIDS", "the list material IDs", "NUMMAT");
    AddNamedSeparator(m, "END", "indicating end of line");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material collection with reactions (thon 09/14)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_matlist_reactions",
        "list/collection of materials, i.e. material IDs and list of reactions",
        INPAR::MAT::m_matlist_reactions));

    AddNamedBool(m, "LOCAL", "individual materials allocated per element or only at global scope");
    AddNamedInt(m, "NUMMAT", "number of materials in list");
    AddNamedIntVector(m, "MATIDS", "the list material IDs", "NUMMAT");
    AddNamedInt(m, "NUMREAC", "number of reactions for these elements", 0);
    AddNamedIntVector(m, "REACIDS", "advanced reaction list", "NUMREAC", 0);
    AddNamedSeparator(m, "END", "indicating end of line");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material collection with chemotaxis (thon 06/15)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_matlist_chemotaxis",
        "list/collection of materials, i.e. material IDs and list of chemotactic pairs",
        INPAR::MAT::m_matlist_chemotaxis));

    AddNamedBool(m, "LOCAL", "individual materials allocated per element or only at global scope");
    AddNamedInt(m, "NUMMAT", "number of materials in list");
    AddNamedIntVector(m, "MATIDS", "the list material IDs", "NUMMAT");
    AddNamedInt(m, "NUMPAIR", "number of pairs for these elements", 0);
    AddNamedIntVector(m, "PAIRIDS", "chemotaxis pairs list", "NUMPAIR", 0);
    AddNamedSeparator(m, "END", "indicating end of line");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material collection with reactions AND chemotaxis (thon 06/15)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_matlist_chemo_reac",
        "list/collection of materials, i.e. material IDs and list of reactive/chemotactic pairs",
        INPAR::MAT::m_matlist_chemoreac));

    AddNamedBool(m, "LOCAL", "individual materials allocated per element or only at global scope");
    AddNamedInt(m, "NUMMAT", "number of materials in list");
    AddNamedIntVector(m, "MATIDS", "the list material IDs", "NUMMAT");
    AddNamedInt(m, "NUMPAIR", "number of pairs for these elements", 0);
    AddNamedIntVector(m, "PAIRIDS", "chemotaxis pairs list", "NUMPAIR", 0);
    AddNamedInt(m, "NUMREAC", "number of reactions for these elements", 0);
    AddNamedIntVector(m, "REACIDS", "advanced reaction list", "NUMREAC", 0);
    AddNamedSeparator(m, "END", "indicating end of line");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material collection (ehrl 11/12)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_elchmat",
        "specific list/collection of species and phases for elch applications",
        INPAR::MAT::m_elchmat));

    AddNamedBool(m, "LOCAL", "individual materials allocated per element or only at global scope",
        false, true);
    AddNamedInt(m, "NUMDOF", "number of dof's per node");
    AddNamedInt(m, "NUMSCAL", "number of transported scalars per node");
    AddNamedInt(m, "NUMPHASE", "number of phases in electrolyte");
    AddNamedIntVector(m, "PHASEIDS", "the list phasel IDs", "NUMPHASE");
    AddNamedSeparator(m, "END", "indicating end of line");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material collection (ehrl 11/12)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_elchphase",
        "material parameters for ion species in electrolyte solution", INPAR::MAT::m_elchphase));

    AddNamedBool(m, "LOCAL", "individual materials allocated per element or only at global scope",
        false, true);
    AddNamedReal(m, "EPSILON", "phase porosity");
    AddNamedReal(m, "TORTUOSITY", "inverse (!) of phase tortuosity");
    AddNamedInt(m, "NUMMAT", "number of materials in electrolyte");
    AddNamedIntVector(m, "MATIDS", "the list phasel IDs", "NUMMAT");
    AddNamedSeparator(m, "END", "indicating end of line");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // St.Venant--Kirchhoff
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_Struct_StVenantKirchhoff", "St.Venant--Kirchhoff material", INPAR::MAT::m_stvenant));

    AddNamedReal(m, "YOUNG", "Young's modulus");
    AddNamedReal(m, "NUE", "Poisson's ratio");
    AddNamedReal(m, "DENS", "mass density");
    AddNamedReal(m, "THEXPANS", "coefficient of linear thermal expansion", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // St.Venant--Kirchhoff with temperature
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Struct_ThrStVenantK",
        "Thermo St.Venant--Kirchhoff material", INPAR::MAT::m_thermostvenant));

    AddNamedInt(m, "YOUNGNUM",
        "number of Young's modulus in list (if 1 Young is const, if >1 Young is temperature) "
        "dependent");
    AddNamedRealVector(m, "YOUNG", "Young's modulus", "YOUNGNUM");
    AddNamedReal(m, "NUE", "Poisson's ratio");
    AddNamedReal(m, "DENS", "mass density");
    AddNamedReal(m, "THEXPANS", "constant coefficient of linear thermal expansion");
    AddNamedReal(m, "CAPA", "capacity");
    AddNamedReal(m, "CONDUCT", "conductivity");
    AddNamedReal(m, "INITTEMP", "initial temperature");
    AddNamedInt(m, "THERMOMAT", "mat id of thermal material part", -1, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Linear thermo-elastic St.Venant Kirchhoff / plastic von Mises
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Struct_ThrPlasticLinElast",
        "Thermo-elastic St.Venant Kirchhoff / plastic von Mises material",
        INPAR::MAT::m_thermopllinelast));

    AddNamedReal(m, "YOUNG", "Young's modulus");
    AddNamedReal(m, "NUE", "Poisson's ratio");
    AddNamedReal(m, "DENS", "mass density");
    AddNamedReal(m, "THEXPANS", "coefficient of linear thermal expansion");
    AddNamedReal(m, "INITTEMP", "initial temperature");
    AddNamedReal(m, "YIELD", "yield stress");
    AddNamedReal(m, "ISOHARD", "isotropic hardening modulus");
    AddNamedReal(m, "KINHARD", "kinematic hardening modulus");
    AddNamedInt(m, "SAMPLENUM", "number of stress-strain pairs in list");
    AddNamedRealVector(m, "SIGMA_Y", "yield stress", "SAMPLENUM");
    AddNamedRealVector(
        m, "EPSBAR_P", "accumulated plastic strain corresponding to SIGMA_Y", "SAMPLENUM");
    AddNamedReal(m, "TOL", "tolerance for local Newton iteration");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Finite strain superelasticity of shape memory alloys
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Struct_SuperElastSMA",
        "finite strain superelastic shape memory alloy", INPAR::MAT::m_superelast));

    AddNamedReal(m, "DENS", "mass density");
    AddNamedReal(m, "YOUNG", "Young's modulus");
    AddNamedReal(m, "NUE", "Poisson's ratio");
    AddNamedReal(m, "EPSILON_L",
        "parameter representing the maximum deformation obtainable only by detwinning of the "
        "multiple-variant martensite");
    AddNamedReal(m, "T_AS_s",
        "Temperature at which the phase transformation from austenite to martensite starts");
    AddNamedReal(m, "T_AS_f",
        "Temperature at which the phase transformation from austenite to martensite finishes");
    AddNamedReal(m, "T_SA_s",
        "Temperature at which the phase transformation from martensite to autenite starts");
    AddNamedReal(m, "T_SA_f",
        "Temperature at which the phase transformation from martensite to autenite finishes");
    AddNamedReal(m, "C_AS", "Coefficient of the linear temperature dependence of T_AS");
    AddNamedReal(m, "C_SA", "Coefficient of the linear temperature dependence of T_SA");
    AddNamedReal(m, "SIGMA_AS_s",
        "stress at which the phase transformation from austenite to martensite begins");
    AddNamedReal(m, "SIGMA_AS_f",
        "stress at which the phase transformation from austenite to martensite finishes");
    AddNamedReal(m, "SIGMA_SA_s",
        "stress at which the phase transformation from martensite to austenite begins");
    AddNamedReal(m, "SIGMA_SA_f",
        "stress at which the phase transformation from martensite to austenite finishes");
    AddNamedReal(m, "ALPHA", "pressure dependency in the drucker-prager-type loading");
    AddNamedInt(m, "MODEL",
        "Model used for the evolution of martensitic fraction (1=exponential; 2=linear)");
    AddNamedReal(m, "BETA_AS",
        "parameter, measuring the speed of the transformation from austenite to martensite", 0.,
        true);
    AddNamedReal(m, "BETA_SA",
        "parameter, measuring the speed of the transformation from martensite to austenite", 0.,
        true);


    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Thermo-hyperelasticity / finite strain von-Mises plasticity
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Struct_ThrPlasticHyperElast",
        "Thermo-hyperelastic / finite strain plastic von Mises material",
        INPAR::MAT::m_thermoplhyperelast));

    AddNamedReal(m, "YOUNG", "Young's modulus");
    AddNamedReal(m, "NUE", "Poisson's ratio");
    AddNamedReal(m, "DENS", "mass density");
    AddNamedReal(m, "CTE", "coefficient of thermal expansion", 0., true);
    AddNamedReal(m, "INITTEMP", "initial, reference temperature", 0., true);
    AddNamedReal(m, "YIELD", "initial yield stress");
    AddNamedReal(m, "ISOHARD", "isotropic hardening modulus", 0., true);
    AddNamedReal(m, "SATHARDENING", "saturation hardening", 0., true);
    AddNamedReal(m, "HARDEXPO", "hardening exponent", 0., true);
    AddNamedReal(m, "YIELDSOFT", "yield stress softening", 0., true);
    AddNamedReal(m, "HARDSOFT", "hardening softening", 0., true);
    AddNamedReal(m, "TOL", "tolerance for local Newton iteration", 1.e-8, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Hyperelasticity / finite strain von-Mises plasticity
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Struct_PlasticNlnLogNeoHooke",
        "hyperelastic / finite strain plastic von Mises material", INPAR::MAT::m_plnlnlogneohooke));

    AddNamedReal(m, "YOUNG", "Young's modulus");
    AddNamedReal(m, "NUE", "Poisson's ratio");
    AddNamedReal(m, "DENS", "mass density");
    AddNamedReal(m, "YIELD", "yield stress");
    AddNamedReal(m, "ISOHARD", "isotropic hardening modulus");
    AddNamedReal(m, "SATHARDENING", "saturation hardening");
    AddNamedReal(m, "HARDEXPO", "hardening exponent");
    AddNamedReal(m, "VISC", "VISCOSITY", 0., true);
    AddNamedReal(m, "RATE_DEPENDENCY", "rate dependency", 0., true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Plastic linear elastic St.Venant Kirchhoff / von Mises
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Struct_PlasticLinElast",
        "elastic St.Venant Kirchhoff / plastic von Mises material", INPAR::MAT::m_pllinelast));

    AddNamedReal(m, "YOUNG", "Young's modulus");
    AddNamedReal(m, "NUE", "Poisson's ratio");
    AddNamedReal(m, "DENS", "mass density");
    AddNamedReal(m, "YIELD", "yield stress");
    AddNamedReal(m, "ISOHARD", "isotropic hardening modulus");
    AddNamedReal(m, "KINHARD", "kinematic hardening modulus");
    AddNamedReal(m, "TOL", "tolerance for local Newton iteration");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Elastic visco-plastic finite strain material law without yield surface
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Struct_Viscoplastic_No_Yield_Surface",
        "Elastic visco-plastic finite strain material law without yield surface",
        INPAR::MAT::m_vp_no_yield_surface));

    // elasticity parameters
    AddNamedReal(m, "YOUNG", "Young's modulus");
    AddNamedReal(m, "NUE", "Poisson's ratio");
    AddNamedReal(m, "DENS", "material mass density");
    // visco-plasticity parameters
    AddNamedReal(m, "TEMPERATURE", "temperature in Kelvin");
    AddNamedReal(m, "PRE_EXP_FAC", "pre-exponential factor of plastic shear strain rate 'A'");
    AddNamedReal(m, "ACTIVATION_ENERGY", "activation energy 'Q'");
    AddNamedReal(m, "GAS_CONSTANT", "gas constant 'R'");
    AddNamedReal(m, "STRAIN_RATE_SENS", "strain-rate-sensitivity 'm'");
    AddNamedReal(m, "INIT_FLOW_RES", "initial isotropic flow resistance 'S^0'");
    AddNamedReal(m, "FLOW_RES_PRE_FAC", "flow resistance factor 'H_0'");
    AddNamedReal(m, "FLOW_RES_EXP", "flow resistance exponential value 'a'");
    AddNamedReal(m, "FLOW_RES_SAT_FAC", "flow resistance saturation factor 'S_*'");
    AddNamedReal(m, "FLOW_RES_SAT_EXP", "flow resistance saturation exponent 'b'");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Robinson's visco-plastic material
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_Struct_Robinson", "Robinson's visco-plastic material", INPAR::MAT::m_vp_robinson));

    AddNamedString(m, "KIND",
        "kind of Robinson material: "
        "Butler, Arya, Arya_NarloyZ (default), Arya_CrMoSteel",
        "Arya_NarloyZ");
    AddNamedInt(m, "YOUNGNUM", "number of Young's modulus in list");
    AddNamedRealVector(m, "YOUNG", "Young's modulus", "YOUNGNUM");
    AddNamedReal(m, "NUE", "Poisson's ratio");
    AddNamedReal(m, "DENS", "mass density");
    AddNamedReal(m, "THEXPANS", "coefficient of linear thermal expansion");
    AddNamedReal(m, "INITTEMP", "initial temperature");
    AddNamedReal(m, "HRDN_FACT", "hardening factor 'A'");
    AddNamedReal(m, "HRDN_EXPO", "hardening power 'n'");
    AddNamedInt(m, "SHRTHRSHLDNUM", "number of shear stress threshold 'K^2'in list");
    AddNamedRealVector(
        m, "SHRTHRSHLD", "Bingam-Prager shear stress threshold 'K^2'", "SHRTHRSHLDNUM");
    AddNamedReal(m, "RCVRY", "recovery factor 'R_0'");
    AddNamedReal(m, "ACTV_ERGY", "activation energy 'Q_0'");
    AddNamedReal(m, "ACTV_TMPR", "activation temperature 'T_0'");
    AddNamedReal(m, "G0", "'G_0'");
    AddNamedReal(m, "M_EXPO", "'m'");
    AddNamedInt(m, "BETANUM", "number of 'beta' in list");
    AddNamedRealVector(m, "BETA", "beta", "BETANUM");
    AddNamedReal(m, "H_FACT", "'H'");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Elasto-plastic material with damage, based on MAT_Struct_PlasticLinElast
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Struct_Damage",
        "elasto-plastic von Mises material with ductile damage", INPAR::MAT::m_elpldamage));

    AddNamedReal(m, "YOUNG", "Young's modulus");
    AddNamedReal(m, "NUE", "Poisson's ratio");
    AddNamedReal(m, "DENS", "mass density");
    AddNamedInt(m, "SAMPLENUM", "number of stress-strain pairs in list");
    AddNamedRealVector(m, "SIGMA_Y", "yield stress", "SAMPLENUM");
    AddNamedRealVector(
        m, "EPSBAR_P", "accumulated plastic strain corresponding to SIGMA_Y", "SAMPLENUM");
    AddNamedReal(m, "DAMDEN", "denominator of damage evoluation law");
    AddNamedReal(m, "DAMEXP", "exponent of damage evoluation law");
    AddNamedReal(m, "DAMTHRESHOLD", "damage threshold");
    AddNamedReal(m, "KINHARD", "kinematic hardening modulus, stress-like variable");
    AddNamedReal(m, "KINHARD_REC", "recovery factor, scalar-valued variable");
    AddNamedReal(m, "SATHARDENING", "saturation hardening");
    AddNamedReal(m, "HARDEXPO", "hardening exponent");
    AddNamedReal(m, "TOL", "tolerance for local Newton iteration");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // aneurysm wall material according to Raghavan and Vorp [2000]
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Struct_AAANeoHooke",
        "aneurysm wall material according to Raghavan and Vorp [2000]", INPAR::MAT::m_aaaneohooke));

    AddNamedReal(m, "YOUNG", "Young's modulus");
    AddNamedReal(m, "BETA", "2nd parameter");
    AddNamedReal(m, "NUE", "Poisson's ratio");
    AddNamedReal(m, "DENS", "mass density");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // aneurysm wall material according to Raghavan and Vorp [2000]
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Struct_AAANeoHookeStopro",
        "aneurysm wall material according to Raghavan and Vorp [2000] with stochastic "
        "modelling of beta",
        INPAR::MAT::m_aaaneohooke_stopro));

    AddNamedReal(m, "YOUNG", "Young's modulus");
    AddNamedReal(m, "BETA", "2nd parameter");
    AddNamedReal(m, "NUE", "Poisson's ratio");
    AddNamedReal(m, "DENS", "mass density");
    // Stochastic properties are set via randomfield class

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // AAA thrombus material according to GASSER et. al. [2008]
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Struct_AAAGasser",
        "AAA thrombus material according to GASSER [2008]", INPAR::MAT::m_aaagasser));

    AddNamedReal(m, "DENS", "mass density");
    AddNamedString(
        m, "VOL", "Type of volumetric Strain Energy Density: OSM (default),SuBa,SiTa", "OSM");
    AddNamedReal(m, "NUE", "Poisson's ratio (0.49)");
    AddNamedReal(m, "BETA", "empiric constant for OSM (-2.0)");
    AddNamedReal(m, "CLUM", "luminal stiffness parameter (2.62e3)");
    AddNamedReal(m, "CMED", "medial stiffness parameter (2.62e3)");
    AddNamedReal(m, "CABLUM", "abluminal stiffness parameter (2.62e3)");

    /*
     AddNamedReal(m,"DENS","mass density");
     AddNamedReal(m,"KAPPA","dilatation modulus");
     AddNamedReal(m,"BETA","empiric constant");
     AddNamedReal(m,"CLUM","luminal stiffness parameter");
     AddNamedReal(m,"CMED","medial stiffness parameter");
     AddNamedReal(m,"CABLUM","abluminal stiffness parameter");
     */

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // aneurysm wall material according to Raghavan and Vorp [2000] with damage Simo
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Raghavan_Damage",
        "aneurysm wall material according to Raghavan and Vorp [2000] with damage",
        INPAR::MAT::m_aaaraghavanvorp_damage));

    AddNamedReal(m, "BULK", "Bulk's modulus");
    AddNamedReal(m, "ALPHA", "1nd parameter,alpha");
    AddNamedReal(m, "BETA", "2nd parameter,beta");
    AddNamedReal(m, "EQSTRMIN", "equivalent strain initial damage");
    AddNamedReal(m, "A", "1st parameter, a");
    AddNamedReal(m, "B", "2nd parameter, b");
    AddNamedReal(m, "DENS", "mass density");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // aneurysm wall material SEF according  to Raghavan and Vorp [2000],
  // parameters according to mixed effects model
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Struct_AAA_MixedEffects",
        "aneurysm wall material according to Mixed Effects Model", INPAR::MAT::m_aaa_mixedeffects));

    AddNamedReal(m, "AGE", "age");
    AddNamedReal(m, "REFDIA", "subrenal diameter");
    AddNamedReal(m, "NUE", "Poisson's ratio");
    AddNamedReal(m, "DENS", "mass density");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Visco-elastic Neo-Hookean material law
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_VISCONEOHOOKE",
        "visco-elastic neo-Hookean material law", INPAR::MAT::m_visconeohooke));
    AddNamedReal(m, "YOUNGS_SLOW", "???");
    AddNamedReal(m, "POISSON", "???");
    AddNamedReal(m, "DENS", "???");
    AddNamedReal(m, "YOUNGS_FAST", "???");
    AddNamedReal(m, "RELAX", "???");
    AddNamedReal(m, "THETA", "???");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Visco-elastic anisotropic fiber material law
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_VISCOANISO",
        "visco-elastic anisotropic fibre material law", INPAR::MAT::m_viscoanisotropic));

    AddNamedReal(m, "KAPPA", "dilatation modulus");
    AddNamedReal(m, "MUE", "Shear Modulus");
    AddNamedReal(m, "DENS", "Density");
    AddNamedReal(m, "K1", "Parameter for linear fiber stiffness");
    AddNamedReal(m, "K2", "Parameter for exponetial fiber stiffness");
    AddNamedReal(m, "GAMMA", "angle between fibers");
    AddNamedReal(m, "BETA_ISO", "ratio between elasticities in generalized Maxweel body");
    AddNamedReal(m, "BETA_ANISO", "ratio between elasticities in generalized Maxweel body");
    AddNamedReal(m, "RELAX_ISO", "isotropic relaxation time");
    AddNamedReal(m, "RELAX_ANISO", "anisotropic relaxation time");
    AddNamedReal(m, "MINSTRETCH", "minimal principal stretch fibers do respond to");
    AddNamedInt(m, "ELETHICKDIR", "Element thickness direction applies also to fibers (only sosh)");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Structural micro-scale approach: material parameters are calculated from microscale simulation
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Struct_Multiscale",
        "Structural micro-scale approach: material parameters are calculated from microscale "
        "simulation",
        INPAR::MAT::m_struct_multiscale));

    AddNamedString(m, "MICROFILE", "inputfile for microstructure", "filename.dat");
    AddNamedInt(m, "MICRODIS_NUM", "Number of microscale discretization");
    AddNamedReal(m, "INITVOL", "Initial volume of RVE", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // collection of hyperelastic materials
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_ElastHyper",
        "list/collection of hyperelastic materials, i.e. material IDs", INPAR::MAT::m_elasthyper));

    AddNamedInt(m, "NUMMAT", "number of materials/potentials in list");
    AddNamedIntVector(m, "MATIDS", "the list material/potential IDs", "NUMMAT");
    AddNamedReal(m, "DENS", "material mass density");
    AddNamedInt(m, "POLYCONVEX", "1.0 if polyconvexity of system is checked", 0., true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // viscohyperelastic material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_ViscoElastHyper",
        "Viscohyperelastic material compatible with the collection of hyperelastic materials",
        INPAR::MAT::m_viscoelasthyper));

    AddNamedInt(m, "NUMMAT", "number of materials/potentials in list");
    AddNamedIntVector(m, "MATIDS", "the list material/potential IDs", "NUMMAT");
    AddNamedReal(m, "DENS", "material mass density");
    AddNamedInt(m, "POLYCONVEX", "1.0 if polyconvexity of system is checked", 0., true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // collection of hyperelastic materials for finite strain plasticity
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_PlasticElastHyper",
        "list/collection of hyperelastic materials, i.e. material IDs",
        INPAR::MAT::m_plelasthyper));

    AddNamedInt(m, "NUMMAT", "number of materials/potentials in list");
    AddNamedIntVector(m, "MATIDS", "the list material/potential IDs", "NUMMAT");
    AddNamedReal(m, "DENS", "material mass density");
    AddNamedReal(m, "INITYIELD", "initial yield stress");
    AddNamedInt(m, "POLYCONVEX", "1.0 if polyconvexity of system is checked", 0., true);
    AddNamedReal(m, "ISOHARD", "linear isotropic hardening modulus", 0., true);
    AddNamedReal(m, "EXPISOHARD", "nonlinear isotropic hardening exponent", 0., true);
    AddNamedReal(
        m, "INFYIELD", "saturation yield stress for nonlinear isotropic hardening", 0., true);
    AddNamedReal(m, "KINHARD", "linear kinematic hardening modulus", 0., true);

    // visco-plasticity
    AddNamedReal(m, "VISC", "Visco-Plasticity parameter 'eta' in Perzyna model", 0., true);
    AddNamedReal(
        m, "RATE_DEPENDENCY", "Visco-Plasticity parameter 'eta' in Perzyna model", 1., true);
    AddNamedReal(m, "VISC_SOFT",
        "Visco-Plasticity temperature dependency (eta = eta_0 * (1-(T-T_0)*x)", 0., true);

    // optional pastic spin parameter
    AddNamedReal(
        m, "PL_SPIN_CHI", "Plastic spin coupling parameter chi (often called eta)", 0.0, true);

    // optional Hill yield parameters
    AddNamedReal(m, "rY_11", "relative yield stress in fiber1-direction (Y_11/Y_0)", 0.0, true);
    AddNamedReal(m, "rY_22", "relative yield stress in fiber2-direction (Y_22/Y_0)", 0.0, true);
    AddNamedReal(m, "rY_33", "relative yield stress in fiber3-direction (Y_33/Y_0)", 0.0, true);
    AddNamedReal(m, "rY_12", "relative shear yield stress in 12-direction (Y_12/Y_0)", 0.0, true);
    AddNamedReal(m, "rY_23", "relative shear yield stress in 23-direction (Y_23/Y_0)", 0.0, true);
    AddNamedReal(m, "rY_13", "relative shear yield stress in 13-direction (Y_13/Y_0)", 0.0, true);

    // optional TSI parameters
    AddNamedReal(m, "CTE", "coefficient of thermal expansion", 0., true);
    AddNamedReal(m, "INITTEMP", "initial, reference temperature", 0., true);
    AddNamedReal(m, "YIELDSOFT", "yield stress softening", 0., true);
    AddNamedReal(m, "HARDSOFT", "hardening softening", 0., true);
    AddNamedReal(
        m, "TAYLOR_QUINNEY", "Taylor-Quinney factor for plastic heat conversion", 1., true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // collection of hyperelastic materials for finite strain plasticity
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_PlasticElastHyperVCU",
        "list/collection of hyperelastic materials, i.e. material IDs",
        INPAR::MAT::m_plelasthyperVCU));

    AddNamedInt(m, "NUMMAT", "number of materials/potentials in list");
    AddNamedIntVector(m, "MATIDS", "the list material/potential IDs", "NUMMAT");
    AddNamedReal(m, "DENS", "material mass density");
    AddNamedReal(m, "INITYIELD", "initial yield stress");
    AddNamedReal(m, "ISOHARD", "linear isotropic hardening modulus", 0., true);
    AddNamedReal(m, "EXPISOHARD", "nonlinear isotropic hardening exponent", 0., true);
    AddNamedReal(
        m, "INFYIELD", "saturation yield stress for nonlinear isotropic hardening", 0., true);
    AddNamedReal(m, "KINHARD", "linear kinematic hardening modulus", 0., true);

    // visco-plasticity
    AddNamedReal(m, "VISC", "Visco-Plasticity parameter 'eta' in Perzyna model", 0., true);
    AddNamedReal(
        m, "RATE_DEPENDENCY", "Visco-Plasticity parameter 'eta' in Perzyna model", 1., true);
    AddNamedReal(m, "VISC_SOFT",
        "Visco-Plasticity temperature dependency (eta = eta_0 * (1-(T-T_0)*x)", 0., true);

    // optional pastic spin parameter
    AddNamedReal(
        m, "PL_SPIN_CHI", "Plastic spin coupling parameter chi (often called eta)", 0.0, true);

    // optional Hill yield parameters
    AddNamedReal(m, "rY_11", "relative yield stress in fiber1-direction (Y_11/Y_0)", 0.0, true);
    AddNamedReal(m, "rY_22", "relative yield stress in fiber2-direction (Y_22/Y_0)", 0.0, true);
    AddNamedReal(m, "rY_33", "relative yield stress in fiber3-direction (Y_33/Y_0)", 0.0, true);
    AddNamedReal(m, "rY_12", "relative shear yield stress in 12-direction (Y_12/Y_0)", 0.0, true);
    AddNamedReal(m, "rY_23", "relative shear yield stress in 23-direction (Y_23/Y_0)", 0.0, true);
    AddNamedReal(m, "rY_13", "relative shear yield stress in 13-direction (Y_13/Y_0)", 0.0, true);

    // optional TSI parameters
    AddNamedReal(m, "CTE", "coefficient of thermal expansion", 0., true);
    AddNamedReal(m, "INITTEMP", "initial, reference temperature", 0., true);
    AddNamedReal(m, "YIELDSOFT", "yield stress softening", 0., true);
    AddNamedReal(m, "HARDSOFT", "hardening softening", 0., true);
    AddNamedReal(
        m, "TAYLOR_QUINNEY", "Taylor-Quinney factor for plastic heat conversion", 1., true);

    AddNamedInt(m, "POLYCONVEX", "1.0 if polyconvexity of system is checked", 0., true);


    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // logarithmic neo-Hooke material acc. to Bonet and Wood
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_CoupLogNeoHooke",
        "logarithmic neo-Hooke material acc. to Bonet and Wood", INPAR::MAT::mes_couplogneohooke));

    AddNamedString(m, "MODE",
        "parameter set: YN (Young's modulus and Poisson's ration; default) or Lame (mue and "
        "lambda)",
        "YN");
    AddNamedReal(m, "C1", "E or mue");
    AddNamedReal(m, "C2", "nue or lambda");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // Saint-Venant-Kirchhoff as elastic summand
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "ELAST_CoupSVK", "Saint-Venant-Kirchhoff as elastic summand", INPAR::MAT::mes_coupSVK));

    AddNamedReal(m, "YOUNG", "Young's modulus");
    AddNamedReal(m, "NUE", "Poisson's ratio");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // Simo-Pister type material
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "ELAST_CoupSimoPister", "Simo-Pister type material", INPAR::MAT::mes_coupsimopister));

    AddNamedReal(m, "MUE", "material constant");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // logarithmic mixed neo-Hooke material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_CoupLogMixNeoHooke",
        "mixed logarithmic neo-Hooke material", INPAR::MAT::mes_couplogmixneohooke));

    AddNamedString(m, "MODE",
        "parameter set: YN (Young's modulus and Poisson's ration; default) or Lame (mue and "
        "lambda)",
        "YN");
    AddNamedReal(m, "C1", "E or mue");
    AddNamedReal(m, "C2", "nue or lambda");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled exponential material for compressible material (according to Weikenmeier_2014)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_CoupExpPol",
        "compressible, isochoric exponential material law for soft tissue",
        INPAR::MAT::mes_coupexppol));
    AddNamedReal(m, "A", "material constant");
    AddNamedReal(m, "B", "material constant linear I_1");
    AddNamedReal(m, "C", "material constant linear J");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // compressible neo-Hooke material acc. to Holzapfel
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_CoupNeoHooke",
        "compressible neo-Hooke material acc. to Holzapfel", INPAR::MAT::mes_coupneohooke));

    AddNamedReal(m, "YOUNG", "Young's modulus", 0.0, true);
    AddNamedReal(m, "NUE", "Poisson's ratio", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }
  // Mooney Rivlin  material acc. to Holzapfel
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_CoupMooneyRivlin",
        "Mooney - Rivlin material acc. to Holzapfel", INPAR::MAT::mes_coupmooneyrivlin));

    AddNamedReal(m, "C1", "material constant", 0.0, true);
    AddNamedReal(m, "C2", "material constant", 0.0, true);
    AddNamedReal(m, "C3", "material constant", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled Blatz and Ko material acc. to Holzapfel
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_CoupBlatzKo",
        "Blatz and Ko material acc. to Holzapfel", INPAR::MAT::mes_coupblatzko));

    AddNamedReal(m, "MUE", "Shear modulus");
    AddNamedReal(m, "NUE", "Poisson's ratio");
    AddNamedReal(m, "F", "interpolation parameter");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of Neo-Hooke
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_IsoNeoHooke",
        "isochoric part of neo-Hooke material acc. to Holzapfel", INPAR::MAT::mes_isoneohooke));

    AddNamedReal(m, "MUE", "Shear modulus");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of one-term Ogden material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_IsoOgden",
        "isochoric part of the one-term Ogden material", INPAR::MAT::mes_isoogden));

    AddNamedReal(m, "MUE", "Shear modulus");
    AddNamedReal(m, "ALPHA", "Nonlinearity parameter");
    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric and volumetric contribution of HU dependent NeoHooke
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_IsoVolHUDependentNeoHooke",
        "isochoric and volumetric part of HU dependent neo-Hooke material",
        INPAR::MAT::mes_isovolHUdependentneohooke));

    AddNamedReal(m, "ALPHA_MAX", "");
    AddNamedReal(m, "CT_MIN", "");
    AddNamedReal(m, "CT_MAX", "");
    AddNamedReal(m, "NUE", "");
    AddNamedReal(m, "BETA", "");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric and volumetric contribution of AAAGasser
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_IsoVolAAAGasser",
        "isochoric and volumetric part of AAAGasser material (thrombus)",
        INPAR::MAT::mes_isovolaaagasser));

    AddNamedReal(m, "CLUM", "luminal stiffness parameter (2.62e3)");
    AddNamedReal(m, "CMED", "medial stiffness parameter (2.62e3)");
    AddNamedReal(m, "CABLUM", "abluminal stiffness parameter (2.62e3)");
    AddNamedReal(m, "NUE", "");
    AddNamedReal(m, "BETA", "");
    // optional parameters for uncertainty quantification
    AddNamedReal(
        m, "MULUM", "mu for luminal pdf, irrelevant for deterministic analysis", 0.0, true);
    AddNamedReal(m, "MUMED", "mu for medial pdf, irrelevant for deterministic analysis", 0.0, true);
    AddNamedReal(
        m, "MUABLUM", "mu for abluminal pdf, irrelevant for deterministic analysis", 0.0, true);
    AddNamedReal(
        m, "SIGMALUM", "std for luminal pdf, irrelevant for deterministic analysis", 0.0, true);
    AddNamedReal(
        m, "SIGMAMED", "std for medial pdf, irrelevant for deterministic analysis", 0.0, true);
    AddNamedReal(
        m, "SIGMAABLUM", "std for abluminal pdf, irrelevant for deterministic analysis", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of Yeoh
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_IsoYeoh",
        "isochoric part of  Yeoh material acc. to Holzapfel", INPAR::MAT::mes_isoyeoh));

    AddNamedReal(m, "C1", "Linear modulus");
    AddNamedReal(m, "C2", "Quadratic modulus");
    AddNamedReal(m, "C3", "Cubic modulus");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of iso1pow
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "ELAST_Iso1Pow", "isochoric part of general power material", INPAR::MAT::mes_iso1pow));

    AddNamedReal(m, "C", "material parameter");
    AddNamedInt(m, "D", "exponent");
    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of iso2pow
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "ELAST_Iso2Pow", "isochoric part of general power material", INPAR::MAT::mes_iso2pow));

    AddNamedReal(m, "C", "material parameter");
    AddNamedInt(m, "D", "exponent");
    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup1pow
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "ELAST_Coup1Pow", "part of general power material", INPAR::MAT::mes_coup1pow));

    AddNamedReal(m, "C", "material parameter");
    AddNamedInt(m, "D", "exponent");
    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup2pow
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "ELAST_Coup2Pow", "part of general power material", INPAR::MAT::mes_coup2pow));

    AddNamedReal(m, "C", "material parameter");
    AddNamedInt(m, "D", "exponent");
    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup3pow
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "ELAST_Coup3Pow", "part of general power material", INPAR::MAT::mes_coup3pow));

    AddNamedReal(m, "C", "material parameter");
    AddNamedInt(m, "D", "exponent");
    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup13apow
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_Coup13aPow",
        "hyperelastic potential summand for multiplicative coupled invariants I1 and I3",
        INPAR::MAT::mes_coup13apow));

    AddNamedReal(m, "C", "material parameter");
    AddNamedInt(m, "D", "exponent of all");
    AddNamedReal(m, "A", "negative exponent of I3");
    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of expo
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_IsoExpoPow",
        "isochoric part of  exponential material acc. to Holzapfel", INPAR::MAT::mes_isoexpopow));

    AddNamedReal(m, "K1", "material parameter");
    AddNamedReal(m, "K2", "material parameter");
    AddNamedInt(m, "C", "exponent");
    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of mooney rivlin
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_IsoMooneyRivlin",
        "isochoric part of  Mooney-Rivlin material acc. to Holzapfel",
        INPAR::MAT::mes_isomooneyrivlin));

    AddNamedReal(m, "C1", "Linear modulus for first invariant");
    AddNamedReal(m, "C2", "Linear modulus for second invariant");
    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric anisotropic material with one exponential fiber family
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_IsoMuscle_Blemker",
        "anisotropic Blemker muscle material", INPAR::MAT::mes_isomuscleblemker));

    AddNamedReal(m, "G1", "muscle along fiber shear modulus");
    AddNamedReal(m, "G2", "muscle cross fiber shear modulus");
    AddNamedReal(m, "P1", "linear material parameter for passive along-fiber response");
    AddNamedReal(m, "P2", "exponential material parameter for passive along-fiber response");
    AddNamedReal(m, "SIGMAMAX", "maximal active isometric stress");
    AddNamedReal(m, "LAMBDAOFL", "optimal fiber stretch");
    AddNamedReal(
        m, "LAMBDASTAR", "stretch at which the normalized passive fiber force becomes linear");
    AddNamedReal(m, "ALPHA", "tetanised activation level,");
    AddNamedReal(m, "BETA", "constant scaling tanh-type activation function");
    AddNamedReal(m, "ACTSTARTTIME", "starting time of muscle activation");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // test material to test elasthyper-toolbox
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_IsoTestMaterial",
        "test material to test elasthyper-toolbox", INPAR::MAT::mes_isotestmaterial));

    AddNamedReal(m, "C1", "Modulus for first invariant");
    AddNamedReal(m, "C2", "Modulus for second invariant");
    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // general fiber material for remodeling
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_RemodelFiber",
        "General fiber material for remodeling", INPAR::MAT::mes_remodelfiber));

    AddNamedInt(m, "NUMMAT", "number of materials/potentials in list");
    AddNamedIntVector(m, "MATIDS", "the list material/potential IDs", "NUMMAT");
    AddNamedReal(m, "TDECAY", "decay time of Poisson (degradation) process");
    AddNamedReal(m, "GROWTHFAC", "time constant for collagen growth", 0.0, true);
    AddNamedRealVector(m, "COLMASSFRAC",
        "initial mass fraction of first collagen fiber family in constraint mixture", "NUMMAT", 0.0,
        true);
    AddNamedReal(m, "DEPOSITIONSTRETCH", "deposition stretch");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // volumetric contribution of Sussman Bathe
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_VolSussmanBathe",
        "volumetric part of  SussmanBathe material", INPAR::MAT::mes_volsussmanbathe));

    AddNamedReal(m, "KAPPA", "dilatation modulus");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // volumetric penalty contribution
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_VolPenalty",
        "Penalty formulation for the volumetric part", INPAR::MAT::mes_volpenalty));

    AddNamedReal(m, "EPSILON", "penalty parameter");
    AddNamedReal(m, "GAMMA", "penalty parameter");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // volumetric contribution of Ogden
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "ELAST_VolOgden", "Ogden formulation for the volumetric part", INPAR::MAT::mes_vologden));

    AddNamedReal(m, "KAPPA", "dilatation modulus");
    AddNamedReal(m, "BETA", "empiric constant");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // volumetric power law contribution
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "ELAST_VolPow", "Power law formulation for the volumetric part", INPAR::MAT::mes_volpow));

    AddNamedReal(m, "A", "prefactor of power law");
    AddNamedReal(m, "EXPON", "exponent of power law");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with one exponential fiber family
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_CoupAnisoExpoActive",
        "anisotropic active fiber", INPAR::MAT::mes_coupanisoexpoactive));

    AddNamedReal(m, "K1", "linear constant");
    AddNamedReal(m, "K2", "exponential constant");
    AddNamedReal(m, "GAMMA", "angle");
    AddNamedReal(m, "K1COMP", "linear constant");
    AddNamedReal(m, "K2COMP", "exponential constant");
    AddNamedInt(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    AddNamedInt(m, "INIT", "initialization modus for fiber alignment", 1, true);
    AddNamedBool(m, "ADAPT_ANGLE", "adapt angle during remodeling", false, true);
    AddNamedReal(m, "S", "maximum contractile stress");
    AddNamedReal(m, "LAMBDAMAX", "stretch at maximum active force generation");
    AddNamedReal(m, "LAMBDA0", "stretch at zero active force generation");
    AddNamedReal(m, "DENS", "total reference mass density of constrained mixture");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with one exponential fiber family
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_CoupAnisoExpo",
        "anisotropic part with one exp. fiber", INPAR::MAT::mes_coupanisoexpo));

    AddNamedReal(m, "K1", "linear constant");
    AddNamedReal(m, "K2", "exponential constant");
    AddNamedReal(m, "GAMMA", "angle");
    AddNamedReal(m, "K1COMP", "linear constant");
    AddNamedReal(m, "K2COMP", "exponential constant");
    AddNamedInt(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    AddNamedInt(m, "INIT", "initialization modus for fiber alignment", 1, true);
    AddNamedBool(m, "ADAPT_ANGLE", "adapt angle during remodeling", false, true);
    AddNamedInt(m, "FIBER_ID", "Id of the fiber to be used (1 for first fiber, default)", 1, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with one exponential shear behavior between two fibers
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_CoupAnisoExpoShear",
        "Exponential shear behavior between two fibers", INPAR::MAT::mes_coupanisoexposhear));

    AddNamedReal(m, "K1", "linear constant");
    AddNamedReal(m, "K2", "exponential constant");
    AddNamedReal(m, "GAMMA", "angle");
    AddNamedReal(m, "K1COMP", "linear constant");
    AddNamedReal(m, "K2COMP", "exponential constant");
    AddNamedInt(m, "INIT", "initialization modus for fiber alignment", 1, true);
    AddNamedIntVector(m, "FIBER_IDS",
        "Ids of the two fibers to be used (1 for the first fiber, 2 for the second, default)", 2);

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with one pow-like fiber family
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_CoupAnisoPow",
        "anisotropic part with one pow-like fiber", INPAR::MAT::mes_coupanisopow));

    AddNamedReal(m, "K", "linear constant");
    AddNamedReal(m, "D1", "exponential constant for fiber invariant");
    AddNamedReal(m, "D2", "exponential constant for system");
    AddNamedReal(m, "ACTIVETHRES",
        "Deformation threshold for activating fibers. Default:"
        " 1.0 (off at compression); If 0.0 (always active)",
        1.0, true);
    AddNamedInt(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    AddNamedInt(m, "FIBER", "Number of the fiber family contained in the element", 1, true);
    AddNamedReal(m, "GAMMA", "angle", 0.0, true);
    AddNamedInt(m, "INIT", "initialization modus for fiber alignment", 1, true);
    AddNamedBool(m, "ADAPT_ANGLE", "adapt angle during remodeling", false, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with two exponential fiber families
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_CoupAnisoExpoTwoCoup",
        "anisotropic part with two exp. fibers", INPAR::MAT::mes_coupanisoexpotwocoup));

    AddNamedReal(m, "A4", "linear anisotropic constant for fiber 1");
    AddNamedReal(m, "B4", "exponential anisotropic constant for fiber 1");
    AddNamedReal(m, "A6", "linear anisotropic constant for fiber 2");
    AddNamedReal(m, "B6", "exponential anisotropic constant for fiber 2");
    AddNamedReal(m, "A8", "linear anisotropic constant for fiber 1 relating fiber 2");
    AddNamedReal(m, "B8", "exponential anisotropic constant for fiber 1 relating fiber 2");
    AddNamedReal(m, "GAMMA", "angle");
    AddNamedInt(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    AddNamedInt(m, "INIT", "initialization modus for fiber alignment", 1, true);
    AddNamedBool(m, "FIB_COMP", "fibers support compression: yes (true) or no (false)", true, true);
    AddNamedBool(m, "ADAPT_ANGLE", "adapt angle during remodeling", false, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with two exponential fiber families
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_CoupAnisoNeoHooke",
        "anisotropic part with one neo Hookean fiber", INPAR::MAT::mes_coupanisoneohooke));

    AddNamedReal(m, "C", "linear constant");
    AddNamedReal(m, "GAMMA", "angle");
    AddNamedInt(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    AddNamedInt(m, "INIT", "initialization modus for fiber alignment", 1, true);
    AddNamedBool(m, "ADAPT_ANGLE", "adapt angle during remodeling", false, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with the stress given by a simplified version of the contraction
  // law of Bestel-Clement-Sorine
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_AnisoActiveStress_Evolution",
        "anisotropic part with one fiber with coefficient given by a simplification of the "
        "activation-contraction law of Bestel-Clement-Sorine-2001",
        INPAR::MAT::mes_anisoactivestress_evolution));

    AddNamedReal(m, "SIGMA", "Contractility (maximal stress)");
    AddNamedReal(m, "TAUC0", "Initial value for the active stress");
    AddNamedReal(m, "MAX_ACTIVATION", "Maximal value for the rescaled activation");
    AddNamedReal(m, "MIN_ACTIVATION", "Minimal value for the rescaled activation");
    AddNamedInt(
        m, "SOURCE_ACTIVATION", "Where the activation comes from: 0=scatra , >0 Id for FUNCT");
    AddNamedReal(m, "ACTIVATION_THRES",
        "Threshold for activation (contraction starts when activation function is larger than this "
        "value, relaxes otherwise)");
    AddNamedBool(m, "STRAIN_DEPENDENCY",
        "model strain dependency of contractility (Frank-Starling law): no (false) or yes (true)",
        false, true);
    AddNamedReal(m, "LAMBDA_LOWER", "lower fiber stretch for Frank-Starling law", 1.0, true);
    AddNamedReal(m, "LAMBDA_UPPER", "upper fiber stretch for Frank-Starling law", 1.0, true);
    AddNamedReal(m, "GAMMA", "angle", 0.0, true);
    AddNamedInt(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    AddNamedInt(m, "INIT", "initialization mode for fiber alignment", 1, true);
    AddNamedBool(m, "ADAPT_ANGLE", "adapt angle during remodeling", false, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with variable stress coefficient
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_CoupAnisoNeoHooke_VarProp",
        "anisotropic part with one neo Hookean fiber with variable coefficient",
        INPAR::MAT::mes_coupanisoneohooke_varprop));

    AddNamedReal(m, "C", "linear constant");
    AddNamedInt(
        m, "SOURCE_ACTIVATION", "Where the activation comes from: 0=scatra , >0 Id for FUNCT");
    AddNamedReal(m, "GAMMA", "azimuth angle", 0.0, true);
    AddNamedReal(m, "THETA", "polar angle", 0.0, true);
    AddNamedInt(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    AddNamedInt(m, "INIT", "initialization mode for fiber alignment", 1, true);
    AddNamedBool(m, "ADAPT_ANGLE", "adapt angle during remodeling", false, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric anisotropic material with one exponential fiber family
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_IsoAnisoExpo",
        "anisotropic part with one exp. fiber", INPAR::MAT::mes_isoanisoexpo));

    AddNamedReal(m, "K1", "linear constant");
    AddNamedReal(m, "K2", "exponential constant");
    AddNamedReal(m, "GAMMA", "angle");
    AddNamedReal(m, "K1COMP", "linear constant");
    AddNamedReal(m, "K2COMP", "exponential constant");
    AddNamedInt(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    AddNamedInt(m, "INIT", "initialization modus for fiber alignment", 1, true);
    AddNamedBool(m, "ADAPT_ANGLE", "adapt angle during remodeling", false, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // structural tensor
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_StructuralTensor",
        "Parameter for structural tensor strategy in anisotropic materials",
        INPAR::MAT::mes_structuraltensorstratgy));

    AddNamedString(m, "STRATEGY",
        "Strategy for evaluation of structural tensor: "
        "Standard (default), ByDistributionFunction, DispersedTransverselyIsotropic",
        "Standard");

    // choose between:
    // "none"
    // "Bingham"
    // "vonMisesFisher"
    //  rauch 10/17
    AddNamedString(m, "DISTR",
        "Type of distribution function around mean direction: "
        "none, Bingham, vonMisesFisher",
        "none", true);

    AddNamedReal(m, "C1", "constant 1 for distribution function", 1.0, true);
    AddNamedReal(m, "C2", "constant 2 for distribution function", 0.0, true);
    AddNamedReal(m, "C3", "constant 3 for distribution function", 0.0, true);
    AddNamedReal(m, "C4", "constant 4 for distribution function", 1e16, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // transversely isotropic material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("ELAST_CoupTransverselyIsotropic",
        "transversely part of a simple othotropic, transversely "
        "isotropic hyperelastic constitutive equation",
        INPAR::MAT::mes_couptransverselyisotropic));

    AddNamedReal(m, "ALPHA", "1-st constant");
    AddNamedReal(m, "BETA", "2-nd constant");
    AddNamedReal(m, "GAMMA", "3-rd constant");
    AddNamedReal(m, "ANGLE", "fiber angle");
    AddNamedInt(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    AddNamedInt(m, "FIBER", "exponential constant", 1, true);
    AddNamedInt(m, "INIT", "initialization modus for fiber alignment", 1, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled Varga material acc. to Holzapfel
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "ELAST_CoupVarga", "Varga material acc. to Holzapfel", INPAR::MAT::mes_coupvarga));

    AddNamedReal(m, "MUE", "Shear modulus");
    AddNamedReal(m, "BETA", "'Anti-modulus'");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric Varga material acc. to Holzapfel
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "ELAST_IsoVarga", "Isochoric Varga material acc. to Holzapfel", INPAR::MAT::mes_isovarga));

    AddNamedReal(m, "MUE", "Shear modulus");
    AddNamedReal(m, "BETA", "'Anti-modulus'");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isotropic viscous contribution of myocardial matrix (chapelle12)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("VISCO_CoupMyocard",
        "Isotropic viscous contribution of myocardial matrix", INPAR::MAT::mes_coupmyocard));

    AddNamedReal(m, "N", "material parameter");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric rate dependent viscos material, modified from Pioletti,1997
  {
    auto m = Teuchos::rcp(new MaterialDefinition("VISCO_IsoRateDep",
        "Isochoric rate dependent viscous material", INPAR::MAT::mes_isoratedep));

    AddNamedReal(m, "N", "material parameter");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // viscos contribution to visohyperelastic material according to SLS-Model
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "VISCO_GenMax", "Viscous contribution according to SLS-Model", INPAR::MAT::mes_genmax));

    AddNamedReal(m, "TAU", "relaxation parameter");
    AddNamedReal(m, "BETA", "emphasis of viscous to elastic part");
    AddNamedString(m, "SOLVE",
        "Solution of evolution equation via: OST (default) or CONVOL (convolution integral)",
        "OST");


    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // viscos contribution to visohyperelastic material according to FSLS-Model
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "VISCO_Fract", "Viscous contribution according to FSLS-Model", INPAR::MAT::mes_fract));

    AddNamedReal(m, "TAU", "relaxation parameter");
    AddNamedReal(m, "ALPHA", "fractional order derivative");
    AddNamedReal(m, "BETA", "emphasis of viscous to elastic part");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // viscous contribution of a branch of a generalized Maxwell model
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "VISCO_PART", "Viscous contribution of a viscoelastic Branch", INPAR::MAT::mes_viscopart));

    AddNamedReal(m, "TAU", "dynamic viscosity divided by young's modulus of the branch");

    AppendMaterialDefinition(matlist, m);
  }
  /*--------------------------------------------------------------------*/
  // viscoelatic branches of a generalized Maxwell model
  {
    auto m = Teuchos::rcp(new MaterialDefinition("VISCO_GeneralizedGenMax",
        "Viscoelastic Branches of generalized Maxwell", INPAR::MAT::mes_generalizedgenmax));

    AddNamedInt(m, "NUMBRANCH", "number of viscoelastic branches");
    AddNamedIntVector(m, "MATIDS", "the list material IDs", "NUMBRANCH");
    AddNamedString(m, "SOLVE",
        "Solution for evolution equation: OST (default) or CONVOL (convolution integral)",
        "CONVOL");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // description of a viscoelatic branch of a generalized Maxwell model
  {
    auto m = Teuchos::rcp(new MaterialDefinition("VISCO_BRANCH",
        "Viscoelastic Branch (viscous and elastic contribution)", INPAR::MAT::mes_viscobranch));

    AddNamedInt(m, "NUMMAT", "number of materials in the viscoelastic branch");
    AddNamedIntVector(m, "MATIDS", "the list material IDs", "NUMMAT");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // 1D Artery material with constant properties
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_CNST_ART", "artery with constant properties", INPAR::MAT::m_cnst_art));

    AddNamedReal(m, "VISCOSITY",
        "viscosity (for CONSTANT viscosity law taken as blood viscosity, for BLOOD viscosity law "
        "taken as the viscosity of blood plasma)");
    AddNamedReal(m, "DENS", "density of blood");
    AddNamedReal(m, "YOUNG", "artery Youngs modulus of elasticity");
    AddNamedReal(m, "NUE", "Poissons ratio of artery fiber");
    AddNamedReal(m, "TH", "artery thickness");
    AddNamedReal(m, "PEXT1", "artery fixed external pressure 1");
    AddNamedReal(m, "PEXT2", "artery fixed external pressure 2");
    AddNamedString(
        m, "VISCOSITYLAW", "type of viscosity law, CONSTANT (default) or BLOOD", "CONSTANT", true);
    AddNamedReal(m, "BLOOD_VISC_SCALE_DIAM_TO_MICRONS",
        "used to scale the diameter for blood viscosity law to microns if your problem is not "
        "given in microns, e.g., if you use mms, set this parameter to 1.0e3",
        1.0, true);
    AddNamedString(m, "VARYING_DIAMETERLAW",
        "type of varying diameter law, CONSTANT (default) or BY_FUNCTION", "CONSTANT", true);
    AddNamedInt(m, "VARYING_DIAMETER_FUNCTION", "function for varying diameter law", -1, true);
    AddNamedReal(m, "COLLAPSE_THRESHOLD",
        "Collapse threshold for diameter (below this diameter element is assumed to be collapsed "
        "with zero diameter and is not evaluated)",
        -1.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // Fourier's law
  {
    auto m = Teuchos::rcp(new MaterialDefinition("THERM_FourierIso",
        "isotropic (linear) Fourier's law of heat conduction", INPAR::MAT::m_th_fourier_iso));

    AddNamedReal(m, "CAPA", "volumetric heat capacity");
    AddNamedReal(m, "CONDUCT", "thermal conductivity");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material for heat transport due to Fourier-type thermal conduction and the Soret effect (fang
  // 06/15)
  {
    auto matsoret = Teuchos::rcp(new MaterialDefinition("MAT_soret",
        "material for heat transport due to Fourier-type thermal conduction and the Soret effect",
        INPAR::MAT::m_soret));

    // mandatory parameters
    AddNamedReal(matsoret, "CAPA", "volumetric heat capacity");
    AddNamedReal(matsoret, "CONDUCT", "thermal conductivity");
    AddNamedReal(matsoret, "SORET", "Soret coefficient");

    // add Soret material to global list of valid materials
    AppendMaterialDefinition(matlist, matsoret);
  }

  /*----------------------------------------------------------------------*/
  // integration point based growth
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_GrowthVolumetric", "volumetric growth", INPAR::MAT::m_growth_volumetric));

    AddNamedInt(m, "GROWTHLAW", "number of growth law in input file");
    AddNamedInt(
        m, "IDMATELASTIC", "number of elastic material in input file: MAT IDMATELASTIC ...");
    AddNamedReal(m, "STARTTIME", "start growth after this time");
    AddNamedReal(m, "ENDTIME", "end growth after this time");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // collection of hyperelastic materials for membranes
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Membrane_ElastHyper",
        "list/collection of hyperelastic materials for membranes, i.e. material IDs",
        INPAR::MAT::m_membrane_elasthyper));

    AddNamedInt(m, "NUMMAT", "number of materials/potentials in list");
    AddNamedIntVector(m, "MATIDS", "the list material/potential IDs", "NUMMAT");
    AddNamedReal(m, "DENS", "material mass density");
    AddNamedInt(m, "POLYCONVEX", "1.0 if polyconvexity of system is checked", 0., true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // active strain membrane material for gastric electromechanics
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_Membrane_ActiveStrain",
        "active strain membrane material", INPAR::MAT::m_membrane_activestrain));

    AddNamedInt(m, "MATIDPASSIVE", "MATID for the passive material", false);
    AddNamedInt(m, "SCALIDVOLTAGE", "ID of the scalar that represents the (SMC) voltage", false);
    AddNamedReal(m, "DENS", "material mass density", false);
    AddNamedReal(m, "BETA1", "Ca2+ dynamics", false);
    AddNamedReal(m, "BETA2", "opening dynamics of the VDCC", false);
    AddNamedReal(m, "VOLTHRESH", "voltage threshold for activation", false);
    AddNamedReal(m, "ALPHA1", "intensity of contraction in fiber direction 1", false);
    AddNamedReal(m, "ALPHA2", "intensity of contraction in fiber direction 2", false);
    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // growth and remodeling (homogenized constrained mixture model)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_GrowthRemodel_ElastHyper",
        "growth and remodeling", INPAR::MAT::m_growthremodel_elasthyper));

    AddNamedInt(m, "NUMMATRF", "number of remodelfiber materials in list", false);
    AddNamedInt(
        m, "NUMMATEL3D", "number of 3d elastin matrix materials/potentials in list", 0, true);
    AddNamedInt(m, "NUMMATEL2D", "number of 2d elastin matrix materials/potentials in list", false);
    AddNamedIntVector(m, "MATIDSRF", "the list remodelfiber material IDs", "NUMMATRF", false);
    AddNamedIntVector(m, "MATIDSEL3D", "the list 3d elastin matrix material/potential IDs",
        "NUMMATEL3D", -1, true);
    AddNamedIntVector(
        m, "MATIDSEL2D", "the list 2d elastin matrix material/potential IDs", "NUMMATEL2D", false);
    AddNamedInt(m, "MATIDELPENALTY", "penalty material ID", -1, true);
    AddNamedReal(
        m, "ELMASSFRAC", "initial mass fraction of elastin matrix in constraint mixture", false);
    AddNamedReal(m, "DENS", "material mass density", false);
    AddNamedReal(m, "PRESTRETCHELASTINCIR", "circumferential prestretch of elastin matrix", false);
    AddNamedReal(m, "PRESTRETCHELASTINAX", "axial prestretch of elastin matrix", false);
    AddNamedReal(m, "THICKNESS",
        "reference wall thickness of the idealized cylindrical aneurysm [m]", -1, true);
    AddNamedReal(m, "MEANPRESSURE", "mean blood pressure [Pa]", -1.0, true);
    AddNamedReal(m, "RADIUS", "inner radius of the idealized cylindrical aneurysm [m]", -1.0, true);
    AddNamedInt(m, "DAMAGE", "1: elastin damage after prestressing,0: no elastin damage", false);
    AddNamedInt(m, "GROWTHTYPE",
        "flag to decide what type of collagen growth is used: 1: anisotropic growth; 0: isotropic "
        "growth",
        false);
    AddNamedInt(m, "LOCTIMEINT",
        "flag to decide what type of local time integration scheme is used: 1: Backward Euler "
        "Method; 0: Forward Euler Method",
        false);
    AddNamedInt(m, "MEMBRANE",
        "Flag whether Hex or Membrane elements are used ( Membrane: 1, Hex: Everything else )", -1,
        true);
    AddNamedInt(m, "CYLINDER",
        "Flag that geometry is a cylinder. 1: aligned in x-direction; 2: y-direction; 3: "
        "z-direction",
        -1, true);
    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // multiplicative split of deformation gradient in elastic and inelastic parts
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_MultiplicativeSplitDefgradElastHyper",
        "multiplicative split of deformation gradient",
        INPAR::MAT::m_multiplicative_split_defgrad_elasthyper));

    AddNamedInt(m, "NUMMATEL", "number of elastic materials/potentials in list", 0, false);
    AddNamedIntVector(
        m, "MATIDSEL", "the list of elastic material/potential IDs", "NUMMATEL", -1, false);
    AddNamedInt(m, "NUMFACINEL", "number of factors of inelastic deformation gradient", false);
    AddNamedIntVector(m, "INELDEFGRADFACIDS",
        "the list of inelastic deformation gradient factor IDs", "NUMFACINEL", false);
    AddNamedReal(m, "DENS", "material mass density", false);
    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // simple inelastic material law featuring no volume change
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_InelasticDefgradNoGrowth",
        "no volume change, i.e. the inelastic deformation gradient is the identity tensor",
        INPAR::MAT::mfi_no_growth));

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // simple isotropic, volumetric growth; growth is linearly dependent on scalar mapped to material
  // configuration, constant material density
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_InelasticDefgradLinScalarIso",
        "scalar dependent isotropic growth law; volume change linearly dependent on scalar (in "
        "material configuration)",
        INPAR::MAT::mfi_lin_scalar_iso));

    AddNamedInt(m, "SCALAR1", "number of growth inducing scalar");
    AddNamedReal(m, "SCALAR1_MolarGrowthFac", "isotropic molar growth factor due to scalar 1");
    AddNamedReal(m, "SCALAR1_RefConc", "reference concentration of scalar 1 causing no strains");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // simple anisotropic, volumetric growth; growth direction prescribed in input-file;
  // growth is linearly dependent on scalar mapped to material configuration, constant material
  // density
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_InelasticDefgradLinScalarAniso",
        "scalar dependent anisotropic growth law; growth in direction as given in input-file; "
        "volume change linearly dependent on scalar (in material configuration)",
        INPAR::MAT::mfi_lin_scalar_aniso));

    AddNamedInt(m, "SCALAR1", "number of growth inducing scalar");
    AddNamedReal(m, "SCALAR1_MolarGrowthFac", "anisotropic molar growth factor due to scalar 1");
    AddNamedReal(m, "SCALAR1_RefConc", "reference concentration of scalar 1 causing no strains");
    AddNamedInt(m, "NUMSPACEDIM", "Number of space dimension (only 3 valid)");
    AddNamedRealVector(
        m, "GrowthDirection", "vector that defines the growth direction", "NUMSPACEDIM");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // non-linear isotropic volumetric growth; growth is dependent on the degree of lithiation,
  // constant material density, nonlinear behavior prescribed by polynomial in input file
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_InelasticDefgradPolyIntercalFracIso",
        "scalar dependent isotropic growth law; volume change nonlinearly dependent on the "
        "intercalation fraction, that is calculated using the scalar concentration (in material "
        "configuration)",
        INPAR::MAT::mfi_poly_intercal_frac_iso));

    AddNamedInt(m, "SCALAR1", "number of growth inducing scalar");
    AddNamedReal(m, "SCALAR1_RefConc", "reference concentration of scalar 1 causing no strains");
    AddNamedInt(m, "POLY_PARA_NUM", "number of polynomial coefficients");
    AddNamedRealVector(m, "POLY_PARAMS", "coefficients of polynomial", "POLY_PARA_NUM");
    AddNamedReal(m, "X_min", "lower bound of validity of polynomial");
    AddNamedReal(m, "X_max", "upper bound of validity of polynomial");
    AddNamedInt(m, "MATID", "material ID of the corresponding scatra material");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // non-linear anisotropic volumetric growth; growth direction prescribed in input-file;
  // growth is dependent on the degree of lithiation, constant material density, nonlinear behavior
  // prescribed by polynomial in input file
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_InelasticDefgradPolyIntercalFracAniso",
        "scalar dependent anisotropic growth law; growth in direction as given in input-file; "
        "volume change nonlinearly dependent on the intercalation fraction, that is calculated "
        "using the scalar concentration (in material configuration)",
        INPAR::MAT::mfi_poly_intercal_frac_aniso));

    AddNamedInt(m, "SCALAR1", "number of growth inducing scalar");
    AddNamedReal(m, "SCALAR1_RefConc", "reference concentration of scalar 1 causing no strains");
    AddNamedInt(m, "NUMSPACEDIM", "Number of space dimension (only 3 valid)");
    AddNamedRealVector(
        m, "GrowthDirection", "vector that defines the growth direction", "NUMSPACEDIM");
    AddNamedInt(m, "POLY_PARA_NUM", "number of polynomial coefficients");
    AddNamedRealVector(m, "POLY_PARAMS", "coefficients of polynomial", "POLY_PARA_NUM");
    AddNamedReal(m, "X_min", "lower bound of validity of polynomial");
    AddNamedReal(m, "X_max", "upper bound of validity of polynomial");
    AddNamedInt(m, "MATID", "material ID of the corresponding scatra material");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_InelasticDefgradLinTempIso",
        "Temperature dependent growth law. Volume change linearly dependent on temperature",
        INPAR::MAT::mfi_lin_temp_iso));

    AddNamedReal(m, "Temp_GrowthFac", "isotropic growth factor due to temperature");
    AddNamedReal(m, "RefTemp", "reference temperature causing no strains");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // integration point based and scalar dependent interpolation between to materials
  {
    auto mm = Teuchos::rcp(new MaterialDefinition("MAT_ScDepInterp",
        "integration point based and scalar dependent interpolation between to materials",
        INPAR::MAT::m_sc_dep_interp));

    AddNamedInt(mm, "IDMATZEROSC", "material for lambda equal to zero");
    AddNamedInt(mm, "IDMATUNITSC", "material for lambda equal to one");
    //      AddNamedReal(mm,"ALPHA","size of ",-1.0,true);

    AppendMaterialDefinition(matlist, mm);
  }

  /*----------------------------------------------------------------------*/
  // anisotropic strain-dependent growth law (Goektepe et al., J Theor Biol 2010, Lee et al., BMMB
  // 2017)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_GrowthAnisoStrain",
        "growth law depending on elastic stretch in fiber direction, growth in fiber direction",
        INPAR::MAT::m_growth_aniso_strain));

    AddNamedReal(m, "TAU", "growth time scale");
    AddNamedReal(m, "TAU_REV", "reverse growth time scale");
    AddNamedReal(m, "THETA_MIN", "lower limit for growth stretch");
    AddNamedReal(m, "THETA_MAX", "upper limit for growth stretch");
    AddNamedReal(m, "GAMMA", "growth non-linearity");
    AddNamedReal(m, "GAMMA_REV", "reverse growth non-linearity");
    AddNamedReal(m, "LAMBDA_CRIT", "critical fiber stretch");
    AddNamedReal(m, "TOL", "tolerance for local Newton iteration");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // anisotropic strain-dependent growth law (Goektepe et al., J Theor Biol 2010, Lee et al., BMMB
  // 2017)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_GrowthAnisoStress",
        "growth law depending on elastic Mandel stress, growth perpendicular to fiber direction",
        INPAR::MAT::m_growth_aniso_stress));

    AddNamedReal(m, "TAU", "growth time scale");
    AddNamedReal(m, "TAU_REV", "reverse growth time scale");
    AddNamedReal(m, "THETA_MIN", "lower limit for growth stretch");
    AddNamedReal(m, "THETA_MAX", "upper limit for growth stretch");
    AddNamedReal(m, "GAMMA", "growth non-linearity");
    AddNamedReal(m, "GAMMA_REV", "reverse growth non-linearity");
    AddNamedReal(m, "P_CRIT", "critical pressure");
    AddNamedReal(m, "TOL", "tolerance for local Newton iteration");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // anisotropic strain-dependent growth law with constant prescribed trigger (for multiscale in
  // time)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_GrowthAnisoStrainConstTrig",
        "growth law depending on prescribed constant elastic stretch in fiber direction, "
        "growth in fiber direction",
        INPAR::MAT::m_growth_aniso_strain_const_trig));

    AddNamedReal(m, "TAU", "growth time scale");
    AddNamedReal(m, "TAU_REV", "reverse growth time scale");
    AddNamedReal(m, "THETA_MIN", "lower limit for growth stretch");
    AddNamedReal(m, "THETA_MAX", "upper limit for growth stretch");
    AddNamedReal(m, "GAMMA", "growth non-linearity");
    AddNamedReal(m, "GAMMA_REV", "reverse growth non-linearity");
    AddNamedReal(m, "LAMBDA_CRIT", "critical fiber stretch");
    AddNamedReal(m, "TOL", "tolerance for local Newton iteration");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // anisotropic strain-dependent growth law with constant prescribed trigger (for multiscale in
  // time)
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_GrowthAnisoStressConstTrig",
        "growth law depending on prescribed constant elastic Mandel stress, growth "
        "perpendicular to fiber direction",
        INPAR::MAT::m_growth_aniso_stress_const_trig));

    AddNamedReal(m, "TAU", "growth time scale");
    AddNamedReal(m, "TAU_REV", "reverse growth time scale");
    AddNamedReal(m, "THETA_MIN", "lower limit for growth stretch");
    AddNamedReal(m, "THETA_MAX", "upper limit for growth stretch");
    AddNamedReal(m, "GAMMA", "growth non-linearity");
    AddNamedReal(m, "GAMMA_REV", "reverse growth non-linearity");
    AddNamedReal(m, "P_CRIT", "critical pressure");
    AddNamedReal(m, "TOL", "tolerance for local Newton iteration");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // isotropic growth law (cf. Diss Tinkl 2015, LNM)
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_GrowthIsoStress", "stress-dependent growth law", INPAR::MAT::m_growth_iso_stress));

    AddNamedReal(m, "THETAPLUS", "maximal growth stretch");
    AddNamedReal(m, "KPLUS", "growth law parameter kthetaplus");
    AddNamedReal(m, "MPLUS", "growth law parameter mthetaplus");
    AddNamedReal(m, "THETAMINUS", "minimal growth stretch");
    AddNamedReal(m, "KMINUS", "growth law parameter kthetaminus");
    AddNamedReal(m, "MMINUS", "growth law parameter mthetaminus");
    AddNamedReal(m, "HOMMANDEL", "homeostatic value for mandelstress");
    AddNamedReal(m, "TOL", "tolerance for local Newton iteration");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // simple atherosclerosis growth law, scalar-dependent volumetric growth
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_GrowthAC", "scalar depended volumetric growth", INPAR::MAT::m_growth_ac));

    AddNamedInt(m, "SCALAR1", "number of first growth inducing scalar");
    AddNamedReal(m, "ALPHA", "volume per first scalar's mass density");
    AddNamedInt(m, "SCALAR2", "number of second growth inducing scalar", 1, true);
    AddNamedReal(m, "BETA", "volume per second scalar's mass density", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // atherosclerosis growth law, scalar depended growth in radial direction
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_GrowthACRadial",
        "scalar depended growth in radial direction", INPAR::MAT::m_growth_ac_radial));

    AddNamedInt(m, "SCALAR1", "number of first growth inducing scalar");
    AddNamedReal(m, "ALPHA", "volume per first scalar's mass density");
    AddNamedInt(m, "SCALAR2", "number of second growth inducing scalar", 1, true);
    AddNamedReal(m, "BETA", "volume per second scalar's mass density", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // atherosclerosis growth law, scalar depended growth in radial direction
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_GrowthACRadialRefConc",
        "scalar depended growth in radial direction", INPAR::MAT::m_growth_ac_radial_refconc));

    AddNamedInt(m, "SCALAR1", "number of first growth inducing scalar");
    AddNamedReal(m, "ALPHA", "volume per first scalar's mass density");
    AddNamedInt(m, "SCALAR2", "number of second growth inducing scalar", 1, true);
    AddNamedReal(m, "BETA", "volume per second scalar's mass density", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // constant rate growth law
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_GrowthConst", "constant growth law", INPAR::MAT::m_growth_const));

    AddNamedReal(m, "THETARATE", "reference value for mandelstress");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // growth and remodeling of arteries
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_ConstraintMixture",
        "growth and remodeling of arteries", INPAR::MAT::m_constraintmixture));

    AddNamedReal(m, "DENS", "Density");
    AddNamedReal(m, "MUE", "Shear Modulus");
    AddNamedReal(m, "NUE", "Poisson's ratio");
    AddNamedReal(m, "PHIE", "mass fraction of elastin");
    AddNamedReal(m, "PREELA", "prestretch of elastin");
    AddNamedReal(m, "K1", "Parameter for linear collagen fiber stiffness");
    AddNamedReal(m, "K2", "Parameter for exponential collagen fiber stiffness");
    AddNamedInt(m, "NUMHOM", "Number of homeostatic parameters", 1);
    AddNamedRealVector(m, "PRECOLL", "prestretch of collagen fibers", "NUMHOM");
    AddNamedReal(m, "DAMAGE", "damage stretch of collagen fibers");
    AddNamedReal(m, "K1M", "Parameter for linear smooth muscle fiber stiffness");
    AddNamedReal(m, "K2M", "Parameter for exponential smooth muscle fiber stiffness");
    AddNamedReal(m, "PHIM", "mass fraction of smooth muscle");
    AddNamedReal(m, "PREMUS", "prestretch of smooth muscle fibers");
    AddNamedReal(m, "SMAX", "maximal active stress");
    AddNamedReal(m, "KAPPA", "dilatation modulus");
    AddNamedReal(m, "LIFETIME", "lifetime of collagen fibers");
    AddNamedReal(m, "GROWTHFAC", "growth factor for stress");
    AddNamedRealVector(m, "HOMSTR", "homeostatic target value of scalar stress measure", "NUMHOM");
    AddNamedReal(m, "SHEARGROWTHFAC", "growth factor for shear");
    AddNamedReal(m, "HOMRAD", "homeostatic target value of inner radius");
    AddNamedReal(m, "STARTTIME", "at this time turnover of collagen starts");
    AddNamedString(m, "INTEGRATION",
        "time integration scheme: "
        "Explicit (default), or Implicit",
        "Explicit");
    AddNamedReal(m, "TOL", "tolerance for local Newton iteration, only for implicit integration");
    AddNamedString(m, "GROWTHFORCE",
        "driving force of growth: "
        "Single (default), All, ElaCol",
        "Single");
    AddNamedString(m, "ELASTINDEGRAD",
        "how elastin is degraded: "
        "None (default), Rectangle, Time",
        "None");
    AddNamedString(m, "MASSPROD",
        "how mass depends on driving force: "
        "Lin (default), CosCos",
        "Lin");
    AddNamedString(m, "INITSTRETCH",
        "how to set stretches in the beginning (None, Homeo, UpdatePrestretch)", "None");
    AddNamedInt(m, "CURVE", "number of timecurve for increase of prestretch in time", 0);
    AddNamedString(m, "DEGOPTION",
        "Type of degradation function: "
        "Lin (default), Cos, Exp, ExpVar",
        "Lin");
    AddNamedReal(m, "MAXMASSPRODFAC", "maximal factor of mass production");
    AddNamedReal(m, "ELASTINFAC", "factor for elastin content", 0.0, true);
    AddNamedBool(m, "STOREHISTORY",
        "store all history variables, not recommended for forward simulations", false, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // hyperelastic material for poroelasticity
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_StructPoro", "wrapper for structure poroelastic material", INPAR::MAT::m_structporo));

    AddNamedInt(m, "MATID", "ID of structure material");
    AddNamedInt(m, "POROLAWID", "ID of porosity law");
    AddNamedReal(m, "INITPOROSITY", "initial porosity of porous medium");

    AppendMaterialDefinition(matlist, m);
  }
  /*----------------------------------------------------------------------*/
  // linear law for porosity in porous media problems
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_PoroLawLinear",
        "linear constitutive law for porosity", INPAR::MAT::m_poro_law_linear));

    AddNamedReal(m, "BULKMODULUS", "bulk modulus of porous medium");

    AppendMaterialDefinition(matlist, m);
  }
  /*----------------------------------------------------------------------*/
  // constant law for porosity in porous media problems
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_PoroLawConstant",
        "constant constitutive law for porosity", INPAR::MAT::m_poro_law_constant));

    AppendMaterialDefinition(matlist, m);
  }
  /*----------------------------------------------------------------------*/
  // neo-hookean law for porosity in porous media problems
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_PoroLawNeoHooke",
        "NeoHookean-like constitutive law for porosity",
        INPAR::MAT::m_poro_law_logNeoHooke_Penalty));

    AddNamedReal(m, "BULKMODULUS", "bulk modulus of porous medium");
    AddNamedReal(m, "PENALTYPARAMETER", "penalty paramter of porous medium");

    AppendMaterialDefinition(matlist, m);
  }
  /*----------------------------------------------------------------------*/
  // incompressible skeleton law for porosity in porous media problems
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_PoroLawIncompSkel",
        "porosity law for incompressible skeleton phase", INPAR::MAT::m_poro_law_incompr_skeleton));

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // incompressible skeleton law for porosity in porous media problems
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_PoroLawLinBiot",
        "linear biot model for porosity law", INPAR::MAT::m_poro_law_linear_biot));

    AddNamedReal(m, "INVBIOTMODULUS", "inverse Biot modulus of porous medium");
    AddNamedReal(m, "BIOTCEOFF", "Biot coefficient of porous medium");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // incompressible skeleton law for porosity depending on the density
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_PoroLawDensityDependent",
        "porosity depending on the density", INPAR::MAT::m_poro_law_density_dependent));

    AddNamedInt(m, "DENSITYLAWID", "material ID of density law");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // density law for constant density in porous multiphase medium
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_PoroDensityLawConstant",
        "density law for constant density in porous multiphase medium",
        INPAR::MAT::m_poro_densitylaw_constant));

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // density law for constant density in porous multiphase medium
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_PoroDensityLawExp",
        "density law for pressure dependent exponential function",
        INPAR::MAT::m_poro_densitylaw_exp));

    AddNamedReal(m, "BULKMODULUS", "bulk modulus of porous medium");
    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // permeability law for constant permeability in porous multiphase medium
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_FluidPoroRelPermeabilityLawConstant",
        "permeability law for constant permeability in porous multiphase medium",
        INPAR::MAT::m_fluidporo_relpermeabilitylaw_constant));

    AddNamedReal(m, "VALUE", "constant value of permeability");
    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // permeability law for permeability depending on saturation according to (saturation)^exp
  // in porous multiphase medium
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_FluidPoroRelPermeabilityLawExp",
        "permeability law depending on saturation in porous multiphase medium",
        INPAR::MAT::m_fluidporo_relpermeabilitylaw_exp));

    AddNamedReal(m, "EXP", "exponent of the saturation of this phase");
    AddNamedReal(m, "MIN_SAT", "minimum saturation which is used for calculation");
    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // viscosity law for constant viscosity in porous multiphase medium
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_FluidPoroViscosityLawConstant",
        "viscosity law for constant viscosity in porous multiphase medium",
        INPAR::MAT::m_fluidporo_viscositylaw_constant));

    AddNamedReal(m, "VALUE", "constant value of viscosity");
    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // viscosity law for viscosity-dependency modelling cell adherence
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_FluidPoroViscosityLawCellAdherence",
        "visosity law depending on pressure gradient in porous multiphase medium",
        INPAR::MAT::m_fluidporo_viscositylaw_celladh));

    AddNamedReal(m, "VISC_0", "Visc0 parameter for modelling cell adherence");
    AddNamedReal(m, "XI", "xi parameter for modelling cell adherence");
    AddNamedReal(m, "PSI", "psi parameter for modelling cell adherence");
    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // hyperelastic material for poroelasticity with reaction
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_StructPoroReaction",
        "wrapper for structure porelastic material with reaction",
        INPAR::MAT::m_structpororeaction));

    AddNamedInt(m, "MATID", "ID of structure material");
    AddNamedInt(m, "POROLAWID", "ID of porosity law");
    AddNamedReal(m, "INITPOROSITY", "initial porosity of porous medium");
    AddNamedInt(m, "DOFIDREACSCALAR",
        "Id of DOF within scalar transport problem, which controls the reaction");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // hyperelastic material for poroelasticity with reaction
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_StructPoroReactionECM",
        "wrapper for structure porelastic material with reaction",
        INPAR::MAT::m_structpororeactionECM));

    AddNamedInt(m, "MATID", "ID of structure material");
    AddNamedInt(m, "POROLAWID", "ID of porosity law");
    AddNamedReal(m, "INITPOROSITY", "initial porosity of porous medium");
    AddNamedReal(m, "DENSCOLLAGEN", "density of collagen");
    AddNamedInt(m, "DOFIDREACSCALAR",
        "Id of DOF within scalar transport problem, which controls the reaction");
    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // fluid flow in a poroelastic material
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_FluidPoro", "flow in deformable porous media", INPAR::MAT::m_fluidporo));

    AddNamedReal(m, "DYNVISCOSITY", "dynamic viscosity");
    AddNamedReal(m, "DENSITY", "density");
    AddNamedReal(m, "PERMEABILITY", "permeability of medium", 0.0, true);
    AddNamedReal(m, "AXIALPERMEABILITY", "axial permeability for transverse isotropy", 0.0, true);
    AddNamedReal(m, "ORTHOPERMEABILITY1", "first permeability for orthotropy", 0.0, true);
    AddNamedReal(m, "ORTHOPERMEABILITY2", "second permeability for orthotropy", 0.0, true);
    AddNamedReal(m, "ORTHOPERMEABILITY3", "third permeability for orthotropy", 0.0, true);
    AddNamedString(m, "TYPE", "Problem type: Darcy (default) or Darcy-Brinkman", "Darcy");
    // optional parameter
    AddNamedString(m, "PERMEABILITYFUNCTION",
        "Permeability function: Const(Default) or Kozeny_Carman", "Const", true);
    //  AddNamedReal(m,"BULKMODULUS","bulk modulus of medium");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // multiphase flow in a poroelastic material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_FluidPoroMultiPhase",
        "multi phase flow in deformable porous media", INPAR::MAT::m_fluidporo_multiphase));

    AddNamedBool(m, "LOCAL", "individual materials allocated per element or only at global scope");
    AddNamedReal(m, "PERMEABILITY", "permeability of medium");
    AddNamedInt(m, "NUMMAT", "number of materials in list");
    AddNamedIntVector(m, "MATIDS", "the list material IDs", "NUMMAT");
    AddNamedInt(m, "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE", "number of fluid phases");
    AddNamedSeparator(m, "END", "indicating end of line");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // multiphase flow in a poroelastic material with reactions
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_FluidPoroMultiPhaseReactions",
        "multi phase flow in deformable porous media and list of reactions",
        INPAR::MAT::m_fluidporo_multiphase_reactions));

    AddNamedBool(m, "LOCAL", "individual materials allocated per element or only at global scope");
    AddNamedReal(m, "PERMEABILITY", "permeability of medium");
    AddNamedInt(m, "NUMMAT", "number of materials in list");
    AddNamedIntVector(m, "MATIDS", "the list material IDs", "NUMMAT");
    AddNamedInt(m, "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE", "number of fluid phases");
    AddNamedInt(m, "NUMREAC", "number of reactions for these elements", 0);
    AddNamedIntVector(m, "REACIDS", "advanced reaction list", "NUMREAC", 0);
    AddNamedSeparator(m, "END", "indicating end of line");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one reaction for multiphase flow in a poroelastic material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_FluidPoroSingleReaction",
        "advanced reaction material", INPAR::MAT::m_fluidporo_singlereaction));

    AddNamedInt(m, "NUMSCAL", "number of scalars coupled with this problem");
    AddNamedInt(m, "TOTALNUMDOF", "total number of multiphase-dofs");
    AddNamedInt(m, "NUMVOLFRAC", "number of volfracs");
    AddNamedIntVector(m, "SCALE", "advanced reaction list", "TOTALNUMDOF");
    AddNamedString(m, "COUPLING",
        "type of coupling: "
        "scalar_by_function, no_coupling (default)",
        "no_coupling", false);
    AddNamedInt(m, "FUNCTID", "function ID defining the reaction");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one phase for multiphase flow in a poroelastic material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_FluidPoroSinglePhase",
        "one phase for multiphase flow in deformable porous media",
        INPAR::MAT::m_fluidporo_singlephase));

    AddNamedInt(m, "DENSITYLAWID", "ID of density law");
    AddNamedReal(m, "DENSITY", "reference/initial density");
    AddNamedInt(m, "RELPERMEABILITYLAWID", "ID of relative permeability law");
    AddNamedInt(m, "VISCOSITYLAWID", "ID of viscosity law");
    AddNamedInt(m, "DOFTYPEID", "ID of dof definition");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one volume fraction for multiphase flow in a poroelastic material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_FluidPoroSingleVolFrac",
        "one phase for multiphase flow in deformable porous media",
        INPAR::MAT::m_fluidporo_singlevolfrac));

    AddNamedReal(m, "DENSITY", "reference/initial density");
    AddNamedReal(m, "DIFFUSIVITY", "diffusivity of phase");
    AddNamedBool(
        m, "AddScalarDependentFlux", "Is there additional scalar dependent flux (yes) or (no)");
    AddNamedInt(m, "NUMSCAL", "Number of scalars", 0, true);
    AddNamedRealVector(m, "SCALARDIFFS", "Diffusivities for additional scalar-dependent flux",
        "NUMSCAL", 0.0, true);
    AddNamedRealVector(
        m, "OMEGA_HALF", "Constant for receptor kinetic law", "NUMSCAL", 1.0e13, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one volume fraction pressure for multiphase flow in a poroelastic material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_FluidPoroVolFracPressure",
        "one volume fraction pressure for multiphase flow in deformable porous media",
        INPAR::MAT::m_fluidporo_volfracpressure));

    AddNamedReal(m, "PERMEABILITY", "permeability of phase");
    AddNamedInt(m, "VISCOSITYLAWID", "ID of viscosity law");
    AddNamedReal(m, "MIN_VOLFRAC",
        "Minimum volume fraction under which we assume that VolfracPressure is zero", 1.0e-3, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one degree of freedom for on single phase of a multiphase flow in a poroelastic material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_FluidPoroSinglePhaseDofDiffPressure",
        "one degrree of freedom for multiphase flow in deformable porous media",
        INPAR::MAT::m_fluidporo_phasedof_diffpressure));

    AddNamedInt(m, "PHASELAWID", "ID of pressure-saturation law");
    AddNamedInt(m, "NUMDOF", "number of DoFs", 0);
    AddNamedIntVector(m, "PRESCOEFF", "pressure IDs for differential pressure", "NUMDOF", 0);
    AddNamedSeparator(m, "END", "indicating end of line");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one degree of freedom for on single phase of a multiphase flow in a poroelastic material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_FluidPoroSinglePhaseDofPressure",
        "one degrree of freedom for multiphase flow in deformable porous media",
        INPAR::MAT::m_fluidporo_phasedof_pressure));

    AddNamedInt(m, "PHASELAWID", "ID of pressure-saturation law");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one degree of freedom for on single phase of a multiphase flow in a poroelastic material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_FluidPoroSinglePhaseDofSaturation",
        "one degrree of freedom for multiphase flow in deformable porous media",
        INPAR::MAT::m_fluidporo_phasedof_saturation));

    AddNamedInt(m, "PHASELAWID", "ID of pressure-saturation law");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // saturated law for pressure-saturation law in porous media problems
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_PhaseLawLinear",
        "saturated fluid phase of porous medium", INPAR::MAT::m_fluidporo_phaselaw_linear));

    AddNamedReal(m, "RELTENSION", "relative interface tensions");
    AddNamedReal(m, "SATURATION_0", "saturation at zero differential pressure");
    AddNamedInt(m, "NUMDOF", "number of DoFs", 0);
    AddNamedIntVector(m, "PRESCOEFF", "Coefficients for pressure dependence", "NUMDOF", 0);
    AddNamedSeparator(m, "END", "indicating end of line");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // tangent law for pressure-saturation law in porous media multiphase problems
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_PhaseLawTangent",
        "tangent fluid phase of porous medium", INPAR::MAT::m_fluidporo_phaselaw_tangent));

    AddNamedReal(m, "RELTENSION", "relative interface tensions");
    AddNamedReal(m, "EXP", "exponent in pressure-saturation law");
    AddNamedReal(m, "SATURATION_0", "saturation at zero differential pressure");
    AddNamedInt(m, "NUMDOF", "number of DoFs", 0);
    AddNamedIntVector(m, "PRESCOEFF", "Coefficients for pressure dependence", "NUMDOF", 0);
    AddNamedSeparator(m, "END", "indicating end of line");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // constraint law for pressure-saturation law in porous media multiphase problems
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_PhaseLawConstraint",
        "constraint fluid phase of porous medium", INPAR::MAT::m_fluidporo_phaselaw_constraint));

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // pressure-saturation law defined by functions in porous media multiphase problems
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_PhaseLawByFunction",
        "fluid phase of porous medium defined by functions",
        INPAR::MAT::m_fluidporo_phaselaw_byfunction));

    AddNamedInt(m, "FUNCTPRES", "ID of function for differential pressure", 0);
    AddNamedInt(m, "FUNCTSAT", "ID of function for saturation", 0);
    AddNamedInt(m, "NUMDOF", "number of DoFs", 0);
    AddNamedIntVector(m, "PRESCOEFF", "Coefficients for pressure dependence", "NUMDOF", 0);
    AddNamedSeparator(m, "END", "indicating end of line");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // elastic spring
  {
    auto m = Teuchos::rcp(
        new MaterialDefinition("MAT_Struct_Spring", "elastic spring", INPAR::MAT::m_spring));

    AddNamedReal(m, "STIFFNESS", "spring constant");
    AddNamedReal(m, "DENS", "density");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // materials for beam elements (grill 02/17):

  /* The constitutive laws used in beam formulations are consistently
   * derived from a 3D solid continuum mechanics material law, e.g. a hyperelastic
   * stored energy function. The conceptual difference is that they are
   * formulated for stress and strain resultants, i.e. cross-section quantities.
   * Hence, the constitutive parameters that naturally occur in constitutive
   * relations of beam formulations are strongly related to the cross-section
   * specification (shape and dimensions) and can be identified as 'modal'
   * constitutive parameters (axial/shear/torsion/bending rigidity). See
   * Diss Meier, chapters 2.2.4 and 2.2.5 for formulae and details.
   *
   * This justifies the implementation and use of the following beam material
   * definitions. They combine cross-section specification and material definition
   * which can be done in two distinct ways:
   *
   * 1) by providing individual parameter values for cross-section specs
   *    (area, (polar) area moment of inertia, shear-correction factor, ...) and
   *    material (Young's modulus, Poisson's ratio).
   *
   * 2) by directly providing parameter values for modal constitutive parameters
   *    (axial/shear/torsion/bending rigidity).
   *    This is especially useful if experimentally determined values are used
   *    or artificial scaling of individual modes is desired in tests/debugging.
   *
   * The same logic applies to parameters required to model mass inertia.
   *
   * Reduced formulations such as Kirchhoff and isotropic/torsion-free Kirchhoff
   * beams of course require only a subset of parameters and hence use specific
   * material parameter definitions. Nevertheless, the material relations are
   * general enough such that only one class is used for the material relations of
   *  all types of beam formulations.
   */

  /*--------------------------------------------------------------------*/
  // material parameter definition for a Simo-Reissner type beam element
  {
    auto matdef = Teuchos::rcp(new MaterialDefinition("MAT_BeamReissnerElastHyper",
        "material parameters for a Simo-Reissner type beam element based on "
        "hyperelastic stored energy function",
        INPAR::MAT::m_beam_reissner_elast_hyper));


    AddNamedReal(matdef, "YOUNG", "Young's modulus");

    /* note: we define both of the two following (redundant) parameters to be optional.
     *       upon initialization of the material, we assure that one of them is
     *       properly defined. */
    AddNamedReal(matdef, "SHEARMOD", "shear modulus", -1.0, true);
    AddNamedReal(matdef, "POISSONRATIO", "Poisson's ratio", -1.0, true);

    AddNamedReal(matdef, "DENS", "mass density");

    AddNamedReal(matdef, "CROSSAREA", "cross-section area");
    AddNamedReal(matdef, "SHEARCORR", "shear correction factor");

    AddNamedReal(matdef, "MOMINPOL", "polar/axial area moment of inertia");
    AddNamedReal(matdef, "MOMIN2",
        "area moment of inertia w.r.t. first principal "
        "axis of inertia (i.e. second base vector)");
    AddNamedReal(matdef, "MOMIN3",
        "area moment of inertia w.r.t. second principal "
        "axis of inertia (i.e. third base vector)");
    AddNamedBool(matdef, "FAD", "Does automatic differentiation have to be used", false, true);


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    AddNamedReal(matdef, "INTERACTIONRADIUS",
        "radius of a circular cross-section which "
        "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
        -1.0, true);

    AppendMaterialDefinition(matlist, matdef);
  }
  /*--------------------------------------------------------------------*/
  // material parameter definition for a Simo-Reissner type elasto-plastic beam element
  {
    auto matdef = Teuchos::rcp(new MaterialDefinition("MAT_BeamReissnerElastPlastic",
        "material parameters for a Simo-Reissner type beam element based on "
        "hyperelastic stored energy function",
        INPAR::MAT::m_beam_reissner_elast_plastic));


    AddNamedReal(matdef, "YOUNG", "Young's modulus");

    // optional parameters for plasticity
    AddNamedReal(matdef, "YIELDN", "initial yield stress N", -1.0, true);
    AddNamedReal(matdef, "YIELDM", "initial yield stress M", -1.0, true);
    AddNamedReal(matdef, "ISOHARDN", "isotropic hardening modulus of forces", -1.0, true);
    AddNamedReal(matdef, "ISOHARDM", "isotropic hardening modulus of moments", -1.0, true);
    AddNamedReal(matdef, "TORSIONPLAST",
        "defines whether torsional moment contributes to plasticity", 0, true);

    /* note: we define both of the two following (redundant) parameters to be optional.
     *       upon initialization of the material, we assure that one of them is
     *       properly defined. */
    AddNamedReal(matdef, "SHEARMOD", "shear modulus", -1.0, true);
    AddNamedReal(matdef, "POISSONRATIO", "Poisson's ratio", -1.0, true);

    AddNamedReal(matdef, "DENS", "mass density");

    AddNamedReal(matdef, "CROSSAREA", "cross-section area");
    AddNamedReal(matdef, "SHEARCORR", "shear correction factor");

    AddNamedReal(matdef, "MOMINPOL", "polar/axial area moment of inertia");
    AddNamedReal(matdef, "MOMIN2",
        "area moment of inertia w.r.t. first principal "
        "axis of inertia (i.e. second base vector)");
    AddNamedReal(matdef, "MOMIN3",
        "area moment of inertia w.r.t. second principal "
        "axis of inertia (i.e. third base vector)");
    AddNamedBool(matdef, "FAD", "Does automatic differentiation have to be used", false, true);


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    AddNamedReal(matdef, "INTERACTIONRADIUS",
        "radius of a circular cross-section which "
        "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
        -1.0, true);

    AppendMaterialDefinition(matlist, matdef);
  }
  /*--------------------------------------------------------------------*/
  // material parameter definition for a Simo-Reissner type beam element,
  // specified via 'modal' constitutive parameters (see comment above)
  {
    auto matdef = Teuchos::rcp(new MaterialDefinition("MAT_BeamReissnerElastHyper_ByModes",
        "material parameters for a Simo-Reissner type beam element based on "
        "hyperelastic stored energy function, specified for individual "
        "deformation modes",
        INPAR::MAT::m_beam_reissner_elast_hyper_bymodes));


    AddNamedReal(matdef, "EA", "axial rigidity");
    AddNamedReal(matdef, "GA2", "shear rigidity w.r.t first principal axis of inertia");
    AddNamedReal(matdef, "GA3", "shear rigidity w.r.t second principal axis of inertia");

    AddNamedReal(matdef, "GI_T", "torsional rigidity");
    AddNamedReal(matdef, "EI2",
        "flexural/bending rigidity w.r.t. first principal "
        "axis of inertia");
    AddNamedReal(matdef, "EI3",
        "flexural/bending rigidity w.r.t. second principal "
        "axis of inertia");

    AddNamedReal(matdef, "RhoA", "translational inertia: mass density * cross-section area");

    AddNamedReal(matdef, "MASSMOMINPOL",
        "polar mass moment of inertia, i.e. w.r.t. "
        "rotation around beam axis");
    AddNamedReal(matdef, "MASSMOMIN2",
        "mass moment of inertia w.r.t. first principal "
        "axis of inertia");
    AddNamedReal(matdef, "MASSMOMIN3",
        "mass moment of inertia w.r.t. second principal "
        "axis of inertia");
    AddNamedBool(matdef, "FAD", "Does automatic differentiation have to be used", false, true);


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    AddNamedReal(matdef, "INTERACTIONRADIUS",
        "radius of a circular cross-section which "
        "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
        -1.0, true);

    AppendMaterialDefinition(matlist, matdef);
  }

  /*--------------------------------------------------------------------*/
  // material parameter definition for a Kirchhoff-Love type beam element
  {
    auto matdef = Teuchos::rcp(new MaterialDefinition("MAT_BeamKirchhoffElastHyper",
        "material parameters for a Kirchhoff-Love type beam element based on "
        "hyperelastic stored energy function",
        INPAR::MAT::m_beam_kirchhoff_elast_hyper));


    AddNamedReal(matdef, "YOUNG", "Young's modulus");

    /* note: we define both of the two following (redundant) parameters to be optional.
     *       upon initialization of the material, we assure that one of them is
     *       properly defined. */
    AddNamedReal(matdef, "SHEARMOD", "shear modulus", -1.0, true);
    AddNamedReal(matdef, "POISSONRATIO", "Poisson's ratio", -1.0, true);

    AddNamedReal(matdef, "DENS", "mass density");

    AddNamedReal(matdef, "CROSSAREA", "cross-section area");

    AddNamedReal(matdef, "MOMINPOL", "polar/axial area moment of inertia");
    AddNamedReal(matdef, "MOMIN2",
        "area moment of inertia w.r.t. first principal "
        "axis of inertia (i.e. second base vector)");
    AddNamedReal(matdef, "MOMIN3",
        "area moment of inertia w.r.t. second principal "
        "axis of inertia (i.e. third base vector)");
    AddNamedBool(matdef, "FAD", "Does automatic differentiation have to be used", false, true);


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    AddNamedReal(matdef, "INTERACTIONRADIUS",
        "radius of a circular cross-section which "
        "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
        -1.0, true);

    AppendMaterialDefinition(matlist, matdef);
  }

  /*--------------------------------------------------------------------*/
  // material parameter definition for a Kirchhoff-Love type beam element,
  // specified via 'modal' constitutive parameters (see comment above)
  {
    auto matdef = Teuchos::rcp(new MaterialDefinition("MAT_BeamKirchhoffElastHyper_ByModes",
        "material parameters for a Kirchhoff-Love type beam element based on "
        "hyperelastic stored energy function, specified for individual "
        "deformation modes",
        INPAR::MAT::m_beam_kirchhoff_elast_hyper_bymodes));


    AddNamedReal(matdef, "EA", "axial rigidity");

    AddNamedReal(matdef, "GI_T", "torsional rigidity");
    AddNamedReal(matdef, "EI2",
        "flexural/bending rigidity w.r.t. first principal "
        "axis of inertia");
    AddNamedReal(matdef, "EI3",
        "flexural/bending rigidity w.r.t. second principal "
        "axis of inertia");

    AddNamedReal(matdef, "RhoA", "translational inertia: mass density * cross-section area");

    AddNamedReal(matdef, "MASSMOMINPOL",
        "polar mass moment of inertia, i.e. w.r.t. "
        "rotation around beam axis");
    AddNamedReal(matdef, "MASSMOMIN2",
        "mass moment of inertia w.r.t. first principal "
        "axis of inertia");
    AddNamedReal(matdef, "MASSMOMIN3",
        "mass moment of inertia w.r.t. second principal "
        "axis of inertia");
    AddNamedBool(matdef, "FAD", "Does automatic differentiation have to be used", false, true);


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    AddNamedReal(matdef, "INTERACTIONRADIUS",
        "radius of a circular cross-section which "
        "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
        -1.0, true);

    AppendMaterialDefinition(matlist, matdef);
  }

  /*--------------------------------------------------------------------*/
  // material parameter definition for a torsion-free, isotropic
  // Kirchhoff-Love type beam element
  {
    auto matdef = Teuchos::rcp(new MaterialDefinition("MAT_BeamKirchhoffTorsionFreeElastHyper",
        "material parameters for a torsion-free, isotropic Kirchhoff-Love "
        "type beam element based on hyperelastic stored energy function",
        INPAR::MAT::m_beam_kirchhoff_torsionfree_elast_hyper));


    AddNamedReal(matdef, "YOUNG", "Young's modulus");

    AddNamedReal(matdef, "DENS", "mass density");

    AddNamedReal(matdef, "CROSSAREA", "cross-section area");

    AddNamedReal(matdef, "MOMIN", "area moment of inertia");
    AddNamedBool(matdef, "FAD", "Does automatic differentiation have to be used", false, true);


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    AddNamedReal(matdef, "INTERACTIONRADIUS",
        "radius of a circular cross-section which "
        "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
        -1.0, true);

    AppendMaterialDefinition(matlist, matdef);
  }

  /*--------------------------------------------------------------------*/
  // material parameter definition for a torsion-free, isotropic
  // Kirchhoff-Love type beam element,
  // specified via 'modal' constitutive parameters (see comment above)
  {
    auto matdef =
        Teuchos::rcp(new MaterialDefinition("MAT_BeamKirchhoffTorsionFreeElastHyper_ByModes",
            "material parameters for a torsion-free, isotropic Kirchhoff-Love "
            "type beam element based on hyperelastic stored energy function, "
            "specified for individual deformation modes",
            INPAR::MAT::m_beam_kirchhoff_torsionfree_elast_hyper_bymodes));


    AddNamedReal(matdef, "EA", "axial rigidity");

    AddNamedReal(matdef, "EI", "flexural/bending rigidity");


    AddNamedReal(matdef, "RhoA", "translational inertia: mass density * cross-section area");
    AddNamedBool(matdef, "FAD", "Does automatic differentiation have to be used", false, true);

    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    AddNamedReal(matdef, "INTERACTIONRADIUS",
        "radius of a circular cross-section which "
        "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
        -1.0, true);

    AppendMaterialDefinition(matlist, matdef);
  }

  /*--------------------------------------------------------------------*/
  // material for a crosslinker in a biopolymer simulation
  {
    auto matdef = Teuchos::rcp(new MaterialDefinition(
        "MAT_Crosslinker", "material for a linkage between beams", INPAR::MAT::m_crosslinkermat));

    AddNamedReal(matdef, "MATNUM", "number of beam elasthyper material");
    AddNamedString(matdef, "JOINTTYPE",
        "type of joint: "
        "beam3rline2rigid (default), beam3rline2pin or truss",
        "beam3rline2rigid");
    AddNamedReal(matdef, "LINKINGLENGTH", "distance between the two binding domains of a linker");
    AddNamedReal(matdef, "LINKINGLENGTHTOL",
        "tolerance for linker length in the sense: length +- tolerance");
    AddNamedReal(matdef, "LINKINGANGLE",
        "preferred binding angle enclosed by two filaments' axes in radians");
    AddNamedReal(matdef, "LINKINGANGLETOL",
        "tolerance for preferred binding angle in radians in the sense of: angle +- tolerance");
    AddNamedReal(matdef, "K_ON", "chemical association-rate");
    AddNamedReal(matdef, "K_OFF", "chemical dissociation-rate");

    // optional parameter
    AddNamedReal(
        matdef, "DELTABELLEQ", "deltaD in Bell's equation for force dependent off rate", 0.0, true);
    AddNamedReal(matdef, "NOBONDDISTSPHERE",
        "distance to sphere elements in which no double bonded linker is allowed", 0.0, true);
    AddNamedString(matdef, "TYPE",
        "type of crosslinker: "
        "arbitrary (default), actin, collagen, integrin",
        "arbitrary", true);

    AppendMaterialDefinition(matlist, matdef);
  }

  /*--------------------------------------------------------------------*/
  // 0D Acinar material base
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_0D_MAXWELL_ACINUS", "0D acinar material", INPAR::MAT::m_0d_maxwell_acinus));

    AddNamedReal(m, "Stiffness1", "first stiffness");
    AddNamedReal(m, "Stiffness2", "second stiffness");
    AddNamedReal(m, "Viscosity1", "first viscosity");
    AddNamedReal(m, "Viscosity2", "second viscosity");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // 0D NeoHookean Acinar material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_0D_MAXWELL_ACINUS_NEOHOOKEAN",
        "0D acinar material neohookean", INPAR::MAT::m_0d_maxwell_acinus_neohookean));

    AddNamedReal(m, "Stiffness1", "first stiffness");
    AddNamedReal(m, "Stiffness2", "second stiffness");
    AddNamedReal(m, "Viscosity1", "first viscosity");
    AddNamedReal(m, "Viscosity2", "second viscosity");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // 0D Exponential Acinar material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_0D_MAXWELL_ACINUS_EXPONENTIAL",
        "0D acinar material exponential", INPAR::MAT::m_0d_maxwell_acinus_exponential));

    AddNamedReal(m, "Stiffness1", "first stiffness");
    AddNamedReal(m, "Stiffness2", "second stiffness");
    AddNamedReal(m, "Viscosity1", "first viscosity");
    AddNamedReal(m, "Viscosity2", "second viscosity");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // 0D Exponential Acinar material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_0D_MAXWELL_ACINUS_DOUBLEEXPONENTIAL",
        "0D acinar material doubleexponential", INPAR::MAT::m_0d_maxwell_acinus_doubleexponential));

    AddNamedReal(m, "Stiffness1", "first stiffness");
    AddNamedReal(m, "Stiffness2", "second stiffness");
    AddNamedReal(m, "Viscosity1", "first viscosity");
    AddNamedReal(m, "Viscosity2", "second viscosity");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // 0D Ogden Acinar material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_0D_MAXWELL_ACINUS_OGDEN",
        "0D acinar material ogden", INPAR::MAT::m_0d_maxwell_acinus_ogden));

    AddNamedReal(m, "Stiffness1", "first stiffness");
    AddNamedReal(m, "Stiffness2", "second stiffness");
    AddNamedReal(m, "Viscosity1", "first viscosity");
    AddNamedReal(m, "Viscosity2", "second viscosity");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // O2 hemoglobin saturation material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_0D_O2_HEMOGLOBIN_SATURATION",
        "0D O2 hemoglobin saturation material", INPAR::MAT::m_0d_o2_hemoglobin_saturation));

    AddNamedReal(m, "PerVolumeBlood", "how much of blood satisfies this rule (usually 100ml)");
    AddNamedReal(m, "O2SaturationPerVolBlood",
        "O2 saturation per volume blood (In healthy blood 21.36ml/100ml of blood)");
    AddNamedReal(m, "PressureHalf", "PO2 of 50\% saturated O2 (In healthy blood 26mmHg)");
    AddNamedReal(m, "Power", "Power of the Sigmoidal saturation curve (2.5)");
    AddNamedReal(m, "NumberOfO2PerVO2", "Number of O2 moles per unit volume of O2");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // O2 air saturation material
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_0D_O2_AIR_SATURATION",
        "0D O2 air saturation material", INPAR::MAT::m_0d_o2_air_saturation));

    AddNamedReal(m, "AtmosphericPressure", "The atmospheric pressure");
    AddNamedReal(m, "NumberOfO2PerVO2", "Number of O2 moles per unit volume of O2");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // particle material sph fluid
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_ParticleSPHFluid",
        "particle material for SPH fluid", INPAR::MAT::m_particle_sph_fluid));

    AddNamedReal(m, "INITRADIUS", "initial radius");
    AddNamedReal(m, "INITDENSITY", "initial density");
    AddNamedReal(m, "REFDENSFAC", "reference density factor in equation of state");
    AddNamedReal(m, "EXPONENT", "exponent in equation of state");
    AddNamedReal(m, "BACKGROUNDPRESSURE", "background pressure for transport velocity formulation");
    AddNamedReal(m, "BULK_MODULUS", "bulk modulus");
    AddNamedReal(m, "DYNAMIC_VISCOSITY", "dynamic shear viscosity");
    AddNamedReal(m, "BULK_VISCOSITY", "bulk viscosity");
    AddNamedReal(m, "ARTIFICIAL_VISCOSITY", "artificial viscosity");
    AddNamedReal(m, "INITTEMPERATURE", "initial temperature", 0.0, true);
    AddNamedReal(m, "THERMALCAPACITY", "thermal capacity", 0.0, true);
    AddNamedReal(m, "THERMALCONDUCTIVITY", "thermal conductivity", 0.0, true);
    AddNamedReal(m, "THERMALABSORPTIVITY", "thermal absorptivity", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // particle material sph boundary
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_ParticleSPHBoundary",
        "particle material for SPH boundary", INPAR::MAT::m_particle_sph_boundary));

    AddNamedReal(m, "INITRADIUS", "initial radius");
    AddNamedReal(m, "INITDENSITY", "initial density");
    AddNamedReal(m, "INITTEMPERATURE", "initial temperature", 0.0, true);
    AddNamedReal(m, "THERMALCAPACITY", "thermal capacity", 0.0, true);
    AddNamedReal(m, "THERMALCONDUCTIVITY", "thermal conductivity", 0.0, true);
    AddNamedReal(m, "THERMALABSORPTIVITY", "thermal absorptivity", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // particle material dem
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_ParticleDEM", "particle material for DEM", INPAR::MAT::m_particle_dem));

    AddNamedReal(m, "INITRADIUS", "initial radius of particle");
    AddNamedReal(m, "INITDENSITY", "initial density of particle");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // particle wall material dem
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_ParticleWallDEM", "particle wall material for DEM", INPAR::MAT::m_particle_wall_dem));

    AddNamedReal(m, "FRICT_COEFF_TANG", "friction coefficient for tangential contact", -1.0, true);
    AddNamedReal(m, "FRICT_COEFF_ROLL", "friction coefficient for rolling contact", -1.0, true);
    AddNamedReal(m, "ADHESION_SURFACE_ENERGY", "adhesion surface energy", -1.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // electromagnetic material
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_Electromagnetic", "Electromagnetic material", INPAR::MAT::m_electromagneticmat));

    AddNamedReal(m, "CONDUCTIVITY", "electrical conductivity");
    AddNamedReal(m, "PERMITTIVITY", "Permittivity");
    AddNamedReal(m, "PERMEABILITY", "Permeability");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // active fiber formation for the modeling of living cells
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_ACTIVEFIBER",
        "active fiber formation for the modeling of living cells", INPAR::MAT::m_activefiber));

    AddNamedReal(m, "DENS", "Density");
    AddNamedReal(m, "DECAY", "decay constant of activation signal");
    AddNamedInt(
        m, "IDMATPASSIVE", "number of passive material in input file: MAT IDMATPASSIVE ...");
    AddNamedReal(m, "KFOR", "formation rate parameter kforwards");
    AddNamedReal(m, "KBACK", "dissociation parameter kbackwards");
    AddNamedReal(m, "KVAR", "fiber rate sensitivity");
    AddNamedReal(m, "SIGMAX", "maximum tension exerted by stress fibres");
    AddNamedReal(m, "EPSNULL", "reference strain rate of cross-bridge dynamics law");


    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // General mixture models (used for prestretching and for homogenized constrained mixture models)
  {
    auto m = Teuchos::rcp(
        new MaterialDefinition("MAT_Mixture", "General mixture model", INPAR::MAT::m_mixture));

    AddNamedInt(m, "NUMCONST", "number of mixture constituents");
    AddNamedInt(m, "MATIDMIXTURERULE", "material id of the mixturerule");
    AddNamedIntVector(
        m, "MATIDSCONST", "list material IDs of the mixture constituents", "NUMCONST");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for ElastHyper toolbox
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MIX_Constituent_ElastHyper", "ElastHyper toolbox", INPAR::MAT::mix_elasthyper));

    AddNamedInt(m, "NUMMAT", "number of summands");
    AddNamedIntVector(m, "MATIDS", "list material IDs of the summands", "NUMMAT");
    AddNamedInt(m, "PRESTRESS_STRATEGY",
        "Material id of the prestress strategy (optional, by default no prestretch)", 0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for ElastHyper toolbox with a damage process
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MIX_Constituent_ElastHyper_Damage",
        "ElastHyper toolbox with damage", INPAR::MAT::mix_elasthyper_damage));

    AddNamedInt(m, "NUMMAT", "number of summands");
    AddNamedIntVector(m, "MATIDS", "list material IDs of the membrane summands", "NUMMAT");
    AddNamedInt(m, "PRESTRESS_STRATEGY",
        "Material id of the prestress strategy (optional, by default no prestretch)", 0, true);
    AddNamedInt(m, "DAMAGE_FUNCT",
        "Reference to the function that is a gain for the increase/decrease of the reference mass "
        "density.");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for ElastHyper toolbox with a damage process and a membrane constituent
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MIX_Constituent_ElastHyper_ElastinMembrane",
        "ElastHyper toolbox with damage and 2D membrane material",
        INPAR::MAT::mix_elasthyper_elastin_membrane));

    AddNamedInt(m, "NUMMAT", "number of summands");
    AddNamedIntVector(m, "MATIDS", "list material IDs of the membrane summands", "NUMMAT");
    AddNamedInt(m, "MEMBRANENUMMAT", "number of summands");
    AddNamedIntVector(
        m, "MEMBRANEMATIDS", "list material IDs of the membrane summands", "MEMBRANENUMMAT");
    AddNamedInt(m, "PRESTRESS_STRATEGY",
        "Material id of the prestress strategy (optional, by default no prestretch)", 0, true);
    AddNamedInt(m, "DAMAGE_FUNCT",
        "Reference to the function that is a gain for the increase/decrease of the reference mass "
        "density.");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for solid material
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MIX_Constituent_SolidMaterial", "Solid material", INPAR::MAT::mix_solid_material));

    AddNamedInt(m, "MATID", "ID of the solid material");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Isotropic growth
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MIX_GrowthStrategy_Isotropic", "isotropic growth",
        INPAR::MAT::mix_growth_strategy_isotropic));

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Anisotropic growth
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MIX_GrowthStrategy_Anisotropic",
        "anisotropic growth", INPAR::MAT::mix_growth_strategy_anisotropic));


    AddNamedInt(m, "INIT", "initialization modus for growth direction alignment", 1, true);
    AddNamedInt(m, "FIBER_ID",
        "Id of the fiber to point the growth direction (1 for first fiber, default)", 1, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Extension of all constituents simultaneously -> Growth happens mainly in the direction with the
  // smallest stiffness
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MIX_GrowthStrategy_Stiffness",
        "Extension of all constituents simultaneously", INPAR::MAT::mix_growth_strategy_stiffness));

    AddNamedReal(
        m, "KAPPA", "Penalty parameter for the modified penalty term for incompressibility");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Constant predefined prestretch
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MIX_Prestress_Strategy_Constant",
        "Simple predefined prestress", INPAR::MAT::mix_prestress_strategy_constant));

    AddNamedRealVector(m, "PRESTRETCH", "Definition of the prestretch as a 9x1 vector", 9);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Prestress strategy for a cylinder
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MIX_Prestress_Strategy_Cylinder",
        "Simple prestress strategy for a cylinder", INPAR::MAT::mix_prestress_strategy_cylinder));

    AddNamedReal(m, "INNER_RADIUS", "Inner radius of the cylinder");
    AddNamedReal(m, "WALL_THICKNESS", "Wall thickness of the cylinder");
    AddNamedReal(m, "AXIAL_PRESTRETCH", "Prestretch in axial direction");
    AddNamedReal(m, "CIRCUMFERENTIAL_PRESTRETCH", "Prestretch in circumferential direction");
    AddNamedReal(m, "PRESSURE", "Pressure in the inner of the cylinder");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Iterative prestress strategy for any geometry
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MIX_Prestress_Strategy_Iterative",
        "Simple iterative prestress strategy for any geometry. Needed to be used within the "
        "mixture framework.",
        INPAR::MAT::mix_prestress_strategy_iterative));
    AddNamedBool(m, "ISOCHORIC", "Flag whether prestretch tensor is isochoric", false, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for a remodel fiber
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MIX_Constituent_ExplicitRemodelFiber",
        "A 1D constituent that remodels", INPAR::MAT::mix_remodelfiber_expl));

    AddNamedInt(m, "FIBER_ID", "Id of the fiber", 1, true);
    AddNamedInt(m, "FIBER_MATERIAL_ID", "Id of fiber material");

    AddNamedBool(m, "GROWTH_ENABLED", "Switch for the growth (default true)", true, true);
    AddNamedReal(m, "DECAY_TIME", "Decay time of deposited tissue");
    AddNamedReal(m, "GROWTH_CONSTANT", "Growth constant of the tissue");
    AddNamedReal(m, "DEPOSITION_STRETCH", "Stretch at with the fiber is deposited");
    AddNamedInt(m, "DEPOSITION_STRETCH_TIMEFUNCT",
        "Id of the time function to scale the deposition stretch (Default: 0=None)", 0, true);
    AddNamedBool(
        m, "INELASTIC_GROWTH", "Mixture rule has inelastic growth (default false)", false, true);
    AddNamedInt(m, "INIT", "Initialization mode for fibers (1=element fibers, 2=nodal fibers)");
    AddNamedReal(
        m, "GAMMA", "Angle of fiber alignment in degree (default = 0.0 degrees)", 0.0, true);

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for a remodel fiber
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MIX_Constituent_ImplicitRemodelFiber",
        "A 1D constituent that remodels", INPAR::MAT::mix_remodelfiber_impl));

    AddNamedInt(m, "FIBER_ID", "Id of the fiber");
    AddNamedInt(m, "FIBER_MATERIAL_ID", "Id of fiber material");

    AddNamedBool(m, "GROWTH_ENABLED", "Switch for the growth (default true)", true, true);
    AddNamedReal(m, "DECAY_TIME", "Decay time of deposited tissue");
    AddNamedReal(m, "GROWTH_CONSTANT", "Growth constant of the tissue");
    AddNamedReal(m, "DEPOSITION_STRETCH", "Stretch at with the fiber is deposited");
    AddNamedInt(m, "DEPOSITION_STRETCH_TIMEFUNCT",
        "Id of the time function to scale the deposition stretch (Default: 0=None)", 0, true);
    AddNamedInt(m, "INIT", "Initialization mode for fibers (1=element fibers, 2=nodal fibers)");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent material for a remodel fiber with exponential strain energy function
  {
    auto m =
        Teuchos::rcp(new MaterialDefinition("MIX_Constituent_RemodelFiber_Material_Exponential",
            "An exponential strain energy function for the remodel fiber",
            INPAR::MAT::mix_remodelfiber_material_exponential));


    AddNamedReal(m, "K1", "First parameter of exponential strain energy function");
    AddNamedReal(m, "K2", "Second parameter of exponential strain energy function");
    AddNamedBool(
        m, "COMPRESSION", "Bool, whether the fiber material also supports compressive forces.");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent material for a remodel fiber with exponential strain energy function and an
  // active contribution
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MIX_Constituent_RemodelFiber_Material_Exponential_Active",
        "An exponential strain energy function for the remodel fiber with an active contribution",
        INPAR::MAT::mix_remodelfiber_material_exponential_active));


    AddNamedReal(m, "K1", "First parameter of exponential strain energy function");
    AddNamedReal(m, "K2", "Second parameter of exponential strain energy function");
    AddNamedBool(
        m, "COMPRESSION", "Bool, whether the fiber material also supports compressive forces.");
    AddNamedReal(m, "SIGMA_MAX", "Maximum active Cauchy-stress");
    AddNamedReal(m, "LAMBDAMAX", "Stretch at maximum active Cauchy-stress");
    AddNamedReal(m, "LAMBDA0", "Stretch at zero active Cauchy-stress");
    AddNamedReal(m, "LAMBDAACT", "Current stretch", 1.0, true);
    AddNamedReal(m, "DENS", "Density of the whole mixture");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Base mixture rule for solid mixtures
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MIX_Rule_Simple", "Simple mixture rule", INPAR::MAT::mix_rule_simple));

    AddNamedReal(m, "DENS", "");
    AddNamedInt(m, "NUMCONST", "number of mixture constituents");
    AddNamedRealVector(
        m, "MASSFRAC", "list mass fractions of the mixture constituents", "NUMCONST");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Base mixture rule for solid mixtures
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MIX_GrowthRemodelMixtureRule",
        "Mixture rule for growth/remodel homogenized constrained mixture models",
        INPAR::MAT::mix_rule_growthremodel));

    AddNamedInt(m, "GROWTH_STRATEGY", "Material id of the growth strategy");
    AddNamedReal(m, "DENS", "");
    AddNamedInt(m, "NUMCONST", "number of mixture constituents");
    AddNamedRealVector(
        m, "MASSFRAC", "list mass fractions of the mixture constituents", "NUMCONST");

    AppendMaterialDefinition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // crystal plasticity
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_crystal_plasticity", " Crystal plasticity ", INPAR::MAT::m_crystplast));
    AddNamedReal(m, "TOL", "tolerance for internal Newton iteration");
    AddNamedReal(m, "YOUNG", "Young's modulus");
    AddNamedReal(m, "NUE", "Poisson's ratio");
    AddNamedReal(m, "DENS", "Mass density");
    AddNamedString(m, "LAT", "lattice type: FCC, BCC, HCP, D019 or L10", "FCC");
    AddNamedReal(m, "CTOA", "c to a ratio of crystal unit cell");
    AddNamedReal(m, "ABASE", "base length a of the crystal unit cell");
    AddNamedInt(m, "NUMSLIPSYS", "number of slip systems");
    AddNamedInt(m, "NUMSLIPSETS", "number of slip system sets");
    AddNamedIntVector(m, "SLIPSETMEMBERS",
        "vector of NUMSLIPSYS indices ranging from 1 to NUMSLIPSETS that indicate to which set "
        "each slip system belongs",
        "NUMSLIPSYS");
    AddNamedIntVector(m, "SLIPRATEEXP",
        "vector containing NUMSLIPSETS entries for the rate sensitivity exponent", "NUMSLIPSETS");
    AddNamedRealVector(m, "GAMMADOTSLIPREF",
        "vector containing NUMSLIPSETS entries for the reference slip shear rate", "NUMSLIPSETS");
    AddNamedRealVector(m, "DISDENSINIT",
        "vector containing NUMSLIPSETS entries for the initial dislocation density", "NUMSLIPSETS");
    AddNamedRealVector(m, "DISGENCOEFF",
        "vector containing NUMSLIPSETS entries for the dislocation generation coefficients",
        "NUMSLIPSETS");
    AddNamedRealVector(m, "DISDYNRECCOEFF",
        "vector containing NUMSLIPSETS entries for the coefficients for dynamic dislocation "
        "removal",
        "NUMSLIPSETS");
    AddNamedRealVector(m, "TAUY0",
        "vector containing NUMSLIPSETS entries for the lattice resistance to slip, e.g. the "
        "Peierls barrier",
        "NUMSLIPSETS");
    AddNamedRealVector(m, "MFPSLIP",
        "vector containing NUMSLIPSETS microstructural parameters that are relevant for Hall-Petch "
        "strengthening, e.g., grain size",
        "NUMSLIPSETS");
    AddNamedRealVector(m, "SLIPHPCOEFF",
        "vector containing NUMSLIPSETS entries for the Hall-Petch coefficients corresponding to "
        "the "
        "microstructural parameters given in MFPSLIP",
        "NUMSLIPSETS");
    AddNamedRealVector(m, "SLIPBYTWIN",
        "(optional) vector containing NUMSLIPSETS entries for the work hardening coefficients by "
        "twinning on non-coplanar systems",
        "NUMSLIPSETS", 0., true);
    AddNamedInt(m, "NUMTWINSYS", "(optional) number of twinning systems", 0, true);
    AddNamedInt(m, "NUMTWINSETS", "(optional) number of sets of twinning systems", 0, true);
    AddNamedIntVector(m, "TWINSETMEMBERS",
        "(optional) vector of NUMTWINSYS indices ranging from 1 to NUMTWINSETS that indicate to "
        "which set each slip system belongs",
        "NUMTWINSYS", 0, true);
    AddNamedIntVector(m, "TWINRATEEXP",
        "(optional) vector containing NUMTWINSETS entries for the rate sensitivity exponent",
        "NUMTWINSETS", 0, true);
    AddNamedRealVector(m, "GAMMADOTTWINREF",
        "(optional) vector containing NUMTWINSETS entries for the reference slip shear rate",
        "NUMTWINSETS", 0., true);
    AddNamedRealVector(m, "TAUT0",
        "(optional) vector containing NUMTWINSETS entries for the lattice resistance to twinning, "
        "e.g. the Peierls "
        "barrier",
        "NUMTWINSETS", 0., true);
    AddNamedRealVector(m, "MFPTWIN",
        "(optional) vector containing NUMTWINSETS microstructural parameters that are relevant for "
        "Hall-Petch "
        "strengthening of twins, e.g., grain size",
        "NUMTWINSETS", 0., true);
    AddNamedRealVector(m, "TWINHPCOEFF",
        "(optional) vector containing NUMTWINSETS entries for the Hall-Petch coefficients "
        "corresponding to the "
        "microstructural parameters given in MFPTWIN",
        "NUMTWINSETS", 0., true);
    AddNamedRealVector(m, "TWINBYSLIP",
        "(optional) vector containing NUMTWINSETS entries for the work hardening coefficients by "
        "slip",
        "NUMTWINSETS", 0., true);
    AddNamedRealVector(m, "TWINBYTWIN",
        "(optional) vector containing NUMTWINSETS entries for the work hardening coefficients by "
        "twins on non-coplanar systems",
        "NUMTWINSETS", 0., true);
    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // linear elastic material in one direction
  {
    auto m = Teuchos::rcp(new MaterialDefinition(
        "MAT_LinElast1D", "linear elastic material in one direction", INPAR::MAT::m_linelast1D));

    AddNamedReal(m, "YOUNG", "Young's modulus");
    AddNamedReal(m, "DENS", "mass density");

    AppendMaterialDefinition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // linear elastic material with growth in one direction
  {
    auto m = Teuchos::rcp(new MaterialDefinition("MAT_LinElast1DGrowth",
        "linear elastic material with growth in one direction", INPAR::MAT::m_linelast1D_growth));

    AddNamedReal(m, "YOUNG", "Young's modulus");
    AddNamedReal(m, "DENS", "mass density");
    AddNamedReal(m, "C0", "reference concentration");
    AddNamedBool(m, "AOS_PROP_GROWTH",
        "growth proportional to amount of substance (AOS) if true or proportional to concentration "
        "if false");
    AddNamedInt(m, "POLY_PARA_NUM", "number of polynomial coefficients");
    AddNamedRealVector(m, "POLY_PARAMS", "coefficients of polynomial", "POLY_PARA_NUM");

    AppendMaterialDefinition(matlist, m);
  }

  // deliver
  return vm;
}
