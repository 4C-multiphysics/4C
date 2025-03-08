// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_global_legacy_module_validmaterials.hpp"

#include "4C_global_data.hpp"
#include "4C_inpar_material.hpp"
#include "4C_io_control.hpp"
#include "4C_io_file_reader.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_electrode.hpp"
#include "4C_mat_materialdefinition.hpp"

#include <filesystem>
#include <string>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<std::vector<std::shared_ptr<Mat::MaterialDefinition>>> Global::valid_materials()
{
  using namespace Core::IO::InputSpecBuilders;

  // a list containing all valid materials
  std::shared_ptr<std::vector<std::shared_ptr<Mat::MaterialDefinition>>> vm =
      std::make_shared<std::vector<std::shared_ptr<Mat::MaterialDefinition>>>();

  // convenience
  std::vector<std::shared_ptr<Mat::MaterialDefinition>>& matlist = *vm;


  /*----------------------------------------------------------------------*/
  // Newtonian fluid
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_fluid", "Newtonian fluid", Core::Materials::m_fluid);

    m->add_component(parameter<double>("DYNVISCOSITY", {.description = "dynamic viscosity"}));
    m->add_component(parameter<double>("DENSITY", {.description = "spatial mass density"}));
    m->add_component(parameter<double>(
        "GAMMA", {.description = "surface tension coefficient", .default_value = 0.0}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Weakly compressible fluid according to Murnaghan-Tait
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_fluid_murnaghantait",
        "Weakly compressible fluid according to Murnaghan-Tait",
        Core::Materials::m_fluid_murnaghantait);

    m->add_component(parameter<double>("DYNVISCOSITY", {.description = "dynamic viscosity"}));
    m->add_component(
        parameter<double>("REFDENSITY", {.description = "reference spatial mass density"}));
    m->add_component(parameter<double>("REFPRESSURE", {.description = "reference pressure"}));
    m->add_component(
        parameter<double>("REFBULKMODULUS", {.description = "reference bulk modulus"}));
    m->add_component(parameter<double>(
        "MATPARAMETER", {.description = "material parameter according to Murnaghan-Tait"}));
    m->add_component(parameter<double>(
        "GAMMA", {.description = "surface tension coefficient", .default_value = 0.0}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Linear law (pressure-dependent) for the density and the viscosity
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_fluid_linear_density_viscosity",
        "Linear law (pressure-dependent) for the density and the viscosity",
        Core::Materials::m_fluid_linear_density_viscosity);

    m->add_component(parameter<double>("REFDENSITY", {.description = "reference density"}));
    m->add_component(parameter<double>("REFVISCOSITY", {.description = "reference viscosity"}));
    m->add_component(parameter<double>("REFPRESSURE", {.description = "reference pressure"}));
    m->add_component(
        parameter<double>("COEFFDENSITY", {.description = "density-pressure coefficient"}));
    m->add_component(
        parameter<double>("COEFFVISCOSITY", {.description = "viscosity-pressure coefficient"}));
    m->add_component(parameter<double>(
        "GAMMA", {.description = "surface tension coefficient", .default_value = 0.0}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Weakly compressible fluid
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_fluid_weakly_compressible",
        "Weakly compressible fluid", Core::Materials::m_fluid_weakly_compressible);

    m->add_component(parameter<double>("VISCOSITY", {.description = "viscosity"}));
    m->add_component(parameter<double>("REFDENSITY", {.description = "reference density"}));
    m->add_component(parameter<double>("REFPRESSURE", {.description = "reference pressure"}));
    m->add_component(
        parameter<double>("COMPRCOEFF", {.description = "compressibility coefficient"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // fluid with non-linear viscosity according to Carreau-Yasuda
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_carreauyasuda",
        "fluid with non-linear viscosity according to Carreau-Yasuda",
        Core::Materials::m_carreauyasuda);

    m->add_component(parameter<double>("NU_0", {.description = "zero-shear viscosity"}));
    m->add_component(parameter<double>("NU_INF", {.description = "infinite-shear viscosity"}));
    m->add_component(parameter<double>("LAMBDA", {.description = "characteristic time"}));
    m->add_component(parameter<double>("APARAM", {.description = "constant parameter"}));
    m->add_component(parameter<double>("BPARAM", {.description = "constant parameter"}));
    m->add_component(parameter<double>("DENSITY", {.description = "density"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // fluid with nonlinear viscosity according to a modified power law
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_modpowerlaw",
        "fluid with nonlinear viscosity according to a modified power law",
        Core::Materials::m_modpowerlaw);

    m->add_component(parameter<double>("MCONS", {.description = "consistency"}));
    m->add_component(parameter<double>("DELTA", {.description = "safety factor"}));
    m->add_component(parameter<double>("AEXP", {.description = "exponent"}));
    m->add_component(parameter<double>("DENSITY", {.description = "density"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // fluid with non-linear viscosity according to Herschel-Bulkley
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_herschelbulkley",
        "fluid with non-linear viscosity according to Herschel-Bulkley",
        Core::Materials::m_herschelbulkley);

    m->add_component(parameter<double>("TAU_0", {.description = "yield stress"}));
    m->add_component(parameter<double>("KFAC", {.description = "constant factor"}));
    m->add_component(parameter<double>("NEXP", {.description = "exponent"}));
    m->add_component(parameter<double>("MEXP", {.description = "exponent"}));
    m->add_component(
        parameter<double>("LOLIMSHEARRATE", {.description = "lower limit of shear rate"}));
    m->add_component(
        parameter<double>("UPLIMSHEARRATE", {.description = "upper limit of shear rate"}));
    m->add_component(parameter<double>("DENSITY", {.description = "density"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // lubrication material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_lubrication", "lubrication material", Core::Materials::m_lubrication);

    m->add_component(parameter<int>("LUBRICATIONLAWID", {.description = "lubrication law id"}));
    m->add_component(parameter<double>("DENSITY", {.description = "lubricant density"}));

    Mat::append_material_definition(matlist, m);
  }


  /*----------------------------------------------------------------------*/
  // constant lubrication material law
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_lubrication_law_constant",
        "constant lubrication material law", Core::Materials::m_lubrication_law_constant);

    m->add_component(parameter<double>("VISCOSITY", {.description = "lubricant viscosity"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Barus viscosity lubrication material law
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_lubrication_law_barus",
        "barus lubrication material law", Core::Materials::m_lubrication_law_barus);

    m->add_component(
        parameter<double>("ABSViscosity", {.description = "absolute lubricant viscosity"}));
    m->add_component(
        parameter<double>("PreVisCoeff", {.description = "pressure viscosity coefficient"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Roeland viscosity lubrication material law
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_lubrication_law_roeland",
        "roeland lubrication material law", Core::Materials::m_lubrication_law_roeland);

    m->add_component(
        parameter<double>("ABSViscosity", {.description = "absolute lubricant viscosity"}));
    m->add_component(
        parameter<double>("PreVisCoeff", {.description = "pressure viscosity coefficient"}));
    m->add_component(parameter<double>("RefVisc", {.description = "reference viscosity"}));
    m->add_component(parameter<double>("RefPress", {.description = "reference Pressure"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport material (with potential reaction coefficient)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_scatra", "scalar transport material", Core::Materials::m_scatra);

    m->add_component(parameter<double>("DIFFUSIVITY", {.description = "kinematic diffusivity"}));
    m->add_component(parameter<double>(
        "REACOEFF", {.description = "reaction coefficient", .default_value = 0.0}));
    m->add_component(
        parameter<double>("SCNUM", {.description = "schmidt number", .default_value = 0.0}));
    m->add_component(parameter<double>(
        "DENSIFICATION", {.description = "densification coefficient", .default_value = 0.0}));
    m->add_component(parameter<bool>("REACTS_TO_EXTERNAL_FORCE",
        {.description = "reacts to external force", .default_value = false}));

    Mat::append_material_definition(matlist, m);
  }


  /*----------------------------------------------------------------------*/
  // scalar transport material (with potential reaction coefficient)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_scatra_reaction_poro",
        "scalar transport material", Core::Materials::m_scatra_reaction_poroECM);

    m->add_component(
        parameter<int>("NUMSCAL", {.description = "number of scalars for these elements"}));
    m->add_component(parameter<std::vector<int>>("STOICH",
        {.description = "reaction stoichometrie list", .size = from_parameter<int>("NUMSCAL")}));
    m->add_component(parameter<double>("REACCOEFF", {.description = "reaction coefficient"}));
    m->add_component(
        parameter<double>("REACSCALE", {.description = "scaling for reaction coefficient"}));
    // reacscale could now be done by constant distribution function
    m->add_component(parameter<int>("DISTRFUNCT",
        {.description = "spatial distribution of reaction coefficient", .default_value = 0}));
    m->add_component(parameter<std::string>("COUPLING",
        {.description = "type of coupling: simple_multiplicative, power_multiplicative, "
                        "constant, michaelis_menten, by_function, no_coupling (default)",
            .default_value = "no_coupling"}));
    m->add_component(parameter<std::vector<double>>(
        "ROLE", {.description = "role in michaelis-menten like reactions",
                    .size = from_parameter<int>("NUMSCAL")}));
    m->add_component(parameter<std::optional<std::vector<double>>>("REACSTART",
        {.description = "starting point of reaction", .size = from_parameter<int>("NUMSCAL")}));

    Mat::append_material_definition(matlist, m);
  }
  /*----------------------------------------------------------------------*/
  // scalar transport reaction material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_scatra_reaction", "advanced reaction material", Core::Materials::m_scatra_reaction);

    m->add_component(
        parameter<int>("NUMSCAL", {.description = "number of scalars for these elements"}));
    m->add_component(parameter<std::vector<int>>("STOICH",
        {.description = "reaction stoichometrie list", .size = from_parameter<int>("NUMSCAL")}));
    m->add_component(parameter<double>("REACCOEFF", {.description = "reaction coefficient"}));
    m->add_component(parameter<int>("DISTRFUNCT",
        {.description = "spatial distribution of reaction coefficient", .default_value = 0}));
    m->add_component(parameter<std::string>("COUPLING",
        {.description = "type of coupling: simple_multiplicative, power_multiplicative, "
                        "constant, michaelis_menten, by_function, no_coupling (default)",
            .default_value = "no_coupling"}));
    m->add_component(parameter<std::vector<double>>(
        "ROLE", {.description = "role in michaelis-menten like reactions",
                    .size = from_parameter<int>("NUMSCAL")}));
    m->add_component(parameter<std::optional<std::vector<double>>>("REACSTART",
        {.description = "starting point of reaction", .size = from_parameter<int>("NUMSCAL")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport reaction material (species in fluid)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_scatra_multiporo_fluid",
        "advanced reaction material for multiphase porous flow (species in fluid)",
        Core::Materials::m_scatra_multiporo_fluid);

    m->add_component(parameter<double>("DIFFUSIVITY", {.description = "kinematic diffusivity"}));
    m->add_component(parameter<int>(
        "PHASEID", {.description = "ID of fluid phase the scalar is associated with"}));
    m->add_component(parameter<double>(
        "REACOEFF", {.description = "reaction coefficient", .default_value = 0.0}));
    m->add_component(
        parameter<double>("SCNUM", {.description = "schmidt number", .default_value = 0.0}));
    m->add_component(parameter<double>(
        "DENSIFICATION", {.description = "densification coefficient", .default_value = 0.0}));
    m->add_component(parameter<double>("DELTA", {.description = "delta", .default_value = 0.0}));
    m->add_component(parameter<double>("MIN_SAT",
        {.description =
                "minimum saturation under which also corresponding mass fraction is equal to zero",
            .default_value = 1.0e-9}));
    m->add_component(parameter<bool>("REACTS_TO_EXTERNAL_FORCE",
        {.description = "reacts to external force", .default_value = false}));
    m->add_component(parameter<int>("RELATIVE_MOBILITY_FUNCTION_ID",
        {.description = "relative mobility function ID", .default_value = 0}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport reaction material (species in volume fraction)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_scatra_multiporo_volfrac",
        "advanced reaction material for multiphase porous flow (species in volfrac)",
        Core::Materials::m_scatra_multiporo_volfrac);

    m->add_component(parameter<double>("DIFFUSIVITY", {.description = "kinematic diffusivity"}));
    m->add_component(parameter<int>(
        "PHASEID", {.description = "ID of fluid phase the scalar is associated with"}));
    m->add_component(parameter<double>(
        "REACOEFF", {.description = "reaction coefficient", .default_value = 0.0}));
    m->add_component(
        parameter<double>("SCNUM", {.description = "schmidt number", .default_value = 0.0}));
    m->add_component(parameter<double>(
        "DENSIFICATION", {.description = "densification coefficient", .default_value = 0.0}));
    m->add_component(parameter<double>("DELTA", {.description = "delta", .default_value = 0.0}));
    m->add_component(parameter<bool>("REACTS_TO_EXTERNAL_FORCE",
        {.description = "reacts to external force", .default_value = false}));
    m->add_component(parameter<int>("RELATIVE_MOBILITY_FUNCTION_ID",
        {.description = "relative mobility function ID", .default_value = 0}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport reaction material (species in solid)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_scatra_multiporo_solid",
        "advanced reaction material for multiphase "
        "porous flow (species in solid)",
        Core::Materials::m_scatra_multiporo_solid);

    m->add_component(parameter<double>("DIFFUSIVITY", {.description = "kinematic diffusivity"}));
    // no phaseID because only one solid phase
    m->add_component(parameter<double>(
        "REACOEFF", {.description = "reaction coefficient", .default_value = 0.0}));
    m->add_component(
        parameter<double>("SCNUM", {.description = "schmidt number", .default_value = 0.0}));
    m->add_component(parameter<double>(
        "DENSIFICATION", {.description = "densification coefficient", .default_value = 0.0}));
    m->add_component(parameter<double>("DELTA", {.description = "delta", .default_value = 0.0}));
    m->add_component(parameter<bool>("REACTS_TO_EXTERNAL_FORCE",
        {.description = "reacts to external force", .default_value = false}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport reaction material (temperature)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_scatra_multiporo_temperature",
        "advanced reaction material for multiphase porous flow (temperature)",
        Core::Materials::m_scatra_multiporo_temperature);

    m->add_component(parameter<int>(
        "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE", {.description = "number of fluid dofs"}));
    m->add_component(parameter<std::vector<double>>(
        "CP_FLUID", {.description = "heat capacity fluid phases",
                        .size = from_parameter<int>("NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE")}));
    m->add_component(parameter<int>("NUMVOLFRAC", {.description = "number of volfrac dofs"}));
    m->add_component(parameter<std::vector<double>>("CP_VOLFRAC",
        {.description = "heat capacity volfrac", .size = from_parameter<int>("NUMVOLFRAC")}));
    m->add_component(parameter<double>("CP_SOLID", {.description = "heat capacity solid"}));
    m->add_component(parameter<std::vector<double>>(
        "KAPPA_FLUID", {.description = "thermal diffusivity fluid phases",
                           .size = from_parameter<int>("NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE")}));
    m->add_component(parameter<std::vector<double>>("KAPPA_VOLFRAC",
        {.description = "thermal diffusivity volfrac", .size = from_parameter<int>("NUMVOLFRAC")}));
    m->add_component(parameter<double>("KAPPA_SOLID", {.description = "heat capacity solid"}));
    m->add_component(parameter<double>(
        "DIFFUSIVITY", {.description = "kinematic diffusivity", .default_value = 1.0}));
    m->add_component(parameter<double>(
        "REACOEFF", {.description = "reaction coefficient", .default_value = 0.0}));
    m->add_component(
        parameter<double>("SCNUM", {.description = "schmidt number", .default_value = 0.0}));
    m->add_component(parameter<double>(
        "DENSIFICATION", {.description = "densification coefficient", .default_value = 0.0}));
    m->add_component(parameter<bool>("REACTS_TO_EXTERNAL_FORCE",
        {.description = "reacts to external force", .default_value = false}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport chemotaxis material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_scatra_chemotaxis", "chemotaxis material", Core::Materials::m_scatra_chemotaxis);

    m->add_component(parameter<int>(
        "NUMSCAL", {.description = "number of chemotactic pairs for these elements"}));
    m->add_component(parameter<std::vector<int>>(
        "PAIR", {.description = "chemotaxis pairing", .size = from_parameter<int>("NUMSCAL")}));
    m->add_component(parameter<double>("CHEMOCOEFF", {.description = "chemotaxis coefficient"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport material for multi-scale approach
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_scatra_multiscale",
        "scalar transport material for multi-scale approach", Core::Materials::m_scatra_multiscale);

    m->add_component(parameter<std::string>("MICROFILE",
        {.description = "input file for micro scale", .default_value = "filename.dat"}));
    m->add_component(
        parameter<int>("MICRODIS_NUM", {.description = "number of micro-scale discretization"}));
    m->add_component(parameter<double>("POROSITY", {.description = "porosity"}));
    m->add_component(parameter<double>("TORTUOSITY", {.description = "tortuosity"}));
    m->add_component(
        parameter<double>("A_s", {.description = "specific micro-scale surface area"}));
    m->add_component(parameter<double>("DIFFUSIVITY", {.description = "kinematic diffusivity"}));
    m->add_component(parameter<double>(
        "REACOEFF", {.description = "reaction coefficient", .default_value = 0.0}));
    m->add_component(
        parameter<double>("SCNUM", {.description = "Schmidt number", .default_value = 0.0}));
    m->add_component(parameter<double>(
        "DENSIFICATION", {.description = "densification coefficient", .default_value = 0.0}));
    m->add_component(parameter<bool>("REACTS_TO_EXTERNAL_FORCE",
        {.description = "reacts to external force", .default_value = false}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Weickenmeier muscle material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Muscle_Weickenmeier",
        "Weickenmeier muscle material", Core::Materials::m_muscle_weickenmeier);

    m->add_component(
        parameter<double>("ALPHA", {.description = "experimentally fitted material parameter"}));
    m->add_component(
        parameter<double>("BETA", {.description = "experimentally fitted material parameter"}));
    m->add_component(
        parameter<double>("GAMMA", {.description = "experimentally fitted material parameter"}));
    m->add_component(parameter<double>(
        "KAPPA", {.description = "material parameter for coupled volumetric contribution"}));
    m->add_component(parameter<double>(
        "OMEGA0", {.description = "weighting factor for isotropic tissue constituents"}));
    m->add_component(parameter<double>("ACTMUNUM",
        {.description =
                "number of active motor units per undeformed muscle cross-sectional area"}));
    m->add_component(parameter<int>("MUTYPESNUM", {.description = "number of motor unit types"}));
    m->add_component(parameter<std::vector<double>>("INTERSTIM",
        {.description = "interstimulus interval", .size = from_parameter<int>("MUTYPESNUM")}));
    m->add_component(parameter<std::vector<double>>("FRACACTMU",
        {.description = "fraction of motor unit type", .size = from_parameter<int>("MUTYPESNUM")}));
    m->add_component(
        parameter<std::vector<double>>("FTWITCH", {.description = "twitch force of motor unit type",
                                                      .size = from_parameter<int>("MUTYPESNUM")}));
    m->add_component(parameter<std::vector<double>>(
        "TTWITCH", {.description = "twitch contraction time of motor unit type",
                       .size = from_parameter<int>("MUTYPESNUM")}));
    m->add_component(
        parameter<double>("LAMBDAMIN", {.description = "minimal active fiber stretch"}));
    m->add_component(parameter<double>("LAMBDAOPT",
        {.description = "optimal active fiber stretch related to active nominal stress maximum"}));
    m->add_component(parameter<double>("DOTLAMBDAMIN", {.description = "minimal stretch rate"}));
    m->add_component(parameter<double>(
        "KE", {.description = "parameter controlling the curvature of the velocity "
                              "dependent activation function in the "
                              "eccentric case"}));
    m->add_component(parameter<double>(
        "KC", {.description = "parameter controlling the curvature of the velocity "
                              "dependent activation function in the "
                              "concentric case"}));
    m->add_component(parameter<double>(
        "DE", {.description = "parameter controlling the amplitude of the velocity "
                              "dependent activation function in the "
                              "eccentric case"}));
    m->add_component(parameter<double>(
        "DC", {.description = "parameter controlling the amplitude of the velocity "
                              "dependent activation function in the "
                              "concentric case"}));
    m->add_component(parameter<int>(
        "ACTTIMESNUM", {.description = "number of time boundaries to prescribe activation"}));
    m->add_component(parameter<std::vector<double>>(
        "ACTTIMES", {.description = "time boundaries between intervals",
                        .size = from_parameter<int>("ACTTIMESNUM")}));
    m->add_component(parameter<int>(
        "ACTINTERVALSNUM", {.description = "number of time intervals to prescribe activation"}));
    m->add_component(parameter<std::vector<double>>("ACTVALUES",
        {.description = "scaling factor in intervals (1=full activation, 0=no activation)",
            .size = from_parameter<int>("ACTINTERVALSNUM")}));
    m->add_component(parameter<double>("DENS", {.description = "density"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Combo muscle material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_Muscle_Combo", "Combo muscle material", Core::Materials::m_muscle_combo);

    m->add_component(
        parameter<double>("ALPHA", {.description = "experimentally fitted material parameter"}));
    m->add_component(
        parameter<double>("BETA", {.description = "experimentally fitted material parameter"}));
    m->add_component(
        parameter<double>("GAMMA", {.description = "experimentally fitted material parameter"}));
    m->add_component(parameter<double>(
        "KAPPA", {.description = "material parameter for coupled volumetric contribution"}));
    m->add_component(parameter<double>(
        "OMEGA0", {.description = "weighting factor for isotropic tissue constituents"}));
    m->add_component(
        parameter<double>("POPT", {.description = "tetanised optimal (maximal) active stress"}));
    m->add_component(
        parameter<double>("LAMBDAMIN", {.description = "minimal active fiber stretch"}));
    m->add_component(parameter<double>("LAMBDAOPT",
        {.description = "optimal active fiber stretch related to active nominal stress maximum"}));

    m->add_component(deprecated_selection<Inpar::Mat::ActivationType>("ACTEVALTYPE",
        {{"function", Inpar::Mat::ActivationType::function_of_space_time},
            {"map", Inpar::Mat::ActivationType::map}},
        {.description = "type of activation evaluation"}));

    using actMapType = std::unordered_map<int, std::vector<std::pair<double, double>>>;

    auto on_parse = [](Core::IO::InputParameterContainer& container)
    {
      const auto& map_file = container.get<std::filesystem::path>("MAPFILE");
      std::ifstream file_stream(map_file);

      if (file_stream.fail()) FOUR_C_THROW("Invalid file %s!", map_file.string().c_str());

      auto map_reduction_operation = [](actMapType acc, const actMapType& next)
      {
        for (const auto& [key, value] : next)
        {
          acc[key] = value;
        }
        return acc;
      };

      container.add("MAPFILE_CONTENT",
          Core::IO::convert_lines<actMapType, actMapType>(file_stream, map_reduction_operation));
    };

    auto activation = one_of({
        parameter<int>("FUNCTID",
            {.description = "function id for time- and space-dependency of muscle activation"}),
        parameter<std::filesystem::path>("MAPFILE",
            {.description = "pattern file containing a map of elementwise-defined discrete values "
                            "for time- and space-dependency of muscle activation",
                .on_parse_callback = on_parse}),
    });
    m->add_component(std::move(activation));

    m->add_component(parameter<double>("DENS", {.description = "density"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Active strain Giantesio muscle material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Muscle_Giantesio",
        "Giantesio active strain muscle material", Core::Materials::m_muscle_giantesio);

    m->add_component(
        parameter<double>("ALPHA", {.description = "experimentally fitted material parameter"}));
    m->add_component(
        parameter<double>("BETA", {.description = "experimentally fitted material parameter"}));
    m->add_component(
        parameter<double>("GAMMA", {.description = "experimentally fitted material parameter"}));
    m->add_component(parameter<double>(
        "KAPPA", {.description = "material parameter for coupled volumetric contribution"}));
    m->add_component(parameter<double>(
        "OMEGA0", {.description = "weighting factor for isotropic tissue constituents"}));
    m->add_component(parameter<double>("ACTMUNUM",
        {.description =
                "number of active motor units per undeformed muscle cross-sectional area"}));
    m->add_component(parameter<int>("MUTYPESNUM", {.description = "number of motor unit types"}));
    m->add_component(parameter<std::vector<double>>("INTERSTIM",
        {.description = "interstimulus interval", .size = from_parameter<int>("MUTYPESNUM")}));
    m->add_component(parameter<std::vector<double>>("FRACACTMU",
        {.description = "fraction of motor unit type", .size = from_parameter<int>("MUTYPESNUM")}));
    m->add_component(
        parameter<std::vector<double>>("FTWITCH", {.description = "twitch force of motor unit type",
                                                      .size = from_parameter<int>("MUTYPESNUM")}));
    m->add_component(parameter<std::vector<double>>(
        "TTWITCH", {.description = "twitch contraction time of motor unit type",
                       .size = from_parameter<int>("MUTYPESNUM")}));
    m->add_component(
        parameter<double>("LAMBDAMIN", {.description = "minimal active fiber stretch"}));
    m->add_component(parameter<double>("LAMBDAOPT",
        {.description = "optimal active fiber stretch related to active nominal stress maximum"}));
    m->add_component(parameter<double>("DOTLAMBDAMIN", {.description = "minimal stretch rate"}));
    m->add_component(parameter<double>(
        "KE", {.description = "parameter controlling the curvature of the velocity "
                              "dependent activation function in the "
                              "eccentric case"}));
    m->add_component(parameter<double>(
        "KC", {.description = "parameter controlling the curvature of the velocity "
                              "dependent activation function in the "
                              "concentric case"}));
    m->add_component(parameter<double>(
        "DE", {.description = "parameter controlling the amplitude of the velocity "
                              "dependent activation function in the "
                              "eccentric case"}));
    m->add_component(parameter<double>(
        "DC", {.description = "parameter controlling the amplitude of the velocity "
                              "dependent activation function in the "
                              "concentric case"}));
    m->add_component(parameter<int>(
        "ACTTIMESNUM", {.description = "number of time boundaries to prescribe activation"}));
    m->add_component(parameter<std::vector<double>>(
        "ACTTIMES", {.description = "time boundaries between intervals",
                        .size = from_parameter<int>("ACTTIMESNUM")}));
    m->add_component(parameter<int>(
        "ACTINTERVALSNUM", {.description = "number of time intervals to prescribe activation"}));
    m->add_component(parameter<std::vector<double>>("ACTVALUES",
        {.description = "scaling factor in intervals (1=full activation, 0=no activation)",
            .size = from_parameter<int>("ACTINTERVALSNUM")}));
    m->add_component(parameter<double>("DENS", {.description = "density"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Myocard muscle material (with complicated reaction coefficient)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_myocard", "Myocard muscle material", Core::Materials::m_myocard);

    m->add_component(
        parameter<double>("DIFF1", {.description = "conductivity in fiber direction"}));
    m->add_component(parameter<double>(
        "DIFF2", {.description = "conductivity perpendicular to fiber direction"}));
    m->add_component(parameter<double>(
        "DIFF3", {.description = "conductivity perpendicular to fiber direction"}));
    m->add_component(parameter<double>("PERTURBATION_DERIV",
        {.description = "perturbation for calculation of reaction coefficient derivative"}));
    m->add_component(parameter<std::string>(
        "MODEL", {.description = "Model type: MV (default), FHN, TNNP, SAN or INADA",
                     .default_value = "MV"}));
    m->add_component(parameter<std::string>("TISSUE",
        {.description = "Tissue type: M (default), ENDO, EPI, AN, N or NH", .default_value = "M"}));
    m->add_component(
        parameter<double>("TIME_SCALE", {.description = "Scale factor for time units of Model"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material according to Sutherland law
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_sutherland", "material according to Sutherland law", Core::Materials::m_sutherland);

    m->add_component(
        parameter<double>("REFVISC", {.description = "reference dynamic viscosity (kg/(m*s))"}));
    m->add_component(parameter<double>("REFTEMP", {.description = "reference temperature (K)"}));
    m->add_component(parameter<double>("SUTHTEMP", {.description = "Sutherland temperature (K)"}));
    m->add_component(parameter<double>(
        "SHC", {.description = "specific heat capacity at constant pressure (J/(kg*K))"}));
    m->add_component(parameter<double>("PRANUM", {.description = "Prandtl number"}));
    m->add_component(parameter<double>(
        "THERMPRESS", {.description = "(initial) thermodynamic pressure (J/m^3)"}));
    m->add_component(
        parameter<double>("GASCON", {.description = "specific gas constant R (J/(kg*K))"}));

    Mat::append_material_definition(matlist, m);
  }


  /*----------------------------------------------------------------------*/
  // material parameters for ion species in electrolyte solution (gjb 07/08)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_ion",
        "material parameters for ion species in electrolyte solution", Core::Materials::m_ion);

    m->add_component(parameter<double>("DIFFUSIVITY", {.description = "kinematic diffusivity"}));
    m->add_component(parameter<double>("VALENCE", {.description = "valence (= charge number)"}));
    m->add_component(parameter<double>(
        "DENSIFICATION", {.description = "densification coefficient", .default_value = 0.0}));
    // via these two optional parameters we can bring the material parameters
    // of one eliminated ionic species into 4C if needed
    m->add_component(parameter<double>("ELIM_DIFFUSIVITY",
        {.description = "kinematic diffusivity of elim. species", .default_value = 0.0}));
    m->add_component(parameter<double>(
        "ELIM_VALENCE", {.description = "valence of elim. species", .default_value = 0.0}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material parameters for ion species in electrolyte solution (ehrl 07/12)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_newman",
        "material parameters for ion species in electrolyte solution", Core::Materials::m_newman);

    m->add_component(parameter<double>("VALENCE", {.description = "valence (= charge number)"}));
    m->add_component(parameter<int>("DIFF_COEF_CONC_DEP_FUNCT",
        {.description = "function number of function describing concentration dependence of "
                        "diffusion coefficient"}));
    m->add_component(parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT",
        {.description = "FUNCT number describing temperature scaling of diffusion coefficient"}));
    m->add_component(
        parameter<int>("TRANSNR", {.description = "curve number for transference number"}));
    m->add_component(
        parameter<int>("THERMFAC", {.description = "curve number for thermodynamic factor"}));
    m->add_component(parameter<int>(
        "COND_CONC_DEP_FUNCT", {.description = "function number of function describing "
                                               "concentration dependence of conductivity"}));
    m->add_component(parameter<int>("COND_TEMP_SCALE_FUNCT",
        {.description = "FUNCT number describing temperature scaling of conductivity"}));
    // optional parameter for implemented concentration depending function
    m->add_component(parameter<int>("DIFF_PARA_NUM",
        {.description = "number of parameters for diffusion coefficient", .default_value = 0}));
    m->add_component(parameter<std::vector<double>>(
        "DIFF_PARA", {.description = "parameters for diffusion coefficient",
                         .default_value = std::vector<double>{},
                         .size = from_parameter<int>("DIFF_PARA_NUM")}));
    m->add_component(parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM",
        {.description = "number of parameters for scaling function describing temperature "
                        "dependence of diffusion "
                        "coefficient",
            .default_value = 0}));
    m->add_component(parameter<std::vector<double>>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA",
        {.description = "parameters for function describing temperature dependence of diffusion "
                        "coefficient",
            .default_value = std::vector<double>{},
            .size = from_parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM")}));
    m->add_component(parameter<int>("TRANS_PARA_NUM",
        {.description = "number of parameters for transference number", .default_value = 0}));
    m->add_component(parameter<std::vector<double>>(
        "TRANS_PARA", {.description = "parameters for transference number",
                          .default_value = std::vector<double>{},
                          .size = from_parameter<int>("TRANS_PARA_NUM")}));
    m->add_component(parameter<int>("THERM_PARA_NUM",
        {.description = "number of parameters for thermodynamic factor", .default_value = 0}));
    m->add_component(parameter<std::vector<double>>(
        "THERM_PARA", {.description = "parameters for thermodynamic factor",
                          .default_value = std::vector<double>{},
                          .size = from_parameter<int>("THERM_PARA_NUM")}));
    m->add_component(parameter<int>("COND_PARA_NUM",
        {.description = "number of parameters for conductivity", .default_value = 0}));
    m->add_component(parameter<std::vector<double>>(
        "COND_PARA", {.description = "parameters for conductivity",
                         .default_value = std::vector<double>{},
                         .size = from_parameter<int>("COND_PARA_NUM")}));
    m->add_component(parameter<int>("COND_TEMP_SCALE_FUNCT_PARA_NUM",
        {.description = "number of parameters for temperature scaling of conductivity",
            .default_value = 0}));
    m->add_component(parameter<std::vector<double>>("COND_TEMP_SCALE_FUNCT_PARA",
        {.description = "parameters for temperature scaling of conductivity",
            .default_value = std::vector<double>{},
            .size = from_parameter<int>("COND_TEMP_SCALE_FUNCT_PARA_NUM")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material parameters for ion species in electrolyte solution for multi-scale approach (fang
  // 07/17)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_newman_multiscale",
        "material parameters for ion species in electrolyte solution for multi-scale approach",
        Core::Materials::m_newman_multiscale);

    m->add_component(parameter<double>("VALENCE", {.description = "valence (= charge number)"}));
    m->add_component(parameter<int>("DIFF_COEF_CONC_DEP_FUNCT",
        {.description = "function number of function describing concentration dependence of "
                        "diffusion coefficient"}));
    m->add_component(parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT",
        {.description = "FUNCT number describing temperature scaling of diffusion coefficient"}));
    m->add_component(
        parameter<int>("TRANSNR", {.description = "curve number for transference number"}));
    m->add_component(
        parameter<int>("THERMFAC", {.description = "curve number for thermodynamic factor"}));
    m->add_component(parameter<int>(
        "COND_CONC_DEP_FUNCT", {.description = "function number of function describing "
                                               "concentration dependence of conductivity"}));
    m->add_component(parameter<int>("COND_TEMP_SCALE_FUNCT",
        {.description = "FUNCT number describing temperature scaling of conductivity"}));
    m->add_component(
        parameter<double>("ELECTRONIC_COND", {.description = "electronic conductivity"}));
    m->add_component(parameter<int>("ELECTRONIC_COND_CONC_SCALE_FUNC_NUM",
        {.description =
                "FUNCT number describing concentration dependence of electronic conductivity"}));
    m->add_component(
        parameter<double>("A_s", {.description = "specific micro-scale surface area"}));
    m->add_component(parameter<std::string>("MICROFILE",
        {.description = "input file for micro scale", .default_value = "filename.dat"}));
    m->add_component(
        parameter<int>("MICRODIS_NUM", {.description = "number of micro-scale discretization"}));
    // optional parameters for implemented concentration-depending functions
    m->add_component(parameter<int>("DIFF_PARA_NUM",
        {.description = "number of parameters for diffusion coefficient", .default_value = 0}));
    m->add_component(parameter<std::vector<double>>(
        "DIFF_PARA", {.description = "parameters for diffusion coefficient",
                         .default_value = std::vector<double>{},
                         .size = from_parameter<int>("DIFF_PARA_NUM")}));
    m->add_component(parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM",
        {.description =
                "number of parameters for scaling function describing temperature dependence of "
                "diffusion coefficient",
            .default_value = 0}));
    m->add_component(parameter<std::vector<double>>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA",
        {.description = "parameters for function describing temperature dependence of diffusion "
                        "coefficient",
            .default_value = std::vector<double>{},
            .size = from_parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM")}));
    m->add_component(parameter<int>("TRANS_PARA_NUM",
        {.description = "number of parameters for transference number", .default_value = 0}));
    m->add_component(parameter<std::vector<double>>(
        "TRANS_PARA", {.description = "parameters for transference number",
                          .default_value = std::vector<double>{},
                          .size = from_parameter<int>("TRANS_PARA_NUM")}));
    m->add_component(parameter<int>("THERM_PARA_NUM",
        {.description = "number of parameters for thermodynamic factor", .default_value = 0}));
    m->add_component(parameter<std::vector<double>>(
        "THERM_PARA", {.description = "parameters for thermodynamic factor",
                          .default_value = std::vector<double>{},
                          .size = from_parameter<int>("THERM_PARA_NUM")}));
    m->add_component(parameter<int>("COND_PARA_NUM",
        {.description = "number of parameters for ionic conductivity", .default_value = 0}));
    m->add_component(parameter<std::vector<double>>(
        "COND_PARA", {.description = "parameters for ionic conductivity",
                         .default_value = std::vector<double>{},
                         .size = from_parameter<int>("COND_PARA_NUM")}));
    m->add_component(parameter<int>("COND_TEMP_SCALE_FUNCT_PARA_NUM",
        {.description = "number of parameters for temperature scaling of conductivity",
            .default_value = 0}));
    m->add_component(parameter<std::vector<double>>("COND_TEMP_SCALE_FUNCT_PARA",
        {.description = "parameters for temperature scaling of conductivity",
            .default_value = std::vector<double>{},
            .size = from_parameter<int>("COND_TEMP_SCALE_FUNCT_PARA_NUM")}));

    Mat::append_material_definition(matlist, m);
  }
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_scl", "material parameters for space charge layers", Core::Materials::m_scl);

    m->add_component(parameter<double>("VALENCE", {.description = "valence/charge number"}));
    m->add_component(parameter<int>("DIFF_COEF_CONC_DEP_FUNCT",
        {.description = "function number of function describing concentration dependence of "
                        "diffusion coefficient"}));
    m->add_component(parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT",
        {.description =
                "function number describing temperature scaling of diffusion coefficient"}));
    m->add_component(
        parameter<int>("TRANSNR", {.description = "curve number for transference number"}));
    m->add_component(parameter<int>(
        "COND_CONC_DEP_FUNCT", {.description = "function number of function describing "
                                               "concentration dependence of conductivity"}));
    m->add_component(parameter<int>("COND_TEMP_SCALE_FUNCT",
        {.description = "function number describing temperature scaling of conductivity"}));
    m->add_component(parameter<int>("DIFF_PARA_NUM",
        {.description = "number of parameters for diffusion coefficient", .default_value = 0}));
    m->add_component(parameter<std::vector<double>>(
        "DIFF_PARA", {.description = "parameters for diffusion coefficient",
                         .default_value = std::vector<double>{},
                         .size = from_parameter<int>("DIFF_PARA_NUM")}));
    m->add_component(parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM",
        {.description = "number of parameters for scaling function describing temperature "
                        "dependence of diffusion "
                        "coefficient",
            .default_value = 0}));
    m->add_component(parameter<std::vector<double>>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA",
        {.description = "parameters for function describing temperature dependence of diffusion "
                        "coefficient",
            .default_value = std::vector<double>{},
            .size = from_parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM")}));
    m->add_component(parameter<int>("TRANS_PARA_NUM",
        {.description = "number of parameters for transference number", .default_value = 0}));
    m->add_component(parameter<std::vector<double>>(
        "TRANS_PARA", {.description = "parameters for transference number",
                          .default_value = std::vector<double>{},
                          .size = from_parameter<int>("TRANS_PARA_NUM")}));
    m->add_component(parameter<int>("COND_PARA_NUM",
        {.description = "number of parameters for conductivity", .default_value = 0}));
    m->add_component(parameter<std::vector<double>>(
        "COND_PARA", {.description = "parameters for conductivity",
                         .default_value = std::vector<double>{},
                         .size = from_parameter<int>("COND_PARA_NUM")}));
    m->add_component(parameter<int>("COND_TEMP_SCALE_FUNCT_PARA_NUM",
        {.description = "number of parameters for temperature scaling of conductivity",
            .default_value = 0}));
    m->add_component(parameter<std::vector<double>>("COND_TEMP_SCALE_FUNCT_PARA",
        {.description = "parameters for temperature scaling of conductivity",
            .default_value = std::vector<double>{},
            .size = from_parameter<int>("COND_TEMP_SCALE_FUNCT_PARA_NUM")}));
    m->add_component(
        parameter<double>("MAX_CONC", {.description = "maximum cation concentration"}));
    m->add_component(parameter<int>(
        "EXTRAPOL_DIFF", {.description = "strategy for extrapolation of diffusion "
                                         "coefficient below 0 and above MAX_CONC (-1: "
                                         "disabled, 0: constant)"}));
    m->add_component(parameter<double>("LIM_CONC",
        {.description = "limiting concentration for extrapolation", .default_value = 1.0}));
    m->add_component(parameter<double>("BULK_CONC", {.description = "bulk ion concentration"}));
    m->add_component(parameter<double>("SUSCEPT", {.description = "susceptibility"}));
    m->add_component(parameter<double>(
        "DELTA_NU", {.description = "difference of partial molar volumes (vacancy & cation)"}));

    Mat::append_material_definition(matlist, m);
  }


  /*----------------------------------------------------------------------*/
  // electrode material (fang 02/15)
  {
    auto matelectrode = std::make_shared<Mat::MaterialDefinition>(
        "MAT_electrode", "electrode material", Core::Materials::m_electrode);

    // diffusivity and electronic conductivity
    matelectrode->add_component(parameter<int>("DIFF_COEF_CONC_DEP_FUNCT",
        {.description = "function number of function describing concentration dependence of "
                        "diffusion coefficient"}));
    matelectrode->add_component(parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT",
        {.description = "FUNCT number describing temperature scaling of diffusion coefficient"}));
    matelectrode->add_component(parameter<int>(
        "COND_CONC_DEP_FUNCT", {.description = "function number of function describing "
                                               "concentration dependence of conductivity"}));
    matelectrode->add_component(parameter<int>("COND_TEMP_SCALE_FUNCT",
        {.description = "FUNCT number describing temperature scaling of conductivity"}));

    // optional parameters for concentration dependency of diffusivity and electronic conductivity
    matelectrode->add_component(parameter<int>("DIFF_PARA_NUM",
        {.description = "number of parameters for diffusion coefficient", .default_value = 0}));
    matelectrode->add_component(parameter<std::vector<double>>(
        "DIFF_PARA", {.description = "parameters for diffusion coefficient",
                         .default_value = std::vector<double>{},
                         .size = from_parameter<int>("DIFF_PARA_NUM")}));
    matelectrode->add_component(parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM",
        {.description = "number of parameters for scaling function describing temperature "
                        "dependence of diffusion "
                        "coefficient",
            .default_value = 0}));
    matelectrode->add_component(parameter<std::vector<double>>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA",
        {.description = "parameters for function describing temperature dependence of diffusion "
                        "coefficient",
            .default_value = std::vector<double>{},
            .size = from_parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM")}));
    matelectrode->add_component(parameter<int>("COND_PARA_NUM",
        {.description = "number of parameters for electronic conductivity", .default_value = 0}));
    matelectrode->add_component(parameter<std::vector<double>>(
        "COND_PARA", {.description = "parameters for electronic conductivity",
                         .default_value = std::vector<double>{},
                         .size = from_parameter<int>("COND_PARA_NUM")}));
    matelectrode->add_component(parameter<int>("COND_TEMP_SCALE_FUNCT_PARA_NUM",
        {.description = "number of parameters for temperature scaling of conductivity",
            .default_value = 0}));
    matelectrode->add_component(parameter<std::vector<double>>("COND_TEMP_SCALE_FUNCT_PARA",
        {.description = "parameters for temperature scaling of conductivity",
            .default_value = std::vector<double>{},
            .size = from_parameter<int>("COND_TEMP_SCALE_FUNCT_PARA_NUM")}));
    // saturation value of intercalated Lithium concentration
    matelectrode->add_component(parameter<double>(
        "C_MAX", {.description = "saturation value of intercalated Lithium concentration"}));

    // lithiation value corresponding to saturation value of intercalated Lithium concentration
    matelectrode->add_component(
        parameter<double>("CHI_MAX", {.description = "lithiation value corresponding to saturation "
                                                     "value of intercalated Lithium concentration "
                                                     "'C_MAX'"}));

    // model for half cell open circuit potential of electrode
    using namespace Core::IO::InputSpecBuilders;
    matelectrode->add_component(group("OCP_MODEL",
        {
            one_of(
                {
                    group("Function",
                        {
                            parameter<int>("OCP_FUNCT_NUM",
                                {
                                    .description = "function number of function that is used to "
                                                   "model the open circuit potential",
                                }),
                        }),
                    group("Redlich-Kister",
                        {
                            parameter<int>("OCP_PARA_NUM",
                                {
                                    .description = "number of parameters underlying half "
                                                   "cell open circuit potential model",
                                }),
                            parameter<std::vector<double>>(
                                "OCP_PARA", {.description = "parameters underlying half cell open "
                                                            "circuit potential model",
                                                .size = from_parameter<int>("OCP_PARA_NUM")}),
                        }),
                    group("Taralov",
                        {
                            parameter<std::vector<double>>(
                                "OCP_PARA", {.description = "parameters underlying half cell open "
                                                            "circuit potential model",
                                                .size = 13}),
                        }),
                },
                store_index_as<Mat::PAR::OCPModels>("OCP_MODEL")),
            parameter<double>(
                "X_MIN", {.description = "lower bound of range of validity as a fraction of "
                                         "C_MAX for ocp calculation model"}),
            parameter<double>(
                "X_MAX", {.description = "upper bound of range of validity as a fraction of "
                                         "C_MAX for ocp calculation model"}),
        }));

    // add electrode material to global list of valid materials
    Mat::append_material_definition(matlist, matelectrode);
  }

  /*----------------------------------------------------------------------*/
  // material collection (gjb 07/08)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_matlist",
        "list/collection of materials, i.e. material IDs", Core::Materials::m_matlist);

    m->add_component(parameter<bool>("LOCAL",
        {.description = "individual materials allocated per element or only at global scope"}));
    m->add_component(parameter<int>("NUMMAT", {.description = "number of materials in list"}));
    m->add_component(parameter<std::vector<int>>(
        "MATIDS", {.description = "the list material IDs", .size = from_parameter<int>("NUMMAT")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material collection with reactions (thon 09/14)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_matlist_reactions",
        "list/collection of materials, i.e. material IDs and list of reactions",
        Core::Materials::m_matlist_reactions);

    m->add_component(parameter<bool>("LOCAL",
        {.description = "individual materials allocated per element or only at global scope"}));
    m->add_component(parameter<int>("NUMMAT", {.description = "number of materials in list"}));
    m->add_component(parameter<std::vector<int>>(
        "MATIDS", {.description = "the list material IDs", .size = from_parameter<int>("NUMMAT")}));
    m->add_component(
        parameter<int>("NUMREAC", {.description = "number of reactions for these elements"}));
    m->add_component(
        parameter<std::vector<int>>("REACIDS", {.description = "advanced reaction list",
                                                   .default_value = std::vector{0},
                                                   .size = from_parameter<int>("NUMREAC")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material collection with chemotaxis (thon 06/15)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_matlist_chemotaxis",
        "list/collection of materials, i.e. material IDs and list of chemotactic pairs",
        Core::Materials::m_matlist_chemotaxis);

    m->add_component(parameter<bool>("LOCAL",
        {.description = "individual materials allocated per element or only at global scope"}));
    m->add_component(parameter<int>("NUMMAT", {.description = "number of materials in list"}));
    m->add_component(parameter<std::vector<int>>(
        "MATIDS", {.description = "the list material IDs", .size = from_parameter<int>("NUMMAT")}));
    m->add_component(
        parameter<int>("NUMPAIR", {.description = "number of pairs for these elements"}));
    m->add_component(
        parameter<std::vector<int>>("PAIRIDS", {.description = "chemotaxis pairs list",
                                                   .default_value = std::vector{0},
                                                   .size = from_parameter<int>("NUMPAIR")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material collection with reactions AND chemotaxis (thon 06/15)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_matlist_chemo_reac",
        "list/collection of materials, i.e. material IDs and list of reactive/chemotactic pairs",
        Core::Materials::m_matlist_chemoreac);

    m->add_component(parameter<bool>("LOCAL",
        {.description = "individual materials allocated per element or only at global scope"}));
    m->add_component(parameter<int>("NUMMAT", {.description = "number of materials in list"}));
    m->add_component(parameter<std::vector<int>>(
        "MATIDS", {.description = "the list material IDs", .size = from_parameter<int>("NUMMAT")}));
    m->add_component(
        parameter<int>("NUMPAIR", {.description = "number of pairs for these elements"}));
    m->add_component(
        parameter<std::vector<int>>("PAIRIDS", {.description = "chemotaxis pairs list",
                                                   .default_value = std::vector{0},
                                                   .size = from_parameter<int>("NUMPAIR")}));
    m->add_component(
        parameter<int>("NUMREAC", {.description = "number of reactions for these elements"}));
    m->add_component(
        parameter<std::vector<int>>("REACIDS", {.description = "advanced reaction list",
                                                   .default_value = std::vector{0},
                                                   .size = from_parameter<int>("NUMREAC")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material collection (ehrl 11/12)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_elchmat",
        "specific list/collection of species and phases for elch applications",
        Core::Materials::m_elchmat);

    m->add_component(parameter<bool>("LOCAL",
        {.description = "individual materials allocated per element or only at global scope",
            .default_value = false}));
    m->add_component(parameter<int>("NUMDOF", {.description = "number of dof's per node"}));
    m->add_component(
        parameter<int>("NUMSCAL", {.description = "number of transported scalars per node"}));
    m->add_component(
        parameter<int>("NUMPHASE", {.description = "number of phases in electrolyte"}));
    m->add_component(parameter<std::vector<int>>("PHASEIDS",
        {.description = "the list phasel IDs", .size = from_parameter<int>("NUMPHASE")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material collection (ehrl 11/12)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_elchphase",
        "material parameters for ion species in electrolyte solution",
        Core::Materials::m_elchphase);

    m->add_component(parameter<bool>("LOCAL",
        {.description = "individual materials allocated per element or only at global scope",
            .default_value = false}));
    m->add_component(parameter<double>("EPSILON", {.description = "phase porosity"}));
    m->add_component(
        parameter<double>("TORTUOSITY", {.description = "inverse (!) of phase tortuosity"}));
    m->add_component(
        parameter<int>("NUMMAT", {.description = "number of materials in electrolyte"}));
    m->add_component(parameter<std::vector<int>>(
        "MATIDS", {.description = "the list phasel IDs", .size = from_parameter<int>("NUMMAT")}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // St.Venant--Kirchhoff
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_StVenantKirchhoff",
        "St.Venant--Kirchhoff material", Core::Materials::m_stvenant);

    m->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));
    m->add_component(parameter<double>("NUE", {.description = "Poisson's ratio"}));
    m->add_component(parameter<double>("DENS", {.description = "mass density"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // St.Venant--Kirchhoff with temperature
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_ThermoStVenantK",
        "Thermo St.Venant--Kirchhoff material", Core::Materials::m_thermostvenant);

    m->add_component(parameter<int>(
        "YOUNGNUM", {.description = "number of Young's modulus in list (if 1 Young is "
                                    "const, if >1 Young is temperature) "
                                    "dependent"}));
    m->add_component(parameter<std::vector<double>>(
        "YOUNG", {.description = "Young's modulus", .size = from_parameter<int>("YOUNGNUM")}));
    m->add_component(parameter<double>("NUE", {.description = "Poisson's ratio"}));
    m->add_component(parameter<double>("DENS", {.description = "mass density"}));
    m->add_component(parameter<double>(
        "THEXPANS", {.description = "constant coefficient of linear thermal expansion"}));
    m->add_component(parameter<double>("CAPA", {.description = "capacity"}));
    m->add_component(parameter<double>("CONDUCT", {.description = "conductivity"}));
    m->add_component(parameter<double>("INITTEMP", {.description = "initial temperature"}));
    m->add_component(parameter<int>(
        "THERMOMAT", {.description = "mat id of thermal material part", .default_value = -1}));

    Mat::append_material_definition(matlist, m);
  }
  /*----------------------------------------------------------------------*/
  // Plastic linear elastic St.Venant Kirchhoff / Drucker Prager plasticity
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_DruckerPrager",
        "elastic St.Venant Kirchhoff / plastic drucker prager", Core::Materials::m_pldruckprag);

    m->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));
    m->add_component(parameter<double>("NUE", {.description = "Poisson's ratio"}));
    m->add_component(parameter<double>("DENS", {.description = "Density"}));
    m->add_component(parameter<double>("ISOHARD", {.description = "linear isotropic hardening"}));
    m->add_component(parameter<double>("TOL", {.description = "Local Newton iteration tolerance"}));
    m->add_component(parameter<double>("C", {.description = "cohesion"}));
    m->add_component(parameter<double>("ETA", {.description = "Drucker Prager Constant Eta"}));
    m->add_component(parameter<double>("XI", {.description = "Drucker Prager Constant Xi"}));
    m->add_component(
        parameter<double>("ETABAR", {.description = "Drucker Prager Constant Etabar"}));
    m->add_component(parameter<std::string>("TANG",
        {.description = "Method to compute the material tangent", .default_value = "consistent"}));
    m->add_component(parameter<int>("MAXITER",
        {.description = "Maximum Iterations for local Neutron Raphson", .default_value = 50}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Linear thermo-elastic St.Venant Kirchhoff / plastic von Mises
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_ThermoPlasticLinElast",
        "Thermo-elastic St.Venant Kirchhoff / plastic von Mises material",
        Core::Materials::m_thermopllinelast);

    m->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));
    m->add_component(parameter<double>("NUE", {.description = "Poisson's ratio"}));
    m->add_component(parameter<double>("DENS", {.description = "mass density"}));
    m->add_component(
        parameter<double>("THEXPANS", {.description = "coefficient of linear thermal expansion"}));
    m->add_component(parameter<double>("INITTEMP", {.description = "initial temperature"}));
    m->add_component(parameter<double>("YIELD", {.description = "yield stress"}));
    m->add_component(parameter<double>("ISOHARD", {.description = "isotropic hardening modulus"}));
    m->add_component(parameter<double>("KINHARD", {.description = "kinematic hardening modulus"}));
    m->add_component(
        parameter<int>("SAMPLENUM", {.description = "number of stress-strain pairs in list"}));
    m->add_component(parameter<std::vector<double>>(
        "SIGMA_Y", {.description = "yield stress", .size = from_parameter<int>("SAMPLENUM")}));
    m->add_component(parameter<std::vector<double>>(
        "EPSBAR_P", {.description = "accumulated plastic strain corresponding to SIGMA_Y",
                        .size = from_parameter<int>("SAMPLENUM")}));
    m->add_component(
        parameter<double>("TOL", {.description = "tolerance for local Newton iteration"}));
    m->add_component(parameter<int>(
        "THERMOMAT", {.description = "mat id of thermal material part", .default_value = -1}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Plastic linear elastic St.Venant Kirchhoff / GTN plasticity
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_PlasticGTN",
        "elastic St.Venant Kirchhoff / plastic GTN", Core::Materials::m_plgtn);

    m->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));
    m->add_component(parameter<double>("NUE", {.description = "Poisson's ratio"}));
    m->add_component(parameter<double>("DENS", {.description = "Density"}));
    m->add_component(parameter<double>("YIELD", {.description = "yield stress"}));
    m->add_component(parameter<double>("ISOHARD", {.description = "linear isotropic hardening"}));
    m->add_component(parameter<int>("HARDENING_FUNC",
        {.description = "Function number for isotropic hardening", .default_value = 0}));
    m->add_component(parameter<double>("TOL", {.description = "Local Newton iteration tolerance"}));
    m->add_component(parameter<int>(
        "MAXITER", {.description = "Maximum Neutron Raphson Iterations", .default_value = 50}));
    m->add_component(parameter<double>("K1", {.description = "GTN Constant k1"}));
    m->add_component(parameter<double>("K2", {.description = "GTN Constant k2"}));
    m->add_component(parameter<double>("K3", {.description = "GTN constant k3"}));
    m->add_component(
        parameter<double>("F0", {.description = "GTN constant f0 for initial damage"}));
    m->add_component(
        parameter<double>("FN", {.description = "GTN constant fN for damage nucleation"}));
    m->add_component(
        parameter<double>("EN", {.description = "GTN constant eN for damage nucleation"}));
    m->add_component(
        parameter<double>("SN", {.description = "GTN constant sN for damage nucleation"}));
    m->add_component(
        parameter<double>("FC", {.description = "GTN constant fC for damage coalescence"}));
    m->add_component(
        parameter<double>("KAPPA", {.description = "GTN constant kappa for damage coalescence"}));
    m->add_component(parameter<double>(
        "EF", {.description = "GTN stabilization parameter ef for damage coalescence",
                  .default_value = 0.0}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Finite strain superelasticity of shape memory alloys
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_SuperElastSMA",
        "finite strain superelastic shape memory alloy", Core::Materials::m_superelast);

    m->add_component(parameter<double>("DENS", {.description = "mass density"}));
    m->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));
    m->add_component(parameter<double>("NUE", {.description = "Poisson's ratio"}));
    m->add_component(parameter<double>(
        "EPSILON_L", {.description = "parameter representing the maximum deformation "
                                     "obtainable only by detwinning of the "
                                     "multiple-variant martensite"}));
    m->add_component(
        parameter<double>("T_AS_s", {.description = "Temperature at which the phase transformation "
                                                    "from austenite to martensite starts"}));
    m->add_component(
        parameter<double>("T_AS_f", {.description = "Temperature at which the phase transformation "
                                                    "from austenite to martensite finishes"}));
    m->add_component(
        parameter<double>("T_SA_s", {.description = "Temperature at which the phase transformation "
                                                    "from martensite to autenite starts"}));
    m->add_component(
        parameter<double>("T_SA_f", {.description = "Temperature at which the phase transformation "
                                                    "from martensite to autenite finishes"}));
    m->add_component(parameter<double>(
        "C_AS", {.description = "Coefficient of the linear temperature dependence of T_AS"}));
    m->add_component(parameter<double>(
        "C_SA", {.description = "Coefficient of the linear temperature dependence of T_SA"}));
    m->add_component(parameter<double>("SIGMA_AS_s",
        {.description =
                "stress at which the phase transformation from austenite to martensite begins"}));
    m->add_component(parameter<double>("SIGMA_AS_f",
        {.description =
                "stress at which the phase transformation from austenite to martensite finishes"}));
    m->add_component(parameter<double>("SIGMA_SA_s",
        {.description =
                "stress at which the phase transformation from martensite to austenite begins"}));
    m->add_component(parameter<double>("SIGMA_SA_f",
        {.description =
                "stress at which the phase transformation from martensite to austenite finishes"}));
    m->add_component(parameter<double>(
        "ALPHA", {.description = "pressure dependency in the drucker-prager-type loading"}));
    m->add_component(parameter<int>("MODEL",
        {.description =
                "Model used for the evolution of martensitic fraction (1=exponential; 2=linear)"}));
    m->add_component(parameter<double>("BETA_AS",
        {.description =
                "parameter, measuring the speed of the transformation from austenite to martensite",
            .default_value = 0.}));
    m->add_component(parameter<double>("BETA_SA",
        {.description =
                "parameter, measuring the speed of the transformation from martensite to austenite",
            .default_value = 0.}));


    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Thermo-hyperelasticity / finite strain von-Mises plasticity
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_ThermoPlasticHyperElast",
        "Thermo-hyperelastic / finite strain plastic von Mises material "
        "with linear and exponential isotropic hardening",
        Core::Materials::m_thermoplhyperelast);

    m->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));
    m->add_component(parameter<double>("NUE", {.description = "Poisson's ratio"}));
    m->add_component(parameter<double>("DENS", {.description = "mass density"}));
    m->add_component(parameter<double>(
        "CTE", {.description = "coefficient of thermal expansion", .default_value = 0.}));
    m->add_component(parameter<double>(
        "INITTEMP", {.description = "initial, reference temperature", .default_value = 0.}));
    m->add_component(parameter<double>("YIELD", {.description = "initial yield stress"}));
    m->add_component(parameter<double>(
        "ISOHARD", {.description = "linear isotropic hardening modulus", .default_value = 0.}));
    m->add_component(parameter<double>(
        "SATHARDENING", {.description = "saturation hardening", .default_value = 0.}));
    m->add_component(
        parameter<double>("HARDEXPO", {.description = "hardening exponent", .default_value = 0.}));
    m->add_component(parameter<double>(
        "YIELDSOFT", {.description = "thermal yield stress softening", .default_value = 0.}));
    m->add_component(parameter<double>("HARDSOFT",
        {.description = "thermal hardening softening (acting on SATHARDENING and ISOHARD)",
            .default_value = 0.}));
    m->add_component(parameter<double>(
        "TOL", {.description = "tolerance for local Newton iteration", .default_value = 1.e-8}));
    m->add_component(parameter<int>(
        "THERMOMAT", {.description = "mat id of thermal material part", .default_value = -1}));

    Mat::append_material_definition(matlist, m);
  }


  /*----------------------------------------------------------------------*/
  // Hyperelasticity / finite strain von-Mises plasticity
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_PlasticNlnLogNeoHooke",
        "hyperelastic / finite strain plastic von Mises material "
        "with linear and exponential isotropic hardening or the definition of a hardening function "
        "(VARFUNCTION using the variable epsp)",
        Core::Materials::m_plnlnlogneohooke);

    m->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));
    m->add_component(parameter<double>("NUE", {.description = "Poisson's ratio"}));
    m->add_component(parameter<double>("DENS", {.description = "mass density"}));
    m->add_component(
        parameter<double>("YIELD", {.description = "yield stress", .default_value = 0.}));
    m->add_component(parameter<double>(
        "ISOHARD", {.description = "isotropic hardening modulus", .default_value = 0.}));
    m->add_component(parameter<double>(
        "SATHARDENING", {.description = "saturation hardening", .default_value = 0.}));
    m->add_component(parameter<double>(
        "HARDEXPO", {.description = "linear hardening exponent", .default_value = 0.}));
    m->add_component(parameter<double>("VISC", {.description = "VISCOSITY", .default_value = 0.}));
    m->add_component(parameter<double>(
        "RATE_DEPENDENCY", {.description = "rate dependency", .default_value = 0.}));
    m->add_component(parameter<double>("TOL",
        {.description = "Tolerance for local Newton-Raphson iteration", .default_value = 1.e-08}));
    m->add_component(parameter<int>("HARDENING_FUNC",
        {.description = "Function number for isotropic hardening", .default_value = 0}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Plastic linear elastic St.Venant Kirchhoff / von Mises
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_PlasticLinElast",
        "elastic St.Venant Kirchhoff / plastic von Mises material "
        "with linear isotropic and kineamtic hardening",
        Core::Materials::m_pllinelast);

    m->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));
    m->add_component(parameter<double>("NUE", {.description = "Poisson's ratio"}));
    m->add_component(parameter<double>("DENS", {.description = "mass density"}));
    m->add_component(parameter<double>("YIELD", {.description = "yield stress"}));
    m->add_component(
        parameter<double>("ISOHARD", {.description = "linear isotropic hardening modulus"}));
    m->add_component(
        parameter<double>("KINHARD", {.description = "linear kinematic hardening modulus"}));
    m->add_component(
        parameter<double>("TOL", {.description = "tolerance for local Newton iteration"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Elastic visco-plastic finite strain material law without yield surface
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_Viscoplastic_No_Yield_Surface",
        "Elastic visco-plastic finite strain material law without yield surface",
        Core::Materials::m_vp_no_yield_surface);

    // elasticity parameters
    m->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));
    m->add_component(parameter<double>("NUE", {.description = "Poisson's ratio"}));
    m->add_component(parameter<double>("DENS", {.description = "material mass density"}));
    // visco-plasticity parameters
    m->add_component(parameter<double>("TEMPERATURE", {.description = "temperature in Kelvin"}));
    m->add_component(parameter<double>(
        "PRE_EXP_FAC", {.description = "pre-exponential factor of plastic shear strain rate 'A'"}));
    m->add_component(
        parameter<double>("ACTIVATION_ENERGY", {.description = "activation energy 'Q'"}));
    m->add_component(parameter<double>("GAS_CONSTANT", {.description = "gas constant 'R'"}));
    m->add_component(
        parameter<double>("STRAIN_RATE_SENS", {.description = "strain-rate-sensitivity 'm'"}));
    m->add_component(parameter<double>(
        "INIT_FLOW_RES", {.description = "initial isotropic flow resistance 'S^0'"}));
    m->add_component(
        parameter<double>("FLOW_RES_PRE_FAC", {.description = "flow resistance factor 'H_0'"}));
    m->add_component(parameter<double>(
        "FLOW_RES_EXP", {.description = "flow resistance exponential value 'a'"}));
    m->add_component(parameter<double>(
        "FLOW_RES_SAT_FAC", {.description = "flow resistance saturation factor 'S_*'"}));
    m->add_component(parameter<double>(
        "FLOW_RES_SAT_EXP", {.description = "flow resistance saturation exponent 'b'"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Robinson's visco-plastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_Struct_Robinson", "Robinson's visco-plastic material", Core::Materials::m_vp_robinson);

    m->add_component(parameter<std::string>(
        "KIND", {.description = "kind of Robinson material: "
                                "Butler, Arya, Arya_NarloyZ (default), Arya_CrMoSteel"}));
    m->add_component(
        parameter<int>("YOUNGNUM", {.description = "number of Young's modulus in list"}));
    m->add_component(parameter<std::vector<double>>(
        "YOUNG", {.description = "Young's modulus", .size = from_parameter<int>("YOUNGNUM")}));
    m->add_component(parameter<double>("NUE", {.description = "Poisson's ratio"}));
    m->add_component(parameter<double>("DENS", {.description = "mass density"}));
    m->add_component(
        parameter<double>("THEXPANS", {.description = "coefficient of linear thermal expansion"}));
    m->add_component(parameter<double>("INITTEMP", {.description = "initial temperature"}));
    m->add_component(parameter<double>("HRDN_FACT", {.description = "hardening factor 'A'"}));
    m->add_component(parameter<double>("HRDN_EXPO", {.description = "hardening power 'n'"}));
    m->add_component(parameter<int>(
        "SHRTHRSHLDNUM", {.description = "number of shear stress threshold 'K^2'in list"}));
    m->add_component(parameter<std::vector<double>>(
        "SHRTHRSHLD", {.description = "Bingam-Prager shear stress threshold 'K^2'",
                          .size = from_parameter<int>("SHRTHRSHLDNUM")}));
    m->add_component(parameter<double>("RCVRY", {.description = "recovery factor 'R_0'"}));
    m->add_component(parameter<double>("ACTV_ERGY", {.description = "activation energy 'Q_0'"}));
    m->add_component(
        parameter<double>("ACTV_TMPR", {.description = "activation temperature 'T_0'"}));
    m->add_component(parameter<double>("G0", {.description = "'G_0'"}));
    m->add_component(parameter<double>("M_EXPO", {.description = "'m'"}));
    m->add_component(parameter<int>("BETANUM", {.description = "number of 'beta' in list"}));
    m->add_component(parameter<std::vector<double>>(
        "BETA", {.description = "beta", .size = from_parameter<int>("BETANUM")}));
    m->add_component(parameter<double>("H_FACT", {.description = "'H'"}));
    m->add_component(parameter<int>(
        "THERMOMAT", {.description = "mat id of thermal material part", .default_value = -1}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Elasto-plastic material with damage, based on MAT_Struct_PlasticLinElast
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_Damage",
        "elasto-plastic von Mises material with ductile damage", Core::Materials::m_elpldamage);

    m->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));
    m->add_component(parameter<double>("NUE", {.description = "Poisson's ratio"}));
    m->add_component(parameter<double>("DENS", {.description = "mass density"}));
    m->add_component(
        parameter<int>("SAMPLENUM", {.description = "number of stress-strain pairs in list"}));
    m->add_component(parameter<std::vector<double>>(
        "SIGMA_Y", {.description = "yield stress", .size = from_parameter<int>("SAMPLENUM")}));
    m->add_component(parameter<std::vector<double>>(
        "EPSBAR_P", {.description = "accumulated plastic strain corresponding to SIGMA_Y",
                        .size = from_parameter<int>("SAMPLENUM")}));
    m->add_component(
        parameter<double>("DAMDEN", {.description = "denominator of damage evaluations law"}));
    m->add_component(
        parameter<double>("DAMEXP", {.description = "exponent of damage evaluations law"}));
    m->add_component(parameter<double>("DAMTHRESHOLD", {.description = "damage threshold"}));
    m->add_component(parameter<double>(
        "KINHARD", {.description = "kinematic hardening modulus, stress-like variable"}));
    m->add_component(parameter<double>(
        "KINHARD_REC", {.description = "recovery factor, scalar-valued variable"}));
    m->add_component(parameter<double>("SATHARDENING", {.description = "saturation hardening"}));
    m->add_component(parameter<double>("HARDEXPO", {.description = "hardening exponent"}));
    m->add_component(
        parameter<double>("TOL", {.description = "tolerance for local Newton iteration"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // aneurysm wall material according to Raghavan and Vorp [2000]
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_AAANeoHooke",
        "aneurysm wall material according to Raghavan and Vorp [2000]",
        Core::Materials::m_aaaneohooke);

    m->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));
    m->add_component(parameter<double>("BETA", {.description = "2nd parameter"}));
    m->add_component(parameter<double>("NUE", {.description = "Poisson's ratio"}));
    m->add_component(parameter<double>("DENS", {.description = "mass density"}));

    Mat::append_material_definition(matlist, m);
  }


  /*----------------------------------------------------------------------*/
  // Visco-elastic Neo-Hookean material law
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_VISCONEOHOOKE",
        "visco-elastic neo-Hookean material law", Core::Materials::m_visconeohooke);
    m->add_component(parameter<double>("YOUNGS_SLOW", {.description = "???"}));
    m->add_component(parameter<double>("POISSON", {.description = "???"}));
    m->add_component(parameter<double>("DENS", {.description = "???"}));
    m->add_component(parameter<double>("YOUNGS_FAST", {.description = "???"}));
    m->add_component(parameter<double>("RELAX", {.description = "???"}));
    m->add_component(parameter<double>("THETA", {.description = "???"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Visco-elastic anisotropic fiber material law
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_VISCOANISO",
        "visco-elastic anisotropic fibre material law", Core::Materials::m_viscoanisotropic);

    m->add_component(parameter<double>("KAPPA", {.description = "dilatation modulus"}));
    m->add_component(parameter<double>("MUE", {.description = "Shear Modulus"}));
    m->add_component(parameter<double>("DENS", {.description = "Density"}));
    m->add_component(
        parameter<double>("K1", {.description = "Parameter for linear fiber stiffness"}));
    m->add_component(
        parameter<double>("K2", {.description = "Parameter for exponential fiber stiffness"}));
    m->add_component(parameter<double>("GAMMA", {.description = "angle between fibers"}));
    m->add_component(parameter<double>(
        "BETA_ISO", {.description = "ratio between elasticities in generalized Maxweel body"}));
    m->add_component(parameter<double>(
        "BETA_ANISO", {.description = "ratio between elasticities in generalized Maxweel body"}));
    m->add_component(parameter<double>("RELAX_ISO", {.description = "isotropic relaxation time"}));
    m->add_component(
        parameter<double>("RELAX_ANISO", {.description = "anisotropic relaxation time"}));
    m->add_component(parameter<double>(
        "MINSTRETCH", {.description = "minimal principal stretch fibers do respond to"}));
    m->add_component(parameter<int>("ELETHICKDIR",
        {.description = "Element thickness direction applies also to fibers (only sosh)"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Structural micro-scale approach: material parameters are calculated from microscale simulation
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_Multiscale",
        "Structural micro-scale approach: material parameters are calculated from microscale "
        "simulation",
        Core::Materials::m_struct_multiscale);

    m->add_component(parameter<std::string>("MICROFILE",
        {.description = "inputfile for microstructure", .default_value = "filename.dat"}));
    m->add_component(
        parameter<int>("MICRODIS_NUM", {.description = "Number of microscale discretization"}));
    m->add_component(parameter<double>(
        "INITVOL", {.description = "Initial volume of RVE", .default_value = 0.0}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // collection of hyperelastic materials
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_ElastHyper",
        "list/collection of hyperelastic materials, i.e. material IDs",
        Core::Materials::m_elasthyper);

    m->add_component(
        parameter<int>("NUMMAT", {.description = "number of materials/potentials in list"}));
    m->add_component(parameter<std::vector<int>>("MATIDS",
        {.description = "the list material/potential IDs", .size = from_parameter<int>("NUMMAT")}));
    m->add_component(parameter<double>("DENS", {.description = "material mass density"}));
    m->add_component(parameter<int>("POLYCONVEX",
        {.description = "1.0 if polyconvexity of system is checked", .default_value = 0}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // viscohyperelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_ViscoElastHyper",
        "Viscohyperelastic material compatible with the collection of hyperelastic materials",
        Core::Materials::m_viscoelasthyper);

    m->add_component(
        parameter<int>("NUMMAT", {.description = "number of materials/potentials in list"}));
    m->add_component(parameter<std::vector<int>>("MATIDS",
        {.description = "the list material/potential IDs", .size = from_parameter<int>("NUMMAT")}));
    m->add_component(parameter<double>("DENS", {.description = "material mass density"}));
    m->add_component(parameter<int>("POLYCONVEX",
        {.description = "1.0 if polyconvexity of system is checked", .default_value = 0}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // collection of hyperelastic materials for finite strain plasticity
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PlasticElastHyper",
        "list/collection of hyperelastic materials, i.e. material IDs",
        Core::Materials::m_plelasthyper);

    m->add_component(
        parameter<int>("NUMMAT", {.description = "number of materials/potentials in list"}));
    m->add_component(parameter<std::vector<int>>("MATIDS",
        {.description = "the list material/potential IDs", .size = from_parameter<int>("NUMMAT")}));
    m->add_component(parameter<double>("DENS", {.description = "material mass density"}));
    m->add_component(parameter<double>("INITYIELD", {.description = "initial yield stress"}));
    m->add_component(parameter<int>("POLYCONVEX",
        {.description = "1.0 if polyconvexity of system is checked", .default_value = 0}));
    m->add_component(parameter<double>(
        "ISOHARD", {.description = "linear isotropic hardening modulus", .default_value = 0.}));
    m->add_component(parameter<double>("EXPISOHARD",
        {.description = "nonlinear isotropic hardening exponent", .default_value = 0.}));
    m->add_component(parameter<double>(
        "INFYIELD", {.description = "saturation yield stress for nonlinear isotropic hardening",
                        .default_value = 0.}));
    m->add_component(parameter<double>(
        "KINHARD", {.description = "linear kinematic hardening modulus", .default_value = 0.}));

    // visco-plasticity
    m->add_component(parameter<double>("VISC",
        {.description = "Visco-Plasticity parameter 'eta' in Perzyna model", .default_value = 0.}));
    m->add_component(parameter<double>("RATE_DEPENDENCY",
        {.description = "Visco-Plasticity parameter 'eta' in Perzyna model", .default_value = 1.}));
    m->add_component(parameter<double>("VISC_SOFT",
        {.description = "Visco-Plasticity temperature dependency (eta = eta_0 * (1-(T-T_0)*x)",
            .default_value = 0.}));

    // optional pastic spin parameter
    m->add_component(parameter<double>(
        "PL_SPIN_CHI", {.description = "Plastic spin coupling parameter chi (often called eta)",
                           .default_value = 0.0}));

    // optional Hill yield parameters
    m->add_component(parameter<double>(
        "rY_11", {.description = "relative yield stress in fiber1-direction (Y_11/Y_0)",
                     .default_value = 0.0}));
    m->add_component(parameter<double>(
        "rY_22", {.description = "relative yield stress in fiber2-direction (Y_22/Y_0)",
                     .default_value = 0.0}));
    m->add_component(parameter<double>(
        "rY_33", {.description = "relative yield stress in fiber3-direction (Y_33/Y_0)",
                     .default_value = 0.0}));
    m->add_component(parameter<double>(
        "rY_12", {.description = "relative shear yield stress in 12-direction (Y_12/Y_0)",
                     .default_value = 0.0}));
    m->add_component(parameter<double>(
        "rY_23", {.description = "relative shear yield stress in 23-direction (Y_23/Y_0)",
                     .default_value = 0.0}));
    m->add_component(parameter<double>(
        "rY_13", {.description = "relative shear yield stress in 13-direction (Y_13/Y_0)",
                     .default_value = 0.0}));

    // optional TSI parameters
    m->add_component(parameter<double>(
        "CTE", {.description = "coefficient of thermal expansion", .default_value = 0.}));
    m->add_component(parameter<double>(
        "INITTEMP", {.description = "initial, reference temperature", .default_value = 0.}));
    m->add_component(parameter<double>(
        "YIELDSOFT", {.description = "yield stress softening", .default_value = 0.}));
    m->add_component(
        parameter<double>("HARDSOFT", {.description = "hardening softening", .default_value = 0.}));
    m->add_component(parameter<double>("TAYLOR_QUINNEY",
        {.description = "Taylor-Quinney factor for plastic heat conversion", .default_value = 1.}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // collection of hyperelastic materials for finite strain plasticity
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PlasticElastHyperVCU",
        "list/collection of hyperelastic materials, i.e. material IDs",
        Core::Materials::m_plelasthyperVCU);

    m->add_component(
        parameter<int>("NUMMAT", {.description = "number of materials/potentials in list"}));
    m->add_component(parameter<std::vector<int>>("MATIDS",
        {.description = "the list material/potential IDs", .size = from_parameter<int>("NUMMAT")}));
    m->add_component(parameter<double>("DENS", {.description = "material mass density"}));
    m->add_component(parameter<double>("INITYIELD", {.description = "initial yield stress"}));
    m->add_component(parameter<double>(
        "ISOHARD", {.description = "linear isotropic hardening modulus", .default_value = 0.}));
    m->add_component(parameter<double>("EXPISOHARD",
        {.description = "nonlinear isotropic hardening exponent", .default_value = 0.}));
    m->add_component(parameter<double>(
        "INFYIELD", {.description = "saturation yield stress for nonlinear isotropic hardening",
                        .default_value = 0.}));
    m->add_component(parameter<double>(
        "KINHARD", {.description = "linear kinematic hardening modulus", .default_value = 0.}));

    // visco-plasticity
    m->add_component(parameter<double>("VISC",
        {.description = "Visco-Plasticity parameter 'eta' in Perzyna model", .default_value = 0.}));
    m->add_component(parameter<double>("RATE_DEPENDENCY",
        {.description = "Visco-Plasticity parameter 'eta' in Perzyna model", .default_value = 1.}));
    m->add_component(parameter<double>("VISC_SOFT",
        {.description = "Visco-Plasticity temperature dependency (eta = eta_0 * (1-(T-T_0)*x)",
            .default_value = 0.}));

    // optional pastic spin parameter
    m->add_component(parameter<double>(
        "PL_SPIN_CHI", {.description = "Plastic spin coupling parameter chi (often called eta)",
                           .default_value = 0.0}));

    // optional Hill yield parameters
    m->add_component(parameter<double>(
        "rY_11", {.description = "relative yield stress in fiber1-direction (Y_11/Y_0)",
                     .default_value = 0.0}));
    m->add_component(parameter<double>(
        "rY_22", {.description = "relative yield stress in fiber2-direction (Y_22/Y_0)",
                     .default_value = 0.0}));
    m->add_component(parameter<double>(
        "rY_33", {.description = "relative yield stress in fiber3-direction (Y_33/Y_0)",
                     .default_value = 0.0}));
    m->add_component(parameter<double>(
        "rY_12", {.description = "relative shear yield stress in 12-direction (Y_12/Y_0)",
                     .default_value = 0.0}));
    m->add_component(parameter<double>(
        "rY_23", {.description = "relative shear yield stress in 23-direction (Y_23/Y_0)",
                     .default_value = 0.0}));
    m->add_component(parameter<double>(
        "rY_13", {.description = "relative shear yield stress in 13-direction (Y_13/Y_0)",
                     .default_value = 0.0}));

    // optional TSI parameters
    m->add_component(parameter<double>(
        "CTE", {.description = "coefficient of thermal expansion", .default_value = 0.}));
    m->add_component(parameter<double>(
        "INITTEMP", {.description = "initial, reference temperature", .default_value = 0.}));
    m->add_component(parameter<double>(
        "YIELDSOFT", {.description = "yield stress softening", .default_value = 0.}));
    m->add_component(
        parameter<double>("HARDSOFT", {.description = "hardening softening", .default_value = 0.}));
    m->add_component(parameter<double>("TAYLOR_QUINNEY",
        {.description = "Taylor-Quinney factor for plastic heat conversion", .default_value = 1.}));

    m->add_component(parameter<int>("POLYCONVEX",
        {.description = "1.0 if polyconvexity of system is checked", .default_value = 0}));


    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // logarithmic neo-Hooke material acc. to Bonet and Wood
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupLogNeoHooke",
        "logarithmic neo-Hooke material acc. to Bonet and Wood",
        Core::Materials::mes_couplogneohooke);

    m->add_component(parameter<std::string>(
        "MODE", {.description = "parameter set: YN (Young's modulus and "
                                "Poisson's ration; default) or Lame (mue and "
                                "lambda)"}));
    m->add_component(parameter<double>("C1", {.description = "E or mue"}));
    m->add_component(parameter<double>("C2", {.description = "nue or lambda"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // Saint-Venant-Kirchhoff as elastic summand
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "ELAST_CoupSVK", "Saint-Venant-Kirchhoff as elastic summand", Core::Materials::mes_coupSVK);

    m->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));
    m->add_component(parameter<double>("NUE", {.description = "Poisson's ratio"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // Simo-Pister type material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "ELAST_CoupSimoPister", "Simo-Pister type material", Core::Materials::mes_coupsimopister);

    m->add_component(parameter<double>("MUE", {.description = "material constant"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // logarithmic mixed neo-Hooke material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupLogMixNeoHooke",
        "mixed logarithmic neo-Hooke material", Core::Materials::mes_couplogmixneohooke);

    m->add_component(parameter<std::string>(
        "MODE", {.description = "parameter set: YN (Young's modulus and "
                                "Poisson's ration; default) or Lame (mue and "
                                "lambda)"}));
    m->add_component(parameter<double>("C1", {.description = "E or mue"}));
    m->add_component(parameter<double>("C2", {.description = "nue or lambda"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled exponential material for compressible material (according to Weikenmeier_2014)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupExpPol",
        "compressible, isochoric exponential material law for soft tissue",
        Core::Materials::mes_coupexppol);
    m->add_component(parameter<double>("A", {.description = "material constant"}));
    m->add_component(parameter<double>("B", {.description = "material constant linear I_1"}));
    m->add_component(parameter<double>("C", {.description = "material constant linear J"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // compressible neo-Hooke material acc. to Holzapfel
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupNeoHooke",
        "compressible neo-Hooke material acc. to Holzapfel", Core::Materials::mes_coupneohooke);

    m->add_component(
        parameter<double>("YOUNG", {.description = "Young's modulus", .default_value = 0.0}));
    m->add_component(
        parameter<double>("NUE", {.description = "Poisson's ratio", .default_value = 0.0}));

    Mat::append_material_definition(matlist, m);
  }
  // Mooney Rivlin  material acc. to Holzapfel
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupMooneyRivlin",
        "Mooney - Rivlin material acc. to Holzapfel", Core::Materials::mes_coupmooneyrivlin);

    m->add_component(
        parameter<double>("C1", {.description = "material constant", .default_value = 0.0}));
    m->add_component(
        parameter<double>("C2", {.description = "material constant", .default_value = 0.0}));
    m->add_component(
        parameter<double>("C3", {.description = "material constant", .default_value = 0.0}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled Blatz and Ko material acc. to Holzapfel
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupBlatzKo",
        "Blatz and Ko material acc. to Holzapfel", Core::Materials::mes_coupblatzko);

    m->add_component(parameter<double>("MUE", {.description = "Shear modulus"}));
    m->add_component(parameter<double>("NUE", {.description = "Poisson's ratio"}));
    m->add_component(parameter<double>("F", {.description = "interpolation parameter"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of Neo-Hooke
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoNeoHooke",
        "isochoric part of neo-Hooke material acc. to Holzapfel", Core::Materials::mes_isoneohooke);

    m->add_component(parameter<double>("MUE", {.description = "Shear modulus"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of one-term Ogden material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoOgden",
        "isochoric part of the one-term Ogden material", Core::Materials::mes_isoogden);

    m->add_component(parameter<double>("MUE", {.description = "Shear modulus"}));
    m->add_component(parameter<double>("ALPHA", {.description = "Nonlinearity parameter"}));
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of Yeoh
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoYeoh",
        "isochoric part of  Yeoh material acc. to Holzapfel", Core::Materials::mes_isoyeoh);

    m->add_component(parameter<double>("C1", {.description = "Linear modulus"}));
    m->add_component(parameter<double>("C2", {.description = "Quadratic modulus"}));
    m->add_component(parameter<double>("C3", {.description = "Cubic modulus"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of iso1pow
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "ELAST_Iso1Pow", "isochoric part of general power material", Core::Materials::mes_iso1pow);

    m->add_component(parameter<double>("C", {.description = "material parameter"}));
    m->add_component(parameter<int>("D", {.description = "exponent"}));
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of iso2pow
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "ELAST_Iso2Pow", "isochoric part of general power material", Core::Materials::mes_iso2pow);

    m->add_component(parameter<double>("C", {.description = "material parameter"}));
    m->add_component(parameter<int>("D", {.description = "exponent"}));
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup1pow
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "ELAST_Coup1Pow", "part of general power material", Core::Materials::mes_coup1pow);

    m->add_component(parameter<double>("C", {.description = "material parameter"}));
    m->add_component(parameter<int>("D", {.description = "exponent"}));
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup2pow
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "ELAST_Coup2Pow", "part of general power material", Core::Materials::mes_coup2pow);

    m->add_component(parameter<double>("C", {.description = "material parameter"}));
    m->add_component(parameter<int>("D", {.description = "exponent"}));
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup3pow
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "ELAST_Coup3Pow", "part of general power material", Core::Materials::mes_coup3pow);

    m->add_component(parameter<double>("C", {.description = "material parameter"}));
    m->add_component(parameter<int>("D", {.description = "exponent"}));
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup13apow
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_Coup13aPow",
        "hyperelastic potential summand for multiplicative coupled invariants I1 and I3",
        Core::Materials::mes_coup13apow);

    m->add_component(parameter<double>("C", {.description = "material parameter"}));
    m->add_component(parameter<int>("D", {.description = "exponent of all"}));
    m->add_component(parameter<double>("A", {.description = "negative exponent of I3"}));
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of expo
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoExpoPow",
        "isochoric part of  exponential material acc. to Holzapfel",
        Core::Materials::mes_isoexpopow);

    m->add_component(parameter<double>("K1", {.description = "material parameter"}));
    m->add_component(parameter<double>("K2", {.description = "material parameter"}));
    m->add_component(parameter<int>("C", {.description = "exponent"}));
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of mooney rivlin
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoMooneyRivlin",
        "isochoric part of  Mooney-Rivlin material acc. to Holzapfel",
        Core::Materials::mes_isomooneyrivlin);

    m->add_component(
        parameter<double>("C1", {.description = "Linear modulus for first invariant"}));
    m->add_component(
        parameter<double>("C2", {.description = "Linear modulus for second invariant"}));
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric anisotropic material with one exponential fiber family
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoMuscle_Blemker",
        "anisotropic Blemker muscle material", Core::Materials::mes_isomuscleblemker);

    m->add_component(parameter<double>("G1", {.description = "muscle along fiber shear modulus"}));
    m->add_component(parameter<double>("G2", {.description = "muscle cross fiber shear modulus"}));
    m->add_component(parameter<double>(
        "P1", {.description = "linear material parameter for passive along-fiber response"}));
    m->add_component(parameter<double>(
        "P2", {.description = "exponential material parameter for passive along-fiber response"}));
    m->add_component(
        parameter<double>("SIGMAMAX", {.description = "maximal active isometric stress"}));
    m->add_component(parameter<double>("LAMBDAOFL", {.description = "optimal fiber stretch"}));
    m->add_component(parameter<double>("LAMBDASTAR",
        {.description = "stretch at which the normalized passive fiber force becomes linear"}));
    m->add_component(parameter<double>("ALPHA", {.description = "tetanised activation level,"}));
    m->add_component(parameter<double>(
        "BETA", {.description = "constant scaling tanh-type activation function"}));
    m->add_component(
        parameter<double>("ACTSTARTTIME", {.description = "starting time of muscle activation"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // test material to test elasthyper-toolbox
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoTestMaterial",
        "test material to test elasthyper-toolbox", Core::Materials::mes_isotestmaterial);

    m->add_component(parameter<double>("C1", {.description = "Modulus for first invariant"}));
    m->add_component(parameter<double>("C2", {.description = "Modulus for second invariant"}));
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // general fiber material for remodeling
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_RemodelFiber",
        "General fiber material for remodeling", Core::Materials::mes_remodelfiber);

    m->add_component(
        parameter<int>("NUMMAT", {.description = "number of materials/potentials in list"}));
    m->add_component(parameter<std::vector<int>>("MATIDS",
        {.description = "the list material/potential IDs", .size = from_parameter<int>("NUMMAT")}));
    m->add_component(parameter<double>(
        "TDECAY", {.description = "decay time of Poisson (degradation) process"}));
    m->add_component(parameter<double>(
        "GROWTHFAC", {.description = "time constant for collagen growth", .default_value = 0.0}));
    m->add_component(parameter<std::vector<double>>("COLMASSFRAC",
        {.description =
                "initial mass fraction of first collagen fiber family in constraint mixture",
            .default_value = std::vector{0.0},
            .size = from_parameter<int>("NUMMAT")}));
    m->add_component(parameter<double>("DEPOSITIONSTRETCH", {.description = "deposition stretch"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // volumetric contribution of Sussman Bathe
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_VolSussmanBathe",
        "volumetric part of  SussmanBathe material", Core::Materials::mes_volsussmanbathe);

    m->add_component(parameter<double>("KAPPA", {.description = "dilatation modulus"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // volumetric penalty contribution
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_VolPenalty",
        "Penalty formulation for the volumetric part", Core::Materials::mes_volpenalty);

    m->add_component(parameter<double>("EPSILON", {.description = "penalty parameter"}));
    m->add_component(parameter<double>("GAMMA", {.description = "penalty parameter"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // volumetric contribution of Ogden
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_VolOgden",
        "Ogden formulation for the volumetric part", Core::Materials::mes_vologden);

    m->add_component(parameter<double>("KAPPA", {.description = "dilatation modulus"}));
    m->add_component(parameter<double>("BETA", {.description = "empiric constant"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // volumetric power law contribution
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_VolPow",
        "Power law formulation for the volumetric part", Core::Materials::mes_volpow);

    m->add_component(parameter<double>("A", {.description = "prefactor of power law"}));
    m->add_component(parameter<double>("EXPON", {.description = "exponent of power law"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with one exponential fiber family
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupAnisoExpoActive",
        "anisotropic active fiber", Core::Materials::mes_coupanisoexpoactive);

    m->add_component(parameter<double>("K1", {.description = "linear constant"}));
    m->add_component(parameter<double>("K2", {.description = "exponential constant"}));
    m->add_component(parameter<double>("GAMMA", {.description = "angle"}));
    m->add_component(parameter<double>("K1COMP", {.description = "linear constant"}));
    m->add_component(parameter<double>("K2COMP", {.description = "exponential constant"}));
    m->add_component(parameter<int>(
        "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}));
    m->add_component(parameter<int>(
        "INIT", {.description = "initialization modus for fiber alignment", .default_value = 1}));
    m->add_component(parameter<bool>(
        "ADAPT_ANGLE", {.description = "adapt angle during remodeling", .default_value = false}));
    m->add_component(parameter<double>("S", {.description = "maximum contractile stress"}));
    m->add_component(parameter<double>(
        "LAMBDAMAX", {.description = "stretch at maximum active force generation"}));
    m->add_component(
        parameter<double>("LAMBDA0", {.description = "stretch at zero active force generation"}));
    m->add_component(parameter<double>(
        "DENS", {.description = "total reference mass density of constrained mixture"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with one exponential fiber family
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupAnisoExpo",
        "anisotropic part with one exp. fiber", Core::Materials::mes_coupanisoexpo);

    m->add_component(parameter<double>("K1", {.description = "linear constant"}));
    m->add_component(parameter<double>("K2", {.description = "exponential constant"}));
    m->add_component(parameter<double>("GAMMA", {.description = "angle"}));
    m->add_component(parameter<double>("K1COMP", {.description = "linear constant"}));
    m->add_component(parameter<double>("K2COMP", {.description = "exponential constant"}));
    m->add_component(parameter<int>(
        "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}));
    m->add_component(parameter<int>(
        "INIT", {.description = "initialization modus for fiber alignment", .default_value = 1}));
    m->add_component(parameter<bool>(
        "ADAPT_ANGLE", {.description = "adapt angle during remodeling", .default_value = false}));
    m->add_component(parameter<int>(
        "FIBER_ID", {.description = "Id of the fiber to be used (1 for first fiber, default)",
                        .default_value = 1}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with one exponential shear behavior between two fibers
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupAnisoExpoShear",
        "Exponential shear behavior between two fibers", Core::Materials::mes_coupanisoexposhear);

    m->add_component(parameter<double>("K1", {.description = "linear constant"}));
    m->add_component(parameter<double>("K2", {.description = "exponential constant"}));
    m->add_component(parameter<double>("GAMMA", {.description = "angle"}));
    m->add_component(parameter<double>("K1COMP", {.description = "linear constant"}));
    m->add_component(parameter<double>("K2COMP", {.description = "exponential constant"}));
    m->add_component(parameter<int>(
        "INIT", {.description = "initialization modus for fiber alignment", .default_value = 1}));
    m->add_component(parameter<std::vector<int>>("FIBER_IDS",
        {.description =
                "Ids of the two fibers to be used (1 for the first fiber, 2 for the second, "
                "default)",
            .size = 2}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with one pow-like fiber family
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupAnisoPow",
        "anisotropic part with one pow-like fiber", Core::Materials::mes_coupanisopow);

    m->add_component(parameter<double>("K", {.description = "linear constant"}));
    m->add_component(
        parameter<double>("D1", {.description = "exponential constant for fiber invariant"}));
    m->add_component(parameter<double>("D2", {.description = "exponential constant for system"}));
    m->add_component(parameter<double>(
        "ACTIVETHRES", {.description = "Deformation threshold for activating fibers. Default:"
                                       " 1.0 (off at compression); If 0.0 (always active)",
                           .default_value = 1.0}));
    m->add_component(parameter<int>(
        "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}));
    m->add_component(parameter<int>(
        "FIBER", {.description = "Number of the fiber family contained in the element",
                     .default_value = 1}));
    m->add_component(parameter<double>("GAMMA", {.description = "angle", .default_value = 0.0}));
    m->add_component(parameter<int>(
        "INIT", {.description = "initialization modus for fiber alignment", .default_value = 1}));
    m->add_component(parameter<bool>(
        "ADAPT_ANGLE", {.description = "adapt angle during remodeling", .default_value = false}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with two exponential fiber families
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupAnisoExpoTwoCoup",
        "anisotropic part with two exp. fibers", Core::Materials::mes_coupanisoexpotwocoup);

    m->add_component(
        parameter<double>("A4", {.description = "linear anisotropic constant for fiber 1"}));
    m->add_component(
        parameter<double>("B4", {.description = "exponential anisotropic constant for fiber 1"}));
    m->add_component(
        parameter<double>("A6", {.description = "linear anisotropic constant for fiber 2"}));
    m->add_component(
        parameter<double>("B6", {.description = "exponential anisotropic constant for fiber 2"}));
    m->add_component(parameter<double>(
        "A8", {.description = "linear anisotropic constant for fiber 1 relating fiber 2"}));
    m->add_component(parameter<double>(
        "B8", {.description = "exponential anisotropic constant for fiber 1 relating fiber 2"}));
    m->add_component(parameter<double>("GAMMA", {.description = "angle"}));
    m->add_component(parameter<int>(
        "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}));
    m->add_component(parameter<int>(
        "INIT", {.description = "initialization modus for fiber alignment", .default_value = 1}));
    m->add_component(parameter<bool>(
        "FIB_COMP", {.description = "fibers support compression: yes (true) or no (false)",
                        .default_value = true}));
    m->add_component(parameter<bool>(
        "ADAPT_ANGLE", {.description = "adapt angle during remodeling", .default_value = false}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with two exponential fiber families
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupAnisoNeoHooke",
        "anisotropic part with one neo Hookean fiber", Core::Materials::mes_coupanisoneohooke);

    m->add_component(parameter<double>("C", {.description = "linear constant"}));
    m->add_component(parameter<double>("GAMMA", {.description = "angle"}));
    m->add_component(parameter<int>(
        "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}));
    m->add_component(parameter<int>(
        "INIT", {.description = "initialization modus for fiber alignment", .default_value = 1}));
    m->add_component(parameter<bool>(
        "ADAPT_ANGLE", {.description = "adapt angle during remodeling", .default_value = false}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with the stress given by a simplified version of the contraction
  // law of Bestel-Clement-Sorine
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_AnisoActiveStress_Evolution",
        "anisotropic part with one fiber with coefficient given by a simplification of the "
        "activation-contraction law of Bestel-Clement-Sorine-2001",
        Core::Materials::mes_anisoactivestress_evolution);

    m->add_component(parameter<double>("SIGMA", {.description = "Contractility (maximal stress)"}));
    m->add_component(
        parameter<double>("TAUC0", {.description = "Initial value for the active stress"}));
    m->add_component(parameter<double>(
        "MAX_ACTIVATION", {.description = "Maximal value for the rescaled activation"}));
    m->add_component(parameter<double>(
        "MIN_ACTIVATION", {.description = "Minimal value for the rescaled activation"}));
    m->add_component(parameter<int>("SOURCE_ACTIVATION",
        {.description = "Where the activation comes from: 0=scatra , >0 Id for FUNCT"}));
    m->add_component(parameter<double>(
        "ACTIVATION_THRES", {.description = "Threshold for activation (contraction starts when "
                                            "activation function is larger than this "
                                            "value, relaxes otherwise)"}));
    m->add_component(parameter<bool>(
        "STRAIN_DEPENDENCY", {.description = "model strain dependency of contractility "
                                             "(Frank-Starling law): no (false) or yes (true)",
                                 .default_value = false}));
    m->add_component(parameter<double>("LAMBDA_LOWER",
        {.description = "lower fiber stretch for Frank-Starling law", .default_value = 1.0}));
    m->add_component(parameter<double>("LAMBDA_UPPER",
        {.description = "upper fiber stretch for Frank-Starling law", .default_value = 1.0}));
    m->add_component(parameter<double>("GAMMA", {.description = "angle", .default_value = 0.0}));
    m->add_component(parameter<int>(
        "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}));
    m->add_component(parameter<int>(
        "INIT", {.description = "initialization mode for fiber alignment", .default_value = 1}));
    m->add_component(parameter<bool>(
        "ADAPT_ANGLE", {.description = "adapt angle during remodeling", .default_value = false}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with variable stress coefficient
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupAnisoNeoHooke_VarProp",
        "anisotropic part with one neo Hookean fiber with variable coefficient",
        Core::Materials::mes_coupanisoneohooke_varprop);

    m->add_component(parameter<double>("C", {.description = "linear constant"}));
    m->add_component(parameter<int>("SOURCE_ACTIVATION",
        {.description = "Where the activation comes from: 0=scatra , >0 Id for FUNCT"}));
    m->add_component(
        parameter<double>("GAMMA", {.description = "azimuth angle", .default_value = 0.0}));
    m->add_component(
        parameter<double>("THETA", {.description = "polar angle", .default_value = 0.0}));
    m->add_component(parameter<int>(
        "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}));
    m->add_component(parameter<int>(
        "INIT", {.description = "initialization mode for fiber alignment", .default_value = 1}));
    m->add_component(parameter<bool>(
        "ADAPT_ANGLE", {.description = "adapt angle during remodeling", .default_value = false}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric anisotropic material with one exponential fiber family
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoAnisoExpo",
        "anisotropic part with one exp. fiber", Core::Materials::mes_isoanisoexpo);

    m->add_component(parameter<double>("K1", {.description = "linear constant"}));
    m->add_component(parameter<double>("K2", {.description = "exponential constant"}));
    m->add_component(parameter<double>("GAMMA", {.description = "angle"}));
    m->add_component(parameter<double>("K1COMP", {.description = "linear constant"}));
    m->add_component(parameter<double>("K2COMP", {.description = "exponential constant"}));
    m->add_component(parameter<int>(
        "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}));
    m->add_component(parameter<int>(
        "INIT", {.description = "initialization modus for fiber alignment", .default_value = 1}));
    m->add_component(parameter<bool>(
        "ADAPT_ANGLE", {.description = "adapt angle during remodeling", .default_value = false}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // structural tensor
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_StructuralTensor",
        "Parameter for structural tensor strategy in anisotropic materials",
        Core::Materials::mes_structuraltensorstratgy);

    m->add_component(parameter<std::string>("STRATEGY",
        {.description =
                "Strategy for evaluation of structural tensor: "
                "Standard (default), ByDistributionFunction, DispersedTransverselyIsotropic"}));

    // choose between:
    // "none"
    // "Bingham"
    // "vonMisesFisher"
    //  rauch 10/17
    m->add_component(parameter<std::string>(
        "DISTR", {.description = "Type of distribution function around mean direction: "
                                 "none, Bingham, vonMisesFisher",
                     .default_value = "none"}));

    m->add_component(parameter<double>(
        "C1", {.description = "constant 1 for distribution function", .default_value = 1.0}));
    m->add_component(parameter<double>(
        "C2", {.description = "constant 2 for distribution function", .default_value = 0.0}));
    m->add_component(parameter<double>(
        "C3", {.description = "constant 3 for distribution function", .default_value = 0.0}));
    m->add_component(parameter<double>(
        "C4", {.description = "constant 4 for distribution function", .default_value = 1e16}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // transversely isotropic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupTransverselyIsotropic",
        "transversely part of a simple othotropic, transversely "
        "isotropic hyperelastic constitutive equation",
        Core::Materials::mes_couptransverselyisotropic);

    m->add_component(parameter<double>("ALPHA", {.description = "1-st constant"}));
    m->add_component(parameter<double>("BETA", {.description = "2-nd constant"}));
    m->add_component(parameter<double>("GAMMA", {.description = "3-rd constant"}));
    m->add_component(parameter<double>("ANGLE", {.description = "fiber angle"}));
    m->add_component(parameter<int>(
        "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}));
    m->add_component(
        parameter<int>("FIBER", {.description = "exponential constant", .default_value = 1}));
    m->add_component(parameter<int>(
        "INIT", {.description = "initialization modus for fiber alignment", .default_value = 1}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled Varga material acc. to Holzapfel
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "ELAST_CoupVarga", "Varga material acc. to Holzapfel", Core::Materials::mes_coupvarga);

    m->add_component(parameter<double>("MUE", {.description = "Shear modulus"}));
    m->add_component(parameter<double>("BETA", {.description = "'Anti-modulus'"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric Varga material acc. to Holzapfel
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoVarga",
        "Isochoric Varga material acc. to Holzapfel", Core::Materials::mes_isovarga);

    m->add_component(parameter<double>("MUE", {.description = "Shear modulus"}));
    m->add_component(parameter<double>("BETA", {.description = "'Anti-modulus'"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isotropic viscous contribution of myocardial matrix (chapelle12)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("VISCO_CoupMyocard",
        "Isotropic viscous contribution of myocardial matrix", Core::Materials::mes_coupmyocard);

    m->add_component(parameter<double>("N", {.description = "material parameter"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric rate dependent viscos material, modified from Pioletti,1997
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("VISCO_IsoRateDep",
        "Isochoric rate dependent viscous material", Core::Materials::mes_isoratedep);

    m->add_component(parameter<double>("N", {.description = "material parameter"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // viscos contribution to visohyperelastic material according to SLS-Model
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "VISCO_GenMax", "Viscous contribution according to SLS-Model", Core::Materials::mes_genmax);

    m->add_component(parameter<double>("TAU", {.description = "relaxation parameter"}));
    m->add_component(
        parameter<double>("BETA", {.description = "emphasis of viscous to elastic part"}));
    m->add_component(parameter<std::string>("SOLVE",
        {.description = "Solution of evolution equation via: OST (default) or CONVOL (convolution "
                        "integral)"}));


    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // viscos contribution to visohyperelastic material according to FSLS-Model
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "VISCO_Fract", "Viscous contribution according to FSLS-Model", Core::Materials::mes_fract);

    m->add_component(parameter<double>("TAU", {.description = "relaxation parameter"}));
    m->add_component(parameter<double>("ALPHA", {.description = "fractional order derivative"}));
    m->add_component(
        parameter<double>("BETA", {.description = "emphasis of viscous to elastic part"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // viscous contribution of a branch of a generalized Maxwell model
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("VISCO_PART",
        "Viscous contribution of a viscoelastic Branch", Core::Materials::mes_viscopart);

    m->add_component(parameter<double>(
        "TAU", {.description = "dynamic viscosity divided by young's modulus of the branch"}));

    Mat::append_material_definition(matlist, m);
  }
  /*--------------------------------------------------------------------*/
  // viscoelatic branches of a generalized Maxwell model
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("VISCO_GeneralizedGenMax",
        "Viscoelastic Branches of generalized Maxwell", Core::Materials::mes_generalizedgenmax);

    m->add_component(
        parameter<int>("NUMBRANCH", {.description = "number of viscoelastic branches"}));
    m->add_component(parameter<std::vector<int>>("MATIDS",
        {.description = "the list material IDs", .size = from_parameter<int>("NUMBRANCH")}));
    m->add_component(parameter<std::string>("SOLVE",
        {.description = "Solution for evolution equation: OST (default) or CONVOL (convolution "
                        "integral)"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // description of a viscoelatic branch of a generalized Maxwell model
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("VISCO_BRANCH",
        "Viscoelastic Branch (viscous and elastic contribution)", Core::Materials::mes_viscobranch);

    m->add_component(parameter<int>(
        "NUMMAT", {.description = "number of materials in the viscoelastic branch"}));
    m->add_component(parameter<std::vector<int>>(
        "MATIDS", {.description = "the list material IDs", .size = from_parameter<int>("NUMMAT")}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // 1D Artery material with constant properties
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_CNST_ART", "artery with constant properties", Core::Materials::m_cnst_art);

    m->add_component(parameter<double>(
        "VISCOSITY", {.description = "viscosity (for CONSTANT viscosity law taken as "
                                     "blood viscosity, for BLOOD viscosity law "
                                     "taken as the viscosity of blood plasma)"}));
    m->add_component(parameter<double>("DENS", {.description = "density of blood"}));
    m->add_component(
        parameter<double>("YOUNG", {.description = "artery Youngs modulus of elasticity"}));
    m->add_component(parameter<double>("NUE", {.description = "Poissons ratio of artery fiber"}));
    m->add_component(parameter<double>("TH", {.description = "artery thickness"}));
    m->add_component(
        parameter<double>("PEXT1", {.description = "artery fixed external pressure 1"}));
    m->add_component(
        parameter<double>("PEXT2", {.description = "artery fixed external pressure 2"}));
    m->add_component(parameter<std::string>(
        "VISCOSITYLAW", {.description = "type of viscosity law, CONSTANT (default) or BLOOD",
                            .default_value = "CONSTANT"}));
    m->add_component(parameter<double>("BLOOD_VISC_SCALE_DIAM_TO_MICRONS",
        {.description = "used to scale the diameter for blood viscosity law to microns if your "
                        "problem is not "
                        "given in microns, e.g., if you use mms, set this parameter to 1.0e3",
            .default_value = 1.0}));
    m->add_component(parameter<std::string>("VARYING_DIAMETERLAW",
        {.description = "type of varying diameter law, CONSTANT (default) or BY_FUNCTION",
            .default_value = "CONSTANT"}));
    m->add_component(parameter<int>("VARYING_DIAMETER_FUNCTION",
        {.description = "function for varying diameter law", .default_value = -1}));
    m->add_component(parameter<double>(
        "COLLAPSE_THRESHOLD", {.description = "Collapse threshold for diameter (below this "
                                              "diameter element is assumed to be collapsed "
                                              "with zero diameter and is not evaluated)",
                                  .default_value = -1.0}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // Fourier's law for linear and possibly anisotropic heat transport
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Fourier",
        "anisotropic linear Fourier's law of heat conduction", Core::Materials::m_thermo_fourier);

    m->add_component(parameter<double>("CAPA", {.description = "volumetric heat capacity"}));
    m->add_component(parameter<int>("CONDUCT_PARA_NUM",
        {.description = "Parameter representing the relevant number of entries in the thermal "
                        "conductivity tensor. Setting its value to 1 resembles a scalar "
                        "conductivity, 2 or 3 a diagonal conductivity and 4 or 9 the full "
                        "conductivity tensor in two and three dimensions respectively."}));
    m->add_component(parameter<std::vector<double>>("CONDUCT",
        {.description = "Vector of values representing the thermal conductivity tensor in a "
                        "row-wise ordering, with the vector size given by CONDUCT_PARA_NUM.",
            .size = from_parameter<int>("CONDUCT_PARA_NUM")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material for heat transport due to Fourier-type thermal conduction and the Soret effect
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_soret",
        "material for heat transport due to Fourier-type thermal conduction and the Soret effect",
        Core::Materials::m_soret);

    m->add_component(parameter<double>("CAPA", {.description = "volumetric heat capacity"}));
    m->add_component(parameter<int>("CONDUCT_PARA_NUM",
        {.description = "Parameter representing the relevant number of entries in the thermal "
                        "conductivity tensor. Setting its value to 1 resembles a scalar "
                        "conductivity, 2 or 3 a diagonal conductivity and 4 or 9 the full "
                        "conductivity tensor in two and three dimensions respectively."}));
    m->add_component(parameter<std::vector<double>>("CONDUCT",
        {.description = "Vector of values representing the thermal conductivity tensor in a "
                        "row-wise ordering, with the vector size given by CONDUCT_PARA_NUM.",
            .size = from_parameter<int>("CONDUCT_PARA_NUM")}));
    m->add_component(parameter<double>("SORET", {.description = "Soret coefficient"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // collection of hyperelastic materials for membranes
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Membrane_ElastHyper",
        "list/collection of hyperelastic materials for membranes, i.e. material IDs",
        Core::Materials::m_membrane_elasthyper);

    m->add_component(
        parameter<int>("NUMMAT", {.description = "number of materials/potentials in list"}));
    m->add_component(parameter<std::vector<int>>("MATIDS",
        {.description = "the list material/potential IDs", .size = from_parameter<int>("NUMMAT")}));
    m->add_component(parameter<double>("DENS", {.description = "material mass density"}));
    m->add_component(parameter<int>("POLYCONVEX",
        {.description = "1.0 if polyconvexity of system is checked", .default_value = 0}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // active strain membrane material for gastric electromechanics
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Membrane_ActiveStrain",
        "active strain membrane material", Core::Materials::m_membrane_activestrain);

    m->add_component(
        parameter<int>("MATIDPASSIVE", {.description = "MATID for the passive material"}));
    m->add_component(parameter<int>(
        "SCALIDVOLTAGE", {.description = "ID of the scalar that represents the (SMC) voltage"}));
    m->add_component(parameter<double>("DENS", {.description = "material mass density"}));
    m->add_component(parameter<double>("BETA1", {.description = "Ca2+ dynamics"}));
    m->add_component(parameter<double>("BETA2", {.description = "opening dynamics of the VDCC"}));
    m->add_component(
        parameter<double>("VOLTHRESH", {.description = "voltage threshold for activation"}));
    m->add_component(parameter<double>(
        "ALPHA1", {.description = "intensity of contraction in fiber direction 1"}));
    m->add_component(parameter<double>(
        "ALPHA2", {.description = "intensity of contraction in fiber direction 2"}));
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // growth and remodeling (homogenized constrained mixture model)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_GrowthRemodel_ElastHyper",
        "growth and remodeling", Core::Materials::m_growthremodel_elasthyper);

    m->add_component(
        parameter<int>("NUMMATRF", {.description = "number of remodelfiber materials in list"}));
    m->add_component(parameter<int>(
        "NUMMATEL3D", {.description = "number of 3d elastin matrix materials/potentials in list",
                          .default_value = 0}));
    m->add_component(parameter<int>(
        "NUMMATEL2D", {.description = "number of 2d elastin matrix materials/potentials in list"}));
    m->add_component(parameter<std::vector<int>>(
        "MATIDSRF", {.description = "the list remodelfiber material IDs",
                        .default_value = std::vector{0},
                        .size = from_parameter<int>("NUMMATRF")}));
    m->add_component(parameter<std::vector<int>>(
        "MATIDSEL3D", {.description = "the list 3d elastin matrix material/potential IDs",
                          .default_value = std::vector{-1},
                          .size = from_parameter<int>("NUMMATEL3D")}));
    m->add_component(parameter<std::vector<int>>(
        "MATIDSEL2D", {.description = "the list 2d elastin matrix material/potential IDs",
                          .default_value = std::vector{0},
                          .size = from_parameter<int>("NUMMATEL2D")}));
    m->add_component(parameter<int>(
        "MATIDELPENALTY", {.description = "penalty material ID", .default_value = -1}));
    m->add_component(parameter<double>("ELMASSFRAC",
        {.description = "initial mass fraction of elastin matrix in constraint mixture"}));
    m->add_component(parameter<double>("DENS", {.description = "material mass density"}));
    m->add_component(parameter<double>(
        "PRESTRETCHELASTINCIR", {.description = "circumferential prestretch of elastin matrix"}));
    m->add_component(parameter<double>(
        "PRESTRETCHELASTINAX", {.description = "axial prestretch of elastin matrix"}));
    m->add_component(parameter<double>("THICKNESS",
        {.description = "reference wall thickness of the idealized cylindrical aneurysm [m]",
            .default_value = -1.}));
    m->add_component(parameter<double>(
        "MEANPRESSURE", {.description = "mean blood pressure [Pa]", .default_value = -1.0}));
    m->add_component(parameter<double>(
        "RADIUS", {.description = "inner radius of the idealized cylindrical aneurysm [m]",
                      .default_value = -1.0}));
    m->add_component(parameter<int>(
        "DAMAGE", {.description = "1: elastin damage after prestressing,0: no elastin damage"}));
    m->add_component(parameter<int>("GROWTHTYPE",
        {.description =
                "flag to decide what type of collagen growth is used: 1: anisotropic growth; "
                "0: isotropic growth"}));
    m->add_component(parameter<int>("LOCTIMEINT",
        {.description = "flag to decide what type of local time integration scheme is used: 1: "
                        "Backward Euler Method; 0: Forward Euler Method"}));
    m->add_component(parameter<int>(
        "MEMBRANE", {.description = "Flag whether Hex or Membrane elements are used ( "
                                    "Membrane: 1, Hex: Everything else )",
                        .default_value = -1}));
    m->add_component(
        parameter<int>("CYLINDER", {.description = "Flag that geometry is a cylinder. 1: "
                                                   "aligned in x-direction; 2: y-direction; 3: "
                                                   "z-direction",
                                       .default_value = -1}));
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // multiplicative split of deformation gradient in elastic and inelastic parts
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_MultiplicativeSplitDefgradElastHyper",
        "multiplicative split of deformation gradient",
        Core::Materials::m_multiplicative_split_defgrad_elasthyper);

    m->add_component(parameter<int>(
        "NUMMATEL", {.description = "number of elastic materials/potentials in list"}));
    m->add_component(parameter<std::vector<int>>(
        "MATIDSEL", {.description = "the list of elastic material/potential IDs",
                        .default_value = std::vector{-1},
                        .size = from_parameter<int>("NUMMATEL")}));
    m->add_component(parameter<int>(
        "NUMFACINEL", {.description = "number of factors of inelastic deformation gradient"}));
    m->add_component(parameter<std::vector<int>>("INELDEFGRADFACIDS",
        {.description = "the list of inelastic deformation gradient factor IDs",
            .default_value = std::vector{0},
            .size = from_parameter<int>("NUMFACINEL")}));
    m->add_component(parameter<double>("DENS", {.description = "material mass density"}));
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // simple inelastic material law featuring no volume change
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_InelasticDefgradNoGrowth",
        "no volume change, i.e. the inelastic deformation gradient is the identity tensor",
        Core::Materials::mfi_no_growth);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // simple isotropic, volumetric growth; growth is linearly dependent on scalar mapped to material
  // configuration, constant material density
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_InelasticDefgradLinScalarIso",
        "scalar dependent isotropic growth law; volume change linearly dependent on scalar (in "
        "material configuration)",
        Core::Materials::mfi_lin_scalar_iso);

    m->add_component(
        parameter<int>("SCALAR1", {.description = "number of growth inducing scalar"}));
    m->add_component(parameter<double>("SCALAR1_MolarGrowthFac",
        {.description = "isotropic molar growth factor due to scalar 1"}));
    m->add_component(parameter<double>("SCALAR1_RefConc",
        {.description = "reference concentration of scalar 1 causing no strains"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // simple anisotropic, volumetric growth; growth direction prescribed in input-file;
  // growth is linearly dependent on scalar mapped to material configuration, constant material
  // density
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_InelasticDefgradLinScalarAniso",
        "scalar dependent anisotropic growth law; growth in direction as given in input-file; "
        "volume change linearly dependent on scalar (in material configuration)",
        Core::Materials::mfi_lin_scalar_aniso);

    m->add_component(
        parameter<int>("SCALAR1", {.description = "number of growth inducing scalar"}));
    m->add_component(parameter<double>("SCALAR1_MolarGrowthFac",
        {.description = "anisotropic molar growth factor due to scalar 1"}));
    m->add_component(parameter<double>("SCALAR1_RefConc",
        {.description = "reference concentration of scalar 1 causing no strains"}));
    m->add_component(
        parameter<int>("NUMSPACEDIM", {.description = "Number of space dimension (only 3 valid)"}));
    m->add_component(parameter<std::vector<double>>(
        "GrowthDirection", {.description = "vector that defines the growth direction",
                               .size = from_parameter<int>("NUMSPACEDIM")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // non-linear isotropic volumetric growth; growth is dependent on the degree of lithiation,
  // constant material density, nonlinear behavior prescribed by polynomial in input file
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_InelasticDefgradPolyIntercalFracIso",
        "scalar dependent isotropic growth law; volume change nonlinearly dependent on the "
        "intercalation fraction, that is calculated using the scalar concentration (in material "
        "configuration)",
        Core::Materials::mfi_poly_intercal_frac_iso);

    m->add_component(
        parameter<int>("SCALAR1", {.description = "number of growth inducing scalar"}));
    m->add_component(parameter<double>("SCALAR1_RefConc",
        {.description = "reference concentration of scalar 1 causing no strains"}));
    m->add_component(
        parameter<int>("POLY_PARA_NUM", {.description = "number of polynomial coefficients"}));
    m->add_component(parameter<std::vector<double>>(
        "POLY_PARAMS", {.description = "coefficients of polynomial",
                           .size = from_parameter<int>("POLY_PARA_NUM")}));
    m->add_component(
        parameter<double>("X_min", {.description = "lower bound of validity of polynomial"}));
    m->add_component(
        parameter<double>("X_max", {.description = "upper bound of validity of polynomial"}));
    m->add_component(parameter<int>(
        "MATID", {.description = "material ID of the corresponding scatra material"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // non-linear anisotropic volumetric growth; growth direction prescribed in input-file;
  // growth is dependent on the degree of lithiation, constant material density, nonlinear behavior
  // prescribed by polynomial in input file
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_InelasticDefgradPolyIntercalFracAniso",
        "scalar dependent anisotropic growth law; growth in direction as given in input-file; "
        "volume change nonlinearly dependent on the intercalation fraction, that is calculated "
        "using the scalar concentration (in material configuration)",
        Core::Materials::mfi_poly_intercal_frac_aniso);

    m->add_component(
        parameter<int>("SCALAR1", {.description = "number of growth inducing scalar"}));
    m->add_component(parameter<double>("SCALAR1_RefConc",
        {.description = "reference concentration of scalar 1 causing no strains"}));
    m->add_component(
        parameter<int>("NUMSPACEDIM", {.description = "Number of space dimension (only 3 valid)"}));
    m->add_component(parameter<std::vector<double>>(
        "GrowthDirection", {.description = "vector that defines the growth direction",
                               .size = from_parameter<int>("NUMSPACEDIM")}));
    m->add_component(
        parameter<int>("POLY_PARA_NUM", {.description = "number of polynomial coefficients"}));
    m->add_component(parameter<std::vector<double>>(
        "POLY_PARAMS", {.description = "coefficients of polynomial",
                           .size = from_parameter<int>("POLY_PARA_NUM")}));
    m->add_component(
        parameter<double>("X_min", {.description = "lower bound of validity of polynomial"}));
    m->add_component(
        parameter<double>("X_max", {.description = "upper bound of validity of polynomial"}));
    m->add_component(parameter<int>(
        "MATID", {.description = "material ID of the corresponding scatra material"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_InelasticDefgradLinTempIso",
        "Temperature dependent growth law. Volume change linearly dependent on temperature",
        Core::Materials::mfi_lin_temp_iso);

    m->add_component(parameter<double>(
        "Temp_GrowthFac", {.description = "isotropic growth factor due to temperature"}));
    m->add_component(
        parameter<double>("RefTemp", {.description = "reference temperature causing no strains"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_InelasticDefgradTimeFunct",
        "Time-dependent growth law. determinant of volume change dependent on time function "
        "defined "
        "by 'FUNCT_NUM",
        Core::Materials::mfi_time_funct);

    m->add_component(parameter<int>(
        "FUNCT_NUM", {.description = "Time-dependent function of the determinant of the "
                                     "inelastic deformation gradient"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_InelasticDefgradTransvIsotropElastViscoplast",
        "Versatile transversely isotropic (or isotropic) viscoplasticity model for finite "
        "deformations with isotropic hardening, using user-defined viscoplasticity laws (flow rule "
        "+ hardening model)",
        Core::Materials::mfi_transv_isotrop_elast_viscoplast);

    m->add_component(parameter<int>(
        "VISCOPLAST_LAW_ID", {.description = "MAT ID of the corresponding viscoplastic law"}));
    m->add_component(parameter<int>("FIBER_READER_ID",
        {.description =
                "MAT ID of the used fiber direction reader for transversely isotropic behavior"}));
    m->add_component(parameter<double>("YIELD_COND_A",
        {.description =
                "transversely isotropic version of the Hill(1948) yield condition: parameter A, "
                "following "
                "the notation in Dafalias 1989, International Journal of Plasticity, Vol. 5"}));
    m->add_component(parameter<double>("YIELD_COND_B",
        {.description =
                "transversely isotropic version of the Hill(1948) yield condition: parameter B, "
                "following "
                "the notation in Dafalias 1989, International Journal of Plasticity, Vol. 5"}));
    m->add_component(parameter<double>("YIELD_COND_F",
        {.description =
                "transversely isotropic version of the Hill(1948) yield condition: parameter F, "
                "following "
                "the notation in Dafalias 1989, International Journal of Plasticity, Vol. 5"}));
    m->add_component(parameter<std::string>("ANISOTROPY",
        {.description =
                "Anisotropy type: transversely isotropic (transvisotrop; transverseisotropic; "
                "transverselyisotropic) | isotropic (isotrop; isotropic; Default)"}));
    m->add_component(parameter<bool>(
        "LOG_SUBSTEP", {.description = "boolean: time integration of internal variables using "
                                       "logarithmic substepping (True) or "
                                       "standard substepping (False)?"}));
    m->add_component(parameter<int>(
        "MAX_HALVE_NUM_SUBSTEP", {.description = "maximum number of times the global time step can "
                                                 "be halved in the substepping procedure"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_ViscoplasticLawReformulatedJohnsonCook",
        "Reformulation of the Johnson-Cook viscoplastic law (comprising flow rule $\\dot{P} = "
        "\\dot{P}_0 \\exp \\left( \\frac{ \\Sigma_{eq}}{C \\Sigma_y} - \\frac{1}{C} \\right) - "
        "\\dot{P}_0$ and hardening "
        "law), as shown in Mareau et al. (Mechanics of Materials 143, 2020)",
        Core::Materials::mvl_reformulated_Johnson_Cook);

    m->add_component(parameter<double>(
        "STRAIN_RATE_PREFAC", {.description = "reference plastic strain rate $\\dot{P}_0$ "}));
    m->add_component(parameter<double>(
        "STRAIN_RATE_EXP_FAC", {.description = "exponential factor of plastic strain rate $C$"}));
    m->add_component(parameter<double>(
        "INIT_YIELD_STRENGTH", {.description = "initial yield strength of the material $A_0$"}));
    m->add_component(parameter<double>("ISOTROP_HARDEN_PREFAC",
        {.description = "prefactor of the isotropic hardening stress $B_0$"}));
    m->add_component(parameter<double>(
        "ISOTROP_HARDEN_EXP", {.description = "exponent of the isotropic hardening stress $n$"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // integration point based and scalar dependent interpolation between to materials
  {
    auto mm = std::make_shared<Mat::MaterialDefinition>("MAT_ScDepInterp",
        "integration point based and scalar dependent interpolation between to materials",
        Core::Materials::m_sc_dep_interp);

    mm->add_component(
        parameter<int>("IDMATZEROSC", {.description = "material for lambda equal to zero"}));
    mm->add_component(
        parameter<int>("IDMATUNITSC", {.description = "material for lambda equal to one"}));

    Mat::append_material_definition(matlist, mm);
  }

  /*----------------------------------------------------------------------*/
  // growth and remodeling of arteries
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_ConstraintMixture",
        "growth and remodeling of arteries", Core::Materials::m_constraintmixture);

    m->add_component(parameter<double>("DENS", {.description = "Density"}));
    m->add_component(parameter<double>("MUE", {.description = "Shear Modulus"}));
    m->add_component(parameter<double>("NUE", {.description = "Poisson's ratio"}));
    m->add_component(parameter<double>("PHIE", {.description = "mass fraction of elastin"}));
    m->add_component(parameter<double>("PREELA", {.description = "prestretch of elastin"}));
    m->add_component(
        parameter<double>("K1", {.description = "Parameter for linear collagen fiber stiffness"}));
    m->add_component(parameter<double>(
        "K2", {.description = "Parameter for exponential collagen fiber stiffness"}));
    m->add_component(parameter<int>("NUMHOM", {.description = "Number of homeostatic parameters"}));
    m->add_component(parameter<std::vector<double>>("PRECOLL",
        {.description = "prestretch of collagen fibers", .size = from_parameter<int>("NUMHOM")}));
    m->add_component(
        parameter<double>("DAMAGE", {.description = "damage stretch of collagen fibers"}));
    m->add_component(parameter<double>(
        "K1M", {.description = "Parameter for linear smooth muscle fiber stiffness"}));
    m->add_component(parameter<double>(
        "K2M", {.description = "Parameter for exponential smooth muscle fiber stiffness"}));
    m->add_component(parameter<double>("PHIM", {.description = "mass fraction of smooth muscle"}));
    m->add_component(
        parameter<double>("PREMUS", {.description = "prestretch of smooth muscle fibers"}));
    m->add_component(parameter<double>("SMAX", {.description = "maximal active stress"}));
    m->add_component(parameter<double>("KAPPA", {.description = "dilatation modulus"}));
    m->add_component(parameter<double>("LIFETIME", {.description = "lifetime of collagen fibers"}));
    m->add_component(parameter<double>("GROWTHFAC", {.description = "growth factor for stress"}));
    m->add_component(parameter<std::vector<double>>(
        "HOMSTR", {.description = "homeostatic target value of scalar stress measure",
                      .size = from_parameter<int>("NUMHOM")}));
    m->add_component(
        parameter<double>("SHEARGROWTHFAC", {.description = "growth factor for shear"}));
    m->add_component(
        parameter<double>("HOMRAD", {.description = "homeostatic target value of inner radius"}));
    m->add_component(parameter<double>(
        "STARTTIME", {.description = "at this time turnover of collagen starts"}));
    m->add_component(
        parameter<std::string>("INTEGRATION", {.description = "time integration scheme: "
                                                              "Explicit (default), or Implicit"}));
    m->add_component(parameter<double>("TOL",
        {.description = "tolerance for local Newton iteration, only for implicit integration"}));
    m->add_component(
        parameter<std::string>("GROWTHFORCE", {.description = "driving force of growth: "
                                                              "Single (default), All, ElaCol"}));
    m->add_component(parameter<std::string>(
        "ELASTINDEGRAD", {.description = "how elastin is degraded: "
                                         "None (default), Rectangle, Time"}));
    m->add_component(
        parameter<std::string>("MASSPROD", {.description = "how mass depends on driving force: "
                                                           "Lin (default), CosCos"}));
    m->add_component(parameter<std::string>("INITSTRETCH",
        {.description = "how to set stretches in the beginning (None, Homeo, UpdatePrestretch)",
            .default_value = "None"}));
    m->add_component(parameter<int>(
        "CURVE", {.description = "number of timecurve for increase of prestretch in time"}));
    m->add_component(
        parameter<std::string>("DEGOPTION", {.description = "Type of degradation function: "
                                                            "Lin (default), Cos, Exp, ExpVar"}));
    m->add_component(
        parameter<double>("MAXMASSPRODFAC", {.description = "maximal factor of mass production"}));
    m->add_component(parameter<double>(
        "ELASTINFAC", {.description = "factor for elastin content", .default_value = 0.0}));
    m->add_component(parameter<bool>("STOREHISTORY",
        {.description = "store all history variables, not recommended for forward simulations",
            .default_value = false}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // hyperelastic material for poroelasticity
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_StructPoro",
        "wrapper for structure poroelastic material", Core::Materials::m_structporo);

    m->add_component(parameter<int>("MATID", {.description = "ID of structure material"}));
    m->add_component(parameter<int>("POROLAWID", {.description = "ID of porosity law"}));
    m->add_component(
        parameter<double>("INITPOROSITY", {.description = "initial porosity of porous medium"}));

    Mat::append_material_definition(matlist, m);
  }
  /*----------------------------------------------------------------------*/
  // linear law for porosity in porous media problems
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PoroLawLinear",
        "linear constitutive law for porosity", Core::Materials::m_poro_law_linear);

    m->add_component(
        parameter<double>("BULKMODULUS", {.description = "bulk modulus of porous medium"}));

    Mat::append_material_definition(matlist, m);
  }
  /*----------------------------------------------------------------------*/
  // constant law for porosity in porous media problems
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PoroLawConstant",
        "constant constitutive law for porosity", Core::Materials::m_poro_law_constant);

    Mat::append_material_definition(matlist, m);
  }
  /*----------------------------------------------------------------------*/
  // neo-hookean law for porosity in porous media problems
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PoroLawNeoHooke",
        "NeoHookean-like constitutive law for porosity",
        Core::Materials::m_poro_law_logNeoHooke_Penalty);

    m->add_component(
        parameter<double>("BULKMODULUS", {.description = "bulk modulus of porous medium"}));
    m->add_component(parameter<double>(
        "PENALTYPARAMETER", {.description = "penalty parameter of porous medium"}));

    Mat::append_material_definition(matlist, m);
  }
  /*----------------------------------------------------------------------*/
  // incompressible skeleton law for porosity in porous media problems
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PoroLawIncompSkel",
        "porosity law for incompressible skeleton phase",
        Core::Materials::m_poro_law_incompr_skeleton);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // incompressible skeleton law for porosity in porous media problems
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PoroLawLinBiot",
        "linear biot model for porosity law", Core::Materials::m_poro_law_linear_biot);

    m->add_component(parameter<double>(
        "INVBIOTMODULUS", {.description = "inverse Biot modulus of porous medium"}));
    m->add_component(
        parameter<double>("BIOTCEOFF", {.description = "Biot coefficient of porous medium"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // incompressible skeleton law for porosity depending on the density
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PoroLawDensityDependent",
        "porosity depending on the density", Core::Materials::m_poro_law_density_dependent);

    m->add_component(parameter<int>("DENSITYLAWID", {.description = "material ID of density law"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // density law for constant density in porous multiphase medium
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PoroDensityLawConstant",
        "density law for constant density in porous multiphase medium",
        Core::Materials::m_poro_densitylaw_constant);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // density law for constant density in porous multiphase medium
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PoroDensityLawExp",
        "density law for pressure dependent exponential function",
        Core::Materials::m_poro_densitylaw_exp);

    m->add_component(
        parameter<double>("BULKMODULUS", {.description = "bulk modulus of porous medium"}));
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // permeability law for constant permeability in porous multiphase medium
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroRelPermeabilityLawConstant",
        "permeability law for constant permeability in porous multiphase medium",
        Core::Materials::m_fluidporo_relpermeabilitylaw_constant);

    m->add_component(parameter<double>("VALUE", {.description = "constant value of permeability"}));
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // permeability law for permeability depending on saturation according to (saturation)^exp
  // in porous multiphase medium
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroRelPermeabilityLawExp",
        "permeability law depending on saturation in porous multiphase medium",
        Core::Materials::m_fluidporo_relpermeabilitylaw_exp);

    m->add_component(
        parameter<double>("EXP", {.description = "exponent of the saturation of this phase"}));
    m->add_component(parameter<double>(
        "MIN_SAT", {.description = "minimum saturation which is used for calculation"}));
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // viscosity law for constant viscosity in porous multiphase medium
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroViscosityLawConstant",
        "viscosity law for constant viscosity in porous multiphase medium",
        Core::Materials::m_fluidporo_viscositylaw_constant);

    m->add_component(parameter<double>("VALUE", {.description = "constant value of viscosity"}));
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // viscosity law for viscosity-dependency modelling cell adherence
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroViscosityLawCellAdherence",
        "visosity law depending on pressure gradient in porous multiphase medium",
        Core::Materials::m_fluidporo_viscositylaw_celladh);

    m->add_component(parameter<double>(
        "VISC_0", {.description = "Visc0 parameter for modelling cell adherence"}));
    m->add_component(
        parameter<double>("XI", {.description = "xi parameter for modelling cell adherence"}));
    m->add_component(
        parameter<double>("PSI", {.description = "psi parameter for modelling cell adherence"}));
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // hyperelastic material for poroelasticity with reaction
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_StructPoroReaction",
        "wrapper for structure porelastic material with reaction",
        Core::Materials::m_structpororeaction);

    m->add_component(parameter<int>("MATID", {.description = "ID of structure material"}));
    m->add_component(parameter<int>("POROLAWID", {.description = "ID of porosity law"}));
    m->add_component(
        parameter<double>("INITPOROSITY", {.description = "initial porosity of porous medium"}));
    m->add_component(parameter<int>("DOFIDREACSCALAR",
        {.description = "Id of DOF within scalar transport problem, which controls the reaction"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // hyperelastic material for poroelasticity with reaction
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_StructPoroReactionECM",
        "wrapper for structure porelastic material with reaction",
        Core::Materials::m_structpororeactionECM);

    m->add_component(parameter<int>("MATID", {.description = "ID of structure material"}));
    m->add_component(parameter<int>("POROLAWID", {.description = "ID of porosity law"}));
    m->add_component(
        parameter<double>("INITPOROSITY", {.description = "initial porosity of porous medium"}));
    m->add_component(parameter<double>("DENSCOLLAGEN", {.description = "density of collagen"}));
    m->add_component(parameter<int>("DOFIDREACSCALAR",
        {.description = "Id of DOF within scalar transport problem, which controls the reaction"}));
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // fluid flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_FluidPoro", "flow in deformable porous media", Core::Materials::m_fluidporo);

    m->add_component(parameter<double>("DYNVISCOSITY", {.description = "dynamic viscosity"}));
    m->add_component(parameter<double>("DENSITY", {.description = "density"}));
    m->add_component(parameter<double>(
        "PERMEABILITY", {.description = "permeability of medium", .default_value = 0.0}));
    m->add_component(parameter<double>("AXIALPERMEABILITY",
        {.description = "axial permeability for transverse isotropy", .default_value = 0.0}));
    m->add_component(parameter<double>("ORTHOPERMEABILITY1",
        {.description = "first permeability for orthotropy", .default_value = 0.0}));
    m->add_component(parameter<double>("ORTHOPERMEABILITY2",
        {.description = "second permeability for orthotropy", .default_value = 0.0}));
    m->add_component(parameter<double>("ORTHOPERMEABILITY3",
        {.description = "third permeability for orthotropy", .default_value = 0.0}));
    m->add_component(parameter<std::string>(
        "TYPE", {.description = "Problem type: Darcy (default) or Darcy-Brinkman",
                    .default_value = "Darcy"}));
    // optional parameter
    m->add_component(parameter<std::string>("PERMEABILITYFUNCTION",
        {.description = "Permeability function: Const(Default) or Kozeny_Carman",
            .default_value = "Const"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // multiphase flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroMultiPhase",
        "multi phase flow in deformable porous media", Core::Materials::m_fluidporo_multiphase);

    m->add_component(parameter<bool>("LOCAL",
        {.description = "individual materials allocated per element or only at global scope"}));
    m->add_component(parameter<double>("PERMEABILITY", {.description = "permeability of medium"}));
    m->add_component(parameter<int>("NUMMAT", {.description = "number of materials in list"}));
    m->add_component(parameter<std::vector<int>>(
        "MATIDS", {.description = "the list material IDs", .size = from_parameter<int>("NUMMAT")}));
    m->add_component(parameter<int>(
        "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE", {.description = "number of fluid phases"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // multiphase flow in a poroelastic material with reactions
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroMultiPhaseReactions",
        "multi phase flow in deformable porous media and list of reactions",
        Core::Materials::m_fluidporo_multiphase_reactions);

    m->add_component(parameter<bool>("LOCAL",
        {.description = "individual materials allocated per element or only at global scope"}));
    m->add_component(parameter<double>("PERMEABILITY", {.description = "permeability of medium"}));
    m->add_component(parameter<int>("NUMMAT", {.description = "number of materials in list"}));
    m->add_component(parameter<std::vector<int>>(
        "MATIDS", {.description = "the list material IDs", .size = from_parameter<int>("NUMMAT")}));
    m->add_component(parameter<int>(
        "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE", {.description = "number of fluid phases"}));
    m->add_component(
        parameter<int>("NUMREAC", {.description = "number of reactions for these elements"}));
    m->add_component(
        parameter<std::vector<int>>("REACIDS", {.description = "advanced reaction list",
                                                   .default_value = std::vector{0},
                                                   .size = from_parameter<int>("NUMREAC")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one reaction for multiphase flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroSingleReaction",
        "advanced reaction material", Core::Materials::m_fluidporo_singlereaction);

    m->add_component(
        parameter<int>("NUMSCAL", {.description = "number of scalars coupled with this problem"}));
    m->add_component(
        parameter<int>("TOTALNUMDOF", {.description = "total number of multiphase-dofs"}));
    m->add_component(parameter<int>("NUMVOLFRAC", {.description = "number of volfracs"}));
    m->add_component(parameter<std::vector<int>>("SCALE",
        {.description = "advanced reaction list", .size = from_parameter<int>("TOTALNUMDOF")}));
    m->add_component(
        parameter<std::string>("COUPLING", {.description = "type of coupling: "
                                                           "scalar_by_function, no_coupling "
                                                           "(default)"}));
    m->add_component(
        parameter<int>("FUNCTID", {.description = "function ID defining the reaction"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one phase for multiphase flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroSinglePhase",
        "one phase for multiphase flow in deformable porous media",
        Core::Materials::m_fluidporo_singlephase);

    m->add_component(parameter<int>("DENSITYLAWID", {.description = "ID of density law"}));
    m->add_component(parameter<double>("DENSITY", {.description = "reference/initial density"}));
    m->add_component(
        parameter<int>("RELPERMEABILITYLAWID", {.description = "ID of relative permeability law"}));
    m->add_component(parameter<int>("VISCOSITYLAWID", {.description = "ID of viscosity law"}));
    m->add_component(parameter<int>("DOFTYPEID", {.description = "ID of dof definition"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one volume fraction for multiphase flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroSingleVolFrac",
        "one phase for multiphase flow in deformable porous media",
        Core::Materials::m_fluidporo_singlevolfrac);

    m->add_component(parameter<double>("DENSITY", {.description = "reference/initial density"}));
    m->add_component(parameter<double>("DIFFUSIVITY", {.description = "diffusivity of phase"}));
    m->add_component(parameter<bool>("AddScalarDependentFlux",
        {.description = "Is there additional scalar dependent flux (yes) or (no)"}));
    m->add_component(
        parameter<int>("NUMSCAL", {.description = "Number of scalars", .default_value = 0}));
    m->add_component(parameter<std::vector<double>>(
        "SCALARDIFFS", {.description = "Diffusivities for additional scalar-dependent flux",
                           .default_value = std::vector<double>{},
                           .size = from_parameter<int>("NUMSCAL")}));
    m->add_component(parameter<std::optional<std::vector<double>>>(
        "OMEGA_HALF", {.description = "Constant for receptor kinetic law",
                          .size = from_parameter<int>("NUMSCAL")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one volume fraction pressure for multiphase flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroVolFracPressure",
        "one volume fraction pressure for multiphase flow in deformable porous media",
        Core::Materials::m_fluidporo_volfracpressure);

    m->add_component(parameter<double>("PERMEABILITY", {.description = "permeability of phase"}));
    m->add_component(parameter<int>("VISCOSITYLAWID", {.description = "ID of viscosity law"}));
    m->add_component(parameter<double>("MIN_VOLFRAC",
        {.description =
                "Minimum volume fraction under which we assume that VolfracPressure is zero",
            .default_value = 1.0e-3}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one degree of freedom for on single phase of a multiphase flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroSinglePhaseDofDiffPressure",
        "one degrree of freedom for multiphase flow in deformable porous media",
        Core::Materials::m_fluidporo_phasedof_diffpressure);

    m->add_component(
        parameter<int>("PHASELAWID", {.description = "ID of pressure-saturation law"}));
    m->add_component(parameter<int>("NUMDOF", {.description = "number of DoFs"}));
    m->add_component(parameter<std::vector<int>>(
        "PRESCOEFF", {.description = "pressure IDs for differential pressure",
                         .default_value = std::vector{0},
                         .size = from_parameter<int>("NUMDOF")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one degree of freedom for on single phase of a multiphase flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroSinglePhaseDofPressure",
        "one degrree of freedom for multiphase flow in deformable porous media",
        Core::Materials::m_fluidporo_phasedof_pressure);

    m->add_component(
        parameter<int>("PHASELAWID", {.description = "ID of pressure-saturation law"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one degree of freedom for on single phase of a multiphase flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroSinglePhaseDofSaturation",
        "one degrree of freedom for multiphase flow in deformable porous media",
        Core::Materials::m_fluidporo_phasedof_saturation);

    m->add_component(
        parameter<int>("PHASELAWID", {.description = "ID of pressure-saturation law"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // saturated law for pressure-saturation law in porous media problems
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PhaseLawLinear",
        "saturated fluid phase of porous medium", Core::Materials::m_fluidporo_phaselaw_linear);

    m->add_component(
        parameter<double>("RELTENSION", {.description = "relative interface tensions"}));
    m->add_component(parameter<double>(
        "SATURATION_0", {.description = "saturation at zero differential pressure"}));
    m->add_component(parameter<int>("NUMDOF", {.description = "number of DoFs"}));
    m->add_component(parameter<std::vector<int>>(
        "PRESCOEFF", {.description = "Coefficients for pressure dependence",
                         .default_value = std::vector{0},
                         .size = from_parameter<int>("NUMDOF")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // tangent law for pressure-saturation law in porous media multiphase problems
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PhaseLawTangent",
        "tangent fluid phase of porous medium", Core::Materials::m_fluidporo_phaselaw_tangent);

    m->add_component(
        parameter<double>("RELTENSION", {.description = "relative interface tensions"}));
    m->add_component(
        parameter<double>("EXP", {.description = "exponent in pressure-saturation law"}));
    m->add_component(parameter<double>(
        "SATURATION_0", {.description = "saturation at zero differential pressure"}));
    m->add_component(parameter<int>("NUMDOF", {.description = "number of DoFs"}));
    m->add_component(parameter<std::vector<int>>(
        "PRESCOEFF", {.description = "Coefficients for pressure dependence",
                         .default_value = std::vector{0},
                         .size = from_parameter<int>("NUMDOF")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // constraint law for pressure-saturation law in porous media multiphase problems
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PhaseLawConstraint",
        "constraint fluid phase of porous medium",
        Core::Materials::m_fluidporo_phaselaw_constraint);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // pressure-saturation law defined by functions in porous media multiphase problems
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PhaseLawByFunction",
        "fluid phase of porous medium defined by functions",
        Core::Materials::m_fluidporo_phaselaw_byfunction);

    m->add_component(
        parameter<int>("FUNCTPRES", {.description = "ID of function for differential pressure"}));
    m->add_component(parameter<int>("FUNCTSAT", {.description = "ID of function for saturation"}));
    m->add_component(parameter<int>("NUMDOF", {.description = "number of DoFs"}));
    m->add_component(parameter<std::vector<int>>(
        "PRESCOEFF", {.description = "Coefficients for pressure dependence",
                         .default_value = std::vector{0},
                         .size = from_parameter<int>("NUMDOF")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // elastic spring
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_Struct_Spring", "elastic spring", Core::Materials::m_spring);

    m->add_component(parameter<double>("STIFFNESS", {.description = "spring constant"}));
    m->add_component(parameter<double>("DENS", {.description = "density"}));

    Mat::append_material_definition(matlist, m);
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
    auto matdef = std::make_shared<Mat::MaterialDefinition>("MAT_BeamReissnerElastHyper",
        "material parameters for a Simo-Reissner type beam element based on "
        "hyperelastic stored energy function",
        Core::Materials::m_beam_reissner_elast_hyper);


    matdef->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));

    /* note: we define both of the two following (redundant) parameters to be optional.
     *       upon initialization of the material, we assure that one of them is
     *       properly defined. */
    matdef->add_component(
        parameter<double>("SHEARMOD", {.description = "shear modulus", .default_value = -1.0}));
    matdef->add_component(parameter<double>(
        "POISSONRATIO", {.description = "Poisson's ratio", .default_value = -1.0}));

    matdef->add_component(parameter<double>("DENS", {.description = "mass density"}));

    matdef->add_component(parameter<double>("CROSSAREA", {.description = "cross-section area"}));
    matdef->add_component(
        parameter<double>("SHEARCORR", {.description = "shear correction factor"}));

    matdef->add_component(
        parameter<double>("MOMINPOL", {.description = "polar/axial area moment of inertia"}));
    matdef->add_component(
        parameter<double>("MOMIN2", {.description = "area moment of inertia w.r.t. first principal "
                                                    "axis of inertia (i.e. second base vector)"}));
    matdef->add_component(parameter<double>(
        "MOMIN3", {.description = "area moment of inertia w.r.t. second principal "
                                  "axis of inertia (i.e. third base vector)"}));
    matdef->add_component(parameter<bool>("FAD",
        {.description = "Does automatic differentiation have to be used", .default_value = false}));


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    matdef->add_component(parameter<double>("INTERACTIONRADIUS",
        {.description =
                "radius of a circular cross-section which "
                "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
            .default_value = -1.0}));

    Mat::append_material_definition(matlist, matdef);
  }
  /*--------------------------------------------------------------------*/
  // material parameter definition for a Simo-Reissner type elasto-plastic beam element
  {
    auto matdef = std::make_shared<Mat::MaterialDefinition>("MAT_BeamReissnerElastPlastic",
        "material parameters for a Simo-Reissner type beam element based on "
        "hyperelastic stored energy function",
        Core::Materials::m_beam_reissner_elast_plastic);


    matdef->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));

    // optional parameters for plasticity
    matdef->add_component(parameter<double>(
        "YIELDN", {.description = "initial yield stress N", .default_value = -1.0}));
    matdef->add_component(parameter<double>(
        "YIELDM", {.description = "initial yield stress M", .default_value = -1.0}));
    matdef->add_component(parameter<double>("ISOHARDN",
        {.description = "isotropic hardening modulus of forces", .default_value = -1.0}));
    matdef->add_component(parameter<double>("ISOHARDM",
        {.description = "isotropic hardening modulus of moments", .default_value = -1.0}));
    matdef->add_component(parameter<bool>("TORSIONPLAST",
        {.description = "defines whether torsional moment contributes to plasticity",
            .default_value = false}));

    /* note: we define both of the two following (redundant) parameters to be optional.
     *       upon initialization of the material, we assure that one of them is
     *       properly defined. */
    matdef->add_component(
        parameter<double>("SHEARMOD", {.description = "shear modulus", .default_value = -1.0}));
    matdef->add_component(parameter<double>(
        "POISSONRATIO", {.description = "Poisson's ratio", .default_value = -1.0}));

    matdef->add_component(parameter<double>("DENS", {.description = "mass density"}));

    matdef->add_component(parameter<double>("CROSSAREA", {.description = "cross-section area"}));
    matdef->add_component(
        parameter<double>("SHEARCORR", {.description = "shear correction factor"}));

    matdef->add_component(
        parameter<double>("MOMINPOL", {.description = "polar/axial area moment of inertia"}));
    matdef->add_component(
        parameter<double>("MOMIN2", {.description = "area moment of inertia w.r.t. first principal "
                                                    "axis of inertia (i.e. second base vector)"}));
    matdef->add_component(parameter<double>(
        "MOMIN3", {.description = "area moment of inertia w.r.t. second principal "
                                  "axis of inertia (i.e. third base vector)"}));
    matdef->add_component(parameter<bool>("FAD",
        {.description = "Does automatic differentiation have to be used", .default_value = false}));


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    matdef->add_component(parameter<double>("INTERACTIONRADIUS",
        {.description =
                "radius of a circular cross-section which "
                "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
            .default_value = -1.0}));

    Mat::append_material_definition(matlist, matdef);
  }
  /*--------------------------------------------------------------------*/
  // material parameter definition for a Simo-Reissner type beam element,
  // specified via 'modal' constitutive parameters (see comment above)
  {
    auto matdef = std::make_shared<Mat::MaterialDefinition>("MAT_BeamReissnerElastHyper_ByModes",
        "material parameters for a Simo-Reissner type beam element based on "
        "hyperelastic stored energy function, specified for individual "
        "deformation modes",
        Core::Materials::m_beam_reissner_elast_hyper_bymodes);


    matdef->add_component(parameter<double>("EA", {.description = "axial rigidity"}));
    matdef->add_component(parameter<double>(
        "GA2", {.description = "shear rigidity w.r.t first principal axis of inertia"}));
    matdef->add_component(parameter<double>(
        "GA3", {.description = "shear rigidity w.r.t second principal axis of inertia"}));

    matdef->add_component(parameter<double>("GI_T", {.description = "torsional rigidity"}));
    matdef->add_component(
        parameter<double>("EI2", {.description = "flexural/bending rigidity w.r.t. first principal "
                                                 "axis of inertia"}));
    matdef->add_component(parameter<double>(
        "EI3", {.description = "flexural/bending rigidity w.r.t. second principal "
                               "axis of inertia"}));

    matdef->add_component(parameter<double>(
        "RhoA", {.description = "translational inertia: mass density * cross-section area"}));

    matdef->add_component(parameter<double>(
        "MASSMOMINPOL", {.description = "polar mass moment of inertia, i.e. w.r.t. "
                                        "rotation around beam axis"}));
    matdef->add_component(parameter<double>(
        "MASSMOMIN2", {.description = "mass moment of inertia w.r.t. first principal "
                                      "axis of inertia"}));
    matdef->add_component(parameter<double>(
        "MASSMOMIN3", {.description = "mass moment of inertia w.r.t. second principal "
                                      "axis of inertia"}));
    matdef->add_component(parameter<bool>("FAD",
        {.description = "Does automatic differentiation have to be used", .default_value = false}));


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    matdef->add_component(parameter<double>("INTERACTIONRADIUS",
        {.description =
                "radius of a circular cross-section which "
                "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
            .default_value = -1.0}));

    Mat::append_material_definition(matlist, matdef);
  }

  /*--------------------------------------------------------------------*/
  // material parameter definition for a Kirchhoff-Love type beam element
  {
    auto matdef = std::make_shared<Mat::MaterialDefinition>("MAT_BeamKirchhoffElastHyper",
        "material parameters for a Kirchhoff-Love type beam element based on "
        "hyperelastic stored energy function",
        Core::Materials::m_beam_kirchhoff_elast_hyper);


    matdef->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));

    /* note: we define both of the two following (redundant) parameters to be optional.
     *       upon initialization of the material, we assure that one of them is
     *       properly defined. */
    matdef->add_component(
        parameter<double>("SHEARMOD", {.description = "shear modulus", .default_value = -1.0}));
    matdef->add_component(parameter<double>(
        "POISSONRATIO", {.description = "Poisson's ratio", .default_value = -1.0}));

    matdef->add_component(parameter<double>("DENS", {.description = "mass density"}));

    matdef->add_component(parameter<double>("CROSSAREA", {.description = "cross-section area"}));

    matdef->add_component(
        parameter<double>("MOMINPOL", {.description = "polar/axial area moment of inertia"}));
    matdef->add_component(
        parameter<double>("MOMIN2", {.description = "area moment of inertia w.r.t. first principal "
                                                    "axis of inertia (i.e. second base vector)"}));
    matdef->add_component(parameter<double>(
        "MOMIN3", {.description = "area moment of inertia w.r.t. second principal "
                                  "axis of inertia (i.e. third base vector)"}));
    matdef->add_component(parameter<bool>("FAD",
        {.description = "Does automatic differentiation have to be used", .default_value = false}));


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    matdef->add_component(parameter<double>("INTERACTIONRADIUS",
        {.description =
                "radius of a circular cross-section which "
                "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
            .default_value = -1.0}));

    Mat::append_material_definition(matlist, matdef);
  }

  /*--------------------------------------------------------------------*/
  // material parameter definition for a Kirchhoff-Love type beam element,
  // specified via 'modal' constitutive parameters (see comment above)
  {
    auto matdef = std::make_shared<Mat::MaterialDefinition>("MAT_BeamKirchhoffElastHyper_ByModes",
        "material parameters for a Kirchhoff-Love type beam element based on "
        "hyperelastic stored energy function, specified for individual "
        "deformation modes",
        Core::Materials::m_beam_kirchhoff_elast_hyper_bymodes);


    matdef->add_component(parameter<double>("EA", {.description = "axial rigidity"}));

    matdef->add_component(parameter<double>("GI_T", {.description = "torsional rigidity"}));
    matdef->add_component(
        parameter<double>("EI2", {.description = "flexural/bending rigidity w.r.t. first principal "
                                                 "axis of inertia"}));
    matdef->add_component(parameter<double>(
        "EI3", {.description = "flexural/bending rigidity w.r.t. second principal "
                               "axis of inertia"}));

    matdef->add_component(parameter<double>(
        "RhoA", {.description = "translational inertia: mass density * cross-section area"}));

    matdef->add_component(parameter<double>(
        "MASSMOMINPOL", {.description = "polar mass moment of inertia, i.e. w.r.t. "
                                        "rotation around beam axis"}));
    matdef->add_component(parameter<double>(
        "MASSMOMIN2", {.description = "mass moment of inertia w.r.t. first principal "
                                      "axis of inertia"}));
    matdef->add_component(parameter<double>(
        "MASSMOMIN3", {.description = "mass moment of inertia w.r.t. second principal "
                                      "axis of inertia"}));
    matdef->add_component(parameter<bool>("FAD",
        {.description = "Does automatic differentiation have to be used", .default_value = false}));


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    matdef->add_component(parameter<double>("INTERACTIONRADIUS",
        {.description =
                "radius of a circular cross-section which "
                "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
            .default_value = -1.0}));

    Mat::append_material_definition(matlist, matdef);
  }

  /*--------------------------------------------------------------------*/
  // material parameter definition for a torsion-free, isotropic
  // Kirchhoff-Love type beam element
  {
    auto matdef =
        std::make_shared<Mat::MaterialDefinition>("MAT_BeamKirchhoffTorsionFreeElastHyper",
            "material parameters for a torsion-free, isotropic Kirchhoff-Love "
            "type beam element based on hyperelastic stored energy function",
            Core::Materials::m_beam_kirchhoff_torsionfree_elast_hyper);


    matdef->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));

    matdef->add_component(parameter<double>("DENS", {.description = "mass density"}));

    matdef->add_component(parameter<double>("CROSSAREA", {.description = "cross-section area"}));

    matdef->add_component(parameter<double>("MOMIN", {.description = "area moment of inertia"}));
    matdef->add_component(parameter<bool>("FAD",
        {.description = "Does automatic differentiation have to be used", .default_value = false}));


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    matdef->add_component(parameter<double>("INTERACTIONRADIUS",
        {.description =
                "radius of a circular cross-section which "
                "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
            .default_value = -1.0}));

    Mat::append_material_definition(matlist, matdef);
  }

  /*--------------------------------------------------------------------*/
  // material parameter definition for a torsion-free, isotropic
  // Kirchhoff-Love type beam element,
  // specified via 'modal' constitutive parameters (see comment above)
  {
    auto matdef =
        std::make_shared<Mat::MaterialDefinition>("MAT_BeamKirchhoffTorsionFreeElastHyper_ByModes",
            "material parameters for a torsion-free, isotropic Kirchhoff-Love "
            "type beam element based on hyperelastic stored energy function, "
            "specified for individual deformation modes",
            Core::Materials::m_beam_kirchhoff_torsionfree_elast_hyper_bymodes);


    matdef->add_component(parameter<double>("EA", {.description = "axial rigidity"}));

    matdef->add_component(parameter<double>("EI", {.description = "flexural/bending rigidity"}));


    matdef->add_component(parameter<double>(
        "RhoA", {.description = "translational inertia: mass density * cross-section area"}));
    matdef->add_component(parameter<bool>("FAD",
        {.description = "Does automatic differentiation have to be used", .default_value = false}));

    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    matdef->add_component(parameter<double>("INTERACTIONRADIUS",
        {.description =
                "radius of a circular cross-section which "
                "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
            .default_value = -1.0}));

    Mat::append_material_definition(matlist, matdef);
  }

  /*----------------------------------------------------------------------*/
  // material for an elastic Kirchhoff-Love shell
  {
    auto matdef = std::make_shared<Mat::MaterialDefinition>("MAT_Kirchhoff_Love_shell",
        "Material for an elastic Kichhhoff-Love shell ", Core::Materials::m_shell_kirchhoff_love);

    matdef->add_component(parameter<double>("YOUNG_MODULUS", {.description = "Young's modulus"}));
    matdef->add_component(parameter<double>("POISSON_RATIO", {.description = "Poisson's ratio"}));
    matdef->add_component(
        parameter<double>("THICKNESS", {.description = "Thickness of the shell"}));

    append_material_definition(matlist, matdef);
  }

  /*--------------------------------------------------------------------*/
  // material for a crosslinker in a biopolymer simulation
  {
    auto matdef = std::make_shared<Mat::MaterialDefinition>("MAT_Crosslinker",
        "material for a linkage between beams", Core::Materials::m_crosslinkermat);

    matdef->add_component(
        parameter<double>("MATNUM", {.description = "number of beam elasthyper material"}));
    matdef->add_component(parameter<std::string>(
        "JOINTTYPE", {.description = "type of joint: "
                                     "beam3rline2rigid (default), beam3rline2pin or truss"}));
    matdef->add_component(parameter<double>(
        "LINKINGLENGTH", {.description = "distance between the two binding domains of a linker"}));
    matdef->add_component(parameter<double>("LINKINGLENGTHTOL",
        {.description = "tolerance for linker length in the sense: length +- tolerance"}));
    matdef->add_component(parameter<double>("LINKINGANGLE",
        {.description = "preferred binding angle enclosed by two filaments' axes in radians"}));
    matdef->add_component(parameter<double>(
        "LINKINGANGLETOL", {.description = "tolerance for preferred binding angle in radians in "
                                           "the sense of: angle +- tolerance"}));
    matdef->add_component(parameter<double>("K_ON", {.description = "chemical association-rate"}));
    matdef->add_component(
        parameter<double>("K_OFF", {.description = "chemical dissociation-rate"}));

    // optional parameter
    matdef->add_component(parameter<double>(
        "DELTABELLEQ", {.description = "deltaD in Bell's equation for force dependent off rate",
                           .default_value = 0.0}));
    matdef->add_component(parameter<double>("NOBONDDISTSPHERE",
        {.description = "distance to sphere elements in which no double bonded linker is allowed",
            .default_value = 0.0}));
    matdef->add_component(parameter<std::string>(
        "TYPE", {.description = "type of crosslinker: "
                                "arbitrary (default), actin, collagen, integrin",
                    .default_value = "arbitrary"}));

    Mat::append_material_definition(matlist, matdef);
  }

  /*--------------------------------------------------------------------*/
  // 0D Acinar material base
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_0D_MAXWELL_ACINUS", "0D acinar material", Core::Materials::m_0d_maxwell_acinus);

    m->add_component(parameter<double>("Stiffness1", {.description = "first stiffness"}));
    m->add_component(parameter<double>("Stiffness2", {.description = "second stiffness"}));
    m->add_component(parameter<double>("Viscosity1", {.description = "first viscosity"}));
    m->add_component(parameter<double>("Viscosity2", {.description = "second viscosity"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // 0D NeoHookean Acinar material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_0D_MAXWELL_ACINUS_NEOHOOKEAN",
        "0D acinar material neohookean", Core::Materials::m_0d_maxwell_acinus_neohookean);

    m->add_component(parameter<double>("Stiffness1", {.description = "first stiffness"}));
    m->add_component(parameter<double>("Stiffness2", {.description = "second stiffness"}));
    m->add_component(parameter<double>("Viscosity1", {.description = "first viscosity"}));
    m->add_component(parameter<double>("Viscosity2", {.description = "second viscosity"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // 0D Exponential Acinar material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_0D_MAXWELL_ACINUS_EXPONENTIAL",
        "0D acinar material exponential", Core::Materials::m_0d_maxwell_acinus_exponential);

    m->add_component(parameter<double>("Stiffness1", {.description = "first stiffness"}));
    m->add_component(parameter<double>("Stiffness2", {.description = "second stiffness"}));
    m->add_component(parameter<double>("Viscosity1", {.description = "first viscosity"}));
    m->add_component(parameter<double>("Viscosity2", {.description = "second viscosity"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // 0D Exponential Acinar material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_0D_MAXWELL_ACINUS_DOUBLEEXPONENTIAL",
        "0D acinar material doubleexponential",
        Core::Materials::m_0d_maxwell_acinus_doubleexponential);

    m->add_component(parameter<double>("Stiffness1", {.description = "first stiffness"}));
    m->add_component(parameter<double>("Stiffness2", {.description = "second stiffness"}));
    m->add_component(parameter<double>("Viscosity1", {.description = "first viscosity"}));
    m->add_component(parameter<double>("Viscosity2", {.description = "second viscosity"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // 0D Ogden Acinar material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_0D_MAXWELL_ACINUS_OGDEN",
        "0D acinar material ogden", Core::Materials::m_0d_maxwell_acinus_ogden);

    m->add_component(parameter<double>("Stiffness1", {.description = "first stiffness"}));
    m->add_component(parameter<double>("Stiffness2", {.description = "second stiffness"}));
    m->add_component(parameter<double>("Viscosity1", {.description = "first viscosity"}));
    m->add_component(parameter<double>("Viscosity2", {.description = "second viscosity"}));

    Mat::append_material_definition(matlist, m);
  }


  /*----------------------------------------------------------------------*/
  // particle material sph fluid
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_ParticleSPHFluid",
        "particle material for SPH fluid", Core::Materials::m_particle_sph_fluid);

    m->add_component(parameter<double>("INITRADIUS", {.description = "initial radius"}));
    m->add_component(parameter<double>("INITDENSITY", {.description = "initial density"}));
    m->add_component(parameter<double>(
        "REFDENSFAC", {.description = "reference density factor in equation of state"}));
    m->add_component(
        parameter<double>("EXPONENT", {.description = "exponent in equation of state"}));
    m->add_component(parameter<double>("BACKGROUNDPRESSURE",
        {.description = "background pressure for transport velocity formulation"}));
    m->add_component(parameter<double>("BULK_MODULUS", {.description = "bulk modulus"}));
    m->add_component(
        parameter<double>("DYNAMIC_VISCOSITY", {.description = "dynamic shear viscosity"}));
    m->add_component(parameter<double>("BULK_VISCOSITY", {.description = "bulk viscosity"}));
    m->add_component(
        parameter<double>("ARTIFICIAL_VISCOSITY", {.description = "artificial viscosity"}));
    m->add_component(parameter<double>(
        "INITTEMPERATURE", {.description = "initial temperature", .default_value = 0.0}));
    m->add_component(parameter<double>(
        "THERMALCAPACITY", {.description = "thermal capacity", .default_value = 0.0}));
    m->add_component(parameter<double>(
        "THERMALCONDUCTIVITY", {.description = "thermal conductivity", .default_value = 0.0}));
    m->add_component(parameter<double>(
        "THERMALABSORPTIVITY", {.description = "thermal absorptivity", .default_value = 0.0}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // particle material sph boundary
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_ParticleSPHBoundary",
        "particle material for SPH boundary", Core::Materials::m_particle_sph_boundary);

    m->add_component(parameter<double>("INITRADIUS", {.description = "initial radius"}));
    m->add_component(parameter<double>("INITDENSITY", {.description = "initial density"}));
    m->add_component(parameter<double>(
        "INITTEMPERATURE", {.description = "initial temperature", .default_value = 0.0}));
    m->add_component(parameter<double>(
        "THERMALCAPACITY", {.description = "thermal capacity", .default_value = 0.0}));
    m->add_component(parameter<double>(
        "THERMALCONDUCTIVITY", {.description = "thermal conductivity", .default_value = 0.0}));
    m->add_component(parameter<double>(
        "THERMALABSORPTIVITY", {.description = "thermal absorptivity", .default_value = 0.0}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // particle material dem
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_ParticleDEM", "particle material for DEM", Core::Materials::m_particle_dem);

    m->add_component(
        parameter<double>("INITRADIUS", {.description = "initial radius of particle"}));
    m->add_component(
        parameter<double>("INITDENSITY", {.description = "initial density of particle"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // particle wall material dem
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_ParticleWallDEM",
        "particle wall material for DEM", Core::Materials::m_particle_wall_dem);

    m->add_component(parameter<double>("FRICT_COEFF_TANG",
        {.description = "friction coefficient for tangential contact", .default_value = -1.0}));
    m->add_component(parameter<double>("FRICT_COEFF_ROLL",
        {.description = "friction coefficient for rolling contact", .default_value = -1.0}));
    m->add_component(parameter<double>("ADHESION_SURFACE_ENERGY",
        {.description = "adhesion surface energy", .default_value = -1.0}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // electromagnetic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_Electromagnetic", "Electromagnetic material", Core::Materials::m_electromagneticmat);

    m->add_component(parameter<double>("CONDUCTIVITY", {.description = "electrical conductivity"}));
    m->add_component(parameter<double>("PERMITTIVITY", {.description = "Permittivity"}));
    m->add_component(parameter<double>("PERMEABILITY", {.description = "Permeability"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // General mixture models (used for prestretching and for homogenized constrained mixture models)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_Mixture", "General mixture model", Core::Materials::m_mixture);

    m->add_component(parameter<int>("NUMCONST", {.description = "number of mixture constituents"}));
    m->add_component(
        parameter<int>("MATIDMIXTURERULE", {.description = "material id of the mixturerule"}));
    m->add_component(parameter<std::vector<int>>(
        "MATIDSCONST", {.description = "list material IDs of the mixture constituents",
                           .size = from_parameter<int>("NUMCONST")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for ElastHyper toolbox
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MIX_Constituent_ElastHyper", "ElastHyper toolbox", Core::Materials::mix_elasthyper);

    m->add_component(parameter<int>("NUMMAT", {.description = "number of summands"}));
    m->add_component(
        parameter<std::vector<int>>("MATIDS", {.description = "list material IDs of the summands",
                                                  .size = from_parameter<int>("NUMMAT")}));
    m->add_component(parameter<int>("PRESTRESS_STRATEGY",
        {.description =
                "Material id of the prestress strategy (optional, by default no prestretch)",
            .default_value = 0}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for ElastHyper toolbox with a damage process
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Constituent_ElastHyper_Damage",
        "ElastHyper toolbox with damage", Core::Materials::mix_elasthyper_damage);

    m->add_component(parameter<int>("NUMMAT", {.description = "number of summands"}));
    m->add_component(parameter<std::vector<int>>(
        "MATIDS", {.description = "list material IDs of the membrane summands",
                      .size = from_parameter<int>("NUMMAT")}));
    m->add_component(parameter<int>("PRESTRESS_STRATEGY",
        {.description =
                "Material id of the prestress strategy (optional, by default no prestretch)",
            .default_value = 0}));
    m->add_component(parameter<int>(
        "DAMAGE_FUNCT", {.description = "Reference to the function that is a gain for "
                                        "the increase/decrease of the reference "
                                        "mass density."}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for ElastHyper toolbox with a damage process and a membrane constituent
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Constituent_ElastHyper_ElastinMembrane",
        "ElastHyper toolbox with damage and 2D membrane material",
        Core::Materials::mix_elasthyper_elastin_membrane);

    m->add_component(parameter<int>("NUMMAT", {.description = "number of summands"}));
    m->add_component(parameter<std::vector<int>>(
        "MATIDS", {.description = "list material IDs of the membrane summands",
                      .size = from_parameter<int>("NUMMAT")}));
    m->add_component(parameter<int>("MEMBRANENUMMAT", {.description = "number of summands"}));
    m->add_component(parameter<std::vector<int>>(
        "MEMBRANEMATIDS", {.description = "list material IDs of the membrane summands",
                              .size = from_parameter<int>("MEMBRANENUMMAT")}));
    m->add_component(parameter<int>("PRESTRESS_STRATEGY",
        {.description =
                "Material id of the prestress strategy (optional, by default no prestretch)",
            .default_value = 0}));
    m->add_component(parameter<int>(
        "DAMAGE_FUNCT", {.description = "Reference to the function that is a gain for "
                                        "the increase/decrease of the reference "
                                        "mass density."}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for solid material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MIX_Constituent_SolidMaterial", "Solid material", Core::Materials::mix_solid_material);

    m->add_component(parameter<int>("MATID", {.description = "ID of the solid material"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Isotropic growth
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_GrowthStrategy_Isotropic",
        "isotropic growth", Core::Materials::mix_growth_strategy_isotropic);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Anisotropic growth
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_GrowthStrategy_Anisotropic",
        "anisotropic growth", Core::Materials::mix_growth_strategy_anisotropic);


    m->add_component(parameter<int>(
        "INIT", {.description = "initialization modus for growth direction alignment",
                    .default_value = 1}));
    m->add_component(parameter<int>("FIBER_ID",
        {.description =
                "Id of the fiber to point the growth direction (1 for first fiber, default)",
            .default_value = 1}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Extension of all constituents simultaneously -> Growth happens mainly in the direction with the
  // smallest stiffness
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_GrowthStrategy_Stiffness",
        "Extension of all constituents simultaneously",
        Core::Materials::mix_growth_strategy_stiffness);

    m->add_component(parameter<double>("KAPPA",
        {.description = "Penalty parameter for the modified penalty term for incompressibility"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // General material wrapper enabling iterative prestressing
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_IterativePrestress",
        "General material wrapper enabling iterative pretressing for any material",
        Core::Materials::m_iterative_prestress);

    m->add_component(parameter<int>("MATID", {.description = "Id of the material"}));
    m->add_component(parameter<bool>("ACTIVE",
        {.description =
                "Set to True during prestressing and to false afterwards using a restart of the "
                "simulation."}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Constant predefined prestretch
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Prestress_Strategy_Constant",
        "Simple predefined prestress", Core::Materials::mix_prestress_strategy_constant);

    m->add_component(parameter<std::vector<double>>(
        "PRESTRETCH", {.description = "Definition of the prestretch as a "
                                      "9x1 vector",
                          .size = 9}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Prestress strategy for a cylinder
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Prestress_Strategy_Cylinder",
        "Simple prestress strategy for a cylinder",
        Core::Materials::mix_prestress_strategy_cylinder);

    m->add_component(
        parameter<double>("INNER_RADIUS", {.description = "Inner radius of the cylinder"}));
    m->add_component(
        parameter<double>("WALL_THICKNESS", {.description = "Wall thickness of the cylinder"}));
    m->add_component(
        parameter<double>("AXIAL_PRESTRETCH", {.description = "Prestretch in axial direction"}));
    m->add_component(parameter<double>(
        "CIRCUMFERENTIAL_PRESTRETCH", {.description = "Prestretch in circumferential direction"}));
    m->add_component(
        parameter<double>("PRESSURE", {.description = "Pressure in the inner of the cylinder"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Iterative prestress strategy for any geometry
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Prestress_Strategy_Iterative",
        "Simple iterative prestress strategy for any geometry. Needed to be used within the "
        "mixture framework.",
        Core::Materials::mix_prestress_strategy_iterative);
    m->add_component(parameter<bool>(
        "ACTIVE", {.description = "Flag whether prestretch tensor should be updated"}));
    m->add_component(parameter<bool>("ISOCHORIC",
        {.description = "Flag whether prestretch tensor is isochoric", .default_value = false}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for a full constrained mixture fiber
  {
    auto m =
        std::make_shared<Mat::MaterialDefinition>("MIX_Constituent_FullConstrainedMixtureFiber",
            "A 1D constituent that grows with the full constrained mixture fiber theory",
            Core::Materials::mix_full_constrained_mixture_fiber);

    m->add_component(parameter<int>("FIBER_ID", {.description = "Id of the fiber"}));
    m->add_component(parameter<int>("FIBER_MATERIAL_ID", {.description = "Id of fiber material"}));
    m->add_component(parameter<bool>("ENABLE_GROWTH",
        {.description = "Switch for the growth (default true)", .default_value = true}));
    m->add_component(parameter<bool>("ENABLE_BASAL_MASS_PRODUCTION",
        {.description = "Switch to enable the basal mass production rate (default true)",
            .default_value = true}));
    m->add_component(
        parameter<double>("DECAY_TIME", {.description = "Decay time of deposited tissue"}));
    m->add_component(
        parameter<double>("GROWTH_CONSTANT", {.description = "Growth constant of the tissue"}));
    m->add_component(parameter<double>(
        "DEPOSITION_STRETCH", {.description = "Stretch at which the fiber is deposited"}));
    m->add_component(parameter<int>("INITIAL_DEPOSITION_STRETCH_TIMEFUNCT",
        {.description = "Id of the time function to scale the deposition stretch (Default: 0=None)",
            .default_value = 0}));
    m->add_component(parameter<int>("INIT",
        {.description = "Initialization mode for fibers (1=element fibers, 3=nodal fibers)"}));
    m->add_component(parameter<std::string>("ADAPTIVE_HISTORY_STRATEGY",
        {.description = "Strategy for adaptive history integration (none, model_equation, "
                        "higher_order)",
            .default_value = "none"}));
    m->add_component(parameter<double>("ADAPTIVE_HISTORY_TOLERANCE",
        {.description = "Tolerance of the adaptive history", .default_value = 1e-6}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for a remodel fiber
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Constituent_ExplicitRemodelFiber",
        "A 1D constituent that remodels", Core::Materials::mix_remodelfiber_expl);

    m->add_component(
        parameter<int>("FIBER_ID", {.description = "Id of the fiber", .default_value = 1}));
    m->add_component(parameter<int>("FIBER_MATERIAL_ID", {.description = "Id of fiber material"}));

    m->add_component(parameter<bool>("ENABLE_GROWTH",
        {.description = "Switch for the growth (default true)", .default_value = true}));
    m->add_component(parameter<bool>("ENABLE_BASAL_MASS_PRODUCTION",
        {.description = "Switch to enable the basal mass production rate (default true)",
            .default_value = true}));
    m->add_component(
        parameter<double>("DECAY_TIME", {.description = "Decay time of deposited tissue"}));
    m->add_component(
        parameter<double>("GROWTH_CONSTANT", {.description = "Growth constant of the tissue"}));
    m->add_component(parameter<double>(
        "DEPOSITION_STRETCH", {.description = "Stretch at with the fiber is deposited"}));
    m->add_component(parameter<int>("DEPOSITION_STRETCH_TIMEFUNCT",
        {.description = "Id of the time function to scale the deposition stretch (Default: 0=None)",
            .default_value = 0}));
    m->add_component(parameter<bool>(
        "INELASTIC_GROWTH", {.description = "Mixture rule has inelastic growth (default false)",
                                .default_value = false}));
    m->add_component(parameter<int>("INIT",
        {.description = "Initialization mode for fibers (1=element fibers, 2=nodal fibers)"}));
    m->add_component(parameter<double>(
        "GAMMA", {.description = "Angle of fiber alignment in degree (default = 0.0 degrees)",
                     .default_value = 0.0}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for a remodel fiber
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Constituent_ImplicitRemodelFiber",
        "A 1D constituent that remodels", Core::Materials::mix_remodelfiber_impl);

    m->add_component(parameter<int>("FIBER_ID", {.description = "Id of the fiber"}));
    m->add_component(parameter<int>("FIBER_MATERIAL_ID", {.description = "Id of fiber material"}));

    m->add_component(parameter<bool>("ENABLE_GROWTH",
        {.description = "Switch for the growth (default true)", .default_value = true}));
    m->add_component(parameter<bool>("ENABLE_BASAL_MASS_PRODUCTION",
        {.description = "Switch to enable the basal mass production rate (default true)",
            .default_value = true}));
    m->add_component(
        parameter<double>("DECAY_TIME", {.description = "Decay time of deposited tissue"}));
    m->add_component(
        parameter<double>("GROWTH_CONSTANT", {.description = "Growth constant of the tissue"}));
    m->add_component(parameter<double>(
        "DEPOSITION_STRETCH", {.description = "Stretch at with the fiber is deposited"}));
    m->add_component(parameter<int>("DEPOSITION_STRETCH_TIMEFUNCT",
        {.description = "Id of the time function to scale the deposition stretch (Default: 0=None)",
            .default_value = 0}));
    m->add_component(parameter<int>("INIT",
        {.description = "Initialization mode for fibers (1=element fibers, 2=nodal fibers)"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent material for a remodel fiber with exponential strain energy function
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MIX_Constituent_RemodelFiber_Material_Exponential",
        "An exponential strain energy function for the remodel fiber",
        Core::Materials::mix_remodelfiber_material_exponential);


    m->add_component(parameter<double>(
        "K1", {.description = "First parameter of exponential strain energy function"}));
    m->add_component(parameter<double>(
        "K2", {.description = "Second parameter of exponential strain energy function"}));
    m->add_component(parameter<bool>("COMPRESSION",
        {.description = "Bool, whether the fiber material also supports compressive forces."}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent material for a remodel fiber with exponential strain energy function and an
  // active contribution
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MIX_Constituent_RemodelFiber_Material_Exponential_Active",
        "An exponential strain energy function for the remodel fiber with an active contribution",
        Core::Materials::mix_remodelfiber_material_exponential_active);


    m->add_component(parameter<double>(
        "K1", {.description = "First parameter of exponential strain energy function"}));
    m->add_component(parameter<double>(
        "K2", {.description = "Second parameter of exponential strain energy function"}));
    m->add_component(parameter<bool>("COMPRESSION",
        {.description = "Bool, whether the fiber material also supports compressive forces."}));
    m->add_component(
        parameter<double>("SIGMA_MAX", {.description = "Maximum active Cauchy-stress"}));
    m->add_component(
        parameter<double>("LAMBDAMAX", {.description = "Stretch at maximum active Cauchy-stress"}));
    m->add_component(
        parameter<double>("LAMBDA0", {.description = "Stretch at zero active Cauchy-stress"}));
    m->add_component(
        parameter<double>("LAMBDAACT", {.description = "Current stretch", .default_value = 1.0}));
    m->add_component(parameter<double>("DENS", {.description = "Density of the whole mixture"}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Function mixture rule for solid mixtures
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Rule_Function",
        "A mixture rule where the mass fractions are scaled by functions of space and time",
        Core::Materials::mix_rule_function);

    m->add_component(parameter<double>("DENS", {.description = ""}));
    m->add_component(parameter<int>("NUMCONST", {.description = "number of mixture constituents"}));
    m->add_component(parameter<std::vector<int>>("MASSFRACFUNCT",
        {.description = "list of functions (their ids) defining the mass fractions of the mixture "
                        "constituents",
            .size = from_parameter<int>("NUMCONST")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Map mixture rule for solid mixtures
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Rule_Map",
        "A mixture rule where the mass fractions are defined elementwise as discrete values",
        Core::Materials::mix_rule_map);

    m->add_component(parameter<double>("DENS", {.description = ""}));
    m->add_component(parameter<int>("NUMCONST", {.description = "number of mixture constituents"}));

    // definition of operation and print string for post processed component "MASSFRACMAPFILE"
    using mapType = std::unordered_map<int, std::vector<double>>;

    auto on_parse = [](Core::IO::InputParameterContainer& container)
    {
      const auto& map_file = container.get<std::filesystem::path>("MASSFRACMAPFILE");
      std::ifstream file_stream(map_file);

      if (file_stream.fail()) FOUR_C_THROW("Invalid file %s!", map_file.string().c_str());

      auto map_reduction_operation = [](mapType acc, const mapType& next)
      {
        for (const auto& [key, value] : next)
        {
          acc[key] = value;
        }
        return acc;
      };

      container.add("MASSFRACMAPFILE_CONTENT",
          Core::IO::convert_lines<mapType, mapType>(file_stream, map_reduction_operation));
    };

    m->add_component(parameter<std::filesystem::path>("MASSFRACMAPFILE",
        {
            .description =
                "file path of pattern file defining the massfractions as discrete values",
            .on_parse_callback = on_parse,
        }));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Base mixture rule for solid mixtures
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MIX_Rule_Simple", "Simple mixture rule", Core::Materials::mix_rule_simple);

    m->add_component(parameter<double>("DENS", {.description = ""}));
    m->add_component(parameter<int>("NUMCONST", {.description = "number of mixture constituents"}));
    m->add_component(parameter<std::vector<double>>(
        "MASSFRAC", {.description = "list mass fractions of the mixture constituents",
                        .size = from_parameter<int>("NUMCONST")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Base mixture rule for solid mixtures
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_GrowthRemodelMixtureRule",
        "Mixture rule for growth/remodel homogenized constrained mixture models",
        Core::Materials::mix_rule_growthremodel);

    m->add_component(
        parameter<int>("GROWTH_STRATEGY", {.description = "Material id of the growth strategy"}));
    m->add_component(parameter<double>("DENS", {.description = ""}));
    m->add_component(parameter<int>("NUMCONST", {.description = "number of mixture constituents"}));
    m->add_component(parameter<std::vector<double>>(
        "MASSFRAC", {.description = "list mass fractions of the mixture constituents",
                        .size = from_parameter<int>("NUMCONST")}));

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // crystal plasticity
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_crystal_plasticity", " Crystal plasticity ", Core::Materials::m_crystplast);
    m->add_component(
        parameter<double>("TOL", {.description = "tolerance for internal Newton iteration"}));
    m->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));
    m->add_component(parameter<double>("NUE", {.description = "Poisson's ratio"}));
    m->add_component(parameter<double>("DENS", {.description = "Mass density"}));
    m->add_component(parameter<std::string>("LAT",
        {.description = "lattice type: FCC, BCC, HCP, D019 or L10", .default_value = "FCC"}));
    m->add_component(
        parameter<double>("CTOA", {.description = "c to a ratio of crystal unit cell"}));
    m->add_component(
        parameter<double>("ABASE", {.description = "base length a of the crystal unit cell"}));
    m->add_component(parameter<int>("NUMSLIPSYS", {.description = "number of slip systems"}));
    m->add_component(parameter<int>("NUMSLIPSETS", {.description = "number of slip system sets"}));
    m->add_component(parameter<std::vector<int>>("SLIPSETMEMBERS",
        {.description = "vector of NUMSLIPSYS indices ranging from 1 to NUMSLIPSETS that indicate "
                        "to which set each slip system belongs",
            .size = from_parameter<int>("NUMSLIPSYS")}));
    m->add_component(parameter<std::vector<int>>("SLIPRATEEXP",
        {.description = "vector containing NUMSLIPSETS entries for the rate sensitivity exponent",
            .size = from_parameter<int>("NUMSLIPSETS")}));
    m->add_component(parameter<std::vector<double>>("GAMMADOTSLIPREF",
        {.description = "vector containing NUMSLIPSETS entries for the reference slip shear rate",
            .size = from_parameter<int>("NUMSLIPSETS")}));
    m->add_component(parameter<std::vector<double>>("DISDENSINIT",
        {.description = "vector containing NUMSLIPSETS entries for the initial dislocation density",
            .size = from_parameter<int>("NUMSLIPSETS")}));
    m->add_component(parameter<std::vector<double>>("DISGENCOEFF",
        {.description =
                "vector containing NUMSLIPSETS entries for the dislocation generation coefficients",
            .size = from_parameter<int>("NUMSLIPSETS")}));
    m->add_component(parameter<std::vector<double>>(
        "DISDYNRECCOEFF", {.description = "vector containing NUMSLIPSETS entries for "
                                          "the coefficients for dynamic dislocation "
                                          "removal",
                              .size = from_parameter<int>("NUMSLIPSETS")}));
    m->add_component(parameter<std::vector<double>>(
        "TAUY0", {.description = "vector containing NUMSLIPSETS entries for the "
                                 "lattice resistance to slip, e.g. the "
                                 "Peierls barrier",
                     .size = from_parameter<int>("NUMSLIPSETS")}));
    m->add_component(parameter<std::vector<double>>(
        "MFPSLIP", {.description = "vector containing NUMSLIPSETS microstructural "
                                   "parameters that are relevant for Hall-Petch "
                                   "strengthening, e.g., grain size",
                       .size = from_parameter<int>("NUMSLIPSETS")}));
    m->add_component(parameter<std::vector<double>>(
        "SLIPHPCOEFF", {.description = "vector containing NUMSLIPSETS entries for the Hall-Petch "
                                       "coefficients corresponding to "
                                       "the microstructural parameters given in MFPSLIP",
                           .size = from_parameter<int>("NUMSLIPSETS")}));
    m->add_component(parameter<std::vector<double>>(
        "SLIPBYTWIN", {.description = "(optional) vector containing NUMSLIPSETS entries for the "
                                      "work hardening coefficients by "
                                      "twinning on non-coplanar systems",
                          .default_value = std::vector{0.},
                          .size = from_parameter<int>("NUMSLIPSETS")}));
    m->add_component(parameter<int>("NUMTWINSYS",
        {.description = "(optional) number of twinning systems", .default_value = 0}));
    m->add_component(parameter<int>("NUMTWINSETS",
        {.description = "(optional) number of sets of twinning systems", .default_value = 0}));
    m->add_component(parameter<std::vector<int>>("TWINSETMEMBERS",
        {.description = "(optional) vector of NUMTWINSYS indices ranging from 1 to NUMTWINSETS "
                        "that indicate to which set each slip system belongs",
            .default_value = std::vector{0},
            .size = from_parameter<int>("NUMTWINSYS")}));
    m->add_component(parameter<std::vector<int>>("TWINRATEEXP",
        {.description = "(optional) vector containing NUMTWINSETS entries for the rate sensitivity "
                        "exponent",
            .default_value = std::vector{0},
            .size = from_parameter<int>("NUMTWINSETS")}));
    m->add_component(parameter<std::vector<double>>(
        "GAMMADOTTWINREF", {.description = "(optional) vector containing NUMTWINSETS entries for "
                                           "the reference slip shear rate",
                               .default_value = std::vector{0.},
                               .size = from_parameter<int>("NUMTWINSETS")}));
    m->add_component(parameter<std::vector<double>>(
        "TAUT0", {.description = "(optional) vector containing NUMTWINSETS entries for the lattice "
                                 "resistance to twinning, "
                                 "e.g. the Peierls barrier",
                     .default_value = std::vector{0.},
                     .size = from_parameter<int>("NUMTWINSETS")}));
    m->add_component(parameter<std::vector<double>>(
        "MFPTWIN", {.description = "(optional) vector containing NUMTWINSETS microstructural "
                                   "parameters that are relevant for "
                                   "Hall-Petch strengthening of twins, e.g., grain size",
                       .default_value = std::vector{0.},
                       .size = from_parameter<int>("NUMTWINSETS")}));
    m->add_component(parameter<std::vector<double>>("TWINHPCOEFF",
        {.description =
                "(optional) vector containing NUMTWINSETS entries for the Hall-Petch coefficients "
                "corresponding to the microstructural parameters given in MFPTWIN",
            .default_value = std::vector{0.},
            .size = from_parameter<int>("NUMTWINSETS")}));
    m->add_component(parameter<std::vector<double>>(
        "TWINBYSLIP", {.description = "(optional) vector containing NUMTWINSETS entries for the "
                                      "work hardening coefficients by "
                                      "slip",
                          .default_value = std::vector{0.},
                          .size = from_parameter<int>("NUMTWINSETS")}));
    m->add_component(parameter<std::vector<double>>(
        "TWINBYTWIN", {.description = "(optional) vector containing NUMTWINSETS entries for the "
                                      "work hardening coefficients by "
                                      "twins on non-coplanar systems",
                          .default_value = std::vector{0.},
                          .size = from_parameter<int>("NUMTWINSETS")}));
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // linear elastic material in one direction
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_LinElast1D",
        "linear elastic material in one direction", Core::Materials::m_linelast1D);

    m->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));
    m->add_component(parameter<double>("DENS", {.description = "mass density"}));

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // linear elastic material with growth in one direction
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_LinElast1DGrowth",
        "linear elastic material with growth in one direction",
        Core::Materials::m_linelast1D_growth);

    m->add_component(parameter<double>("YOUNG", {.description = "Young's modulus"}));
    m->add_component(parameter<double>("DENS", {.description = "mass density"}));
    m->add_component(parameter<double>("C0", {.description = "reference concentration"}));
    m->add_component(parameter<bool>(
        "AOS_PROP_GROWTH", {.description = "growth proportional to amount of substance (AOS) if "
                                           "true or proportional to concentration "
                                           "if false"}));
    m->add_component(
        parameter<int>("POLY_PARA_NUM", {.description = "number of polynomial coefficients"}));
    m->add_component(parameter<std::vector<double>>(
        "POLY_PARAMS", {.description = "coefficients of polynomial",
                           .size = from_parameter<int>("POLY_PARA_NUM")}));

    Mat::append_material_definition(matlist, m);
  }

  // deliver
  return vm;
}

FOUR_C_NAMESPACE_CLOSE
