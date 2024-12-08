// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_validmaterials.hpp"

#include "4C_global_data.hpp"
#include "4C_inpar_material.hpp"
#include "4C_io_control.hpp"
#include "4C_io_file_reader.hpp"
#include "4C_io_linecomponent.hpp"
#include "4C_mat_materialdefinition.hpp"

#include <filesystem>
#include <string>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Input::print_empty_material_definitions(
    std::ostream& stream, std::vector<std::shared_ptr<Mat::MaterialDefinition>>& matlist)
{
  const std::string sectionname = "MATERIALS";
  const unsigned l = sectionname.length();
  stream << "--" << std::string(std::max<int>(65 - l, 0), '-');
  stream << sectionname << '\n';

  for (auto& i : matlist)
  {
    i->print(stream, nullptr);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void print_material_dat_header()
{
  std::shared_ptr<std::vector<std::shared_ptr<Mat::MaterialDefinition>>> matlist =
      Input::valid_materials();
  Input::print_empty_material_definitions(std::cout, *matlist);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<std::vector<std::shared_ptr<Mat::MaterialDefinition>>> Input::valid_materials()
{
  using Teuchos::tuple;

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

    add_named_real(m, "DYNVISCOSITY", "dynamic viscosity");
    add_named_real(m, "DENSITY", "spatial mass density");
    add_named_real(m, "GAMMA", "surface tension coefficient", 0.0, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Weakly compressible fluid according to Murnaghan-Tait
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_fluid_murnaghantait",
        "Weakly compressible fluid according to Murnaghan-Tait",
        Core::Materials::m_fluid_murnaghantait);

    add_named_real(m, "DYNVISCOSITY", "dynamic viscosity");
    add_named_real(m, "REFDENSITY", "reference spatial mass density");
    add_named_real(m, "REFPRESSURE", "reference pressure");
    add_named_real(m, "REFBULKMODULUS", "reference bulk modulus");
    add_named_real(m, "MATPARAMETER", "material parameter according to Murnaghan-Tait");
    add_named_real(m, "GAMMA", "surface tension coefficient", 0.0, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Linear law (pressure-dependent) for the density and the viscosity
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_fluid_linear_density_viscosity",
        "Linear law (pressure-dependent) for the density and the viscosity",
        Core::Materials::m_fluid_linear_density_viscosity);

    add_named_real(m, "REFDENSITY", "reference density");
    add_named_real(m, "REFVISCOSITY", "reference viscosity");
    add_named_real(m, "REFPRESSURE", "reference pressure");
    add_named_real(m, "COEFFDENSITY", "density-pressure coefficient");
    add_named_real(m, "COEFFVISCOSITY", "viscosity-pressure coefficient");
    add_named_real(m, "GAMMA", "surface tension coefficient", 0.0, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Weakly compressible fluid
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_fluid_weakly_compressible",
        "Weakly compressible fluid", Core::Materials::m_fluid_weakly_compressible);

    add_named_real(m, "VISCOSITY", "viscosity");
    add_named_real(m, "REFDENSITY", "reference density");
    add_named_real(m, "REFPRESSURE", "reference pressure");
    add_named_real(m, "COMPRCOEFF", "compressibility coefficient");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // fluid with non-linear viscosity according to Carreau-Yasuda
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_carreauyasuda",
        "fluid with non-linear viscosity according to Carreau-Yasuda",
        Core::Materials::m_carreauyasuda);

    add_named_real(m, "NU_0", "zero-shear viscosity");
    add_named_real(m, "NU_INF", "infinite-shear viscosity");
    add_named_real(m, "LAMBDA", "characteristic time");
    add_named_real(m, "APARAM", "constant parameter");
    add_named_real(m, "BPARAM", "constant parameter");
    add_named_real(m, "DENSITY", "density");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // fluid with nonlinear viscosity according to a modified power law
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_modpowerlaw",
        "fluid with nonlinear viscosity according to a modified power law",
        Core::Materials::m_modpowerlaw);

    add_named_real(m, "MCONS", "consistency");
    add_named_real(m, "DELTA", "safety factor");
    add_named_real(m, "AEXP", "exponent");
    add_named_real(m, "DENSITY", "density");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // fluid with non-linear viscosity according to Herschel-Bulkley
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_herschelbulkley",
        "fluid with non-linear viscosity according to Herschel-Bulkley",
        Core::Materials::m_herschelbulkley);

    add_named_real(m, "TAU_0", "yield stress");
    add_named_real(m, "KFAC", "constant factor");
    add_named_real(m, "NEXP", "exponent");
    add_named_real(m, "MEXP", "exponent");
    add_named_real(m, "LOLIMSHEARRATE", "lower limit of shear rate");
    add_named_real(m, "UPLIMSHEARRATE", "upper limit of shear rate");
    add_named_real(m, "DENSITY", "density");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // lubrication material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_lubrication", "lubrication material", Core::Materials::m_lubrication);

    add_named_int(m, "LUBRICATIONLAWID", "lubrication law id");
    add_named_real(m, "DENSITY", "lubricant density");

    Mat::append_material_definition(matlist, m);
  }


  /*----------------------------------------------------------------------*/
  // constant lubrication material law
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_lubrication_law_constant",
        "constant lubrication material law", Core::Materials::m_lubrication_law_constant);

    add_named_real(m, "VISCOSITY", "lubricant viscosity");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Barus viscosity lubrication material law
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_lubrication_law_barus",
        "barus lubrication material law", Core::Materials::m_lubrication_law_barus);

    add_named_real(m, "ABSViscosity", "absolute lubricant viscosity");
    add_named_real(m, "PreVisCoeff", "pressure viscosity coefficient");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Roeland viscosity lubrication material law
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_lubrication_law_roeland",
        "roeland lubrication material law", Core::Materials::m_lubrication_law_roeland);

    add_named_real(m, "ABSViscosity", "absolute lubricant viscosity");
    add_named_real(m, "PreVisCoeff", "pressure viscosity coefficient");
    add_named_real(m, "RefVisc", "reference viscosity");
    add_named_real(m, "RefPress", "reference Pressure");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport material (with potential reaction coefficient)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_scatra", "scalar transport material", Core::Materials::m_scatra);

    add_named_real(m, "DIFFUSIVITY", "kinematic diffusivity");
    add_named_real(m, "REACOEFF", "reaction coefficient", 0.0, true);
    add_named_real(m, "SCNUM", "schmidt number", 0.0, true);
    add_named_real(m, "DENSIFICATION", "densification coefficient", 0.0, true);
    add_named_bool(m, "REACTS_TO_EXTERNAL_FORCE", "reacts to external force", false, true);

    Mat::append_material_definition(matlist, m);
  }


  /*----------------------------------------------------------------------*/
  // scalar transport material (with potential reaction coefficient)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_scatra_reaction_poro",
        "scalar transport material", Core::Materials::m_scatra_reaction_poroECM);

    add_named_int(m, "NUMSCAL", "number of scalars for these elements");
    add_named_int_vector(m, "STOICH", "reaction stoichometrie list", "NUMSCAL");
    add_named_real(m, "REACCOEFF", "reaction coefficient");
    add_named_real(m, "REACSCALE", "scaling for reaction coefficient");
    // reacscale could now be done by constant distribution function
    add_named_int(m, "DISTRFUNCT", "spatial distribution of reaction coefficient", 0, true);
    add_named_string(m, "COUPLING",
        "type of coupling: "
        "simple_multiplicative, power_multiplicative, constant, michaelis_menten, by_function, "
        "no_coupling (default)",
        "no_coupling", false);
    add_named_real_vector(m, "ROLE", "role in michaelis-menten like reactions", "NUMSCAL");
    add_named_real_vector(m, "REACSTART", "starting point of reaction", "NUMSCAL", 0.0, true);

    Mat::append_material_definition(matlist, m);
  }
  /*----------------------------------------------------------------------*/
  // scalar transport reaction material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_scatra_reaction", "advanced reaction material", Core::Materials::m_scatra_reaction);

    add_named_int(m, "NUMSCAL", "number of scalars for these elements");
    add_named_int_vector(m, "STOICH", "reaction stoichometrie list", "NUMSCAL");
    add_named_real(m, "REACCOEFF", "reaction coefficient");
    add_named_int(m, "DISTRFUNCT", "spatial distribution of reaction coefficient", 0, true);
    add_named_string(m, "COUPLING",
        "type of coupling: "
        "simple_multiplicative, power_multiplicative, constant, michaelis_menten, by_function, "
        "no_coupling (default)",
        "no_coupling", false);
    add_named_real_vector(m, "ROLE", "role in michaelis-menten like reactions", "NUMSCAL");
    add_named_real_vector(m, "REACSTART", "starting point of reaction", "NUMSCAL", 0.0, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport reaction material (species in fluid)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_scatra_multiporo_fluid",
        "advanced reaction material for multiphase porous flow (species in fluid)",
        Core::Materials::m_scatra_multiporo_fluid);

    add_named_real(m, "DIFFUSIVITY", "kinematic diffusivity");
    add_named_int(m, "PHASEID", "ID of fluid phase the scalar is associated with");
    add_named_real(m, "REACOEFF", "reaction coefficient", 0.0, true);
    add_named_real(m, "SCNUM", "schmidt number", 0.0, true);
    add_named_real(m, "DENSIFICATION", "densification coefficient", 0.0, true);
    add_named_real(m, "DELTA", "delta", 0.0, true);
    add_named_real(m, "MIN_SAT",
        "minimum saturation under which also corresponding mass fraction is equal to zero", 1.0e-9,
        true);
    add_named_bool(m, "REACTS_TO_EXTERNAL_FORCE", "reacts to external force", false, true);
    add_named_int(m, "RELATIVE_MOBILITY_FUNCTION_ID", "relative mobility function ID", 0, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport reaction material (species in volume fraction)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_scatra_multiporo_volfrac",
        "advanced reaction material for multiphase porous flow (species in volfrac)",
        Core::Materials::m_scatra_multiporo_volfrac);

    add_named_real(m, "DIFFUSIVITY", "kinematic diffusivity");
    add_named_int(m, "PHASEID", "ID of fluid phase the scalar is associated with");
    add_named_real(m, "REACOEFF", "reaction coefficient", 0.0, true);
    add_named_real(m, "SCNUM", "schmidt number", 0.0, true);
    add_named_real(m, "DENSIFICATION", "densification coefficient", 0.0, true);
    add_named_real(m, "DELTA", "delta", 0.0, true);
    add_named_bool(m, "REACTS_TO_EXTERNAL_FORCE", "reacts to external force", false, true);
    add_named_int(m, "RELATIVE_MOBILITY_FUNCTION_ID", "relative mobility function ID", 0, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport reaction material (species in solid)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_scatra_multiporo_solid",
        "advanced reaction material for multiphase "
        "porous flow (species in solid)",
        Core::Materials::m_scatra_multiporo_solid);

    add_named_real(m, "DIFFUSIVITY", "kinematic diffusivity");
    // no phaseID because only one solid phase
    add_named_real(m, "REACOEFF", "reaction coefficient", 0.0, true);
    add_named_real(m, "SCNUM", "schmidt number", 0.0, true);
    add_named_real(m, "DENSIFICATION", "densification coefficient", 0.0, true);
    add_named_real(m, "DELTA", "delta", 0.0, true);
    add_named_bool(m, "REACTS_TO_EXTERNAL_FORCE", "reacts to external force", false, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport reaction material (temperature)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_scatra_multiporo_temperature",
        "advanced reaction material for multiphase porous flow (temperature)",
        Core::Materials::m_scatra_multiporo_temperature);

    add_named_int(m, "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE", "number of fluid dofs");
    add_named_real_vector(
        m, "CP_FLUID", "heat capacity fluid phases", "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE");
    add_named_int(m, "NUMVOLFRAC", "number of volfrac dofs");
    add_named_real_vector(m, "CP_VOLFRAC", "heat capacity volfrac", "NUMVOLFRAC");
    add_named_real(m, "CP_SOLID", "heat capacity solid");
    add_named_real_vector(m, "KAPPA_FLUID", "thermal diffusivity fluid phases",
        "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE");
    add_named_real_vector(m, "KAPPA_VOLFRAC", "thermal diffusivity volfrac", "NUMVOLFRAC");
    add_named_real(m, "KAPPA_SOLID", "heat capacity solid");
    add_named_real(m, "DIFFUSIVITY", "kinematic diffusivity", 1.0, true);
    add_named_real(m, "REACOEFF", "reaction coefficient", 0.0, true);
    add_named_real(m, "SCNUM", "schmidt number", 0.0, true);
    add_named_real(m, "DENSIFICATION", "densification coefficient", 0.0, true);
    add_named_bool(m, "REACTS_TO_EXTERNAL_FORCE", "reacts to external force", false, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport chemotaxis material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_scatra_chemotaxis", "chemotaxis material", Core::Materials::m_scatra_chemotaxis);

    add_named_int(m, "NUMSCAL", "number of chemotactic pairs for these elements");
    add_named_int_vector(m, "PAIR", "chemotaxis pairing", "NUMSCAL");
    add_named_real(m, "CHEMOCOEFF", "chemotaxis coefficient");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // scalar transport material for multi-scale approach
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_scatra_multiscale",
        "scalar transport material for multi-scale approach", Core::Materials::m_scatra_multiscale);

    add_named_string(m, "MICROFILE", "input file for micro scale", "filename.dat");
    add_named_int(m, "MICRODIS_NUM", "number of micro-scale discretization");
    add_named_real(m, "POROSITY", "porosity");
    add_named_real(m, "TORTUOSITY", "tortuosity");
    add_named_real(m, "A_s", "specific micro-scale surface area");
    add_named_real(m, "DIFFUSIVITY", "kinematic diffusivity");
    add_named_real(m, "REACOEFF", "reaction coefficient", 0.0, true);
    add_named_real(m, "SCNUM", "Schmidt number", 0.0, true);
    add_named_real(m, "DENSIFICATION", "densification coefficient", 0.0, true);
    add_named_bool(m, "REACTS_TO_EXTERNAL_FORCE", "reacts to external force", false, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Weickenmeier muscle material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Muscle_Weickenmeier",
        "Weickenmeier muscle material", Core::Materials::m_muscle_weickenmeier);

    add_named_real(m, "ALPHA", "experimentally fitted material parameter");
    add_named_real(m, "BETA", "experimentally fitted material parameter");
    add_named_real(m, "GAMMA", "experimentally fitted material parameter");
    add_named_real(m, "KAPPA", "material parameter for coupled volumetric contribution");
    add_named_real(m, "OMEGA0", "weighting factor for isotropic tissue constituents");
    add_named_real(
        m, "ACTMUNUM", "number of active motor units per undeformed muscle cross-sectional area");
    add_named_int(m, "MUTYPESNUM", "number of motor unit types");
    add_named_real_vector(m, "INTERSTIM", "interstimulus interval", "MUTYPESNUM");
    add_named_real_vector(m, "FRACACTMU", "fraction of motor unit type", "MUTYPESNUM");
    add_named_real_vector(m, "FTWITCH", "twitch force of motor unit type", "MUTYPESNUM");
    add_named_real_vector(m, "TTWITCH", "twitch contraction time of motor unit type", "MUTYPESNUM");
    add_named_real(m, "LAMBDAMIN", "minimal active fiber stretch");
    add_named_real(
        m, "LAMBDAOPT", "optimal active fiber stretch related to active nominal stress maximum");
    add_named_real(m, "DOTLAMBDAMIN", "minimal stretch rate");
    add_named_real(m, "KE",
        "parameter controlling the curvature of the velocity dependent activation function in the "
        "eccentric case");
    add_named_real(m, "KC",
        "parameter controlling the curvature of the velocity dependent activation function in the "
        "concentric case");
    add_named_real(m, "DE",
        "parameter controlling the amplitude of the velocity dependent activation function in the "
        "eccentric case");
    add_named_real(m, "DC",
        "parameter controlling the amplitude of the velocity dependent activation function in the "
        "concentric case");
    add_named_int(m, "ACTTIMESNUM", "number of time boundaries to prescribe activation");
    add_named_real_vector(m, "ACTTIMES", "time boundaries between intervals", "ACTTIMESNUM");
    add_named_int(m, "ACTINTERVALSNUM", "number of time intervals to prescribe activation");
    add_named_real_vector(m, "ACTVALUES",
        "scaling factor in intervals (1=full activation, 0=no activation)", "ACTINTERVALSNUM");
    add_named_real(m, "DENS", "density");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Combo muscle material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_Muscle_Combo", "Combo muscle material", Core::Materials::m_muscle_combo);

    add_named_real(m, "ALPHA", "experimentally fitted material parameter");
    add_named_real(m, "BETA", "experimentally fitted material parameter");
    add_named_real(m, "GAMMA", "experimentally fitted material parameter");
    add_named_real(m, "KAPPA", "material parameter for coupled volumetric contribution");
    add_named_real(m, "OMEGA0", "weighting factor for isotropic tissue constituents");
    add_named_real(m, "POPT", "tetanised optimal (maximal) active stress");
    add_named_real(m, "LAMBDAMIN", "minimal active fiber stretch");
    add_named_real(
        m, "LAMBDAOPT", "optimal active fiber stretch related to active nominal stress maximum");

    std::map<int, std::pair<std::string, std::vector<std::shared_ptr<Input::LineComponent>>>>
        activation_evaluation_choices;

    // choice function
    std::vector<std::shared_ptr<Input::LineComponent>> activation_function;
    activation_function.emplace_back(std::make_shared<Input::SeparatorComponent>(
        "FUNCTID", "function id for time- and space-dependency of muscle activation"));
    activation_function.emplace_back(std::make_shared<Input::IntComponent>("FUNCTID"));
    activation_evaluation_choices.emplace(
        static_cast<int>(Inpar::Mat::ActivationType::function_of_space_time),
        std::make_pair("function", activation_function));

    // choice map
    // definition of operation and print string for post processed component "MAPFILE"
    using actMapType = std::unordered_map<int, std::vector<std::pair<double, double>>>;
    std::function<const actMapType(const std::string&)> operation =
        [](const std::string& map_file) -> actMapType
    {
      // map pattern file needs to be placed in same folder as input file
      std::filesystem::path input_file_path =
          Global::Problem::instance()->output_control_file()->input_file_name();
      const auto map_file_path = input_file_path.replace_filename(map_file);

      std::ifstream file_stream(map_file_path);

      if (file_stream.fail()) FOUR_C_THROW("Invalid file %s!", map_file_path.c_str());

      auto map_reduction_operation = [](actMapType acc, const actMapType& next)
      {
        for (const auto& [key, value] : next)
        {
          acc[key] = value;
        }
        return acc;
      };

      return Core::IO::convert_lines<actMapType, actMapType>(file_stream, map_reduction_operation);
    };
    const std::string print_string = std::string(
        "map of activation values retrieved from pattern file with rows in the format \"eleid: "
        "time_1, act_value_1; time_2, act_value_2; ...\"");

    std::vector<std::shared_ptr<Input::LineComponent>> activation_map;
    activation_map.emplace_back(std::make_shared<Input::SeparatorComponent>("MAPFILE",
        "pattern file containing a map of elementwise-defined discrete values for time- and "
        "space-dependency of muscle activation"));
    activation_map.emplace_back(
        std::make_shared<Input::ProcessedComponent>("MAPFILE", operation, print_string, false));
    activation_evaluation_choices.emplace(
        static_cast<int>(Inpar::Mat::ActivationType::map), std::make_pair("map", activation_map));

    m->add_component(std::make_shared<Input::SeparatorComponent>("ACTEVALTYPE",
        "type of time- and space-dependency of muscle activation: "
        "function or map",
        false));
    m->add_component(std::make_shared<Input::SwitchComponent>("ACTEVALTYPE",
        Inpar::Mat::ActivationType::function_of_space_time, activation_evaluation_choices));

    add_named_real(m, "DENS", "density");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Active strain Giantesio muscle material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Muscle_Giantesio",
        "Giantesio active strain muscle material", Core::Materials::m_muscle_giantesio);

    add_named_real(m, "ALPHA", "experimentally fitted material parameter");
    add_named_real(m, "BETA", "experimentally fitted material parameter");
    add_named_real(m, "GAMMA", "experimentally fitted material parameter");
    add_named_real(m, "KAPPA", "material parameter for coupled volumetric contribution");
    add_named_real(m, "OMEGA0", "weighting factor for isotropic tissue constituents");
    add_named_real(
        m, "ACTMUNUM", "number of active motor units per undeformed muscle cross-sectional area");
    add_named_int(m, "MUTYPESNUM", "number of motor unit types");
    add_named_real_vector(m, "INTERSTIM", "interstimulus interval", "MUTYPESNUM");
    add_named_real_vector(m, "FRACACTMU", "fraction of motor unit type", "MUTYPESNUM");
    add_named_real_vector(m, "FTWITCH", "twitch force of motor unit type", "MUTYPESNUM");
    add_named_real_vector(m, "TTWITCH", "twitch contraction time of motor unit type", "MUTYPESNUM");
    add_named_real(m, "LAMBDAMIN", "minimal active fiber stretch");
    add_named_real(
        m, "LAMBDAOPT", "optimal active fiber stretch related to active nominal stress maximum");
    add_named_real(m, "DOTLAMBDAMIN", "minimal stretch rate");
    add_named_real(m, "KE",
        "parameter controlling the curvature of the velocity dependent activation function in the "
        "eccentric case");
    add_named_real(m, "KC",
        "parameter controlling the curvature of the velocity dependent activation function in the "
        "concentric case");
    add_named_real(m, "DE",
        "parameter controlling the amplitude of the velocity dependent activation function in the "
        "eccentric case");
    add_named_real(m, "DC",
        "parameter controlling the amplitude of the velocity dependent activation function in the "
        "concentric case");
    add_named_int(m, "ACTTIMESNUM", "number of time boundaries to prescribe activation");
    add_named_real_vector(m, "ACTTIMES", "time boundaries between intervals", "ACTTIMESNUM");
    add_named_int(m, "ACTINTERVALSNUM", "number of time intervals to prescribe activation");
    add_named_real_vector(m, "ACTVALUES",
        "scaling factor in intervals (1=full activation, 0=no activation)", "ACTINTERVALSNUM");
    add_named_real(m, "DENS", "density");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Myocard muscle material (with complicated reaction coefficient)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_myocard", "Myocard muscle material", Core::Materials::m_myocard);

    add_named_real(m, "DIFF1", "conductivity in fiber direction");
    add_named_real(m, "DIFF2", "conductivity perpendicular to fiber direction");
    add_named_real(m, "DIFF3", "conductivity perpendicular to fiber direction");
    add_named_real(
        m, "PERTUBATION_DERIV", "pertubation for calculation of reaction coefficient derivative");
    add_named_string(m, "MODEL", "Model type: MV (default), FHN, TNNP, SAN or INADA", "MV");
    add_named_string(m, "TISSUE", "Tissue type: M (default), ENDO, EPI, AN, N or NH", "M");
    add_named_real(m, "TIME_SCALE", "Scale factor for time units of Model");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material according to Sutherland law
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_sutherland", "material according to Sutherland law", Core::Materials::m_sutherland);

    add_named_real(m, "REFVISC", "reference dynamic viscosity (kg/(m*s))");
    add_named_real(m, "REFTEMP", "reference temperature (K)");
    add_named_real(m, "SUTHTEMP", "Sutherland temperature (K)");
    add_named_real(m, "SHC", "specific heat capacity at constant pressure (J/(kg*K))");
    add_named_real(m, "PRANUM", "Prandtl number");
    add_named_real(m, "THERMPRESS", "(initial) thermodynamic pressure (J/m^3)");
    add_named_real(m, "GASCON", "specific gas constant R (J/(kg*K))");

    Mat::append_material_definition(matlist, m);
  }


  /*----------------------------------------------------------------------*/
  // material parameters for ion species in electrolyte solution (gjb 07/08)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_ion",
        "material parameters for ion species in electrolyte solution", Core::Materials::m_ion);

    add_named_real(m, "DIFFUSIVITY", "kinematic diffusivity");
    add_named_real(m, "VALENCE", "valence (= charge number)");
    add_named_real(m, "DENSIFICATION", "densification coefficient", 0.0, true);
    // via these two optional parameters we can bring the material parameters
    // of one eliminated ionic species into 4C if needed
    add_named_real(m, "ELIM_DIFFUSIVITY", "kinematic diffusivity of elim. species", 0.0, true);
    add_named_real(m, "ELIM_VALENCE", "valence of elim. species", 0.0, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material parameters for ion species in electrolyte solution (ehrl 07/12)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_newman",
        "material parameters for ion species in electrolyte solution", Core::Materials::m_newman);

    add_named_real(m, "VALENCE", "valence (= charge number)");
    add_named_int(m, "DIFF_COEF_CONC_DEP_FUNCT",
        "function number of function describing concentration dependence of diffusion coefficient",
        0);
    add_named_int(m, "DIFF_COEF_TEMP_SCALE_FUNCT",
        "FUNCT number describing temperature scaling of diffusion coefficient", 0);
    add_named_int(m, "TRANSNR", "curve number for transference number");
    add_named_int(m, "THERMFAC", "curve number for thermodynamic factor");
    add_named_int(m, "COND_CONC_DEP_FUNCT",
        "function number of function describing concentration dependence of conductivity", 0);
    add_named_int(m, "COND_TEMP_SCALE_FUNCT",
        "FUNCT number describing temperature scaling of conductivity", 0);
    // optional parameter for implemented concentration depending function
    add_named_int(m, "DIFF_PARA_NUM", "number of parameters for diffusion coefficient", 0, true);
    add_named_real_vector(
        m, "DIFF_PARA", "parameters for diffusion coefficient", "DIFF_PARA_NUM", 0.0, true);
    add_named_int(m, "DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM",
        "number of parameters for scaling function describing temperature dependence of diffusion "
        "coefficient",
        0, true);
    add_named_real_vector(m, "DIFF_COEF_TEMP_SCALE_FUNCT_PARA",
        "parameters for function describing temperature dependence of diffusion coefficient",
        "DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM", 0.0, true);
    add_named_int(m, "TRANS_PARA_NUM", "number of parameters for transference number", 0, true);
    add_named_real_vector(
        m, "TRANS_PARA", "parameters for transference number", "TRANS_PARA_NUM", 0.0, true);
    add_named_int(m, "THERM_PARA_NUM", "number of parameters for thermodynamic factor", 0, true);
    add_named_real_vector(
        m, "THERM_PARA", "parameters for thermodynamic factor", "THERM_PARA_NUM", 0.0, true);
    add_named_int(m, "COND_PARA_NUM", "number of parameters for conductivity", 0, true);
    add_named_real_vector(
        m, "COND_PARA", "parameters for conductivity", "COND_PARA_NUM", 0.0, true);
    add_named_int(m, "COND_TEMP_SCALE_FUNCT_PARA_NUM",
        "number of parameters for temperature scaling of conductivity", 0, true);
    add_named_real_vector(m, "COND_TEMP_SCALE_FUNCT_PARA",
        "parameters for temperature scaling of conductivity", "COND_TEMP_SCALE_FUNCT_PARA_NUM", 0.0,
        true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material parameters for ion species in electrolyte solution for multi-scale approach (fang
  // 07/17)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_newman_multiscale",
        "material parameters for ion species in electrolyte solution for multi-scale approach",
        Core::Materials::m_newman_multiscale);

    add_named_real(m, "VALENCE", "valence (= charge number)");
    add_named_int(m, "DIFF_COEF_CONC_DEP_FUNCT",
        "function number of function describing concentration dependence of diffusion coefficient",
        0);
    add_named_int(m, "DIFF_COEF_TEMP_SCALE_FUNCT",
        "FUNCT number describing temperature scaling of diffusion coefficient", 0);
    add_named_int(m, "TRANSNR", "curve number for transference number");
    add_named_int(m, "THERMFAC", "curve number for thermodynamic factor");
    add_named_int(m, "COND_CONC_DEP_FUNCT",
        "function number of function describing concentration dependence of conductivity", 0);
    add_named_int(m, "COND_TEMP_SCALE_FUNCT",
        "FUNCT number describing temperature scaling of conductivity", 0);
    add_named_real(m, "ELECTRONIC_COND", "electronic conductivity");
    add_named_int(m, "ELECTRONIC_COND_CONC_SCALE_FUNC_NUM",
        "FUNCT number describing concentration dependence of electronic conductivity", 0);
    add_named_real(m, "A_s", "specific micro-scale surface area");
    add_named_string(m, "MICROFILE", "input file for micro scale", "filename.dat");
    add_named_int(m, "MICRODIS_NUM", "number of micro-scale discretization");
    // optional parameters for implemented concentration-depending functions
    add_named_int(m, "DIFF_PARA_NUM", "number of parameters for diffusion coefficient", 0, true);
    add_named_real_vector(
        m, "DIFF_PARA", "parameters for diffusion coefficient", "DIFF_PARA_NUM", 0.0, true);
    add_named_int(m, "DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM",
        "number of parameters for scaling function describing temperature dependence of diffusion "
        "coefficient",
        0, true);
    add_named_real_vector(m, "DIFF_COEF_TEMP_SCALE_FUNCT_PARA",
        "parameters for function describing temperature dependence of diffusion coefficient",
        "DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM", 0.0, true);
    add_named_int(m, "TRANS_PARA_NUM", "number of parameters for transference number", 0, true);
    add_named_real_vector(
        m, "TRANS_PARA", "parameters for transference number", "TRANS_PARA_NUM", 0.0, true);
    add_named_int(m, "THERM_PARA_NUM", "number of parameters for thermodynamic factor", 0, true);
    add_named_real_vector(
        m, "THERM_PARA", "parameters for thermodynamic factor", "THERM_PARA_NUM", 0.0, true);
    add_named_int(m, "COND_PARA_NUM", "number of parameters for ionic conductivity", 0, true);
    add_named_real_vector(
        m, "COND_PARA", "parameters for ionic conductivity", "COND_PARA_NUM", 0.0, true);
    add_named_int(m, "COND_TEMP_SCALE_FUNCT_PARA_NUM",
        "number of parameters for temperature scaling of conductivity", 0, true);
    add_named_real_vector(m, "COND_TEMP_SCALE_FUNCT_PARA",
        "parameters for temperature scaling of conductivity", "COND_TEMP_SCALE_FUNCT_PARA_NUM", 0.0,
        true);

    Mat::append_material_definition(matlist, m);
  }
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_scl", "material parameters for space charge layers", Core::Materials::m_scl);

    add_named_real(m, "VALENCE", "valence/charge number");
    add_named_int(m, "DIFF_COEF_CONC_DEP_FUNCT",
        "function number of function describing concentration dependence of diffusion coefficient",
        0);
    add_named_int(m, "DIFF_COEF_TEMP_SCALE_FUNCT",
        "function number describing temperature scaling of diffusion coefficient", 0);
    add_named_int(m, "TRANSNR", "curve number for transference number");
    add_named_int(m, "COND_CONC_DEP_FUNCT",
        "function number of function describing concentration dependence of conductivity", 0);
    add_named_int(m, "COND_TEMP_SCALE_FUNCT",
        "function number describing temperature scaling of conductivity", 0);
    add_named_int(m, "DIFF_PARA_NUM", "number of parameters for diffusion coefficient", 0, true);
    add_named_real_vector(
        m, "DIFF_PARA", "parameters for diffusion coefficient", "DIFF_PARA_NUM", 0.0, true);
    add_named_int(m, "DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM",
        "number of parameters for scaling function describing temperature dependence of diffusion "
        "coefficient",
        0, true);
    add_named_real_vector(m, "DIFF_COEF_TEMP_SCALE_FUNCT_PARA",
        "parameters for function describing temperature dependence of diffusion coefficient",
        "DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM", 0.0, true);
    add_named_int(m, "TRANS_PARA_NUM", "number of parameters for transference number", 0, true);
    add_named_real_vector(
        m, "TRANS_PARA", "parameters for transference number", "TRANS_PARA_NUM", 0.0, true);
    add_named_int(m, "COND_PARA_NUM", "number of parameters for conductivity", 0, true);
    add_named_real_vector(
        m, "COND_PARA", "parameters for conductivity", "COND_PARA_NUM", 0.0, true);
    add_named_int(m, "COND_TEMP_SCALE_FUNCT_PARA_NUM",
        "number of parameters for temperature scaling of conductivity", 0, true);
    add_named_real_vector(m, "COND_TEMP_SCALE_FUNCT_PARA",
        "parameters for temperature scaling of conductivity", "COND_TEMP_SCALE_FUNCT_PARA_NUM", 0.0,
        true);
    add_named_real(m, "MAX_CONC", "maximum cation concentration", 1.0);
    add_named_int(m, "EXTRAPOL_DIFF",
        "strategy for extrapolation of diffusion coefficient below 0 and above MAX_CONC (-1: "
        "disabled, 0: constant)",
        0);
    add_named_real(m, "LIM_CONC", "limiting concentration for extrapolation", 1.0, true);
    add_named_real(m, "BULK_CONC", "bulk ion concentration", 1.0);
    add_named_real(m, "SUSCEPT", "susceptibility", 1.0);
    add_named_real(m, "DELTA_NU", "difference of partial molar volumes (vacancy & cation)", 0.0);

    Mat::append_material_definition(matlist, m);
  }


  /*----------------------------------------------------------------------*/
  // electrode material (fang 02/15)
  {
    auto matelectrode = std::make_shared<Mat::MaterialDefinition>(
        "MAT_electrode", "electrode material", Core::Materials::m_electrode);

    // diffusivity and electronic conductivity
    add_named_int(matelectrode, "DIFF_COEF_CONC_DEP_FUNCT",
        "function number of function describing concentration dependence of diffusion coefficient",
        0);
    add_named_int(matelectrode, "DIFF_COEF_TEMP_SCALE_FUNCT",
        "FUNCT number describing temperature scaling of diffusion coefficient", 0);
    add_named_int(matelectrode, "COND_CONC_DEP_FUNCT",
        "function number of function describing concentration dependence of conductivity", 0);
    add_named_int(matelectrode, "COND_TEMP_SCALE_FUNCT",
        "FUNCT number describing temperature scaling of conductivity", 0);

    // optional parameters for concentration dependency of diffusivity and electronic conductivity
    add_named_int(
        matelectrode, "DIFF_PARA_NUM", "number of parameters for diffusion coefficient", 0, true);
    add_named_real_vector(matelectrode, "DIFF_PARA", "parameters for diffusion coefficient",
        "DIFF_PARA_NUM", 0.0, true);
    add_named_int(matelectrode, "DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM",
        "number of parameters for scaling function describing temperature dependence of diffusion "
        "coefficient",
        0, true);
    add_named_real_vector(matelectrode, "DIFF_COEF_TEMP_SCALE_FUNCT_PARA",
        "parameters for function describing temperature dependence of diffusion coefficient",
        "DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM", 0.0, true);
    add_named_int(
        matelectrode, "COND_PARA_NUM", "number of parameters for electronic conductivity", 0, true);
    add_named_real_vector(matelectrode, "COND_PARA", "parameters for electronic conductivity",
        "COND_PARA_NUM", 0.0, true);
    add_named_int(matelectrode, "COND_TEMP_SCALE_FUNCT_PARA_NUM",
        "number of parameters for temperature scaling of conductivity", 0, true);
    add_named_real_vector(matelectrode, "COND_TEMP_SCALE_FUNCT_PARA",
        "parameters for temperature scaling of conductivity", "COND_TEMP_SCALE_FUNCT_PARA_NUM", 0.0,
        true);
    // saturation value of intercalated Lithium concentration
    add_named_real(matelectrode, "C_MAX", "saturation value of intercalated Lithium concentration");

    // lithiation value corresponding to saturation value of intercalated Lithium concentration
    add_named_real(matelectrode, "CHI_MAX",
        "lithiation value corresponding to saturation value of intercalated Lithium concentration "
        "'C_MAX'");

    // model for half cell open circuit potential of electrode
    add_named_string(matelectrode, "OCP_MODEL",
        "model for half cell open circuit potential of electrode: "
        "Redlich-Kister, Taralov, Polynomial, csv",
        "none");

    // lower bound of range of validity as a fraction of C_MAX for ocp calculation model
    add_named_real(matelectrode, "X_MIN",
        "lower bound of range of validity as a fraction of C_MAX for ocp calculation model", 2.0,
        false);

    // upper bound of range of validity as a fraction of C_MAX for ocp calculation model
    add_named_real(matelectrode, "X_MAX",
        "upper bound of range of validity as a fraction of C_MAX for ocp calculation model", 2.0,
        false);

    // number of parameters underlying half cell open circuit potential model
    add_named_int(matelectrode, "OCP_PARA_NUM",
        "number of parameters underlying half cell open circuit potential model", 0, true);

    // parameters underlying half cell open circuit potential model
    add_named_real_vector(matelectrode, "OCP_PARA",
        "parameters underlying half cell open circuit potential model", "OCP_PARA_NUM", 0., true);

    // *.csv file with data points for half cell open circuit potential
    add_named_string(matelectrode, "OCP_CSV",
        "\\*.csv file with data points for half cell open circuit potential", "", true);

    // add electrode material to global list of valid materials
    Mat::append_material_definition(matlist, matelectrode);
  }

  /*----------------------------------------------------------------------*/
  // material collection (gjb 07/08)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_matlist",
        "list/collection of materials, i.e. material IDs", Core::Materials::m_matlist);

    add_named_bool(
        m, "LOCAL", "individual materials allocated per element or only at global scope");
    // add_named_int(m,"LOCAL","individual materials allocated per element or only at global
    // scope");
    add_named_int(m, "NUMMAT", "number of materials in list");
    add_named_int_vector(m, "MATIDS", "the list material IDs", "NUMMAT");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material collection with reactions (thon 09/14)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_matlist_reactions",
        "list/collection of materials, i.e. material IDs and list of reactions",
        Core::Materials::m_matlist_reactions);

    add_named_bool(
        m, "LOCAL", "individual materials allocated per element or only at global scope");
    add_named_int(m, "NUMMAT", "number of materials in list");
    add_named_int_vector(m, "MATIDS", "the list material IDs", "NUMMAT");
    add_named_int(m, "NUMREAC", "number of reactions for these elements", 0);
    add_named_int_vector(m, "REACIDS", "advanced reaction list", "NUMREAC", 0);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material collection with chemotaxis (thon 06/15)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_matlist_chemotaxis",
        "list/collection of materials, i.e. material IDs and list of chemotactic pairs",
        Core::Materials::m_matlist_chemotaxis);

    add_named_bool(
        m, "LOCAL", "individual materials allocated per element or only at global scope");
    add_named_int(m, "NUMMAT", "number of materials in list");
    add_named_int_vector(m, "MATIDS", "the list material IDs", "NUMMAT");
    add_named_int(m, "NUMPAIR", "number of pairs for these elements", 0);
    add_named_int_vector(m, "PAIRIDS", "chemotaxis pairs list", "NUMPAIR", 0);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material collection with reactions AND chemotaxis (thon 06/15)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_matlist_chemo_reac",
        "list/collection of materials, i.e. material IDs and list of reactive/chemotactic pairs",
        Core::Materials::m_matlist_chemoreac);

    add_named_bool(
        m, "LOCAL", "individual materials allocated per element or only at global scope");
    add_named_int(m, "NUMMAT", "number of materials in list");
    add_named_int_vector(m, "MATIDS", "the list material IDs", "NUMMAT");
    add_named_int(m, "NUMPAIR", "number of pairs for these elements", 0);
    add_named_int_vector(m, "PAIRIDS", "chemotaxis pairs list", "NUMPAIR", 0);
    add_named_int(m, "NUMREAC", "number of reactions for these elements", 0);
    add_named_int_vector(m, "REACIDS", "advanced reaction list", "NUMREAC", 0);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material collection (ehrl 11/12)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_elchmat",
        "specific list/collection of species and phases for elch applications",
        Core::Materials::m_elchmat);

    add_named_bool(m, "LOCAL", "individual materials allocated per element or only at global scope",
        false, true);
    add_named_int(m, "NUMDOF", "number of dof's per node");
    add_named_int(m, "NUMSCAL", "number of transported scalars per node");
    add_named_int(m, "NUMPHASE", "number of phases in electrolyte");
    add_named_int_vector(m, "PHASEIDS", "the list phasel IDs", "NUMPHASE");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material collection (ehrl 11/12)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_elchphase",
        "material parameters for ion species in electrolyte solution",
        Core::Materials::m_elchphase);

    add_named_bool(m, "LOCAL", "individual materials allocated per element or only at global scope",
        false, true);
    add_named_real(m, "EPSILON", "phase porosity");
    add_named_real(m, "TORTUOSITY", "inverse (!) of phase tortuosity");
    add_named_int(m, "NUMMAT", "number of materials in electrolyte");
    add_named_int_vector(m, "MATIDS", "the list phasel IDs", "NUMMAT");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // St.Venant--Kirchhoff
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_StVenantKirchhoff",
        "St.Venant--Kirchhoff material", Core::Materials::m_stvenant);

    add_named_real(m, "YOUNG", "Young's modulus");
    add_named_real(m, "NUE", "Poisson's ratio");
    add_named_real(m, "DENS", "mass density");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // St.Venant--Kirchhoff with temperature
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_ThrStVenantK",
        "Thermo St.Venant--Kirchhoff material", Core::Materials::m_thermostvenant);

    add_named_int(m, "YOUNGNUM",
        "number of Young's modulus in list (if 1 Young is const, if >1 Young is temperature) "
        "dependent");
    add_named_real_vector(m, "YOUNG", "Young's modulus", "YOUNGNUM");
    add_named_real(m, "NUE", "Poisson's ratio");
    add_named_real(m, "DENS", "mass density");
    add_named_real(m, "THEXPANS", "constant coefficient of linear thermal expansion");
    add_named_real(m, "CAPA", "capacity");
    add_named_real(m, "CONDUCT", "conductivity");
    add_named_real(m, "INITTEMP", "initial temperature");
    add_named_int(m, "THERMOMAT", "mat id of thermal material part", -1, true);

    Mat::append_material_definition(matlist, m);
  }
  /*----------------------------------------------------------------------*/
  // Plastic linear elastic St.Venant Kirchhoff / Drucker Prager plasticity
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_DruckerPrager",
        "elastic St.Venant Kirchhoff / plastic drucker prager", Core::Materials::m_pldruckprag);

    add_named_real(m, "YOUNG", "Young's modulus");
    add_named_real(m, "NUE", "Poisson's ratio");
    add_named_real(m, "DENS", "Density");
    add_named_real(m, "ISOHARD", "linear isotropic hardening");
    add_named_real(m, "TOL", "Local Newton iteration tolerance");
    add_named_real(m, "C", "cohesion");
    add_named_real(m, "ETA", "Drucker Prager Constant Eta");
    add_named_real(m, "XI", "Drucker Prager Constant Xi");
    add_named_real(m, "ETABAR", "Drucker Prager Constant Etabar");
    add_named_string(m, "TANG", "Method to compute the material tangent", "consistent", true);
    add_named_int(m, "MAXITER", "Maximum Iterations for local Neuton Raphson", 50, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Linear thermo-elastic St.Venant Kirchhoff / plastic von Mises
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_ThrPlasticLinElast",
        "Thermo-elastic St.Venant Kirchhoff / plastic von Mises material",
        Core::Materials::m_thermopllinelast);

    add_named_real(m, "YOUNG", "Young's modulus");
    add_named_real(m, "NUE", "Poisson's ratio");
    add_named_real(m, "DENS", "mass density");
    add_named_real(m, "THEXPANS", "coefficient of linear thermal expansion");
    add_named_real(m, "INITTEMP", "initial temperature");
    add_named_real(m, "YIELD", "yield stress");
    add_named_real(m, "ISOHARD", "isotropic hardening modulus");
    add_named_real(m, "KINHARD", "kinematic hardening modulus");
    add_named_int(m, "SAMPLENUM", "number of stress-strain pairs in list");
    add_named_real_vector(m, "SIGMA_Y", "yield stress", "SAMPLENUM");
    add_named_real_vector(
        m, "EPSBAR_P", "accumulated plastic strain corresponding to SIGMA_Y", "SAMPLENUM");
    add_named_real(m, "TOL", "tolerance for local Newton iteration");
    add_named_int(m, "THERMOMAT", "mat id of thermal material part", -1, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Plastic linear elastic St.Venant Kirchhoff / GTN plasticity
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_PlasticGTN",
        "elastic St.Venant Kirchhoff / plastic GTN", Core::Materials::m_plgtn);

    add_named_real(m, "YOUNG", "Young's modulus");
    add_named_real(m, "NUE", "Poisson's ratio");
    add_named_real(m, "DENS", "Density");
    add_named_real(m, "YIELD", "yield stress");
    add_named_real(m, "ISOHARD", "linear isotropic hardening");
    add_named_int(m, "HARDENING_FUNC", "Function number for isotropic hardening", 0, true);
    add_named_real(m, "TOL", "Local Newton iteration tolerance");
    add_named_int(m, "MAXITER", "Maximum Neuton Raphson Iterations", 50, true);
    add_named_real(m, "K1", "GTN Constant k1");
    add_named_real(m, "K2", "GTN Constant k2");
    add_named_real(m, "K3", "GTN constant k3");
    add_named_real(m, "F0", "GTN constant f0 for initial damage");
    add_named_real(m, "FN", "GTN constant fN for damage nucleation");
    add_named_real(m, "EN", "GTN constant eN for damage nucleation");
    add_named_real(m, "SN", "GTN constant sN for damage nucleation");
    add_named_real(m, "FC", "GTN constant fC for damage coalescence");
    add_named_real(m, "KAPPA", "GTN constant kappa for damage coalescence");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Finite strain superelasticity of shape memory alloys
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_SuperElastSMA",
        "finite strain superelastic shape memory alloy", Core::Materials::m_superelast);

    add_named_real(m, "DENS", "mass density");
    add_named_real(m, "YOUNG", "Young's modulus");
    add_named_real(m, "NUE", "Poisson's ratio");
    add_named_real(m, "EPSILON_L",
        "parameter representing the maximum deformation obtainable only by detwinning of the "
        "multiple-variant martensite");
    add_named_real(m, "T_AS_s",
        "Temperature at which the phase transformation from austenite to martensite starts");
    add_named_real(m, "T_AS_f",
        "Temperature at which the phase transformation from austenite to martensite finishes");
    add_named_real(m, "T_SA_s",
        "Temperature at which the phase transformation from martensite to autenite starts");
    add_named_real(m, "T_SA_f",
        "Temperature at which the phase transformation from martensite to autenite finishes");
    add_named_real(m, "C_AS", "Coefficient of the linear temperature dependence of T_AS");
    add_named_real(m, "C_SA", "Coefficient of the linear temperature dependence of T_SA");
    add_named_real(m, "SIGMA_AS_s",
        "stress at which the phase transformation from austenite to martensite begins");
    add_named_real(m, "SIGMA_AS_f",
        "stress at which the phase transformation from austenite to martensite finishes");
    add_named_real(m, "SIGMA_SA_s",
        "stress at which the phase transformation from martensite to austenite begins");
    add_named_real(m, "SIGMA_SA_f",
        "stress at which the phase transformation from martensite to austenite finishes");
    add_named_real(m, "ALPHA", "pressure dependency in the drucker-prager-type loading");
    add_named_int(m, "MODEL",
        "Model used for the evolution of martensitic fraction (1=exponential; 2=linear)");
    add_named_real(m, "BETA_AS",
        "parameter, measuring the speed of the transformation from austenite to martensite", 0.,
        true);
    add_named_real(m, "BETA_SA",
        "parameter, measuring the speed of the transformation from martensite to austenite", 0.,
        true);


    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Thermo-hyperelasticity / finite strain von-Mises plasticity
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_ThrPlasticHyperElast",
        "Thermo-hyperelastic / finite strain plastic von Mises material "
        "with linear and exponential isotropic hardening",
        Core::Materials::m_thermoplhyperelast);

    add_named_real(m, "YOUNG", "Young's modulus");
    add_named_real(m, "NUE", "Poisson's ratio");
    add_named_real(m, "DENS", "mass density");
    add_named_real(m, "CTE", "coefficient of thermal expansion", 0., true);
    add_named_real(m, "INITTEMP", "initial, reference temperature", 0., true);
    add_named_real(m, "YIELD", "initial yield stress");
    add_named_real(m, "ISOHARD", "linear isotropic hardening modulus", 0., true);
    add_named_real(m, "SATHARDENING", "saturation hardening", 0., true);
    add_named_real(m, "HARDEXPO", "hardening exponent", 0., true);
    add_named_real(m, "YIELDSOFT", "thermal yield stress softening", 0., true);
    add_named_real(m, "HARDSOFT",
        "thermal hardening softening (acting on SATHARDENING and ISOHARD)", 0., true);
    add_named_real(m, "TOL", "tolerance for local Newton iteration", 1.e-8, true);
    add_named_int(m, "THERMOMAT", "mat id of thermal material part", -1, true);

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

    add_named_real(m, "YOUNG", "Young's modulus");
    add_named_real(m, "NUE", "Poisson's ratio");
    add_named_real(m, "DENS", "mass density");
    add_named_real(m, "YIELD", "yield stress", 0, true);
    add_named_real(m, "ISOHARD", "isotropic hardening modulus", 0, true);
    add_named_real(m, "SATHARDENING", "saturation hardening", 0, true);
    add_named_real(m, "HARDEXPO", "linear hardening exponent", 0, true);
    add_named_real(m, "VISC", "VISCOSITY", 0., true);
    add_named_real(m, "RATE_DEPENDENCY", "rate dependency", 0., true);
    add_named_int(m, "HARDENING_FUNC", "Function number for isotropic hardening", 0, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Plastic linear elastic St.Venant Kirchhoff / von Mises
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_PlasticLinElast",
        "elastic St.Venant Kirchhoff / plastic von Mises material "
        "with linear isotropic and kineamtic hardening",
        Core::Materials::m_pllinelast);

    add_named_real(m, "YOUNG", "Young's modulus");
    add_named_real(m, "NUE", "Poisson's ratio");
    add_named_real(m, "DENS", "mass density");
    add_named_real(m, "YIELD", "yield stress");
    add_named_real(m, "ISOHARD", "linear isotropic hardening modulus");
    add_named_real(m, "KINHARD", "linear kinematic hardening modulus");
    add_named_real(m, "TOL", "tolerance for local Newton iteration");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Elastic visco-plastic finite strain material law without yield surface
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_Viscoplastic_No_Yield_Surface",
        "Elastic visco-plastic finite strain material law without yield surface",
        Core::Materials::m_vp_no_yield_surface);

    // elasticity parameters
    add_named_real(m, "YOUNG", "Young's modulus");
    add_named_real(m, "NUE", "Poisson's ratio");
    add_named_real(m, "DENS", "material mass density");
    // visco-plasticity parameters
    add_named_real(m, "TEMPERATURE", "temperature in Kelvin");
    add_named_real(m, "PRE_EXP_FAC", "pre-exponential factor of plastic shear strain rate 'A'");
    add_named_real(m, "ACTIVATION_ENERGY", "activation energy 'Q'");
    add_named_real(m, "GAS_CONSTANT", "gas constant 'R'");
    add_named_real(m, "STRAIN_RATE_SENS", "strain-rate-sensitivity 'm'");
    add_named_real(m, "INIT_FLOW_RES", "initial isotropic flow resistance 'S^0'");
    add_named_real(m, "FLOW_RES_PRE_FAC", "flow resistance factor 'H_0'");
    add_named_real(m, "FLOW_RES_EXP", "flow resistance exponential value 'a'");
    add_named_real(m, "FLOW_RES_SAT_FAC", "flow resistance saturation factor 'S_*'");
    add_named_real(m, "FLOW_RES_SAT_EXP", "flow resistance saturation exponent 'b'");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Robinson's visco-plastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_Struct_Robinson", "Robinson's visco-plastic material", Core::Materials::m_vp_robinson);

    add_named_string(m, "KIND",
        "kind of Robinson material: "
        "Butler, Arya, Arya_NarloyZ (default), Arya_CrMoSteel",
        "Arya_NarloyZ");
    add_named_int(m, "YOUNGNUM", "number of Young's modulus in list");
    add_named_real_vector(m, "YOUNG", "Young's modulus", "YOUNGNUM");
    add_named_real(m, "NUE", "Poisson's ratio");
    add_named_real(m, "DENS", "mass density");
    add_named_real(m, "THEXPANS", "coefficient of linear thermal expansion");
    add_named_real(m, "INITTEMP", "initial temperature");
    add_named_real(m, "HRDN_FACT", "hardening factor 'A'");
    add_named_real(m, "HRDN_EXPO", "hardening power 'n'");
    add_named_int(m, "SHRTHRSHLDNUM", "number of shear stress threshold 'K^2'in list");
    add_named_real_vector(
        m, "SHRTHRSHLD", "Bingam-Prager shear stress threshold 'K^2'", "SHRTHRSHLDNUM");
    add_named_real(m, "RCVRY", "recovery factor 'R_0'");
    add_named_real(m, "ACTV_ERGY", "activation energy 'Q_0'");
    add_named_real(m, "ACTV_TMPR", "activation temperature 'T_0'");
    add_named_real(m, "G0", "'G_0'");
    add_named_real(m, "M_EXPO", "'m'");
    add_named_int(m, "BETANUM", "number of 'beta' in list");
    add_named_real_vector(m, "BETA", "beta", "BETANUM");
    add_named_real(m, "H_FACT", "'H'");
    add_named_int(m, "THERMOMAT", "mat id of thermal material part", -1, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Elasto-plastic material with damage, based on MAT_Struct_PlasticLinElast
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_Damage",
        "elasto-plastic von Mises material with ductile damage", Core::Materials::m_elpldamage);

    add_named_real(m, "YOUNG", "Young's modulus");
    add_named_real(m, "NUE", "Poisson's ratio");
    add_named_real(m, "DENS", "mass density");
    add_named_int(m, "SAMPLENUM", "number of stress-strain pairs in list");
    add_named_real_vector(m, "SIGMA_Y", "yield stress", "SAMPLENUM");
    add_named_real_vector(
        m, "EPSBAR_P", "accumulated plastic strain corresponding to SIGMA_Y", "SAMPLENUM");
    add_named_real(m, "DAMDEN", "denominator of damage evoluation law");
    add_named_real(m, "DAMEXP", "exponent of damage evoluation law");
    add_named_real(m, "DAMTHRESHOLD", "damage threshold");
    add_named_real(m, "KINHARD", "kinematic hardening modulus, stress-like variable");
    add_named_real(m, "KINHARD_REC", "recovery factor, scalar-valued variable");
    add_named_real(m, "SATHARDENING", "saturation hardening");
    add_named_real(m, "HARDEXPO", "hardening exponent");
    add_named_real(m, "TOL", "tolerance for local Newton iteration");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // aneurysm wall material according to Raghavan and Vorp [2000]
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_AAANeoHooke",
        "aneurysm wall material according to Raghavan and Vorp [2000]",
        Core::Materials::m_aaaneohooke);

    add_named_real(m, "YOUNG", "Young's modulus");
    add_named_real(m, "BETA", "2nd parameter");
    add_named_real(m, "NUE", "Poisson's ratio");
    add_named_real(m, "DENS", "mass density");

    Mat::append_material_definition(matlist, m);
  }


  /*----------------------------------------------------------------------*/
  // Visco-elastic Neo-Hookean material law
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_VISCONEOHOOKE",
        "visco-elastic neo-Hookean material law", Core::Materials::m_visconeohooke);
    add_named_real(m, "YOUNGS_SLOW", "???");
    add_named_real(m, "POISSON", "???");
    add_named_real(m, "DENS", "???");
    add_named_real(m, "YOUNGS_FAST", "???");
    add_named_real(m, "RELAX", "???");
    add_named_real(m, "THETA", "???");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Visco-elastic anisotropic fiber material law
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_VISCOANISO",
        "visco-elastic anisotropic fibre material law", Core::Materials::m_viscoanisotropic);

    add_named_real(m, "KAPPA", "dilatation modulus");
    add_named_real(m, "MUE", "Shear Modulus");
    add_named_real(m, "DENS", "Density");
    add_named_real(m, "K1", "Parameter for linear fiber stiffness");
    add_named_real(m, "K2", "Parameter for exponetial fiber stiffness");
    add_named_real(m, "GAMMA", "angle between fibers");
    add_named_real(m, "BETA_ISO", "ratio between elasticities in generalized Maxweel body");
    add_named_real(m, "BETA_ANISO", "ratio between elasticities in generalized Maxweel body");
    add_named_real(m, "RELAX_ISO", "isotropic relaxation time");
    add_named_real(m, "RELAX_ANISO", "anisotropic relaxation time");
    add_named_real(m, "MINSTRETCH", "minimal principal stretch fibers do respond to");
    add_named_int(
        m, "ELETHICKDIR", "Element thickness direction applies also to fibers (only sosh)");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Structural micro-scale approach: material parameters are calculated from microscale simulation
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Struct_Multiscale",
        "Structural micro-scale approach: material parameters are calculated from microscale "
        "simulation",
        Core::Materials::m_struct_multiscale);

    add_named_string(m, "MICROFILE", "inputfile for microstructure", "filename.dat");
    add_named_int(m, "MICRODIS_NUM", "Number of microscale discretization");
    add_named_real(m, "INITVOL", "Initial volume of RVE", 0.0, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // collection of hyperelastic materials
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_ElastHyper",
        "list/collection of hyperelastic materials, i.e. material IDs",
        Core::Materials::m_elasthyper);

    add_named_int(m, "NUMMAT", "number of materials/potentials in list");
    add_named_int_vector(m, "MATIDS", "the list material/potential IDs", "NUMMAT");
    add_named_real(m, "DENS", "material mass density");
    add_named_int(m, "POLYCONVEX", "1.0 if polyconvexity of system is checked", 0., true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // viscohyperelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_ViscoElastHyper",
        "Viscohyperelastic material compatible with the collection of hyperelastic materials",
        Core::Materials::m_viscoelasthyper);

    add_named_int(m, "NUMMAT", "number of materials/potentials in list");
    add_named_int_vector(m, "MATIDS", "the list material/potential IDs", "NUMMAT");
    add_named_real(m, "DENS", "material mass density");
    add_named_int(m, "POLYCONVEX", "1.0 if polyconvexity of system is checked", 0., true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // collection of hyperelastic materials for finite strain plasticity
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PlasticElastHyper",
        "list/collection of hyperelastic materials, i.e. material IDs",
        Core::Materials::m_plelasthyper);

    add_named_int(m, "NUMMAT", "number of materials/potentials in list");
    add_named_int_vector(m, "MATIDS", "the list material/potential IDs", "NUMMAT");
    add_named_real(m, "DENS", "material mass density");
    add_named_real(m, "INITYIELD", "initial yield stress");
    add_named_int(m, "POLYCONVEX", "1.0 if polyconvexity of system is checked", 0., true);
    add_named_real(m, "ISOHARD", "linear isotropic hardening modulus", 0., true);
    add_named_real(m, "EXPISOHARD", "nonlinear isotropic hardening exponent", 0., true);
    add_named_real(
        m, "INFYIELD", "saturation yield stress for nonlinear isotropic hardening", 0., true);
    add_named_real(m, "KINHARD", "linear kinematic hardening modulus", 0., true);

    // visco-plasticity
    add_named_real(m, "VISC", "Visco-Plasticity parameter 'eta' in Perzyna model", 0., true);
    add_named_real(
        m, "RATE_DEPENDENCY", "Visco-Plasticity parameter 'eta' in Perzyna model", 1., true);
    add_named_real(m, "VISC_SOFT",
        "Visco-Plasticity temperature dependency (eta = eta_0 * (1-(T-T_0)*x)", 0., true);

    // optional pastic spin parameter
    add_named_real(
        m, "PL_SPIN_CHI", "Plastic spin coupling parameter chi (often called eta)", 0.0, true);

    // optional Hill yield parameters
    add_named_real(m, "rY_11", "relative yield stress in fiber1-direction (Y_11/Y_0)", 0.0, true);
    add_named_real(m, "rY_22", "relative yield stress in fiber2-direction (Y_22/Y_0)", 0.0, true);
    add_named_real(m, "rY_33", "relative yield stress in fiber3-direction (Y_33/Y_0)", 0.0, true);
    add_named_real(m, "rY_12", "relative shear yield stress in 12-direction (Y_12/Y_0)", 0.0, true);
    add_named_real(m, "rY_23", "relative shear yield stress in 23-direction (Y_23/Y_0)", 0.0, true);
    add_named_real(m, "rY_13", "relative shear yield stress in 13-direction (Y_13/Y_0)", 0.0, true);

    // optional TSI parameters
    add_named_real(m, "CTE", "coefficient of thermal expansion", 0., true);
    add_named_real(m, "INITTEMP", "initial, reference temperature", 0., true);
    add_named_real(m, "YIELDSOFT", "yield stress softening", 0., true);
    add_named_real(m, "HARDSOFT", "hardening softening", 0., true);
    add_named_real(
        m, "TAYLOR_QUINNEY", "Taylor-Quinney factor for plastic heat conversion", 1., true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // collection of hyperelastic materials for finite strain plasticity
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PlasticElastHyperVCU",
        "list/collection of hyperelastic materials, i.e. material IDs",
        Core::Materials::m_plelasthyperVCU);

    add_named_int(m, "NUMMAT", "number of materials/potentials in list");
    add_named_int_vector(m, "MATIDS", "the list material/potential IDs", "NUMMAT");
    add_named_real(m, "DENS", "material mass density");
    add_named_real(m, "INITYIELD", "initial yield stress");
    add_named_real(m, "ISOHARD", "linear isotropic hardening modulus", 0., true);
    add_named_real(m, "EXPISOHARD", "nonlinear isotropic hardening exponent", 0., true);
    add_named_real(
        m, "INFYIELD", "saturation yield stress for nonlinear isotropic hardening", 0., true);
    add_named_real(m, "KINHARD", "linear kinematic hardening modulus", 0., true);

    // visco-plasticity
    add_named_real(m, "VISC", "Visco-Plasticity parameter 'eta' in Perzyna model", 0., true);
    add_named_real(
        m, "RATE_DEPENDENCY", "Visco-Plasticity parameter 'eta' in Perzyna model", 1., true);
    add_named_real(m, "VISC_SOFT",
        "Visco-Plasticity temperature dependency (eta = eta_0 * (1-(T-T_0)*x)", 0., true);

    // optional pastic spin parameter
    add_named_real(
        m, "PL_SPIN_CHI", "Plastic spin coupling parameter chi (often called eta)", 0.0, true);

    // optional Hill yield parameters
    add_named_real(m, "rY_11", "relative yield stress in fiber1-direction (Y_11/Y_0)", 0.0, true);
    add_named_real(m, "rY_22", "relative yield stress in fiber2-direction (Y_22/Y_0)", 0.0, true);
    add_named_real(m, "rY_33", "relative yield stress in fiber3-direction (Y_33/Y_0)", 0.0, true);
    add_named_real(m, "rY_12", "relative shear yield stress in 12-direction (Y_12/Y_0)", 0.0, true);
    add_named_real(m, "rY_23", "relative shear yield stress in 23-direction (Y_23/Y_0)", 0.0, true);
    add_named_real(m, "rY_13", "relative shear yield stress in 13-direction (Y_13/Y_0)", 0.0, true);

    // optional TSI parameters
    add_named_real(m, "CTE", "coefficient of thermal expansion", 0., true);
    add_named_real(m, "INITTEMP", "initial, reference temperature", 0., true);
    add_named_real(m, "YIELDSOFT", "yield stress softening", 0., true);
    add_named_real(m, "HARDSOFT", "hardening softening", 0., true);
    add_named_real(
        m, "TAYLOR_QUINNEY", "Taylor-Quinney factor for plastic heat conversion", 1., true);

    add_named_int(m, "POLYCONVEX", "1.0 if polyconvexity of system is checked", 0., true);


    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // logarithmic neo-Hooke material acc. to Bonet and Wood
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupLogNeoHooke",
        "logarithmic neo-Hooke material acc. to Bonet and Wood",
        Core::Materials::mes_couplogneohooke);

    add_named_string(m, "MODE",
        "parameter set: YN (Young's modulus and Poisson's ration; default) or Lame (mue and "
        "lambda)",
        "YN");
    add_named_real(m, "C1", "E or mue");
    add_named_real(m, "C2", "nue or lambda");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // Saint-Venant-Kirchhoff as elastic summand
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "ELAST_CoupSVK", "Saint-Venant-Kirchhoff as elastic summand", Core::Materials::mes_coupSVK);

    add_named_real(m, "YOUNG", "Young's modulus");
    add_named_real(m, "NUE", "Poisson's ratio");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // Simo-Pister type material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "ELAST_CoupSimoPister", "Simo-Pister type material", Core::Materials::mes_coupsimopister);

    add_named_real(m, "MUE", "material constant");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // logarithmic mixed neo-Hooke material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupLogMixNeoHooke",
        "mixed logarithmic neo-Hooke material", Core::Materials::mes_couplogmixneohooke);

    add_named_string(m, "MODE",
        "parameter set: YN (Young's modulus and Poisson's ration; default) or Lame (mue and "
        "lambda)",
        "YN");
    add_named_real(m, "C1", "E or mue");
    add_named_real(m, "C2", "nue or lambda");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled exponential material for compressible material (according to Weikenmeier_2014)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupExpPol",
        "compressible, isochoric exponential material law for soft tissue",
        Core::Materials::mes_coupexppol);
    add_named_real(m, "A", "material constant");
    add_named_real(m, "B", "material constant linear I_1");
    add_named_real(m, "C", "material constant linear J");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // compressible neo-Hooke material acc. to Holzapfel
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupNeoHooke",
        "compressible neo-Hooke material acc. to Holzapfel", Core::Materials::mes_coupneohooke);

    add_named_real(m, "YOUNG", "Young's modulus", 0.0, true);
    add_named_real(m, "NUE", "Poisson's ratio", 0.0, true);

    Mat::append_material_definition(matlist, m);
  }
  // Mooney Rivlin  material acc. to Holzapfel
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupMooneyRivlin",
        "Mooney - Rivlin material acc. to Holzapfel", Core::Materials::mes_coupmooneyrivlin);

    add_named_real(m, "C1", "material constant", 0.0, true);
    add_named_real(m, "C2", "material constant", 0.0, true);
    add_named_real(m, "C3", "material constant", 0.0, true);

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled Blatz and Ko material acc. to Holzapfel
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupBlatzKo",
        "Blatz and Ko material acc. to Holzapfel", Core::Materials::mes_coupblatzko);

    add_named_real(m, "MUE", "Shear modulus");
    add_named_real(m, "NUE", "Poisson's ratio");
    add_named_real(m, "F", "interpolation parameter");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of Neo-Hooke
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoNeoHooke",
        "isochoric part of neo-Hooke material acc. to Holzapfel", Core::Materials::mes_isoneohooke);

    add_named_real(m, "MUE", "Shear modulus");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of one-term Ogden material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoOgden",
        "isochoric part of the one-term Ogden material", Core::Materials::mes_isoogden);

    add_named_real(m, "MUE", "Shear modulus");
    add_named_real(m, "ALPHA", "Nonlinearity parameter");
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of Yeoh
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoYeoh",
        "isochoric part of  Yeoh material acc. to Holzapfel", Core::Materials::mes_isoyeoh);

    add_named_real(m, "C1", "Linear modulus");
    add_named_real(m, "C2", "Quadratic modulus");
    add_named_real(m, "C3", "Cubic modulus");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of iso1pow
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "ELAST_Iso1Pow", "isochoric part of general power material", Core::Materials::mes_iso1pow);

    add_named_real(m, "C", "material parameter");
    add_named_int(m, "D", "exponent");
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of iso2pow
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "ELAST_Iso2Pow", "isochoric part of general power material", Core::Materials::mes_iso2pow);

    add_named_real(m, "C", "material parameter");
    add_named_int(m, "D", "exponent");
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup1pow
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "ELAST_Coup1Pow", "part of general power material", Core::Materials::mes_coup1pow);

    add_named_real(m, "C", "material parameter");
    add_named_int(m, "D", "exponent");
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup2pow
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "ELAST_Coup2Pow", "part of general power material", Core::Materials::mes_coup2pow);

    add_named_real(m, "C", "material parameter");
    add_named_int(m, "D", "exponent");
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup3pow
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "ELAST_Coup3Pow", "part of general power material", Core::Materials::mes_coup3pow);

    add_named_real(m, "C", "material parameter");
    add_named_int(m, "D", "exponent");
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup13apow
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_Coup13aPow",
        "hyperelastic potential summand for multiplicative coupled invariants I1 and I3",
        Core::Materials::mes_coup13apow);

    add_named_real(m, "C", "material parameter");
    add_named_int(m, "D", "exponent of all");
    add_named_real(m, "A", "negative exponent of I3");
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of expo
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoExpoPow",
        "isochoric part of  exponential material acc. to Holzapfel",
        Core::Materials::mes_isoexpopow);

    add_named_real(m, "K1", "material parameter");
    add_named_real(m, "K2", "material parameter");
    add_named_int(m, "C", "exponent");
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of mooney rivlin
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoMooneyRivlin",
        "isochoric part of  Mooney-Rivlin material acc. to Holzapfel",
        Core::Materials::mes_isomooneyrivlin);

    add_named_real(m, "C1", "Linear modulus for first invariant");
    add_named_real(m, "C2", "Linear modulus for second invariant");
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric anisotropic material with one exponential fiber family
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoMuscle_Blemker",
        "anisotropic Blemker muscle material", Core::Materials::mes_isomuscleblemker);

    add_named_real(m, "G1", "muscle along fiber shear modulus");
    add_named_real(m, "G2", "muscle cross fiber shear modulus");
    add_named_real(m, "P1", "linear material parameter for passive along-fiber response");
    add_named_real(m, "P2", "exponential material parameter for passive along-fiber response");
    add_named_real(m, "SIGMAMAX", "maximal active isometric stress");
    add_named_real(m, "LAMBDAOFL", "optimal fiber stretch");
    add_named_real(
        m, "LAMBDASTAR", "stretch at which the normalized passive fiber force becomes linear");
    add_named_real(m, "ALPHA", "tetanised activation level,");
    add_named_real(m, "BETA", "constant scaling tanh-type activation function");
    add_named_real(m, "ACTSTARTTIME", "starting time of muscle activation");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // test material to test elasthyper-toolbox
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoTestMaterial",
        "test material to test elasthyper-toolbox", Core::Materials::mes_isotestmaterial);

    add_named_real(m, "C1", "Modulus for first invariant");
    add_named_real(m, "C2", "Modulus for second invariant");
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // general fiber material for remodeling
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_RemodelFiber",
        "General fiber material for remodeling", Core::Materials::mes_remodelfiber);

    add_named_int(m, "NUMMAT", "number of materials/potentials in list");
    add_named_int_vector(m, "MATIDS", "the list material/potential IDs", "NUMMAT");
    add_named_real(m, "TDECAY", "decay time of Poisson (degradation) process");
    add_named_real(m, "GROWTHFAC", "time constant for collagen growth", 0.0, true);
    add_named_real_vector(m, "COLMASSFRAC",
        "initial mass fraction of first collagen fiber family in constraint mixture", "NUMMAT", 0.0,
        true);
    add_named_real(m, "DEPOSITIONSTRETCH", "deposition stretch");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // volumetric contribution of Sussman Bathe
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_VolSussmanBathe",
        "volumetric part of  SussmanBathe material", Core::Materials::mes_volsussmanbathe);

    add_named_real(m, "KAPPA", "dilatation modulus");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // volumetric penalty contribution
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_VolPenalty",
        "Penalty formulation for the volumetric part", Core::Materials::mes_volpenalty);

    add_named_real(m, "EPSILON", "penalty parameter");
    add_named_real(m, "GAMMA", "penalty parameter");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // volumetric contribution of Ogden
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_VolOgden",
        "Ogden formulation for the volumetric part", Core::Materials::mes_vologden);

    add_named_real(m, "KAPPA", "dilatation modulus");
    add_named_real(m, "BETA", "empiric constant");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // volumetric power law contribution
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_VolPow",
        "Power law formulation for the volumetric part", Core::Materials::mes_volpow);

    add_named_real(m, "A", "prefactor of power law");
    add_named_real(m, "EXPON", "exponent of power law");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with one exponential fiber family
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupAnisoExpoActive",
        "anisotropic active fiber", Core::Materials::mes_coupanisoexpoactive);

    add_named_real(m, "K1", "linear constant");
    add_named_real(m, "K2", "exponential constant");
    add_named_real(m, "GAMMA", "angle");
    add_named_real(m, "K1COMP", "linear constant");
    add_named_real(m, "K2COMP", "exponential constant");
    add_named_int(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    add_named_int(m, "INIT", "initialization modus for fiber alignment", 1, true);
    add_named_bool(m, "ADAPT_ANGLE", "adapt angle during remodeling", false, true);
    add_named_real(m, "S", "maximum contractile stress");
    add_named_real(m, "LAMBDAMAX", "stretch at maximum active force generation");
    add_named_real(m, "LAMBDA0", "stretch at zero active force generation");
    add_named_real(m, "DENS", "total reference mass density of constrained mixture");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with one exponential fiber family
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupAnisoExpo",
        "anisotropic part with one exp. fiber", Core::Materials::mes_coupanisoexpo);

    add_named_real(m, "K1", "linear constant");
    add_named_real(m, "K2", "exponential constant");
    add_named_real(m, "GAMMA", "angle");
    add_named_real(m, "K1COMP", "linear constant");
    add_named_real(m, "K2COMP", "exponential constant");
    add_named_int(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    add_named_int(m, "INIT", "initialization modus for fiber alignment", 1, true);
    add_named_bool(m, "ADAPT_ANGLE", "adapt angle during remodeling", false, true);
    add_named_int(
        m, "FIBER_ID", "Id of the fiber to be used (1 for first fiber, default)", 1, true);

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with one exponential shear behavior between two fibers
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupAnisoExpoShear",
        "Exponential shear behavior between two fibers", Core::Materials::mes_coupanisoexposhear);

    add_named_real(m, "K1", "linear constant");
    add_named_real(m, "K2", "exponential constant");
    add_named_real(m, "GAMMA", "angle");
    add_named_real(m, "K1COMP", "linear constant");
    add_named_real(m, "K2COMP", "exponential constant");
    add_named_int(m, "INIT", "initialization modus for fiber alignment", 1, true);
    add_named_int_vector(m, "FIBER_IDS",
        "Ids of the two fibers to be used (1 for the first fiber, 2 for the second, default)", 2);

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with one pow-like fiber family
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupAnisoPow",
        "anisotropic part with one pow-like fiber", Core::Materials::mes_coupanisopow);

    add_named_real(m, "K", "linear constant");
    add_named_real(m, "D1", "exponential constant for fiber invariant");
    add_named_real(m, "D2", "exponential constant for system");
    add_named_real(m, "ACTIVETHRES",
        "Deformation threshold for activating fibers. Default:"
        " 1.0 (off at compression); If 0.0 (always active)",
        1.0, true);
    add_named_int(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    add_named_int(m, "FIBER", "Number of the fiber family contained in the element", 1, true);
    add_named_real(m, "GAMMA", "angle", 0.0, true);
    add_named_int(m, "INIT", "initialization modus for fiber alignment", 1, true);
    add_named_bool(m, "ADAPT_ANGLE", "adapt angle during remodeling", false, true);

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with two exponential fiber families
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupAnisoExpoTwoCoup",
        "anisotropic part with two exp. fibers", Core::Materials::mes_coupanisoexpotwocoup);

    add_named_real(m, "A4", "linear anisotropic constant for fiber 1");
    add_named_real(m, "B4", "exponential anisotropic constant for fiber 1");
    add_named_real(m, "A6", "linear anisotropic constant for fiber 2");
    add_named_real(m, "B6", "exponential anisotropic constant for fiber 2");
    add_named_real(m, "A8", "linear anisotropic constant for fiber 1 relating fiber 2");
    add_named_real(m, "B8", "exponential anisotropic constant for fiber 1 relating fiber 2");
    add_named_real(m, "GAMMA", "angle");
    add_named_int(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    add_named_int(m, "INIT", "initialization modus for fiber alignment", 1, true);
    add_named_bool(
        m, "FIB_COMP", "fibers support compression: yes (true) or no (false)", true, true);
    add_named_bool(m, "ADAPT_ANGLE", "adapt angle during remodeling", false, true);

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with two exponential fiber families
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupAnisoNeoHooke",
        "anisotropic part with one neo Hookean fiber", Core::Materials::mes_coupanisoneohooke);

    add_named_real(m, "C", "linear constant");
    add_named_real(m, "GAMMA", "angle");
    add_named_int(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    add_named_int(m, "INIT", "initialization modus for fiber alignment", 1, true);
    add_named_bool(m, "ADAPT_ANGLE", "adapt angle during remodeling", false, true);

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

    add_named_real(m, "SIGMA", "Contractility (maximal stress)");
    add_named_real(m, "TAUC0", "Initial value for the active stress");
    add_named_real(m, "MAX_ACTIVATION", "Maximal value for the rescaled activation");
    add_named_real(m, "MIN_ACTIVATION", "Minimal value for the rescaled activation");
    add_named_int(
        m, "SOURCE_ACTIVATION", "Where the activation comes from: 0=scatra , >0 Id for FUNCT");
    add_named_real(m, "ACTIVATION_THRES",
        "Threshold for activation (contraction starts when activation function is larger than this "
        "value, relaxes otherwise)");
    add_named_bool(m, "STRAIN_DEPENDENCY",
        "model strain dependency of contractility (Frank-Starling law): no (false) or yes (true)",
        false, true);
    add_named_real(m, "LAMBDA_LOWER", "lower fiber stretch for Frank-Starling law", 1.0, true);
    add_named_real(m, "LAMBDA_UPPER", "upper fiber stretch for Frank-Starling law", 1.0, true);
    add_named_real(m, "GAMMA", "angle", 0.0, true);
    add_named_int(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    add_named_int(m, "INIT", "initialization mode for fiber alignment", 1, true);
    add_named_bool(m, "ADAPT_ANGLE", "adapt angle during remodeling", false, true);

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with variable stress coefficient
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupAnisoNeoHooke_VarProp",
        "anisotropic part with one neo Hookean fiber with variable coefficient",
        Core::Materials::mes_coupanisoneohooke_varprop);

    add_named_real(m, "C", "linear constant");
    add_named_int(
        m, "SOURCE_ACTIVATION", "Where the activation comes from: 0=scatra , >0 Id for FUNCT");
    add_named_real(m, "GAMMA", "azimuth angle", 0.0, true);
    add_named_real(m, "THETA", "polar angle", 0.0, true);
    add_named_int(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    add_named_int(m, "INIT", "initialization mode for fiber alignment", 1, true);
    add_named_bool(m, "ADAPT_ANGLE", "adapt angle during remodeling", false, true);

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric anisotropic material with one exponential fiber family
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoAnisoExpo",
        "anisotropic part with one exp. fiber", Core::Materials::mes_isoanisoexpo);

    add_named_real(m, "K1", "linear constant");
    add_named_real(m, "K2", "exponential constant");
    add_named_real(m, "GAMMA", "angle");
    add_named_real(m, "K1COMP", "linear constant");
    add_named_real(m, "K2COMP", "exponential constant");
    add_named_int(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    add_named_int(m, "INIT", "initialization modus for fiber alignment", 1, true);
    add_named_bool(m, "ADAPT_ANGLE", "adapt angle during remodeling", false, true);

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // structural tensor
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_StructuralTensor",
        "Parameter for structural tensor strategy in anisotropic materials",
        Core::Materials::mes_structuraltensorstratgy);

    add_named_string(m, "STRATEGY",
        "Strategy for evaluation of structural tensor: "
        "Standard (default), ByDistributionFunction, DispersedTransverselyIsotropic",
        "Standard");

    // choose between:
    // "none"
    // "Bingham"
    // "vonMisesFisher"
    //  rauch 10/17
    add_named_string(m, "DISTR",
        "Type of distribution function around mean direction: "
        "none, Bingham, vonMisesFisher",
        "none", true);

    add_named_real(m, "C1", "constant 1 for distribution function", 1.0, true);
    add_named_real(m, "C2", "constant 2 for distribution function", 0.0, true);
    add_named_real(m, "C3", "constant 3 for distribution function", 0.0, true);
    add_named_real(m, "C4", "constant 4 for distribution function", 1e16, true);

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // transversely isotropic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_CoupTransverselyIsotropic",
        "transversely part of a simple othotropic, transversely "
        "isotropic hyperelastic constitutive equation",
        Core::Materials::mes_couptransverselyisotropic);

    add_named_real(m, "ALPHA", "1-st constant");
    add_named_real(m, "BETA", "2-nd constant");
    add_named_real(m, "GAMMA", "3-rd constant");
    add_named_real(m, "ANGLE", "fiber angle");
    add_named_int(m, "STR_TENS_ID", "MAT ID for definition of Structural Tensor");
    add_named_int(m, "FIBER", "exponential constant", 1, true);
    add_named_int(m, "INIT", "initialization modus for fiber alignment", 1, true);

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // coupled Varga material acc. to Holzapfel
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "ELAST_CoupVarga", "Varga material acc. to Holzapfel", Core::Materials::mes_coupvarga);

    add_named_real(m, "MUE", "Shear modulus");
    add_named_real(m, "BETA", "'Anti-modulus'");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric Varga material acc. to Holzapfel
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("ELAST_IsoVarga",
        "Isochoric Varga material acc. to Holzapfel", Core::Materials::mes_isovarga);

    add_named_real(m, "MUE", "Shear modulus");
    add_named_real(m, "BETA", "'Anti-modulus'");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isotropic viscous contribution of myocardial matrix (chapelle12)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("VISCO_CoupMyocard",
        "Isotropic viscous contribution of myocardial matrix", Core::Materials::mes_coupmyocard);

    add_named_real(m, "N", "material parameter");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // isochoric rate dependent viscos material, modified from Pioletti,1997
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("VISCO_IsoRateDep",
        "Isochoric rate dependent viscous material", Core::Materials::mes_isoratedep);

    add_named_real(m, "N", "material parameter");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // viscos contribution to visohyperelastic material according to SLS-Model
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "VISCO_GenMax", "Viscous contribution according to SLS-Model", Core::Materials::mes_genmax);

    add_named_real(m, "TAU", "relaxation parameter");
    add_named_real(m, "BETA", "emphasis of viscous to elastic part");
    add_named_string(m, "SOLVE",
        "Solution of evolution equation via: OST (default) or CONVOL (convolution integral)",
        "OST");


    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // viscos contribution to visohyperelastic material according to FSLS-Model
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "VISCO_Fract", "Viscous contribution according to FSLS-Model", Core::Materials::mes_fract);

    add_named_real(m, "TAU", "relaxation parameter");
    add_named_real(m, "ALPHA", "fractional order derivative");
    add_named_real(m, "BETA", "emphasis of viscous to elastic part");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // viscous contribution of a branch of a generalized Maxwell model
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("VISCO_PART",
        "Viscous contribution of a viscoelastic Branch", Core::Materials::mes_viscopart);

    add_named_real(m, "TAU", "dynamic viscosity divided by young's modulus of the branch");

    Mat::append_material_definition(matlist, m);
  }
  /*--------------------------------------------------------------------*/
  // viscoelatic branches of a generalized Maxwell model
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("VISCO_GeneralizedGenMax",
        "Viscoelastic Branches of generalized Maxwell", Core::Materials::mes_generalizedgenmax);

    add_named_int(m, "NUMBRANCH", "number of viscoelastic branches");
    add_named_int_vector(m, "MATIDS", "the list material IDs", "NUMBRANCH");
    add_named_string(m, "SOLVE",
        "Solution for evolution equation: OST (default) or CONVOL (convolution integral)",
        "CONVOL");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // description of a viscoelatic branch of a generalized Maxwell model
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("VISCO_BRANCH",
        "Viscoelastic Branch (viscous and elastic contribution)", Core::Materials::mes_viscobranch);

    add_named_int(m, "NUMMAT", "number of materials in the viscoelastic branch");
    add_named_int_vector(m, "MATIDS", "the list material IDs", "NUMMAT");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // 1D Artery material with constant properties
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_CNST_ART", "artery with constant properties", Core::Materials::m_cnst_art);

    add_named_real(m, "VISCOSITY",
        "viscosity (for CONSTANT viscosity law taken as blood viscosity, for BLOOD viscosity law "
        "taken as the viscosity of blood plasma)");
    add_named_real(m, "DENS", "density of blood");
    add_named_real(m, "YOUNG", "artery Youngs modulus of elasticity");
    add_named_real(m, "NUE", "Poissons ratio of artery fiber");
    add_named_real(m, "TH", "artery thickness");
    add_named_real(m, "PEXT1", "artery fixed external pressure 1");
    add_named_real(m, "PEXT2", "artery fixed external pressure 2");
    add_named_string(
        m, "VISCOSITYLAW", "type of viscosity law, CONSTANT (default) or BLOOD", "CONSTANT", true);
    add_named_real(m, "BLOOD_VISC_SCALE_DIAM_TO_MICRONS",
        "used to scale the diameter for blood viscosity law to microns if your problem is not "
        "given in microns, e.g., if you use mms, set this parameter to 1.0e3",
        1.0, true);
    add_named_string(m, "VARYING_DIAMETERLAW",
        "type of varying diameter law, CONSTANT (default) or BY_FUNCTION", "CONSTANT", true);
    add_named_int(m, "VARYING_DIAMETER_FUNCTION", "function for varying diameter law", -1, true);
    add_named_real(m, "COLLAPSE_THRESHOLD",
        "Collapse threshold for diameter (below this diameter element is assumed to be collapsed "
        "with zero diameter and is not evaluated)",
        -1.0, true);

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // Fourier's law
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("THERM_FourierIso",
        "isotropic (linear) Fourier's law of heat conduction", Core::Materials::m_th_fourier_iso);

    add_named_real(m, "CAPA", "volumetric heat capacity");
    add_named_real(m, "CONDUCT", "thermal conductivity");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // material for heat transport due to Fourier-type thermal conduction and the Soret effect (fang
  // 06/15)
  {
    auto matsoret = std::make_shared<Mat::MaterialDefinition>("MAT_soret",
        "material for heat transport due to Fourier-type thermal conduction and the Soret effect",
        Core::Materials::m_soret);

    // mandatory parameters
    add_named_real(matsoret, "CAPA", "volumetric heat capacity");
    add_named_real(matsoret, "CONDUCT", "thermal conductivity");
    add_named_real(matsoret, "SORET", "Soret coefficient");

    // add Soret material to global list of valid materials
    Mat::append_material_definition(matlist, matsoret);
  }

  /*----------------------------------------------------------------------*/
  // collection of hyperelastic materials for membranes
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Membrane_ElastHyper",
        "list/collection of hyperelastic materials for membranes, i.e. material IDs",
        Core::Materials::m_membrane_elasthyper);

    add_named_int(m, "NUMMAT", "number of materials/potentials in list");
    add_named_int_vector(m, "MATIDS", "the list material/potential IDs", "NUMMAT");
    add_named_real(m, "DENS", "material mass density");
    add_named_int(m, "POLYCONVEX", "1.0 if polyconvexity of system is checked", 0., true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // active strain membrane material for gastric electromechanics
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_Membrane_ActiveStrain",
        "active strain membrane material", Core::Materials::m_membrane_activestrain);

    add_named_int(m, "MATIDPASSIVE", "MATID for the passive material", false);
    add_named_int(m, "SCALIDVOLTAGE", "ID of the scalar that represents the (SMC) voltage", false);
    add_named_real(m, "DENS", "material mass density", false);
    add_named_real(m, "BETA1", "Ca2+ dynamics", false);
    add_named_real(m, "BETA2", "opening dynamics of the VDCC", false);
    add_named_real(m, "VOLTHRESH", "voltage threshold for activation", false);
    add_named_real(m, "ALPHA1", "intensity of contraction in fiber direction 1", false);
    add_named_real(m, "ALPHA2", "intensity of contraction in fiber direction 2", false);
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // growth and remodeling (homogenized constrained mixture model)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_GrowthRemodel_ElastHyper",
        "growth and remodeling", Core::Materials::m_growthremodel_elasthyper);

    add_named_int(m, "NUMMATRF", "number of remodelfiber materials in list", false);
    add_named_int(
        m, "NUMMATEL3D", "number of 3d elastin matrix materials/potentials in list", 0, true);
    add_named_int(
        m, "NUMMATEL2D", "number of 2d elastin matrix materials/potentials in list", false);
    add_named_int_vector(m, "MATIDSRF", "the list remodelfiber material IDs", "NUMMATRF", false);
    add_named_int_vector(m, "MATIDSEL3D", "the list 3d elastin matrix material/potential IDs",
        "NUMMATEL3D", -1, true);
    add_named_int_vector(
        m, "MATIDSEL2D", "the list 2d elastin matrix material/potential IDs", "NUMMATEL2D", false);
    add_named_int(m, "MATIDELPENALTY", "penalty material ID", -1, true);
    add_named_real(
        m, "ELMASSFRAC", "initial mass fraction of elastin matrix in constraint mixture", false);
    add_named_real(m, "DENS", "material mass density", false);
    add_named_real(
        m, "PRESTRETCHELASTINCIR", "circumferential prestretch of elastin matrix", false);
    add_named_real(m, "PRESTRETCHELASTINAX", "axial prestretch of elastin matrix", false);
    add_named_real(m, "THICKNESS",
        "reference wall thickness of the idealized cylindrical aneurysm [m]", -1, true);
    add_named_real(m, "MEANPRESSURE", "mean blood pressure [Pa]", -1.0, true);
    add_named_real(
        m, "RADIUS", "inner radius of the idealized cylindrical aneurysm [m]", -1.0, true);
    add_named_int(m, "DAMAGE", "1: elastin damage after prestressing,0: no elastin damage", false);
    add_named_int(m, "GROWTHTYPE",
        "flag to decide what type of collagen growth is used: 1: anisotropic growth; 0: isotropic "
        "growth",
        false);
    add_named_int(m, "LOCTIMEINT",
        "flag to decide what type of local time integration scheme is used: 1: Backward Euler "
        "Method; 0: Forward Euler Method",
        false);
    add_named_int(m, "MEMBRANE",
        "Flag whether Hex or Membrane elements are used ( Membrane: 1, Hex: Everything else )", -1,
        true);
    add_named_int(m, "CYLINDER",
        "Flag that geometry is a cylinder. 1: aligned in x-direction; 2: y-direction; 3: "
        "z-direction",
        -1, true);
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // multiplicative split of deformation gradient in elastic and inelastic parts
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_MultiplicativeSplitDefgradElastHyper",
        "multiplicative split of deformation gradient",
        Core::Materials::m_multiplicative_split_defgrad_elasthyper);

    add_named_int(m, "NUMMATEL", "number of elastic materials/potentials in list", 0, false);
    add_named_int_vector(
        m, "MATIDSEL", "the list of elastic material/potential IDs", "NUMMATEL", -1, false);
    add_named_int(m, "NUMFACINEL", "number of factors of inelastic deformation gradient", false);
    add_named_int_vector(m, "INELDEFGRADFACIDS",
        "the list of inelastic deformation gradient factor IDs", "NUMFACINEL", false);
    add_named_real(m, "DENS", "material mass density", false);
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

    add_named_int(m, "SCALAR1", "number of growth inducing scalar");
    add_named_real(m, "SCALAR1_MolarGrowthFac", "isotropic molar growth factor due to scalar 1");
    add_named_real(m, "SCALAR1_RefConc", "reference concentration of scalar 1 causing no strains");

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

    add_named_int(m, "SCALAR1", "number of growth inducing scalar");
    add_named_real(m, "SCALAR1_MolarGrowthFac", "anisotropic molar growth factor due to scalar 1");
    add_named_real(m, "SCALAR1_RefConc", "reference concentration of scalar 1 causing no strains");
    add_named_int(m, "NUMSPACEDIM", "Number of space dimension (only 3 valid)");
    add_named_real_vector(
        m, "GrowthDirection", "vector that defines the growth direction", "NUMSPACEDIM");

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

    add_named_int(m, "SCALAR1", "number of growth inducing scalar");
    add_named_real(m, "SCALAR1_RefConc", "reference concentration of scalar 1 causing no strains");
    add_named_int(m, "POLY_PARA_NUM", "number of polynomial coefficients");
    add_named_real_vector(m, "POLY_PARAMS", "coefficients of polynomial", "POLY_PARA_NUM");
    add_named_real(m, "X_min", "lower bound of validity of polynomial");
    add_named_real(m, "X_max", "upper bound of validity of polynomial");
    add_named_int(m, "MATID", "material ID of the corresponding scatra material");

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

    add_named_int(m, "SCALAR1", "number of growth inducing scalar");
    add_named_real(m, "SCALAR1_RefConc", "reference concentration of scalar 1 causing no strains");
    add_named_int(m, "NUMSPACEDIM", "Number of space dimension (only 3 valid)");
    add_named_real_vector(
        m, "GrowthDirection", "vector that defines the growth direction", "NUMSPACEDIM");
    add_named_int(m, "POLY_PARA_NUM", "number of polynomial coefficients");
    add_named_real_vector(m, "POLY_PARAMS", "coefficients of polynomial", "POLY_PARA_NUM");
    add_named_real(m, "X_min", "lower bound of validity of polynomial");
    add_named_real(m, "X_max", "upper bound of validity of polynomial");
    add_named_int(m, "MATID", "material ID of the corresponding scatra material");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_InelasticDefgradLinTempIso",
        "Temperature dependent growth law. Volume change linearly dependent on temperature",
        Core::Materials::mfi_lin_temp_iso);

    add_named_real(m, "Temp_GrowthFac", "isotropic growth factor due to temperature");
    add_named_real(m, "RefTemp", "reference temperature causing no strains");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_InelasticDefgradTimeFunct",
        "Time-dependent growth law. determinant of volume change dependent on time function "
        "defined "
        "by 'FUNCT_NUM",
        Core::Materials::mfi_time_funct);

    add_named_int(m, "FUNCT_NUM",
        "Time-dependent function of the determinant of the inelastic deformation gradient");

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

    add_named_int(m, "VISCOPLAST_LAW_ID", "MAT ID of the corresponding viscoplastic law");
    add_named_int(m, "FIBER_READER_ID",
        "MAT ID of the used fiber direction reader for transversely isotropic behavior");
    add_named_real(m, "YIELD_COND_A",
        "transversely isotropic version of the Hill(1948) yield condition: parameter A, following "
        "the notation in Dafalias 1989, International Journal of Plasticity, Vol. 5");
    add_named_real(m, "YIELD_COND_B",
        "transversely isotropic version of the Hill(1948) yield condition: parameter B, following "
        "the notation in Dafalias 1989, International Journal of Plasticity, Vol. 5");
    add_named_real(m, "YIELD_COND_F",
        "transversely isotropic version of the Hill(1948) yield condition: parameter F, following "
        "the notation in Dafalias 1989, International Journal of Plasticity, Vol. 5");
    add_named_string(m, "ANISOTROPY",
        "Anisotropy type: transversely isotropic (transvisotrop; transverseisotropic; "
        "transverselyisotropic) | isotropic (isotrop; isotropic; Default)",
        "isotrop");
    add_named_bool(m, "LOG_SUBSTEP",
        "boolean: time integration of internal variables using logarithmic substepping (True) or "
        "standard substepping (False)?");
    add_named_int(m, "MAX_HALVE_NUM_SUBSTEP",
        "maximum number of times the global time step can be halved in the substepping procedure");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_ViscoplasticLawReformulatedJohnsonCook",
        "Reformulation of the Johnson-Cook viscoplastic law (comprising flow rule and hardening "
        "law), as shown in Mareau et al. (Mechanics of Materials 143, 2020)",
        Core::Materials::mvl_reformulated_Johnson_Cook);

    add_named_real(m, "STRAIN_RATE_PREFAC", "plastic strain rate prefactor $\\dot{P}_0$");
    add_named_real(m, "STRAIN_RATE_EXP_FAC", "exponential factor of plastic strain rate $C$");
    add_named_real(m, "INIT_YIELD_STRENGTH", "initial yield strength of the material $A_0$");
    add_named_real(m, "ISOTROP_HARDEN_PREFAC", "prefactor of the isotropic hardening stress $B_0$");
    add_named_real(m, "ISOTROP_HARDEN_EXP", "exponent of the isotropic hardening stress $n$");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // integration point based and scalar dependent interpolation between to materials
  {
    auto mm = std::make_shared<Mat::MaterialDefinition>("MAT_ScDepInterp",
        "integration point based and scalar dependent interpolation between to materials",
        Core::Materials::m_sc_dep_interp);

    add_named_int(mm, "IDMATZEROSC", "material for lambda equal to zero");
    add_named_int(mm, "IDMATUNITSC", "material for lambda equal to one");
    //      add_named_real(mm,"ALPHA","size of ",-1.0,true);

    Mat::append_material_definition(matlist, mm);
  }

  /*----------------------------------------------------------------------*/
  // growth and remodeling of arteries
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_ConstraintMixture",
        "growth and remodeling of arteries", Core::Materials::m_constraintmixture);

    add_named_real(m, "DENS", "Density");
    add_named_real(m, "MUE", "Shear Modulus");
    add_named_real(m, "NUE", "Poisson's ratio");
    add_named_real(m, "PHIE", "mass fraction of elastin");
    add_named_real(m, "PREELA", "prestretch of elastin");
    add_named_real(m, "K1", "Parameter for linear collagen fiber stiffness");
    add_named_real(m, "K2", "Parameter for exponential collagen fiber stiffness");
    add_named_int(m, "NUMHOM", "Number of homeostatic parameters", 1);
    add_named_real_vector(m, "PRECOLL", "prestretch of collagen fibers", "NUMHOM");
    add_named_real(m, "DAMAGE", "damage stretch of collagen fibers");
    add_named_real(m, "K1M", "Parameter for linear smooth muscle fiber stiffness");
    add_named_real(m, "K2M", "Parameter for exponential smooth muscle fiber stiffness");
    add_named_real(m, "PHIM", "mass fraction of smooth muscle");
    add_named_real(m, "PREMUS", "prestretch of smooth muscle fibers");
    add_named_real(m, "SMAX", "maximal active stress");
    add_named_real(m, "KAPPA", "dilatation modulus");
    add_named_real(m, "LIFETIME", "lifetime of collagen fibers");
    add_named_real(m, "GROWTHFAC", "growth factor for stress");
    add_named_real_vector(
        m, "HOMSTR", "homeostatic target value of scalar stress measure", "NUMHOM");
    add_named_real(m, "SHEARGROWTHFAC", "growth factor for shear");
    add_named_real(m, "HOMRAD", "homeostatic target value of inner radius");
    add_named_real(m, "STARTTIME", "at this time turnover of collagen starts");
    add_named_string(m, "INTEGRATION",
        "time integration scheme: "
        "Explicit (default), or Implicit",
        "Explicit");
    add_named_real(m, "TOL", "tolerance for local Newton iteration, only for implicit integration");
    add_named_string(m, "GROWTHFORCE",
        "driving force of growth: "
        "Single (default), All, ElaCol",
        "Single");
    add_named_string(m, "ELASTINDEGRAD",
        "how elastin is degraded: "
        "None (default), Rectangle, Time",
        "None");
    add_named_string(m, "MASSPROD",
        "how mass depends on driving force: "
        "Lin (default), CosCos",
        "Lin");
    add_named_string(m, "INITSTRETCH",
        "how to set stretches in the beginning (None, Homeo, UpdatePrestretch)", "None");
    add_named_int(m, "CURVE", "number of timecurve for increase of prestretch in time", 0);
    add_named_string(m, "DEGOPTION",
        "Type of degradation function: "
        "Lin (default), Cos, Exp, ExpVar",
        "Lin");
    add_named_real(m, "MAXMASSPRODFAC", "maximal factor of mass production");
    add_named_real(m, "ELASTINFAC", "factor for elastin content", 0.0, true);
    add_named_bool(m, "STOREHISTORY",
        "store all history variables, not recommended for forward simulations", false, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // hyperelastic material for poroelasticity
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_StructPoro",
        "wrapper for structure poroelastic material", Core::Materials::m_structporo);

    add_named_int(m, "MATID", "ID of structure material");
    add_named_int(m, "POROLAWID", "ID of porosity law");
    add_named_real(m, "INITPOROSITY", "initial porosity of porous medium");

    Mat::append_material_definition(matlist, m);
  }
  /*----------------------------------------------------------------------*/
  // linear law for porosity in porous media problems
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PoroLawLinear",
        "linear constitutive law for porosity", Core::Materials::m_poro_law_linear);

    add_named_real(m, "BULKMODULUS", "bulk modulus of porous medium");

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

    add_named_real(m, "BULKMODULUS", "bulk modulus of porous medium");
    add_named_real(m, "PENALTYPARAMETER", "penalty paramter of porous medium");

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

    add_named_real(m, "INVBIOTMODULUS", "inverse Biot modulus of porous medium");
    add_named_real(m, "BIOTCEOFF", "Biot coefficient of porous medium");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // incompressible skeleton law for porosity depending on the density
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PoroLawDensityDependent",
        "porosity depending on the density", Core::Materials::m_poro_law_density_dependent);

    add_named_int(m, "DENSITYLAWID", "material ID of density law");

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

    add_named_real(m, "BULKMODULUS", "bulk modulus of porous medium");
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // permeability law for constant permeability in porous multiphase medium
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroRelPermeabilityLawConstant",
        "permeability law for constant permeability in porous multiphase medium",
        Core::Materials::m_fluidporo_relpermeabilitylaw_constant);

    add_named_real(m, "VALUE", "constant value of permeability");
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // permeability law for permeability depending on saturation according to (saturation)^exp
  // in porous multiphase medium
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroRelPermeabilityLawExp",
        "permeability law depending on saturation in porous multiphase medium",
        Core::Materials::m_fluidporo_relpermeabilitylaw_exp);

    add_named_real(m, "EXP", "exponent of the saturation of this phase");
    add_named_real(m, "MIN_SAT", "minimum saturation which is used for calculation");
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // viscosity law for constant viscosity in porous multiphase medium
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroViscosityLawConstant",
        "viscosity law for constant viscosity in porous multiphase medium",
        Core::Materials::m_fluidporo_viscositylaw_constant);

    add_named_real(m, "VALUE", "constant value of viscosity");
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // viscosity law for viscosity-dependency modelling cell adherence
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroViscosityLawCellAdherence",
        "visosity law depending on pressure gradient in porous multiphase medium",
        Core::Materials::m_fluidporo_viscositylaw_celladh);

    add_named_real(m, "VISC_0", "Visc0 parameter for modelling cell adherence");
    add_named_real(m, "XI", "xi parameter for modelling cell adherence");
    add_named_real(m, "PSI", "psi parameter for modelling cell adherence");
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // hyperelastic material for poroelasticity with reaction
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_StructPoroReaction",
        "wrapper for structure porelastic material with reaction",
        Core::Materials::m_structpororeaction);

    add_named_int(m, "MATID", "ID of structure material");
    add_named_int(m, "POROLAWID", "ID of porosity law");
    add_named_real(m, "INITPOROSITY", "initial porosity of porous medium");
    add_named_int(m, "DOFIDREACSCALAR",
        "Id of DOF within scalar transport problem, which controls the reaction");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // hyperelastic material for poroelasticity with reaction
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_StructPoroReactionECM",
        "wrapper for structure porelastic material with reaction",
        Core::Materials::m_structpororeactionECM);

    add_named_int(m, "MATID", "ID of structure material");
    add_named_int(m, "POROLAWID", "ID of porosity law");
    add_named_real(m, "INITPOROSITY", "initial porosity of porous medium");
    add_named_real(m, "DENSCOLLAGEN", "density of collagen");
    add_named_int(m, "DOFIDREACSCALAR",
        "Id of DOF within scalar transport problem, which controls the reaction");
    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // fluid flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_FluidPoro", "flow in deformable porous media", Core::Materials::m_fluidporo);

    add_named_real(m, "DYNVISCOSITY", "dynamic viscosity");
    add_named_real(m, "DENSITY", "density");
    add_named_real(m, "PERMEABILITY", "permeability of medium", 0.0, true);
    add_named_real(m, "AXIALPERMEABILITY", "axial permeability for transverse isotropy", 0.0, true);
    add_named_real(m, "ORTHOPERMEABILITY1", "first permeability for orthotropy", 0.0, true);
    add_named_real(m, "ORTHOPERMEABILITY2", "second permeability for orthotropy", 0.0, true);
    add_named_real(m, "ORTHOPERMEABILITY3", "third permeability for orthotropy", 0.0, true);
    add_named_string(m, "TYPE", "Problem type: Darcy (default) or Darcy-Brinkman", "Darcy");
    // optional parameter
    add_named_string(m, "PERMEABILITYFUNCTION",
        "Permeability function: Const(Default) or Kozeny_Carman", "Const", true);
    //  add_named_real(m,"BULKMODULUS","bulk modulus of medium");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // multiphase flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroMultiPhase",
        "multi phase flow in deformable porous media", Core::Materials::m_fluidporo_multiphase);

    add_named_bool(
        m, "LOCAL", "individual materials allocated per element or only at global scope");
    add_named_real(m, "PERMEABILITY", "permeability of medium");
    add_named_int(m, "NUMMAT", "number of materials in list");
    add_named_int_vector(m, "MATIDS", "the list material IDs", "NUMMAT");
    add_named_int(m, "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE", "number of fluid phases");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // multiphase flow in a poroelastic material with reactions
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroMultiPhaseReactions",
        "multi phase flow in deformable porous media and list of reactions",
        Core::Materials::m_fluidporo_multiphase_reactions);

    add_named_bool(
        m, "LOCAL", "individual materials allocated per element or only at global scope");
    add_named_real(m, "PERMEABILITY", "permeability of medium");
    add_named_int(m, "NUMMAT", "number of materials in list");
    add_named_int_vector(m, "MATIDS", "the list material IDs", "NUMMAT");
    add_named_int(m, "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE", "number of fluid phases");
    add_named_int(m, "NUMREAC", "number of reactions for these elements", 0);
    add_named_int_vector(m, "REACIDS", "advanced reaction list", "NUMREAC", 0);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one reaction for multiphase flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroSingleReaction",
        "advanced reaction material", Core::Materials::m_fluidporo_singlereaction);

    add_named_int(m, "NUMSCAL", "number of scalars coupled with this problem");
    add_named_int(m, "TOTALNUMDOF", "total number of multiphase-dofs");
    add_named_int(m, "NUMVOLFRAC", "number of volfracs");
    add_named_int_vector(m, "SCALE", "advanced reaction list", "TOTALNUMDOF");
    add_named_string(m, "COUPLING",
        "type of coupling: "
        "scalar_by_function, no_coupling (default)",
        "no_coupling", false);
    add_named_int(m, "FUNCTID", "function ID defining the reaction");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one phase for multiphase flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroSinglePhase",
        "one phase for multiphase flow in deformable porous media",
        Core::Materials::m_fluidporo_singlephase);

    add_named_int(m, "DENSITYLAWID", "ID of density law");
    add_named_real(m, "DENSITY", "reference/initial density");
    add_named_int(m, "RELPERMEABILITYLAWID", "ID of relative permeability law");
    add_named_int(m, "VISCOSITYLAWID", "ID of viscosity law");
    add_named_int(m, "DOFTYPEID", "ID of dof definition");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one volume fraction for multiphase flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroSingleVolFrac",
        "one phase for multiphase flow in deformable porous media",
        Core::Materials::m_fluidporo_singlevolfrac);

    add_named_real(m, "DENSITY", "reference/initial density");
    add_named_real(m, "DIFFUSIVITY", "diffusivity of phase");
    add_named_bool(
        m, "AddScalarDependentFlux", "Is there additional scalar dependent flux (yes) or (no)");
    add_named_int(m, "NUMSCAL", "Number of scalars", 0, true);
    add_named_real_vector(m, "SCALARDIFFS", "Diffusivities for additional scalar-dependent flux",
        "NUMSCAL", 0.0, true);
    add_named_real_vector(
        m, "OMEGA_HALF", "Constant for receptor kinetic law", "NUMSCAL", 1.0e13, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one volume fraction pressure for multiphase flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroVolFracPressure",
        "one volume fraction pressure for multiphase flow in deformable porous media",
        Core::Materials::m_fluidporo_volfracpressure);

    add_named_real(m, "PERMEABILITY", "permeability of phase");
    add_named_int(m, "VISCOSITYLAWID", "ID of viscosity law");
    add_named_real(m, "MIN_VOLFRAC",
        "Minimum volume fraction under which we assume that VolfracPressure is zero", 1.0e-3, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one degree of freedom for on single phase of a multiphase flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroSinglePhaseDofDiffPressure",
        "one degrree of freedom for multiphase flow in deformable porous media",
        Core::Materials::m_fluidporo_phasedof_diffpressure);

    add_named_int(m, "PHASELAWID", "ID of pressure-saturation law");
    add_named_int(m, "NUMDOF", "number of DoFs", 0);
    add_named_int_vector(m, "PRESCOEFF", "pressure IDs for differential pressure", "NUMDOF", 0);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one degree of freedom for on single phase of a multiphase flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroSinglePhaseDofPressure",
        "one degrree of freedom for multiphase flow in deformable porous media",
        Core::Materials::m_fluidporo_phasedof_pressure);

    add_named_int(m, "PHASELAWID", "ID of pressure-saturation law");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // one degree of freedom for on single phase of a multiphase flow in a poroelastic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_FluidPoroSinglePhaseDofSaturation",
        "one degrree of freedom for multiphase flow in deformable porous media",
        Core::Materials::m_fluidporo_phasedof_saturation);

    add_named_int(m, "PHASELAWID", "ID of pressure-saturation law");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // saturated law for pressure-saturation law in porous media problems
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PhaseLawLinear",
        "saturated fluid phase of porous medium", Core::Materials::m_fluidporo_phaselaw_linear);

    add_named_real(m, "RELTENSION", "relative interface tensions");
    add_named_real(m, "SATURATION_0", "saturation at zero differential pressure");
    add_named_int(m, "NUMDOF", "number of DoFs", 0);
    add_named_int_vector(m, "PRESCOEFF", "Coefficients for pressure dependence", "NUMDOF", 0);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // tangent law for pressure-saturation law in porous media multiphase problems
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_PhaseLawTangent",
        "tangent fluid phase of porous medium", Core::Materials::m_fluidporo_phaselaw_tangent);

    add_named_real(m, "RELTENSION", "relative interface tensions");
    add_named_real(m, "EXP", "exponent in pressure-saturation law");
    add_named_real(m, "SATURATION_0", "saturation at zero differential pressure");
    add_named_int(m, "NUMDOF", "number of DoFs", 0);
    add_named_int_vector(m, "PRESCOEFF", "Coefficients for pressure dependence", "NUMDOF", 0);

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

    add_named_int(m, "FUNCTPRES", "ID of function for differential pressure", 0);
    add_named_int(m, "FUNCTSAT", "ID of function for saturation", 0);
    add_named_int(m, "NUMDOF", "number of DoFs", 0);
    add_named_int_vector(m, "PRESCOEFF", "Coefficients for pressure dependence", "NUMDOF", 0);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // elastic spring
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_Struct_Spring", "elastic spring", Core::Materials::m_spring);

    add_named_real(m, "STIFFNESS", "spring constant");
    add_named_real(m, "DENS", "density");

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


    add_named_real(matdef, "YOUNG", "Young's modulus");

    /* note: we define both of the two following (redundant) parameters to be optional.
     *       upon initialization of the material, we assure that one of them is
     *       properly defined. */
    add_named_real(matdef, "SHEARMOD", "shear modulus", -1.0, true);
    add_named_real(matdef, "POISSONRATIO", "Poisson's ratio", -1.0, true);

    add_named_real(matdef, "DENS", "mass density");

    add_named_real(matdef, "CROSSAREA", "cross-section area");
    add_named_real(matdef, "SHEARCORR", "shear correction factor");

    add_named_real(matdef, "MOMINPOL", "polar/axial area moment of inertia");
    add_named_real(matdef, "MOMIN2",
        "area moment of inertia w.r.t. first principal "
        "axis of inertia (i.e. second base vector)");
    add_named_real(matdef, "MOMIN3",
        "area moment of inertia w.r.t. second principal "
        "axis of inertia (i.e. third base vector)");
    add_named_bool(matdef, "FAD", "Does automatic differentiation have to be used", false, true);


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    add_named_real(matdef, "INTERACTIONRADIUS",
        "radius of a circular cross-section which "
        "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
        -1.0, true);

    Mat::append_material_definition(matlist, matdef);
  }
  /*--------------------------------------------------------------------*/
  // material parameter definition for a Simo-Reissner type elasto-plastic beam element
  {
    auto matdef = std::make_shared<Mat::MaterialDefinition>("MAT_BeamReissnerElastPlastic",
        "material parameters for a Simo-Reissner type beam element based on "
        "hyperelastic stored energy function",
        Core::Materials::m_beam_reissner_elast_plastic);


    add_named_real(matdef, "YOUNG", "Young's modulus");

    // optional parameters for plasticity
    add_named_real(matdef, "YIELDN", "initial yield stress N", -1.0, true);
    add_named_real(matdef, "YIELDM", "initial yield stress M", -1.0, true);
    add_named_real(matdef, "ISOHARDN", "isotropic hardening modulus of forces", -1.0, true);
    add_named_real(matdef, "ISOHARDM", "isotropic hardening modulus of moments", -1.0, true);
    add_named_bool(matdef, "TORSIONPLAST",
        "defines whether torsional moment contributes to plasticity", false, true);

    /* note: we define both of the two following (redundant) parameters to be optional.
     *       upon initialization of the material, we assure that one of them is
     *       properly defined. */
    add_named_real(matdef, "SHEARMOD", "shear modulus", -1.0, true);
    add_named_real(matdef, "POISSONRATIO", "Poisson's ratio", -1.0, true);

    add_named_real(matdef, "DENS", "mass density");

    add_named_real(matdef, "CROSSAREA", "cross-section area");
    add_named_real(matdef, "SHEARCORR", "shear correction factor");

    add_named_real(matdef, "MOMINPOL", "polar/axial area moment of inertia");
    add_named_real(matdef, "MOMIN2",
        "area moment of inertia w.r.t. first principal "
        "axis of inertia (i.e. second base vector)");
    add_named_real(matdef, "MOMIN3",
        "area moment of inertia w.r.t. second principal "
        "axis of inertia (i.e. third base vector)");
    add_named_bool(matdef, "FAD", "Does automatic differentiation have to be used", false, true);


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    add_named_real(matdef, "INTERACTIONRADIUS",
        "radius of a circular cross-section which "
        "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
        -1.0, true);

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


    add_named_real(matdef, "EA", "axial rigidity");
    add_named_real(matdef, "GA2", "shear rigidity w.r.t first principal axis of inertia");
    add_named_real(matdef, "GA3", "shear rigidity w.r.t second principal axis of inertia");

    add_named_real(matdef, "GI_T", "torsional rigidity");
    add_named_real(matdef, "EI2",
        "flexural/bending rigidity w.r.t. first principal "
        "axis of inertia");
    add_named_real(matdef, "EI3",
        "flexural/bending rigidity w.r.t. second principal "
        "axis of inertia");

    add_named_real(matdef, "RhoA", "translational inertia: mass density * cross-section area");

    add_named_real(matdef, "MASSMOMINPOL",
        "polar mass moment of inertia, i.e. w.r.t. "
        "rotation around beam axis");
    add_named_real(matdef, "MASSMOMIN2",
        "mass moment of inertia w.r.t. first principal "
        "axis of inertia");
    add_named_real(matdef, "MASSMOMIN3",
        "mass moment of inertia w.r.t. second principal "
        "axis of inertia");
    add_named_bool(matdef, "FAD", "Does automatic differentiation have to be used", false, true);


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    add_named_real(matdef, "INTERACTIONRADIUS",
        "radius of a circular cross-section which "
        "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
        -1.0, true);

    Mat::append_material_definition(matlist, matdef);
  }

  /*--------------------------------------------------------------------*/
  // material parameter definition for a Kirchhoff-Love type beam element
  {
    auto matdef = std::make_shared<Mat::MaterialDefinition>("MAT_BeamKirchhoffElastHyper",
        "material parameters for a Kirchhoff-Love type beam element based on "
        "hyperelastic stored energy function",
        Core::Materials::m_beam_kirchhoff_elast_hyper);


    add_named_real(matdef, "YOUNG", "Young's modulus");

    /* note: we define both of the two following (redundant) parameters to be optional.
     *       upon initialization of the material, we assure that one of them is
     *       properly defined. */
    add_named_real(matdef, "SHEARMOD", "shear modulus", -1.0, true);
    add_named_real(matdef, "POISSONRATIO", "Poisson's ratio", -1.0, true);

    add_named_real(matdef, "DENS", "mass density");

    add_named_real(matdef, "CROSSAREA", "cross-section area");

    add_named_real(matdef, "MOMINPOL", "polar/axial area moment of inertia");
    add_named_real(matdef, "MOMIN2",
        "area moment of inertia w.r.t. first principal "
        "axis of inertia (i.e. second base vector)");
    add_named_real(matdef, "MOMIN3",
        "area moment of inertia w.r.t. second principal "
        "axis of inertia (i.e. third base vector)");
    add_named_bool(matdef, "FAD", "Does automatic differentiation have to be used", false, true);


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    add_named_real(matdef, "INTERACTIONRADIUS",
        "radius of a circular cross-section which "
        "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
        -1.0, true);

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


    add_named_real(matdef, "EA", "axial rigidity");

    add_named_real(matdef, "GI_T", "torsional rigidity");
    add_named_real(matdef, "EI2",
        "flexural/bending rigidity w.r.t. first principal "
        "axis of inertia");
    add_named_real(matdef, "EI3",
        "flexural/bending rigidity w.r.t. second principal "
        "axis of inertia");

    add_named_real(matdef, "RhoA", "translational inertia: mass density * cross-section area");

    add_named_real(matdef, "MASSMOMINPOL",
        "polar mass moment of inertia, i.e. w.r.t. "
        "rotation around beam axis");
    add_named_real(matdef, "MASSMOMIN2",
        "mass moment of inertia w.r.t. first principal "
        "axis of inertia");
    add_named_real(matdef, "MASSMOMIN3",
        "mass moment of inertia w.r.t. second principal "
        "axis of inertia");
    add_named_bool(matdef, "FAD", "Does automatic differentiation have to be used", false, true);


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    add_named_real(matdef, "INTERACTIONRADIUS",
        "radius of a circular cross-section which "
        "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
        -1.0, true);

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


    add_named_real(matdef, "YOUNG", "Young's modulus");

    add_named_real(matdef, "DENS", "mass density");

    add_named_real(matdef, "CROSSAREA", "cross-section area");

    add_named_real(matdef, "MOMIN", "area moment of inertia");
    add_named_bool(matdef, "FAD", "Does automatic differentiation have to be used", false, true);


    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    add_named_real(matdef, "INTERACTIONRADIUS",
        "radius of a circular cross-section which "
        "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
        -1.0, true);

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


    add_named_real(matdef, "EA", "axial rigidity");

    add_named_real(matdef, "EI", "flexural/bending rigidity");


    add_named_real(matdef, "RhoA", "translational inertia: mass density * cross-section area");
    add_named_bool(matdef, "FAD", "Does automatic differentiation have to be used", false, true);

    /* The following is optional because it is only required if we evaluate interactions
     * between beams such as contact, potential-based and whatever more to come.
     * For now, we always assume a circular cross-section if interactions are considered.
     *
     * This should be generalized to a type of cross-section shape (circular, rectangular,
     * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed. */
    add_named_real(matdef, "INTERACTIONRADIUS",
        "radius of a circular cross-section which "
        "is EXCLUSIVELY used to evaluate interactions such as contact, potentials, ...",
        -1.0, true);

    Mat::append_material_definition(matlist, matdef);
  }

  /*----------------------------------------------------------------------*/
  // material for an elastic Kirchhoff-Love shell
  {
    auto matdef = std::make_shared<Mat::MaterialDefinition>("MAT_Kirchhoff_Love_shell",
        "Material for an elastic Kichhhoff-Love shell ", Core::Materials::m_shell_kirchhoff_love);

    add_named_real(matdef, "YOUNG_MODULUS", "Young's modulus");
    add_named_real(matdef, "POISSON_RATIO", "Poisson's ratio");
    add_named_real(matdef, "THICKNESS", "Thickness of the shell");

    append_material_definition(matlist, matdef);
  }

  /*--------------------------------------------------------------------*/
  // material for a crosslinker in a biopolymer simulation
  {
    auto matdef = std::make_shared<Mat::MaterialDefinition>("MAT_Crosslinker",
        "material for a linkage between beams", Core::Materials::m_crosslinkermat);

    add_named_real(matdef, "MATNUM", "number of beam elasthyper material");
    add_named_string(matdef, "JOINTTYPE",
        "type of joint: "
        "beam3rline2rigid (default), beam3rline2pin or truss",
        "beam3rline2rigid");
    add_named_real(matdef, "LINKINGLENGTH", "distance between the two binding domains of a linker");
    add_named_real(matdef, "LINKINGLENGTHTOL",
        "tolerance for linker length in the sense: length +- tolerance");
    add_named_real(matdef, "LINKINGANGLE",
        "preferred binding angle enclosed by two filaments' axes in radians");
    add_named_real(matdef, "LINKINGANGLETOL",
        "tolerance for preferred binding angle in radians in the sense of: angle +- tolerance");
    add_named_real(matdef, "K_ON", "chemical association-rate");
    add_named_real(matdef, "K_OFF", "chemical dissociation-rate");

    // optional parameter
    add_named_real(
        matdef, "DELTABELLEQ", "deltaD in Bell's equation for force dependent off rate", 0.0, true);
    add_named_real(matdef, "NOBONDDISTSPHERE",
        "distance to sphere elements in which no double bonded linker is allowed", 0.0, true);
    add_named_string(matdef, "TYPE",
        "type of crosslinker: "
        "arbitrary (default), actin, collagen, integrin",
        "arbitrary", true);

    Mat::append_material_definition(matlist, matdef);
  }

  /*--------------------------------------------------------------------*/
  // 0D Acinar material base
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_0D_MAXWELL_ACINUS", "0D acinar material", Core::Materials::m_0d_maxwell_acinus);

    add_named_real(m, "Stiffness1", "first stiffness");
    add_named_real(m, "Stiffness2", "second stiffness");
    add_named_real(m, "Viscosity1", "first viscosity");
    add_named_real(m, "Viscosity2", "second viscosity");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // 0D NeoHookean Acinar material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_0D_MAXWELL_ACINUS_NEOHOOKEAN",
        "0D acinar material neohookean", Core::Materials::m_0d_maxwell_acinus_neohookean);

    add_named_real(m, "Stiffness1", "first stiffness");
    add_named_real(m, "Stiffness2", "second stiffness");
    add_named_real(m, "Viscosity1", "first viscosity");
    add_named_real(m, "Viscosity2", "second viscosity");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // 0D Exponential Acinar material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_0D_MAXWELL_ACINUS_EXPONENTIAL",
        "0D acinar material exponential", Core::Materials::m_0d_maxwell_acinus_exponential);

    add_named_real(m, "Stiffness1", "first stiffness");
    add_named_real(m, "Stiffness2", "second stiffness");
    add_named_real(m, "Viscosity1", "first viscosity");
    add_named_real(m, "Viscosity2", "second viscosity");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // 0D Exponential Acinar material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_0D_MAXWELL_ACINUS_DOUBLEEXPONENTIAL",
        "0D acinar material doubleexponential",
        Core::Materials::m_0d_maxwell_acinus_doubleexponential);

    add_named_real(m, "Stiffness1", "first stiffness");
    add_named_real(m, "Stiffness2", "second stiffness");
    add_named_real(m, "Viscosity1", "first viscosity");
    add_named_real(m, "Viscosity2", "second viscosity");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // 0D Ogden Acinar material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_0D_MAXWELL_ACINUS_OGDEN",
        "0D acinar material ogden", Core::Materials::m_0d_maxwell_acinus_ogden);

    add_named_real(m, "Stiffness1", "first stiffness");
    add_named_real(m, "Stiffness2", "second stiffness");
    add_named_real(m, "Viscosity1", "first viscosity");
    add_named_real(m, "Viscosity2", "second viscosity");

    Mat::append_material_definition(matlist, m);
  }


  /*----------------------------------------------------------------------*/
  // particle material sph fluid
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_ParticleSPHFluid",
        "particle material for SPH fluid", Core::Materials::m_particle_sph_fluid);

    add_named_real(m, "INITRADIUS", "initial radius");
    add_named_real(m, "INITDENSITY", "initial density");
    add_named_real(m, "REFDENSFAC", "reference density factor in equation of state");
    add_named_real(m, "EXPONENT", "exponent in equation of state");
    add_named_real(
        m, "BACKGROUNDPRESSURE", "background pressure for transport velocity formulation");
    add_named_real(m, "BULK_MODULUS", "bulk modulus");
    add_named_real(m, "DYNAMIC_VISCOSITY", "dynamic shear viscosity");
    add_named_real(m, "BULK_VISCOSITY", "bulk viscosity");
    add_named_real(m, "ARTIFICIAL_VISCOSITY", "artificial viscosity");
    add_named_real(m, "INITTEMPERATURE", "initial temperature", 0.0, true);
    add_named_real(m, "THERMALCAPACITY", "thermal capacity", 0.0, true);
    add_named_real(m, "THERMALCONDUCTIVITY", "thermal conductivity", 0.0, true);
    add_named_real(m, "THERMALABSORPTIVITY", "thermal absorptivity", 0.0, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // particle material sph boundary
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_ParticleSPHBoundary",
        "particle material for SPH boundary", Core::Materials::m_particle_sph_boundary);

    add_named_real(m, "INITRADIUS", "initial radius");
    add_named_real(m, "INITDENSITY", "initial density");
    add_named_real(m, "INITTEMPERATURE", "initial temperature", 0.0, true);
    add_named_real(m, "THERMALCAPACITY", "thermal capacity", 0.0, true);
    add_named_real(m, "THERMALCONDUCTIVITY", "thermal conductivity", 0.0, true);
    add_named_real(m, "THERMALABSORPTIVITY", "thermal absorptivity", 0.0, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // particle material dem
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_ParticleDEM", "particle material for DEM", Core::Materials::m_particle_dem);

    add_named_real(m, "INITRADIUS", "initial radius of particle");
    add_named_real(m, "INITDENSITY", "initial density of particle");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // particle wall material dem
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_ParticleWallDEM",
        "particle wall material for DEM", Core::Materials::m_particle_wall_dem);

    add_named_real(
        m, "FRICT_COEFF_TANG", "friction coefficient for tangential contact", -1.0, true);
    add_named_real(m, "FRICT_COEFF_ROLL", "friction coefficient for rolling contact", -1.0, true);
    add_named_real(m, "ADHESION_SURFACE_ENERGY", "adhesion surface energy", -1.0, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // electromagnetic material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_Electromagnetic", "Electromagnetic material", Core::Materials::m_electromagneticmat);

    add_named_real(m, "CONDUCTIVITY", "electrical conductivity");
    add_named_real(m, "PERMITTIVITY", "Permittivity");
    add_named_real(m, "PERMEABILITY", "Permeability");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // General mixture models (used for prestretching and for homogenized constrained mixture models)
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_Mixture", "General mixture model", Core::Materials::m_mixture);

    add_named_int(m, "NUMCONST", "number of mixture constituents");
    add_named_int(m, "MATIDMIXTURERULE", "material id of the mixturerule");
    add_named_int_vector(
        m, "MATIDSCONST", "list material IDs of the mixture constituents", "NUMCONST");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for ElastHyper toolbox
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MIX_Constituent_ElastHyper", "ElastHyper toolbox", Core::Materials::mix_elasthyper);

    add_named_int(m, "NUMMAT", "number of summands");
    add_named_int_vector(m, "MATIDS", "list material IDs of the summands", "NUMMAT");
    add_named_int(m, "PRESTRESS_STRATEGY",
        "Material id of the prestress strategy (optional, by default no prestretch)", 0, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for ElastHyper toolbox with a damage process
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Constituent_ElastHyper_Damage",
        "ElastHyper toolbox with damage", Core::Materials::mix_elasthyper_damage);

    add_named_int(m, "NUMMAT", "number of summands");
    add_named_int_vector(m, "MATIDS", "list material IDs of the membrane summands", "NUMMAT");
    add_named_int(m, "PRESTRESS_STRATEGY",
        "Material id of the prestress strategy (optional, by default no prestretch)", 0, true);
    add_named_int(m, "DAMAGE_FUNCT",
        "Reference to the function that is a gain for the increase/decrease of the reference mass "
        "density.");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for ElastHyper toolbox with a damage process and a membrane constituent
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Constituent_ElastHyper_ElastinMembrane",
        "ElastHyper toolbox with damage and 2D membrane material",
        Core::Materials::mix_elasthyper_elastin_membrane);

    add_named_int(m, "NUMMAT", "number of summands");
    add_named_int_vector(m, "MATIDS", "list material IDs of the membrane summands", "NUMMAT");
    add_named_int(m, "MEMBRANENUMMAT", "number of summands");
    add_named_int_vector(
        m, "MEMBRANEMATIDS", "list material IDs of the membrane summands", "MEMBRANENUMMAT");
    add_named_int(m, "PRESTRESS_STRATEGY",
        "Material id of the prestress strategy (optional, by default no prestretch)", 0, true);
    add_named_int(m, "DAMAGE_FUNCT",
        "Reference to the function that is a gain for the increase/decrease of the reference mass "
        "density.");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for solid material
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MIX_Constituent_SolidMaterial", "Solid material", Core::Materials::mix_solid_material);

    add_named_int(m, "MATID", "ID of the solid material");

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


    add_named_int(m, "INIT", "initialization modus for growth direction alignment", 1, true);
    add_named_int(m, "FIBER_ID",
        "Id of the fiber to point the growth direction (1 for first fiber, default)", 1, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Extension of all constituents simultaneously -> Growth happens mainly in the direction with the
  // smallest stiffness
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_GrowthStrategy_Stiffness",
        "Extension of all constituents simultaneously",
        Core::Materials::mix_growth_strategy_stiffness);

    add_named_real(
        m, "KAPPA", "Penalty parameter for the modified penalty term for incompressibility");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // General material wrapper enabling iterative prestressing
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_IterativePrestress",
        "General material wrapper enabling iterative pretressing for any material",
        Core::Materials::m_iterative_prestress);

    add_named_int(m, "MATID", "Id of the material");
    add_named_bool(m, "ACTIVE",
        "Set to True during prestressing and to false afterwards using a restart of the "
        "simulation.");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Constant predefined prestretch
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Prestress_Strategy_Constant",
        "Simple predefined prestress", Core::Materials::mix_prestress_strategy_constant);

    add_named_real_vector(m, "PRESTRETCH", "Definition of the prestretch as a 9x1 vector", 9);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Prestress strategy for a cylinder
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Prestress_Strategy_Cylinder",
        "Simple prestress strategy for a cylinder",
        Core::Materials::mix_prestress_strategy_cylinder);

    add_named_real(m, "INNER_RADIUS", "Inner radius of the cylinder");
    add_named_real(m, "WALL_THICKNESS", "Wall thickness of the cylinder");
    add_named_real(m, "AXIAL_PRESTRETCH", "Prestretch in axial direction");
    add_named_real(m, "CIRCUMFERENTIAL_PRESTRETCH", "Prestretch in circumferential direction");
    add_named_real(m, "PRESSURE", "Pressure in the inner of the cylinder");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Iterative prestress strategy for any geometry
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Prestress_Strategy_Iterative",
        "Simple iterative prestress strategy for any geometry. Needed to be used within the "
        "mixture framework.",
        Core::Materials::mix_prestress_strategy_iterative);
    add_named_bool(m, "ACTIVE", "Flag whether prestretch tensor should be updated");
    add_named_bool(m, "ISOCHORIC", "Flag whether prestretch tensor is isochoric", false, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for a full constrained mixture fiber
  {
    auto m =
        std::make_shared<Mat::MaterialDefinition>("MIX_Constituent_FullConstrainedMixtureFiber",
            "A 1D constituent that grows with the full constrained mixture fiber theory",
            Core::Materials::mix_full_constrained_mixture_fiber);

    add_named_int(m, "FIBER_ID", "Id of the fiber");
    add_named_int(m, "FIBER_MATERIAL_ID", "Id of fiber material");
    add_named_bool(m, "ENABLE_GROWTH", "Switch for the growth (default true)", true, true);
    add_named_bool(m, "ENABLE_BASAL_MASS_PRODUCTION",
        "Switch to enable the basal mass production rate (default true)", true, true);
    add_named_real(m, "DECAY_TIME", "Decay time of deposited tissue");
    add_named_real(m, "GROWTH_CONSTANT", "Growth constant of the tissue");
    add_named_real(m, "DEPOSITION_STRETCH", "Stretch at which the fiber is deposited");
    add_named_int(m, "INITIAL_DEPOSITION_STRETCH_TIMEFUNCT",
        "Id of the time function to scale the deposition stretch (Default: 0=None)", 0, true);
    add_named_int(m, "INIT", "Initialization mode for fibers (1=element fibers, 3=nodal fibers)");
    add_named_string(m, "ADAPTIVE_HISTORY_STRATEGY",
        "Strategy for adaptive history integration (none, model_equation, higher_order)", "none",
        true);
    add_named_real(
        m, "ADAPTIVE_HISTORY_TOLERANCE", "Tolerance of the adaptive history", 1e-6, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for a remodel fiber
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Constituent_ExplicitRemodelFiber",
        "A 1D constituent that remodels", Core::Materials::mix_remodelfiber_expl);

    add_named_int(m, "FIBER_ID", "Id of the fiber", 1, true);
    add_named_int(m, "FIBER_MATERIAL_ID", "Id of fiber material");

    add_named_bool(m, "ENABLE_GROWTH", "Switch for the growth (default true)", true, true);
    add_named_bool(m, "ENABLE_BASAL_MASS_PRODUCTION",
        "Switch to enable the basal mass production rate (default true)", true, true);
    add_named_real(m, "DECAY_TIME", "Decay time of deposited tissue");
    add_named_real(m, "GROWTH_CONSTANT", "Growth constant of the tissue");
    add_named_real(m, "DEPOSITION_STRETCH", "Stretch at with the fiber is deposited");
    add_named_int(m, "DEPOSITION_STRETCH_TIMEFUNCT",
        "Id of the time function to scale the deposition stretch (Default: 0=None)", 0, true);
    add_named_bool(
        m, "INELASTIC_GROWTH", "Mixture rule has inelastic growth (default false)", false, true);
    add_named_int(m, "INIT", "Initialization mode for fibers (1=element fibers, 2=nodal fibers)");
    add_named_real(
        m, "GAMMA", "Angle of fiber alignment in degree (default = 0.0 degrees)", 0.0, true);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for a remodel fiber
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Constituent_ImplicitRemodelFiber",
        "A 1D constituent that remodels", Core::Materials::mix_remodelfiber_impl);

    add_named_int(m, "FIBER_ID", "Id of the fiber");
    add_named_int(m, "FIBER_MATERIAL_ID", "Id of fiber material");

    add_named_bool(m, "ENABLE_GROWTH", "Switch for the growth (default true)", true, true);
    add_named_bool(m, "ENABLE_BASAL_MASS_PRODUCTION",
        "Switch to enable the basal mass production rate (default true)", true, true);
    add_named_real(m, "DECAY_TIME", "Decay time of deposited tissue");
    add_named_real(m, "GROWTH_CONSTANT", "Growth constant of the tissue");
    add_named_real(m, "DEPOSITION_STRETCH", "Stretch at with the fiber is deposited");
    add_named_int(m, "DEPOSITION_STRETCH_TIMEFUNCT",
        "Id of the time function to scale the deposition stretch (Default: 0=None)", 0, true);
    add_named_int(m, "INIT", "Initialization mode for fibers (1=element fibers, 2=nodal fibers)");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent material for a remodel fiber with exponential strain energy function
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MIX_Constituent_RemodelFiber_Material_Exponential",
        "An exponential strain energy function for the remodel fiber",
        Core::Materials::mix_remodelfiber_material_exponential);


    add_named_real(m, "K1", "First parameter of exponential strain energy function");
    add_named_real(m, "K2", "Second parameter of exponential strain energy function");
    add_named_bool(
        m, "COMPRESSION", "Bool, whether the fiber material also supports compressive forces.");

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


    add_named_real(m, "K1", "First parameter of exponential strain energy function");
    add_named_real(m, "K2", "Second parameter of exponential strain energy function");
    add_named_bool(
        m, "COMPRESSION", "Bool, whether the fiber material also supports compressive forces.");
    add_named_real(m, "SIGMA_MAX", "Maximum active Cauchy-stress");
    add_named_real(m, "LAMBDAMAX", "Stretch at maximum active Cauchy-stress");
    add_named_real(m, "LAMBDA0", "Stretch at zero active Cauchy-stress");
    add_named_real(m, "LAMBDAACT", "Current stretch", 1.0, true);
    add_named_real(m, "DENS", "Density of the whole mixture");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Function mixture rule for solid mixtures
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Rule_Function",
        "A mixture rule where the mass fractions are scaled by functions of space and time",
        Core::Materials::mix_rule_function);

    add_named_real(m, "DENS", "");
    add_named_int(m, "NUMCONST", "number of mixture constituents");
    add_named_int_vector(m, "MASSFRACFUNCT",
        "list of functions (their ids) defining the mass fractions of the mixture constituents",
        "NUMCONST");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Map mixture rule for solid mixtures
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_Rule_Map",
        "A mixture rule where the mass fractions are defined elementwise as discrete values",
        Core::Materials::mix_rule_map);

    add_named_real(m, "DENS", "");
    add_named_int(m, "NUMCONST", "number of mixture constituents");

    // definition of operation and print string for post processed component "MASSFRACMAPFILE"
    using mapType = std::unordered_map<int, std::vector<double>>;
    std::function<const mapType(const std::string&)> operation =
        [](const std::string& map_file) -> mapType
    {
      // map pattern file needs to be placed in same folder as input file
      std::filesystem::path input_file_path =
          Global::Problem::instance()->output_control_file()->input_file_name();
      const auto map_file_path = input_file_path.replace_filename(map_file);

      std::ifstream file_stream(map_file_path);

      if (file_stream.fail()) FOUR_C_THROW("Invalid file %s!", map_file_path.c_str());

      auto map_reduction_operation = [](mapType acc, const mapType& next)
      {
        for (const auto& [key, value] : next)
        {
          acc[key] = value;
        }
        return acc;
      };

      return Core::IO::convert_lines<mapType, mapType>(file_stream, map_reduction_operation);
    };
    const std::string print_string = std::string(
        "map of massfraction values retrieved from pattern file with rows in the format \"eleid: "
        "massfrac_1, massfrac_2, ...\"");

    add_named_processed_component(m, "MASSFRACMAPFILE",
        "file path of pattern file defining the massfractions as discrete values", operation,
        print_string, false);

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Base mixture rule for solid mixtures
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MIX_Rule_Simple", "Simple mixture rule", Core::Materials::mix_rule_simple);

    add_named_real(m, "DENS", "");
    add_named_int(m, "NUMCONST", "number of mixture constituents");
    add_named_real_vector(
        m, "MASSFRAC", "list mass fractions of the mixture constituents", "NUMCONST");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // Base mixture rule for solid mixtures
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MIX_GrowthRemodelMixtureRule",
        "Mixture rule for growth/remodel homogenized constrained mixture models",
        Core::Materials::mix_rule_growthremodel);

    add_named_int(m, "GROWTH_STRATEGY", "Material id of the growth strategy");
    add_named_real(m, "DENS", "");
    add_named_int(m, "NUMCONST", "number of mixture constituents");
    add_named_real_vector(
        m, "MASSFRAC", "list mass fractions of the mixture constituents", "NUMCONST");

    Mat::append_material_definition(matlist, m);
  }

  /*----------------------------------------------------------------------*/
  // crystal plasticity
  {
    auto m = std::make_shared<Mat::MaterialDefinition>(
        "MAT_crystal_plasticity", " Crystal plasticity ", Core::Materials::m_crystplast);
    add_named_real(m, "TOL", "tolerance for internal Newton iteration");
    add_named_real(m, "YOUNG", "Young's modulus");
    add_named_real(m, "NUE", "Poisson's ratio");
    add_named_real(m, "DENS", "Mass density");
    add_named_string(m, "LAT", "lattice type: FCC, BCC, HCP, D019 or L10", "FCC");
    add_named_real(m, "CTOA", "c to a ratio of crystal unit cell");
    add_named_real(m, "ABASE", "base length a of the crystal unit cell");
    add_named_int(m, "NUMSLIPSYS", "number of slip systems");
    add_named_int(m, "NUMSLIPSETS", "number of slip system sets");
    add_named_int_vector(m, "SLIPSETMEMBERS",
        "vector of NUMSLIPSYS indices ranging from 1 to NUMSLIPSETS that indicate to which set "
        "each slip system belongs",
        "NUMSLIPSYS");
    add_named_int_vector(m, "SLIPRATEEXP",
        "vector containing NUMSLIPSETS entries for the rate sensitivity exponent", "NUMSLIPSETS");
    add_named_real_vector(m, "GAMMADOTSLIPREF",
        "vector containing NUMSLIPSETS entries for the reference slip shear rate", "NUMSLIPSETS");
    add_named_real_vector(m, "DISDENSINIT",
        "vector containing NUMSLIPSETS entries for the initial dislocation density", "NUMSLIPSETS");
    add_named_real_vector(m, "DISGENCOEFF",
        "vector containing NUMSLIPSETS entries for the dislocation generation coefficients",
        "NUMSLIPSETS");
    add_named_real_vector(m, "DISDYNRECCOEFF",
        "vector containing NUMSLIPSETS entries for the coefficients for dynamic dislocation "
        "removal",
        "NUMSLIPSETS");
    add_named_real_vector(m, "TAUY0",
        "vector containing NUMSLIPSETS entries for the lattice resistance to slip, e.g. the "
        "Peierls barrier",
        "NUMSLIPSETS");
    add_named_real_vector(m, "MFPSLIP",
        "vector containing NUMSLIPSETS microstructural parameters that are relevant for Hall-Petch "
        "strengthening, e.g., grain size",
        "NUMSLIPSETS");
    add_named_real_vector(m, "SLIPHPCOEFF",
        "vector containing NUMSLIPSETS entries for the Hall-Petch coefficients corresponding to "
        "the "
        "microstructural parameters given in MFPSLIP",
        "NUMSLIPSETS");
    add_named_real_vector(m, "SLIPBYTWIN",
        "(optional) vector containing NUMSLIPSETS entries for the work hardening coefficients by "
        "twinning on non-coplanar systems",
        "NUMSLIPSETS", 0., true);
    add_named_int(m, "NUMTWINSYS", "(optional) number of twinning systems", 0, true);
    add_named_int(m, "NUMTWINSETS", "(optional) number of sets of twinning systems", 0, true);
    add_named_int_vector(m, "TWINSETMEMBERS",
        "(optional) vector of NUMTWINSYS indices ranging from 1 to NUMTWINSETS that indicate to "
        "which set each slip system belongs",
        "NUMTWINSYS", 0, true);
    add_named_int_vector(m, "TWINRATEEXP",
        "(optional) vector containing NUMTWINSETS entries for the rate sensitivity exponent",
        "NUMTWINSETS", 0, true);
    add_named_real_vector(m, "GAMMADOTTWINREF",
        "(optional) vector containing NUMTWINSETS entries for the reference slip shear rate",
        "NUMTWINSETS", 0., true);
    add_named_real_vector(m, "TAUT0",
        "(optional) vector containing NUMTWINSETS entries for the lattice resistance to twinning, "
        "e.g. the Peierls "
        "barrier",
        "NUMTWINSETS", 0., true);
    add_named_real_vector(m, "MFPTWIN",
        "(optional) vector containing NUMTWINSETS microstructural parameters that are relevant for "
        "Hall-Petch "
        "strengthening of twins, e.g., grain size",
        "NUMTWINSETS", 0., true);
    add_named_real_vector(m, "TWINHPCOEFF",
        "(optional) vector containing NUMTWINSETS entries for the Hall-Petch coefficients "
        "corresponding to the "
        "microstructural parameters given in MFPTWIN",
        "NUMTWINSETS", 0., true);
    add_named_real_vector(m, "TWINBYSLIP",
        "(optional) vector containing NUMTWINSETS entries for the work hardening coefficients by "
        "slip",
        "NUMTWINSETS", 0., true);
    add_named_real_vector(m, "TWINBYTWIN",
        "(optional) vector containing NUMTWINSETS entries for the work hardening coefficients by "
        "twins on non-coplanar systems",
        "NUMTWINSETS", 0., true);
    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // linear elastic material in one direction
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_LinElast1D",
        "linear elastic material in one direction", Core::Materials::m_linelast1D);

    add_named_real(m, "YOUNG", "Young's modulus");
    add_named_real(m, "DENS", "mass density");

    Mat::append_material_definition(matlist, m);
  }

  /*--------------------------------------------------------------------*/
  // linear elastic material with growth in one direction
  {
    auto m = std::make_shared<Mat::MaterialDefinition>("MAT_LinElast1DGrowth",
        "linear elastic material with growth in one direction",
        Core::Materials::m_linelast1D_growth);

    add_named_real(m, "YOUNG", "Young's modulus");
    add_named_real(m, "DENS", "mass density");
    add_named_real(m, "C0", "reference concentration");
    add_named_bool(m, "AOS_PROP_GROWTH",
        "growth proportional to amount of substance (AOS) if true or proportional to concentration "
        "if false");
    add_named_int(m, "POLY_PARA_NUM", "number of polynomial coefficients");
    add_named_real_vector(m, "POLY_PARAMS", "coefficients of polynomial", "POLY_PARA_NUM");

    Mat::append_material_definition(matlist, m);
  }

  // deliver
  return vm;
}

FOUR_C_NAMESPACE_CLOSE
