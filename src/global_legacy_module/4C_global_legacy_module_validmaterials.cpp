// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_global_legacy_module_validmaterials.hpp"

#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io_file_reader.hpp"
#include "4C_io_input_field.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_input_spec_validators.hpp"
#include "4C_mat_electrode.hpp"
#include "4C_mat_fiber_interpolation.hpp"
#include "4C_mat_fluidporo_singlephase.hpp"
#include "4C_mat_micromaterial.hpp"
#include "4C_mat_muscle_combo.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_input.hpp"

#include <filesystem>
#include <string>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::unordered_map<Core::Materials::MaterialType, Core::IO::InputSpec> Global::valid_materials()
{
  using namespace Core::IO::InputSpecBuilders;
  std::unordered_map<Core::Materials::MaterialType, Core::IO::InputSpec> known_materials;

  /*----------------------------------------------------------------------*/
  // Newtonian fluid
  {
    known_materials[Core::Materials::m_fluid] = group("MAT_fluid",
        {
            parameter<double>("DYNVISCOSITY", {.description = "dynamic viscosity"}),
            parameter<double>("DENSITY", {.description = "spatial mass density"}),
            parameter<double>(
                "GAMMA", {.description = "surface tension coefficient", .default_value = 0.0}),
        },
        {.description = "Newtonian fluid"});
  }

  /*----------------------------------------------------------------------*/
  // Weakly compressible fluid according to Murnaghan-Tait
  {
    known_materials[Core::Materials::m_fluid_murnaghantait] = group("MAT_fluid_murnaghantait",
        {
            parameter<double>("DYNVISCOSITY", {.description = "dynamic viscosity"}),
            parameter<double>("REFDENSITY", {.description = "reference spatial mass density"}),
            parameter<double>("REFPRESSURE", {.description = "reference pressure"}),
            parameter<double>("REFBULKMODULUS", {.description = "reference bulk modulus"}),
            parameter<double>(
                "MATPARAMETER", {.description = "material parameter according to Murnaghan-Tait"}),
            parameter<double>(
                "GAMMA", {.description = "surface tension coefficient", .default_value = 0.0}),
        },
        {.description = "Weakly compressible fluid according to Murnaghan-Tait"});
  }

  /*----------------------------------------------------------------------*/
  // Linear law (pressure-dependent) for the density and the viscosity
  {
    known_materials[Core::Materials::m_fluid_linear_density_viscosity] = group(
        "MAT_fluid_linear_density_viscosity",
        {
            parameter<double>("REFDENSITY", {.description = "reference density"}),
            parameter<double>("REFVISCOSITY", {.description = "reference viscosity"}),
            parameter<double>("REFPRESSURE", {.description = "reference pressure"}),
            parameter<double>("COEFFDENSITY", {.description = "density-pressure coefficient"}),
            parameter<double>("COEFFVISCOSITY", {.description = "viscosity-pressure coefficient"}),
            parameter<double>(
                "GAMMA", {.description = "surface tension coefficient", .default_value = 0.0}),
        },
        {.description = "Linear law (pressure-dependent) for the density and the viscosity"});
  }

  /*----------------------------------------------------------------------*/
  // Weakly compressible fluid
  {
    known_materials[Core::Materials::m_fluid_weakly_compressible] =
        group("MAT_fluid_weakly_compressible",
            {
                parameter<double>("VISCOSITY", {.description = "viscosity"}),
                parameter<double>("REFDENSITY", {.description = "reference density"}),
                parameter<double>("REFPRESSURE", {.description = "reference pressure"}),
                parameter<double>("COMPRCOEFF", {.description = "compressibility coefficient"}),
            },
            {.description = "Weakly compressible fluid"});
  }

  /*----------------------------------------------------------------------*/
  // fluid with non-linear viscosity according to Carreau-Yasuda
  {
    known_materials[Core::Materials::m_carreauyasuda] = group("MAT_carreauyasuda",
        {
            parameter<double>("NU_0", {.description = "zero-shear viscosity"}),
            parameter<double>("NU_INF", {.description = "infinite-shear viscosity"}),
            parameter<double>("LAMBDA", {.description = "characteristic time"}),
            parameter<double>("APARAM", {.description = "constant parameter"}),
            parameter<double>("BPARAM", {.description = "constant parameter"}),
            parameter<double>("DENSITY", {.description = "density"}),
        },
        {.description = "fluid with non-linear viscosity according to Carreau-Yasuda"});
  }

  /*----------------------------------------------------------------------*/
  // fluid with nonlinear viscosity according to a modified power law
  {
    known_materials[Core::Materials::m_modpowerlaw] = group("MAT_modpowerlaw",
        {
            parameter<double>("MCONS", {.description = "consistency"}),
            parameter<double>("DELTA", {.description = "safety factor"}),
            parameter<double>("AEXP", {.description = "exponent"}),
            parameter<double>("DENSITY", {.description = "density"}),
        },
        {.description = "fluid with nonlinear viscosity according to a modified power law"});
  }

  /*----------------------------------------------------------------------*/
  // fluid with non-linear viscosity according to Herschel-Bulkley
  {
    known_materials[Core::Materials::m_herschelbulkley] = group("MAT_herschelbulkley",
        {
            parameter<double>("TAU_0", {.description = "yield stress"}),
            parameter<double>("KFAC", {.description = "constant factor"}),
            parameter<double>("NEXP", {.description = "exponent"}),
            parameter<double>("MEXP", {.description = "exponent"}),
            parameter<double>("LOLIMSHEARRATE", {.description = "lower limit of shear rate"}),
            parameter<double>("UPLIMSHEARRATE", {.description = "upper limit of shear rate"}),
            parameter<double>("DENSITY", {.description = "density"}),
        },
        {.description = "fluid with non-linear viscosity according to Herschel-Bulkley"});
  }

  /*----------------------------------------------------------------------*/
  // lubrication material
  {
    known_materials[Core::Materials::m_lubrication] = group("MAT_lubrication",
        {
            parameter<int>("LUBRICATIONLAWID", {.description = "lubrication law id"}),
            parameter<double>("DENSITY", {.description = "lubricant density"}),
        },
        {.description = "lubrication material"});
  }


  /*----------------------------------------------------------------------*/
  // constant lubrication material law
  {
    known_materials[Core::Materials::m_lubrication_law_constant] =
        group("MAT_lubrication_law_constant",
            {
                parameter<double>("VISCOSITY", {.description = "lubricant viscosity"}),
            },
            {.description = "constant lubrication material law"});
  }

  /*----------------------------------------------------------------------*/
  // Barus viscosity lubrication material law
  {
    known_materials[Core::Materials::m_lubrication_law_barus] = group("MAT_lubrication_law_barus",
        {
            parameter<double>("ABSViscosity", {.description = "absolute lubricant viscosity"}),
            parameter<double>("PreVisCoeff", {.description = "pressure viscosity coefficient"}),
        },
        {.description = "barus lubrication material law"});
  }

  /*----------------------------------------------------------------------*/
  // Roeland viscosity lubrication material law
  {
    known_materials[Core::Materials::m_lubrication_law_roeland] =
        group("MAT_lubrication_law_roeland",
            {
                parameter<double>("ABSViscosity", {.description = "absolute lubricant viscosity"}),
                parameter<double>("PreVisCoeff", {.description = "pressure viscosity coefficient"}),
                parameter<double>("RefVisc", {.description = "reference viscosity"}),
                parameter<double>("RefPress", {.description = "reference Pressure"}),
            },
            {.description = "roeland lubrication material law"});
  }

  /*----------------------------------------------------------------------*/
  // scalar transport material (with potential reaction coefficient)
  {
    known_materials[Core::Materials::m_scatra] = group("MAT_scatra",
        {
            parameter<double>("DIFFUSIVITY", {.description = "kinematic diffusivity"}),
            parameter<double>(
                "REACOEFF", {.description = "reaction coefficient", .default_value = 0.0}),
            parameter<double>("SCNUM", {.description = "schmidt number", .default_value = 0.0}),
            parameter<double>("DENSIFICATION",
                {.description = "densification coefficient", .default_value = 0.0}),
            parameter<bool>("REACTS_TO_EXTERNAL_FORCE",
                {.description = "reacts to external force", .default_value = false}),
        },
        {.description = "scalar transport material"});
  }


  /*----------------------------------------------------------------------*/
  // scalar transport material (with potential reaction coefficient)
  {
    known_materials[Core::Materials::m_scatra_reaction_poroECM] = group("MAT_scatra_reaction_poro",
        {
            parameter<int>("NUMSCAL", {.description = "number of scalars for these elements"}),
            parameter<std::vector<int>>("STOICH", {.description = "reaction stoichometrie list",
                                                      .size = from_parameter<int>("NUMSCAL")}),
            parameter<double>("REACCOEFF", {.description = "reaction coefficient"}),
            parameter<double>("REACSCALE", {.description = "scaling for reaction coefficient"}),
            parameter<int>(
                "DISTRFUNCT", {.description = "spatial distribution of reaction coefficient",
                                  .default_value = 0}),
            parameter<std::string>("COUPLING",
                {.description = "type of coupling: simple_multiplicative, power_multiplicative, "
                                "constant, michaelis_menten, by_function, no_coupling (default)",
                    .default_value = "no_coupling"}),
            parameter<std::vector<double>>(
                "ROLE", {.description = "role in michaelis-menten like reactions",
                            .size = from_parameter<int>("NUMSCAL")}),
            parameter<std::optional<std::vector<double>>>(
                "REACSTART", {.description = "starting point of reaction",
                                 .size = from_parameter<int>("NUMSCAL")}),
        },
        {.description = "scalar transport material"});
  }
  /*----------------------------------------------------------------------*/
  // scalar transport reaction material
  {
    known_materials[Core::Materials::m_scatra_reaction] = group("MAT_scatra_reaction",
        {
            parameter<int>("NUMSCAL", {.description = "number of scalars for these elements"}),
            parameter<std::vector<int>>("STOICH", {.description = "reaction stoichometrie list",
                                                      .size = from_parameter<int>("NUMSCAL")}),
            parameter<double>("REACCOEFF", {.description = "reaction coefficient"}),
            parameter<int>(
                "DISTRFUNCT", {.description = "spatial distribution of reaction coefficient",
                                  .default_value = 0}),
            parameter<std::string>("COUPLING",
                {.description = "type of coupling: simple_multiplicative, power_multiplicative, "
                                "constant, michaelis_menten, by_function, no_coupling (default)",
                    .default_value = "no_coupling"}),
            parameter<std::vector<double>>(
                "ROLE", {.description = "role in michaelis-menten like reactions",
                            .size = from_parameter<int>("NUMSCAL")}),
            parameter<std::optional<std::vector<double>>>(
                "REACSTART", {.description = "starting point of reaction",
                                 .size = from_parameter<int>("NUMSCAL")}),
        },
        {.description = "advanced reaction material"});
  }

  /*----------------------------------------------------------------------*/
  // scalar transport reaction material (species in fluid)
  {
    known_materials[Core::Materials::m_scatra_in_fluid_porofluid_pressure_based] =
        group("MAT_scatra_multiporo_fluid",
            {
                parameter<double>("DIFFUSIVITY", {.description = "kinematic diffusivity"}),
                parameter<int>(
                    "PHASEID", {.description = "ID of fluid phase the "
                                               "scalar is associated with. Starting with zero."}),
                parameter<double>(
                    "REACOEFF", {.description = "reaction coefficient", .default_value = 0.0}),
                parameter<double>("SCNUM", {.description = "schmidt number", .default_value = 0.0}),
                parameter<double>("DENSIFICATION",
                    {.description = "densification coefficient", .default_value = 0.0}),
                parameter<double>("DELTA", {.description = "delta", .default_value = 0.0}),
                parameter<double>("MIN_SAT",
                    {.description = "minimum saturation under which also corresponding mass "
                                    "fraction is equal to zero",
                        .default_value = 1.0e-9}),
                parameter<bool>("REACTS_TO_EXTERNAL_FORCE",
                    {.description = "reacts to external force", .default_value = false}),
                parameter<int>("RELATIVE_MOBILITY_FUNCTION_ID",
                    {.description = "relative mobility function ID", .default_value = 0}),
            },
            {.description =
                    "advanced reaction material for multiphase porous flow (species in fluid)"});
  }

  /*----------------------------------------------------------------------*/
  // scalar transport reaction material (species in volume fraction)
  {
    known_materials[Core::Materials::m_scatra_in_volfrac_porofluid_pressure_based] = group(
        "MAT_scatra_multiporo_volfrac",
        {
            parameter<double>("DIFFUSIVITY", {.description = "kinematic diffusivity"}),
            parameter<int>("PHASEID",
                {.description =
                        "ID of fluid phase the scalar is associated with. Starting with zero."}),
            parameter<double>(
                "REACOEFF", {.description = "reaction coefficient", .default_value = 0.0}),
            parameter<double>("SCNUM", {.description = "schmidt number", .default_value = 0.0}),
            parameter<double>("DENSIFICATION",
                {.description = "densification coefficient", .default_value = 0.0}),
            parameter<double>("DELTA", {.description = "delta", .default_value = 0.0}),
            parameter<bool>("REACTS_TO_EXTERNAL_FORCE",
                {.description = "reacts to external force", .default_value = false}),
            parameter<int>("RELATIVE_MOBILITY_FUNCTION_ID",
                {.description = "relative mobility function ID", .default_value = 0}),
        },
        {.description =
                "advanced reaction material for multiphase porous flow (species in volfrac)"});
  }

  /*----------------------------------------------------------------------*/
  // scalar transport reaction material (species in solid)
  {
    known_materials[Core::Materials::m_scatra_in_solid_porofluid_pressure_based] =
        group("MAT_scatra_multiporo_solid",
            {
                parameter<double>("DIFFUSIVITY", {.description = "kinematic diffusivity"}),
                parameter<double>(
                    "REACOEFF", {.description = "reaction coefficient", .default_value = 0.0}),
                parameter<double>("SCNUM", {.description = "schmidt number", .default_value = 0.0}),
                parameter<double>("DENSIFICATION",
                    {.description = "densification coefficient", .default_value = 0.0}),
                parameter<double>("DELTA", {.description = "delta", .default_value = 0.0}),
                parameter<bool>("REACTS_TO_EXTERNAL_FORCE",
                    {.description = "reacts to external force", .default_value = false}),
            },
            {.description =
                    "advanced reaction material for multiphase porous flow (species in solid)"});
  }

  /*----------------------------------------------------------------------*/
  // scalar transport reaction material (temperature)
  {
    known_materials[Core::Materials::m_scatra_as_temperature_porofluid_pressure_based] =
        group("MAT_scatra_multiporo_temperature",
            {
                parameter<int>("NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE",
                    {.description = "number of fluid dofs"}),
                parameter<std::vector<double>>("CP_FLUID",
                    {.description = "heat capacity fluid phases",
                        .size = from_parameter<int>("NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE")}),
                parameter<int>("NUMVOLFRAC", {.description = "number of volfrac dofs"}),
                parameter<std::vector<double>>(
                    "CP_VOLFRAC", {.description = "heat capacity volfrac",
                                      .size = from_parameter<int>("NUMVOLFRAC")}),
                parameter<double>("CP_SOLID", {.description = "heat capacity solid"}),
                parameter<std::vector<double>>("KAPPA_FLUID",
                    {.description = "thermal diffusivity fluid phases",
                        .size = from_parameter<int>("NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE")}),
                parameter<std::vector<double>>(
                    "KAPPA_VOLFRAC", {.description = "thermal diffusivity volfrac",
                                         .size = from_parameter<int>("NUMVOLFRAC")}),
                parameter<double>("KAPPA_SOLID", {.description = "heat capacity solid"}),
                parameter<double>(
                    "DIFFUSIVITY", {.description = "kinematic diffusivity", .default_value = 1.0}),
                parameter<double>(
                    "REACOEFF", {.description = "reaction coefficient", .default_value = 0.0}),
                parameter<double>("SCNUM", {.description = "schmidt number", .default_value = 0.0}),
                parameter<double>("DENSIFICATION",
                    {.description = "densification coefficient", .default_value = 0.0}),
                parameter<bool>("REACTS_TO_EXTERNAL_FORCE",
                    {.description = "reacts to external force", .default_value = false}),
            },
            {.description = "advanced reaction material for multiphase porous flow (temperature)"});
  }

  /*----------------------------------------------------------------------*/
  // scalar transport chemotaxis material
  {
    known_materials[Core::Materials::m_scatra_chemotaxis] = group("MAT_scatra_chemotaxis",
        {
            parameter<int>(
                "NUMSCAL", {.description = "number of chemotactic pairs for these elements"}),
            parameter<std::vector<int>>("PAIR",
                {.description = "chemotaxis pairing", .size = from_parameter<int>("NUMSCAL")}),
            parameter<double>("CHEMOCOEFF", {.description = "chemotaxis coefficient"}),
        },
        {.description = "chemotaxis material"});
  }

  /*----------------------------------------------------------------------*/
  // scalar transport material for multi-scale approach
  {
    known_materials[Core::Materials::m_scatra_multiscale] = group("MAT_scatra_multiscale",
        {
            parameter<std::string>("MICROFILE",
                {.description = "input file for micro scale", .default_value = "filename.dat"}),
            parameter<int>("MICRODIS_NUM", {.description = "number of micro-scale discretization"}),
            parameter<double>("POROSITY", {.description = "porosity"}),
            parameter<double>("TORTUOSITY", {.description = "tortuosity"}),
            parameter<double>("A_s", {.description = "specific micro-scale surface area"}),
            parameter<double>("DIFFUSIVITY", {.description = "kinematic diffusivity"}),
            parameter<double>(
                "REACOEFF", {.description = "reaction coefficient", .default_value = 0.0}),
            parameter<double>("SCNUM", {.description = "Schmidt number", .default_value = 0.0}),
            parameter<double>("DENSIFICATION",
                {.description = "densification coefficient", .default_value = 0.0}),
            parameter<bool>("REACTS_TO_EXTERNAL_FORCE",
                {.description = "reacts to external force", .default_value = false}),
        },
        {.description = "scalar transport material for multi-scale approach"});
  }

  /*----------------------------------------------------------------------*/
  // Weickenmeier muscle material
  {
    using namespace Core::IO::InputSpecBuilders::Validators;

    known_materials[Core::Materials::m_muscle_weickenmeier] = group("MAT_Muscle_Weickenmeier",
        {
            parameter<double>("ALPHA", {.description = "experimentally fitted material parameter",
                                           .validator = positive<double>()}),
            parameter<double>("BETA", {.description = "experimentally fitted material parameter",
                                          .validator = positive<double>()}),
            parameter<double>("GAMMA", {.description = "experimentally fitted material parameter",
                                           .validator = positive<double>()}),
            parameter<double>(
                "KAPPA", {.description = "material parameter for coupled volumetric contribution"}),
            parameter<double>(
                "OMEGA0", {.description = "weighting factor for isotropic tissue constituents",
                              .validator = in_range<double>(0.0, 1.0)}),
            parameter<double>("ACTMUNUM",
                {.description =
                        "number of active motor units per undeformed muscle cross-sectional area",
                    .validator = positive_or_zero<double>()}),
            parameter<int>("MUTYPESNUM", {.description = "number of motor unit types"}),
            parameter<std::vector<double>>(
                "INTERSTIM", {.description = "interstimulus interval",
                                 .validator = all_elements(positive_or_zero<double>()),
                                 .size = from_parameter<int>("MUTYPESNUM")}),
            parameter<std::vector<double>>(
                "FRACACTMU", {.description = "fraction of motor unit type",
                                 .validator = all_elements(positive_or_zero<double>()),
                                 .size = from_parameter<int>("MUTYPESNUM")}),
            parameter<std::vector<double>>(
                "FTWITCH", {.description = "twitch force of motor unit type",
                               .validator = all_elements(positive_or_zero<double>()),
                               .size = from_parameter<int>("MUTYPESNUM")}),
            parameter<std::vector<double>>(
                "TTWITCH", {.description = "twitch contraction time of motor unit type",
                               .validator = all_elements(positive_or_zero<double>()),
                               .size = from_parameter<int>("MUTYPESNUM")}),
            parameter<double>("LAMBDAMIN",
                {.description = "minimal active fiber stretch", .validator = positive<double>()}),
            parameter<double>("LAMBDAOPT",
                {.description =
                        "optimal active fiber stretch related to active nominal stress maximum",
                    .validator = positive<double>()}),
            parameter<double>("DOTLAMBDAMIN", {.description = "minimal stretch rate"}),
            parameter<double>(
                "KE", {.description = "parameter controlling the curvature of the velocity "
                                      "dependent activation function in the eccentric case",
                          .validator = positive_or_zero<double>()}),
            parameter<double>(
                "KC", {.description = "parameter controlling the curvature of the velocity "
                                      "dependent activation function in the concentric case",
                          .validator = positive_or_zero<double>()}),
            parameter<double>(
                "DE", {.description = "parameter controlling the amplitude of the velocity "
                                      "dependent activation function in the eccentric case",
                          .validator = positive_or_zero<double>()}),
            parameter<double>(
                "DC", {.description = "parameter controlling the amplitude of the velocity "
                                      "dependent activation function in the concentric case",
                          .validator = positive_or_zero<double>()}),
            parameter<int>("ACTTIMESNUM",
                {.description = "number of time boundaries to prescribe activation"}),
            parameter<std::vector<double>>(
                "ACTTIMES", {.description = "time boundaries between intervals",
                                .size = from_parameter<int>("ACTTIMESNUM")}),
            parameter<int>("ACTINTERVALSNUM",
                {.description = "number of time intervals to prescribe activation"}),
            parameter<std::vector<double>>("ACTVALUES",
                {.description = "scaling factor in intervals (1=full activation, 0=no activation)",
                    .size = from_parameter<int>("ACTINTERVALSNUM")}),
            parameter<double>(
                "DENS", {.description = "density", .validator = positive_or_zero<double>()}),
        },
        {.description = "Weickenmeier muscle material"});
  }

  /*----------------------------------------------------------------------*/
  // Combo muscle material
  {
    using namespace Core::IO::InputSpecBuilders::Validators;

    known_materials[Core::Materials::m_muscle_combo] = group("MAT_Muscle_Combo",
        {
            parameter<double>("ALPHA", {.description = "experimentally fitted material parameter",
                                           .validator = positive<double>()}),
            parameter<double>("BETA", {.description = "experimentally fitted material parameter",
                                          .validator = positive<double>()}),
            parameter<double>("GAMMA", {.description = "experimentally fitted material parameter",
                                           .validator = positive<double>()}),
            parameter<double>(
                "KAPPA", {.description = "material parameter for coupled volumetric contribution"}),
            parameter<double>(
                "OMEGA0", {.description = "weighting factor for isotropic tissue constituents",
                              .validator = in_range<double>(0.0, 1.0)}),
            parameter<double>("POPT", {.description = "tetanised optimal (maximal) active stress",
                                          .validator = positive_or_zero<double>()}),
            parameter<double>("LAMBDAMIN",
                {.description = "minimal active fiber stretch", .validator = positive<double>()}),
            parameter<double>("LAMBDAOPT",
                {.description =
                        "optimal active fiber stretch related to active nominal stress maximum",
                    .validator = positive<double>()}),
            one_of({
                parameter<int>("ACTIVATION_FUNCTION_ID",
                    {.description =
                            "function id for time- and space-dependency of muscle activation"}),
                input_field<std::vector<std::pair<double, double>>>("ACTIVATION_VALUES",
                    {.description = "json input file containing a map of "
                                    "elementwise-defined discrete values "
                                    "for time- and space-dependency of muscle activation"}),
            }),
            parameter<double>(
                "DENS", {.description = "density", .validator = positive_or_zero<double>()}),
        },
        {.description = "Combo muscle material"});
  }

  /*----------------------------------------------------------------------*/
  // Active strain Giantesio muscle material
  {
    using namespace Core::IO::InputSpecBuilders::Validators;

    known_materials[Core::Materials::m_muscle_giantesio] = group("MAT_Muscle_Giantesio",
        {
            parameter<double>("ALPHA", {.description = "experimentally fitted material parameter",
                                           .validator = positive<double>()}),
            parameter<double>("BETA", {.description = "experimentally fitted material parameter",
                                          .validator = positive<double>()}),
            parameter<double>("GAMMA", {.description = "experimentally fitted material parameter",
                                           .validator = positive<double>()}),
            parameter<double>(
                "KAPPA", {.description = "material parameter for coupled volumetric contribution"}),
            parameter<double>(
                "OMEGA0", {.description = "weighting factor for isotropic tissue constituents",
                              .validator = in_range<double>(0.0, 1.0)}),
            parameter<double>("ACTMUNUM",
                {.description =
                        "number of active motor units per undeformed muscle cross-sectional area",
                    .validator = positive_or_zero<double>()}),
            parameter<int>("MUTYPESNUM", {.description = "number of motor unit types"}),
            parameter<std::vector<double>>(
                "INTERSTIM", {.description = "interstimulus interval",
                                 .validator = all_elements(positive_or_zero<double>()),
                                 .size = from_parameter<int>("MUTYPESNUM")}),
            parameter<std::vector<double>>(
                "FRACACTMU", {.description = "fraction of motor unit type",
                                 .validator = all_elements(positive_or_zero<double>()),
                                 .size = from_parameter<int>("MUTYPESNUM")}),
            parameter<std::vector<double>>(
                "FTWITCH", {.description = "twitch force of motor unit type",
                               .validator = all_elements(positive_or_zero<double>()),
                               .size = from_parameter<int>("MUTYPESNUM")}),
            parameter<std::vector<double>>(
                "TTWITCH", {.description = "twitch contraction time of motor unit type",
                               .validator = all_elements(positive_or_zero<double>()),
                               .size = from_parameter<int>("MUTYPESNUM")}),
            parameter<double>("LAMBDAMIN",
                {.description = "minimal active fiber stretch", .validator = positive<double>()}),
            parameter<double>("LAMBDAOPT",
                {.description =
                        "optimal active fiber stretch related to active nominal stress maximum",
                    .validator = positive<double>()}),
            parameter<double>("DOTLAMBDAMIN", {.description = "minimal stretch rate"}),
            parameter<double>(
                "KE", {.description = "parameter controlling the curvature of the velocity "
                                      "dependent activation function in the eccentric case",
                          .validator = positive_or_zero<double>()}),
            parameter<double>(
                "KC", {.description = "parameter controlling the curvature of the velocity "
                                      "dependent activation function in the concentric case",
                          .validator = positive_or_zero<double>()}),
            parameter<double>(
                "DE", {.description = "parameter controlling the amplitude of the velocity "
                                      "dependent activation function in the eccentric case",
                          .validator = positive_or_zero<double>()}),
            parameter<double>(
                "DC", {.description = "parameter controlling the amplitude of the velocity "
                                      "dependent activation function in the concentric case",
                          .validator = positive_or_zero<double>()}),
            parameter<int>("ACTTIMESNUM",
                {.description = "number of time boundaries to prescribe activation"}),
            parameter<std::vector<double>>(
                "ACTTIMES", {.description = "time boundaries between intervals",
                                .size = from_parameter<int>("ACTTIMESNUM")}),
            parameter<int>("ACTINTERVALSNUM",
                {.description = "number of time intervals to prescribe activation"}),
            parameter<std::vector<double>>("ACTVALUES",
                {.description = "scaling factor in intervals (1=full activation, 0=no activation)",
                    .size = from_parameter<int>("ACTINTERVALSNUM")}),
            parameter<double>(
                "DENS", {.description = "density", .validator = positive_or_zero<double>()}),
        },
        {.description = "Giantesio active strain muscle material"});
  }

  /*----------------------------------------------------------------------*/
  // Myocard muscle material (with complicated reaction coefficient)
  {
    known_materials[Core::Materials::m_myocard] = group("MAT_myocard",
        {
            parameter<double>("DIFF1", {.description = "conductivity in fiber direction"}),
            parameter<double>(
                "DIFF2", {.description = "conductivity perpendicular to fiber direction"}),
            parameter<double>(
                "DIFF3", {.description = "conductivity perpendicular to fiber direction"}),
            parameter<double>("PERTURBATION_DERIV",
                {.description = "perturbation for calculation of reaction coefficient derivative"}),
            parameter<std::string>(
                "MODEL", {.description = "Model type: MV (default), FHN, TNNP, SAN or INADA",
                             .default_value = "MV"}),
            parameter<std::string>(
                "TISSUE", {.description = "Tissue type: M (default), ENDO, EPI, AN, N or NH",
                              .default_value = "M"}),
            parameter<double>(
                "TIME_SCALE", {.description = "Scale factor for time units of Model"}),
        },
        {.description = "Myocard muscle material"});
  }

  /*----------------------------------------------------------------------*/
  // material according to Sutherland law
  {
    known_materials[Core::Materials::m_sutherland] = group("MAT_sutherland",
        {
            parameter<double>("REFVISC", {.description = "reference dynamic viscosity (kg/(m*s))"}),
            parameter<double>("REFTEMP", {.description = "reference temperature (K)"}),
            parameter<double>("SUTHTEMP", {.description = "Sutherland temperature (K)"}),
            parameter<double>(
                "SHC", {.description = "specific heat capacity at constant pressure (J/(kg*K))"}),
            parameter<double>("PRANUM", {.description = "Prandtl number"}),
            parameter<double>(
                "THERMPRESS", {.description = "(initial) thermodynamic pressure (J/m^3)"}),
            parameter<double>("GASCON", {.description = "specific gas constant R (J/(kg*K))"}),
        },
        {.description = "material according to Sutherland law"});
  }


  /*----------------------------------------------------------------------*/
  // material parameters for ion species in electrolyte solution (gjb 07/08)
  {
    known_materials[Core::Materials::m_ion] = group("MAT_ion",
        {
            parameter<double>("DIFFUSIVITY", {.description = "kinematic diffusivity"}),
            parameter<double>("VALENCE", {.description = "valence (= charge number)"}),
            parameter<double>("DENSIFICATION",
                {.description = "densification coefficient", .default_value = 0.0}),
            parameter<double>("ELIM_DIFFUSIVITY",
                {.description = "kinematic diffusivity of elim. species", .default_value = 0.0}),
            parameter<double>(
                "ELIM_VALENCE", {.description = "valence of elim. species", .default_value = 0.0}),
        },
        {.description = "material parameters for ion species in electrolyte solution"});
  }

  /*----------------------------------------------------------------------*/
  // material parameters for ion species in electrolyte solution (ehrl 07/12)
  {
    known_materials[Core::Materials::m_newman] = group("MAT_newman",
        {
            parameter<double>("VALENCE", {.description = "valence (= charge number)"}),
            parameter<int>("DIFF_COEF_CONC_DEP_FUNCT",
                {.description = "function number of function describing concentration dependence "
                                "of diffusion coefficient"}),
            parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT",
                {.description =
                        "FUNCT number describing temperature scaling of diffusion coefficient"}),
            parameter<int>("TRANSNR", {.description = "curve number for transference number"}),
            parameter<int>("THERMFAC", {.description = "curve number for thermodynamic factor"}),
            parameter<int>(
                "COND_CONC_DEP_FUNCT", {.description = "function number of function describing "
                                                       "concentration dependence of conductivity"}),
            parameter<int>("COND_TEMP_SCALE_FUNCT",
                {.description = "FUNCT number describing temperature scaling of conductivity"}),
            parameter<int>(
                "DIFF_PARA_NUM", {.description = "number of parameters for diffusion coefficient",
                                     .default_value = 0}),
            parameter<std::vector<double>>(
                "DIFF_PARA", {.description = "parameters for diffusion coefficient",
                                 .default_value = std::vector<double>{},
                                 .size = from_parameter<int>("DIFF_PARA_NUM")}),
            parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM",
                {.description = "number of parameters for scaling function describing temperature "
                                "dependence of diffusion coefficient",
                    .default_value = 0}),
            parameter<std::vector<double>>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA",
                {.description = "parameters for function describing temperature dependence of "
                                "diffusion coefficient",
                    .default_value = std::vector<double>{},
                    .size = from_parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM")}),
            parameter<int>(
                "TRANS_PARA_NUM", {.description = "number of parameters for transference number",
                                      .default_value = 0}),
            parameter<std::vector<double>>(
                "TRANS_PARA", {.description = "parameters for transference number",
                                  .default_value = std::vector<double>{},
                                  .size = from_parameter<int>("TRANS_PARA_NUM")}),
            parameter<int>(
                "THERM_PARA_NUM", {.description = "number of parameters for thermodynamic factor",
                                      .default_value = 0}),
            parameter<std::vector<double>>(
                "THERM_PARA", {.description = "parameters for thermodynamic factor",
                                  .default_value = std::vector<double>{},
                                  .size = from_parameter<int>("THERM_PARA_NUM")}),
            parameter<int>("COND_PARA_NUM",
                {.description = "number of parameters for conductivity", .default_value = 0}),
            parameter<std::vector<double>>(
                "COND_PARA", {.description = "parameters for conductivity",
                                 .default_value = std::vector<double>{},
                                 .size = from_parameter<int>("COND_PARA_NUM")}),
            parameter<int>("COND_TEMP_SCALE_FUNCT_PARA_NUM",
                {.description = "number of parameters for temperature scaling of conductivity",
                    .default_value = 0}),
            parameter<std::vector<double>>("COND_TEMP_SCALE_FUNCT_PARA",
                {.description = "parameters for temperature scaling of conductivity",
                    .default_value = std::vector<double>{},
                    .size = from_parameter<int>("COND_TEMP_SCALE_FUNCT_PARA_NUM")}),
        },
        {.description = "material parameters for ion species in electrolyte solution"});
  }

  /*----------------------------------------------------------------------*/
  // material parameters for ion species in electrolyte solution for multi-scale approach (fang
  // 07/17)
  {
    known_materials[Core::Materials::m_newman_multiscale] = group("MAT_newman_multiscale",
        {
            parameter<double>("VALENCE", {.description = "valence (= charge number)"}),
            parameter<int>("DIFF_COEF_CONC_DEP_FUNCT",
                {.description = "function number of function describing concentration dependence "
                                "of diffusion coefficient"}),
            parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT",
                {.description =
                        "FUNCT number describing temperature scaling of diffusion coefficient"}),
            parameter<int>("TRANSNR", {.description = "curve number for transference number"}),
            parameter<int>("THERMFAC", {.description = "curve number for thermodynamic factor"}),
            parameter<int>(
                "COND_CONC_DEP_FUNCT", {.description = "function number of function describing "
                                                       "concentration dependence of conductivity"}),
            parameter<int>("COND_TEMP_SCALE_FUNCT",
                {.description = "FUNCT number describing temperature scaling of conductivity"}),
            parameter<double>("ELECTRONIC_COND", {.description = "electronic conductivity"}),
            parameter<int>("ELECTRONIC_COND_CONC_SCALE_FUNC_NUM",
                {.description = "FUNCT number describing concentration dependence of electronic "
                                "conductivity"}),
            parameter<double>("A_s", {.description = "specific micro-scale surface area"}),
            parameter<std::string>("MICROFILE",
                {.description = "input file for micro scale", .default_value = "filename.dat"}),
            parameter<int>("MICRODIS_NUM", {.description = "number of micro-scale discretization"}),
            // optional parameters for implemented concentration-depending functions
            parameter<int>(
                "DIFF_PARA_NUM", {.description = "number of parameters for diffusion coefficient",
                                     .default_value = 0}),
            parameter<std::vector<double>>(
                "DIFF_PARA", {.description = "parameters for diffusion coefficient",
                                 .default_value = std::vector<double>{},
                                 .size = from_parameter<int>("DIFF_PARA_NUM")}),
            parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM",
                {.description = "number of parameters for scaling function describing temperature "
                                "dependence of diffusion coefficient",
                    .default_value = 0}),
            parameter<std::vector<double>>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA",
                {.description = "parameters for function describing temperature dependence of "
                                "diffusion coefficient",
                    .default_value = std::vector<double>{},
                    .size = from_parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM")}),
            parameter<int>(
                "TRANS_PARA_NUM", {.description = "number of parameters for transference number",
                                      .default_value = 0}),
            parameter<std::vector<double>>(
                "TRANS_PARA", {.description = "parameters for transference number",
                                  .default_value = std::vector<double>{},
                                  .size = from_parameter<int>("TRANS_PARA_NUM")}),
            parameter<int>(
                "THERM_PARA_NUM", {.description = "number of parameters for thermodynamic factor",
                                      .default_value = 0}),
            parameter<std::vector<double>>(
                "THERM_PARA", {.description = "parameters for thermodynamic factor",
                                  .default_value = std::vector<double>{},
                                  .size = from_parameter<int>("THERM_PARA_NUM")}),
            parameter<int>("COND_PARA_NUM",
                {.description = "number of parameters for ionic conductivity", .default_value = 0}),
            parameter<std::vector<double>>(
                "COND_PARA", {.description = "parameters for ionic conductivity",
                                 .default_value = std::vector<double>{},
                                 .size = from_parameter<int>("COND_PARA_NUM")}),
            parameter<int>("COND_TEMP_SCALE_FUNCT_PARA_NUM",
                {.description = "number of parameters for temperature scaling of conductivity",
                    .default_value = 0}),
            parameter<std::vector<double>>("COND_TEMP_SCALE_FUNCT_PARA",
                {.description = "parameters for temperature scaling of conductivity",
                    .default_value = std::vector<double>{},
                    .size = from_parameter<int>("COND_TEMP_SCALE_FUNCT_PARA_NUM")}),
        },
        {.description = "material parameters for ion species in electrolyte solution for "
                        "multi-scale approach"});
  }

  {
    known_materials[Core::Materials::m_scl] = group("MAT_scl",
        {
            parameter<double>("VALENCE", {.description = "valence/charge number"}),
            parameter<int>("DIFF_COEF_CONC_DEP_FUNCT",
                {.description = "function number of function describing concentration dependence "
                                "of diffusion coefficient"}),
            parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT",
                {.description =
                        "function number describing temperature scaling of diffusion coefficient"}),
            parameter<int>("TRANSNR", {.description = "curve number for transference number"}),
            parameter<int>(
                "COND_CONC_DEP_FUNCT", {.description = "function number of function describing "
                                                       "concentration dependence of conductivity"}),
            parameter<int>("COND_TEMP_SCALE_FUNCT",
                {.description = "function number describing temperature scaling of conductivity"}),
            parameter<int>(
                "DIFF_PARA_NUM", {.description = "number of parameters for diffusion coefficient",
                                     .default_value = 0}),
            parameter<std::vector<double>>(
                "DIFF_PARA", {.description = "parameters for diffusion coefficient",
                                 .default_value = std::vector<double>{},
                                 .size = from_parameter<int>("DIFF_PARA_NUM")}),
            parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM",
                {.description = "number of parameters for scaling function describing temperature "
                                "dependence of diffusion coefficient",
                    .default_value = 0}),
            parameter<std::vector<double>>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA",
                {.description = "parameters for function describing temperature dependence of "
                                "diffusion coefficient",
                    .default_value = std::vector<double>{},
                    .size = from_parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM")}),
            parameter<int>(
                "TRANS_PARA_NUM", {.description = "number of parameters for transference number",
                                      .default_value = 0}),
            parameter<std::vector<double>>(
                "TRANS_PARA", {.description = "parameters for transference number",
                                  .default_value = std::vector<double>{},
                                  .size = from_parameter<int>("TRANS_PARA_NUM")}),
            parameter<int>("COND_PARA_NUM",
                {.description = "number of parameters for conductivity", .default_value = 0}),
            parameter<std::vector<double>>(
                "COND_PARA", {.description = "parameters for conductivity",
                                 .default_value = std::vector<double>{},
                                 .size = from_parameter<int>("COND_PARA_NUM")}),
            parameter<int>("COND_TEMP_SCALE_FUNCT_PARA_NUM",
                {.description = "number of parameters for temperature scaling of conductivity",
                    .default_value = 0}),
            parameter<std::vector<double>>("COND_TEMP_SCALE_FUNCT_PARA",
                {.description = "parameters for temperature scaling of conductivity",
                    .default_value = std::vector<double>{},
                    .size = from_parameter<int>("COND_TEMP_SCALE_FUNCT_PARA_NUM")}),
            parameter<double>("MAX_CONC", {.description = "maximum cation concentration"}),
            parameter<int>("EXTRAPOL_DIFF",
                {.description = "strategy for extrapolation of diffusion coefficient below 0 and "
                                "above MAX_CONC (-1: disabled, 0: constant)"}),
            parameter<double>("LIM_CONC",
                {.description = "limiting concentration for extrapolation", .default_value = 1.0}),
            parameter<double>("BULK_CONC", {.description = "bulk ion concentration"}),
            parameter<double>("SUSCEPT", {.description = "susceptibility"}),
            parameter<double>("DELTA_NU",
                {.description = "difference of partial molar volumes (vacancy & cation)"}),
        },
        {.description = "material parameters for space charge layers"});
  }


  /*----------------------------------------------------------------------*/
  // electrode material (fang 02/15)
  {
    known_materials[Core::Materials::m_electrode] = group("MAT_electrode",
        {
            // diffusivity and electronic conductivity
            parameter<int>("DIFF_COEF_CONC_DEP_FUNCT",
                {.description = "function number of function describing concentration dependence "
                                "of diffusion coefficient"}),
            parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT",
                {.description =
                        "FUNCT number describing temperature scaling of diffusion coefficient"}),
            parameter<int>(
                "COND_CONC_DEP_FUNCT", {.description = "function number of function describing "
                                                       "concentration dependence of conductivity"}),
            parameter<int>("COND_TEMP_SCALE_FUNCT",
                {.description = "FUNCT number describing temperature scaling of conductivity"}),

            // optional parameters for concentration dependency of diffusivity and electronic
            // conductivity
            parameter<int>(
                "DIFF_PARA_NUM", {.description = "number of parameters for diffusion coefficient",
                                     .default_value = 0}),
            parameter<std::vector<double>>(
                "DIFF_PARA", {.description = "parameters for diffusion coefficient",
                                 .default_value = std::vector<double>{},
                                 .size = from_parameter<int>("DIFF_PARA_NUM")}),
            parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM",
                {.description = "number of parameters for scaling function describing temperature "
                                "dependence of diffusion coefficient",
                    .default_value = 0}),
            parameter<std::vector<double>>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA",
                {.description = "parameters for function describing temperature dependence of "
                                "diffusion coefficient",
                    .default_value = std::vector<double>{},
                    .size = from_parameter<int>("DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM")}),
            parameter<int>(
                "COND_PARA_NUM", {.description = "number of parameters for electronic conductivity",
                                     .default_value = 0}),
            parameter<std::vector<double>>(
                "COND_PARA", {.description = "parameters for electronic conductivity",
                                 .default_value = std::vector<double>{},
                                 .size = from_parameter<int>("COND_PARA_NUM")}),
            parameter<int>("COND_TEMP_SCALE_FUNCT_PARA_NUM",
                {.description = "number of parameters for temperature scaling of conductivity",
                    .default_value = 0}),
            parameter<std::vector<double>>("COND_TEMP_SCALE_FUNCT_PARA",
                {.description = "parameters for temperature scaling of conductivity",
                    .default_value = std::vector<double>{},
                    .size = from_parameter<int>("COND_TEMP_SCALE_FUNCT_PARA_NUM")}),
            // saturation value of intercalated Lithium concentration
            parameter<double>(
                "C_MAX", {.description = "saturation value of intercalated Lithium concentration"}),

            // lithiation value corresponding to saturation value of intercalated Lithium
            // concentration
            parameter<double>(
                "CHI_MAX", {.description = "lithiation value corresponding to saturation value of "
                                           "intercalated Lithium concentration 'C_MAX'"}),

            // model for half cell open circuit potential of electrode
            group("OCP_MODEL",
                {
                    one_of(
                        {
                            group("Function",
                                {
                                    parameter<int>("OCP_FUNCT_NUM",
                                        {
                                            .description =
                                                "function number of function that is used to model "
                                                "the open circuit potential",
                                        }),
                                }),
                            group("Redlich-Kister",
                                {
                                    parameter<int>("OCP_PARA_NUM",
                                        {
                                            .description = "number of parameters underlying half "
                                                           "cell open circuit potential model",
                                        }),
                                    parameter<std::vector<double>>("OCP_PARA",
                                        {.description = "parameters underlying half cell open "
                                                        "circuit potential model",
                                            .size = from_parameter<int>("OCP_PARA_NUM")}),
                                }),
                            group("Taralov",
                                {
                                    parameter<std::vector<double>>("OCP_PARA",
                                        {.description = "parameters underlying half cell open "
                                                        "circuit potential model",
                                            .size = 13}),
                                }),
                        },
                        store_index_as<Mat::PAR::OCPModels>("OCP_MODEL")),
                    parameter<double>(
                        "X_MIN", {.description = "lower bound of range of validity as a fraction "
                                                 "of C_MAX for ocp calculation model"}),
                    parameter<double>(
                        "X_MAX", {.description = "upper bound of range of validity as a fraction "
                                                 "of C_MAX for ocp calculation model"}),
                }),
        },
        {.description = "electrode material"});
    ;
  }

  /*----------------------------------------------------------------------*/
  // material collection (gjb 07/08)
  {
    known_materials[Core::Materials::m_matlist] = group("MAT_matlist",
        {
            parameter<bool>("LOCAL",
                {.description =
                        "individual materials allocated per element or only at global scope"}),
            parameter<int>("NUMMAT", {.description = "number of materials in list"}),
            parameter<std::vector<int>>("MATIDS",
                {.description = "the list material IDs", .size = from_parameter<int>("NUMMAT")}),
        },
        {.description = "list/collection of materials, i.e. material IDs"});
  }

  /*----------------------------------------------------------------------*/
  // material collection with reactions (thon 09/14)
  {
    known_materials[Core::Materials::m_matlist_reactions] = group("MAT_matlist_reactions",
        {
            parameter<bool>("LOCAL",
                {.description =
                        "individual materials allocated per element or only at global scope"}),
            parameter<int>("NUMMAT", {.description = "number of materials in list"}),
            parameter<std::vector<int>>("MATIDS",
                {.description = "the list material IDs", .size = from_parameter<int>("NUMMAT")}),
            parameter<int>("NUMREAC", {.description = "number of reactions for these elements"}),
            parameter<std::vector<int>>("REACIDS", {.description = "advanced reaction list",
                                                       .default_value = std::vector{0},
                                                       .size = from_parameter<int>("NUMREAC")}),
        },
        {.description = "list/collection of materials, i.e. material IDs and list of reactions"});
  }

  /*----------------------------------------------------------------------*/
  // material collection with chemotaxis (thon 06/15)
  {
    known_materials[Core::Materials::m_matlist_chemotaxis] = group("MAT_matlist_chemotaxis",
        {
            parameter<bool>("LOCAL",
                {.description =
                        "individual materials allocated per element or only at global scope"}),
            parameter<int>("NUMMAT", {.description = "number of materials in list"}),
            parameter<std::vector<int>>("MATIDS",
                {.description = "the list material IDs", .size = from_parameter<int>("NUMMAT")}),
            parameter<int>("NUMPAIR", {.description = "number of pairs for these elements"}),
            parameter<std::vector<int>>("PAIRIDS", {.description = "chemotaxis pairs list",
                                                       .default_value = std::vector{0},
                                                       .size = from_parameter<int>("NUMPAIR")}),
        },
        {.description =
                "list/collection of materials, i.e. material IDs and list of chemotactic pairs"});
  }

  /*----------------------------------------------------------------------*/
  // material collection with reactions AND chemotaxis (thon 06/15)
  {
    known_materials[Core::Materials::m_matlist_chemoreac] = group("MAT_matlist_chemo_reac",
        {
            parameter<bool>("LOCAL",
                {.description =
                        "individual materials allocated per element or only at global scope"}),
            parameter<int>("NUMMAT", {.description = "number of materials in list"}),
            parameter<std::vector<int>>("MATIDS",
                {.description = "the list material IDs", .size = from_parameter<int>("NUMMAT")}),
            parameter<int>("NUMPAIR", {.description = "number of pairs for these elements"}),
            parameter<std::vector<int>>("PAIRIDS", {.description = "chemotaxis pairs list",
                                                       .default_value = std::vector{0},
                                                       .size = from_parameter<int>("NUMPAIR")}),
            parameter<int>("NUMREAC", {.description = "number of reactions for these elements"}),
            parameter<std::vector<int>>("REACIDS", {.description = "advanced reaction list",
                                                       .default_value = std::vector{0},
                                                       .size = from_parameter<int>("NUMREAC")}),
        },
        {.description = "list/collection of materials, i.e. material IDs and list of "
                        "reactive/chemotactic pairs"});
  }

  /*----------------------------------------------------------------------*/
  // material collection (ehrl 11/12)
  {
    known_materials[Core::Materials::m_elchmat] = group("MAT_elchmat",
        {
            parameter<bool>("LOCAL",
                {.description =
                        "individual materials allocated per element or only at global scope",
                    .default_value = false}),
            parameter<int>("NUMDOF", {.description = "number of dof's per node"}),
            parameter<int>("NUMSCAL", {.description = "number of transported scalars per node"}),
            parameter<int>("NUMPHASE", {.description = "number of phases in electrolyte"}),
            parameter<std::vector<int>>("PHASEIDS",
                {.description = "the list phasel IDs", .size = from_parameter<int>("NUMPHASE")}),
        },
        {.description = "specific list/collection of species and phases for elch applications"});
  }

  /*----------------------------------------------------------------------*/
  // material collection (ehrl 11/12)
  {
    known_materials[Core::Materials::m_elchphase] = group("MAT_elchphase",
        {
            parameter<bool>("LOCAL",
                {.description =
                        "individual materials allocated per element or only at global scope",
                    .default_value = false}),
            parameter<double>("EPSILON", {.description = "phase porosity"}),
            parameter<double>("TORTUOSITY", {.description = "inverse (!) of phase tortuosity"}),
            parameter<int>("NUMMAT", {.description = "number of materials in electrolyte"}),
            parameter<std::vector<int>>("MATIDS",
                {.description = "the list phasel IDs", .size = from_parameter<int>("NUMMAT")}),
        },
        {.description = "material parameters for ion species in electrolyte solution"});
  }

  /*--------------------------------------------------------------------*/
  // St.Venant--Kirchhoff
  {
    using namespace Core::IO::InputSpecBuilders::Validators;

    known_materials[Core::Materials::m_stvenant] = group("MAT_Struct_StVenantKirchhoff",
        {
            parameter<double>(
                "YOUNG", {.description = "Young's modulus", .validator = positive<double>()}),
            parameter<double>("NUE",
                {.description = "Poisson's ratio", .validator = in_range<double>(-1.0, excl(0.5))}),
            parameter<double>("DENS", {.description = "mass density"}),
        },
        {.description = "St.Venant--Kirchhoff material"});
  }

  /*--------------------------------------------------------------------*/
  // St.Venant--Kirchhoff with temperature
  {
    known_materials[Core::Materials::m_thermostvenant] = group("MAT_Struct_ThermoStVenantK",
        {
            parameter<int>("YOUNGNUM",
                {.description = "number of Young's modulus in list (if 1 Young is const, "
                                "if >1 Young is temperature) dependent"}),
            parameter<std::vector<double>>("YOUNG",
                {.description = "Young's modulus", .size = from_parameter<int>("YOUNGNUM")}),
            parameter<double>("NUE", {.description = "Poisson's ratio"}),
            parameter<double>("DENS", {.description = "mass density"}),
            parameter<double>(
                "THEXPANS", {.description = "constant coefficient of linear thermal expansion"}),
            parameter<double>("INITTEMP", {.description = "initial temperature"}),
            parameter<int>("THERMOMAT",
                {.description = "mat id of thermal material part", .default_value = -1}),
        },
        {.description = "Thermo St.Venant--Kirchhoff material"});
  }
  /*----------------------------------------------------------------------*/
  // Plastic linear elastic St.Venant Kirchhoff / Drucker Prager plasticity
  {
    known_materials[Core::Materials::m_pldruckprag] = group("MAT_Struct_DruckerPrager",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus"}),
            parameter<double>("NUE", {.description = "Poisson's ratio"}),
            parameter<double>("DENS", {.description = "Density"}),
            parameter<double>("ISOHARD", {.description = "linear isotropic hardening"}),
            parameter<double>("TOL", {.description = "Local Newton iteration tolerance"}),
            parameter<double>("C", {.description = "cohesion"}),
            parameter<double>("ETA", {.description = "Drucker Prager Constant Eta"}),
            parameter<double>("XI", {.description = "Drucker Prager Constant Xi"}),
            parameter<double>("ETABAR", {.description = "Drucker Prager Constant Etabar"}),
            parameter<std::string>("TANG", {.description = "Method to compute the material tangent",
                                               .default_value = "consistent"}),
            parameter<int>(
                "MAXITER", {.description = "Maximum Iterations for local Neutron Raphson",
                               .default_value = 50}),
        },
        {.description = "elastic St.Venant Kirchhoff / plastic drucker prager"});
  }

  /*----------------------------------------------------------------------*/
  // Linear thermo-elastic St.Venant Kirchhoff / plastic von Mises
  {
    known_materials[Core::Materials::m_thermopllinelast] = group("MAT_Struct_ThermoPlasticLinElast",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus"}),
            parameter<double>("NUE", {.description = "Poisson's ratio"}),
            parameter<double>("DENS", {.description = "mass density"}),
            parameter<double>(
                "THEXPANS", {.description = "coefficient of linear thermal expansion"}),
            parameter<double>("INITTEMP", {.description = "initial temperature"}),
            parameter<double>("YIELD", {.description = "yield stress"}),
            parameter<double>("ISOHARD", {.description = "isotropic hardening modulus"}),
            parameter<double>("KINHARD", {.description = "kinematic hardening modulus"}),
            parameter<int>("SAMPLENUM", {.description = "number of stress-strain pairs in list"}),
            parameter<std::vector<double>>("SIGMA_Y",
                {.description = "yield stress", .size = from_parameter<int>("SAMPLENUM")}),
            parameter<std::vector<double>>(
                "EPSBAR_P", {.description = "accumulated plastic strain corresponding to SIGMA_Y",
                                .size = from_parameter<int>("SAMPLENUM")}),
            parameter<double>("TOL", {.description = "tolerance for local Newton iteration"}),
            parameter<int>("THERMOMAT",
                {.description = "mat id of thermal material part", .default_value = -1}),
        },
        {.description = "Thermo-elastic St.Venant Kirchhoff / plastic von Mises material"});
  }

  /*----------------------------------------------------------------------*/
  // Plastic linear elastic St.Venant Kirchhoff / GTN plasticity
  {
    known_materials[Core::Materials::m_plgtn] = group("MAT_Struct_PlasticGTN",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus"}),
            parameter<double>("NUE", {.description = "Poisson's ratio"}),
            parameter<double>("DENS", {.description = "Density"}),
            parameter<double>("YIELD", {.description = "yield stress"}),
            parameter<double>("ISOHARD", {.description = "linear isotropic hardening"}),
            parameter<int>("HARDENING_FUNC",
                {.description = "Function number for isotropic hardening", .default_value = 0}),
            parameter<double>("TOL", {.description = "Local Newton iteration tolerance"}),
            parameter<int>("MAXITER",
                {.description = "Maximum Neutron Raphson Iterations", .default_value = 50}),
            parameter<double>("K1", {.description = "GTN Constant k1"}),
            parameter<double>("K2", {.description = "GTN Constant k2"}),
            parameter<double>("K3", {.description = "GTN constant k3"}),
            parameter<double>("F0", {.description = "GTN constant f0 for initial damage"}),
            parameter<double>("FN", {.description = "GTN constant fN for damage nucleation"}),
            parameter<double>("EN", {.description = "GTN constant eN for damage nucleation"}),
            parameter<double>("SN", {.description = "GTN constant sN for damage nucleation"}),
            parameter<double>("FC", {.description = "GTN constant fC for damage coalescence"}),
            parameter<double>(
                "KAPPA", {.description = "GTN constant kappa for damage coalescence"}),
            parameter<double>(
                "EF", {.description = "GTN stabilization parameter ef for damage coalescence",
                          .default_value = 0.0}),
        },
        {.description = "elastic St.Venant Kirchhoff / plastic GTN"});
  }

  /*----------------------------------------------------------------------*/
  // Finite strain superelasticity of shape memory alloys
  {
    known_materials[Core::Materials::m_superelast] = group("MAT_Struct_SuperElastSMA",
        {
            parameter<double>("DENS", {.description = "mass density"}),
            parameter<double>("YOUNG", {.description = "Young's modulus"}),
            parameter<double>("NUE", {.description = "Poisson's ratio"}),
            parameter<double>("EPSILON_L",
                {.description = "parameter representing the maximum deformation obtainable only by "
                                "detwinning of the multiple-variant martensite"}),
            parameter<double>(
                "T_AS_s", {.description = "Temperature at which the phase transformation from "
                                          "austenite to martensite starts"}),
            parameter<double>(
                "T_AS_f", {.description = "Temperature at which the phase transformation from "
                                          "austenite to martensite finishes"}),
            parameter<double>(
                "T_SA_s", {.description = "Temperature at which the phase transformation from "
                                          "martensite to autenite starts"}),
            parameter<double>(
                "T_SA_f", {.description = "Temperature at which the phase transformation from "
                                          "martensite to autenite finishes"}),
            parameter<double>("C_AS",
                {.description = "Coefficient of the linear temperature dependence of T_AS"}),
            parameter<double>("C_SA",
                {.description = "Coefficient of the linear temperature dependence of T_SA"}),
            parameter<double>(
                "SIGMA_AS_s", {.description = "stress at which the phase transformation from "
                                              "austenite to martensite begins"}),
            parameter<double>(
                "SIGMA_AS_f", {.description = "stress at which the phase transformation from "
                                              "austenite to martensite finishes"}),
            parameter<double>(
                "SIGMA_SA_s", {.description = "stress at which the phase transformation from "
                                              "martensite to austenite begins"}),
            parameter<double>(
                "SIGMA_SA_f", {.description = "stress at which the phase transformation from "
                                              "martensite to austenite finishes"}),
            parameter<double>(
                "ALPHA", {.description = "pressure dependency in the drucker-prager-type loading"}),
            parameter<int>("MODEL", {.description = "Model used for the evolution of martensitic "
                                                    "fraction (1=exponential; 2=linear)"}),
            parameter<double>(
                "BETA_AS", {.description = "parameter, measuring the speed of the transformation "
                                           "from austenite to martensite",
                               .default_value = 0.}),
            parameter<double>(
                "BETA_SA", {.description = "parameter, measuring the speed of the transformation "
                                           "from martensite to austenite",
                               .default_value = 0.}),
        },
        {.description = "finite strain superelastic shape memory alloy"});
  }

  /*----------------------------------------------------------------------*/
  // Thermo-hyperelasticity / finite strain von-Mises plasticity
  {
    known_materials[Core::Materials::m_thermoplhyperelast] = group(
        "MAT_Struct_ThermoPlasticHyperElast",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus"}),
            parameter<double>("NUE", {.description = "Poisson's ratio"}),
            parameter<double>("DENS", {.description = "mass density"}),
            parameter<double>(
                "CTE", {.description = "coefficient of thermal expansion", .default_value = 0.}),
            parameter<double>(
                "INITTEMP", {.description = "initial, reference temperature", .default_value = 0.}),
            parameter<double>("YIELD", {.description = "initial yield stress"}),
            parameter<double>("ISOHARD",
                {.description = "linear isotropic hardening modulus", .default_value = 0.}),
            parameter<double>(
                "SATHARDENING", {.description = "saturation hardening", .default_value = 0.}),
            parameter<double>(
                "HARDEXPO", {.description = "hardening exponent", .default_value = 0.}),
            parameter<double>("YIELDSOFT",
                {.description = "thermal yield stress softening", .default_value = 0.}),
            parameter<double>("HARDSOFT",
                {.description = "thermal hardening softening (acting on SATHARDENING and ISOHARD)",
                    .default_value = 0.}),
            parameter<double>("TOL",
                {.description = "tolerance for local Newton iteration", .default_value = 1.e-8}),
            parameter<int>("THERMOMAT",
                {.description = "mat id of thermal material part", .default_value = -1}),
        },
        {.description = "Thermo-hyperelastic / finite strain plastic von Mises material with "
                        "linear and exponential isotropic hardening"});
  }


  /*----------------------------------------------------------------------*/
  // Hyperelasticity / finite strain von-Mises plasticity
  {
    known_materials[Core::Materials::m_plnlnlogneohooke] = group("MAT_Struct_PlasticNlnLogNeoHooke",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus"}),
            parameter<double>("NUE", {.description = "Poisson's ratio"}),
            parameter<double>("DENS", {.description = "mass density"}),
            parameter<double>("YIELD", {.description = "yield stress", .default_value = 0.}),
            parameter<double>(
                "ISOHARD", {.description = "isotropic hardening modulus", .default_value = 0.}),
            parameter<double>(
                "SATHARDENING", {.description = "saturation hardening", .default_value = 0.}),
            parameter<double>(
                "HARDEXPO", {.description = "linear hardening exponent", .default_value = 0.}),
            parameter<double>("VISC", {.description = "VISCOSITY", .default_value = 0.}),
            parameter<double>(
                "RATE_DEPENDENCY", {.description = "rate dependency", .default_value = 0.}),
            parameter<double>("TOL", {.description = "Tolerance for local Newton-Raphson iteration",
                                         .default_value = 1.e-08}),
            parameter<int>("HARDENING_FUNC",
                {.description = "Function number for isotropic hardening", .default_value = 0}),
        },
        {.description = "hyperelastic / finite strain plastic von Mises material with linear and "
                        "exponential isotropic hardening or the definition of a hardening function "
                        "(VARFUNCTION using the variable epsp)"});
  }

  /*----------------------------------------------------------------------*/
  // Plastic linear elastic St.Venant Kirchhoff / von Mises
  {
    known_materials[Core::Materials::m_pllinelast] = group("MAT_Struct_PlasticLinElast",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus"}),
            parameter<double>("NUE", {.description = "Poisson's ratio"}),
            parameter<double>("DENS", {.description = "mass density"}),
            parameter<double>("YIELD", {.description = "yield stress"}),
            parameter<double>("ISOHARD", {.description = "linear isotropic hardening modulus"}),
            parameter<double>("KINHARD", {.description = "linear kinematic hardening modulus"}),
            parameter<double>("TOL", {.description = "tolerance for local Newton iteration"}),
        },
        {.description = "elastic St.Venant Kirchhoff / plastic von Mises material with linear "
                        "isotropic and kineamtic hardening"});
  }

  /*----------------------------------------------------------------------*/
  // Elastic visco-plastic finite strain material law without yield surface
  {
    known_materials[Core::Materials::m_vp_no_yield_surface] = group(
        "MAT_Struct_Viscoplastic_No_Yield_Surface",
        {
            // elasticity parameters
            parameter<double>("YOUNG", {.description = "Young's modulus"}),
            parameter<double>("NUE", {.description = "Poisson's ratio"}),
            parameter<double>("DENS", {.description = "material mass density"}),
            // visco-plasticity parameters
            parameter<double>("TEMPERATURE", {.description = "temperature in Kelvin"}),
            parameter<double>("PRE_EXP_FAC",
                {.description = "pre-exponential factor of plastic shear strain rate 'A'"}),
            parameter<double>("ACTIVATION_ENERGY", {.description = "activation energy 'Q'"}),
            parameter<double>("GAS_CONSTANT", {.description = "gas constant 'R'"}),
            parameter<double>("STRAIN_RATE_SENS", {.description = "strain-rate-sensitivity 'm'"}),
            parameter<double>(
                "INIT_FLOW_RES", {.description = "initial isotropic flow resistance 'S^0'"}),
            parameter<double>("FLOW_RES_PRE_FAC", {.description = "flow resistance factor 'H_0'"}),
            parameter<double>(
                "FLOW_RES_EXP", {.description = "flow resistance exponential value 'a'"}),
            parameter<double>(
                "FLOW_RES_SAT_FAC", {.description = "flow resistance saturation factor 'S_*'"}),
            parameter<double>(
                "FLOW_RES_SAT_EXP", {.description = "flow resistance saturation exponent 'b'"}),
        },
        {.description = "Elastic visco-plastic finite strain material law without yield surface"});
  }

  /*----------------------------------------------------------------------*/
  // Robinson's visco-plastic material
  {
    known_materials[Core::Materials::m_vp_robinson] = group("MAT_Struct_Robinson",
        {
            parameter<std::string>(
                "KIND", {.description = "kind of Robinson material: Butler, Arya, "
                                        "Arya_NarloyZ (default), Arya_CrMoSteel"}),
            parameter<int>("YOUNGNUM", {.description = "number of Young's modulus in list"}),
            parameter<std::vector<double>>("YOUNG",
                {.description = "Young's modulus", .size = from_parameter<int>("YOUNGNUM")}),
            parameter<double>("NUE", {.description = "Poisson's ratio"}),
            parameter<double>("DENS", {.description = "mass density"}),
            parameter<double>(
                "THEXPANS", {.description = "coefficient of linear thermal expansion"}),
            parameter<double>("INITTEMP", {.description = "initial temperature"}),
            parameter<double>("HRDN_FACT", {.description = "hardening factor 'A'"}),
            parameter<double>("HRDN_EXPO", {.description = "hardening power 'n'"}),
            parameter<int>(
                "SHRTHRSHLDNUM", {.description = "number of shear stress threshold 'K^2'in list"}),
            parameter<std::vector<double>>(
                "SHRTHRSHLD", {.description = "Bingam-Prager shear stress threshold 'K^2'",
                                  .size = from_parameter<int>("SHRTHRSHLDNUM")}),
            parameter<double>("RCVRY", {.description = "recovery factor 'R_0'"}),
            parameter<double>("ACTV_ERGY", {.description = "activation energy 'Q_0'"}),
            parameter<double>("ACTV_TMPR", {.description = "activation temperature 'T_0'"}),
            parameter<double>("G0", {.description = "'G_0'"}),
            parameter<double>("M_EXPO", {.description = "'m'"}),
            parameter<int>("BETANUM", {.description = "number of 'beta' in list"}),
            parameter<std::vector<double>>(
                "BETA", {.description = "beta", .size = from_parameter<int>("BETANUM")}),
            parameter<double>("H_FACT", {.description = "'H'"}),
            parameter<int>("THERMOMAT",
                {.description = "mat id of thermal material part", .default_value = -1}),
        },
        {.description = "Robinson's visco-plastic material"});
  }

  /*----------------------------------------------------------------------*/
  // Elasto-plastic material with damage, based on MAT_Struct_PlasticLinElast
  {
    known_materials[Core::Materials::m_elpldamage] = group("MAT_Struct_Damage",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus"}),
            parameter<double>("NUE", {.description = "Poisson's ratio"}),
            parameter<double>("DENS", {.description = "mass density"}),
            parameter<int>("SAMPLENUM", {.description = "number of stress-strain pairs in list"}),
            parameter<std::vector<double>>("SIGMA_Y",
                {.description = "yield stress", .size = from_parameter<int>("SAMPLENUM")}),
            parameter<std::vector<double>>(
                "EPSBAR_P", {.description = "accumulated plastic strain corresponding to SIGMA_Y",
                                .size = from_parameter<int>("SAMPLENUM")}),
            parameter<double>("DAMDEN", {.description = "denominator of damage evaluations law"}),
            parameter<double>("DAMEXP", {.description = "exponent of damage evaluations law"}),
            parameter<double>("DAMTHRESHOLD", {.description = "damage threshold"}),
            parameter<double>(
                "KINHARD", {.description = "kinematic hardening modulus, stress-like variable"}),
            parameter<double>(
                "KINHARD_REC", {.description = "recovery factor, scalar-valued variable"}),
            parameter<double>("SATHARDENING", {.description = "saturation hardening"}),
            parameter<double>("HARDEXPO", {.description = "hardening exponent"}),
            parameter<double>("TOL", {.description = "tolerance for local Newton iteration"}),
        },
        {.description = "elasto-plastic von Mises material with ductile damage"});
  }

  /*--------------------------------------------------------------------*/
  // aneurysm wall material according to Raghavan and Vorp [2000]
  {
    known_materials[Core::Materials::m_aaaneohooke] = group("MAT_Struct_AAANeoHooke",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus"}),
            parameter<double>("BETA", {.description = "2nd parameter"}),
            parameter<double>("NUE", {.description = "Poisson's ratio"}),
            parameter<double>("DENS", {.description = "mass density"}),
        },
        {.description = "aneurysm wall material according to Raghavan and Vorp [2000]"});
  }


  /*----------------------------------------------------------------------*/
  // Visco-elastic Neo-Hookean material law
  {
    known_materials[Core::Materials::m_visconeohooke] = group("MAT_VISCONEOHOOKE",
        {
            parameter<double>("YOUNGS_SLOW", {.description = "???"}),
            parameter<double>("POISSON", {.description = "???"}),
            parameter<double>("DENS", {.description = "???"}),
            parameter<double>("YOUNGS_FAST", {.description = "???"}),
            parameter<double>("RELAX", {.description = "???"}),
            parameter<double>("THETA", {.description = "???"}),
        },
        {.description = "visco-elastic neo-Hookean material law"});
  }

  /*----------------------------------------------------------------------*/
  // Visco-elastic anisotropic fiber material law
  {
    known_materials[Core::Materials::m_viscoanisotropic] = group("MAT_VISCOANISO",
        {
            parameter<double>("KAPPA", {.description = "dilatation modulus"}),
            parameter<double>("MUE", {.description = "Shear Modulus"}),
            parameter<double>("DENS", {.description = "Density"}),
            parameter<double>("K1", {.description = "Parameter for linear fiber stiffness"}),
            parameter<double>("K2", {.description = "Parameter for exponential fiber stiffness"}),
            parameter<double>("GAMMA", {.description = "angle between fibers"}),
            parameter<double>("BETA_ISO",
                {.description = "ratio between elasticities in generalized Maxweel body"}),
            parameter<double>("BETA_ANISO",
                {.description = "ratio between elasticities in generalized Maxweel body"}),
            parameter<double>("RELAX_ISO", {.description = "isotropic relaxation time"}),
            parameter<double>("RELAX_ANISO", {.description = "anisotropic relaxation time"}),
            parameter<double>(
                "MINSTRETCH", {.description = "minimal principal stretch fibers do respond to"}),
            parameter<int>("ELETHICKDIR",
                {.description = "Element thickness direction applies also to fibers (only sosh)"}),
        },
        {.description = "visco-elastic anisotropic fibre material law"});
  }

  /*----------------------------------------------------------------------*/
  // Structural micro-scale approach: material parameters are calculated from microscale simulation
  {
    known_materials[Core::Materials::m_struct_multiscale] = group("MAT_Struct_Multiscale",
        {
            parameter<std::string>("MICROFILE",
                {.description = "inputfile for microstructure", .default_value = "filename.dat"}),
            parameter<int>("MICRODIS_NUM", {.description = "Number of microscale discretization"}),
            parameter<double>(
                "INITVOL", {.description = "Initial volume of RVE", .default_value = 0.0}),
            parameter<Mat::PAR::MicroMaterial::RuntimeOutputOption>("RUNTIMEOUTPUT_GP",
                {.description = "Specify the Gauss Points of this element for "
                                "which runtime output is generated",
                    .default_value = Mat::PAR::MicroMaterial::RuntimeOutputOption::all}),
        },
        {.description = "Structural micro-scale approach: material parameters are calculated from "
                        "microscale simulation"});
  }

  /*----------------------------------------------------------------------*/
  // collection of hyperelastic materials
  {
    known_materials[Core::Materials::m_elasthyper] = group("MAT_ElastHyper",
        {
            parameter<int>("NUMMAT", {.description = "number of materials/potentials in list"}),
            parameter<std::vector<int>>("MATIDS", {.description = "the list material/potential IDs",
                                                      .size = from_parameter<int>("NUMMAT")}),
            parameter<double>("DENS", {.description = "material mass density"}),
            parameter<int>("POLYCONVEX",
                {.description = "1.0 if polyconvexity of system is checked", .default_value = 0}),
        },
        {.description = "list/collection of hyperelastic materials, i.e. material IDs"});
  }

  /*----------------------------------------------------------------------*/
  // viscohyperelastic material
  {
    known_materials[Core::Materials::m_viscoelasthyper] = group("MAT_ViscoElastHyper",
        {
            parameter<int>("NUMMAT", {.description = "number of materials/potentials in list"}),
            parameter<std::vector<int>>("MATIDS", {.description = "the list material/potential IDs",
                                                      .size = from_parameter<int>("NUMMAT")}),
            parameter<double>("DENS", {.description = "material mass density"}),
            parameter<int>("POLYCONVEX",
                {.description = "1.0 if polyconvexity of system is checked", .default_value = 0}),
        },
        {.description = "Viscohyperelastic material compatible with the collection of hyperelastic "
                        "materials"});
  }

  /*----------------------------------------------------------------------*/
  // collection of hyperelastic materials for finite strain plasticity
  {
    known_materials[Core::Materials::m_plelasthyper] = group("MAT_PlasticElastHyper",
        {
            parameter<int>("NUMMAT", {.description = "number of materials/potentials in list"}),
            parameter<std::vector<int>>("MATIDS", {.description = "the list material/potential IDs",
                                                      .size = from_parameter<int>("NUMMAT")}),
            parameter<double>("DENS", {.description = "material mass density"}),
            parameter<double>("INITYIELD", {.description = "initial yield stress"}),
            parameter<int>("POLYCONVEX",
                {.description = "1.0 if polyconvexity of system is checked", .default_value = 0}),
            parameter<double>("ISOHARD",
                {.description = "linear isotropic hardening modulus", .default_value = 0.}),
            parameter<double>("EXPISOHARD",
                {.description = "nonlinear isotropic hardening exponent", .default_value = 0.}),
            parameter<double>("INFYIELD",
                {.description = "saturation yield stress for nonlinear isotropic hardening",
                    .default_value = 0.}),
            parameter<double>("KINHARD",
                {.description = "linear kinematic hardening modulus", .default_value = 0.}),

            // visco-plasticity
            parameter<double>(
                "VISC", {.description = "Visco-Plasticity parameter 'eta' in Perzyna model",
                            .default_value = 0.}),
            parameter<double>("RATE_DEPENDENCY",
                {.description = "Visco-Plasticity parameter 'eta' in Perzyna model",
                    .default_value = 1.}),
            parameter<double>("VISC_SOFT",
                {.description =
                        "Visco-Plasticity temperature dependency (eta = eta_0 * (1-(T-T_0)*x)",
                    .default_value = 0.}),

            // optional pastic spin parameter
            parameter<double>("PL_SPIN_CHI",
                {.description = "Plastic spin coupling parameter chi (often called eta)",
                    .default_value = 0.0}),

            // optional Hill yield parameters
            parameter<double>(
                "rY_11", {.description = "relative yield stress in fiber1-direction (Y_11/Y_0)",
                             .default_value = 0.0}),
            parameter<double>(
                "rY_22", {.description = "relative yield stress in fiber2-direction (Y_22/Y_0)",
                             .default_value = 0.0}),
            parameter<double>(
                "rY_33", {.description = "relative yield stress in fiber3-direction (Y_33/Y_0)",
                             .default_value = 0.0}),
            parameter<double>(
                "rY_12", {.description = "relative shear yield stress in 12-direction (Y_12/Y_0)",
                             .default_value = 0.0}),
            parameter<double>(
                "rY_23", {.description = "relative shear yield stress in 23-direction (Y_23/Y_0)",
                             .default_value = 0.0}),
            parameter<double>(
                "rY_13", {.description = "relative shear yield stress in 13-direction (Y_13/Y_0)",
                             .default_value = 0.0}),

            // optional TSI parameters
            parameter<double>(
                "CTE", {.description = "coefficient of thermal expansion", .default_value = 0.}),
            parameter<double>(
                "INITTEMP", {.description = "initial, reference temperature", .default_value = 0.}),
            parameter<double>(
                "YIELDSOFT", {.description = "yield stress softening", .default_value = 0.}),
            parameter<double>(
                "HARDSOFT", {.description = "hardening softening", .default_value = 0.}),
            parameter<double>("TAYLOR_QUINNEY",
                {.description = "Taylor-Quinney factor for plastic heat conversion",
                    .default_value = 1.}),
        },
        {.description = "collection of hyperelastic materials for finite strain plasticity"});
  }

  /*----------------------------------------------------------------------*/
  // collection of hyperelastic materials for finite strain plasticity
  {
    known_materials[Core::Materials::m_plelasthyperVCU] = group("MAT_PlasticElastHyperVCU",
        {
            parameter<int>("NUMMAT", {.description = "number of materials/potentials in list"}),
            parameter<std::vector<int>>("MATIDS", {.description = "the list material/potential IDs",
                                                      .size = from_parameter<int>("NUMMAT")}),
            parameter<double>("DENS", {.description = "material mass density"}),
            parameter<double>("INITYIELD", {.description = "initial yield stress"}),
            parameter<double>("ISOHARD",
                {.description = "linear isotropic hardening modulus", .default_value = 0.}),
            parameter<double>("EXPISOHARD",
                {.description = "nonlinear isotropic hardening exponent", .default_value = 0.}),
            parameter<double>("INFYIELD",
                {.description = "saturation yield stress for nonlinear isotropic hardening",
                    .default_value = 0.}),
            parameter<double>("KINHARD",
                {.description = "linear kinematic hardening modulus", .default_value = 0.}),

            // visco-plasticity
            parameter<double>(
                "VISC", {.description = "Visco-Plasticity parameter 'eta' in Perzyna model",
                            .default_value = 0.}),
            parameter<double>("RATE_DEPENDENCY",
                {.description = "Visco-Plasticity parameter 'eta' in Perzyna model",
                    .default_value = 1.}),
            parameter<double>("VISC_SOFT",
                {.description =
                        "Visco-Plasticity temperature dependency (eta = eta_0 * (1-(T-T_0)*x)",
                    .default_value = 0.}),

            // optional pastic spin parameter
            parameter<double>("PL_SPIN_CHI",
                {.description = "Plastic spin coupling parameter chi (often called eta)",
                    .default_value = 0.0}),

            // optional Hill yield parameters
            parameter<double>(
                "rY_11", {.description = "relative yield stress in fiber1-direction (Y_11/Y_0)",
                             .default_value = 0.0}),
            parameter<double>(
                "rY_22", {.description = "relative yield stress in fiber2-direction (Y_22/Y_0)",
                             .default_value = 0.0}),
            parameter<double>(
                "rY_33", {.description = "relative yield stress in fiber3-direction (Y_33/Y_0)",
                             .default_value = 0.0}),
            parameter<double>(
                "rY_12", {.description = "relative shear yield stress in 12-direction (Y_12/Y_0)",
                             .default_value = 0.0}),
            parameter<double>(
                "rY_23", {.description = "relative shear yield stress in 23-direction (Y_23/Y_0)",
                             .default_value = 0.0}),
            parameter<double>(
                "rY_13", {.description = "relative shear yield stress in 13-direction (Y_13/Y_0)",
                             .default_value = 0.0}),

            // optional TSI parameters
            parameter<double>(
                "CTE", {.description = "coefficient of thermal expansion", .default_value = 0.}),
            parameter<double>(
                "INITTEMP", {.description = "initial, reference temperature", .default_value = 0.}),
            parameter<double>(
                "YIELDSOFT", {.description = "yield stress softening", .default_value = 0.}),
            parameter<double>(
                "HARDSOFT", {.description = "hardening softening", .default_value = 0.}),
            parameter<double>("TAYLOR_QUINNEY",
                {.description = "Taylor-Quinney factor for plastic heat conversion",
                    .default_value = 1.}),

            parameter<int>("POLYCONVEX",
                {.description = "1.0 if polyconvexity of system is checked", .default_value = 0}),
        },
        {.description = "collection of hyperelastic materials for finite strain plasticity"});
  }

  /*--------------------------------------------------------------------*/
  // logarithmic neo-Hooke material acc. to Bonet and Wood
  {
    known_materials[Core::Materials::mes_couplogneohooke] = group("ELAST_CoupLogNeoHooke",
        {
            parameter<std::string>(
                "MODE", {.description = "parameter set: YN (Young's modulus and Poisson's ration; "
                                        "default) or Lame (mue and lambda)"}),
            parameter<double>("C1", {.description = "E or mue"}),
            parameter<double>("C2", {.description = "nue or lambda"}),
        },
        {.description = "logarithmic neo-Hooke material acc. to Bonet and Wood"});
  }

  /*--------------------------------------------------------------------*/
  // Saint-Venant-Kirchhoff as elastic summand
  {
    known_materials[Core::Materials::mes_coupSVK] = group("ELAST_CoupSVK",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus"}),
            parameter<double>("NUE", {.description = "Poisson's ratio"}),
        },
        {.description = "Saint-Venant-Kirchhoff as elastic summand"});
  }

  /*--------------------------------------------------------------------*/
  // Simo-Pister type material
  {
    known_materials[Core::Materials::mes_coupsimopister] = group("ELAST_CoupSimoPister",
        {
            parameter<double>("MUE", {.description = "material constant"}),
        },
        {.description = "Simo-Pister type material"});
  }

  /*--------------------------------------------------------------------*/
  // logarithmic mixed neo-Hooke material
  {
    known_materials[Core::Materials::mes_couplogmixneohooke] = group("ELAST_CoupLogMixNeoHooke",
        {
            parameter<std::string>(
                "MODE", {.description = "parameter set: YN (Young's modulus and Poisson's ration; "
                                        "default) or Lame (mue and lambda)"}),
            parameter<double>("C1", {.description = "E or mue"}),
            parameter<double>("C2", {.description = "nue or lambda"}),
        },
        {.description = "mixed logarithmic neo-Hooke material"});
  }

  /*--------------------------------------------------------------------*/
  // coupled exponential material for compressible material (according to Weikenmeier_2014)
  {
    known_materials[Core::Materials::mes_coupexppol] = group("ELAST_CoupExpPol",
        {
            parameter<double>("A", {.description = "material constant"}),
            parameter<double>("B", {.description = "material constant linear I_1"}),
            parameter<double>("C", {.description = "material constant linear J"}),
        },
        {.description = "compressible, isochoric exponential material law for soft tissue"});
  }

  /*--------------------------------------------------------------------*/
  // compressible neo-Hooke material acc. to Holzapfel
  {
    known_materials[Core::Materials::mes_coupneohooke] = group("ELAST_CoupNeoHooke",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus", .default_value = 0.0}),
            parameter<double>("NUE", {.description = "Poisson's ratio", .default_value = 0.0}),
        },
        {.description = "compressible neo-Hooke material acc. to Holzapfel"});
  }
  // Mooney Rivlin  material acc. to Holzapfel
  {
    known_materials[Core::Materials::mes_coupmooneyrivlin] = group("ELAST_CoupMooneyRivlin",
        {
            parameter<double>("C1", {.description = "material constant", .default_value = 0.0}),
            parameter<double>("C2", {.description = "material constant", .default_value = 0.0}),
            parameter<double>("C3", {.description = "material constant", .default_value = 0.0}),
        },
        {.description = "Mooney - Rivlin material acc. to Holzapfel"});
  }

  /*--------------------------------------------------------------------*/
  // coupled Blatz and Ko material acc. to Holzapfel
  {
    known_materials[Core::Materials::mes_coupblatzko] = group("ELAST_CoupBlatzKo",
        {
            parameter<double>("MUE", {.description = "Shear modulus"}),
            parameter<double>("NUE", {.description = "Poisson's ratio"}),
            parameter<double>("F", {.description = "interpolation parameter"}),
        },
        {.description = "Blatz and Ko material acc. to Holzapfel"});
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of Neo-Hooke
  {
    known_materials[Core::Materials::mes_isoneohooke] = group("ELAST_IsoNeoHooke",
        {
            input_field<double>("MUE", {.description = "Shear modulus"}),
        },
        {.description = "isochoric part of neo-Hooke material acc. to Holzapfel"});
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of one-term Ogden material
  {
    known_materials[Core::Materials::mes_isoogden] = group("ELAST_IsoOgden",
        {
            parameter<double>("MUE", {.description = "Shear modulus"}),
            parameter<double>("ALPHA", {.description = "Nonlinearity parameter"}),
        },
        {.description = "isochoric part of the one-term Ogden material"});
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of Yeoh
  {
    known_materials[Core::Materials::mes_isoyeoh] = group("ELAST_IsoYeoh",
        {
            parameter<double>("C1", {.description = "Linear modulus"}),
            parameter<double>("C2", {.description = "Quadratic modulus"}),
            parameter<double>("C3", {.description = "Cubic modulus"}),
        },
        {.description = "isochoric part of  Yeoh material acc. to Holzapfel"});
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of iso1pow
  {
    known_materials[Core::Materials::mes_iso1pow] = group("ELAST_Iso1Pow",
        {
            parameter<double>("C", {.description = "material parameter"}),
            parameter<int>("D", {.description = "exponent"}),
        },
        {.description = "isochoric part of general power material"});
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of iso2pow
  {
    known_materials[Core::Materials::mes_iso2pow] = group("ELAST_Iso2Pow",
        {
            parameter<double>("C", {.description = "material parameter"}),
            parameter<int>("D", {.description = "exponent"}),
        },
        {.description = "isochoric part of general power material"});
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup1pow
  {
    known_materials[Core::Materials::mes_coup1pow] = group("ELAST_Coup1Pow",
        {
            parameter<double>("C", {.description = "material parameter"}),
            parameter<int>("D", {.description = "exponent"}),
        },
        {.description = "part of general power material"});
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup2pow
  {
    known_materials[Core::Materials::mes_coup2pow] = group("ELAST_Coup2Pow",
        {
            parameter<double>("C", {.description = "material parameter"}),
            parameter<int>("D", {.description = "exponent"}),
        },
        {.description = "part of general power material"});
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup3pow
  {
    known_materials[Core::Materials::mes_coup3pow] = group("ELAST_Coup3Pow",
        {
            parameter<double>("C", {.description = "material parameter"}),
            parameter<int>("D", {.description = "exponent"}),
        },
        {.description = "part of general power material"});
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup13apow
  {
    known_materials[Core::Materials::mes_coup13apow] = group("ELAST_Coup13aPow",
        {
            parameter<double>("C", {.description = "material parameter"}),
            parameter<int>("D", {.description = "exponent of all"}),
            parameter<double>("A", {.description = "negative exponent of I3"}),
        },
        {.description =
                "hyperelastic potential summand for multiplicative coupled invariants I1 and I3"});
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of expo
  {
    known_materials[Core::Materials::mes_isoexpopow] = group("ELAST_IsoExpoPow",
        {
            parameter<double>("K1", {.description = "material parameter"}),
            parameter<double>("K2", {.description = "material parameter"}),
            parameter<int>("C", {.description = "exponent"}),
        },
        {.description = "isochoric part of  exponential material acc. to Holzapfel"});
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of mooney rivlin
  {
    known_materials[Core::Materials::mes_isomooneyrivlin] = group("ELAST_IsoMooneyRivlin",
        {
            parameter<double>("C1", {.description = "Linear modulus for first invariant"}),
            parameter<double>("C2", {.description = "Linear modulus for second invariant"}),
        },
        {.description = "isochoric part of  Mooney-Rivlin material acc. to Holzapfel"});
  }

  /*--------------------------------------------------------------------*/
  // isochoric anisotropic material with one exponential fiber family
  {
    using namespace Core::IO::InputSpecBuilders::Validators;

    known_materials[Core::Materials::mes_isomuscleblemker] = group("ELAST_IsoMuscle_Blemker",
        {
            parameter<double>("G1", {.description = "muscle along fiber shear modulus",
                                        .validator = positive_or_zero<double>()}),
            parameter<double>("G2", {.description = "muscle cross fiber shear modulus",
                                        .validator = positive_or_zero<double>()}),
            parameter<double>(
                "P1", {.description = "linear material parameter for passive along-fiber response",
                          .validator = positive<double>()}),
            parameter<double>("P2",
                {.description = "exponential material parameter for passive along-fiber response",
                    .validator = positive<double>()}),
            parameter<double>("SIGMAMAX", {.description = "maximal active isometric stress",
                                              .validator = positive_or_zero<double>()}),
            parameter<double>("LAMBDAOFL",
                {.description = "optimal fiber stretch", .validator = positive<double>()}),
            parameter<double>("LAMBDASTAR",
                {.description =
                        "stretch at which the normalized passive fiber force becomes linear",
                    .validator = positive<double>()}),
            parameter<double>("ALPHA", {.description = "tetanised activation level,",
                                           .validator = positive_or_zero<double>()}),
            parameter<double>(
                "BETA", {.description = "constant scaling tanh-type activation function",
                            .validator = positive_or_zero<double>()}),
            parameter<double>("ACTSTARTTIME", {.description = "starting time of muscle activation",
                                                  .validator = positive_or_zero<double>()}),
        },
        {.description = "anisotropic Blemker muscle material"});
  }

  /*--------------------------------------------------------------------*/
  // test material to test elasthyper-toolbox
  {
    known_materials[Core::Materials::mes_isotestmaterial] = group("ELAST_IsoTestMaterial",
        {
            parameter<double>("C1", {.description = "Modulus for first invariant"}),
            parameter<double>("C2", {.description = "Modulus for second invariant"}),
        },
        {.description = "test material to test elasthyper-toolbox"});
  }

  /*----------------------------------------------------------------------*/
  // general fiber material for remodeling
  {
    known_materials[Core::Materials::mes_remodelfiber] = group("ELAST_RemodelFiber",
        {
            parameter<int>("NUMMAT", {.description = "number of materials/potentials in list"}),
            parameter<std::vector<int>>("MATIDS", {.description = "the list material/potential IDs",
                                                      .size = from_parameter<int>("NUMMAT")}),
            parameter<double>(
                "TDECAY", {.description = "decay time of Poisson (degradation) process"}),
            parameter<double>("GROWTHFAC",
                {.description = "time constant for collagen growth", .default_value = 0.0}),
            parameter<std::vector<double>>(
                "COLMASSFRAC", {.description = "initial mass fraction of first collagen fiber "
                                               "family in constraint mixture",
                                   .default_value = std::vector{0.0},
                                   .size = from_parameter<int>("NUMMAT")}),
            parameter<double>("DEPOSITIONSTRETCH", {.description = "deposition stretch"}),
        },
        {.description = "General fiber material for remodeling"});
  }

  /*--------------------------------------------------------------------*/
  // volumetric contribution of Sussman Bathe
  {
    known_materials[Core::Materials::mes_volsussmanbathe] = group("ELAST_VolSussmanBathe",
        {
            parameter<double>("KAPPA", {.description = "dilatation modulus"}),
        },
        {.description = "volumetric part of  SussmanBathe material"});
  }

  /*--------------------------------------------------------------------*/
  // volumetric penalty contribution
  {
    known_materials[Core::Materials::mes_volpenalty] = group("ELAST_VolPenalty",
        {
            parameter<double>("EPSILON", {.description = "penalty parameter"}),
            parameter<double>("GAMMA", {.description = "penalty parameter"}),
        },
        {.description = "Penalty formulation for the volumetric part"});
  }

  /*--------------------------------------------------------------------*/
  // volumetric contribution of Ogden
  {
    known_materials[Core::Materials::mes_vologden] = group("ELAST_VolOgden",
        {
            parameter<double>("KAPPA", {.description = "dilatation modulus"}),
            parameter<double>("BETA", {.description = "empiric constant"}),
        },
        {.description = "Ogden formulation for the volumetric part"});
  }

  /*--------------------------------------------------------------------*/
  // volumetric power law contribution
  {
    known_materials[Core::Materials::mes_volpow] = group("ELAST_VolPow",
        {
            parameter<double>("A", {.description = "prefactor of power law"}),
            parameter<double>("EXPON", {.description = "exponent of power law"}),
        },
        {.description = "Power law formulation for the volumetric part"});
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with one exponential fiber family
  {
    known_materials[Core::Materials::mes_coupanisoexpoactive] = group("ELAST_CoupAnisoExpoActive",
        {
            parameter<double>("K1", {.description = "linear constant"}),
            parameter<double>("K2", {.description = "exponential constant"}),
            parameter<double>("GAMMA", {.description = "angle"}),
            parameter<double>("K1COMP", {.description = "linear constant"}),
            parameter<double>("K2COMP", {.description = "exponential constant"}),
            parameter<int>(
                "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}),
            parameter<int>("INIT",
                {.description = "initialization modus for fiber alignment", .default_value = 1}),
            parameter<bool>("ADAPT_ANGLE",
                {.description = "adapt angle during remodeling", .default_value = false}),
            parameter<double>("S", {.description = "maximum contractile stress"}),
            parameter<double>(
                "LAMBDAMAX", {.description = "stretch at maximum active force generation"}),
            parameter<double>(
                "LAMBDA0", {.description = "stretch at zero active force generation"}),
            parameter<double>(
                "DENS", {.description = "total reference mass density of constrained mixture"}),
        },
        {.description = "anisotropic active fiber"});
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with one exponential fiber family
  {
    known_materials[Core::Materials::mes_coupanisoexpo] = group("ELAST_CoupAnisoExpo",
        {
            parameter<double>("K1", {.description = "linear constant"}),
            parameter<double>("K2", {.description = "exponential constant"}),
            parameter<double>("GAMMA", {.description = "angle"}),
            parameter<double>("K1COMP", {.description = "linear constant"}),
            parameter<double>("K2COMP", {.description = "exponential constant"}),
            parameter<int>(
                "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}),
            parameter<int>("INIT",
                {.description = "initialization modus for fiber alignment", .default_value = 1}),
            parameter<bool>("ADAPT_ANGLE",
                {.description = "adapt angle during remodeling", .default_value = false}),
            parameter<int>("FIBER_ID",
                {.description = "Id of the fiber to be used (1 for first fiber, default)",
                    .default_value = 1}),
        },
        {.description = "anisotropic part with one exp. fiber"});
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with one exponential shear behavior between two fibers
  {
    known_materials[Core::Materials::mes_coupanisoexposhear] = group("ELAST_CoupAnisoExpoShear",
        {
            parameter<double>("K1", {.description = "linear constant"}),
            parameter<double>("K2", {.description = "exponential constant"}),
            parameter<double>("GAMMA", {.description = "angle"}),
            parameter<double>("K1COMP", {.description = "linear constant"}),
            parameter<double>("K2COMP", {.description = "exponential constant"}),
            parameter<int>("INIT",
                {.description = "initialization modus for fiber alignment", .default_value = 1}),
            parameter<std::vector<int>>(
                "FIBER_IDS", {.description = "Ids of the two fibers to be used (1 for the first "
                                             "fiber, 2 for the second, default)",
                                 .size = 2}),
        },
        {.description = "Exponential shear behavior between two fibers"});
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with one pow-like fiber family
  {
    known_materials[Core::Materials::mes_coupanisopow] = group("ELAST_CoupAnisoPow",
        {
            parameter<double>("K", {.description = "linear constant"}),
            parameter<double>("D1", {.description = "exponential constant for fiber invariant"}),
            parameter<double>("D2", {.description = "exponential constant for system"}),
            parameter<double>("ACTIVETHRES",
                {.description = "Deformation threshold for activating fibers. Default: 1.0 (off at "
                                "compression); If 0.0 (always active)",
                    .default_value = 1.0}),
            parameter<int>(
                "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}),
            parameter<int>(
                "FIBER", {.description = "Number of the fiber family contained in the element",
                             .default_value = 1}),
            parameter<double>("GAMMA", {.description = "angle", .default_value = 0.0}),
            parameter<int>("INIT",
                {.description = "initialization modus for fiber alignment", .default_value = 1}),
            parameter<bool>("ADAPT_ANGLE",
                {.description = "adapt angle during remodeling", .default_value = false}),
        },
        {.description = "anisotropic part with one pow-like fiber"});
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with two exponential fiber families
  {
    known_materials[Core::Materials::mes_coupanisoexpotwocoup] = group("ELAST_CoupAnisoExpoTwoCoup",
        {
            parameter<double>("A4", {.description = "linear anisotropic constant for fiber 1"}),
            parameter<double>(
                "B4", {.description = "exponential anisotropic constant for fiber 1"}),
            parameter<double>("A6", {.description = "linear anisotropic constant for fiber 2"}),
            parameter<double>(
                "B6", {.description = "exponential anisotropic constant for fiber 2"}),
            parameter<double>(
                "A8", {.description = "linear anisotropic constant for fiber 1 relating fiber 2"}),
            parameter<double>("B8",
                {.description = "exponential anisotropic constant for fiber 1 relating fiber 2"}),
            parameter<double>("GAMMA", {.description = "angle"}),
            parameter<int>(
                "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}),
            parameter<int>("INIT",
                {.description = "initialization modus for fiber alignment", .default_value = 1}),
            parameter<bool>(
                "FIB_COMP", {.description = "fibers support compression: yes (true) or no (false)",
                                .default_value = true}),
            parameter<bool>("ADAPT_ANGLE",
                {.description = "adapt angle during remodeling", .default_value = false}),
        },
        {.description = "anisotropic part with two exp. fibers"});
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with two exponential fiber families
  {
    known_materials[Core::Materials::mes_coupanisoneohooke] = group("ELAST_CoupAnisoNeoHooke",
        {
            parameter<double>("C", {.description = "linear constant"}),
            parameter<double>("GAMMA", {.description = "angle"}),
            parameter<int>(
                "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}),
            parameter<int>("INIT",
                {.description = "initialization modus for fiber alignment", .default_value = 1}),
            parameter<bool>("ADAPT_ANGLE",
                {.description = "adapt angle during remodeling", .default_value = false}),
        },
        {.description = "anisotropic part with one neo Hookean fiber"});
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with the stress given by a simplified version of the contraction
  // law of Bestel-Clement-Sorine
  {
    known_materials[Core::Materials::mes_anisoactivestress_evolution] =
        group("ELAST_AnisoActiveStress_Evolution",
            {
                parameter<double>("SIGMA", {.description = "Contractility (maximal stress)"}),
                parameter<double>("TAUC0", {.description = "Initial value for the active stress"}),
                parameter<double>(
                    "MAX_ACTIVATION", {.description = "Maximal value for the rescaled activation"}),
                parameter<double>(
                    "MIN_ACTIVATION", {.description = "Minimal value for the rescaled activation"}),
                parameter<int>("SOURCE_ACTIVATION",
                    {.description = "Where the activation comes from: 0=scatra , >0 Id for FUNCT"}),
                parameter<double>("ACTIVATION_THRES",
                    {.description = "Threshold for activation (contraction starts when activation "
                                    "function is larger than this value, relaxes otherwise)"}),
                parameter<bool>("STRAIN_DEPENDENCY",
                    {.description = "model strain dependency of contractility (Frank-Starling "
                                    "law): no (false) or yes (true)",
                        .default_value = false}),
                parameter<double>(
                    "LAMBDA_LOWER", {.description = "lower fiber stretch for Frank-Starling law",
                                        .default_value = 1.0}),
                parameter<double>(
                    "LAMBDA_UPPER", {.description = "upper fiber stretch for Frank-Starling law",
                                        .default_value = 1.0}),
                parameter<double>("GAMMA", {.description = "angle", .default_value = 0.0}),
                parameter<int>(
                    "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}),
                parameter<int>("INIT",
                    {.description = "initialization mode for fiber alignment", .default_value = 1}),
                parameter<bool>("ADAPT_ANGLE",
                    {.description = "adapt angle during remodeling", .default_value = false}),
            },
            {.description =
                    "anisotropic part with one fiber with coefficient given by a simplification of "
                    "the activation-contraction law of Bestel-Clement-Sorine-2001"});
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with variable stress coefficient
  {
    known_materials[Core::Materials::mes_coupanisoneohooke_varprop] = group(
        "ELAST_CoupAnisoNeoHooke_VarProp",
        {
            parameter<double>("C", {.description = "linear constant"}),
            parameter<int>("SOURCE_ACTIVATION",
                {.description = "Where the activation comes from: 0=scatra , >0 Id for FUNCT"}),
            parameter<double>("GAMMA", {.description = "azimuth angle", .default_value = 0.0}),
            parameter<double>("THETA", {.description = "polar angle", .default_value = 0.0}),
            parameter<int>(
                "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}),
            parameter<int>("INIT",
                {.description = "initialization mode for fiber alignment", .default_value = 1}),
            parameter<bool>("ADAPT_ANGLE",
                {.description = "adapt angle during remodeling", .default_value = false}),
        },
        {.description = "anisotropic part with one neo Hookean fiber with variable coefficient"});
  }

  /*--------------------------------------------------------------------*/
  // isochoric anisotropic material with one exponential fiber family
  {
    known_materials[Core::Materials::mes_isoanisoexpo] = group("ELAST_IsoAnisoExpo",
        {
            parameter<double>("K1", {.description = "linear constant"}),
            parameter<double>("K2", {.description = "exponential constant"}),
            parameter<double>("GAMMA", {.description = "angle"}),
            parameter<double>("K1COMP", {.description = "linear constant"}),
            parameter<double>("K2COMP", {.description = "exponential constant"}),
            parameter<int>(
                "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}),
            parameter<int>("INIT",
                {.description = "initialization modus for fiber alignment", .default_value = 1}),
            parameter<bool>("ADAPT_ANGLE",
                {.description = "adapt angle during remodeling", .default_value = false}),
        },
        {.description = "anisotropic part with one exp. fiber"});
  }

  /*--------------------------------------------------------------------*/
  // structural tensor
  {
    known_materials[Core::Materials::mes_structuraltensorstratgy] = group("ELAST_StructuralTensor",
        {
            parameter<std::string>("STRATEGY",
                {.description = "Strategy for evaluation of structural tensor: Standard (default), "
                                "ByDistributionFunction, DispersedTransverselyIsotropic"}),

            // choose between:
            // "none"
            // "Bingham"
            // "vonMisesFisher"
            //  rauch 10/17
            parameter<std::string>(
                "DISTR", {.description = "Type of distribution function around mean direction: "
                                         "none, Bingham, vonMisesFisher",
                             .default_value = "none"}),

            parameter<double>("C1",
                {.description = "constant 1 for distribution function", .default_value = 1.0}),
            parameter<double>("C2",
                {.description = "constant 2 for distribution function", .default_value = 0.0}),
            parameter<double>("C3",
                {.description = "constant 3 for distribution function", .default_value = 0.0}),
            parameter<double>("C4",
                {.description = "constant 4 for distribution function", .default_value = 1e16}),
        },
        {.description = "Structural tensor strategy in anisotropic materials"});
  }

  /*--------------------------------------------------------------------*/
  // transversely isotropic material
  {
    known_materials[Core::Materials::mes_couptransverselyisotropic] = group(
        "ELAST_CoupTransverselyIsotropic",
        {
            parameter<double>("ALPHA", {.description = "1-st constant"}),
            parameter<double>("BETA", {.description = "2-nd constant"}),
            parameter<double>("GAMMA", {.description = "3-rd constant"}),
            parameter<double>("ANGLE", {.description = "fiber angle"}),
            parameter<int>(
                "STR_TENS_ID", {.description = "MAT ID for definition of Structural Tensor"}),
            parameter<int>("FIBER", {.description = "exponential constant", .default_value = 1}),
            parameter<int>("INIT",
                {.description = "initialization modus for fiber alignment", .default_value = 1}),
        },
        {.description = "transversely part of a simple othotropic, transversely isotropic "
                        "hyperelastic constitutive equation"});
  }

  /*--------------------------------------------------------------------*/
  // coupled Varga material acc. to Holzapfel
  {
    known_materials[Core::Materials::mes_coupvarga] = group("ELAST_CoupVarga",
        {
            parameter<double>("MUE", {.description = "Shear modulus"}),
            parameter<double>("BETA", {.description = "'Anti-modulus'"}),
        },
        {.description = "Varga material acc. to Holzapfel"});
  }

  /*--------------------------------------------------------------------*/
  // isochoric Varga material acc. to Holzapfel
  {
    known_materials[Core::Materials::mes_isovarga] = group("ELAST_IsoVarga",
        {
            parameter<double>("MUE", {.description = "Shear modulus"}),
            parameter<double>("BETA", {.description = "'Anti-modulus'"}),
        },
        {.description = "Isochoric Varga material acc. to Holzapfel"});
  }

  /*--------------------------------------------------------------------*/
  // isotropic viscous contribution of myocardial matrix (chapelle12)
  {
    known_materials[Core::Materials::mes_coupmyocard] = group("VISCO_CoupMyocard",
        {
            parameter<double>("N", {.description = "material parameter"}),
        },
        {.description = "Isotropic viscous contribution of myocardial matrix"});
  }

  /*--------------------------------------------------------------------*/
  // isochoric rate dependent viscos material, modified from Pioletti,1997
  {
    known_materials[Core::Materials::mes_isoratedep] = group("VISCO_IsoRateDep",
        {
            parameter<double>("N", {.description = "material parameter"}),
        },
        {.description = "Isochoric rate dependent viscous material"});
  }

  /*--------------------------------------------------------------------*/
  // viscos contribution to visohyperelastic material according to SLS-Model
  {
    known_materials[Core::Materials::mes_genmax] = group("VISCO_GenMax",
        {
            parameter<double>("TAU", {.description = "relaxation parameter"}),
            parameter<double>("BETA", {.description = "emphasis of viscous to elastic part"}),
            parameter<std::string>(
                "SOLVE", {.description = "Solution of evolution equation via: OST (default) or "
                                         "CONVOL (convolution integral)"}),
        },
        {.description = "Viscous contribution according to SLS-Model"});
  }

  /*--------------------------------------------------------------------*/
  // viscos contribution to visohyperelastic material according to FSLS-Model
  {
    known_materials[Core::Materials::mes_fract] = group("VISCO_Fract",
        {
            parameter<double>("TAU", {.description = "relaxation parameter"}),
            parameter<double>("ALPHA", {.description = "fractional order derivative"}),
            parameter<double>("BETA", {.description = "emphasis of viscous to elastic part"}),
        },
        {.description = "Viscous contribution according to FSLS-Model"});
  }

  /*--------------------------------------------------------------------*/
  // viscous contribution of a branch of a generalized Maxwell model
  {
    known_materials[Core::Materials::mes_viscopart] = group("VISCO_PART",
        {
            parameter<double>("TAU",
                {.description = "dynamic viscosity divided by young's modulus of the branch"}),
        },
        {.description = "Viscous contribution of a viscoelastic Branch"});
  }
  /*--------------------------------------------------------------------*/
  // viscoelatic branches of a generalized Maxwell model
  {
    known_materials[Core::Materials::mes_generalizedgenmax] = group("VISCO_GeneralizedGenMax",
        {
            parameter<int>("NUMBRANCH", {.description = "number of viscoelastic branches"}),
            parameter<std::vector<int>>("MATIDS",
                {.description = "the list material IDs", .size = from_parameter<int>("NUMBRANCH")}),
            parameter<std::string>(
                "SOLVE", {.description = "Solution for evolution equation: OST (default) or CONVOL "
                                         "(convolution integral)"}),
        },
        {.description = "Viscoelastic Branches of generalized Maxwell"});
  }

  /*--------------------------------------------------------------------*/
  // description of a viscoelatic branch of a generalized Maxwell model
  {
    known_materials[Core::Materials::mes_viscobranch] = group("VISCO_BRANCH",
        {
            parameter<int>(
                "NUMMAT", {.description = "number of materials in the viscoelastic branch"}),
            parameter<std::vector<int>>("MATIDS",
                {.description = "the list material IDs", .size = from_parameter<int>("NUMMAT")}),
        },
        {.description = "Viscoelastic Branch (viscous and elastic contribution)"});
  }

  /*--------------------------------------------------------------------*/
  // 1D Artery material with constant properties
  {
    known_materials[Core::Materials::m_cnst_art] = group("MAT_CNST_ART",
        {
            parameter<double>("VISCOSITY",
                {.description =
                        "viscosity (for CONSTANT viscosity law taken as blood viscosity, for "
                        "BLOOD viscosity law taken as the viscosity of blood plasma)"}),
            parameter<double>("DENS", {.description = "density of blood"}),
            parameter<double>("YOUNG", {.description = "artery Youngs modulus of elasticity"}),
            parameter<double>("NUE", {.description = "Poissons ratio of artery fiber"}),
            parameter<double>("TH", {.description = "artery thickness"}),
            parameter<double>("PEXT1", {.description = "artery fixed external pressure 1"}),
            parameter<double>("PEXT2", {.description = "artery fixed external pressure 2"}),
            parameter<std::string>("VISCOSITYLAW",
                {.description = "type of viscosity law, CONSTANT (default) or BLOOD",
                    .default_value = "CONSTANT"}),
            parameter<double>("BLOOD_VISC_SCALE_DIAM_TO_MICRONS",
                {.description = "used to scale the diameter for blood viscosity law to microns if "
                                "your problem is not given in microns, e.g., if you use mms, set "
                                "this parameter to 1.0e3",
                    .default_value = 1.0}),
            parameter<std::string>("VARYING_DIAMETERLAW",
                {.description = "type of variable diameter law, CONSTANT (default) or BY_FUNCTION",
                    .default_value = "CONSTANT"}),
            parameter<int>("VARYING_DIAMETER_FUNCTION",
                {.description = "function for variable diameter law", .default_value = -1}),
            parameter<double>("COLLAPSE_THRESHOLD",
                {.description = "Collapse threshold for diameter (below this diameter element is "
                                "assumed to be collapsed with zero diameter and is not evaluated)",
                    .default_value = -1.0}),
        },
        {.description = "artery with constant properties"});
  }

  /*--------------------------------------------------------------------*/
  // Fourier's law for linear and possibly anisotropic heat transport
  {
    known_materials[Core::Materials::m_thermo_fourier] = group("MAT_Fourier",
        {
            parameter<double>("CAPA", {.description = "volumetric heat capacity"}),
            input_field<std::vector<double>>("CONDUCT",
                {.description = "entries in the thermal "
                                "conductivity tensor. Setting one value resembles a scalar "
                                "conductivity, 2 or "
                                "3 values a diagonal conductivity and 4 or 9 values the full "
                                "conductivity tensor in two and three dimensions respectively."}),
        },
        {.description = "anisotropic linear Fourier's law of heat conduction"});
  }

  /*----------------------------------------------------------------------*/
  // material for heat transport due to Fourier-type thermal conduction and the Soret effect
  {
    known_materials[Core::Materials::m_soret] = group("MAT_soret",
        {
            parameter<double>("CAPA", {.description = "volumetric heat capacity"}),
            input_field<std::vector<double>>("CONDUCT",
                {.description = "entries in the thermal "
                                "conductivity tensor. Setting one value resembles a scalar "
                                "conductivity, 2 or "
                                "3 values a diagonal conductivity and 4 or 9 values the full "
                                "conductivity tensor in two and three dimensions respectively."}),
            parameter<double>("SORET", {.description = "Soret coefficient"}),
        },
        {.description = "material for heat transport due to Fourier-type thermal conduction and "
                        "the Soret effect"});
  }

  /*----------------------------------------------------------------------*/
  // collection of hyperelastic materials for membranes
  {
    known_materials[Core::Materials::m_membrane_elasthyper] = group("MAT_Membrane_ElastHyper",
        {
            parameter<int>("NUMMAT", {.description = "number of materials/potentials in list"}),
            parameter<std::vector<int>>("MATIDS", {.description = "the list material/potential IDs",
                                                      .size = from_parameter<int>("NUMMAT")}),
            parameter<double>("DENS", {.description = "material mass density"}),
            parameter<int>("POLYCONVEX",
                {.description = "1.0 if polyconvexity of system is checked", .default_value = 0}),
        },
        {.description =
                "list/collection of hyperelastic materials for membranes, i.e. material IDs"});
  }

  /*----------------------------------------------------------------------*/
  // active strain membrane material for gastric electromechanics
  {
    known_materials[Core::Materials::m_membrane_activestrain] = group("MAT_Membrane_ActiveStrain",
        {
            parameter<int>("MATIDPASSIVE", {.description = "MATID for the passive material"}),
            parameter<int>("SCALIDVOLTAGE",
                {.description = "ID of the scalar that represents the (SMC) voltage"}),
            parameter<double>("DENS", {.description = "material mass density"}),
            parameter<double>("BETA1", {.description = "Ca2+ dynamics"}),
            parameter<double>("BETA2", {.description = "opening dynamics of the VDCC"}),
            parameter<double>("VOLTHRESH", {.description = "voltage threshold for activation"}),
            parameter<double>(
                "ALPHA1", {.description = "intensity of contraction in fiber direction 1"}),
            parameter<double>(
                "ALPHA2", {.description = "intensity of contraction in fiber direction 2"}),
        },
        {.description = "active strain membrane material"});
  }

  /*----------------------------------------------------------------------*/
  // growth and remodeling (homogenized constrained mixture model)
  {
    known_materials[Core::Materials::m_growthremodel_elasthyper] = group(
        "MAT_GrowthRemodel_ElastHyper",
        {
            parameter<int>("NUMMATRF", {.description = "number of remodelfiber materials in list"}),
            parameter<int>("NUMMATEL3D",
                {.description = "number of 3d elastin matrix materials/potentials in list",
                    .default_value = 0}),
            parameter<int>("NUMMATEL2D",
                {.description = "number of 2d elastin matrix materials/potentials in list"}),
            parameter<std::vector<int>>(
                "MATIDSRF", {.description = "the list remodelfiber material IDs",
                                .default_value = std::vector{0},
                                .size = from_parameter<int>("NUMMATRF")}),
            parameter<std::vector<int>>(
                "MATIDSEL3D", {.description = "the list 3d elastin matrix material/potential IDs",
                                  .default_value = std::vector{-1},
                                  .size = from_parameter<int>("NUMMATEL3D")}),
            parameter<std::vector<int>>(
                "MATIDSEL2D", {.description = "the list 2d elastin matrix material/potential IDs",
                                  .default_value = std::vector{0},
                                  .size = from_parameter<int>("NUMMATEL2D")}),
            parameter<int>(
                "MATIDELPENALTY", {.description = "penalty material ID", .default_value = -1}),
            parameter<double>("ELMASSFRAC",
                {.description = "initial mass fraction of elastin matrix in constraint mixture"}),
            parameter<double>("DENS", {.description = "material mass density"}),
            parameter<double>("PRESTRETCHELASTINCIR",
                {.description = "circumferential prestretch of elastin matrix"}),
            parameter<double>(
                "PRESTRETCHELASTINAX", {.description = "axial prestretch of elastin matrix"}),
            parameter<double>("THICKNESS",
                {.description =
                        "reference wall thickness of the idealized cylindrical aneurysm [m]",
                    .default_value = -1.}),
            parameter<double>(
                "MEANPRESSURE", {.description = "mean blood pressure [Pa]", .default_value = -1.0}),
            parameter<double>(
                "RADIUS", {.description = "inner radius of the idealized cylindrical aneurysm [m]",
                              .default_value = -1.0}),
            parameter<int>("DAMAGE",
                {.description = "1: elastin damage after prestressing,0: no elastin damage"}),
            parameter<int>(
                "GROWTHTYPE", {.description = "flag to decide what type of collagen growth is "
                                              "used: 1: anisotropic growth; 0: isotropic growth"}),
            parameter<int>("LOCTIMEINT",
                {.description = "flag to decide what type of local time integration scheme is "
                                "used: 1: Backward Euler Method; 0: Forward Euler Method"}),
            parameter<int>("MEMBRANE", {.description = "Flag whether Hex or Membrane elements are "
                                                       "used ( Membrane: 1, Hex: Everything else )",
                                           .default_value = -1}),
            parameter<int>(
                "CYLINDER", {.description = "Flag that geometry is a cylinder. 1: aligned in "
                                            "x-direction; 2: y-direction; 3: z-direction",
                                .default_value = -1}),
        },
        {.description = "growth and remodeling"});
  }

  /*----------------------------------------------------------------------*/
  // multiplicative split of deformation gradient in elastic and inelastic parts
  {
    known_materials[Core::Materials::m_multiplicative_split_defgrad_elasthyper] =
        group("MAT_MultiplicativeSplitDefgradElastHyper",
            {
                parameter<int>(
                    "NUMMATEL", {.description = "number of elastic materials/potentials in list"}),
                parameter<std::vector<int>>(
                    "MATIDSEL", {.description = "the list of elastic material/potential IDs",
                                    .default_value = std::vector{-1},
                                    .size = from_parameter<int>("NUMMATEL")}),
                parameter<int>("NUMFACINEL",
                    {.description = "number of factors of inelastic deformation gradient"}),
                parameter<std::vector<int>>("INELDEFGRADFACIDS",
                    {.description = "the list of inelastic deformation gradient factor IDs",
                        .default_value = std::vector{0},
                        .size = from_parameter<int>("NUMFACINEL")}),
                parameter<double>("DENS", {.description = "material mass density"}),
            },
            {.description = "multiplicative split of deformation gradient"});
  }

  /*----------------------------------------------------------------------*/
  // simple inelastic material law featuring no volume change
  {
    known_materials[Core::Materials::mfi_no_growth] = group("MAT_InelasticDefgradNoGrowth", {},
        {.description = "no volume change, i.e. the inelastic deformation gradient is the identity "
                        "tensor"});
  }

  /*----------------------------------------------------------------------*/
  // simple isotropic, volumetric growth; growth is linearly dependent on scalar mapped to material
  // configuration, constant material density
  {
    known_materials[Core::Materials::mfi_lin_scalar_iso] = group("MAT_InelasticDefgradLinScalarIso",
        {
            parameter<int>("SCALAR1", {.description = "number of growth inducing scalar"}),
            parameter<double>("SCALAR1_MolarGrowthFac",
                {.description = "isotropic molar growth factor due to scalar 1"}),
            parameter<double>("SCALAR1_RefConc",
                {.description = "reference concentration of scalar 1 causing no strains"}),
        },
        {.description = "scalar dependent isotropic growth law; volume change linearly dependent "
                        "on scalar (in material configuration)"});
  }

  /*----------------------------------------------------------------------*/
  // simple anisotropic, volumetric growth; growth direction prescribed in input-file;
  // growth is linearly dependent on scalar mapped to material configuration, constant material
  // density
  {
    known_materials[Core::Materials::mfi_lin_scalar_aniso] =
        group("MAT_InelasticDefgradLinScalarAniso",
            {
                parameter<int>("SCALAR1", {.description = "number of growth inducing scalar"}),
                parameter<double>("SCALAR1_MolarGrowthFac",
                    {.description = "anisotropic molar growth factor due to scalar 1"}),
                parameter<double>("SCALAR1_RefConc",
                    {.description = "reference concentration of scalar 1 causing no strains"}),
                parameter<int>(
                    "NUMSPACEDIM", {.description = "Number of space dimension (only 3 valid)"}),
                parameter<std::vector<double>>(
                    "GrowthDirection", {.description = "vector that defines the growth direction",
                                           .size = from_parameter<int>("NUMSPACEDIM")}),
            },
            {.description = "scalar dependent anisotropic growth law; growth in direction as given "
                            "in input-file; volume change linearly dependent on scalar (in "
                            "material configuration)"});
  }

  /*----------------------------------------------------------------------*/
  // non-linear isotropic volumetric growth; growth is dependent on the degree of lithiation,
  // constant material density, nonlinear behavior prescribed by polynomial in input file
  {
    known_materials[Core::Materials::mfi_poly_intercal_frac_iso] = group(
        "MAT_InelasticDefgradPolyIntercalFracIso",
        {
            parameter<int>("SCALAR1", {.description = "number of growth inducing scalar"}),
            parameter<double>("SCALAR1_RefConc",
                {.description = "reference concentration of scalar 1 causing no strains"}),
            parameter<int>("POLY_PARA_NUM", {.description = "number of polynomial coefficients"}),
            parameter<std::vector<double>>(
                "POLY_PARAMS", {.description = "coefficients of polynomial",
                                   .size = from_parameter<int>("POLY_PARA_NUM")}),
            parameter<double>("X_min", {.description = "lower bound of validity of polynomial"}),
            parameter<double>("X_max", {.description = "upper bound of validity of polynomial"}),
            parameter<int>(
                "MATID", {.description = "material ID of the corresponding scatra material"}),
        },
        {.description = "scalar dependent isotropic growth law; volume change nonlinearly "
                        "dependent on the intercalation fraction, that is calculated using the "
                        "scalar concentration (in material configuration)"});
  }

  /*----------------------------------------------------------------------*/
  // non-linear anisotropic volumetric growth; growth direction prescribed in input-file;
  // growth is dependent on the degree of lithiation, constant material density, nonlinear behavior
  // prescribed by polynomial in input file
  {
    known_materials[Core::Materials::mfi_poly_intercal_frac_aniso] = group(
        "MAT_InelasticDefgradPolyIntercalFracAniso",
        {
            parameter<int>("SCALAR1", {.description = "number of growth inducing scalar"}),
            parameter<double>("SCALAR1_RefConc",
                {.description = "reference concentration of scalar 1 causing no strains"}),
            parameter<int>(
                "NUMSPACEDIM", {.description = "Number of space dimension (only 3 valid)"}),
            parameter<std::vector<double>>(
                "GrowthDirection", {.description = "vector that defines the growth direction",
                                       .size = from_parameter<int>("NUMSPACEDIM")}),
            parameter<int>("POLY_PARA_NUM", {.description = "number of polynomial coefficients"}),
            parameter<std::vector<double>>(
                "POLY_PARAMS", {.description = "coefficients of polynomial",
                                   .size = from_parameter<int>("POLY_PARA_NUM")}),
            parameter<double>("X_min", {.description = "lower bound of validity of polynomial"}),
            parameter<double>("X_max", {.description = "upper bound of validity of polynomial"}),
            parameter<int>(
                "MATID", {.description = "material ID of the corresponding scatra material"}),
        },
        {.description =
                "scalar dependent anisotropic growth law; growth in direction as given in "
                "input-file; volume change nonlinearly dependent on the intercalation fraction, "
                "that is calculated using the scalar concentration (in material configuration)"});
  }

  /*----------------------------------------------------------------------*/
  {
    known_materials[Core::Materials::mfi_lin_temp_iso] = group("MAT_InelasticDefgradLinTempIso",
        {
            parameter<double>(
                "Temp_GrowthFac", {.description = "isotropic growth factor due to temperature"}),
            parameter<double>(
                "RefTemp", {.description = "reference temperature causing no strains"}),
        },
        {.description = "Temperature dependent growth law. Volume change linearly dependent on "
                        "temperature"});
  }

  /*----------------------------------------------------------------------*/
  {
    known_materials[Core::Materials::mfi_time_funct] = group("MAT_InelasticDefgradTimeFunct",
        {
            parameter<int>(
                "FUNCT_NUM", {.description = "Time-dependent function of the determinant "
                                             "of the inelastic deformation gradient"}),
        },
        {.description = "Time-dependent growth law. determinant of volume change dependent on time "
                        "function defined by 'FUNCT_NUM"});
  }

  /*----------------------------------------------------------------------*/
  {
    known_materials[Core::Materials::mfi_transv_isotrop_elast_viscoplast] = group(
        "MAT_InelasticDefgradTransvIsotropElastViscoplast",
        {
            parameter<int>("VISCOPLAST_LAW_ID",
                {.description = "MAT ID of the corresponding viscoplastic law"}),
            parameter<int>(
                "FIBER_READER_ID", {.description = "MAT ID of the used fiber direction reader for "
                                                   "transversely isotropic behavior"}),
            parameter<double>("YIELD_COND_A",
                {.description = "transversely isotropic version of the Hill(1948) yield condition: "
                                "parameter A, following the notation in Dafalias 1989, "
                                "International Journal of Plasticity, Vol. 5"}),
            parameter<double>("YIELD_COND_B",
                {.description = "transversely isotropic version of the Hill(1948) yield condition: "
                                "parameter B, following the notation in Dafalias 1989, "
                                "International Journal of Plasticity, Vol. 5"}),
            parameter<double>("YIELD_COND_F",
                {.description = "transversely isotropic version of the Hill(1948) yield condition: "
                                "parameter F, following the notation in Dafalias 1989, "
                                "International Journal of Plasticity, Vol. 5"}),
            parameter<std::string>("ANISOTROPY",
                {.description = "Anisotropy type: transversely isotropic (transvisotrop; "
                                "transverseisotropic; transverselyisotropic) | isotropic (isotrop; "
                                "isotropic; Default)"}),
            parameter<bool>("LOG_SUBSTEP",
                {.description = "boolean: time integration of internal variables using logarithmic "
                                "substepping (True) or standard substepping (False)?"}),
            parameter<int>("MAX_HALVE_NUM_SUBSTEP",
                {.description = "maximum number of times the global time step can be halved in the "
                                "substepping procedure"}),
        },
        {.description = "Versatile transversely isotropic (or isotropic) viscoplasticity model for "
                        "finite deformations with isotropic hardening, using user-defined "
                        "viscoplasticity laws (flow rule + hardening model)"});
  }

  /*----------------------------------------------------------------------*/
  {
    known_materials[Core::Materials::mvl_reformulated_Johnson_Cook] =
        group("MAT_ViscoplasticLawReformulatedJohnsonCook",
            {
                parameter<double>("STRAIN_RATE_PREFAC",
                    {.description = "reference plastic strain rate $\\dot{P}_0$ "}),
                parameter<double>("STRAIN_RATE_EXP_FAC",
                    {.description = "exponential factor of plastic strain rate $C$"}),
                parameter<double>("INIT_YIELD_STRENGTH",
                    {.description = "initial yield strength of the material $A_0$"}),
                parameter<double>("ISOTROP_HARDEN_PREFAC",
                    {.description = "prefactor of the isotropic hardening stress $B_0$"}),
                parameter<double>("ISOTROP_HARDEN_EXP",
                    {.description = "exponent of the isotropic hardening stress $n$"}),
            },
            {.description = "Reformulation of the Johnson-Cook viscoplastic law (comprising flow "
                            "rule $\\dot{P} = \\dot{P}_0 \\exp \\left( \\frac{ \\Sigma_{eq}}{C "
                            "\\Sigma_y} - \\frac{1}{C} \\right) - \\dot{P}_0$ and hardening law), "
                            "as shown in Mareau et al. (Mechanics of Materials 143, 2020)"});
  }

  /*----------------------------------------------------------------------*/
  // integration point based and scalar dependent interpolation between to materials
  {
    known_materials[Core::Materials::m_sc_dep_interp] = group("MAT_ScDepInterp",
        {
            parameter<int>("IDMATZEROSC", {.description = "material for lambda equal to zero"}),
            parameter<int>("IDMATUNITSC", {.description = "material for lambda equal to one"}),
        },
        {.description =
                "integration point based and scalar dependent interpolation between to materials"});
  }

  /*----------------------------------------------------------------------*/
  // growth and remodeling of arteries
  {
    known_materials[Core::Materials::m_constraintmixture] = group("MAT_ConstraintMixture",
        {
            parameter<double>("DENS", {.description = "Density"}),
            parameter<double>("MUE", {.description = "Shear Modulus"}),
            parameter<double>("NUE", {.description = "Poisson's ratio"}),
            parameter<double>("PHIE", {.description = "mass fraction of elastin"}),
            parameter<double>("PREELA", {.description = "prestretch of elastin"}),
            parameter<double>(
                "K1", {.description = "Parameter for linear collagen fiber stiffness"}),
            parameter<double>(
                "K2", {.description = "Parameter for exponential collagen fiber stiffness"}),
            parameter<int>("NUMHOM", {.description = "Number of homeostatic parameters"}),
            parameter<std::vector<double>>(
                "PRECOLL", {.description = "prestretch of collagen fibers",
                               .size = from_parameter<int>("NUMHOM")}),
            parameter<double>("DAMAGE", {.description = "damage stretch of collagen fibers"}),
            parameter<double>(
                "K1M", {.description = "Parameter for linear smooth muscle fiber stiffness"}),
            parameter<double>(
                "K2M", {.description = "Parameter for exponential smooth muscle fiber stiffness"}),
            parameter<double>("PHIM", {.description = "mass fraction of smooth muscle"}),
            parameter<double>("PREMUS", {.description = "prestretch of smooth muscle fibers"}),
            parameter<double>("SMAX", {.description = "maximal active stress"}),
            parameter<double>("KAPPA", {.description = "dilatation modulus"}),
            parameter<double>("LIFETIME", {.description = "lifetime of collagen fibers"}),
            parameter<double>("GROWTHFAC", {.description = "growth factor for stress"}),
            parameter<std::vector<double>>(
                "HOMSTR", {.description = "homeostatic target value of scalar stress measure",
                              .size = from_parameter<int>("NUMHOM")}),
            parameter<double>("SHEARGROWTHFAC", {.description = "growth factor for shear"}),
            parameter<double>(
                "HOMRAD", {.description = "homeostatic target value of inner radius"}),
            parameter<double>(
                "STARTTIME", {.description = "at this time turnover of collagen starts"}),
            parameter<std::string>("INTEGRATION",
                {.description = "time integration scheme: Explicit (default), or Implicit"}),
            parameter<double>("TOL",
                {.description =
                        "tolerance for local Newton iteration, only for implicit integration"}),
            parameter<std::string>("GROWTHFORCE",
                {.description = "driving force of growth: Single (default), All, ElaCol"}),
            parameter<std::string>("ELASTINDEGRAD",
                {.description = "how elastin is degraded: None (default), Rectangle, Time"}),
            parameter<std::string>("MASSPROD",
                {.description = "how mass depends on driving force: Lin (default), CosCos"}),
            parameter<std::string>("INITSTRETCH",
                {.description =
                        "how to set stretches in the beginning (None, Homeo, UpdatePrestretch)",
                    .default_value = "None"}),
            parameter<int>(
                "CURVE", {.description = "number of timecurve for increase of prestretch in time"}),
            parameter<std::string>("DEGOPTION",
                {.description = "Type of degradation function: Lin (default), Cos, Exp, ExpVar"}),
            parameter<double>(
                "MAXMASSPRODFAC", {.description = "maximal factor of mass production"}),
            parameter<double>(
                "ELASTINFAC", {.description = "factor for elastin content", .default_value = 0.0}),
            parameter<bool>("STOREHISTORY",
                {.description =
                        "store all history variables, not recommended for forward simulations",
                    .default_value = false}),
        },
        {.description = "growth and remodeling of arteries"});
  }

  /*----------------------------------------------------------------------*/
  // hyperelastic material for poroelasticity
  {
    known_materials[Core::Materials::m_structporo] = group("MAT_StructPoro",
        {
            parameter<int>("MATID", {.description = "ID of structure material"}),
            parameter<int>("POROLAWID", {.description = "ID of porosity law"}),
            parameter<double>("INITPOROSITY", {.description = "initial porosity of porous medium"}),
        },
        {.description = "wrapper for structure poroelastic material"});
  }
  /*----------------------------------------------------------------------*/
  // linear law for porosity in porous media problems
  {
    known_materials[Core::Materials::m_poro_law_linear] = group("MAT_PoroLawLinear",
        {
            parameter<double>("BULKMODULUS", {.description = "bulk modulus of porous medium"}),
        },
        {.description = "linear constitutive law for porosity"});
  }
  /*----------------------------------------------------------------------*/
  // constant law for porosity in porous media problems
  {
    known_materials[Core::Materials::m_poro_law_constant] =
        group("MAT_PoroLawConstant", {}, {.description = "constant constitutive law for porosity"});
  }
  /*----------------------------------------------------------------------*/
  // neo-hookean law for porosity in porous media problems
  {
    known_materials[Core::Materials::m_poro_law_logNeoHooke_Penalty] = group("MAT_PoroLawNeoHooke",
        {
            parameter<double>("BULKMODULUS", {.description = "bulk modulus of porous medium"}),
            parameter<double>(
                "PENALTYPARAMETER", {.description = "penalty parameter of porous medium"}),
        },
        {.description = "NeoHookean-like constitutive law for porosity"});
  }
  /*----------------------------------------------------------------------*/
  // incompressible skeleton law for porosity in porous media problems
  {
    known_materials[Core::Materials::m_poro_law_incompr_skeleton] = group("MAT_PoroLawIncompSkel",
        {}, {.description = "porosity law for incompressible skeleton phase"});
  }

  /*----------------------------------------------------------------------*/
  // incompressible skeleton law for porosity in porous media problems
  {
    known_materials[Core::Materials::m_poro_law_linear_biot] = group("MAT_PoroLawLinBiot",
        {
            parameter<double>(
                "INVBIOTMODULUS", {.description = "inverse Biot modulus of porous medium"}),
            parameter<double>("BIOTCEOFF", {.description = "Biot coefficient of porous medium"}),
        },
        {.description = "linear biot model for porosity law"});
  }

  /*----------------------------------------------------------------------*/
  // incompressible skeleton law for porosity depending on the density
  {
    known_materials[Core::Materials::m_poro_law_density_dependent] =
        group("MAT_PoroLawDensityDependent",
            {
                parameter<int>("DENSITYLAWID", {.description = "material ID of density law"}),
            },
            {.description = "porosity depending on the density"});
  }

  /*----------------------------------------------------------------------*/
  // density law for constant density in porous multiphase medium
  {
    known_materials[Core::Materials::m_poro_densitylaw_constant] =
        group("MAT_PoroDensityLawConstant", {},
            {.description = "density law for constant density in porous multiphase medium"});
  }

  /*----------------------------------------------------------------------*/
  // density law for constant density in porous multiphase medium
  {
    known_materials[Core::Materials::m_poro_densitylaw_exp] = group("MAT_PoroDensityLawExp",
        {
            parameter<double>("BULKMODULUS", {.description = "bulk modulus of porous medium"}),
        },
        {.description = "density law for pressure dependent exponential function"});
  }

  /*----------------------------------------------------------------------*/
  // permeability law for constant permeability in porous multiphase medium
  {
    known_materials[Core::Materials::m_fluidporo_relpermeabilitylaw_constant] = group(
        "MAT_FluidPoroRelPermeabilityLawConstant",
        {
            parameter<double>("VALUE", {.description = "constant value of permeability"}),
        },
        {.description = "permeability law for constant permeability in porous multiphase medium"});
  }

  /*----------------------------------------------------------------------*/
  // permeability law for permeability depending on saturation according to (saturation)^exp
  // in porous multiphase medium
  {
    known_materials[Core::Materials::m_fluidporo_relpermeabilitylaw_exp] = group(
        "MAT_FluidPoroRelPermeabilityLawExp",
        {
            parameter<double>("EXP", {.description = "exponent of the saturation of this phase"}),
            parameter<double>(
                "MIN_SAT", {.description = "minimum saturation which is used for calculation"}),
        },
        {.description = "permeability law depending on saturation in porous multiphase medium"});
  }

  /*----------------------------------------------------------------------*/
  // viscosity law for constant viscosity in porous multiphase medium
  {
    known_materials[Core::Materials::m_fluidporo_viscositylaw_constant] =
        group("MAT_FluidPoroViscosityLawConstant",
            {
                parameter<double>("VALUE", {.description = "constant value of viscosity"}),
            },
            {.description = "viscosity law for constant viscosity in porous multiphase medium"});
  }

  /*----------------------------------------------------------------------*/
  // viscosity law for viscosity-dependency modelling cell adherence
  {
    known_materials[Core::Materials::m_fluidporo_viscositylaw_celladh] = group(
        "MAT_FluidPoroViscosityLawCellAdherence",
        {
            parameter<double>(
                "VISC_0", {.description = "Visc0 parameter for modelling cell adherence"}),
            parameter<double>("XI", {.description = "xi parameter for modelling cell adherence"}),
            parameter<double>("PSI", {.description = "psi parameter for modelling cell adherence"}),
        },
        {.description = "visosity law depending on pressure gradient in porous multiphase medium"});
  }

  /*----------------------------------------------------------------------*/
  // hyperelastic material for poroelasticity with reaction
  {
    known_materials[Core::Materials::m_structpororeaction] = group("MAT_StructPoroReaction",
        {
            parameter<int>("MATID", {.description = "ID of structure material"}),
            parameter<int>("POROLAWID", {.description = "ID of porosity law"}),
            parameter<double>("INITPOROSITY", {.description = "initial porosity of porous medium"}),
            parameter<int>("DOFIDREACSCALAR",
                {.description =
                        "Id of DOF within scalar transport problem, which controls the reaction"}),
        },
        {.description = "wrapper for structure porelastic material with reaction"});
  }

  /*----------------------------------------------------------------------*/
  // hyperelastic material for poroelasticity with reaction
  {
    known_materials[Core::Materials::m_structpororeactionECM] = group("MAT_StructPoroReactionECM",
        {
            parameter<int>("MATID", {.description = "ID of structure material"}),
            parameter<int>("POROLAWID", {.description = "ID of porosity law"}),
            parameter<double>("INITPOROSITY", {.description = "initial porosity of porous medium"}),
            parameter<double>("DENSCOLLAGEN", {.description = "density of collagen"}),
            parameter<int>("DOFIDREACSCALAR",
                {.description =
                        "Id of DOF within scalar transport problem, which controls the reaction"}),
        },
        {.description = "wrapper for structure porelastic material with reaction"});
  }

  /*----------------------------------------------------------------------*/
  // fluid flow in a poroelastic material
  {
    known_materials[Core::Materials::m_fluidporo] = group("MAT_FluidPoro",
        {
            parameter<double>("DYNVISCOSITY", {.description = "dynamic viscosity"}),
            parameter<double>("DENSITY", {.description = "density"}),
            parameter<double>(
                "PERMEABILITY", {.description = "permeability of medium", .default_value = 0.0}),
            parameter<double>(
                "AXIALPERMEABILITY", {.description = "axial permeability for transverse isotropy",
                                         .default_value = 0.0}),
            parameter<double>("ORTHOPERMEABILITY1",
                {.description = "first permeability for orthotropy", .default_value = 0.0}),
            parameter<double>("ORTHOPERMEABILITY2",
                {.description = "second permeability for orthotropy", .default_value = 0.0}),
            parameter<double>("ORTHOPERMEABILITY3",
                {.description = "third permeability for orthotropy", .default_value = 0.0}),
            parameter<std::string>(
                "TYPE", {.description = "Problem type: Darcy (default) or Darcy-Brinkman",
                            .default_value = "Darcy"}),
            // optional parameter
            parameter<std::string>("PERMEABILITYFUNCTION",
                {.description = "Permeability function: Const(Default) or Kozeny_Carman",
                    .default_value = "Const"}),
        },
        {.description = "fluid flow in deformable porous media"});
  }

  /*----------------------------------------------------------------------*/
  // multiphase flow in a poroelastic material
  {
    known_materials[Core::Materials::m_fluidporo_multiphase] = group("MAT_FluidPoroMultiPhase",
        {
            parameter<bool>("LOCAL",
                {.description =
                        "individual materials allocated per element or only at global scope"}),
            parameter<double>("PERMEABILITY", {.description = "permeability of medium"}),
            parameter<int>("NUMMAT", {.description = "number of materials in list"}),
            parameter<std::vector<int>>("MATIDS",
                {.description = "the list material IDs", .size = from_parameter<int>("NUMMAT")}),
            parameter<int>(
                "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE", {.description = "number of fluid phases"}),
        },
        {.description = "multi phase flow in deformable porous media"});
  }

  /*----------------------------------------------------------------------*/
  // multiphase flow in a poroelastic material with reactions
  {
    known_materials[Core::Materials::m_fluidporo_multiphase_reactions] = group(
        "MAT_FluidPoroMultiPhaseReactions",
        {
            parameter<bool>("LOCAL",
                {.description =
                        "individual materials allocated per element or only at global scope"}),
            parameter<double>("PERMEABILITY", {.description = "permeability of medium"}),
            parameter<int>("NUMMAT", {.description = "number of materials in list"}),
            parameter<std::vector<int>>("MATIDS",
                {.description = "the list material IDs", .size = from_parameter<int>("NUMMAT")}),
            parameter<int>(
                "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE", {.description = "number of fluid phases"}),
            parameter<int>("NUMREAC", {.description = "number of reactions for these elements"}),
            parameter<std::vector<int>>("REACIDS", {.description = "advanced reaction list",
                                                       .default_value = std::vector{0},
                                                       .size = from_parameter<int>("NUMREAC")}),
        },
        {.description = "multi phase flow in deformable porous media and list of reactions"});
  }

  /*----------------------------------------------------------------------*/
  // one reaction for multiphase flow in a poroelastic material
  {
    known_materials[Core::Materials::m_fluidporo_singlereaction] = group(
        "MAT_FluidPoroSingleReaction",
        {
            parameter<int>(
                "NUMSCAL", {.description = "number of scalars coupled with this problem"}),
            parameter<int>("TOTALNUMDOF", {.description = "total number of multiphase-dofs"}),
            parameter<int>("NUMVOLFRAC", {.description = "number of volfracs"}),
            parameter<Mat::PAR::PoroFluidPressureBased::ClosingRelation>("VOLFRAC_CLOSING_RELATION",
                {.description = "type of closing relation for volume fraction material: "
                                "blood_lung, homogenized_vasculature_tumor (default)",
                    .default_value = Mat::PAR::PoroFluidPressureBased::ClosingRelation::
                        evolutionequation_homogenized_vasculature_tumor}),
            parameter<std::vector<int>>("SCALE", {.description = "advanced reaction list",
                                                     .size = from_parameter<int>("TOTALNUMDOF")}),
            parameter<PoroPressureBased::FluidporoReactionCoupling>("COUPLING",
                {.description = "type of coupling: scalar_by_function, no_coupling (default)",
                    .default_value = PoroPressureBased::FluidporoReactionCoupling::no_coupling}),
            parameter<int>("FUNCTID", {.description = "function ID defining the reaction"}),
        },
        {.description = "advanced reaction material"});
  }

  /*----------------------------------------------------------------------*/
  // one phase for multiphase flow in a poroelastic material
  {
    known_materials[Core::Materials::m_fluidporo_singlephase] = group("MAT_FluidPoroSinglePhase",
        {
            parameter<int>("DENSITYLAWID", {.description = "ID of density law"}),
            parameter<double>("DENSITY", {.description = "reference/initial density"}),
            parameter<int>(
                "RELPERMEABILITYLAWID", {.description = "ID of relative permeability law"}),
            parameter<int>("VISCOSITY_LAW_ID", {.description = "ID of viscosity law"}),
            parameter<int>("DOFTYPEID", {.description = "ID of dof definition"}),
        },
        {.description = "one phase for multiphase flow in deformable porous media"});
  }

  /*----------------------------------------------------------------------*/
  // one volume fraction for multiphase flow in a poroelastic material
  {
    known_materials[Core::Materials::m_fluidporo_singlevolfrac] =
        group("MAT_FluidPoroSingleVolFrac",
            {
                parameter<double>("DENSITY", {.description = "reference/initial density"}),
                parameter<double>("DIFFUSIVITY", {.description = "diffusivity of phase"}),
                parameter<bool>("AddScalarDependentFlux",
                    {.description = "Is there additional scalar dependent flux (yes) or (no)"}),
                parameter<int>("NUMSCAL", {.description = "Number of scalars", .default_value = 0}),
                parameter<std::vector<double>>("SCALARDIFFS",
                    {.description = "Diffusivities for additional scalar-dependent flux",
                        .default_value = std::vector<double>{},
                        .size = from_parameter<int>("NUMSCAL")}),
                parameter<std::optional<std::vector<double>>>(
                    "OMEGA_HALF", {.description = "Constant for receptor kinetic law",
                                      .size = from_parameter<int>("NUMSCAL")}),
            },
            {.description = "one phase for multiphase flow in deformable porous media"});
  }

  /*----------------------------------------------------------------------*/
  // one volume fraction pressure for multiphase flow in a poroelastic material
  {
    known_materials[Core::Materials::m_fluidporo_volfracpressure] =
        group("MAT_FluidPoroVolFracPressure",
            {
                parameter<double>("PERMEABILITY", {.description = "permeability of phase"}),
                parameter<int>("VISCOSITY_LAW_ID", {.description = "ID of viscosity law"}),
                parameter<double>(
                    "MIN_VOLFRAC", {.description = "Minimum volume fraction under which we assume "
                                                   "that VolfracPressure is zero",
                                       .default_value = 1.0e-3}),
            },
            {.description =
                    "one volume fraction pressure for multiphase flow in deformable porous media"});
  }

  /*----------------------------------------------------------------------*/
  // one volume fraction pressure material for vascular units in the lungs for multiphase flow in a
  // poroelastic material
  {
    known_materials[Core::Materials::m_fluidporo_volfrac_pressure_blood_lung] =
        group("MAT_FluidPoroVolFracPressureBloodLung",
            {
                parameter<double>("DENSITY", {.description = "density of phase"}),
                parameter<double>("PERMEABILITY", {.description = "permeability of phase"}),
                parameter<int>("VISCOSITY_LAW_ID", {.description = "ID of viscosity law"}),
                parameter<double>("INITIALVOLFRAC",
                    {.description = "Initial volume fraction (usually at end-expiration)"}),
                parameter<double>("SCALING_PARAMETER_DEFORMATION",
                    {.description = "scaling parameter for deformation dependency"}),
                parameter<std::optional<double>>("SCALING_PARAMETER_PRESSURE",
                    {.description = "scaling parameter for pressure dependency"}),

            },
            {.description = "one volume fraction pressure material for vascular units in the lungs "
                            "for multiphase flow in deformable porous media"});
  }

  /*----------------------------------------------------------------------*/
  // one degree of freedom for on single phase of a multiphase flow in a poroelastic material
  {
    known_materials[Core::Materials::m_fluidporo_phasedof_diffpressure] = group(
        "MAT_FluidPoroSinglePhaseDofDiffPressure",
        {
            parameter<int>("PHASELAWID", {.description = "ID of pressure-saturation law"}),
            parameter<int>("NUMDOF", {.description = "number of DoFs"}),
            parameter<std::vector<int>>(
                "PRESCOEFF", {.description = "pressure IDs for differential pressure",
                                 .default_value = std::vector{0},
                                 .size = from_parameter<int>("NUMDOF")}),
        },
        {.description = "one degrree of freedom for multiphase flow in deformable porous media"});
  }

  /*----------------------------------------------------------------------*/
  // one degree of freedom for on single phase of a multiphase flow in a poroelastic material
  {
    known_materials[Core::Materials::m_fluidporo_phasedof_pressure] = group(
        "MAT_FluidPoroSinglePhaseDofPressure",
        {
            parameter<int>("PHASELAWID", {.description = "ID of pressure-saturation law"}),
        },
        {.description = "one degrree of freedom for multiphase flow in deformable porous media"});
  }

  /*----------------------------------------------------------------------*/
  // one degree of freedom for on single phase of a multiphase flow in a poroelastic material
  {
    known_materials[Core::Materials::m_fluidporo_phasedof_saturation] = group(
        "MAT_FluidPoroSinglePhaseDofSaturation",
        {
            parameter<int>("PHASELAWID", {.description = "ID of pressure-saturation law"}),
        },
        {.description = "one degrree of freedom for multiphase flow in deformable porous media"});
  }

  /*----------------------------------------------------------------------*/
  // saturated law for pressure-saturation law in porous media problems
  {
    known_materials[Core::Materials::m_fluidporo_phaselaw_linear] = group("MAT_PhaseLawLinear",
        {
            parameter<double>("RELTENSION", {.description = "relative interface tensions"}),
            parameter<double>(
                "SATURATION_0", {.description = "saturation at zero differential pressure"}),
            parameter<int>("NUMDOF", {.description = "number of DoFs"}),
            parameter<std::vector<int>>(
                "PRESCOEFF", {.description = "Coefficients for pressure dependence",
                                 .default_value = std::vector{0},
                                 .size = from_parameter<int>("NUMDOF")}),
        },
        {.description = "saturated fluid phase of porous medium"});
  }

  /*----------------------------------------------------------------------*/
  // tangent law for pressure-saturation law in porous media multiphase problems
  {
    known_materials[Core::Materials::m_fluidporo_phaselaw_tangent] = group("MAT_PhaseLawTangent",
        {
            parameter<double>("RELTENSION", {.description = "relative interface tensions"}),
            parameter<double>("EXP", {.description = "exponent in pressure-saturation law"}),
            parameter<double>(
                "SATURATION_0", {.description = "saturation at zero differential pressure"}),
            parameter<int>("NUMDOF", {.description = "number of DoFs"}),
            parameter<std::vector<int>>(
                "PRESCOEFF", {.description = "Coefficients for pressure dependence",
                                 .default_value = std::vector{0},
                                 .size = from_parameter<int>("NUMDOF")}),
        },
        {.description = "tangent fluid phase of porous medium"});
  }

  /*----------------------------------------------------------------------*/
  // constraint law for pressure-saturation law in porous media multiphase problems
  {
    known_materials[Core::Materials::m_fluidporo_phaselaw_constraint] = group(
        "MAT_PhaseLawConstraint", {}, {.description = "constraint fluid phase of porous medium"});
  }

  /*----------------------------------------------------------------------*/
  // pressure-saturation law defined by functions in porous media multiphase problems
  {
    known_materials[Core::Materials::m_fluidporo_phaselaw_byfunction] =
        group("MAT_PhaseLawByFunction",
            {
                parameter<int>(
                    "FUNCTPRES", {.description = "ID of function for differential pressure"}),
                parameter<int>("FUNCTSAT", {.description = "ID of function for saturation"}),
                parameter<int>("NUMDOF", {.description = "number of DoFs"}),
                parameter<std::vector<int>>(
                    "PRESCOEFF", {.description = "Coefficients for pressure dependence",
                                     .default_value = std::vector{0},
                                     .size = from_parameter<int>("NUMDOF")}),
            },
            {.description = "fluid phase of porous medium defined by functions"});
  }

  /*----------------------------------------------------------------------*/
  // elastic spring
  {
    known_materials[Core::Materials::m_spring] = group("MAT_Struct_Spring",
        {
            parameter<double>("STIFFNESS", {.description = "spring constant"}),
            parameter<double>("DENS", {.description = "density"}),
        },
        {.description = "elastic spring"});
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
    known_materials[Core::Materials::m_beam_reissner_elast_hyper] = group(
        "MAT_BeamReissnerElastHyper",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus"}),

            /* note: we define both of the two following (redundant) parameters to be optional.
             *       upon initialization of the material, we assure that one of them is
             *       properly defined. */
            parameter<double>("SHEARMOD", {.description = "shear modulus", .default_value = -1.0}),
            parameter<double>(
                "POISSONRATIO", {.description = "Poisson's ratio", .default_value = -1.0}),

            parameter<double>("DENS", {.description = "mass density"}),

            parameter<double>("CROSSAREA", {.description = "cross-section area"}),
            parameter<double>("SHEARCORR", {.description = "shear correction factor"}),

            parameter<double>("MOMINPOL", {.description = "polar/axial area moment of inertia"}),
            parameter<double>(
                "MOMIN2", {.description = "area moment of inertia w.r.t. first principal axis of "
                                          "inertia (i.e. second base vector)"}),
            parameter<double>(
                "MOMIN3", {.description = "area moment of inertia w.r.t. second principal axis of "
                                          "inertia (i.e. third base vector)"}),
            parameter<bool>("FAD", {.description = "Does automatic differentiation have to be used",
                                       .default_value = false}),


            /* The following is optional because it is only required if we evaluate interactions
             * between beams such as contact, potential-based and whatever more to come.
             * For now, we always assume a circular cross-section if interactions are considered.
             *
             * This should be generalized to a type of cross-section shape (circular, rectangular,
             * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed.
             */
            parameter<double>("INTERACTIONRADIUS",
                {.description = "radius of a circular cross-section which is EXCLUSIVELY used to "
                                "evaluate interactions such as contact, potentials, ...",
                    .default_value = -1.0}),
        },
        {.description =
                "material parameters for a Simo-Reissner type beam element based on hyperelastic "
                "stored energy function"});
  }
  /*--------------------------------------------------------------------*/
  // material parameter definition for a Simo-Reissner type elasto-plastic beam element
  {
    known_materials[Core::Materials::m_beam_reissner_elast_plastic] = group(
        "MAT_BeamReissnerElastPlastic",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus"}),

            // optional parameters for plasticity
            parameter<double>(
                "YIELDN", {.description = "initial yield stress N", .default_value = -1.0}),
            parameter<double>(
                "YIELDM", {.description = "initial yield stress M", .default_value = -1.0}),
            parameter<double>("ISOHARDN",
                {.description = "isotropic hardening modulus of forces", .default_value = -1.0}),
            parameter<double>("ISOHARDM",
                {.description = "isotropic hardening modulus of moments", .default_value = -1.0}),
            parameter<bool>("TORSIONPLAST",
                {.description = "defines whether torsional moment contributes to plasticity",
                    .default_value = false}),

            /* note: we define both of the two following (redundant) parameters to be optional.
             *       upon initialization of the material, we assure that one of them is
             *       properly defined. */
            parameter<double>("SHEARMOD", {.description = "shear modulus", .default_value = -1.0}),
            parameter<double>(
                "POISSONRATIO", {.description = "Poisson's ratio", .default_value = -1.0}),

            parameter<double>("DENS", {.description = "mass density"}),

            parameter<double>("CROSSAREA", {.description = "cross-section area"}),
            parameter<double>("SHEARCORR", {.description = "shear correction factor"}),

            parameter<double>("MOMINPOL", {.description = "polar/axial area moment of inertia"}),
            parameter<double>(
                "MOMIN2", {.description = "area moment of inertia w.r.t. first principal axis of "
                                          "inertia (i.e. second base vector)"}),
            parameter<double>(
                "MOMIN3", {.description = "area moment of inertia w.r.t. second principal axis of "
                                          "inertia (i.e. third base vector)"}),
            parameter<bool>("FAD", {.description = "Does automatic differentiation have to be used",
                                       .default_value = false}),


            /* The following is optional because it is only required if we evaluate interactions
             * between beams such as contact, potential-based and whatever more to come.
             * For now, we always assume a circular cross-section if interactions are considered.
             *
             * This should be generalized to a type of cross-section shape (circular, rectangular,
             * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed.
             */
            parameter<double>("INTERACTIONRADIUS",
                {.description = "radius of a circular cross-section which is EXCLUSIVELY used to "
                                "evaluate interactions such as contact, potentials, ...",
                    .default_value = -1.0}),
        },
        {.description =
                "material parameters for a Simo-Reissner type beam element based on hyperelastic "
                "stored energy function"});
  }
  /*--------------------------------------------------------------------*/
  // material parameter definition for a Simo-Reissner type beam element,
  // specified via 'modal' constitutive parameters (see comment above)
  {
    known_materials[Core::Materials::m_beam_reissner_elast_hyper_bymodes] = group(
        "MAT_BeamReissnerElastHyper_ByModes",
        {
            parameter<double>("EA", {.description = "axial rigidity"}),
            parameter<double>(
                "GA2", {.description = "shear rigidity w.r.t first principal axis of inertia"}),
            parameter<double>(
                "GA3", {.description = "shear rigidity w.r.t second principal axis of inertia"}),

            parameter<double>("GI_T", {.description = "torsional rigidity"}),
            parameter<double>(
                "EI2", {.description =
                               "flexural/bending rigidity w.r.t. first principal axis of inertia"}),
            parameter<double>("EI3",
                {.description =
                        "flexural/bending rigidity w.r.t. second principal axis of inertia"}),

            parameter<double>("RhoA",
                {.description = "translational inertia: mass density * cross-section area"}),

            parameter<double>("MASSMOMINPOL",
                {.description =
                        "polar mass moment of inertia, i.e. w.r.t. rotation around beam axis"}),
            parameter<double>("MASSMOMIN2",
                {.description = "mass moment of inertia w.r.t. first principal axis of inertia"}),
            parameter<double>("MASSMOMIN3",
                {.description = "mass moment of inertia w.r.t. second principal axis of inertia"}),
            parameter<bool>("FAD", {.description = "Does automatic differentiation have to be used",
                                       .default_value = false}),


            /* The following is optional because it is only required if we evaluate interactions
             * between beams such as contact, potential-based and whatever more to come.
             * For now, we always assume a circular cross-section if interactions are considered.
             *
             * This should be generalized to a type of cross-section shape (circular, rectangular,
             * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed.
             */
            parameter<double>("INTERACTIONRADIUS",
                {.description = "radius of a circular cross-section which is EXCLUSIVELY used to "
                                "evaluate interactions such as contact, potentials, ...",
                    .default_value = -1.0}),
        },
        {.description =
                "material parameters for a Simo-Reissner type beam element based on hyperelastic "
                "stored energy function, specified for individual deformation modes"});
  }

  /*--------------------------------------------------------------------*/
  // material parameter definition for a Kirchhoff-Love type beam element
  {
    known_materials[Core::Materials::m_beam_kirchhoff_elast_hyper] = group(
        "MAT_BeamKirchhoffElastHyper",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus"}),

            /* note: we define both of the two following (redundant) parameters to be optional.
             *       upon initialization of the material, we assure that one of them is
             *       properly defined. */
            parameter<double>("SHEARMOD", {.description = "shear modulus", .default_value = -1.0}),
            parameter<double>(
                "POISSONRATIO", {.description = "Poisson's ratio", .default_value = -1.0}),

            parameter<double>("DENS", {.description = "mass density"}),

            parameter<double>("CROSSAREA", {.description = "cross-section area"}),

            parameter<double>("MOMINPOL", {.description = "polar/axial area moment of inertia"}),
            parameter<double>(
                "MOMIN2", {.description = "area moment of inertia w.r.t. first principal axis of "
                                          "inertia (i.e. second base vector)"}),
            parameter<double>(
                "MOMIN3", {.description = "area moment of inertia w.r.t. second principal axis of "
                                          "inertia (i.e. third base vector)"}),
            parameter<bool>("FAD", {.description = "Does automatic differentiation have to be used",
                                       .default_value = false}),


            /* The following is optional because it is only required if we evaluate interactions
             * between beams such as contact, potential-based and whatever more to come.
             * For now, we always assume a circular cross-section if interactions are considered.
             *
             * This should be generalized to a type of cross-section shape (circular, rectangular,
             * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed.
             */
            parameter<double>("INTERACTIONRADIUS",
                {.description = "radius of a circular cross-section which is EXCLUSIVELY used to "
                                "evaluate interactions such as contact, potentials, ...",
                    .default_value = -1.0}),
        },
        {.description =
                "material parameters for a Kirchhoff-Love type beam element based on hyperelastic "
                "stored energy function"});
  }

  /*--------------------------------------------------------------------*/
  // material parameter definition for a Kirchhoff-Love type beam element,
  // specified via 'modal' constitutive parameters (see comment above)
  {
    known_materials[Core::Materials::m_beam_kirchhoff_elast_hyper_bymodes] = group(
        "MAT_BeamKirchhoffElastHyper_ByModes",
        {
            parameter<double>("EA", {.description = "axial rigidity"}),

            parameter<double>("GI_T", {.description = "torsional rigidity"}),
            parameter<double>(
                "EI2", {.description =
                               "flexural/bending rigidity w.r.t. first principal axis of inertia"}),
            parameter<double>("EI3",
                {.description =
                        "flexural/bending rigidity w.r.t. second principal axis of inertia"}),

            parameter<double>("RhoA",
                {.description = "translational inertia: mass density * cross-section area"}),

            parameter<double>("MASSMOMINPOL",
                {.description =
                        "polar mass moment of inertia, i.e. w.r.t. rotation around beam axis"}),
            parameter<double>("MASSMOMIN2",
                {.description = "mass moment of inertia w.r.t. first principal axis of inertia"}),
            parameter<double>("MASSMOMIN3",
                {.description = "mass moment of inertia w.r.t. second principal axis of inertia"}),
            parameter<bool>("FAD", {.description = "Does automatic differentiation have to be used",
                                       .default_value = false}),


            /* The following is optional because it is only required if we evaluate interactions
             * between beams such as contact, potential-based and whatever more to come.
             * For now, we always assume a circular cross-section if interactions are considered.
             *
             * This should be generalized to a type of cross-section shape (circular, rectangular,
             * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed.
             */
            parameter<double>("INTERACTIONRADIUS",
                {.description = "radius of a circular cross-section which is EXCLUSIVELY used to "
                                "evaluate interactions such as contact, potentials, ...",
                    .default_value = -1.0}),
        },
        {.description =
                "material parameters for a Kirchhoff-Love type beam element based on hyperelastic "
                "stored energy function, specified for individual deformation modes"});
  }

  /*--------------------------------------------------------------------*/
  // material parameter definition for a torsion-free, isotropic
  // Kirchhoff-Love type beam element
  {
    known_materials[Core::Materials::m_beam_kirchhoff_torsionfree_elast_hyper] = group(
        "MAT_BeamKirchhoffTorsionFreeElastHyper",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus"}),

            parameter<double>("DENS", {.description = "mass density"}),

            parameter<double>("CROSSAREA", {.description = "cross-section area"}),

            parameter<double>("MOMIN", {.description = "area moment of inertia"}),
            parameter<bool>("FAD", {.description = "Does automatic differentiation have to be used",
                                       .default_value = false}),


            /* The following is optional because it is only required if we evaluate interactions
             * between beams such as contact, potential-based and whatever more to come.
             * For now, we always assume a circular cross-section if interactions are considered.
             *
             * This should be generalized to a type of cross-section shape (circular, rectangular,
             * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed.
             */
            parameter<double>("INTERACTIONRADIUS",
                {.description = "radius of a circular cross-section which is EXCLUSIVELY used to "
                                "evaluate interactions such as contact, potentials, ...",
                    .default_value = -1.0}),
        },
        {.description = "material parameters for a torsion-free, isotropic Kirchhoff-Love type "
                        "beam element "
                        "based on hyperelastic stored energy function"});
  }

  /*--------------------------------------------------------------------*/
  // material parameter definition for a torsion-free, isotropic
  // Kirchhoff-Love type beam element,
  // specified via 'modal' constitutive parameters (see comment above)
  {
    known_materials[Core::Materials::m_beam_kirchhoff_torsionfree_elast_hyper_bymodes] = group(
        "MAT_BeamKirchhoffTorsionFreeElastHyper_ByModes",
        {
            parameter<double>("EA", {.description = "axial rigidity"}),

            parameter<double>("EI", {.description = "flexural/bending rigidity"}),


            parameter<double>("RhoA",
                {.description = "translational inertia: mass density * cross-section area"}),
            parameter<bool>("FAD", {.description = "Does automatic differentiation have to be used",
                                       .default_value = false}),

            /* The following is optional because it is only required if we evaluate interactions
             * between beams such as contact, potential-based and whatever more to come.
             * For now, we always assume a circular cross-section if interactions are considered.
             *
             * This should be generalized to a type of cross-section shape (circular, rectangular,
             * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if needed.
             */
            parameter<double>("INTERACTIONRADIUS",
                {.description = "radius of a circular cross-section which is EXCLUSIVELY used to "
                                "evaluate interactions such as contact, potentials, ...",
                    .default_value = -1.0}),
        },
        {.description = "material parameters for a torsion-free, isotropic Kirchhoff-Love type "
                        "beam element based on hyperelastic stored energy function, specified for "
                        "individual deformation modes"});
  }

  /*----------------------------------------------------------------------*/
  // material for an elastic Kirchhoff-Love shell
  {
    known_materials[Core::Materials::m_shell_kirchhoff_love] = group("MAT_Kirchhoff_Love_shell",
        {
            parameter<double>("YOUNG_MODULUS", {.description = "Young's modulus"}),
            parameter<double>("POISSON_RATIO", {.description = "Poisson's ratio"}),
            parameter<double>("THICKNESS", {.description = "Thickness of the shell"}),
        },
        {.description = "Material for an elastic Kichhhoff-Love shell "});
  }

  /*--------------------------------------------------------------------*/
  // material for a crosslinker in a biopolymer simulation
  {
    known_materials[Core::Materials::m_crosslinkermat] = group("MAT_Crosslinker",
        {
            parameter<double>("MATNUM", {.description = "number of beam elasthyper material"}),
            parameter<std::string>("JOINTTYPE",
                {.description =
                        "type of joint: beam3rline2rigid (default), beam3rline2pin or truss"}),
            parameter<double>("LINKINGLENGTH",
                {.description = "distance between the two binding domains of a linker"}),
            parameter<double>("LINKINGLENGTHTOL",
                {.description = "tolerance for linker length in the sense: length +- tolerance"}),
            parameter<double>("LINKINGANGLE",
                {.description =
                        "preferred binding angle enclosed by two filaments' axes in radians"}),
            parameter<double>(
                "LINKINGANGLETOL", {.description = "tolerance for preferred binding angle in "
                                                   "radians in the sense of: angle +- tolerance"}),
            parameter<double>("K_ON", {.description = "chemical association-rate"}),
            parameter<double>("K_OFF", {.description = "chemical dissociation-rate"}),
            parameter<double>("DELTABELLEQ",
                {.description = "deltaD in Bell's equation for force dependent off rate",
                    .default_value = 0.0}),
            parameter<double>("NOBONDDISTSPHERE",
                {.description =
                        "distance to sphere elements in which no double bonded linker is allowed",
                    .default_value = 0.0}),
            parameter<std::string>("TYPE",
                {.description =
                        "type of crosslinker: arbitrary (default), actin, collagen, integrin",
                    .default_value = "arbitrary"}),
        },
        {.description = "material for a linkage between beams"});
  }

  /*--------------------------------------------------------------------*/
  // 0D Acinar material base
  {
    known_materials[Core::Materials::m_0d_maxwell_acinus] = group("MAT_0D_MAXWELL_ACINUS",
        {
            parameter<double>("Stiffness1", {.description = "first stiffness"}),
            parameter<double>("Stiffness2", {.description = "second stiffness"}),
            parameter<double>("Viscosity1", {.description = "first viscosity"}),
            parameter<double>("Viscosity2", {.description = "second viscosity"}),
        },
        {.description = "0D acinar material"});
  }

  /*--------------------------------------------------------------------*/
  // 0D NeoHookean Acinar material
  {
    known_materials[Core::Materials::m_0d_maxwell_acinus_neohookean] =
        group("MAT_0D_MAXWELL_ACINUS_NEOHOOKEAN",
            {
                parameter<double>("Stiffness1", {.description = "first stiffness"}),
                parameter<double>("Stiffness2", {.description = "second stiffness"}),
                parameter<double>("Viscosity1", {.description = "first viscosity"}),
                parameter<double>("Viscosity2", {.description = "second viscosity"}),
            },
            {.description = "0D acinar material neohookean"});
  }

  /*--------------------------------------------------------------------*/
  // 0D Exponential Acinar material
  {
    known_materials[Core::Materials::m_0d_maxwell_acinus_exponential] =
        group("MAT_0D_MAXWELL_ACINUS_EXPONENTIAL",
            {
                parameter<double>("Stiffness1", {.description = "first stiffness"}),
                parameter<double>("Stiffness2", {.description = "second stiffness"}),
                parameter<double>("Viscosity1", {.description = "first viscosity"}),
                parameter<double>("Viscosity2", {.description = "second viscosity"}),
            },
            {.description = "0D acinar material exponential"});
  }

  /*--------------------------------------------------------------------*/
  // 0D Exponential Acinar material
  {
    known_materials[Core::Materials::m_0d_maxwell_acinus_doubleexponential] =
        group("MAT_0D_MAXWELL_ACINUS_DOUBLEEXPONENTIAL",
            {
                parameter<double>("Stiffness1", {.description = "first stiffness"}),
                parameter<double>("Stiffness2", {.description = "second stiffness"}),
                parameter<double>("Viscosity1", {.description = "first viscosity"}),
                parameter<double>("Viscosity2", {.description = "second viscosity"}),
            },
            {.description = "0D acinar material doubleexponential"});
  }

  /*--------------------------------------------------------------------*/
  // 0D Ogden Acinar material
  {
    known_materials[Core::Materials::m_0d_maxwell_acinus_ogden] =
        group("MAT_0D_MAXWELL_ACINUS_OGDEN",
            {
                parameter<double>("Stiffness1", {.description = "first stiffness"}),
                parameter<double>("Stiffness2", {.description = "second stiffness"}),
                parameter<double>("Viscosity1", {.description = "first viscosity"}),
                parameter<double>("Viscosity2", {.description = "second viscosity"}),
            },
            {.description = "0D acinar material ogden"});
  }


  /*----------------------------------------------------------------------*/
  // particle material sph fluid
  {
    known_materials[Core::Materials::m_particle_sph_fluid] = group("MAT_ParticleSPHFluid",
        {
            parameter<double>("INITRADIUS", {.description = "initial radius"}),
            parameter<double>("INITDENSITY", {.description = "initial density"}),
            parameter<double>(
                "REFDENSFAC", {.description = "reference density factor in equation of state"}),
            parameter<double>("EXPONENT", {.description = "exponent in equation of state"}),
            parameter<double>("BACKGROUNDPRESSURE",
                {.description = "background pressure for transport velocity formulation"}),
            parameter<double>("BULK_MODULUS", {.description = "bulk modulus"}),
            parameter<double>("DYNAMIC_VISCOSITY", {.description = "dynamic shear viscosity"}),
            parameter<double>("BULK_VISCOSITY", {.description = "bulk viscosity"}),
            parameter<double>("ARTIFICIAL_VISCOSITY", {.description = "artificial viscosity"}),
            parameter<double>(
                "INITTEMPERATURE", {.description = "initial temperature", .default_value = 0.0}),
            parameter<double>(
                "THERMALCAPACITY", {.description = "thermal capacity", .default_value = 0.0}),
            parameter<double>("THERMALCONDUCTIVITY",
                {.description = "thermal conductivity", .default_value = 0.0}),
            parameter<double>("THERMALABSORPTIVITY",
                {.description = "thermal absorptivity", .default_value = 0.0}),
        },
        {.description = "particle material for SPH fluid"});
  }

  /*----------------------------------------------------------------------*/
  // particle material sph boundary
  {
    known_materials[Core::Materials::m_particle_sph_boundary] = group("MAT_ParticleSPHBoundary",
        {
            parameter<double>("INITRADIUS", {.description = "initial radius"}),
            parameter<double>("INITDENSITY", {.description = "initial density"}),
            parameter<double>(
                "INITTEMPERATURE", {.description = "initial temperature", .default_value = 0.0}),
            parameter<double>(
                "THERMALCAPACITY", {.description = "thermal capacity", .default_value = 0.0}),
            parameter<double>("THERMALCONDUCTIVITY",
                {.description = "thermal conductivity", .default_value = 0.0}),
            parameter<double>("THERMALABSORPTIVITY",
                {.description = "thermal absorptivity", .default_value = 0.0}),
        },
        {.description = "particle material for SPH boundary"});
  }

  /*----------------------------------------------------------------------*/
  // particle material dem
  {
    known_materials[Core::Materials::m_particle_dem] = group("MAT_ParticleDEM",
        {
            parameter<double>("INITRADIUS", {.description = "initial radius of particle"}),
            parameter<double>("INITDENSITY", {.description = "initial density of particle"}),
        },
        {.description = "particle material for DEM"});
  }

  /*----------------------------------------------------------------------*/
  // particle wall material dem
  {
    known_materials[Core::Materials::m_particle_wall_dem] = group("MAT_ParticleWallDEM",
        {
            parameter<double>(
                "FRICT_COEFF_TANG", {.description = "friction coefficient for tangential contact",
                                        .default_value = -1.0}),
            parameter<double>("FRICT_COEFF_ROLL",
                {.description = "friction coefficient for rolling contact", .default_value = -1.0}),
            parameter<double>("ADHESION_SURFACE_ENERGY",
                {.description = "adhesion surface energy", .default_value = -1.0}),
        },
        {.description = "particle wall material for DEM"});
  }

  /*----------------------------------------------------------------------*/
  // General mixture models (used for prestretching and for homogenized constrained mixture models)
  {
    known_materials[Core::Materials::m_mixture] = group("MAT_Mixture",
        {
            parameter<int>("MATIDMIXTURERULE", {.description = "material id of the mixturerule"}),
            parameter<std::vector<int>>(
                "MATIDSCONST", {.description = "list of material IDs of the mixture constituents"}),
        },
        {.description = "General mixture model"});
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for ElastHyper toolbox
  {
    known_materials[Core::Materials::mix_elasthyper] = group("MIX_Constituent_ElastHyper",
        {
            parameter<int>("NUMMAT", {.description = "number of summands"}),
            parameter<std::vector<int>>(
                "MATIDS", {.description = "list material IDs of the summands",
                              .size = from_parameter<int>("NUMMAT")}),
            parameter<int>(
                "PRESTRESS_STRATEGY", {.description = "Material id of the prestress strategy "
                                                      "(optional, by default no prestretch)",
                                          .default_value = 0}),
        },
        {.description = "ElastHyper toolbox"});
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for ElastHyper toolbox with a damage process
  {
    known_materials[Core::Materials::mix_elasthyper_damage] =
        group("MIX_Constituent_ElastHyper_Damage",
            {
                parameter<int>("NUMMAT", {.description = "number of summands"}),
                parameter<std::vector<int>>(
                    "MATIDS", {.description = "list material IDs of the membrane summands",
                                  .size = from_parameter<int>("NUMMAT")}),
                parameter<int>(
                    "PRESTRESS_STRATEGY", {.description = "Material id of the prestress strategy "
                                                          "(optional, by default no prestretch)",
                                              .default_value = 0}),
                parameter<int>("DAMAGE_FUNCT",
                    {.description = "Reference to the function that is a gain for the "
                                    "increase/decrease of the reference mass density."}),
            },
            {.description = "ElastHyper toolbox with damage"});
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for ElastHyper toolbox with a damage process and a membrane constituent
  {
    known_materials[Core::Materials::mix_elasthyper_elastin_membrane] = group(
        "MIX_Constituent_ElastHyper_ElastinMembrane",
        {
            interpolated_input_field<Core::LinAlg::Tensor<double, 3>, Mat::FiberInterpolation>(
                "MEMBRANE_NORMAL",
                {.description =
                        "A unit vector field pointing in the direction of the membrane normal."}),
            parameter<int>("NUMMAT", {.description = "number of summands"}),
            parameter<std::vector<int>>(
                "MATIDS", {.description = "list material IDs of the membrane summands",
                              .size = from_parameter<int>("NUMMAT")}),
            parameter<int>("MEMBRANENUMMAT", {.description = "number of summands"}),
            parameter<std::vector<int>>(
                "MEMBRANEMATIDS", {.description = "list material IDs of the membrane summands",
                                      .size = from_parameter<int>("MEMBRANENUMMAT")}),
            parameter<int>(
                "PRESTRESS_STRATEGY", {.description = "Material id of the prestress strategy "
                                                      "(optional, by default no prestretch)",
                                          .default_value = 0}),
            parameter<int>("DAMAGE_FUNCT",
                {.description = "Reference to the function that is a gain for the "
                                "increase/decrease of the reference mass density."}),
        },
        {.description = "ElastHyper toolbox with damage and 2D membrane material"});
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for solid material
  {
    known_materials[Core::Materials::mix_solid_material] = group("MIX_Constituent_SolidMaterial",
        {
            parameter<int>("MATID", {.description = "ID of the solid material"}),
        },
        {.description = "Solid material"});
  }

  /*----------------------------------------------------------------------*/
  // Isotropic growth
  {
    known_materials[Core::Materials::mix_growth_strategy_isotropic] =
        group("MIX_GrowthStrategy_Isotropic", {}, {.description = "isotropic growth"});
  }

  /*----------------------------------------------------------------------*/
  // Anisotropic growth
  {
    known_materials[Core::Materials::mix_growth_strategy_anisotropic] =
        group("MIX_GrowthStrategy_Anisotropic",
            {
                interpolated_input_field<Core::LinAlg::Tensor<double, 3>, Mat::FiberInterpolation>(
                    "GROWTH_DIRECTION",
                    {.description = "A unit vector field pointing in the direction of growth."}),
            },
            {.description = "anisotropic growth"});
  }

  /*----------------------------------------------------------------------*/
  // Extension of all constituents simultaneously -> Growth happens mainly in the direction with the
  // smallest stiffness
  {
    known_materials[Core::Materials::mix_growth_strategy_stiffness] = group(
        "MIX_GrowthStrategy_Stiffness",
        {
            parameter<double>("KAPPA",
                {.description =
                        "Penalty parameter for the modified penalty term for incompressibility"}),
        },
        {.description = "Extension of all constituents simultaneously"});
  }

  /*----------------------------------------------------------------------*/
  // General material wrapper enabling iterative prestressing
  {
    known_materials[Core::Materials::m_iterative_prestress] = group("MAT_IterativePrestress",
        {
            parameter<int>("MATID", {.description = "Id of the material"}),
            parameter<bool>(
                "ACTIVE", {.description = "Set to True during prestressing and to false afterwards "
                                          "using a restart of the simulation."}),
        },
        {.description =
                "General material wrapper enabling iterative pretressing for any material"});
  }

  /*----------------------------------------------------------------------*/
  // Constant predefined prestretch
  {
    known_materials[Core::Materials::mix_prestress_strategy_constant] =
        group("MIX_Prestress_Strategy_Constant",
            {
                parameter<Core::LinAlg::SymmetricTensor<double, 3, 3>>("PRESTRETCH",
                    {.description = "Definition of the (symmetric) prestretch tensor"}),
            },
            {.description = "Simple predefined prestress"});
  }

  /*----------------------------------------------------------------------*/
  // Prestress strategy for a cylinder
  {
    known_materials[Core::Materials::mix_prestress_strategy_cylinder] = group(
        "MIX_Prestress_Strategy_Cylinder",
        {
            parameter<double>("INNER_RADIUS", {.description = "Inner radius of the cylinder"}),
            parameter<double>("WALL_THICKNESS", {.description = "Wall thickness of the cylinder"}),
            parameter<double>("AXIAL_PRESTRETCH", {.description = "Prestretch in axial direction"}),
            parameter<double>("CIRCUMFERENTIAL_PRESTRETCH",
                {.description = "Prestretch in circumferential direction"}),
            parameter<double>("PRESSURE", {.description = "Pressure in the inner of the cylinder"}),
            interpolated_input_field<Core::LinAlg::Tensor<double, 3>, Mat::FiberInterpolation>(
                "RADIAL", {.description = "A unit vector field pointing in radial direction."}),
            interpolated_input_field<Core::LinAlg::Tensor<double, 3>, Mat::FiberInterpolation>(
                "AXIAL", {.description = "A unit vector field pointing in axial direction."}),
            interpolated_input_field<Core::LinAlg::Tensor<double, 3>, Mat::FiberInterpolation>(
                "CIRCUMFERENTIAL",
                {.description = "A unit vector field pointing in circumferential direction."}),
        },
        {.description = "Simple prestress strategy for a cylinder"});
  }

  /*----------------------------------------------------------------------*/
  // Iterative prestress strategy for any geometry
  {
    known_materials[Core::Materials::mix_prestress_strategy_iterative] =
        group("MIX_Prestress_Strategy_Iterative",
            {
                parameter<bool>(
                    "ACTIVE", {.description = "Flag whether prestretch tensor should be updated"}),
                parameter<bool>(
                    "ISOCHORIC", {.description = "Flag whether prestretch tensor is isochoric",
                                     .default_value = false}),
            },
            {.description = "Simple iterative prestress strategy for any geometry. Needed to be "
                            "used within the mixture framework."});
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for a full constrained mixture fiber
  {
    known_materials[Core::Materials::mix_full_constrained_mixture_fiber] = group(
        "MIX_Constituent_FullConstrainedMixtureFiber",
        {
            interpolated_input_field<Core::LinAlg::Tensor<double, 3>, Mat::FiberInterpolation>(
                "ORIENTATION", {.description = "A unit vector field pointing in the direction of "
                                               "the fiber in the reference configuration."}),
            parameter<int>("FIBER_MATERIAL_ID", {.description = "Id of fiber material"}),
            parameter<bool>("ENABLE_GROWTH",
                {.description = "Switch for the growth (default true)", .default_value = true}),
            parameter<bool>("ENABLE_BASAL_MASS_PRODUCTION",
                {.description = "Switch to enable the basal mass production rate (default true)",
                    .default_value = true}),
            parameter<double>("DECAY_TIME", {.description = "Decay time of deposited tissue"}),
            parameter<double>("GROWTH_CONSTANT", {.description = "Growth constant of the tissue"}),
            parameter<double>(
                "DEPOSITION_STRETCH", {.description = "Stretch at which the fiber is deposited"}),
            parameter<int>("INITIAL_DEPOSITION_STRETCH_TIMEFUNCT",
                {.description =
                        "Id of the time function to scale the deposition stretch (Default: 0=None)",
                    .default_value = 0}),
            parameter<std::string>("ADAPTIVE_HISTORY_STRATEGY",
                {.description = "Strategy for adaptive history integration (none, model_equation, "
                                "higher_order)",
                    .default_value = "none"}),
            parameter<double>("ADAPTIVE_HISTORY_TOLERANCE",
                {.description = "Tolerance of the adaptive history", .default_value = 1e-6}),
        },
        {.description =
                "A 1D constituent that grows with the full constrained mixture fiber theory"});
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for a remodel fiber
  {
    known_materials[Core::Materials::mix_remodelfiber_expl] = group(
        "MIX_Constituent_ExplicitRemodelFiber",
        {
            interpolated_input_field<Core::LinAlg::Tensor<double, 3>, Mat::FiberInterpolation>(
                "ORIENTATION", {.description = "A unit vector field pointing in the direction of "
                                               "the fiber in the reference configuration."}),
            parameter<int>("FIBER_MATERIAL_ID", {.description = "Id of fiber material"}),
            parameter<bool>("ENABLE_GROWTH",
                {.description = "Switch for the growth (default true)", .default_value = true}),
            parameter<bool>("ENABLE_BASAL_MASS_PRODUCTION",
                {.description = "Switch to enable the basal mass production rate (default true)",
                    .default_value = true}),
            parameter<double>("DECAY_TIME", {.description = "Decay time of deposited tissue"}),
            parameter<double>("GROWTH_CONSTANT", {.description = "Growth constant of the tissue"}),
            parameter<double>(
                "DEPOSITION_STRETCH", {.description = "Stretch at with the fiber is deposited"}),
            parameter<int>("DEPOSITION_STRETCH_TIMEFUNCT",
                {.description =
                        "Id of the time function to scale the deposition stretch (Default: 0=None)",
                    .default_value = 0}),
            parameter<bool>("INELASTIC_GROWTH",
                {.description = "Mixture rule has inelastic growth (default false)",
                    .default_value = false}),
            parameter<double>("GAMMA",
                {.description = "Angle of fiber alignment in degree (default = 0.0 degrees)",
                    .default_value = 0.0}),
        },
        {.description = "A 1D constituent that remodels"});
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent for a remodel fiber
  {
    known_materials[Core::Materials::mix_remodelfiber_impl] = group(
        "MIX_Constituent_ImplicitRemodelFiber",
        {
            interpolated_input_field<Core::LinAlg::Tensor<double, 3>, Mat::FiberInterpolation>(
                "ORIENTATION", {.description = "A unit vector field pointing in the direction of "
                                               "the fiber in the reference configuration."}),
            parameter<int>("FIBER_MATERIAL_ID", {.description = "Id of fiber material"}),
            parameter<bool>("ENABLE_GROWTH",
                {.description = "Switch for the growth (default true)", .default_value = true}),
            parameter<bool>("ENABLE_BASAL_MASS_PRODUCTION",
                {.description = "Switch to enable the basal mass production rate (default true)",
                    .default_value = true}),
            parameter<double>("DECAY_TIME", {.description = "Decay time of deposited tissue"}),
            parameter<double>("GROWTH_CONSTANT", {.description = "Growth constant of the tissue"}),
            parameter<double>(
                "DEPOSITION_STRETCH", {.description = "Stretch at with the fiber is deposited"}),
            parameter<int>("DEPOSITION_STRETCH_TIMEFUNCT",
                {.description =
                        "Id of the time function to scale the deposition stretch (Default: 0=None)",
                    .default_value = 0}),
        },
        {.description = "A 1D constituent that remodels"});
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent material for a remodel fiber with exponential strain energy function
  {
    known_materials[Core::Materials::mix_remodelfiber_material_exponential] =
        group("MIX_Constituent_RemodelFiber_Material_Exponential",
            {
                parameter<double>(
                    "K1", {.description = "First parameter of exponential strain energy function"}),
                parameter<double>("K2",
                    {.description = "Second parameter of exponential strain energy function"}),
                parameter<bool>("COMPRESSION",
                    {.description =
                            "Bool, whether the fiber material also supports compressive forces."}),
            },
            {.description = "An exponential strain energy function for the remodel fiber"});
  }

  /*----------------------------------------------------------------------*/
  // Mixture constituent material for a remodel fiber with exponential strain energy function and an
  // active contribution
  {
    known_materials[Core::Materials::mix_remodelfiber_material_exponential_active] = group(
        "MIX_Constituent_RemodelFiber_Material_Exponential_Active",
        {
            parameter<double>(
                "K1", {.description = "First parameter of exponential strain energy function"}),
            parameter<double>(
                "K2", {.description = "Second parameter of exponential strain energy function"}),
            parameter<bool>("COMPRESSION",
                {.description =
                        "Bool, whether the fiber material also supports compressive forces."}),
            parameter<double>("SIGMA_MAX", {.description = "Maximum active Cauchy-stress"}),
            parameter<double>(
                "LAMBDAMAX", {.description = "Stretch at maximum active Cauchy-stress"}),
            parameter<double>("LAMBDA0", {.description = "Stretch at zero active Cauchy-stress"}),
            parameter<double>(
                "LAMBDAACT", {.description = "Current stretch", .default_value = 1.0}),
            parameter<double>("DENS", {.description = "Density of the whole mixture"}),
        },
        {.description = "An exponential strain energy function for the remodel fiber with an "
                        "active contribution"});
  }

  /*----------------------------------------------------------------------*/
  // Function mixture rule for solid mixtures
  {
    known_materials[Core::Materials::mix_rule_function] = group("MIX_Rule_Function",
        {
            parameter<double>("DENS", {.description = ""}),
            parameter<std::vector<int>>(
                "MASSFRACFUNCT", {.description = "list of functions (their ids) defining the mass "
                                                 "fractions of the mixture constituents"}),
        },
        {.description = "A mixture rule where the mass fractions are scaled by functions of space "
                        "and time"});
  }

  /*----------------------------------------------------------------------*/
  // Base mixture rule for solid mixtures
  {
    known_materials[Core::Materials::mix_rule_simple] = group("MIX_Rule_Simple",
        {
            parameter<double>("DENS", {.description = ""}),
            input_field<std::vector<double>>(
                "MASSFRAC", {.description = "list of mass fractions of the mixture constituents"}),
        },
        {.description = "Simple mixture rule"});
  }

  /*----------------------------------------------------------------------*/
  // Base mixture rule for solid mixtures
  {
    known_materials[Core::Materials::mix_rule_growthremodel] = group("MIX_GrowthRemodelMixtureRule",
        {
            parameter<int>(
                "GROWTH_STRATEGY", {.description = "Material id of the growth strategy"}),
            parameter<double>("DENS", {.description = ""}),
            parameter<std::vector<double>>(
                "MASSFRAC", {.description = "list mass fractions of the mixture constituents"}),
        },
        {.description = "Mixture rule for growth/remodel homogenized constrained mixture models"});
  }

  /*----------------------------------------------------------------------*/
  // crystal plasticity
  {
    known_materials[Core::Materials::m_crystplast] = group("MAT_crystal_plasticity",
        {
            parameter<double>("TOL", {.description = "tolerance for internal Newton iteration"}),
            parameter<double>("YOUNG", {.description = "Young's modulus"}),
            parameter<double>("NUE", {.description = "Poisson's ratio"}),
            parameter<double>("DENS", {.description = "Mass density"}),
            parameter<std::string>(
                "LAT", {.description = "lattice type: FCC, BCC, HCP, D019 or L10",
                           .default_value = "FCC"}),
            parameter<double>("CTOA", {.description = "c to a ratio of crystal unit cell"}),
            parameter<double>("ABASE", {.description = "base length a of the crystal unit cell"}),
            parameter<int>("NUMSLIPSYS", {.description = "number of slip systems"}),
            parameter<int>("NUMSLIPSETS", {.description = "number of slip system sets"}),
            parameter<std::vector<int>>("SLIPSETMEMBERS",
                {.description = "vector of NUMSLIPSYS indices ranging from 1 to NUMSLIPSETS that "
                                "indicate to which set each slip system belongs",
                    .size = from_parameter<int>("NUMSLIPSYS")}),
            parameter<std::vector<int>>("SLIPRATEEXP",
                {.description =
                        "vector containing NUMSLIPSETS entries for the rate sensitivity exponent",
                    .size = from_parameter<int>("NUMSLIPSETS")}),
            parameter<std::vector<double>>("GAMMADOTSLIPREF",
                {.description =
                        "vector containing NUMSLIPSETS entries for the reference slip shear rate",
                    .size = from_parameter<int>("NUMSLIPSETS")}),
            parameter<std::vector<double>>("DISDENSINIT",
                {.description =
                        "vector containing NUMSLIPSETS entries for the initial dislocation density",
                    .size = from_parameter<int>("NUMSLIPSETS")}),
            parameter<std::vector<double>>(
                "DISGENCOEFF", {.description = "vector containing NUMSLIPSETS entries for the "
                                               "dislocation generation coefficients",
                                   .size = from_parameter<int>("NUMSLIPSETS")}),
            parameter<std::vector<double>>(
                "DISDYNRECCOEFF", {.description = "vector containing NUMSLIPSETS entries for the "
                                                  "coefficients for dynamic dislocation removal",
                                      .size = from_parameter<int>("NUMSLIPSETS")}),
            parameter<std::vector<double>>(
                "TAUY0", {.description = "vector containing NUMSLIPSETS entries for the lattice "
                                         "resistance to slip, e.g. the Peierls barrier",
                             .size = from_parameter<int>("NUMSLIPSETS")}),
            parameter<std::vector<double>>("MFPSLIP",
                {.description = "vector containing NUMSLIPSETS microstructural parameters that are "
                                "relevant for Hall-Petch strengthening, e.g., grain size",
                    .size = from_parameter<int>("NUMSLIPSETS")}),
            parameter<std::vector<double>>("SLIPHPCOEFF",
                {.description =
                        "vector containing NUMSLIPSETS entries for the Hall-Petch coefficients "
                        "corresponding to the microstructural parameters given in MFPSLIP",
                    .size = from_parameter<int>("NUMSLIPSETS")}),
            parameter<std::vector<double>>("SLIPBYTWIN",
                {.description = "(optional) vector containing NUMSLIPSETS entries for the work "
                                "hardening coefficients by twinning on non-coplanar systems",
                    .default_value = std::vector{0.},
                    .size = from_parameter<int>("NUMSLIPSETS")}),
            parameter<int>("NUMTWINSYS",
                {.description = "(optional) number of twinning systems", .default_value = 0}),
            parameter<int>(
                "NUMTWINSETS", {.description = "(optional) number of sets of twinning systems",
                                   .default_value = 0}),
            parameter<std::vector<int>>("TWINSETMEMBERS",
                {.description = "(optional) vector of NUMTWINSYS indices ranging from 1 to "
                                "NUMTWINSETS that indicate to which set each slip system belongs",
                    .default_value = std::vector{0},
                    .size = from_parameter<int>("NUMTWINSYS")}),
            parameter<std::vector<int>>(
                "TWINRATEEXP", {.description = "(optional) vector containing NUMTWINSETS entries "
                                               "for the rate sensitivity exponent",
                                   .default_value = std::vector{0},
                                   .size = from_parameter<int>("NUMTWINSETS")}),
            parameter<std::vector<double>>(
                "GAMMADOTTWINREF", {.description = "(optional) vector containing NUMTWINSETS "
                                                   "entries for the reference slip shear rate",
                                       .default_value = std::vector{0.},
                                       .size = from_parameter<int>("NUMTWINSETS")}),
            parameter<std::vector<double>>(
                "TAUT0", {.description = "(optional) vector containing NUMTWINSETS entries for the "
                                         "lattice resistance to twinning, e.g. the Peierls barrier",
                             .default_value = std::vector{0.},
                             .size = from_parameter<int>("NUMTWINSETS")}),
            parameter<std::vector<double>>("MFPTWIN",
                {.description =
                        "(optional) vector containing NUMTWINSETS microstructural parameters that "
                        "are relevant for Hall-Petch strengthening of twins, e.g., grain size",
                    .default_value = std::vector{0.},
                    .size = from_parameter<int>("NUMTWINSETS")}),
            parameter<std::vector<double>>(
                "TWINHPCOEFF", {.description = "(optional) vector containing NUMTWINSETS entries "
                                               "for the Hall-Petch coefficients corresponding to "
                                               "the microstructural parameters given in MFPTWIN",
                                   .default_value = std::vector{0.},
                                   .size = from_parameter<int>("NUMTWINSETS")}),
            parameter<std::vector<double>>(
                "TWINBYSLIP", {.description = "(optional) vector containing NUMTWINSETS entries "
                                              "for the work hardening coefficients by slip",
                                  .default_value = std::vector{0.},
                                  .size = from_parameter<int>("NUMTWINSETS")}),
            parameter<std::vector<double>>("TWINBYTWIN",
                {.description = "(optional) vector containing NUMTWINSETS entries for the work "
                                "hardening coefficients by twins on non-coplanar systems",
                    .default_value = std::vector{0.},
                    .size = from_parameter<int>("NUMTWINSETS")}),
        },
        {.description = " Crystal plasticity "});
  }

  /*--------------------------------------------------------------------*/
  // linear elastic material in one direction
  {
    known_materials[Core::Materials::m_linelast1D] = group("MAT_LinElast1D",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus"}),
            parameter<double>("DENS", {.description = "mass density"}),
        },
        {.description = "linear elastic material in one direction"});
  }

  /*--------------------------------------------------------------------*/
  // linear elastic material with growth in one direction
  {
    known_materials[Core::Materials::m_linelast1D_growth] = group("MAT_LinElast1DGrowth",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus"}),
            parameter<double>("DENS", {.description = "mass density"}),
            parameter<double>("C0", {.description = "reference concentration"}),
            parameter<bool>("AOS_PROP_GROWTH",
                {.description = "growth proportional to amount of substance (AOS) if true or "
                                "proportional to concentration if false"}),
            parameter<int>("POLY_PARA_NUM", {.description = "number of polynomial coefficients"}),
            parameter<std::vector<double>>(
                "POLY_PARAMS", {.description = "coefficients of polynomial",
                                   .size = from_parameter<int>("POLY_PARA_NUM")}),
        },
        {.description = "linear elastic material with growth in one direction"});
  }

  return known_materials;
}

FOUR_C_NAMESPACE_CLOSE
