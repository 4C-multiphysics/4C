// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_global_legacy_module_validmaterials.hpp"

#include "4C_global_data.hpp"
#include "4C_io_input_field.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_input_spec_storage.hpp"
#include "4C_io_input_spec_validators.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_utils_densematrix_funct.hpp"
#include "4C_mat_electrode.hpp"
#include "4C_mat_fiber_interpolation.hpp"
#include "4C_mat_fluidporo_singlephase.hpp"
#include "4C_mat_inelastic_defgrad_factors_service.hpp"
#include "4C_mat_micromaterial.hpp"
#include "4C_mat_plasticdruckerprager.hpp"
#include "4C_mat_scatra_growth_remodel.hpp"
#include "4C_mat_scatra_nonlocal_stimulus.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_input.hpp"
#include "4C_structure_new_input.hpp"

#include <filesystem>
#include <optional>
#include <string>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::unordered_map<Core::Materials::MaterialType, Core::IO::InputSpec> Global::valid_materials()
{
  using namespace Core::IO::InputSpecBuilders;
  std::unordered_map<Core::Materials::MaterialType, Core::IO::InputSpec> known_materials;

  const auto size_from_optional_count = [](const std::string& count_parameter_name)
  {
    return [count_parameter_name](const Core::IO::InputParameterContainer& container)
    { return container.get<std::optional<int>>(count_parameter_name).value_or(0); };
  };

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
  // scalar transport material for growth and remodeling
  {
    known_materials[Core::Materials::m_scatra_gr] = group("MAT_scatra_gr",
        {
            parameter<double>("DIFFUSIVITY", {.description = "diffusivity"}),
            parameter<int>("STRUCTURE_MAT_ID",
                {.description = "material ID of the RemodelFiberSsi constituent that provides "
                                "the reaction coefficient"}),
            parameter<Mat::PAR::ScatraGrowthRemodelMat::ScalarQuantity>(
                "SCALAR_QUANTITY", {.description = "scalar mode: growth or remodeling"}),
        },
        {.description = "Simple transport material with linear reaction used in growth "
                        "and remodeling"});
  }

  /*----------------------------------------------------------------------*/
  // scalar transport material for non-local G&R stimulus (Helmholtz smoothing)
  {
    known_materials[Core::Materials::m_scatra_nl_stimulus] = group("MAT_scatra_nl_stimulus",
        {
            parameter<double>("CHAR_LENGTH_SQ",
                {.description =
                        "Squared characteristic length scale lc for the Helmholtz equation"}),
            parameter<int>("STRUCTURE_MAT_ID",
                {.description = "Material ID of the MixtureConstituentRemodelFiberSsi constituent "
                                "that provides the local stimulus"}),
        },
        {.description = "Helmholtz equation for non-local G&R stimulus: "
                        "psi - lc^2 * laplace(psi) = (sigma-sigma_h)"});
  }

  /*----------------------------------------------------------------------*/

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
            interpolated_input_field<Core::LinAlg::Tensor<double, 3>, Mat::FiberInterpolation>(
                "FIBER_ORIENTATION",
                {.description = "A unit vector field pointing in the direction of the fibers."}),
        },
        {.description = "Weickenmeier muscle material"});
  }

  /*----------------------------------------------------------------------*/
  // Combo muscle material
  {
    using namespace Core::IO::InputSpecBuilders::Validators;

    known_materials[Core::Materials::m_muscle_combo] = group("MAT_Muscle_Combo",
        {
            group("PASSIVE",
                {
                    parameter<double>(
                        "ALPHA", {.description = "experimentally fitted material parameter",
                                     .validator = positive<double>()}),
                    parameter<double>(
                        "BETA", {.description = "experimentally fitted material parameter",
                                    .validator = positive<double>()}),
                    parameter<double>(
                        "GAMMA", {.description = "experimentally fitted material parameter",
                                     .validator = positive<double>()}),
                    parameter<double>("OMEGA0",
                        {.description = "weighting factor for isotropic tissue constituents",
                            .validator = in_range<double>(0.0, 1.0)}),
                    parameter<double>("KAPPA",
                        {.description = "material parameter for coupled volumetric contribution"}),
                },
                {.description = "Passive material parameters"}),
            group("ACTIVE",
                {parameter<double>(
                     "POPT", {.description = "tetanised optimal (maximal) active stress",
                                 .validator = positive_or_zero<double>()}),
                    parameter<double>("LAMBDAMIN", {.description = "minimal active fiber stretch",
                                                       .validator = positive<double>()}),
                    parameter<double>(
                        "LAMBDAOPT", {.description = "optimal active fiber stretch related to "
                                                     "active nominal stress maximum",
                                         .validator = positive<double>()}),
                    one_of({
                        parameter<int>("ACTIVATION_FUNCTION_ID",
                            {.description = "function id for time- and space-dependency of muscle "
                                            "activation",
                                .validator = positive<int>()}),
                        input_field<std::vector<std::pair<double, double>>>("ACTIVATION_VALUES",
                            {.description = "json input file containing a map of "
                                            "elementwise-defined discrete values "
                                            "for time- and space-dependency of muscle activation"}),
                    })},
                {.description = "Active material parameters", .required = false}),
            parameter<double>(
                "DENS", {.description = "density", .validator = positive_or_zero<double>()}),
            interpolated_input_field<Core::LinAlg::Tensor<double, 3>, Mat::FiberInterpolation>(
                "FIBER_ORIENTATION",
                {.description = "A unit vector field pointing in the direction of the fibers."}),
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
            interpolated_input_field<Core::LinAlg::Tensor<double, 3>, Mat::FiberInterpolation>(
                "FIBER_ORIENTATION",
                {.description = "A unit vector field pointing in the direction of the fibers."}),
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
  // material parameters for ion species in electrolyte solution
  {
    using namespace Core::IO::InputSpecBuilders::Validators;
    known_materials[Core::Materials::m_newman] = group("MAT_newman",
        {
            parameter<double>("VALENCE", {.description = "valence (= charge number)"}),
            parameter<double>(
                "DIFF_COEF", {.description = "value of the diffusion coefficient without "
                                             "concentration or temperature dependence",
                                 .validator = positive<double>()}),
            parameter<std::optional<int>>("DIFF_COEF_CONC_SCALE_FUNCT",
                {.description = "optional function number describing concentration scaling of the "
                                "diffusion coefficient",
                    .validator = null_or(positive<int>())}),
            parameter<std::optional<int>>("DIFF_COEF_TEMP_SCALE_FUNCT",
                {.description = "optional function number describing temperature scaling of the"
                                "diffusion coefficient",
                    .validator = null_or(positive<int>())}),
            parameter<double>("TRANSFERENCE_NR",
                {.description = "value of the transference number without concentration dependence",
                    .validator = positive<double>()}),
            parameter<std::optional<int>>("TRANSFERENCE_NR_CONC_SCALE_FUNCT",
                {.description = "optional function number describing the concentration scaling of "
                                "the transference number",
                    .validator = null_or(positive<int>())}),
            parameter<double>("THERM_FAC",
                {.description =
                        "value of the thermodynamic factor without concentration dependence",
                    .validator = positive<double>()}),
            parameter<std::optional<int>>("THERM_FAC_CONC_SCALE_FUNCT",
                {.description = "optional function number describing the concentration scaling of "
                                "the thermodynamic factor",
                    .validator = null_or(positive<int>())}),
            parameter<double>("COND",
                {.description =
                        "value of the conductivity without concentration or temperature dependence",
                    .validator = positive<double>()}),
            parameter<std::optional<int>>("COND_CONC_SCALE_FUNCT",
                {.description = "optional function number describing the concentration scaling of "
                                "the conductivity",
                    .validator = null_or(positive<int>())}),
            parameter<std::optional<int>>("COND_TEMP_SCALE_FUNCT",
                {.description = "optional function number describing the temperature scaling of "
                                "the conductivity",
                    .validator = null_or(positive<int>())}),
        },
        {.description = "material parameters for ion species in electrolyte solution"});
  }

  /*----------------------------------------------------------------------*/
  // material parameters for ion species in electrolyte solution for multiscale approach
  {
    using namespace Core::IO::InputSpecBuilders::Validators;
    known_materials[Core::Materials::m_newman_multiscale] = group("MAT_newman_multiscale",
        {
            parameter<double>("VALENCE", {.description = "valence (= charge number)"}),
            parameter<double>(
                "DIFF_COEF", {.description = "value of the diffusion coefficient without "
                                             "concentration or temperature dependence",
                                 .validator = positive<double>()}),
            parameter<std::optional<int>>("DIFF_COEF_CONC_SCALE_FUNCT",
                {.description = "optional function number describing concentration scaling of the "
                                "diffusion coefficient",
                    .validator = null_or(positive<int>())}),
            parameter<std::optional<int>>("DIFF_COEF_TEMP_SCALE_FUNCT",
                {.description = "optional function number describing temperature scaling of the"
                                "diffusion coefficient",
                    .validator = null_or(positive<int>())}),
            parameter<double>("TRANSFERENCE_NR",
                {.description = "value of the transference number without concentration dependence",
                    .validator = positive<double>()}),
            parameter<std::optional<int>>("TRANSFERENCE_NR_CONC_SCALE_FUNCT",
                {.description = "optional function number describing the concentration scaling of "
                                "the transference number",
                    .validator = null_or(positive<int>())}),
            parameter<double>("THERM_FAC",
                {.description =
                        "value of the thermodynamic factor without concentration dependence",
                    .validator = positive<double>()}),
            parameter<std::optional<int>>("THERM_FAC_CONC_SCALE_FUNCT",
                {.description = "optional function number describing the concentration scaling of "
                                "the thermodynamic factor",
                    .validator = null_or(positive<int>())}),
            parameter<double>("COND",
                {.description =
                        "value of the conductivity without concentration or temperature dependence",
                    .validator = positive<double>()}),
            parameter<std::optional<int>>("COND_CONC_SCALE_FUNCT",
                {.description = "optional function number describing the concentration scaling of "
                                "the conductivity",
                    .validator = null_or(positive<int>())}),
            parameter<std::optional<int>>("COND_TEMP_SCALE_FUNCT",
                {.description = "optional function number describing the temperature scaling of "
                                "the conductivity",
                    .validator = null_or(positive<int>())}),
            parameter<double>("ELECTRONIC_COND", {.description = "electronic conductivity"}),
            parameter<int>("ELECTRONIC_COND_CONC_SCALE_FUNC_NUM",
                {.description = "FUNCT number describing concentration dependence of electronic "
                                "conductivity"}),
            parameter<double>("A_s", {.description = "specific micro-scale surface area"}),
            parameter<std::string>("MICROFILE",
                {.description = "input file for micro scale", .default_value = "filename.dat"}),
            parameter<int>("MICRODIS_NUM", {.description = "number of micro-scale discretization"}),
        },
        {.description = "material parameters for ion species in electrolyte solution for "
                        "multi-scale approach"});
  }

  /*----------------------------------------------------------------------*/
  // Hyperelastic Ogden material with tension-compression asymmetry control
  {
    using namespace Core::IO::InputSpecBuilders::Validators;

    known_materials[Core::Materials::m_ogden_tca] = group("MAT_Ogden_TCA",
        {
            parameter<double>("C",
                {.description = "stiffness scaling parameter", .validator = positive<double>()}),
            parameter<double>(
                "M", {.description = "nonlinearity parameter", .validator = positive<double>()}),
            parameter<double>("Q",
                {.description =
                        "tension-compression asymmetry control parameter. "
                        "$q=0.5$ gives a tension--compression symmetric response, "
                        "$q>0.5$ makes the material stiffer in tension than in compression, "
                        "and $q<0.5$ makes the material stiffer in compression than in tension",
                    .validator = in_range(0.0, 1.0)}),
            parameter<double>("KAPPA",
                {.description = "incompressibility parameter", .validator = positive<double>()}),
            parameter<double>(
                "DENS", {.description = "density", .validator = positive_or_zero<double>()}),
        },
        {.description =
                "Hyperelastic Ogden material with tension-compression asymmetry control. The "
                "second Piola-Kirchhoff stress is computed as\n\n"
                "$$\n"
                "\\mathbf{S} = \\sum_{i=1}^{3} \\left[\\frac{c}{m}\\left(q\\,"
                "\\lambda_i^{m-2} - (1-q)\\,\\lambda_i^{-m-2}\\right) "
                "\\mathbf{N}_i \\otimes \\mathbf{N}_i\\right] "
                "+ \\left[\\kappa J (J-1) + \\frac{c}{m}(1-2q)\\right] \\mathbf{C}^{-1}\n"
                "$$\n\n"
                "with $J$ the determinant of the deformation gradient, $\\lambda_i$ the principal "
                "stretches of the right Cauchy-Green deformation tensor $\\mathbf{C}$, and "
                "$\\mathbf{N}_i$ the corresponding principal directions."});
  }

  {
    using namespace Core::IO::InputSpecBuilders::Validators;
    known_materials[Core::Materials::m_scl] = group("MAT_scl",
        {
            parameter<double>("VALENCE", {.description = "valence/charge number"}),
            parameter<double>(
                "DIFF_COEF", {.description = "value of the diffusion coefficient without "
                                             "concentration or temperature dependence",
                                 .validator = positive<double>()}),
            parameter<std::optional<int>>("DIFF_COEF_CONC_SCALE_FUNCT",
                {.description = "optional function number describing concentration scaling of the "
                                "diffusion coefficient",
                    .validator = null_or(positive<int>())}),
            parameter<std::optional<int>>("DIFF_COEF_TEMP_SCALE_FUNCT",
                {.description = "optional function number describing temperature scaling of the"
                                "diffusion coefficient",
                    .validator = null_or(positive<int>())}),
            parameter<double>("COND",
                {.description =
                        "value of the conductivity without concentration or temperature dependence",
                    .validator = positive<double>()}),
            parameter<std::optional<int>>("COND_CONC_SCALE_FUNCT",
                {.description = "optional function number describing the concentration scaling of "
                                "the conductivity",
                    .validator = null_or(positive<int>())}),
            parameter<std::optional<int>>("COND_TEMP_SCALE_FUNCT",
                {.description = "optional function number describing the temperature scaling of "
                                "the conductivity",
                    .validator = null_or(positive<int>())}),
            parameter<double>("TRANSFERENCE_NR",
                {.description = "value of the transference number without concentration dependence",
                    .validator = positive<double>()}),
            parameter<std::optional<int>>("TRANSFERENCE_NR_CONC_SCALE_FUNCT",
                {.description = "optional function number describing the concentration scaling of "
                                "the transference number",
                    .validator = null_or(positive<int>())}),
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
  // electrode material
  {
    using namespace Core::IO::InputSpecBuilders::Validators;
    known_materials[Core::Materials::m_electrode] = group("MAT_electrode",
        {
            // diffusivity and electronic conductivity
            parameter<double>(
                "DIFF_COEF", {.description = "value of the diffusion coefficient without "
                                             "concentration or temperature dependence",
                                 .validator = positive<double>()}),
            parameter<std::optional<int>>("DIFF_COEF_CONC_SCALE_FUNCT",
                {.description = "optional function number describing concentration scaling of the "
                                "diffusion coefficient",
                    .validator = null_or(positive<int>())}),
            parameter<std::optional<int>>("DIFF_COEF_TEMP_SCALE_FUNCT",
                {.description = "optional function number describing temperature scaling of the"
                                "diffusion coefficient",
                    .validator = null_or(positive<int>())}),
            parameter<double>("COND",
                {.description =
                        "value of the conductivity without concentration or temperature dependence",
                    .validator = positive<double>()}),
            parameter<std::optional<int>>("COND_CONC_SCALE_FUNCT",
                {.description = "optional function number describing the concentration scaling of "
                                "the conductivity",
                    .validator = null_or(positive<int>())}),
            parameter<std::optional<int>>("COND_TEMP_SCALE_FUNCT",
                {.description = "optional function number describing the temperature scaling of "
                                "the conductivity",
                    .validator = null_or(positive<int>())}),
            // saturation value of intercalated Lithium concentration
            parameter<double>(
                "C_MAX", {.description = "saturation value of intercalated Lithium concentration",
                             .validator = in_range(1.0e-12, std::numeric_limits<double>::max())}),

            // lithiation value corresponding to saturation value of intercalated Lithium
            // concentration
            parameter<double>(
                "CHI_MAX", {.description = "lithiation value corresponding to saturation value of "
                                           "intercalated Lithium concentration 'C_MAX'",
                               .validator = positive<double>()}),

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
                                    parameter<std::vector<double>>(
                                        "OCP_PARA", {.description = "parameters underlying open "
                                                                    "circuit potential model"}),
                                }),
                            group("Taralov",
                                {
                                    parameter<std::vector<double>>(
                                        "OCP_PARA", {.description = "parameters underlying open "
                                                                    "circuit potential model",
                                                        .size = 13}),
                                }),
                        },
                        store_index_as<Mat::PAR::OCPModels>("OCP_MODEL")),

                    group("LITHIATION_BOUNDS",
                        {
                            parameter<double>("X_MIN",
                                {.description = "lower bound of range of validity as a fraction "
                                                "of C_MAX for ocp calculation model",
                                    .validator = in_range(0.0, 1.0)}),
                            parameter<double>("X_MAX",
                                {.description = "upper bound of range of validity as a fraction "
                                                "of C_MAX for ocp calculation model",
                                    .validator = in_range(0.0, 1.0)}),
                        },
                        {.description =
                                "Optional lithiation bounds of range of validity as a fraction of "
                                "C_MAX for ocp calculation model",
                            .required = false}),
                }),
        },
        {.description = "electrode material"});
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
  // St.Venant--Kirchhoff with orthotropy
  {
    using namespace Core::IO::InputSpecBuilders::Validators;

    known_materials[Core::Materials::m_orthostvenant] = group(
        "MAT_Struct_StVenantKirchhoffOrthotropic",
        {
            input_field<std::array<double, 3>>("YOUNG",
                {.description = "Input vector field of Young's moduli for the principal directions "
                                "with the ordering [E1, E2, E3]."}),
            input_field<std::array<double, 3>>("SHEAR",
                {.description =
                        "Input vector field of shear moduli with the ordering [G12, G23, G13]."}),
            input_field<std::array<double, 3>>(
                "NUE", {.description = "Input vector field of Poisson's ratios with the ordering "
                                       "[nu12, nu23, nu13]."}),
            parameter<double>("DENS", {.description = "mass density"}),
        },
        {.description = "St.Venant--Kirchhoff material with orthotropy. Direction requirements: "
                        "none; its three orthotropic axes are fixed to the Cartesian reference "
                        "axes and cannot be rotated through material or fiber input."});
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
        },
        {.description = "Thermo St.Venant--Kirchhoff material"});
  }
  /*----------------------------------------------------------------------*/
  // Plastic linear elastic St.Venant Kirchhoff / Drucker Prager plasticity
  {
    using namespace Core::IO::InputSpecBuilders::Validators;
    known_materials[Core::Materials::m_pldruckprag] = group("MAT_Struct_DruckerPrager",
        {
            parameter<double>(
                "YOUNG", {.description = "Young's modulus", .validator = positive<double>()}),
            parameter<double>(
                "NUE", {.description = "Poisson's ratio (must be in (-1, 0.5) for stability)",
                           .validator = in_range(excl(-1.0), excl(0.5))}),
            parameter<double>(
                "DENS", {.description = "Density", .validator = positive_or_zero<double>()}),
            parameter<double>(
                "ISOHARD", {.description = "linear isotropic hardening modulus $H_\\mathrm{iso}$",
                               .validator = positive_or_zero<double>()}),
            parameter<double>(
                "C", {.description = "cohesion $c$", .validator = positive<double>()}),
            parameter<double>("ETA", {.description = "Drucker Prager Constant $\\eta$"}),
            parameter<double>("XI", {.description = "Drucker Prager Constant $\\xi$"}),
            parameter<double>("ETABAR",
                {.description = "Drucker Prager Constant $\\overline{\\eta}$ "
                                "(set $\\overline{\\eta} = \\eta$ for associative flow rule)"}),
            parameter<Mat::PAR::PlasticDruckerPrager::TangentType>("TANG",
                {.description = "Method to compute the material tangent (consistent / elastic)",
                    .default_value = Mat::PAR::PlasticDruckerPrager::TangentType::consistent}),
            parameter<double>("TOL", {.description = "Local Newton iteration tolerance",
                                         .default_value = 1.0e-08,
                                         .validator = positive<double>()}),
            parameter<int>("MAXITER", {.description = "Maximum Iterations for local Newton Raphson",
                                          .default_value = 50,
                                          .validator = positive<int>()}),
        },
        {.description = "Linear-elasto-plastic material with a Drucker-Prager yield surface\n\n"
                        "$$\n"
                        "\\phi = \\sqrt{J_2} + \\eta \\, \\mathrm{tr}(\\boldsymbol{\\sigma}) - "
                        "\\xi\\, \\left(c + H_\\mathrm{iso}\\varepsilon_\\mathrm{p}^\\mathrm{acc}"
                        "\\right)\n"
                        "$$\n\n"
                        "and a potentially non-associated plastic potential\n\n"
                        "$$\n"
                        "g = \\sqrt{J_2} + \\overline{\\eta} \\, "
                        "\\mathrm{tr}(\\boldsymbol{\\sigma}) - "
                        "\\xi\\, \\left(c + H_\\mathrm{iso}\\varepsilon_\\mathrm{p}^\\mathrm{acc}"
                        "\\right)\n"
                        "$$"});
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
            parameter<int>("SAMPLENUM",
                {.description =
                        "number of stress-strain pairs for multi-linear isotropic hardening"}),
            parameter<std::vector<double>>(
                "SIGMA_Y", {.description = "yield stress values at specific plastic strains. The "
                                           "value at zero plastic strain is still given in YIELD",
                               .size = from_parameter<int>("SAMPLENUM")}),
            parameter<std::vector<double>>(
                "EPSBAR_P", {.description = "accumulated plastic strain corresponding to SIGMA_Y",
                                .size = from_parameter<int>("SAMPLENUM")}),
            parameter<double>("TOL", {.description = "tolerance for local Newton iteration"}),
        },
        {.description = "Thermo-elastic St.Venant Kirchhoff / plastic von Mises material with "
                        "isotropic and kinematic hardening. The material also includes adiabatic "
                        "heating due to plastic dissipation"});
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
                {.description = "Maximum Newton Raphson Iterations", .default_value = 50}),
            parameter<double>("K1", {.description = "GTN Constant $k_1$"}),
            parameter<double>("K2", {.description = "GTN Constant $k_2$"}),
            parameter<double>("K3", {.description = "GTN constant $k_3$"}),
            parameter<double>(
                "F0", {.description = "GTN constant $f_0$: initial void volume fraction"}),
            parameter<double>("FN", {.description = "GTN constant $f_N$ for damage nucleation"}),
            parameter<double>(
                "EN", {.description = "GTN constant $\\varepsilon_N$ for damage nucleation"}),
            parameter<double>("SN", {.description = "GTN constant $s_N$ for damage nucleation"}),
            parameter<double>("FC", {.description = "GTN constant $f_C$: "
                                                    "void volume fraction at damage coalescence"}),
            parameter<double>("KAPPA",
                {.description = "GTN constant $\\kappa$: Increased damage rate after coalescence"}),
            parameter<double>(
                "EF", {.description = "GTN stabilization parameter ef for damage coalescence",
                          .default_value = 0.0}),
        },
        {.description = "elastic St.Venant Kirchhoff / plastic GTN for porous metal plasticity."
                        "It uses an associated yield function of the form\n\n"
                        "$$\n"
                        "\\Phi = \\left( \\frac{Q}{R^{(3)}} \\right)^2 + "
                        "2 k_1 f^* \\cosh \\left( -\\frac{3}{2} k_2 \\frac{P}{R^{(3)}} \\right) "
                        "- \\left(1 + k_3 {f^*}^2 \\right) = 0\n"
                        "$$\n\n"
                        "with $Q=\\sqrt{\\sigma_\\text{dev} : \\sigma_\\text{dev}}$ "
                        "and P being the trace of the stress.\n\n"
                        "The damage $f^*$ is calculated by\n\n"
                        "$$\n"
                        "f^* = \\begin{cases}\n"
                        "f & f \\leq f_c \\\\\n"
                        "f_c + \\kappa ( f - f_c) & f > f_c\n"
                        "\\end{cases}\n"
                        "$$\n\n"
                        "The rate of the void volume fraction includes growth and nucleation:\n\n"
                        "$$\n"
                        "\\dot{f} = \\dot{f}_\\text{growth} + \\dot{f}_\\text{nucl}\n"
                        "$$\n\n"
                        "with\n\n"
                        "$$\n"
                        "\\dot{f}_\\text{nucl} = \\frac{f_N}{s_N \\sqrt{2\\pi}} "
                        "\\exp \\left\\{ -\\frac{1}{2} "
                        "\\left[ \\frac{\\overline{\\varepsilon}^{pl} - \\epsilon_N}{s_N} "
                        "\\right]^2 \\right\\} "
                        "\\dot{\\overline{\\varepsilon}}^{pl}\n"
                        "$$"});
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
  // 3D solid material superposition
  {
    using namespace Core::IO::InputSpecBuilders::Validators;

    known_materials[Core::Materials::m_superposition] = group("MAT_Solid_Superposition",
        {
            parameter<std::vector<int>>("MATIDS",
                {
                    .description = "List of material IDs to be superimposed",
                    .validator = all_elements(positive<int>()),
                }),
            parameter<double>("DENS", {.description = "mass density of superposition material"}),
        },
        {.description =
                "3D solid material superposition. Each constituent material defined by "
                "MATIDS is evaluated independently, and its responses are accumulated, "
                "such that the stress S = sum_i S_i and the material tangent C = sum_i C_i"});
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
        {.description = "visco-elastic anisotropic fibre material law. Direction requirements: "
                        "2 derived directions; requires the cylindrical coordinate system, from "
                        "which two fiber families are generated using its material angle GAMMA."});
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
        {.description = "list/collection of hyperelastic materials, i.e. material IDs. Direction "
                        "requirements: none imposed by this container itself; the requirements "
                        "are determined by the selected summands. A `STR_TENS_ID` on an "
                        "anisotropic summand controls the structural-tensor strategy but does not "
                        "replace the underlying mean fiber direction."});
  }

  /*----------------------------------------------------------------------*/
  // viscohyperelastic material
  {
    known_materials[Core::Materials::m_viscoelasthyper] = group("MAT_ViscoElastHyper",
        {
            parameter<int>("NUMMAT", {.description = "number of materials/potentials in list"}),
            parameter<std::vector<int>>("MATIDS", {.description = "the list material/potential IDs",
                                                      .size = from_parameter<int>("NUMMAT")}),
            parameter<std::optional<int>>(
                "NUMELAST", {.description = "explicit number of purely elastic summands"}),
            parameter<std::optional<std::vector<int>>>(
                "ELAST_MATIDS", {.description = "explicit purely elastic summand IDs",
                                    .size = size_from_optional_count("NUMELAST")}),
            parameter<std::optional<int>>(
                "NUMVISCO", {.description = "explicit number of visco summands"}),
            parameter<std::optional<std::vector<int>>>(
                "VISCO_MATIDS", {.description = "explicit visco summand IDs",
                                    .size = size_from_optional_count("NUMVISCO")}),
            parameter<double>("DENS", {.description = "material mass density"}),
            parameter<int>(
                "POLYCONVEX", {.description = "1.0 if polyconvexity of system is checked "
                                              "(not supported for viscoelastic combinations.)",
                                  .default_value = 0}),
        },
        {.description = "Viscohyperelastic material. Uses NUMMAT/MATIDS as the complete summand "
                        "list and supports explicit elastic/visco splits with NUMELAST/"
                        "ELAST_MATIDS and NUMVISCO/VISCO_MATIDS. Direction requirements: none "
                        "imposed by this container itself; the requirements are determined by "
                        "the selected summands. A `STR_TENS_ID` on an anisotropic summand "
                        "controls the structural-tensor strategy but does not replace the "
                        "underlying mean fiber direction."});
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
        {.description = "collection of hyperelastic materials for finite strain plasticity. "
                        "Direction requirements: 0 or 3 directions; three explicit, mutually "
                        "orthogonal fibers forming a right-handed basis are required for Hill "
                        "plasticity, while no fibers are used for isotropic von Mises "
                        "plasticity."});
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
        {.description = "collection of hyperelastic materials for finite strain plasticity. "
                        "Direction requirements: 0 or 3 directions; three explicit, mutually "
                        "orthogonal fibers forming a right-handed basis are required for Hill "
                        "plasticity, while no fibers are used for isotropic von Mises "
                        "plasticity."});
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
        {.description = "Logarithmic neo-Hooke material acc. to Bonet and Wood. The strain energy "
                        "is computed as\n\n"
                        "$$\n"
                        "\\Psi = \\frac{\\mu}{2}(I_1-3) - \\mu \\ln(J) + "
                        "\\frac{\\lambda}{2}(\\ln J)^2\n"
                        "$$\n\n"
                        "with $I_1$ the first invariant of the right Cauchy-Green deformation "
                        "tensor and $J$ the determinant of the deformation gradient."});
  }

  /*--------------------------------------------------------------------*/
  // Saint-Venant-Kirchhoff as elastic summand
  {
    known_materials[Core::Materials::mes_coupSVK] = group("ELAST_CoupSVK",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus"}),
            parameter<double>("NUE", {.description = "Poisson's ratio"}),
        },
        {.description = "Saint-Venant-Kirchhoff as elastic summand. The strain energy is computed "
                        "as\n\n"
                        "$$\n"
                        "\\Psi = \\mu\\,\\mathrm{tr}(\\mathbf{E}^2) + "
                        "\\frac{\\lambda}{2}(\\mathrm{tr}\\,\\mathbf{E})^2\n"
                        "$$\n\n"
                        "with $\\mathbf{E}$ the Green-Lagrange strain tensor and $\\mu$, "
                        "$\\lambda$ the Lame constants."});
  }

  /*--------------------------------------------------------------------*/
  // Simo-Pister type material
  {
    known_materials[Core::Materials::mes_coupsimopister] = group("ELAST_CoupSimoPister",
        {
            parameter<double>("MUE", {.description = "material constant"}),
        },
        {.description = "Simo-Pister type material. The strain energy is computed as\n\n"
                        "$$\n"
                        "\\Psi = \\frac{\\mu}{2}(I_1-3) - \\mu \\ln(J)\n"
                        "$$\n\n"
                        "with $I_1$ the first invariant of the right Cauchy-Green deformation "
                        "tensor and $J$ the determinant of the deformation gradient."});
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
        {.description = "Mixed logarithmic neo-Hooke material. The strain energy is computed as\n\n"
                        "$$\n"
                        "\\Psi = \\frac{\\mu}{2}(I_1-3) - \\mu \\ln(J) + "
                        "\\frac{\\lambda}{2}(J-1)^2\n"
                        "$$\n\n"
                        "with $I_1$ the first invariant of the right Cauchy-Green deformation "
                        "tensor and $J$ the determinant of the deformation gradient."});
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
        {.description = "Compressible, isochoric exponential material law for soft tissue. The "
                        "strain energy is computed as\n\n"
                        "$$\n"
                        "\\Psi = a \\exp\\left[b(I_1-3)-(2b+c)\\ln(J)+c(J-1)\\right] - a\n"
                        "$$\n\n"
                        "with $I_1$ the first invariant of the right Cauchy-Green deformation "
                        "tensor and $J$ the determinant of the deformation gradient."});
  }

  /*--------------------------------------------------------------------*/
  // compressible neo-Hooke material acc. to Holzapfel
  {
    known_materials[Core::Materials::mes_coupneohooke] = group("ELAST_CoupNeoHooke",
        {
            parameter<double>("YOUNG", {.description = "Young's modulus", .default_value = 0.0}),
            parameter<double>("NUE", {.description = "Poisson's ratio", .default_value = 0.0}),
        },
        {.description = "Compressible neo-Hooke material acc. to Holzapfel. The strain energy is "
                        "computed as\n\n"
                        "$$\n"
                        "\\Psi = c(I_1-3) + \\frac{c}{\\beta}\\left(I_3^{-\\beta} - 1\\right)\n"
                        "$$\n\n"
                        "with $c = E/(4(1+\\nu))$, $\\beta = \\nu/(1-2\\nu)$, and $I_1$, $I_3$ "
                        "the invariants of the right Cauchy-Green deformation tensor."});
  }
  // Mooney Rivlin  material acc. to Holzapfel
  {
    known_materials[Core::Materials::mes_coupmooneyrivlin] = group("ELAST_CoupMooneyRivlin",
        {
            parameter<double>("C1", {.description = "material constant", .default_value = 0.0}),
            parameter<double>("C2", {.description = "material constant", .default_value = 0.0}),
            parameter<double>("C3", {.description = "material constant", .default_value = 0.0}),
        },
        {.description = "Mooney-Rivlin material acc. to Holzapfel. The strain energy is computed "
                        "as\n\n"
                        "$$\n"
                        "\\Psi = c_1(I_1-3)+c_2(I_2-3)-(2c_1+4c_2)\\ln(J)+c_3(J-1)^2\n"
                        "$$\n\n"
                        "with $I_1$ and $I_2$ the invariants of the right Cauchy-Green deformation "
                        "tensor and $J$ the determinant of the deformation gradient."});
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
        {.description = "Blatz and Ko material acc. to Holzapfel. The strain energy is computed "
                        "as\n\n"
                        "$$\n"
                        "\\Psi = \\frac{\\mu}{2} \\left\\{f\\left[I_1-3+\\frac{I_3^{-\\beta}-1}"
                        "{\\beta}\\right] + (1-f)\\left[\\frac{I_2}{I_3}-3+"
                        "\\frac{I_3^{\\beta}-1}{\\beta}\\right]\\right\\}\n"
                        "$$\n\n"
                        "with $\\beta = \\nu/(1-2\\nu)$, and $I_1$, $I_2$, $I_3$ the invariants of "
                        "the right Cauchy-Green deformation tensor."});
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of Neo-Hooke
  {
    known_materials[Core::Materials::mes_isoneohooke] = group("ELAST_IsoNeoHooke",
        {
            input_field<double>("MUE", {.description = "Shear modulus"}),
        },
        {.description = "Isochoric part of neo-Hooke material acc. to Holzapfel. The strain "
                        "energy is computed as\n\n"
                        "$$\n"
                        "\\Psi = \\frac{\\mu}{2}(\\bar{I}_1 - 3)\n"
                        "$$\n\n"
                        "with $\\bar{I}_1$ the first invariant of the isochoric right "
                        "Cauchy-Green deformation tensor."});
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of one-term Ogden material
  {
    known_materials[Core::Materials::mes_isoogden] = group("ELAST_IsoOgden",
        {
            parameter<double>("MUE", {.description = "Shear modulus"}),
            parameter<double>("ALPHA", {.description = "Nonlinearity parameter"}),
        },
        {.description = "Isochoric part of the one-term Ogden material. The strain energy is "
                        "computed as\n\n"
                        "$$\n"
                        "\\Psi = \\frac{2\\mu}{\\alpha^2}\\left(\\bar{\\lambda}_1^{\\alpha} + "
                        "\\bar{\\lambda}_2^{\\alpha} + \\bar{\\lambda}_3^{\\alpha} - 3\\right)\n"
                        "$$\n\n"
                        "with $\\bar{\\lambda}_i$ the principal stretches of the isochoric right "
                        "Cauchy-Green deformation tensor."});
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
        {.description = "Isochoric part of Yeoh material acc. to Holzapfel. The strain energy is "
                        "computed as\n\n"
                        "$$\n"
                        "\\Psi = c_1(\\bar{I}_1-3) + c_2(\\bar{I}_1-3)^2 + c_3(\\bar{I}_1-3)^3\n"
                        "$$\n\n"
                        "with $\\bar{I}_1$ the first invariant of the isochoric right Cauchy-Green "
                        "deformation tensor."});
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of iso1pow
  {
    known_materials[Core::Materials::mes_iso1pow] = group("ELAST_Iso1Pow",
        {
            parameter<double>("C", {.description = "material parameter"}),
            parameter<int>("D", {.description = "exponent"}),
        },
        {.description = "Isochoric part of general power material. The strain energy is computed "
                        "as\n\n"
                        "$$\n"
                        "\\Psi = c(\\bar{I}_1-3)^d\n"
                        "$$\n\n"
                        "with $\\bar{I}_1$ the first invariant of the isochoric right Cauchy-Green "
                        "deformation tensor."});
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of iso2pow
  {
    known_materials[Core::Materials::mes_iso2pow] = group("ELAST_Iso2Pow",
        {
            parameter<double>("C", {.description = "material parameter"}),
            parameter<int>("D", {.description = "exponent"}),
        },
        {.description = "Isochoric part of general power material. The strain energy is computed "
                        "as\n\n"
                        "$$\n"
                        "\\Psi = c(\\bar{I}_2-3)^d\n"
                        "$$\n\n"
                        "with $\\bar{I}_2$ the second invariant of the isochoric right "
                        "Cauchy-Green deformation tensor."});
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup1pow
  {
    known_materials[Core::Materials::mes_coup1pow] = group("ELAST_Coup1Pow",
        {
            parameter<double>("C", {.description = "material parameter"}),
            parameter<int>("D", {.description = "exponent"}),
        },
        {.description = "Part of general power material. The strain energy is computed as\n\n"
                        "$$\n"
                        "\\Psi = c(I_1-3)^d\n"
                        "$$\n\n"
                        "with $I_1$ the first invariant of the right Cauchy-Green deformation "
                        "tensor."});
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup2pow
  {
    known_materials[Core::Materials::mes_coup2pow] = group("ELAST_Coup2Pow",
        {
            parameter<double>("C", {.description = "material parameter"}),
            parameter<int>("D", {.description = "exponent"}),
        },
        {.description = "Part of general power material. The strain energy is computed as\n\n"
                        "$$\n"
                        "\\Psi = c(I_2-3)^d\n"
                        "$$\n\n"
                        "with $I_2$ the second invariant of the right Cauchy-Green deformation "
                        "tensor."});
  }

  /*--------------------------------------------------------------------*/
  // contribution of coup3pow
  {
    known_materials[Core::Materials::mes_coup3pow] = group("ELAST_Coup3Pow",
        {
            parameter<double>("C", {.description = "material parameter"}),
            parameter<int>("D", {.description = "exponent"}),
        },
        {.description = "Part of general power material. The strain energy is computed as\n\n"
                        "$$\n"
                        "\\Psi = c\\left(I_3^{1/3}-1\\right)^d\n"
                        "$$\n\n"
                        "with $I_3$ the third invariant of the right Cauchy-Green deformation "
                        "tensor."});
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
                "Hyperelastic potential summand for multiplicative coupled invariants I1 and I3. "
                "The strain energy is computed as\n\n"
                "$$\n"
                "\\Psi = c\\left(I_1 I_3^{-a} - 3\\right)^d\n"
                "$$"});
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
        {.description = "Isochoric part of exponential material acc. to Holzapfel. The strain "
                        "energy is computed as\n\n"
                        "$$\n"
                        "\\Psi = \\frac{k_1}{2k_2} \\left\\{\\exp\\left[k_2(\\bar{I}_1-3)^c"
                        "\\right]-1\\right\\}\n"
                        "$$\n\n"
                        "with $\\bar{I}_1$ the first invariant of the isochoric right "
                        "Cauchy-Green deformation tensor."});
  }

  /*--------------------------------------------------------------------*/
  // isochoric contribution of mooney rivlin
  {
    known_materials[Core::Materials::mes_isomooneyrivlin] = group("ELAST_IsoMooneyRivlin",
        {
            parameter<double>("C1", {.description = "Linear modulus for first invariant"}),
            parameter<double>("C2", {.description = "Linear modulus for second invariant"}),
        },
        {.description = "Isochoric part of Mooney-Rivlin material acc. to Holzapfel. The strain "
                        "energy is computed as\n\n"
                        "$$\n"
                        "\\Psi = c_1(\\bar{I}_1-3)+c_2(\\bar{I}_2-3)\n"
                        "$$\n\n"
                        "with $\\bar{I}_1$ and $\\bar{I}_2$ the invariants of the isochoric right "
                        "Cauchy-Green deformation tensor."});
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
            interpolated_input_field<Core::LinAlg::Tensor<double, 3>, Mat::FiberInterpolation>(
                "FIBER_ORIENTATION",
                {.description = "A unit vector field pointing in the direction of the fibers."}),
        },
        {.description = "Anisotropic Blemker muscle material. No scalar potential is evaluated; "
                        "passive and time-dependent active stresses are assembled from the "
                        "piecewise muscle force laws."});
  }

  /*--------------------------------------------------------------------*/
  // test material to test elasthyper-toolbox
  {
    known_materials[Core::Materials::mes_isotestmaterial] = group("ELAST_IsoTestMaterial",
        {
            parameter<double>("C1", {.description = "Modulus for first invariant"}),
            parameter<double>("C2", {.description = "Modulus for second invariant"}),
        },
        {.description = "Test material to test elasthyper-toolbox. The strain energy is computed "
                        "as\n\n"
                        "$$\n"
                        "\\Psi = c_1 x + \\frac{c_1}{2} x^2 + c_2 y + \\frac{c_2}{2} y^2 + "
                        "(c_1+2c_2)\\,x y\n"
                        "$$\n\n"
                        "with $x = \\bar{I}_1-3$ and $y = \\bar{I}_2-3$, where $\\bar{I}_1$ and "
                        "$\\bar{I}_2$ are the invariants of the isochoric right Cauchy-Green "
                        "deformation tensor."});
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
        {.description = "General fiber material for remodeling. No single fixed potential is "
                        "used; it combines the referenced exponential fiber potentials with "
                        "evolving remodeling and growth histories. Direction requirements: "
                        "variable number of directions, one for each referenced "
                        "remodeling-fiber contribution."});
  }

  /*--------------------------------------------------------------------*/
  // volumetric contribution of Sussman Bathe
  {
    known_materials[Core::Materials::mes_volsussmanbathe] = group("ELAST_VolSussmanBathe",
        {
            parameter<double>("KAPPA", {.description = "dilatation modulus"}),
        },
        {.description = "Volumetric part of Sussman-Bathe material. The strain energy is computed "
                        "as\n\n"
                        "$$\n"
                        "\\Psi = \\frac{\\kappa}{2}(J-1)^2\n"
                        "$$\n\n"
                        "with $J$ the determinant of the deformation gradient."});
  }

  /*--------------------------------------------------------------------*/
  // volumetric penalty contribution
  {
    known_materials[Core::Materials::mes_volpenalty] = group("ELAST_VolPenalty",
        {
            parameter<double>("EPSILON", {.description = "penalty parameter"}),
            parameter<double>("GAMMA", {.description = "penalty parameter"}),
        },
        {.description = "Penalty formulation for the volumetric part. The strain energy is "
                        "computed as\n\n"
                        "$$\n"
                        "\\Psi = \\epsilon\\left(J^{\\gamma} + J^{-\\gamma} - 2\\right)\n"
                        "$$\n\n"
                        "with $J$ the determinant of the deformation gradient."});
  }

  /*--------------------------------------------------------------------*/
  // volumetric contribution of Ogden
  {
    known_materials[Core::Materials::mes_vologden] = group("ELAST_VolOgden",
        {
            parameter<double>("KAPPA", {.description = "dilatation modulus"}),
            parameter<double>("BETA", {.description = "empiric constant"}),
        },
        {.description = "Ogden formulation for the volumetric part. The strain energy is computed "
                        "as\n\n"
                        "$$\n"
                        "\\Psi = \\frac{\\kappa}{\\beta^2}\\left[\\beta \\ln(J) + "
                        "J^{-\\beta} - 1\\right]\n"
                        "$$\n\n"
                        "with $J$ the determinant of the deformation gradient; for $\\beta=0$, "
                        "$\\Psi = \\frac{\\kappa}{2}(\\ln J)^2$."});
  }

  /*--------------------------------------------------------------------*/
  // volumetric power law contribution
  {
    known_materials[Core::Materials::mes_volpow] = group("ELAST_VolPow",
        {
            parameter<double>("A", {.description = "prefactor of power law"}),
            parameter<double>("EXPON", {.description = "exponent of power law"}),
        },
        {.description = "Power law formulation for the volumetric part. The strain energy is "
                        "computed as\n\n"
                        "$$\n"
                        "\\Psi = \\frac{a}{n-1} J^{1-n} + a J\n"
                        "$$\n\n"
                        "with $n$ = EXPON and $J$ the determinant of the deformation gradient."});
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
                {.description =
                        "Initialization mode for fiber alignment: "
                        "0 - Fibers defined by material parameters on element basis;"
                        "1 - Fibers defined in input file on element basis;"
                        "4 - Fibers defined in material on gauss point basis, i.e., by nodes;"
                        "3 - Fibers defined in input file on gauss point basis, i.e., by nodes",
                    .default_value = 1}),
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
        {.description = "Anisotropic active fiber. The passive strain energy follows "
                        "ELAST_CoupAnisoExpo. In addition, the active response is computed as\n\n"
                        "$$\n"
                        "\\Psi_\\mathrm{act} = \\frac{s}{\\rho} \\left[\\lambda_\\mathrm{act} + "
                        "\\frac{(\\lambda_\\mathrm{max}-\\lambda_\\mathrm{act})^3}"
                        "{3(\\lambda_\\mathrm{max}-\\lambda_0)^2}\\right]\n"
                        "$$\n\n"
                        "Direction requirements: 1 direction, given by an element fiber or the "
                        "cylindrical coordinate system."});
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
                {.description =
                        "Initialization mode for fiber alignment: "
                        "0 - Fibers defined by material parameters on element basis;"
                        "1 - Fibers defined in input file on element basis;"
                        "4 - Fibers defined in material on gauss point basis, i.e., by nodes;"
                        "3 - Fibers defined in input file on gauss point basis, i.e., by nodes",
                    .default_value = 1}),
            parameter<bool>("ADAPT_ANGLE",
                {.description = "adapt angle during remodeling", .default_value = false}),
            parameter<int>("FIBER_ID",
                {.description = "Id of the fiber to be used (1 for first fiber, default)",
                    .default_value = 1}),
        },
        {.description = "Anisotropic part with one exponential fiber. The strain energy is "
                        "computed as\n\n"
                        "$$\n"
                        "\\Psi = \\frac{k_1}{2k_2} \\left\\{\\exp\\left[k_2(I_4-1)^2\\right]"
                        "-1\\right\\}\n"
                        "$$\n\n"
                        "with $I_4$ the pseudo-invariant associated with the fiber direction. "
                        "K1COMP and K2COMP apply the same law in compression. Direction "
                        "requirements: 1 direction, given by an element fiber or the cylindrical "
                        "coordinate system; `FIBER_ID` selects the element fiber."});
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
                {.description =
                        "Initialization mode for fiber alignment: "
                        "0 - Fibers defined by material parameters on element basis;"
                        "1 - Fibers defined in input file on element basis;"
                        "4 - Fibers defined in material on gauss point basis, i.e., by nodes;"
                        "3 - Fibers defined in input file on gauss point basis, i.e., by nodes",
                    .default_value = 1}),
            parameter<std::vector<int>>(
                "FIBER_IDS", {.description = "Ids of the two fibers to be used (1 for the first "
                                             "fiber, 2 for the second, default)",
                                 .size = 2}),
        },
        {.description = "Exponential shear behavior between two fibers. The strain energy is "
                        "computed as\n\n"
                        "$$\n"
                        "\\Psi = \\frac{k_1}{2k_2} \\left\\{\\exp\\left[k_2(I_8 - "
                        "\\mathbf{a}_1\\cdot\\mathbf{a}_2)^2\\right]-1\\right\\}\n"
                        "$$\n\n"
                        "with $I_8$ the mixed pseudo-invariant of the two fiber directions "
                        "$\\mathbf{a}_1$ and $\\mathbf{a}_2$. Direction requirements: 2 "
                        "directions; requires explicit element or Gauss-point fibers selected by "
                        "`FIBER_IDS`, a cylindrical coordinate system is not supported."});
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
                {.description =
                        "Initialization mode for fiber alignment: "
                        "0 - Fibers defined by material parameters on element basis;"
                        "1 - Fibers defined in input file on element basis;"
                        "4 - Fibers defined in material on gauss point basis, i.e., by nodes;"
                        "3 - Fibers defined in input file on gauss point basis, i.e., by nodes",
                    .default_value = 1}),
            parameter<bool>("ADAPT_ANGLE",
                {.description = "adapt angle during remodeling", .default_value = false}),
        },
        {.description = "Anisotropic part with one pow-like fiber. Where active, the strain "
                        "energy is computed as\n\n"
                        "$$\n"
                        "\\Psi = k\\left(I_4^{d_1}-1\\right)^{d_2}\n"
                        "$$\n\n"
                        "with $I_4$ the pseudo-invariant associated with the fiber direction; "
                        "ACTIVETHRES disables its stress contribution below the selected fiber "
                        "stretch. Direction requirements: 1 direction, given by an element fiber "
                        "or the cylindrical coordinate system; its `FIBER` selector chooses the "
                        "family."});
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
                {.description =
                        "Initialization mode for fiber alignment: "
                        "0 - Fibers defined by material parameters on element basis;"
                        "1 - Fibers defined in input file on element basis;"
                        "4 - Fibers defined in material on gauss point basis, i.e., by nodes;"
                        "3 - Fibers defined in input file on gauss point basis, i.e., by nodes",
                    .default_value = 1}),
            parameter<bool>(
                "FIB_COMP", {.description = "fibers support compression: yes (true) or no (false)",
                                .default_value = true}),
            parameter<bool>("ADAPT_ANGLE",
                {.description = "adapt angle during remodeling", .default_value = false}),
        },
        {.description = "Anisotropic part with two exponential fibers. The strain energy is "
                        "computed as\n\n"
                        "$$\n"
                        "\\Psi = \\sum_{\\alpha=4,6,8} \\frac{a_\\alpha}{2b_\\alpha} "
                        "\\left[\\exp\\left(b_\\alpha x_\\alpha^2\\right)-1\\right]\n"
                        "$$\n\n"
                        "with $x_4 = I_4-1$, $x_6 = I_6-1$, and $x_8 = I_8 - "
                        "\\mathbf{a}_1\\cdot\\mathbf{a}_2$, where $I_4$, $I_6$, $I_8$ are the "
                        "pseudo-invariants of the two fiber directions $\\mathbf{a}_1$, "
                        "$\\mathbf{a}_2$. Direction requirements: 2 directions, given by element "
                        "fibers or the cylindrical coordinate system."});
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
                {.description =
                        "Initialization mode for fiber alignment: "
                        "0 - Fibers defined by material parameters on element basis;"
                        "1 - Fibers defined in input file on element basis;"
                        "4 - Fibers defined in material on gauss point basis, i.e., by nodes;"
                        "3 - Fibers defined in input file on gauss point basis, i.e., by nodes",
                    .default_value = 1}),
            parameter<bool>("ADAPT_ANGLE",
                {.description = "adapt angle during remodeling", .default_value = false}),
        },
        {.description = "Anisotropic part with one neo-Hookean fiber. The strain energy is "
                        "computed as\n\n"
                        "$$\n"
                        "\\Psi = c(I_4-1)\n"
                        "$$\n\n"
                        "with $I_4$ the pseudo-invariant associated with the fiber direction. "
                        "Direction requirements: 1 direction, given by an element fiber or the "
                        "cylindrical coordinate system."});
  }

  /*--------------------------------------------------------------------*/
  // coupled anisotropic material with the stress given by a simplified version of the contraction
  // law of Bestel-Clement-Sorine
  {
    known_materials[Core::Materials::mes_anisoactivestress_evolution] = group(
        "ELAST_AnisoActiveStress_Evolution",
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
                "Anisotropic part with one fiber with coefficient given by a simplification of "
                "the activation-contraction law of Bestel-Clement-Sorine-2001. No stored-energy "
                "potential is evaluated; it supplies the evolving active stress\n\n"
                "$$\n"
                "\\mathbf{S}_\\mathrm{act} = \\tau(t)\\, \\mathbf{A}\n"
                "$$\n\n"
                "with $\\mathbf{A}$ the structural tensor of the fiber direction. Direction "
                "requirements: 1 direction, given by an element fiber or the cylindrical "
                "coordinate system."});
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
        {.description = "Anisotropic part with one neo-Hookean fiber with variable coefficient. "
                        "The strain energy is computed as\n\n"
                        "$$\n"
                        "\\Psi = c(\\mathbf{x},t)(I_4-1)\n"
                        "$$\n\n"
                        "with $I_4$ the pseudo-invariant associated with the fiber direction and "
                        "SOURCE_ACTIVATION defining the spatially and temporally varying "
                        "coefficient $c(\\mathbf{x},t)$. Direction requirements: 1 direction, "
                        "given by an element fiber or the cylindrical coordinate system."});
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
                {.description =
                        "Initialization mode for fiber alignment: "
                        "0 - Fibers defined by material parameters on element basis;"
                        "1 - Fibers defined in input file on element basis;"
                        "4 - Fibers defined in material on gauss point basis, i.e., by nodes;"
                        "3 - Fibers defined in input file on gauss point basis, i.e., by nodes",
                    .default_value = 1}),
            parameter<bool>("ADAPT_ANGLE",
                {.description = "adapt angle during remodeling", .default_value = false}),
        },
        {.description = "Anisotropic part with one exponential fiber, combined isochoric-"
                        "anisotropic response. The strain energy is computed as\n\n"
                        "$$\n"
                        "\\Psi = \\frac{k_1}{2k_2} \\left\\{\\exp\\left[k_2(\\bar{I}_4-1)^2"
                        "\\right]-1\\right\\}\n"
                        "$$\n\n"
                        "with $\\bar{I}_4 = J^{-2/3} I_4$ the isochoric pseudo-invariant "
                        "associated with the fiber direction. Direction requirements: 1 "
                        "direction, given by an element fiber or the cylindrical coordinate "
                        "system."});
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
        {.description =
                "Structural tensor strategy in anisotropic materials. No potential is "
                "evaluated; it supplies the structural tensor $\\mathbf{A}$, for example\n\n"
                "$$\n"
                "\\mathbf{A} = \\mathbf{a} \\otimes \\mathbf{a}\n"
                "$$\n\n"
                "for the Standard strategy, with $\\mathbf{a}$ the fiber direction and "
                "$\\otimes$ the dyadic product. Direction requirements: 1 direction, "
                "defining the mean fiber direction the structural tensor is built from; "
                "this is a helper rather than an independent strain-energy term."});
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
                {.description =
                        "Initialization mode for fiber alignment: "
                        "0 - Fibers defined by material parameters on element basis;"
                        "1 - Fibers defined in input file on element basis;"
                        "4 - Fibers defined in material on gauss point basis, i.e., by nodes;"
                        "3 - Fibers defined in input file on gauss point basis, i.e., by nodes",
                    .default_value = 1}),
        },
        {.description = "Transversely part of a simple orthotropic, transversely isotropic "
                        "hyperelastic constitutive equation. The strain energy is computed as\n\n"
                        "$$\n"
                        "\\Psi = \\left[\\alpha + \\frac{\\beta}{2}\\ln(I_3) + "
                        "\\gamma(I_4-1)\\right](I_4-1) - \\frac{\\alpha}{2}(I_5-1)\n"
                        "$$\n\n"
                        "with $I_3$, $I_4$, $I_5$ invariants of the right Cauchy-Green deformation "
                        "tensor and the fiber direction. Direction requirements: 1 direction, "
                        "given by an element fiber or the cylindrical coordinate system."});
  }

  /*--------------------------------------------------------------------*/
  // coupled Varga material acc. to Holzapfel
  {
    known_materials[Core::Materials::mes_coupvarga] = group("ELAST_CoupVarga",
        {
            parameter<double>("MUE", {.description = "Shear modulus"}),
            parameter<double>("BETA", {.description = "'Anti-modulus'"}),
        },
        {.description = "Varga material acc. to Holzapfel. The strain energy is computed as\n\n"
                        "$$\n"
                        "\\Psi = (2\\mu-\\beta)(\\lambda_1+\\lambda_2+\\lambda_3-3) + "
                        "\\beta\\left(\\lambda_1^{-1}+\\lambda_2^{-1}+\\lambda_3^{-1}-3\\right)\n"
                        "$$\n\n"
                        "with $\\lambda_i$ the principal stretches of the right Cauchy-Green "
                        "deformation tensor."});
  }

  /*--------------------------------------------------------------------*/
  // isochoric Varga material acc. to Holzapfel
  {
    known_materials[Core::Materials::mes_isovarga] = group("ELAST_IsoVarga",
        {
            parameter<double>("MUE", {.description = "Shear modulus"}),
            parameter<double>("BETA", {.description = "'Anti-modulus'"}),
        },
        {.description = "Isochoric Varga material acc. to Holzapfel. The strain energy is "
                        "computed as\n\n"
                        "$$\n"
                        "\\Psi = (2\\mu-\\beta)\\left(\\bar{\\lambda}_1+\\bar{\\lambda}_2+"
                        "\\bar{\\lambda}_3-3\\right) + \\beta\\left(\\bar{\\lambda}_1^{-1}+"
                        "\\bar{\\lambda}_2^{-1}+\\bar{\\lambda}_3^{-1}-3\\right)\n"
                        "$$\n\n"
                        "with $\\bar{\\lambda}_i$ the principal stretches of the isochoric right "
                        "Cauchy-Green deformation tensor."});
  }

  /*--------------------------------------------------------------------*/
  // isotropic viscous contribution of myocardial matrix (chapelle12)
  {
    known_materials[Core::Materials::mes_coupmyocard] = group("VISCO_CoupMyocard",
        {
            parameter<double>("N", {.description = "material parameter"}),
        },
        {.description =
                "Coupled myocardial viscoelastic contribution.\n\n"
                "$$\n"
                "\\Phi_\\mathrm{v} = \\frac{\\eta}{2}\\dot{\\boldsymbol E}:\\dot{\\boldsymbol E} "
                "= \\frac{\\eta}{8}\\dot{\\boldsymbol C}:\\dot{\\boldsymbol C}\n"
                "$$\n\n"
                "hence $\\boldsymbol S_\\mathrm{v}=\\eta\\dot{\\boldsymbol E}$; `N` is "
                "$\\eta$."});
  }

  /*--------------------------------------------------------------------*/
  // isochoric rate dependent viscos material, modified from Pioletti,1997
  {
    known_materials[Core::Materials::mes_isoratedep] = group("VISCO_IsoRateDep",
        {
            parameter<double>("N", {.description = "material parameter"}),
        },
        {.description = "Isochoric rate-dependent contribution.\n\n"
                        "$$\n"
                        "\\Phi_\\mathrm{v} = n\\,\\bar J_2(\\bar I_1-3)\n"
                        "$$\n\n"
                        "with $\\bar J_2=\\frac12\\dot{\\bar{\\boldsymbol C}}:"
                        "\\dot{\\bar{\\boldsymbol C}}$. The rate is evaluated by a backward "
                        "difference."});
  }

  /*--------------------------------------------------------------------*/
  // viscos contribution to visohyperelastic material according to FSLS-Model
  {
    known_materials[Core::Materials::mes_fsls] = group("VISCO_FSLS",
        {
            parameter<double>("TAU", {.description = "relaxation parameter"}),
            parameter<double>("ALPHA", {.description = "fractional order derivative"}),
            parameter<double>("BETA", {.description = "emphasis of viscous to elastic part"}),
        },
        {.description =
                "Fractional standard linear solid.\n"
                "Hereditary viscous stress update from artificial stress $\\boldsymbol Q$. With "
                "\n\n"
                "$$\n"
                "b_0=1,\\quad b_j=\\frac{j-1-\\alpha}{j}b_{j-1}, \\quad"
                "\\lambda_1=\\frac{\\Delta t^\\alpha}{\\Delta t^\\alpha+\\tau^\\alpha},\\quad"
                "\\lambda_2=-\\frac{\\tau^\\alpha}{\\Delta t^\\alpha+\\tau^\\alpha},\n"
                "$$\n\n"
                "the implemented history update is \n\n"
                "$$\n"
                "\\boldsymbol Q^{n+1}=\\lambda_1\\beta\\boldsymbol S_0^{n+1}"
                "+\\lambda_2\\sum_{j=1}^{m}b_j\\boldsymbol Q^{n+1-j},"
                "\\quad \\boldsymbol S_\\mathrm{v}^{n+1}"
                "=\\boldsymbol Q^{n+1}-\\beta\\boldsymbol S_0^{n+1}.\n"
                "$$\n"});
  }

  /*--------------------------------------------------------------------*/
  // viscoelatic branches of a generalized Maxwell model
  {
    known_materials[Core::Materials::mes_generalizedmaxwell] = group("VISCO_GeneralizedMaxwell",
        {
            parameter<int>("NUMBRANCH", {.description = "number of viscoelastic branches"}),
            parameter<std::vector<int>>("MATIDS",
                {.description = "the list material IDs", .size = from_parameter<int>("NUMBRANCH")}),
            parameter<std::string>("SOLVE",
                {.description = "Solution for evolution equation: OneStepTheta (default) or "
                                "ExponentialTimeDiscretization (convolution integral)",
                    .default_value = "OneStepTheta"}),
        },
        {.description =
                "Generalized Maxwell model, obtains a separate elastic law and $\\tau$ from each "
                "VISCO_GeneralizedMaxwellBranch.\n\n"
                "$$\n"
                "\\boldsymbol S_\\mathrm{v} = \\sum_i \\boldsymbol Q_i\n"
                "$$\n\n"
                "where $\\dot{\\boldsymbol Q}_i+\\boldsymbol Q_i/\\tau_i="
                "\\dot{\\boldsymbol S}^{\\,e}_i$. Each $\\boldsymbol S^{\\,e}_i$ comes from "
                "the elastic material referenced by its branch."});
  }

  /*--------------------------------------------------------------------*/
  // quasi-linear Fung-type generalized Maxwell model
  {
    using namespace Core::IO::InputSpecBuilders::Validators;

    known_materials[Core::Materials::mes_quasilineargeneralizedmaxwell] =
        group("VISCO_QuasiLinearGeneralizedMaxwell",
            {
                parameter<std::vector<double>>(
                    "BETA", {.description = "dimensionless relative branch stress weights",
                                .validator = all_elements(positive_or_zero<double>())}),
                parameter<std::vector<double>>(
                    "TAU", {.description = "positive branch relaxation times",
                               .validator = all_elements(positive<double>())}),
                deprecated_selection<std::string>("SOLVE",
                    {"OneStepTheta", "ExponentialTimeDiscretization"},
                    {.description = "Solution for evolution equation: OneStepTheta (default) or "
                                    "ExponentialTimeDiscretization (first-order exponential time "
                                    "discretization with backward-difference forcing)",
                        .default_value = "OneStepTheta"}),
                parameter<double>("VISCOSITY", {.description = "parallel dashpot viscosity",
                                                   .default_value = 0.0,
                                                   .validator = positive_or_zero<double>()}),
            },
            {.description =
                    "Fung-type quasi-linear generalized Maxwell model; "
                    "uses $\\beta$ and $\\tau$ arrays and drives every branch from the same "
                    "surrounding hyperelastic base law. "
                    "Its optional viscosity $\\eta$ defines the parallel-dashpot pseudo-potential "
                    "$\\Phi_\\eta=\\frac{\\eta}{2}\\dot{\\boldsymbol E}:\\dot{\\boldsymbol E}$, "
                    "from which the viscous stress is derived as\n\n"
                    "$$\n"
                    "\\boldsymbol S_\\mathrm{v} = \\sum_i \\boldsymbol Q_i + "
                    "\\eta\\dot{\\boldsymbol E} \\quad \\text{with} \\quad "
                    "\\dot{\\boldsymbol Q}_i+\\boldsymbol Q_i/\\tau_i="
                    "\\beta_i\\dot{\\boldsymbol S}_0.\n"
                    "$$\n"});
  }

  /*--------------------------------------------------------------------*/
  // description of a viscoelastic branch of a generalized Maxwell model
  {
    known_materials[Core::Materials::mes_viscobranch] = group("VISCO_GeneralizedMaxwellBranch",
        {
            parameter<double>(
                "TAU", {.description = "dynamic viscosity divided by branch stiffness"}),
            parameter<int>("MATID", {.description = "material ID of branch elasticity rule"}),
        },
        {.description = "Branch referenced by a generalized Maxwell model.\n\n"
                        "No independent equation; `MATID` defines $\\boldsymbol S^{\\,e}_i$ and "
                        "`TAU` defines $\\tau_i$ in the parent Maxwell law."});
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
                parameter<double>("REF_TEMPERATURE",
                    {.description = "reference temperature for thermoelastic expansion.",
                        .default_value = 0.0,
                        .validator = Validators::positive_or_zero<double>()}),
                parameter<double>("THERMAL_EXPANSION_COEFFICIENT",
                    {.description = "coefficient of thermal expansion $\\alpha_T$",
                        .default_value = 0.0}),
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
    using namespace Core::IO::InputSpecBuilders::Validators;
    known_materials[Core::Materials::mfi_lin_scalar_iso] = group("MAT_InelasticDefgradLinScalarIso",
        {
            parameter<int>("SCALAR1",
                {.description = "number of growth inducing scalar", .validator = positive<int>()}),
            parameter<double>("SCALAR1_MolarGrowthFac",
                {.description = "isotropic molar growth factor due to scalar 1",
                    .validator = positive<double>()}),
            parameter<double>("SCALAR1_RefConc",
                {.description = "reference concentration of scalar 1 causing no strains",
                    .validator = positive_or_zero<double>()}),
        },
        {.description = "scalar dependent isotropic growth law; volume change linearly dependent "
                        "on scalar (in material configuration)"});
  }

  /*----------------------------------------------------------------------*/
  // simple anisotropic, volumetric growth; growth direction prescribed in input-file;
  // growth is linearly dependent on scalar mapped to material configuration, constant material
  // density
  {
    using namespace Core::IO::InputSpecBuilders::Validators;
    known_materials[Core::Materials::mfi_lin_scalar_aniso] = group(
        "MAT_InelasticDefgradLinScalarAniso",
        {
            parameter<int>("SCALAR1",
                {.description = "number of growth inducing scalar", .validator = positive<int>()}),
            parameter<double>("SCALAR1_MolarGrowthFac",
                {.description = "anisotropic molar growth factor due to scalar 1",
                    .validator = positive<double>()}),
            parameter<double>("SCALAR1_RefConc",
                {.description = "reference concentration of scalar 1 causing no strains",
                    .validator = positive_or_zero<double>()}),
            parameter<std::vector<double>>("GrowthDirection",
                {.description = "vector that defines the growth direction", .size = 3}),
        },
        {.description = "scalar dependent anisotropic growth law; growth in direction as given "
                        "in input-file; volume change linearly dependent on scalar (in "
                        "material configuration)"});
  }

  /*----------------------------------------------------------------------*/
  // non-linear isotropic volumetric growth; growth is dependent on the degree of lithiation,
  // constant material density, nonlinear behavior prescribed by polynomial in input file
  {
    using namespace Core::IO::InputSpecBuilders::Validators;
    known_materials[Core::Materials::mfi_poly_intercal_frac_iso] = group(
        "MAT_InelasticDefgradPolyIntercalFracIso",
        {
            parameter<int>("SCALAR1",
                {.description = "number of growth inducing scalar", .validator = positive<int>()}),
            parameter<double>("SCALAR1_RefConc",
                {.description = "reference concentration of scalar 1 causing no strains",
                    .validator = positive_or_zero<double>()}),
            parameter<int>("POLY_PARA_NUM",
                {.description = "number of polynomial coefficients", .validator = positive<int>()}),
            parameter<std::vector<double>>(
                "POLY_PARAMS", {.description = "coefficients of polynomial",
                                   .size = from_parameter<int>("POLY_PARA_NUM")}),
            parameter<double>("X_min", {.description = "lower bound of validity of polynomial",
                                           .validator = positive_or_zero<double>()}),
            parameter<double>("X_max", {.description = "upper bound of validity of polynomial",
                                           .validator = positive<double>()}),
            parameter<int>(
                "MATID", {.description = "material ID of the corresponding scatra material",
                             .validator = positive<int>()}),
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
    using namespace Core::IO::InputSpecBuilders::Validators;
    known_materials[Core::Materials::mfi_poly_intercal_frac_aniso] = group(
        "MAT_InelasticDefgradPolyIntercalFracAniso",
        {
            parameter<int>("SCALAR1",
                {.description = "number of growth inducing scalar", .validator = positive<int>()}),
            parameter<double>("SCALAR1_RefConc",
                {.description = "reference concentration of scalar 1 causing no strains",
                    .validator = positive_or_zero<double>()}),
            parameter<std::vector<double>>("GrowthDirection",
                {.description = "vector that defines the growth direction", .size = 3}),
            parameter<int>("POLY_PARA_NUM",
                {.description = "number of polynomial coefficients", .validator = positive<int>()}),
            parameter<std::vector<double>>(
                "POLY_PARAMS", {.description = "coefficients of polynomial",
                                   .size = from_parameter<int>("POLY_PARA_NUM")}),
            parameter<double>("X_min", {.description = "lower bound of validity of polynomial",
                                           .validator = positive_or_zero<double>()}),
            parameter<double>("X_max", {.description = "upper bound of validity of polynomial",
                                           .validator = positive<double>()}),
            parameter<int>(
                "MATID", {.description = "material ID of the corresponding scatra material",
                             .validator = positive<int>()}),
        },
        {.description =
                "scalar dependent anisotropic growth law; growth in direction as given in "
                "input-file; volume change nonlinearly dependent on the intercalation fraction, "
                "that is calculated using the scalar concentration (in material configuration)"});
  }

  /*----------------------------------------------------------------------*/
  {
    using namespace Core::IO::InputSpecBuilders::Validators;
    known_materials[Core::Materials::mfi_lin_temp_iso] = group("MAT_InelasticDefgradLinTempIso",
        {
            parameter<double>(
                "Temp_GrowthFac", {.description = "isotropic growth factor due to temperature"}),
            parameter<double>("RefTemp", {.description = "reference temperature causing no strains",
                                             .validator = positive_or_zero<double>()}),
        },
        {.description = "Temperature dependent growth law. Volume change linearly dependent on "
                        "temperature"});
  }

  /*----------------------------------------------------------------------*/
  {
    using namespace Core::IO::InputSpecBuilders::Validators;
    known_materials[Core::Materials::mfi_time_funct_aniso] = group(
        "MAT_InelasticDefgradTimeFunctAniso",
        {
            parameter<int>("FUNCT_NUM", {.description = "Time-dependent function used to calculate "
                                                        "the inelastic deformation gradient",
                                            .validator = positive<int>()}),
            parameter<std::vector<double>>("GrowthDirection",
                {.description = "vector that defines the growth direction", .size = 3}),
        },
        {.description = "Time-dependent anisotropic growth law; growth in direction as given "
                        "in input-file. Determinant of volume change dependent on "
                        "(1 + time function value) defined by 'FUNCT_NUM'"});
  }

  /*----------------------------------------------------------------------*/
  {
    using namespace Core::IO::InputSpecBuilders::Validators;
    known_materials[Core::Materials::mfi_time_funct_iso] = group("MAT_InelasticDefgradTimeFunctIso",
        {
            parameter<int>("FUNCT_NUM", {.description = "Time-dependent function used to calculate "
                                                        "the inelastic deformation gradient",
                                            .validator = positive<int>()}),
        },
        {.description = "Time-dependent isotropic growth law. Determinant of volume change "
                        "dependent on (1 + time function value) defined by 'FUNCT_NUM'"});
  }

  /*----------------------------------------------------------------------*/
  {
    using namespace Core::IO::InputSpecBuilders::Validators;
    namespace ViscoplastUtils = Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;

    known_materials[Core::Materials::mfi_transv_isotrop_elast_viscoplast] = group(
        "MAT_InelasticDefgradTransvIsotropElastViscoplast",
        {parameter<int>(
             "VISCOPLAST_LAW_ID", {.description = "MAT ID of the corresponding viscoplastic law",
                                      .validator = positive<int>()}),
            parameter<int>(
                "FIBER_READER_ID", {.description = "MAT ID of the used fiber direction reader for "
                                                   "transversely isotropic behavior",
                                       .validator = positive<int>()}),
            parameter<double>("TAYLOR_QUINNEY_COEFFICIENT",
                {.description =
                        "Taylor-Quinney coefficient $\\xi_{TQ}$ modeling the internal dissipation",
                    .default_value = 0.0,
                    .validator = positive_or_zero<double>()}),
            parameter<std::optional<double>>(
                "YIELD_COND_A", {.description = "transversely isotropic version of the Hill(1948) "
                                                "yield condition: parameter A, following the "
                                                "notation in Dafalias 1989, International "
                                                "Journal of Plasticity, Vol. 5"}),
            parameter<std::optional<double>>(
                "YIELD_COND_B", {.description = "transversely isotropic version of the Hill(1948) "
                                                "yield condition: parameter B, following the "
                                                "notation in Dafalias 1989, International "
                                                "Journal of Plasticity, Vol. 5"}),
            parameter<std::optional<double>>(
                "YIELD_COND_F", {.description = "transversely isotropic version of the Hill(1948) "
                                                "yield condition: parameter F, following the "
                                                "notation in Dafalias 1989, International "
                                                "Journal of Plasticity, Vol. 5"}),
            parameter<ViscoplastUtils::MatBehavior>(
                "MAT_BEHAVIOR", {.description = "Material behavior / anisotropy type"}),
            parameter<ViscoplastUtils::TimIntType>("TIME_INTEGRATION_HIST_VARS",
                {.description =
                        "time integration of internal variables: standard | logarithmic "
                        "(logarithmic transformation of the "
                        "evolution equation for the plastic deformation gradient -> default)",
                    .default_value = ViscoplastUtils::TimIntType::logarithmic}),
            parameter<ViscoplastUtils::LinearizationType>("LINEARIZATION",
                {.description =
                        "utilized material linearization: analytic | perturb_based (based on "
                        "perturbations of the current state)",
                    .default_value = ViscoplastUtils::LinearizationType::analytic}),
            parameter<Core::LinAlg::MatrixExpCalcMethod>("MATRIX_EXP_CALC_METHOD",
                {.description = "chosen computation method for matrix exponential (default: "
                                "automatic method selection based on matrix characteristics)",
                    .default_value = Core::LinAlg::MatrixExpCalcMethod::automatic}),
            parameter<Core::LinAlg::GenMatrixExpFirstDerivCalcMethod>(
                "MATRIX_EXP_DERIV_CALC_METHOD",
                {.description = "chosen computation method for the first derivative of the matrix "
                                "exponential w.r.t. matrix (default: automatic method selection "
                                "based on matrix characteristics)",
                    .default_value = Core::LinAlg::GenMatrixExpFirstDerivCalcMethod::automatic}),
            parameter<Core::LinAlg::MatrixLogCalcMethod>("MATRIX_LOG_CALC_METHOD",
                {.description = "chosen computation method for matrix logarithm",
                    .default_value = Core::LinAlg::MatrixLogCalcMethod::inv_scal_square}),
            parameter<Core::LinAlg::GenMatrixLogFirstDerivCalcMethod>(
                "MATRIX_LOG_DERIV_CALC_METHOD",
                {.description = "chosen computation method for the first derivative of the matrix "
                                "logarithm w.r.t. matrix",
                    .default_value =
                        Core::LinAlg::GenMatrixLogFirstDerivCalcMethod::pade_part_fract}),
            group("LOCAL_SUBSTEPPING",
                {
                    parameter<bool>("USE_SUBSTEPPING",
                        {.description = "use substepping?", .default_value = false}),
                    parameter<int>("MAX_SUBSTEPPING_HALVE_NUM",
                        {.description = "maximum number of times the global time step can "
                                        "be halved in the substepping procedure",
                            .default_value = 10,
                            .validator = positive_or_zero<int>()}),
                },
                {.description = "Settings for the usage of local substepping to integrate the "
                                "viscoplastic evolution equations",
                    .required = false}),
            group<ViscoplastUtils::LocalNewtonParams>("LOCAL_NEWTON",
                {parameter<ViscoplastUtils::LocalNewtonConvCheck>("CONV_CHECK",
                     {.description = "convergence check type",
                         .default_value =
                             ViscoplastUtils::LocalNewtonConvCheck::residual_and_increment_ratio,
                         .store = in_struct(&ViscoplastUtils::LocalNewtonParams::conv_check)}),
                    parameter<int>("MAX_ITER",
                        {.description = "maximum number of iterations",
                            .default_value = 100,
                            .validator = positive<int>(),
                            .store = in_struct(&ViscoplastUtils::LocalNewtonParams::max_iter)}),
                    parameter<double>("RES_TOL",
                        {.description = "residual tolerance (absolute residual 2-norm)",
                            .default_value = 1.0e-8,
                            .validator = positive<double>(),
                            .store = in_struct(&ViscoplastUtils::LocalNewtonParams::res_tol)}),
                    parameter<double>("MAX_EXCEEDANCE_FACT_RES_TOL",
                        {.description =
                                "maximum exceedance factor for the specified residual tolerance "
                                "(Local Newton divergence safeguard for "
                                "continuing the simulation, if specified by the user via "
                                "DIVER_CONT)",
                            .default_value = 1.0e1,
                            .validator = positive_or_zero<double>(),
                            .store = in_struct(
                                &ViscoplastUtils::LocalNewtonParams::max_exceedance_fact_res_tol)}),
                    parameter<double>("INCR_TOL",
                        {.description = "increment tolerance ("
                                        "ratio of |increment| / |solution|)",
                            .default_value = 1.0e-8,
                            .validator = positive<double>(),
                            .store = in_struct(&ViscoplastUtils::LocalNewtonParams::incr_tol)

                        }),
                    parameter<double>("MAX_EXCEEDANCE_FACT_INCR_TOL",
                        {.description =
                                "maximum exceedance factor for the specified increment tolerance "
                                "(Local Newton divergence safeguard for "
                                "continuing the simulation, if specified by the user via "
                                "DIVER_CONT)",
                            .default_value = 1.0e1,
                            .validator = positive_or_zero<double>(),
                            .store = in_struct(
                                &ViscoplastUtils::LocalNewtonParams::max_exceedance_fact_incr_tol)

                        }),
                    parameter<ViscoplastUtils::LocalNewtonDiverCont>("DIVER_CONT",
                        {.description = "strategy to deal with divergence in the Local Newton Loop",
                            .default_value = ViscoplastUtils::LocalNewtonDiverCont::stop,
                            .store = in_struct(&ViscoplastUtils::LocalNewtonParams::diver_cont)

                        })

                },
                {.description = "Parameters used in the Local Newton--Raphson procedure "
                                "(viscoplastic corrector stage)",
                    .required = false}),
            group<ViscoplastUtils::ErrorRegistrationSettings>("ERROR_REGISTRATION_SETTINGS",
                {parameter<bool>("REGISTER_PLASTIC_STRAIN_INCR_OVERFLOW",
                     {.description = "should overflow error be registered via ErrorType when the "
                                     "plastic strain increment exceeds the specified tolerance?",
                         .default_value = true,
                         .store = in_struct(&ViscoplastUtils::ErrorRegistrationSettings::
                                 register_plastic_strain_incr_overflow)}),
                    parameter<double>("MAX_PLASTIC_STRAIN_INCR",
                        {.description = "maximum evaluable plastic strain increment "
                                        "used for registering overflow errors",
                            .default_value = std::exp(30.0),
                            .validator = positive<double>(),
                            .store = in_struct(&ViscoplastUtils::ErrorRegistrationSettings::
                                    max_plastic_strain_incr)}),
                    parameter<bool>("REGISTER_PLASTIC_STRAIN_DERIV_INCR_OVERFLOW",
                        {.description = "should overflow error be registered via ErrorType when "
                                        "any of the plastic strain derivative increments exceeds "
                                        "the specified tolerance?",
                            .default_value = false,
                            .store = in_struct(&ViscoplastUtils::ErrorRegistrationSettings::
                                    register_plastic_strain_deriv_incr_overflow)}),
                    parameter<double>("MAX_PLASTIC_STRAIN_DERIV_INCR",
                        {.description = "maximum evaluable increment of the plastic strain "
                                        "derivatives w.r.t. plastic strain and equivalent "
                                        "stress, used for registering "
                                        "overflow errors",
                            .default_value = std::exp(30.0),
                            .validator = positive<double>(),
                            .store = in_struct(&ViscoplastUtils::ErrorRegistrationSettings::
                                    max_plastic_strain_deriv_incr)})},
                {.description = "Settings for registering errors within the procedures used for "
                                "constitutive update",
                    .required = false})},
        {.description = "Versatile transversely isotropic (or isotropic) viscoplasticity model for "
                        "finite deformations with isotropic hardening, using user-defined "
                        "viscoplasticity laws (flow rule + hardening model)"});
  }

  /*----------------------------------------------------------------------*/
  {
    using namespace Core::IO::InputSpecBuilders::Validators;

    known_materials[Core::Materials::mvl_reformulated_Johnson_Cook] = group(
        "MAT_ViscoplasticLawReformulatedJohnsonCook",
        {
            parameter<double>(
                "STRAIN_RATE_PREFAC", {.description = "reference plastic strain rate $\\dot{P}_0$",
                                          .validator = positive<double>()}),
            parameter<double>("STRAIN_RATE_EXP_FAC",
                {.description = "exponential factor of plastic strain rate $C$",
                    .validator = positive<double>()}),
            parameter<double>("INIT_YIELD_STRENGTH",
                {.description = "initial yield strength of the material $A_0$",
                    .validator = positive<double>()}),
            parameter<double>("ISOTROP_HARDEN_PREFAC",
                {.description =
                        "prefactor of the isotropic hardening stress / hardening modulus $B_0$",
                    .validator = positive_or_zero<double>()}),
            parameter<double>("ISOTROP_HARDEN_EXP",
                {.description = "exponent of the isotropic hardening stress $n$",
                    .validator = positive_or_zero<double>()}),
            parameter<double>("REF_TEMPERATURE",
                {.description = "reference temperature $T_0$ for evaluating the "
                                "yield strength $A_0$ and the hardening "
                                "modulus $B_0$. Has no effect for isothermal simulations.",
                    .validator = positive<double>()}),
            parameter<double>(
                "MELT_TEMPERATURE", {.description = "melting temperature $T_{\\mathrm{melt}}$. Has "
                                                    "no effect for isothermal simulations.",
                                        .validator = positive<double>()}),
            parameter<double>("TEMPERATURE_SENS",
                {.description =
                        "temperature sensitivity $m$. Has no effect for isothermal simulations.",
                    .validator = positive<double>()}),

        },
        {.description = "Reformulation of the Johnson-Cook viscoplastic law (comprising flow "
                        "rule $\\dot{P} = \\dot{P}_0 \\exp \\left( \\frac{ \\sigma_{eq}}{C "
                        "\\sigma_{\\mathrm{Y}}} - \\frac{1}{C} \\right) - \\dot{P}_0$ and "
                        "hardening law $\\sigma_{\\mathrm{Y}} = (A_0 + "
                        "B_0 \\cdot P^{n}) \\cdot (1 - \\frac{T^m - "
                        "T_{0}^m}{T_{\\mathrm{melt}}^m - T_{0}^m})$), "
                        "as shown in Mareau et al. (Mechanics of Materials 143, 2020)"});
  }

  /*----------------------------------------------------------------------*/
  // integration point based and scalar dependent interpolation between two materials
  {
    known_materials[Core::Materials::m_sc_dep_interp] = group("MAT_ScDepInterp",
        {
            parameter<int>("IDMATZEROSC", {.description = "material for lambda equal to zero"}),
            parameter<int>("IDMATUNITSC", {.description = "material for lambda equal to one"}),
        },
        {.description = "integration point based and scalar dependent interpolation between two "
                        "materials"});
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
            parameter<int>(
                "DOFIDREACSCALAR", {.description = "Id of DOF within scalar transport problem, "
                                                   "which controls the reaction"}),
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
            parameter<int>(
                "DOFIDREACSCALAR", {.description = "Id of DOF within scalar transport problem, "
                                                   "which controls the reaction"}),
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
             * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if
             * needed.
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
             * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if
             * needed.
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
             * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if
             * needed.
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
             * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if
             * needed.
             */
            parameter<double>("INTERACTIONRADIUS",
                {.description = "radius of a circular cross-section which is EXCLUSIVELY used to "
                                "evaluate interactions such as contact, potentials, ...",
                    .default_value = -1.0}),
        },
        {.description = "material parameters for a Kirchhoff-Love type beam element based on "
                        "hyperelastic "
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
             * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if
             * needed.
             */
            parameter<double>("INTERACTIONRADIUS",
                {.description = "radius of a circular cross-section which is EXCLUSIVELY used to "
                                "evaluate interactions such as contact, potentials, ...",
                    .default_value = -1.0}),
        },
        {.description = "material parameters for a Kirchhoff-Love type beam element based on "
                        "hyperelastic "
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
             * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if
             * needed.
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
             * elliptic, ...) and corresponding necessary dimensions (radius, sizes, ...) if
             * needed.
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

  // particle material pd
  {
    known_materials[Core::Materials::m_particle_pd] = group("MAT_ParticlePD",
        {
            parameter<double>("INITRADIUS", {.description = "initial radius"}),
            parameter<double>("INITDENSITY", {.description = "mass density"}),
            parameter<double>("YOUNG", {.description = "Young's modulus"}),
            parameter<double>("CRITICAL_STRETCH", {.description = "critical stretch"}),
        },
        {.description = "particle material for PD"});
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
            parameter<std::vector<int>>(
                "MATIDS", {.description = "list material IDs of the summands"}),
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
    known_materials[Core::Materials::mix_prestress_strategy_prescribed] =
        group("MIX_Prestress_Strategy_Prescribed",
            {
                interpolated_input_field<Core::LinAlg::SymmetricTensor<double, 3, 3>>(
                    "PRESTRETCH", {.description = "Field of a symmetric prestretch tensor."}),
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
                interpolated_input_field<Core::LinAlg::SymmetricTensor<double, 3, 3>>("PRESTRETCH",
                    {.description = "Optional initial prestretch tensor used as starting value. "
                                    "If not provided, the identity tensor is used.",
                        .default_value = Core::LinAlg::TensorGenerators::identity<double, 3, 3>}),
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
                {.description = "Id of the time function to scale the deposition stretch "
                                "(Default: 0=None)",
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
    using namespace Core::IO::InputSpecBuilders::Validators;
    known_materials[Core::Materials::mix_remodelfiber_ssi] = group(
        "MIX_Constituent_SsiRemodelFiber",
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
                {.description = "Id of the time function to scale the deposition stretch "
                                "(Default: 0=None)",
                    .default_value = 0}),
            parameter<bool>("INELASTIC_GROWTH",
                {.description = "Mixture rule has inelastic growth (default false)",
                    .default_value = false}),
            parameter<std::optional<int>>("GROWTH_SCALAR_ID",
                {.description =
                        "Index of the corresponding growth scalar material in the scatra matlist "
                        "(leave unset to disable)",
                    .validator = null_or(positive_or_zero<int>())}),
            parameter<std::optional<int>>("REMODELING_SCALAR_ID",
                {.description =
                        "Index of the corresponding remodeling scalar material in the scatra "
                        "matlist (leave unset to disable)",
                    .validator = null_or(positive_or_zero<int>())}),
            parameter<std::optional<int>>("NONLOCAL_STIMULUS_SCALAR_ID",
                {.description = "Index of the non-local stimulus scalar in the scatra matlist "
                                "(leave unset to disable)",
                    .validator = null_or(positive_or_zero<int>())}),
            parameter<bool>("IMPLICIT_INTEGRATION",
                {.description =
                        "Integrate growth_scalar and lambda_r implicitly at each Newton iteration. "
                        "Default: false.",
                    .default_value = false}),
        },
        {.description =
                "A 1D constituent where the g&r evolution ODEs can be solved either as own scatra "
                "dofs (using GROWTH_SCALAR_ID & REMODELING_SCALAR_ID) or Gauss-point wise based on "
                "a non-local stimulus (using NONLOCAL_STIMULUS_SCALAR_ID)."});
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
                {.description = "Id of the time function to scale the deposition stretch "
                                "(Default: 0=None)",
                    .default_value = 0}),
            parameter<bool>("INELASTIC_GROWTH",
                {.description = "Mixture rule has inelastic growth (default false)",
                    .default_value = false}),
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
                {.description = "Id of the time function to scale the deposition stretch "
                                "(Default: 0=None)",
                    .default_value = 0}),
            parameter<bool>("INELASTIC_GROWTH",
                {.description = "Mixture rule has inelastic growth (default false)",
                    .default_value = false}),
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
            parameter<std::vector<double>>(
                "DISDENSINIT", {.description = "vector containing NUMSLIPSETS entries for the "
                                               "initial dislocation density",
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
                        "(optional) vector containing NUMTWINSETS microstructural parameters "
                        "that "
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
        {.description = "Crystal plasticity. Direction requirements: 3 directions; "
                        "`FIBER1`--`FIBER3` are the columns of the crystal-to-global rotation "
                        "matrix."});
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
