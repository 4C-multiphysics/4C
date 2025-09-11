// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_function.hpp"

#include "4C_global_data.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_utils.hpp"
#include "4C_utils_function_manager.hpp"

#include <memory>
#include <string>

FOUR_C_NAMESPACE_OPEN
namespace
{
  template <int dim>
  std::shared_ptr<Core::Utils::FunctionOfAnything> create_poro_function(
      const Core::IO::InputParameterContainer& container)
  {
    const auto type = container.get<std::string>("POROMULTIPHASESCATRA_FUNCTION");
    const auto& parameters = container.group("PARAMS");

    if (type == "TUMOR_GROWTH_LAW_HEAVISIDE")
    {
      return std::make_shared<PoroPressureBased::TumorGrowthLawHeaviside<dim>>(
          PoroPressureBased::TumorGrowthLawHeavisideParameters{
              .gamma_T_growth = parameters.get<double>("gamma_T_growth"),
              .w_nl_crit = parameters.get<double>("w_nl_crit"),
              .w_nl_env = parameters.get<double>("w_nl_env"),
              .lambda = parameters.get<double>("lambda"),
              .p_t_crit = parameters.get<double>("p_t_crit"),
          });
    }
    if (type == "NECROSIS_LAW_HEAVISIDE")
    {
      return std::make_shared<PoroPressureBased::NecrosisLawHeaviside<dim>>(
          PoroPressureBased::NecrosisLawHeavisideParameters{
              .gamma_t_necr = parameters.get<double>("gamma_t_necr"),
              .w_nl_crit = parameters.get<double>("w_nl_crit"),
              .w_nl_env = parameters.get<double>("w_nl_env"),
              .delta_a_t = parameters.get<double>("delta_a_t"),
              .p_t_crit = parameters.get<double>("p_t_crit"),
          });
    }
    if (type == "OXYGEN_CONSUMPTION_LAW_HEAVISIDE")
    {
      return std::make_shared<PoroPressureBased::OxygenConsumptionLawHeaviside<dim>>(
          PoroPressureBased::OxygenConsumptionLawHeavisideParameters{
              .gamma_nl_growth = parameters.get<double>("gamma_nl_growth"),
              .gamma_0_nl = parameters.get<double>("gamma_0_nl"),
              .w_nl_crit = parameters.get<double>("w_nl_crit"),
              .w_nl_env = parameters.get<double>("w_nl_env"),
              .p_t_crit = parameters.get<double>("p_t_crit"),
          });
    }
    if (type == "TUMOR_GROWTH_LAW_HEAVISIDE_OXY")
    {
      return std::make_shared<PoroPressureBased::TumorGrowthLawHeavisideOxy<dim>>(
          PoroPressureBased::TumorGrowthLawHeavisideNecroOxyParameters{
              .gamma_T_growth = parameters.get<double>("gamma_T_growth"),
              .w_nl_crit = parameters.get<double>("w_nl_crit"),
              .w_nl_env = parameters.get<double>("w_nl_env"),
              .lambda = parameters.get<double>("lambda"),
              .p_t_crit = parameters.get<double>("p_t_crit"),
          });
    }
    if (type == "TUMOR_GROWTH_LAW_HEAVISIDE_NECRO")
    {
      return std::make_shared<PoroPressureBased::TumorGrowthLawHeavisideNecro<dim>>(
          PoroPressureBased::TumorGrowthLawHeavisideNecroOxyParameters{
              .gamma_T_growth = parameters.get<double>("gamma_T_growth"),
              .w_nl_crit = parameters.get<double>("w_nl_crit"),
              .w_nl_env = parameters.get<double>("w_nl_env"),
              .lambda = parameters.get<double>("lambda"),
              .p_t_crit = parameters.get<double>("p_t_crit"),
          });
    }
    if (type == "OXYGEN_TRANSVASCULAR_EXCHANGE_LAW_CONT")
    {
      return std::make_shared<PoroPressureBased::OxygenTransvascularExchangeLawCont<dim>>(
          PoroPressureBased::OxygenTransvascularExchangeLawContParameters{
              .n = parameters.get<double>("n"),
              .Pb50 = parameters.get<double>("Pb50"),
              .CaO2_max = parameters.get<double>("CaO2_max"),
              .alpha_bl_eff = parameters.get<double>("alpha_bl_eff"),
              .gamma_rho_SV = parameters.get<double>("gamma_rho_SV"),
              .rho_oxy = parameters.get<double>("rho_oxy"),
              .rho_IF = parameters.get<double>("rho_IF"),
              .rho_bl = parameters.get<double>("rho_bl"),
              .alpha_IF = parameters.get<double>("alpha_IF"),
          });
    }
    if (type == "OXYGEN_TRANSVASCULAR_EXCHANGE_LAW_DISC")
    {
      return std::make_shared<PoroPressureBased::OxygenTransvascularExchangeLawDisc<dim>>(
          PoroPressureBased::OxygenTransvascularExchangeLawDiscParameters{
              .n = parameters.get<double>("n"),
              .Pb50 = parameters.get<double>("Pb50"),
              .CaO2_max = parameters.get<double>("CaO2_max"),
              .alpha_bl_eff = parameters.get<double>("alpha_bl_eff"),
              .gamma_rho = parameters.get<double>("gamma_rho"),
              .rho_oxy = parameters.get<double>("rho_oxy"),
              .rho_IF = parameters.get<double>("rho_IF"),
              .rho_bl = parameters.get<double>("rho_bl"),
              .S2_max = parameters.get<double>("S2_max"),
              .alpha_IF = parameters.get<double>("alpha_IF"),
          });
    }
    if (type == "LUNG_OXYGEN_EXCHANGE_LAW")
    {
      return std::make_shared<PoroPressureBased::LungOxygenExchangeLaw<dim>>(
          PoroPressureBased::LungOxygenExchangeLawParameters{
              .rho_oxy = parameters.get<double>("rho_oxy"),
              .DiffAdVTLC = parameters.get<double>("DiffAdVTLC"),
              .alpha_oxy = parameters.get<double>("alpha_oxy"),
              .rho_air = parameters.get<double>("rho_air"),
              .rho_bl = parameters.get<double>("rho_bl"),
              .n = parameters.get<double>("n"),
              .P_oB50 = parameters.get<double>("P_oB50"),
              .NC_Hb = parameters.get<double>("NC_Hb"),
              .P_atmospheric = parameters.get<double>("P_atmospheric"),
              .volfrac_blood_ref = parameters.get<double>("volfrac_blood_ref")});
    }
    if (type == "LUNG_CARBONDIOXIDE_EXCHANGE_LAW")
    {
      return std::make_shared<PoroPressureBased::LungCarbonDioxideExchangeLaw<dim>>(
          PoroPressureBased::LungCarbonDioxideExchangeLawParameters{
              .rho_CO2 = parameters.get<double>("rho_CO2"),
              .DiffsolAdVTLC = parameters.get<double>("DiffsolAdVTLC"),
              .pH = parameters.get<double>("pH"),
              .rho_air = parameters.get<double>("rho_air"),
              .rho_bl = parameters.get<double>("rho_bl"),
              .rho_oxy = parameters.get<double>("rho_oxy"),
              .n = parameters.get<double>("n"),
              .P_oB50 = parameters.get<double>("P_oB50"),
              .C_Hb = parameters.get<double>("C_Hb"),
              .NC_Hb = parameters.get<double>("NC_Hb"),
              .alpha_oxy = parameters.get<double>("alpha_oxy"),
              .P_atmospheric = parameters.get<double>("P_atmospheric"),
              .ScalingFormmHg = parameters.get<double>("ScalingFormmHg"),
              .volfrac_blood_ref = parameters.get<double>("volfrac_blood_ref")});
    }
    FOUR_C_THROW("Wrong type of POROMULTIPHASESCATRA_FUNCTION");
  }


  template <int dim>
  std::shared_ptr<Core::Utils::FunctionOfAnything> try_create_poro_function(
      const std::vector<Core::IO::InputParameterContainer>& parameters)
  {
    if (parameters.size() != 1) return nullptr;

    const auto& container = parameters.front();

    if (container.get_if<std::string>("POROMULTIPHASESCATRA_FUNCTION") != nullptr)
    {
      return create_poro_function<dim>(container);
    }
    return std::shared_ptr<Core::Utils::FunctionOfAnything>(nullptr);
  }

  auto try_create_poro_function_dispatch(
      const std::vector<Core::IO::InputParameterContainer>& parameters)
  {
    switch (Global::Problem::instance()->n_dim())
    {
      case 1:
        return try_create_poro_function<1>(parameters);
      case 2:
        return try_create_poro_function<2>(parameters);
      case 3:
        return try_create_poro_function<3>(parameters);
      default:
        FOUR_C_THROW("Unsupported dimension {}.", Global::Problem::instance()->n_dim());
    }
  }
}  // namespace

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
PoroPressureBased::PoroMultiPhaseScaTraFunction<dim>::PoroMultiPhaseScaTraFunction()
    : order_checked_(false)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void PoroPressureBased::add_valid_poro_functions(Core::Utils::FunctionManager& function_manager)
{
  using namespace Core::IO::InputSpecBuilders;

  auto spec = one_of({
      all_of({
          deprecated_selection<std::string>("POROMULTIPHASESCATRA_FUNCTION",
              {"TUMOR_GROWTH_LAW_HEAVISIDE", "TUMOR_GROWTH_LAW_HEAVISIDE_OXY",
                  "TUMOR_GROWTH_LAW_HEAVISIDE_NECRO"}),
          group("PARAMS",
              {
                  parameter<double>("gamma_T_growth"),
                  parameter<double>("w_nl_crit"),
                  parameter<double>("w_nl_env"),
                  parameter<double>("lambda", {.default_value = 0.0}),
                  parameter<double>("p_t_crit", {.default_value = 1.0e9}),
              }),
      }),
      all_of({
          deprecated_selection<std::string>(
              "POROMULTIPHASESCATRA_FUNCTION", {"NECROSIS_LAW_HEAVISIDE"}),
          group("PARAMS",
              {
                  parameter<double>("gamma_t_necr"),
                  parameter<double>("w_nl_crit"),
                  parameter<double>("w_nl_env"),
                  parameter<double>("delta_a_t", {.default_value = 0.0}),
                  parameter<double>("p_t_crit", {.default_value = 1.0e9}),
              }),
      }),
      all_of({
          deprecated_selection<std::string>(
              "POROMULTIPHASESCATRA_FUNCTION", {"OXYGEN_CONSUMPTION_LAW_HEAVISIDE"}),
          group("PARAMS",
              {
                  parameter<double>("gamma_nl_growth"),
                  parameter<double>("gamma_0_nl"),
                  parameter<double>("w_nl_crit"),
                  parameter<double>("w_nl_env"),
                  parameter<double>("p_t_crit", {.default_value = 1.0e9}),
              }),
      }),
      all_of({
          deprecated_selection<std::string>(
              "POROMULTIPHASESCATRA_FUNCTION", {"OXYGEN_TRANSVASCULAR_EXCHANGE_LAW_CONT"}),
          group("PARAMS",
              {
                  parameter<double>("n"),
                  parameter<double>("Pb50"),
                  parameter<double>("CaO2_max"),
                  parameter<double>("alpha_bl_eff"),
                  parameter<double>("gamma_rho_SV"),
                  parameter<double>("rho_oxy"),
                  parameter<double>("rho_IF"),
                  parameter<double>("rho_bl"),
                  parameter<double>("alpha_IF"),
              }),
      }),
      all_of({
          deprecated_selection<std::string>(
              "POROMULTIPHASESCATRA_FUNCTION", {"OXYGEN_TRANSVASCULAR_EXCHANGE_LAW_DISC"}),
          group("PARAMS",
              {
                  parameter<double>("n"),
                  parameter<double>("Pb50"),
                  parameter<double>("CaO2_max"),
                  parameter<double>("alpha_bl_eff"),
                  parameter<double>("gamma_rho"),
                  parameter<double>("rho_oxy"),
                  parameter<double>("rho_IF"),
                  parameter<double>("rho_bl"),
                  parameter<double>("S2_max"),
                  parameter<double>("alpha_IF"),
              }),
      }),
      all_of({
          deprecated_selection<std::string>(
              "POROMULTIPHASESCATRA_FUNCTION", {"LUNG_OXYGEN_EXCHANGE_LAW"}),
          group("PARAMS",
              {parameter<double>("rho_oxy"), parameter<double>("DiffAdVTLC"),
                  parameter<double>("alpha_oxy"), parameter<double>("rho_air"),
                  parameter<double>("rho_bl"), parameter<double>("n"), parameter<double>("P_oB50"),
                  parameter<double>("NC_Hb"), parameter<double>("P_atmospheric"),
                  parameter<double>("volfrac_blood_ref")}),
      }),
      all_of({
          deprecated_selection<std::string>(
              "POROMULTIPHASESCATRA_FUNCTION", {"LUNG_CARBONDIOXIDE_EXCHANGE_LAW"}),
          group("PARAMS",
              {parameter<double>("rho_CO2"), parameter<double>("DiffsolAdVTLC"),
                  parameter<double>("pH"), parameter<double>("rho_air"),
                  parameter<double>("rho_bl"), parameter<double>("rho_oxy"), parameter<double>("n"),
                  parameter<double>("P_oB50"), parameter<double>("C_Hb"),
                  parameter<double>("NC_Hb"), parameter<double>("alpha_oxy"),
                  parameter<double>("P_atmospheric"), parameter<double>("ScalingFormmHg"),
                  parameter<double>("volfrac_blood_ref")}),
      }),
  });

  function_manager.add_function_definition(std::move(spec), try_create_poro_function_dispatch);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
PoroPressureBased::TumorGrowthLawHeaviside<dim>::TumorGrowthLawHeaviside(
    const TumorGrowthLawHeavisideParameters& parameters)
    : PoroMultiPhaseScaTraFunction<dim>(), parameters_(parameters)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
void PoroPressureBased::TumorGrowthLawHeaviside<dim>::check_order(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants) const
{
  // safety check for correct ordering of variables and constants
  // they should have been added in exactly the same way in fluidporo_multiphase_singlereaction,
  // but order might be different if we do not use exactly three fluid phases
  if (variables[1].first != "p2")
    FOUR_C_THROW("wrong order in variable vector, p2 not at position 2");
  if (variables[4].first != "S2")
    FOUR_C_THROW("wrong order in variable vector, S2 not at position 5");
  if (variables[6].first != "porosity")
    FOUR_C_THROW("wrong order in variable vector, porosity not at position 7");
  if (variables[7].first != "phi1")
    FOUR_C_THROW("wrong order in variable vector, phi1 (oxygen mass fraction) not at position 8");
  if (variables[8].first != "phi2")
    FOUR_C_THROW("wrong order in variable vector, phi2 (necrotic mass fraction) not at position 9");

  // order is correct
  this->order_checked_ = true;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
double PoroPressureBased::TumorGrowthLawHeaviside<dim>::evaluate(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // Check order (only once since it does not change)
  if (not this->order_checked_) check_order(variables, constants);

  // read variables and constants (order is crucial)
  const double p2 = variables[1].second;
  const double S2 = variables[4].second;
  const double porosity = variables[6].second;
  const double oxy_mass_frac = variables[7].second;
  const double necr_frac = variables[8].second;

  // evaluate heaviside
  const double heaviside_oxy((oxy_mass_frac - parameters_.w_nl_crit) > 0. ? 1. : 0.);
  const double heaviside_pres((parameters_.p_t_crit - p2) > 0. ? 1. : 0.);
  const double macaulay = parameters_.gamma_T_growth * (oxy_mass_frac - parameters_.w_nl_crit) /
                          (parameters_.w_nl_env - parameters_.w_nl_crit) * heaviside_oxy;

  // evaluate function
  const double functval = (macaulay * heaviside_pres) * (1 - necr_frac) * porosity * S2 -
                          parameters_.lambda * porosity * oxy_mass_frac * S2;
  return functval;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
std::vector<double> PoroPressureBased::TumorGrowthLawHeaviside<dim>::evaluate_derivative(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // create derivative vector (should have size of variables)
  std::vector<double> deriv(variables.size(), 0.0);

  // read variables and constants
  const double p2 = variables[1].second;
  const double S2 = variables[4].second;
  const double porosity = variables[6].second;
  const double oxy_mass_frac = variables[7].second;
  const double necr_frac = variables[8].second;

  // evaluate heaviside
  const double heaviside_oxy((oxy_mass_frac - parameters_.w_nl_crit) > 0. ? 1. : 0.);
  const double heaviside_pres((parameters_.p_t_crit - p2) > 0. ? 1. : 0.);
  const double macaulay = parameters_.gamma_T_growth * (oxy_mass_frac - parameters_.w_nl_crit) /
                          (parameters_.w_nl_env - parameters_.w_nl_crit) * heaviside_oxy;

  // set saturation derivs w.r.t. S2
  const double saturationderiv = (macaulay * heaviside_pres) * (1 - necr_frac) * porosity -
                                 parameters_.lambda * necr_frac * porosity;
  deriv[4] = saturationderiv;

  // set porosity derivs
  const double porosityderiv =
      (macaulay * heaviside_pres) * (1 - necr_frac) * S2 - parameters_.lambda * necr_frac * S2;
  deriv[6] = porosityderiv;

  // set scalar derivs w.r.t. phi1
  const double oxygenderiv = parameters_.gamma_T_growth * heaviside_oxy * heaviside_pres * 1.0 /
                             (parameters_.w_nl_env - parameters_.w_nl_crit) * (1 - necr_frac) *
                             porosity * S2;
  deriv[7] = oxygenderiv;

  // set scalar derivs w.r.t. phi2
  const double necroticderiv =
      -(macaulay * heaviside_pres) * porosity * S2 - parameters_.lambda * porosity * S2;
  deriv[8] = necroticderiv;

  return deriv;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
PoroPressureBased::NecrosisLawHeaviside<dim>::NecrosisLawHeaviside(
    const NecrosisLawHeavisideParameters& parameters)
    : PoroMultiPhaseScaTraFunction<dim>(), parameters_(parameters)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
void PoroPressureBased::NecrosisLawHeaviside<dim>::check_order(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants) const
{
  // safety check for correct ordering of variables and constants
  // they should have been added in exactly the same way in scatra_ele_calc_multiporo_reaction,
  // but order might be different if we do not use exactly three fluid phases
  if (constants[1].first != "p2")
    FOUR_C_THROW("wrong order in variable vector, p2 not at position 2");
  if (constants[4].first != "S2")
    FOUR_C_THROW("wrong order in variable vector, S2 not at position 5");
  if (constants[6].first != "porosity")
    FOUR_C_THROW("wrong order in variable vector, porosity not at position 7");
  if (variables[0].first != "phi1")
    FOUR_C_THROW("wrong order in variable vector, phi1 (oxygen mass fraction) not at position 1");
  if (variables[1].first != "phi2")
    FOUR_C_THROW("wrong order in variable vector, phi1 (necrotic mass fraction) not at position 2");

  // order is correct
  this->order_checked_ = true;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
double PoroPressureBased::NecrosisLawHeaviside<dim>::evaluate(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // Check order (only once since it does not change)
  if (not this->order_checked_) check_order(variables, constants);

  // read variables and constants (order is crucial)
  const double p2 = constants[1].second;
  const double S2 = constants[4].second;
  const double porosity = constants[6].second;
  const double oxy_mass_frac = variables[0].second;
  const double necr_frac = variables[1].second;

  // evaluate heaviside
  const double heaviside_oxy((-(oxy_mass_frac - parameters_.w_nl_crit)) > 0. ? 1. : 0.);
  const double macaulay = -parameters_.gamma_t_necr * (oxy_mass_frac - parameters_.w_nl_crit) /
                          (parameters_.w_nl_env - parameters_.w_nl_crit) * heaviside_oxy;
  const double heaviside_pres((p2 - parameters_.p_t_crit) > 0. ? 1. : 0.);

  // evaluate the function
  const double functval =
      (1.0 - necr_frac) * S2 * porosity * (macaulay + parameters_.delta_a_t * heaviside_pres);

  return functval;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
std::vector<double> PoroPressureBased::NecrosisLawHeaviside<dim>::evaluate_derivative(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // create derivative vector (should have size of variables)
  std::vector<double> deriv(variables.size(), 0.0);

  if (variables[0].first == "phi1")  // maindiag-derivative
  {
    // read variables and constants (order is crucial)
    const double p2 = constants[1].second;
    const double S2 = constants[4].second;
    const double porosity = constants[6].second;
    const double oxy_mass_frac = variables[0].second;
    const double necr_frac = variables[1].second;

    // evaluate heaviside
    const double heaviside_oxy((-(oxy_mass_frac - parameters_.w_nl_crit)) > 0. ? 1. : 0.);
    const double macaulay = -parameters_.gamma_t_necr * (oxy_mass_frac - parameters_.w_nl_crit) /
                            (parameters_.w_nl_env - parameters_.w_nl_crit) * heaviside_oxy;
    const double heaviside_pres((p2 - parameters_.p_t_crit) > 0. ? 1. : 0.);

    // derivative w.r.t. oxygen mass fraction
    const double oxy_deriv = (1.0 - necr_frac) * porosity * S2 *
                             (-parameters_.gamma_t_necr * heaviside_oxy * 1.0 /
                                 (parameters_.w_nl_env - parameters_.w_nl_crit));
    deriv[0] = oxy_deriv;

    // derivative w.r.t. necrotic cell mass fraction
    const double necro_deriv =
        porosity * S2 * (macaulay + parameters_.delta_a_t * heaviside_pres) * (-1.0);
    deriv[1] = necro_deriv;
  }
  else if (variables[0].first == "p1")  // OD-derivative
  {
    // read variables and constants (order is crucial)
    const double S2 = variables[4].second;
    const double porosity = variables[6].second;
    const double p2 = variables[1].second;
    const double oxy_mass_frac = constants[0].second;
    const double necr_frac = constants[1].second;

    // evaluate heaviside
    const double heaviside_oxy((-(oxy_mass_frac - parameters_.w_nl_crit)) > 0. ? 1. : 0.);
    const double macaulay = -parameters_.gamma_t_necr * (oxy_mass_frac - parameters_.w_nl_crit) /
                            (parameters_.w_nl_env - parameters_.w_nl_crit) * heaviside_oxy;
    const double heaviside_pres((p2 - parameters_.p_t_crit) > 0. ? 1. : 0.);

    // derivative w.r.t. tumor cell saturation S2
    const double tc_deriv =
        (1.0 - necr_frac) * porosity * (macaulay + parameters_.delta_a_t * heaviside_pres);
    deriv[4] = tc_deriv;

    // derivative w.r.t. porosity
    const double poro_deriv =
        (1.0 - necr_frac) * S2 * (macaulay + parameters_.delta_a_t * heaviside_pres);
    deriv[6] = poro_deriv;

    // Note: no pressure derivative, only coupling is with heaviside --> derivative zero
  }
  else
    FOUR_C_THROW("Something went wrong in derivative evaluation of NECROSIS_LAW_HEAVISIDE");

  return deriv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
PoroPressureBased::OxygenConsumptionLawHeaviside<dim>::OxygenConsumptionLawHeaviside(
    const OxygenConsumptionLawHeavisideParameters& parameters)
    : PoroMultiPhaseScaTraFunction<dim>(), parameters_(parameters)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
void PoroPressureBased::OxygenConsumptionLawHeaviside<dim>::check_order(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants) const
{
  // safety check for correct ordering of variables and constants
  // they should have been added in exactly the same way in scatra_ele_calc_multiporo_reaction,
  // but order might be different if we do not use exactly three fluid phases
  if (constants[1].first != "p2")
    FOUR_C_THROW("wrong order in variable vector, p2 not at position 2");
  if (constants[4].first != "S2")
    FOUR_C_THROW("wrong order in variable vector, S2 not at position 5");
  if (constants[6].first != "porosity")
    FOUR_C_THROW("wrong order in variable vector, porosity not at position 7");
  if (variables[0].first != "phi1")
    FOUR_C_THROW("wrong order in variable vector, phi1 (oxygen mass fraction) not at position 1");
  if (variables[1].first != "phi2")
    FOUR_C_THROW("wrong order in variable vector, phi1 (necrotic mass fraction) not at position 2");

  // order is correct
  this->order_checked_ = true;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
double PoroPressureBased::OxygenConsumptionLawHeaviside<dim>::evaluate(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // Check order (only once since it does not change)
  if (not this->order_checked_) check_order(variables, constants);

  // read variables and constants (order is crucial)
  const double S2 = constants[4].second;
  const double porosity = constants[6].second;
  const double p2 = constants[1].second;
  const double oxy_mass_frac = variables[0].second;
  const double necr_frac = variables[1].second;

  // evaluate heaviside
  const double heaviside_oxy((oxy_mass_frac - parameters_.w_nl_crit) > 0. ? 1. : 0.);
  const double macaulay = parameters_.gamma_nl_growth * (oxy_mass_frac - parameters_.w_nl_crit) /
                          (parameters_.w_nl_env - parameters_.w_nl_crit) * heaviside_oxy;
  const double heaviside_pres((parameters_.p_t_crit - p2) > 0. ? 1. : 0.);

  // evaluate the function
  const double functval = (1.0 - necr_frac) * S2 * porosity *
                          (macaulay * heaviside_pres +
                              parameters_.gamma_0_nl * sin(std::numbers::pi / 2.0 * oxy_mass_frac /
                                                           parameters_.w_nl_env));

  return functval;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
std::vector<double> PoroPressureBased::OxygenConsumptionLawHeaviside<dim>::evaluate_derivative(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // create derivative vector (should have size of variables)
  std::vector<double> deriv(variables.size(), 0.0);

  if (variables[0].first == "phi1")  // maindiag-derivative
  {
    // read variables and constants (order is crucial)
    const double S2 = constants[4].second;
    const double porosity = constants[6].second;
    const double p2 = constants[1].second;
    const double oxy_mass_frac = variables[0].second;
    const double necr_frac = variables[1].second;

    // evaluate heaviside
    const double heaviside_oxy((oxy_mass_frac - parameters_.w_nl_crit) > 0. ? 1. : 0.);
    const double macaulay = parameters_.gamma_nl_growth * (oxy_mass_frac - parameters_.w_nl_crit) /
                            (parameters_.w_nl_env - parameters_.w_nl_crit) * heaviside_oxy;
    const double heaviside_pres((parameters_.p_t_crit - p2) > 0. ? 1. : 0.);

    // derivative w.r.t. oxygen mass fraction
    const double oxy_deriv =
        (1.0 - necr_frac) * S2 * porosity *
        (parameters_.gamma_nl_growth * heaviside_oxy * heaviside_pres * 1.0 /
                (parameters_.w_nl_env - parameters_.w_nl_crit) +
            parameters_.gamma_0_nl * std::numbers::pi / 2.0 / parameters_.w_nl_env *
                cos(std::numbers::pi / 2.0 * oxy_mass_frac / parameters_.w_nl_env));
    deriv[0] = oxy_deriv;

    // derivative w.r.t. necrotic cell mass fraction
    const double necro_deriv =
        S2 * porosity * (-1.0) *
        (macaulay * heaviside_pres +
            parameters_.gamma_0_nl *
                sin(std::numbers::pi / 2.0 * oxy_mass_frac / parameters_.w_nl_env));
    deriv[1] = necro_deriv;
  }
  else if (variables[0].first == "p1")  // OD-derivative
  {
    // read variables and constants (order is crucial)
    const double S2 = variables[4].second;
    const double porosity = variables[6].second;
    const double p2 = variables[1].second;
    const double oxy_mass_frac = constants[0].second;
    const double necr_frac = constants[1].second;

    // evaluate heaviside
    const double heaviside_oxy((oxy_mass_frac - parameters_.w_nl_crit) > 0. ? 1. : 0.);
    const double macaulay = parameters_.gamma_nl_growth * (oxy_mass_frac - parameters_.w_nl_crit) /
                            (parameters_.w_nl_env - parameters_.w_nl_crit) * heaviside_oxy;
    const double heaviside_pres((parameters_.p_t_crit - p2) > 0. ? 1. : 0.);

    // derivative w.r.t. tumor cell saturation S2
    const double tc_deriv = (1.0 - necr_frac) * porosity *
                            (macaulay * heaviside_pres +
                                parameters_.gamma_0_nl * sin(std::numbers::pi / 2.0 *
                                                             oxy_mass_frac / parameters_.w_nl_env));
    deriv[4] = tc_deriv;

    // derivative w.r.t. porosity
    const double poro_deriv =
        (1.0 - necr_frac) * S2 *
        (macaulay * heaviside_pres +
            parameters_.gamma_0_nl *
                sin(std::numbers::pi / 2.0 * oxy_mass_frac / parameters_.w_nl_env));
    deriv[6] = poro_deriv;
  }
  else
    FOUR_C_THROW(
        "Something went wrong in derivative evaluation of OXYGEN_CONSUMPTION_LAW_HEAVISIDE");

  return deriv;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
PoroPressureBased::TumorGrowthLawHeavisideOxy<dim>::TumorGrowthLawHeavisideOxy(
    const TumorGrowthLawHeavisideNecroOxyParameters& parameters)
    : PoroMultiPhaseScaTraFunction<dim>(), parameters_(parameters)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
void PoroPressureBased::TumorGrowthLawHeavisideOxy<dim>::check_order(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants) const
{
  // safety check for correct ordering of variables and constants
  // they should have been added in exactly the same way in scatra_ele_calc_multiporo_reaction,
  // but order might be different if we do not use exactly three fluid phases
  if (constants[1].first != "p2")
    FOUR_C_THROW("wrong order in variable vector, p2 not at position 2");
  if (constants[4].first != "S2")
    FOUR_C_THROW("wrong order in variable vector, S2 not at position 5");
  if (constants[6].first != "porosity")
    FOUR_C_THROW("wrong order in variable vector, porosity not at position 7");
  if (variables[0].first != "phi1")
    FOUR_C_THROW("wrong order in variable vector, phi1 (oxygen mass fraction) not at position 1");
  if (variables[1].first != "phi2")
    FOUR_C_THROW("wrong order in variable vector, phi1 (necrotic mass fraction) not at position 2");

  // order is correct
  this->order_checked_ = true;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
double PoroPressureBased::TumorGrowthLawHeavisideOxy<dim>::evaluate(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // Check order (only once since it does not change)
  if (not this->order_checked_) check_order(variables, constants);

  // read variables and constants (order is crucial)
  const double p2 = constants[1].second;
  const double S2 = constants[4].second;
  const double porosity = constants[6].second;
  const double oxy_mass_frac = variables[0].second;
  const double necr_frac = variables[1].second;

  // evaluate heaviside
  const double heaviside_oxy((oxy_mass_frac - parameters_.w_nl_crit) > 0. ? 1. : 0.);
  const double heaviside_pres((parameters_.p_t_crit - p2) > 0. ? 1. : 0.);
  const double macaulay = parameters_.gamma_T_growth * (oxy_mass_frac - parameters_.w_nl_crit) /
                          (parameters_.w_nl_env - parameters_.w_nl_crit) * heaviside_oxy;

  // evaluate function
  const double functval =
      oxy_mass_frac * S2 * porosity *
      ((macaulay * heaviside_pres) * (1 - necr_frac) - parameters_.lambda * necr_frac);

  return functval;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
std::vector<double> PoroPressureBased::TumorGrowthLawHeavisideOxy<dim>::evaluate_derivative(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // create derivative vector (should have size of variables)
  std::vector<double> deriv(variables.size(), 0.0);

  if (variables[0].first == "phi1")  // maindiag-derivative
  {
    // read variables and constants (order is crucial)
    const double S2 = constants[4].second;
    const double porosity = constants[6].second;
    const double p2 = constants[1].second;
    const double oxy_mass_frac = variables[0].second;
    const double necr_frac = variables[1].second;

    // evaluate heaviside
    const double heaviside_oxy((oxy_mass_frac - parameters_.w_nl_crit) > 0. ? 1. : 0.);
    const double heaviside_pres((parameters_.p_t_crit - p2) > 0. ? 1. : 0.);
    const double macaulay = parameters_.gamma_T_growth * (oxy_mass_frac - parameters_.w_nl_crit) /
                            (parameters_.w_nl_env - parameters_.w_nl_crit) * heaviside_oxy;

    // derivative w.r.t. oxygen mass fraction
    const double oxy_deriv =
        (parameters_.gamma_T_growth * heaviside_oxy * heaviside_pres * 1.0 /
            (parameters_.w_nl_env - parameters_.w_nl_crit) * (1 - necr_frac)) *
            oxy_mass_frac * S2 * porosity +
        ((macaulay * heaviside_pres) * (1 - necr_frac) - parameters_.lambda * necr_frac) * S2 *
            porosity;
    deriv[0] = oxy_deriv;

    // derivative w.r.t. necrotic cell mass fraction
    const double necro_deriv =
        ((macaulay * heaviside_pres) * (-1.0) - parameters_.lambda) * oxy_mass_frac * S2 * porosity;
    deriv[1] = necro_deriv;
  }
  else if (variables[0].first == "p1")  // OD-derivative
  {
    // read variables and constants (order is crucial)
    const double S2 = variables[4].second;
    const double porosity = variables[6].second;
    const double p2 = variables[1].second;
    const double oxy_mass_frac = constants[0].second;
    const double necr_frac = constants[1].second;

    // evaluate heaviside
    const double heaviside_oxy((oxy_mass_frac - parameters_.w_nl_crit) > 0. ? 1. : 0.);
    const double heaviside_pres((parameters_.p_t_crit - p2) > 0. ? 1. : 0.);
    const double macaulay = parameters_.gamma_T_growth * (oxy_mass_frac - parameters_.w_nl_crit) /
                            (parameters_.w_nl_env - parameters_.w_nl_crit) * heaviside_oxy;

    // derivative w.r.t. tumor cell saturation S2
    const double tc_deriv =
        ((macaulay * heaviside_pres) * (1.0 - necr_frac) - parameters_.lambda * necr_frac) *
        oxy_mass_frac * porosity;
    deriv[4] = tc_deriv;

    // derivative w.r.t. porosity
    const double poro_deriv =
        (macaulay * heaviside_pres * (1.0 - necr_frac) - parameters_.lambda * necr_frac) *
        oxy_mass_frac * S2;
    deriv[6] = poro_deriv;
  }
  else
    FOUR_C_THROW("Something went wrong in derivative evaluation of TUMOR_GROWTH_LAW_HEAVISIDE_OXY");

  return deriv;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
PoroPressureBased::TumorGrowthLawHeavisideNecro<dim>::TumorGrowthLawHeavisideNecro(
    const TumorGrowthLawHeavisideNecroOxyParameters& parameters)
    : PoroMultiPhaseScaTraFunction<dim>(), parameters_(parameters)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
void PoroPressureBased::TumorGrowthLawHeavisideNecro<dim>::check_order(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants) const
{
  // safety check for correct ordering of variables and constants
  // they should have been added in exactly the same way in scatra_ele_calc_multiporo_reaction,
  // but order might be different if we do not use exactly three fluid phases
  if (constants[1].first != "p2")
    FOUR_C_THROW("wrong order in variable vector, p2 not at position 2");
  if (constants[4].first != "S2")
    FOUR_C_THROW("wrong order in variable vector, S2 not at position 5");
  if (constants[6].first != "porosity")
    FOUR_C_THROW("wrong order in variable vector, porosity not at position 7");
  if (variables[0].first != "phi1")
    FOUR_C_THROW("wrong order in variable vector, phi1 (oxygen mass fraction) not at position 1");
  if (variables[1].first != "phi2")
    FOUR_C_THROW("wrong order in variable vector, phi1 (necrotic mass fraction) not at position 2");

  // order is correct
  this->order_checked_ = true;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
double PoroPressureBased::TumorGrowthLawHeavisideNecro<dim>::evaluate(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // Check order (only once since it does not change)
  if (not this->order_checked_) check_order(variables, constants);

  // read variables and constants (order is crucial)
  const double p2 = constants[1].second;
  const double S2 = constants[4].second;
  const double porosity = constants[6].second;
  const double oxy_mass_frac = variables[0].second;
  const double necr_frac = variables[1].second;

  // evaluate heaviside
  const double heaviside_oxy((oxy_mass_frac - parameters_.w_nl_crit) > 0. ? 1. : 0.);
  const double heaviside_pres((parameters_.p_t_crit - p2) > 0. ? 1. : 0.);
  const double macaulay = parameters_.gamma_T_growth * (oxy_mass_frac - parameters_.w_nl_crit) /
                          (parameters_.w_nl_env - parameters_.w_nl_crit) * heaviside_oxy;

  // evaluate function
  const double functval =
      porosity * S2 *
      (((macaulay * heaviside_pres) * (1 - necr_frac) - parameters_.lambda * necr_frac) *
              necr_frac +
          parameters_.lambda * necr_frac);

  return functval;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
std::vector<double> PoroPressureBased::TumorGrowthLawHeavisideNecro<dim>::evaluate_derivative(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // create derivative vector (should have size of variables)
  std::vector<double> deriv(variables.size(), 0.0);

  if (variables[0].first == "phi1")  // maindiag-derivative
  {
    // read variables and constants (order is crucial)
    const double p2 = constants[1].second;
    const double S2 = constants[4].second;
    const double porosity = constants[6].second;
    const double oxy_mass_frac = variables[0].second;
    const double necr_frac = variables[1].second;

    // evaluate heaviside
    const double heaviside_oxy((oxy_mass_frac - parameters_.w_nl_crit) > 0. ? 1. : 0.);
    const double heaviside_pres((parameters_.p_t_crit - p2) > 0. ? 1. : 0.);
    const double macaulay = parameters_.gamma_T_growth * (oxy_mass_frac - parameters_.w_nl_crit) /
                            (parameters_.w_nl_env - parameters_.w_nl_crit) * heaviside_oxy;

    // derivative w.r.t. oxygen mass fraction
    const double oxy_deriv = (parameters_.gamma_T_growth * heaviside_oxy * heaviside_pres * 1.0 /
                                 (parameters_.w_nl_env - parameters_.w_nl_crit)) *
                             (1 - necr_frac) * necr_frac * porosity * S2;
    deriv[0] = oxy_deriv;

    // derivative w.r.t. necrotic cell mass fraction
    const double necro_deriv = ((macaulay * heaviside_pres) * (1 - 2.0 * necr_frac) -
                                   2.0 * parameters_.lambda * necr_frac + parameters_.lambda) *
                               porosity * S2;
    deriv[1] = necro_deriv;
  }
  else if (variables[0].first == "p1")  // OD-derivative
  {
    // read variables and constants (order is crucial)
    const double S2 = variables[4].second;
    const double porosity = variables[6].second;
    const double p2 = variables[1].second;
    const double oxy_mass_frac = constants[0].second;
    const double necr_frac = constants[1].second;

    // evaluate heaviside
    const double heaviside_oxy((oxy_mass_frac - parameters_.w_nl_crit) > 0. ? 1. : 0.);
    const double heaviside_pres((parameters_.p_t_crit - p2) > 0. ? 1. : 0.);
    const double macaulay = parameters_.gamma_T_growth * (oxy_mass_frac - parameters_.w_nl_crit) /
                            (parameters_.w_nl_env - parameters_.w_nl_crit) * heaviside_oxy;

    // derivative w.r.t. tumor cell saturation S2
    const double tc_deriv =
        porosity *
        (((macaulay * heaviside_pres) * (1 - necr_frac) - parameters_.lambda * necr_frac) *
                necr_frac +
            parameters_.lambda * necr_frac);
    deriv[4] = tc_deriv;

    // derivative w.r.t. porosity
    const double poro_deriv =
        S2 * (((macaulay * heaviside_pres) * (1 - necr_frac) - parameters_.lambda * necr_frac) *
                     necr_frac +
                 parameters_.lambda * necr_frac);
    deriv[6] = poro_deriv;

    // Note: no pressure derivative, only coupling is with heaviside --> derivative zero
  }
  else
    FOUR_C_THROW(
        "Something went wrong in derivative evaluation of TUMOR_GROWTH_LAW_HEAVISIDE_NECRO");

  return deriv;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
PoroPressureBased::OxygenTransvascularExchangeLawCont<dim>::OxygenTransvascularExchangeLawCont(
    const OxygenTransvascularExchangeLawContParameters& parameters)
    : PoroMultiPhaseScaTraFunction<dim>(), parameters_(parameters)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
void PoroPressureBased::OxygenTransvascularExchangeLawCont<dim>::check_order(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants) const
{
  // safety check for correct ordering of variables and constants
  // they should have been added in exactly the same way in scatra_ele_calc_multiporo_reaction,
  // but order might be different if we do not use exactly three fluid phases
  if (constants[7].first != "VF1")
    FOUR_C_THROW("wrong order in variable vector, porosity not at position 8");
  if (variables[0].first != "phi1")
    FOUR_C_THROW("wrong order in variable vector, phi1 (oxygen mass fraction) not at position 1");
  if (variables[1].first != "phi2")
    FOUR_C_THROW("wrong order in variable vector, phi2 (necrotic mass fraction) not at position 2");
  if (variables[2].first != "phi3")
    FOUR_C_THROW("wrong order in variable vector, phi3 (necrotic mass fraction) not at position 3");
  if (variables[3].first != "phi4")
    FOUR_C_THROW("wrong order in variable vector, phi4 (necrotic mass fraction) not at position 4");

  // order is correct
  this->order_checked_ = true;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
double PoroPressureBased::OxygenTransvascularExchangeLawCont<dim>::evaluate(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // Check order (only once since it does not change)
  if (not this->order_checked_) check_order(variables, constants);

  const double fac_if = parameters_.rho_oxy / parameters_.rho_IF * parameters_.alpha_IF;

  // read variables and constants (order is crucial)
  double VF1 = constants[7].second;
  double oxy_mass_frac_if = variables[0].second;
  double oxy_mass_frac_nv = variables[3].second;

  double Pb = 0.0;
  double CaO2 = oxy_mass_frac_nv * parameters_.rho_bl / parameters_.rho_oxy;
  // safety check --> should not be larger than CaO2_max, which already corresponds to partial
  // pressures of ~250Pa
  CaO2 = std::max(0.0, std::min(CaO2, 1.0 * parameters_.CaO2_max));
  PoroPressureBased::get_oxy_partial_pressure_from_concentration<double>(
      Pb, CaO2, parameters_.CaO2_max, parameters_.Pb50, parameters_.n, parameters_.alpha_bl_eff);

  // evaluate function
  const double heaviside_oxy((Pb - oxy_mass_frac_if / fac_if) > 0. ? 1. : 0.);
  const double functval =
      parameters_.gamma_rho_SV * heaviside_oxy * (Pb - oxy_mass_frac_if / fac_if) * VF1;

  return functval;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
std::vector<double> PoroPressureBased::OxygenTransvascularExchangeLawCont<dim>::evaluate_derivative(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // create derivative vector (should have size of variables)
  std::vector<double> deriv(variables.size(), 0.0);

  const double fac_if = parameters_.rho_oxy / parameters_.rho_IF * parameters_.alpha_IF;

  double VF1 = 0.0, oxy_mass_frac_if = 0.0;

  // define Fad object for evaluation
  using FAD = Sacado::Fad::DFad<double>;
  FAD oxy_mass_frac_nv = 0.0;
  oxy_mass_frac_nv.diff(0, 1);       // independent variable 0 out of a total of 1
  if (variables[0].first == "phi1")  // maindiag-derivative
  {
    // read variables and constants (order is crucial)
    VF1 = constants[7].second;
    oxy_mass_frac_if = variables[0].second;
    oxy_mass_frac_nv.val() = variables[3].second;
  }
  else if (variables[0].first == "p1")  // OD-derivative
  {
    // read variables and constants (order is crucial)
    VF1 = variables[7].second;
    oxy_mass_frac_if = constants[0].second;
    oxy_mass_frac_nv.val() = constants[3].second;
  }
  else
    FOUR_C_THROW(
        "Something went wrong in derivative evaluation of "
        "OXYGEN_TRANSVASCULAR_EXCHANGE_LAW_CONT");

  FAD Pb = 0.0;
  FAD CaO2 = oxy_mass_frac_nv * parameters_.rho_bl / parameters_.rho_oxy;
  // safety check --> should not be larger than CaO2_max, which already corresponds to partial
  // pressures of ~250Pa
  CaO2 = std::max(0.0, std::min(CaO2, 1.0 * parameters_.CaO2_max));
  PoroPressureBased::get_oxy_partial_pressure_from_concentration<FAD>(
      Pb, CaO2, parameters_.CaO2_max, parameters_.Pb50, parameters_.n, parameters_.alpha_bl_eff);
  const double heaviside_oxy((Pb - oxy_mass_frac_if / fac_if) > 0. ? 1. : 0.);

  if (variables[0].first == "phi1")  // maindiag-derivative
  {
    deriv[0] = parameters_.gamma_rho_SV * VF1 * (-1.0 / fac_if) * heaviside_oxy;
    deriv[3] = parameters_.gamma_rho_SV * VF1 * (Pb.fastAccessDx(0)) * heaviside_oxy;
  }
  else if (variables[0].first == "p1")  // OD-derivative
  {
    deriv[7] = parameters_.gamma_rho_SV * (Pb.val() - oxy_mass_frac_if / fac_if) * heaviside_oxy;
  }
  else
    FOUR_C_THROW(
        "Something went wrong in derivative evaluation of "
        "OXYGEN_TRANSVASCULAR_EXCHANGE_LAW_CONT");

  return deriv;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
PoroPressureBased::OxygenTransvascularExchangeLawDisc<dim>::OxygenTransvascularExchangeLawDisc(
    const OxygenTransvascularExchangeLawDiscParameters& parameters)
    : PoroMultiPhaseScaTraFunction<dim>(), parameters_(parameters), pos_oxy_art_(-1), pos_diam_(-1)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
void PoroPressureBased::OxygenTransvascularExchangeLawDisc<dim>::check_order(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants) const
{
  // safety check for correct ordering of variables and constants
  // they should have been added in exactly the same way in scatra_ele_calc_multiporo_reaction,
  // but order might be different if we do not use exactly three fluid phases
  if (variables[0].first != "phi1")
    FOUR_C_THROW("wrong order in variable vector, phi1 (oxygen mass fraction) not at position 1");

  // oxygen in artery
  // we have no neo-vasculature --> at position 2, species 1: oxy in IF, species 2: NTC
  if (variables[2].first == "phi_art1") pos_oxy_art_ = 2;
  // we have no neo-vasculature --> at position 4, species 1: oxy in IF, species 2: NTC,
  // species 3: TAF, species 4: oxy in NV
  if (variables.size() >= 5 && variables[4].first == "phi_art1") pos_oxy_art_ = 4;
  if (pos_oxy_art_ == -1) FOUR_C_THROW("cannot find position of oxygen in arteries");

  // fluid variables
  if (constants[1].first != "p2")
    FOUR_C_THROW("wrong order in constants vector, p2 (pressure of tumor cells) not at position 2");
  if (constants[4].first != "S2")
    FOUR_C_THROW(
        "wrong order in constants vector, S2 (saturation of tumor cells) not at position 5");

  // diameter
  if (constants[8].first == "D") pos_diam_ = 8;
  if (constants.size() >= 11 && constants[10].first == "D") pos_diam_ = 10;
  if (pos_diam_ == -1) FOUR_C_THROW("cannot find position of artery diameter");

  // order is correct
  this->order_checked_ = true;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
double PoroPressureBased::OxygenTransvascularExchangeLawDisc<dim>::evaluate(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // Check order (only once since it does not change)
  if (not this->order_checked_) check_order(variables, constants);

  const double fac_if = parameters_.rho_oxy / parameters_.rho_IF * parameters_.alpha_IF;

  // read variables and constants (order is crucial)
  double oxy_mass_frac_if = variables[0].second;
  double oxy_mass_frac_nv = variables[pos_oxy_art_].second;
  const double D = constants[pos_diam_].second;
  const double S2 = constants[4].second;

  double Pb = 0.0;
  double CaO2 = oxy_mass_frac_nv * parameters_.rho_bl / parameters_.rho_oxy;
  // safety check --> should not be larger than CaO2_max, which already corresponds to partial
  // pressures of ~250Pa
  CaO2 = std::max(0.0, std::min(CaO2, 1.0 * parameters_.CaO2_max));
  PoroPressureBased::get_oxy_partial_pressure_from_concentration<double>(
      Pb, CaO2, parameters_.CaO2_max, parameters_.Pb50, parameters_.n, parameters_.alpha_bl_eff);

  // evaluate function
  const double heaviside_S2 = ((S2 - parameters_.S2_max) > 0. ? 0. : 1.);
  const double functval = parameters_.gamma_rho * std::numbers::pi * D *
                          (Pb - oxy_mass_frac_if / fac_if) * heaviside_S2;

  return functval;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
std::vector<double> PoroPressureBased::OxygenTransvascularExchangeLawDisc<dim>::evaluate_derivative(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // create derivative vector (should have size of variables)
  std::vector<double> deriv(variables.size(), 0.0);

  const double fac_if = parameters_.rho_oxy / parameters_.rho_IF * parameters_.alpha_IF;

  // define Fad object for evaluation
  using FAD = Sacado::Fad::DFad<double>;
  FAD oxy_mass_frac_nv = 0.0;
  oxy_mass_frac_nv.diff(0, 1);  // independent variable 0 out of a total of 1

  // read variables and constants (order is crucial)
  oxy_mass_frac_nv.val() = variables[pos_oxy_art_].second;
  const double D = constants[pos_diam_].second;
  const double S2 = constants[4].second;

  FAD Pb = 0.0;
  FAD CaO2 = oxy_mass_frac_nv * parameters_.rho_bl / parameters_.rho_oxy;
  // safety check --> should not be larger than CaO2_max, which already corresponds to partial
  // pressures of ~250Pa
  CaO2 = std::max(0.0, std::min(CaO2, 1.0 * parameters_.CaO2_max));
  PoroPressureBased::get_oxy_partial_pressure_from_concentration<FAD>(
      Pb, CaO2, parameters_.CaO2_max, parameters_.Pb50, parameters_.n, parameters_.alpha_bl_eff);

  // evaluate function
  const double heaviside_S2 = ((S2 - parameters_.S2_max) > 0. ? 0. : 1.);

  deriv[0] = parameters_.gamma_rho * std::numbers::pi * D * (-1.0 / fac_if) * heaviside_S2;
  deriv[pos_oxy_art_] =
      parameters_.gamma_rho * std::numbers::pi * D * (Pb.fastAccessDx(0)) * heaviside_S2;

  return deriv;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
PoroPressureBased::LungOxygenExchangeLaw<dim>::LungOxygenExchangeLaw(
    const LungOxygenExchangeLawParameters& parameters)
    : PoroMultiPhaseScaTraFunction<dim>(), parameters_(parameters)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
void PoroPressureBased::LungOxygenExchangeLaw<dim>::check_order(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants) const
{
  // safety check for correct ordering of variables and constants
  if (variables[0].first == "phi1")
  {
    if (constants[0].first != "p1")
      FOUR_C_THROW("wrong order in constants vector, P1 (Pressure of air) not at position 1");
    if (constants[3].first != "VF1")
      FOUR_C_THROW(
          "wrong order in constants vector, VF1 (volume fraction of additional porous network "
          "(blood phase)) not at position 4");
    if (constants[4].first != "VFP1")
    {
      FOUR_C_THROW(
          "wrong order in variable vector, p2 (volume fraction pressure of additional porous "
          "network "
          "(blood)) not at position 5");
    }
    if (variables[1].first != "phi2")
      FOUR_C_THROW(
          "wrong order in variable vector, phi2 (oxygen mass fraction in blood) not at position "
          "2");
  }
  else if (variables[0].first == "p1")
  {
    if (variables[3].first != "VF1")
    {
      FOUR_C_THROW(
          "wrong order in variable vector, VF1 (volume fraction of additional porous network "
          "(blood)) not at position 4");
    }
    if (variables[4].first != "VFP1")
    {
      FOUR_C_THROW(
          "wrong order in variable vector, p2 (volume fraction pressure of additional porous "
          "network "
          "(blood)) not at position 5");
    }

    if (constants[0].first != "phi1")
      FOUR_C_THROW(
          "wrong order in variable vector, phi1 (oxygen mass fraction in air) not at position 1");
    if (constants[1].first != "phi2")
      FOUR_C_THROW(
          "wrong order in variable vector, phi2 (oxygen mass fraction in blood) not at position "
          "2");
  }
  else
  {
    FOUR_C_THROW("Variable <{}> not supported on position 0. Wrong order in variable vector! ",
        variables[0].first);
  }

  // order is correct
  this->order_checked_ = true;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
double PoroPressureBased::LungOxygenExchangeLaw<dim>::evaluate(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // Check order of variables and constants vector only once (since it does not change)
  if (not this->order_checked_) check_order(variables, constants);

  // In debug mode, check order of variables and constants vector on every call
#ifdef FOUR_C_ENABLE_ASSERTIONS
  check_order(variables, constants);
#endif

  // read variables (order is crucial)
  const double oxy_mass_frac_air = variables[0].second;
  const double oxy_mass_frac_bl = variables[1].second;

  // read constants (order is crucial)
  const double P_air = constants[0].second;
  const double volfrac_blood = constants[3].second;

  // partial pressure of oxygen in air
  const double P_oA = oxy_mass_frac_air * (P_air + parameters_.P_atmospheric) *
                      parameters_.rho_air / parameters_.rho_oxy;

  // CoB_total is total concentration of oxygen in blood (physically dissolved and bound to
  // hemoglobin)
  const double CoB_total = oxy_mass_frac_bl * parameters_.rho_bl / parameters_.rho_oxy;

  // partial pressure of oxygen in blood
  double P_oB = 0.0;

  // Calculate partial pressure of oxygen in blood
  PoroPressureBased::get_oxy_partial_pressure_from_concentration<double>(
      P_oB, CoB_total, parameters_.NC_Hb, parameters_.P_oB50, parameters_.n, parameters_.alpha_oxy);

  // evaluate function
  const double functval = parameters_.rho_oxy * parameters_.DiffAdVTLC * parameters_.alpha_oxy *
                          (volfrac_blood / parameters_.volfrac_blood_ref) * (P_oA - P_oB);

  return functval;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
std::vector<double> PoroPressureBased::LungOxygenExchangeLaw<dim>::evaluate_derivative(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // In debug mode, check order of variables and constants vector on every call
#ifdef FOUR_C_ENABLE_ASSERTIONS
  check_order(variables, constants);
#endif
  // Check order of variables and constants vector only once (since it does not change)
  if (not this->order_checked_) check_order(variables, constants);

  // create derivative vector (should have size of variables)
  std::vector<double> deriv(variables.size(), 0.0);

  // define Fad object for evaluation
  using FAD = Sacado::Fad::DFad<double>;
  FAD oxy_mass_frac_bl = 0.0;
  oxy_mass_frac_bl.diff(0, 1);  // independent variable 0 out of a total of 1

  double oxy_mass_frac_air = 0.0, P_air = 0.0, volfrac_blood = 0.0;

  if (variables[0].first == "phi1")  // maindiag-derivative
  {
    // read variables and constants (order is crucial)
    oxy_mass_frac_air = variables[0].second;
    oxy_mass_frac_bl.val() = variables[1].second;
    P_air = constants[0].second;
    volfrac_blood = constants[3].second;
  }
  else if (variables[0].first == "p1")  // OD-derivative
  {
    // read variables and constants (order is crucial)
    oxy_mass_frac_air = constants[0].second;
    oxy_mass_frac_bl.val() = constants[1].second;
    P_air = variables[0].second;
    volfrac_blood = variables[3].second;
  }
  else
    FOUR_C_THROW(
        "Derivative w.r.t. <{}> not supported in LUNG_OXYGEN_EXCHANGE_LAW.", variables[0].first);

  // volfrac relation
  const double volfrac_relation = (volfrac_blood / parameters_.volfrac_blood_ref);


  FAD P_oB = 0.0;
  FAD C_oB_total = oxy_mass_frac_bl * parameters_.rho_bl / parameters_.rho_oxy;

  PoroPressureBased::get_oxy_partial_pressure_from_concentration<FAD>(P_oB, C_oB_total,
      parameters_.NC_Hb, parameters_.P_oB50, parameters_.n, parameters_.alpha_oxy);


  if (variables[0].first == "phi1")  // maindiag-derivative
  {
    deriv[0] = parameters_.rho_oxy * parameters_.DiffAdVTLC * volfrac_relation *
               parameters_.alpha_oxy *
               ((P_air + parameters_.P_atmospheric) * parameters_.rho_air / parameters_.rho_oxy);
    deriv[1] = parameters_.rho_oxy * parameters_.DiffAdVTLC * volfrac_relation *
               parameters_.alpha_oxy * (-1.0) * P_oB.fastAccessDx(0);
  }
  else if (variables[0].first == "p1")  // OD-derivative
  {
    // derivatives of dvolfrac/dPressure_fluid not yet included
    // partial pressure of oxygen in air
    const double P_oA = oxy_mass_frac_air * (P_air + parameters_.P_atmospheric) *
                        parameters_.rho_air / parameters_.rho_oxy;
    deriv[0] =
        (parameters_.rho_oxy * parameters_.DiffAdVTLC * volfrac_relation * parameters_.alpha_oxy) *
        ((oxy_mass_frac_air * parameters_.rho_air) /
            parameters_.rho_oxy);  // derivative wrt P_air (dFunc/dP_oA * dP_oA/P_air)

    deriv[3] = parameters_.rho_oxy * parameters_.DiffAdVTLC * parameters_.alpha_oxy *
               (1 / parameters_.volfrac_blood_ref) * (P_oA - P_oB.val());
  }
  else
    FOUR_C_THROW(
        "Derivative w.r.t. <{}> not supported in LUNG_OXYGEN_EXCHANGE_LAW.", variables[0].first);

  return deriv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
PoroPressureBased::LungCarbonDioxideExchangeLaw<dim>::LungCarbonDioxideExchangeLaw(
    const LungCarbonDioxideExchangeLawParameters& parameters)
    : PoroMultiPhaseScaTraFunction<dim>(), parameters_(parameters)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
void PoroPressureBased::LungCarbonDioxideExchangeLaw<dim>::check_order(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants) const
{
  // safety check for correct ordering of variables and constants
  if (variables[0].first == "phi1")
  {
    if (constants[0].first != "p1")
      FOUR_C_THROW("wrong order in constants vector, P1 (Pressure of air) not at position 1");
    if (constants[1].first != "S1")
      FOUR_C_THROW("wrong order in constants vector, S1 (Saturation of air) not at position 2");
    if (constants[3].first != "VF1")
      FOUR_C_THROW("wrong order in constants vector, VF1 (volfrac 1) not at position 4");
    if (constants[4].first != "VFP1")
      FOUR_C_THROW("wrong order in constants vector, p2 (pressure of blood) not at position 5");
    if (variables[1].first != "phi2")
      FOUR_C_THROW(
          "wrong order in variable vector, phi2 (oxygen mass fraction in blood) not at position "
          "2");
  }
  else if (variables[0].first == "p1")
  {
    if (constants[0].first != "phi1")
      FOUR_C_THROW(
          "wrong order in variable vector, phi1 (oxygen mass fraction in air) not at position 1");
    if (constants[1].first != "phi2")
      FOUR_C_THROW(
          "wrong order in variable vector, phi2 (oxygen mass fraction in blood) not at position "
          "2");
    if (constants[2].first != "phi3")
      FOUR_C_THROW(
          "wrong order in variable vector, phi3 (carbon dioxide mass fraction in blood) not at "
          "position "
          "3");
    if (constants[3].first != "phi4")
      FOUR_C_THROW(
          "wrong order in variable vector, phi4 (carbon dioxide mass fraction in blood) not at "
          "position "
          "4");
    if (variables[4].first != "VFP1")
      FOUR_C_THROW(
          "wrong order in variable vector, p2 (pressure of blood) not at "
          "position 5");
    if (variables[3].first != "VF1")
      FOUR_C_THROW("wrong order in variable vector, VF1 (volfrac 1) not at position 4");
  }
  else
  {
    FOUR_C_THROW("Variable <{}> not supported on position 0. Wrong order in variable vector! ",
        variables[0].first);
  }

  // order is correct
  this->order_checked_ = true;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
double PoroPressureBased::LungCarbonDioxideExchangeLaw<dim>::evaluate(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // Check order of variables and constants vector only once (since it does not change)
  if (not this->order_checked_) check_order(variables, constants);

  // In debug mode, check order of variables and constants vector on every call
#ifdef FOUR_C_ENABLE_ASSERTIONS
  check_order(variables, constants);
#endif

  // read variables (order is crucial)
  const double O2_mass_frac_bl = variables[1].second;
  const double CO2_mass_frac_air = variables[2].second;
  const double CO2_mass_frac_bl = variables[3].second;

  // read constants (order is crucial)
  const double P_air = constants[0].second;
  const double volfrac_blood = constants[3].second;

  // partial pressure of carbon dioxide in air
  const double P_CO2A = CO2_mass_frac_air * (P_air + parameters_.P_atmospheric) *
                        parameters_.rho_air / parameters_.rho_CO2;

  // CoB_total is total concentration of oxygen in blood (physically dissolved and bound to
  // hemoglobin)
  const double CoB_total = O2_mass_frac_bl * parameters_.rho_bl / parameters_.rho_oxy;

  // partial pressure of oxygen in blood
  double P_O2B = 0.0;

  // Calculate partial pressure of oxygen in blood
  PoroPressureBased::get_oxy_partial_pressure_from_concentration<double>(P_O2B, CoB_total,
      parameters_.NC_Hb, parameters_.P_oB50, parameters_.n, parameters_.alpha_oxy);

  // saturation of hemoglobin with oxygen from hill equation
  const double SO2 = pow(P_O2B, parameters_.n) /
                     (pow(P_O2B, parameters_.n) + pow(parameters_.P_oB50, parameters_.n));

  // temporary help variable for calculating partial pressure of carbon dioxide in blood
  const double temp =
      (1.0 - (0.02924 * parameters_.C_Hb) / ((2.244 - 0.422 * SO2) * (8.740 - parameters_.pH))) *
      0.0301 * 2.226 * (1 + pow(10, parameters_.pH - 6.1));

  // partial pressure of carbon dioxide in blood
  double P_CO2B = (CO2_mass_frac_bl * parameters_.rho_bl) / (parameters_.rho_CO2 * temp);

  // scaling of P_CO2B to get from mmHg to the used pressure unit in the input file
  P_CO2B *= parameters_.ScalingFormmHg;

  // evaluate function
  const double functval = parameters_.rho_CO2 * parameters_.DiffsolAdVTLC *
                          (volfrac_blood / parameters_.volfrac_blood_ref) * (P_CO2B - P_CO2A);

  return functval;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <int dim>
std::vector<double> PoroPressureBased::LungCarbonDioxideExchangeLaw<dim>::evaluate_derivative(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const size_t component) const
{
  // In debug mode, check order of variables and constants vector on every call
#ifdef FOUR_C_ENABLE_ASSERTIONS
  check_order(variables, constants);
#endif

  // Check order of variables and constants vector only once (since it does not change)
  if (not this->order_checked_) check_order(variables, constants);

  // create derivative vector (should have size of variables)
  std::vector<double> deriv(variables.size(), 0.0);

  // define Fad object for evaluation
  using FAD = Sacado::Fad::DFad<double>;
  FAD O2_mass_frac_bl = 0.0;
  O2_mass_frac_bl.diff(0, 1);

  double CO2_mass_frac_air = 0.0, P_air = 0.0, CO2_mass_frac_bl = 0.0, volfrac_blood = 0.0;

  if (variables[0].first == "phi1")  // maindiag-derivative
  {
    // read variables and constants (order is crucial)
    O2_mass_frac_bl.val() = variables[1].second;
    CO2_mass_frac_air = variables[2].second;
    CO2_mass_frac_bl = variables[3].second;
    P_air = constants[0].second;
    volfrac_blood = constants[3].second;
  }
  else if (variables[0].first == "p1")  // OD-derivative
  {
    // read variables and constants (order is crucial)
    O2_mass_frac_bl.val() = constants[1].second;
    CO2_mass_frac_air = constants[2].second;
    CO2_mass_frac_bl = constants[3].second;
    P_air = variables[0].second;
    volfrac_blood = variables[3].second;
  }
  else
    FOUR_C_THROW("Derivative w.r.t. <{}> not supported in LUNG_CARBONDIOXIDE_EXCHANGE_LAW.",
        variables[0].first);

  // volfrac relation
  const double volfrac_relation = (volfrac_blood / parameters_.volfrac_blood_ref);

  FAD P_O2B = 0.0;
  FAD C_oB_total = O2_mass_frac_bl * parameters_.rho_bl / parameters_.rho_oxy;

  PoroPressureBased::get_oxy_partial_pressure_from_concentration<FAD>(P_O2B, C_oB_total,
      parameters_.NC_Hb, parameters_.P_oB50, parameters_.n, parameters_.alpha_oxy);

  // saturation of hemoglobin with oxygen from hill equation
  const double SO2 = pow(P_O2B.val(), parameters_.n) /
                     (pow(P_O2B.val(), parameters_.n) + pow(parameters_.P_oB50, parameters_.n));

  // temporary help variable for calculating partial pressure of carbon dioxide in blood
  const double temp =
      (1.0 - (0.02924 * parameters_.C_Hb) / ((2.244 - 0.422 * SO2) * (8.740 - parameters_.pH))) *
      0.0301 * 2.226 * (1.0 + pow(10.0, parameters_.pH - 6.1));


  if (variables[0].first == "phi1")  // maindiag-derivative
  {
    // linearization w.r.t. phi2 (oxygen in blood) = dMassexchangeCO2/dwO2B =
    // dMassexchangeCO2/dPCO2
    // * dPCO2/dSO2 * dSO2/dPO2B * dPO2B/dwO2B
    double dMassexchangeCO2dPCO2 = parameters_.rho_CO2 * parameters_.DiffsolAdVTLC *
                                   volfrac_relation * parameters_.ScalingFormmHg;
    double dPCO2dSO2 = CO2_mass_frac_bl * (parameters_.rho_bl / parameters_.rho_CO2) *
                       pow(temp, -2.0) * 0.0301 * (1.0 + pow(10.0, parameters_.pH - 6.10)) * 2.226 *
                       (0.02924 * parameters_.C_Hb) / ((8.740 - parameters_.pH)) *
                       pow(2.244 - 0.422 * SO2, -2.0) * 0.422;
    double dSO2dPO2B =
        parameters_.n * pow(P_O2B.val(), parameters_.n - 1.0) *
            pow(pow(P_O2B.val(), parameters_.n) + pow(parameters_.P_oB50, parameters_.n), -1.0) -
        pow(pow(P_O2B.val(), parameters_.n) + pow(parameters_.P_oB50, parameters_.n), -2.0) *
            pow(P_O2B.val(), parameters_.n) * parameters_.n * pow(P_O2B.val(), parameters_.n - 1.0);
    double dPO2BdwO2B = P_O2B.fastAccessDx(0);
    deriv[1] = dMassexchangeCO2dPCO2 * dPCO2dSO2 * dSO2dPO2B * dPO2BdwO2B;

    deriv[2] = (-1.0) * parameters_.rho_CO2 * parameters_.DiffsolAdVTLC * volfrac_relation *
               ((P_air + parameters_.P_atmospheric) * parameters_.rho_air / parameters_.rho_CO2);

    deriv[3] = parameters_.rho_CO2 * parameters_.DiffsolAdVTLC * volfrac_relation *
               parameters_.ScalingFormmHg * parameters_.rho_bl / (parameters_.rho_CO2 * temp);
  }
  else if (variables[0].first == "p1")  // OD-derivative
  {
    // derivatives of dvolfrac/dPressure_fluid not yet included
    // partial pressure of carbon dioxide in air
    const double P_CO2A = CO2_mass_frac_air * (P_air + parameters_.P_atmospheric) *
                          parameters_.rho_air / parameters_.rho_CO2;
    // partial pressure of carbon dioxide in blood
    double P_CO2B = ((CO2_mass_frac_bl * parameters_.rho_bl / parameters_.rho_CO2) / temp) *
                    parameters_.ScalingFormmHg;

    // derivative w.r.t. P_air (dFunc/dP_CO2A * dP_CO2A/P_air) +  (dFunc/dvolfrac * dvolfrac/dP_air)
    deriv[0] = (-1.0) * (parameters_.rho_CO2 * parameters_.DiffsolAdVTLC * volfrac_relation) *
               ((CO2_mass_frac_air * parameters_.rho_air) / parameters_.rho_CO2);

    deriv[3] = parameters_.rho_CO2 * parameters_.DiffsolAdVTLC *
               (1 / parameters_.volfrac_blood_ref) * (P_CO2B - P_CO2A);
  }
  else
    FOUR_C_THROW("Derivative w.r.t. <{}> not supported in LUNG_CARBONDIOXIDE_EXCHANGE_LAW.",
        variables[0].first);

  return deriv;
}

// explicit instantiations

template class PoroPressureBased::TumorGrowthLawHeaviside<1>;
template class PoroPressureBased::TumorGrowthLawHeaviside<2>;
template class PoroPressureBased::TumorGrowthLawHeaviside<3>;

template class PoroPressureBased::NecrosisLawHeaviside<1>;
template class PoroPressureBased::NecrosisLawHeaviside<2>;
template class PoroPressureBased::NecrosisLawHeaviside<3>;

template class PoroPressureBased::OxygenConsumptionLawHeaviside<1>;
template class PoroPressureBased::OxygenConsumptionLawHeaviside<2>;
template class PoroPressureBased::OxygenConsumptionLawHeaviside<3>;

template class PoroPressureBased::TumorGrowthLawHeavisideOxy<1>;
template class PoroPressureBased::TumorGrowthLawHeavisideOxy<2>;
template class PoroPressureBased::TumorGrowthLawHeavisideOxy<3>;

template class PoroPressureBased::TumorGrowthLawHeavisideNecro<1>;
template class PoroPressureBased::TumorGrowthLawHeavisideNecro<2>;
template class PoroPressureBased::TumorGrowthLawHeavisideNecro<3>;

template class PoroPressureBased::OxygenTransvascularExchangeLawCont<1>;
template class PoroPressureBased::OxygenTransvascularExchangeLawCont<2>;
template class PoroPressureBased::OxygenTransvascularExchangeLawCont<3>;

template class PoroPressureBased::OxygenTransvascularExchangeLawDisc<1>;
template class PoroPressureBased::OxygenTransvascularExchangeLawDisc<2>;
template class PoroPressureBased::OxygenTransvascularExchangeLawDisc<3>;

template class PoroPressureBased::LungOxygenExchangeLaw<1>;
template class PoroPressureBased::LungOxygenExchangeLaw<2>;
template class PoroPressureBased::LungOxygenExchangeLaw<3>;

template class PoroPressureBased::LungCarbonDioxideExchangeLaw<1>;
template class PoroPressureBased::LungCarbonDioxideExchangeLaw<2>;
template class PoroPressureBased::LungCarbonDioxideExchangeLaw<3>;

FOUR_C_NAMESPACE_CLOSE
