// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROMULTIPHASE_SCATRA_FUNCTION_HPP
#define FOUR_C_POROMULTIPHASE_SCATRA_FUNCTION_HPP


#include "4C_config.hpp"

#include "4C_poromultiphase_scatra_function_parameters.hpp"
#include "4C_utils_function.hpp"

#include <Sacado.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::Utils
{
  class FunctionManager;
}

namespace PoroMultiPhaseScaTra
{
  /*!
   * @brief abstract class derived from FunctionOfAnything since reaction functions are cast to
   * FunctionOfAnything in scatra_reaction_coupling and fluidporo_multiphase_singlereaction
   */
  template <int dim>
  class PoroMultiPhaseScaTraFunction : public Core::Utils::FunctionOfAnything
  {
   public:
    PoroMultiPhaseScaTraFunction();


    double evaluate(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const override = 0;


    std::vector<double> evaluate_derivative(
        const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const override = 0;

    //! only one component possible for PoroMultiPhaseScatraFunction so far
    [[nodiscard]] std::size_t number_components() const override { return 1; }

    /*!
     * \brief check for correct order of input parameters variables and constants this check is
     * performed only once
     *
     * \param variables (i) A vector containing a pair (variablename, value) for each variable
     * \param constants (i) A vector containing a pair (variablename, value) for each constant
     */
    virtual void check_order(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants) const = 0;

   protected:
    /// indicating if the order has already been check for this function
    mutable bool order_checked_;
  };

  /// Attach poro-specific functions to the @p function_manager.
  void add_valid_poro_functions(Core::Utils::FunctionManager& function_manager);

  /*!
   * @brief standard growth law for tumor cells <--> IF (with lysis) and pressure dependency:
   * TUMOR_GROWTH_LAW_HEAVISIDE
   *
   * (gamma_T_growth*(phi1-w_nl_crit)/(w_nl_env-w_nl_crit)*heaviside((phi1-w_nl_crit)/(w_nl_env-w_nl_crit))*heaviside(p_t_crit-p2))*(1-phi2)*porosity*S2
   * - lambda*phi2*porosity*S2
   *
   * with phi1: mass fraction of oxygen, phi2: mass fraction of necrotic tumor cells
   * Furthermore, we assume that phase 1: healthy cells, phase2: tumor cells, phase3: IF
   *
   * (possible) INPUT DEFINITION: POROMULTIPHASESCATRA_FUNCTION TUMOR_GROWTH_LAW_HEAVISIDE NUMPARAMS
   * 5 PARAMS gamma_T_growth 9.6e-6 w_nl_crit 2.0e-6 w_nl_env 4.2e-6 lambda 0.0 p_t_crit 1.0e9
   */
  template <int dim>
  class TumorGrowthLawHeaviside : public PoroMultiPhaseScaTraFunction<dim>
  {
   public:
    /*!
     * \brief Constructor creating empty object. Add the function parameters (read from the
     * input file) to the function
     */
    TumorGrowthLawHeaviside(const TumorGrowthLawHeavisideParameters& parameters);

    double evaluate(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const override;

    std::vector<double> evaluate_derivative(
        const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const override;

    void check_order(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants) const override;

   private:
    const TumorGrowthLawHeavisideParameters parameters_;
  };

  /*!
   * @brief standard necrosis law for tumor growth model: NECROSIS_LAW_HEAVISIDE
   *
   * (1-phi2)*porosity*S2*(-gamma_t_necr*(phi1-w_nl_crit)/(w_nl_env-w_nl_crit)*heaviside(-(phi1-w_nl_crit)/(w_nl_env-w_nl_crit))+
   * delta_a_t*heaviside(p2-p_t_crit) )
   *
   * with phi1: mass fraction of oxygen, phi2: mass fraction of necrotic tumor cells,
   * S2: volume fraction of tumor cells
   *
   * Furthermore, we assume that phase 1: healthy cells, phase2: tumor cells, phase3: IF
   *
   * (possible) INPUT DEFINITION with exactly 5 function parameters:
   * POROMULTIPHASESCATRA_FUNCTION NECROSIS_LAW_HEAVISIDE NUMPARAMS 5 PARAMS gamma_t_necr 9.6e-6
   * w_nl_crit 2.0e-6 w_nl_env 4.2e-6 delta_a_t 0.0 p_t_crit 1.0e9
   */
  template <int dim>
  class NecrosisLawHeaviside : public PoroMultiPhaseScaTraFunction<dim>
  {
   public:
    /*!
     * \brief Constructor creating empty object. Add the function parameters (read from the
     * input file) to the function
     */
    NecrosisLawHeaviside(const NecrosisLawHeavisideParameters& parameters);

    double evaluate(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const override;

    std::vector<double> evaluate_derivative(
        const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const override;

    void check_order(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants) const override;

   private:
    const NecrosisLawHeavisideParameters parameters_;
  };

  /*!
   * @brief standard oxygen consumption law for tumor growth model: OXYGEN_CONSUMPTION_LAW_HEAVISIDE
   *
   * porosity*(1-phi2)*S2*(gamma_nl_growth*(phi1-w_nl_crit)/(w_nl_env-w_nl_crit)*heaviside((phi1-w_nl_crit)/(w_nl_env-w_nl_crit))*heaviside(p_t_crit-p2)+
   * gamma_0_nl*sin(pi/2.0*phi1/w_nl_env))
   *
   * with phi1: mass fraction of oxygen, phi2: mass fraction of necrotic tumor cells, S2: volume
   * fraction of tumor cells
   *
   * Furthermore, we assume that phase 1: healthy cells, phase2: tumor cells, phase3: IF
   *
   * (possible) INPUT DEFINITION with exactly 5 function parameters:
   * POROMULTIPHASESCATRA_FUNCTION OXYGEN_CONSUMPTION_LAW_HEAVISIDE NUMPARAMS 5 PARAMS
   * gamma_nl_growth 2.4e-7 gamma_0_nl 6e-7 w_nl_crit 2.0e-6 w_nl_env 4.2e-6 p_t_crit 1.0e9
   */
  template <int dim>
  class OxygenConsumptionLawHeaviside : public PoroMultiPhaseScaTraFunction<dim>
  {
   public:
    /*!
     * \brief Constructor creating empty object. Add the function parameters (read from the
     * input file) to the function
     */
    OxygenConsumptionLawHeaviside(const OxygenConsumptionLawHeavisideParameters& parameters);

    double evaluate(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const override;

    std::vector<double> evaluate_derivative(
        const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const override;

    void check_order(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants) const override;

   private:
    const OxygenConsumptionLawHeavisideParameters parameters_;
  };

  /*!
   * @brief standard growth law for tumor cells <--> IF (with lysis) and pressure dependency as
   * introduced: TUMOR_GROWTH_LAW_HEAVISIDE_OXY
   *
   * phi1*S2*porosity*((gamma_T_growth*(phi1-w_nl_crit)/(w_nl_env-w_nl_crit)*heaviside((phi1-w_nl_crit)/(w_nl_env-w_nl_crit))*heaviside(p_t_crit-p2))*(1-phi2)-
   * lambda*phi2)
   *
   * with phi1: mass fraction of oxygen, phi2: mass fraction of necrotic tumor cells, S2: volume
   * fraction of tumor cells
   *
   * Furthermore, we assume that phase 1: healthy cells, phase2: tumor cells, phase3: IF
   *
   * (possible) INPUT DEFINITION:
   * POROMULTIPHASESCATRA_FUNCTION TUMOR_GROWTH_LAW_HEAVISIDE_OXY NUMPARAMS 5 PARAMS
   * gamma_T_growth 9.6e-6 w_nl_crit 2.0e-6 w_nl_env 4.2e-6 lambda 0.0 p_t_crit 1.0e9
   */
  template <int dim>
  class TumorGrowthLawHeavisideOxy : public PoroMultiPhaseScaTraFunction<dim>
  {
   public:
    /*!
     * \brief Constructor creating empty object. Add the function parameters (read from the
     * input file) to the function
     */
    TumorGrowthLawHeavisideOxy(const TumorGrowthLawHeavisideNecroOxyParameters& parameters);


    double evaluate(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const override;

    std::vector<double> evaluate_derivative(
        const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const override;

    void check_order(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants) const override;

   private:
    const TumorGrowthLawHeavisideNecroOxyParameters parameters_;
  };

  /*!
   * @brief standard growth law for tumor cells <--> IF (with lysis) and pressure dependency as
   * introduced into balance of mass of necrotic cells: TUMOR_GROWTH_LAW_HEAVISIDE_NECRO
   *
   * porosity*S2*(((gamma_T_growth*(phi1-w_nl_crit)/(w_nl_env-w_nl_crit)*heaviside((phi1-w_nl_crit)/(w_nl_env-w_nl_crit))*heaviside(p_t_crit-p2))*(1-phi2)
   * - lambda*phi2)*phi2 + lambda*phi2)
   *
   * with phi1: mass fraction of oxygen, phi2: mass fraction of necrotic tumor cells
   *
   * Furthermore, we assume that phase 1: healthy cells, phase2: tumor cells, phase3: IF
   *
   * (possible) INPUT DEFINITION:
   * POROMULTIPHASESCATRA_FUNCTION TUMOR_GROWTH_LAW_HEAVISIDE_NECRO NUMPARAMS 5 PARAMS
   * gamma_T_growth 9.6e-6 w_nl_crit 2.0e-6 w_nl_env 4.2e-6 lambda 0.0 p_t_crit 1.0e9 |
   */
  template <int dim>
  class TumorGrowthLawHeavisideNecro : public PoroMultiPhaseScaTraFunction<dim>
  {
   public:
    /*!
     * \brief Constructor creating empty object. Add the function parameters (read from the
     * input file) to the function
     */
    TumorGrowthLawHeavisideNecro(const TumorGrowthLawHeavisideNecroOxyParameters& parameters);


    double evaluate(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const override;

    std::vector<double> evaluate_derivative(
        const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const override;

    void check_order(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants) const override;

   private:
    const TumorGrowthLawHeavisideNecroOxyParameters parameters_;
  };

  /*!
   * @brief transvascular exchange of oxygen from neovasculature into interstitial fluid:
   * OXYGEN_TRANSVASCULAR_EXCHANGE_LAW_CONT
   *
   * based on: Welter M, Fredrich T, Rinneberg H, Rieger H (2016) Computational Model for Tumor
   * Oxygenation Applied to Clinical Data on Breast Tumor Hemoglobin Concentrations Suggests
   * Vascular Dilatation and Compression. PLoS ONE 11(8)
   *
   * S/V*gamma*rho*(P_nv-P_if)*heaviside(P_nv-P_if)*VF1 with partial pressures of oxygen in
   * neovasculature P_nv and in IF P_if
   */
  template <int dim>
  class OxygenTransvascularExchangeLawCont : public PoroMultiPhaseScaTraFunction<dim>
  {
   public:
    /*!
     * \brief Constructor creating empty object. Add the function parameters (read from the
     * input file) to the function
     */
    OxygenTransvascularExchangeLawCont(
        const OxygenTransvascularExchangeLawContParameters& parameters);

    double evaluate(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const override;

    std::vector<double> evaluate_derivative(
        const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const override;

    void check_order(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants) const override;

   private:
    const OxygenTransvascularExchangeLawContParameters parameters_;
  };

  /*!
   * @brief transvascular exchange of oxygen from pre-existing (1D) vasculature into interstitial
   * fluid: OXYGEN_TRANSVASCULAR_EXCHANGE_LAW_DISC
   *
   * based on: Welter M, Fredrich T, Rinneberg H, Rieger H (2016) Computational Model for Tumor
   * Oxygenation Applied to Clinical Data on Breast Tumor Hemoglobin Concentrations Suggests
   * Vascular Dilatation and Compression. PLoS ONE 11(8)
   *
   * pi*D*gamma*rho*(P_v-P_if)*VF1*heaviside(S2_max-S2) with partial pressures of oxygen in
   * vasculature P_v and in IF P_if
   */
  template <int dim>
  class OxygenTransvascularExchangeLawDisc : public PoroMultiPhaseScaTraFunction<dim>
  {
   public:
    /*!
     * \brief Constructor creating empty object. Add the function parameters (read from the
     * input file) to the function
     */
    OxygenTransvascularExchangeLawDisc(
        const OxygenTransvascularExchangeLawDiscParameters& parameters);


    double evaluate(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const size_t component) const override;

    std::vector<double> evaluate_derivative(
        const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const size_t component) const override;

    void check_order(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants) const override;

   private:
    const OxygenTransvascularExchangeLawDiscParameters parameters_;
    mutable int pos_oxy_art_;
    mutable int pos_diam_;
  };

  /*!
   * @brief exchange of oxygen between air (phase in multiphase pore space) and blood (additional
   * porous network) in the lung: LUNG_OXYGEN_EXCHANGE_LAW
   *
   * rho_oxy*alpha_oxy*Diff_oxy*(1/d)*(A/V_TLC)*(P_oA - P_oB)
   *
   * with P_oA: partial pressures of oxygen in air, P_oB: partial pressures of oxygen in blood
   *
   * (possible) INPUT DEFINITION:
   * POROMULTIPHASESCATRA_FUNCTION LUNG_OXYGEN_EXCHANGE_LAW NUMPARAMS 9 PARAMS rho_oxy 1.429e-9
   * DiffAdVTLC 5.36 alpha_oxy 2.1e-4 rho_air 1.0e-9 rho_bl 1.03e-6 n 3 P_oB50 3.6 NC_Hb 0.25
   * P_atmospheric 101.3
   */
  template <int dim>
  class LungOxygenExchangeLaw : public PoroMultiPhaseScaTraFunction<dim>
  {
   public:
    /*!
     * \brief Constructor creating empty object. Add the function parameters (read from the
     * input file) to the function
     */
    LungOxygenExchangeLaw(const LungOxygenExchangeLawParameters& parameters);


    double evaluate(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const size_t component) const override;

    std::vector<double> evaluate_derivative(
        const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const size_t component) const override;

    void check_order(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants) const override;

   private:
    const LungOxygenExchangeLawParameters parameters_;
  };

  /*!
   * @brief exchange of carbon dioxide between air (phase in multiphase pore space) and blood
   * (additional porous network) in the lung: LUNG_CARBONDIOXIDE_EXCHANGE_LAW
   *
   * * rho_CO2*alpha_CO2*Diff_CO2*(1/d)*(A/V_TLC)*(P_CO2B - P_CO2B)
   *
   * with
   * P_CO2A: partial pressures of carbon dioxide in air
   * P_CO2B: partial pressures of carbon dioxide in blood (calculation based on "QUANTITATIVE
   * DESCRIPTION OF WHOLE BLOOD CO2 DISSOCIATION CURVE AND HALDANE EFFECT"
   * (DOI: 10.1016/0034-5687(83)90038-5))
   *
   * (possible) INPUT DEFINITION:
   * POROMULTIPHASESCATRA_FUNCTION LUNG_CARBONDIOXIDE_EXCHANGE_LAW NUMPARAMS 14 PARAMS
   * rho_CO2 1.98e-9 DiffsolAdVTLC 4.5192e-3 pH 7.352 rho_air 1.0e-9 rho_bl 1.03e-6 rho_oxy 1.429e-9
   * n 3 P_oB50 3.6 C_Hb 18.2 NC_Hb 0.25 alpha_oxy 2.1e-4 P_atmospheric 101.3 ScalingFormmHg
   * 133.3e-3 volfrac_blood_ref 0.1
   */
  template <int dim>
  class LungCarbonDioxideExchangeLaw : public PoroMultiPhaseScaTraFunction<dim>
  {
   public:
    /*!
     * \brief Constructor creating empty object. Add the function parameters (read from the
     * input file) to the function
     */
    LungCarbonDioxideExchangeLaw(const LungCarbonDioxideExchangeLawParameters& parameters);

    /*!
     * \brief evaluate function for a given set of variables
     *
     * \param[in] index For vector-valued functions, index defines the function-component which
     * should be evaluated
     * \param[in] variables A vector containing a pair (variablename, value) for
     * each variable
     * \param[in] constants A vector containing a pair (variablename, value) for each
     * constant
     */
    double evaluate(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const size_t component) const override;

    /*!
     * \brief evaluate derivative at given position in space for a given set of variables and
     * constants
     *
     * \param[in] index For vector-valued functions, index defines the function-component which
     * should be evaluated
     * \param[in] variables A vector containing a pair (variablename, value) for
     * each variable
     * \param[in] constants A vector containing a pair (variablename, value) for each
     * constant
     */
    std::vector<double> evaluate_derivative(
        const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const size_t component) const override;

    /*!
     * \brief check for correct order of input parameters variables and constants this check is
     * performed only once
     *
     * \param[in] variables A vector containing a pair (variablename, value) for each variable
     * \param[in] constants A vector containing a pair (variablename, value) for each constant
     */
    void check_order(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants) const override;


   private:
    const LungCarbonDioxideExchangeLawParameters parameters_;
  };

}  // namespace PoroMultiPhaseScaTra

FOUR_C_NAMESPACE_CLOSE

#endif
