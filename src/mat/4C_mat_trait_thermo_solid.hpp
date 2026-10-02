// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_mat_monolithic_solid_scalar_material.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_utils_exceptions.hpp"

#include <optional>

#ifndef FOUR_C_MAT_TRAIT_THERMO_SOLID_HPP
#define FOUR_C_MAT_TRAIT_THERMO_SOLID_HPP

FOUR_C_NAMESPACE_OPEN

namespace Mat
{

  /*!
   * @brief Mechanical heat source of a material and the linearizations needed by
   * thermo-mechanical coupling. Default constructed with zero values.
   *
   * The heat source per unit reference volume and time is
   * \f[
   *   R_\text{mech}(\mathbf{E}, \dot{\mathbf{E}}, T)
   *     = T \frac{\partial \mathbf{S}}{\partial T} : \dot{\mathbf{E}}
   *       + R_\text{add}(\mathbf{E}, T),
   * \f]
   * i.e., the thermoelastic heating plus the additional heat source of the material (see
   * \ref AdditionalHeatSource). \f$\mathbf{E}\f$ is the Green-Lagrange strain for finite strains
   * and the linear strain for small strains. Positive values heat the material.
   */
  struct ThermoMechanicalHeatSource
  {
    //! \f$R_\text{mech}\f$
    double value = 0.0;
    //! \f$\mathrm{d} R_\text{mech} / \mathrm{d} T\f$
    double derivative_wrt_temperature = 0.0;
    //! \f$\mathrm{d} R_\text{mech} / \mathrm{d} \mathbf{E}\f$
    Core::LinAlg::SymmetricTensor<double, 3, 3> derivative_wrt_strain{};
    //! \f$\mathrm{d} R_\text{mech} / \mathrm{d} \dot{\mathbf{E}}\f$
    Core::LinAlg::SymmetricTensor<double, 3, 3> derivative_wrt_strain_rate{};
  };

  /*!
   * @brief Heat source \f$R_\text{add}(\mathbf{E}, T)\f$ of the internal variables of a material,
   * e.g., the dissipation, and its linearizations. See \ref ThermoMechanicalHeatSource for the
   * notation. Default constructed with zero values.
   */
  struct AdditionalHeatSource
  {
    //! \f$R_\text{add}\f$
    double value = 0.0;
    //! \f$\mathrm{d} R_\text{add} / \mathrm{d} T\f$
    double derivative_wrt_temperature = 0.0;
    //! \f$\mathrm{d} R_\text{add} / \mathrm{d} \mathbf{E}\f$
    Core::LinAlg::SymmetricTensor<double, 3, 3> derivative_wrt_strain{};
  };

  /*!
   * @brief Stress-temperature modulus, i.e. the partial derivative
   * \f$\partial \mathbf{S} / \partial T\f$ (at fixed internal variables), and its linearizations.
   * See \ref ThermoMechanicalHeatSource for the notation. Default constructed with zero values.
   */
  struct StressTemperatureModulus
  {
    //! \f$\partial \mathbf{S} / \partial T\f$
    Core::LinAlg::SymmetricTensor<double, 3, 3> value{};
    //! \f$\mathrm{d} (\partial \mathbf{S} / \partial T) / \mathrm{d} T\f$
    Core::LinAlg::SymmetricTensor<double, 3, 3> derivative_wrt_temperature{};
    //! \f$\mathrm{d} (\partial \mathbf{S} / \partial T) / \mathrm{d} \mathbf{E}\f$
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> derivative_wrt_strain{};
  };

  /*!
   * @brief Deformation state, storing the (Green-Lagrange) strain and optionally the deformation
   * gradient for finite strains.
   */
  class DeformationState
  {
   public:
    /*!
     * @brief Construct a deformation state for small strains.
     * @param strain The linear strain.
     */
    explicit DeformationState(const Core::LinAlg::SymmetricTensor<double, 3, 3>& strain)
        : strain_(strain)
    {
    }
    /*!
     * @brief Construct a deformation state for finite strains.
     * @param strain The Green-Lagrange strain.
     * @param defgrad The deformation gradient.
     */
    explicit DeformationState(const Core::LinAlg::SymmetricTensor<double, 3, 3>& strain,
        const Core::LinAlg::Tensor<double, 3, 3>& defgrad)
        : strain_(strain), defgrad_(defgrad)
    {
    }

    //! Green-Lagrange strain for finite strains or linear strain for small strains
    [[nodiscard]] const Core::LinAlg::SymmetricTensor<double, 3, 3>& strain() const
    {
      return strain_;
    }

    //! Deformation gradient, only available for finite strains
    [[nodiscard]] const Core::LinAlg::Tensor<double, 3, 3>& defgrad() const
    {
      if (!defgrad_.has_value())
        FOUR_C_THROW("The deformation gradient is only available for finite strains.");
      return *defgrad_;
    }

   private:
    Core::LinAlg::SymmetricTensor<double, 3, 3> strain_;
    std::optional<Core::LinAlg::Tensor<double, 3, 3>> defgrad_;
  };

  namespace Trait
  {
    class ThermoSolid : public So3Material, public MonolithicSolidScalarMaterial
    {
     public:
      /*!
       * Evaluate the mechanical heat source and its derivatives, see
       * \ref ThermoMechanicalHeatSource. Only the thermoelastic heating depends on the strain rate.
       *
       * @param temperature current temperature
       * @param deformation_state current deformation
       * @param strain_rate current strain rate \f$\dot{\mathbf{E}}\f$
       * @param context evaluation context
       * @param gp Gauss-point index
       * @param eleGID global element id
       */
      [[nodiscard]] ThermoMechanicalHeatSource evaluate_mechanical_heat_source(
          const double temperature, const DeformationState& deformation_state,
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& strain_rate,
          const EvaluationContext<3>& context, const int gp, const int eleGID)
      {
        const StressTemperatureModulus stm = evaluate_stress_temperature_modulus(
            temperature, deformation_state, context, gp, eleGID);
        const AdditionalHeatSource additional_source =
            evaluate_additional_heat_source(temperature, deformation_state, context, gp, eleGID);

        return {
            .value =
                temperature * Core::LinAlg::ddot(stm.value, strain_rate) + additional_source.value,
            .derivative_wrt_temperature =
                Core::LinAlg::ddot(
                    stm.value + temperature * stm.derivative_wrt_temperature, strain_rate) +
                additional_source.derivative_wrt_temperature,
            .derivative_wrt_strain =
                temperature * Core::LinAlg::ddot(strain_rate, stm.derivative_wrt_strain) +
                additional_source.derivative_wrt_strain,
            .derivative_wrt_strain_rate = temperature * stm.value,
        };
      }

     protected:
      /*!
       * Evaluate the stress-temperature modulus and its derivatives w.r.t. temperature and strain.
       *
       * @param temperature current temperature
       * @param deformation_state current deformation state
       * @param context evaluation context
       * @param gp Gauss-point index
       * @param eleGID global element id
       */
      [[nodiscard]] virtual StressTemperatureModulus evaluate_stress_temperature_modulus(
          double temperature, const DeformationState& deformation_state,
          const EvaluationContext<3>& context, int gp, int eleGID) = 0;

      /*!
       * Evaluate heat sources in addition to the thermoelastic heating, e.g., plastic dissipation.
       * The default implementation returns zero values.
       *
       * @param temperature current temperature
       * @param deformation_state current deformation state
       * @param context evaluation context
       * @param gp Gauss-point index
       * @param eleGID global element id
       */
      [[nodiscard]] virtual AdditionalHeatSource evaluate_additional_heat_source(
          const double temperature, const DeformationState& deformation_state,
          const EvaluationContext<3>& context, const int gp, const int eleGID)
      {
        return {};
      }
    };
  }  // namespace Trait
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif