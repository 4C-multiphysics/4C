// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MULTIPLICATIVE_SPLIT_DEFGRAD_ELASTHYPER_SERVICE_HPP
#define FOUR_C_MAT_MULTIPLICATIVE_SPLIT_DEFGRAD_ELASTHYPER_SERVICE_HPP
#include "4C_config.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_utils_densematrix_determinant.hpp"
#include "4C_mat_elasthyper_service.hpp"
#include "4C_mat_service.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  /*!
   * @brief Mechanical heat source contribution produced by the material and the linearizations
   * needed by thermomechanical coupling. Default constructed with zero values.
   */
  struct HeatSource
  {
    double value = 0.0;
    //! derivative of the mechanical dissipation heat source w.r.t. temperature
    double derivative_wrt_temperature = 0.0;
    //! derivative of the mechanical dissipation heat source w.r.t. the right Cauchy-Green tensor
    //! (stress-form)
    Core::LinAlg::Matrix<1, 6> derivative_wrt_cauchy_green{Core::LinAlg::Initialization::zero};

    //! pack a vector of MechanicalDissipation
    static void pack(
        Core::Communication::PackBuffer& data, const std::vector<HeatSource>& heat_source)
    {
      Core::Communication::add_to_pack(data, static_cast<int>(heat_source.size()));

      for (const auto& md : heat_source)
      {
        Core::Communication::add_to_pack(data, md.value);
        Core::Communication::add_to_pack(data, md.derivative_wrt_temperature);
        Core::Communication::add_to_pack(data, md.derivative_wrt_cauchy_green);
      }
    }
  };


  /// Free-energy related stress factors, as presented in Holzapfel - Nonlinear Solid Mechanics.
  struct StressFactors
  {
    /// 2nd Piola-Kirchhoff stress factors, cf. Holzapfel p. 216.
    Core::LinAlg::Matrix<3, 1> gamma{Core::LinAlg::Initialization::zero};
    /// Constitutive tensor factors, cf. Holzapfel p. 261.
    Core::LinAlg::Matrix<8, 1> delta{Core::LinAlg::Initialization::zero};
  };

  /// helper to store a second order tensor and its derivative w.r.t. temperature
  struct TensorAndTemperatureDerivative
  {
    /// value of the second order tensor
    Core::LinAlg::SymmetricTensor<double, 3, 3> value{};
    /// derivative of the second order tensor w.r.t. temperature
    Core::LinAlg::SymmetricTensor<double, 3, 3> temperature_derivative{};
  };

  /// Thermal stretch quantities used by multiplicative-split thermoelastic stress evaluation.
  namespace ThermalExpansion
  {

    /*!
     * @brief compute contribution to the thermo-elastic stress due to thermal expansion,
     * \f$\boldsymbol{S}_T = S_\mathrm{he}(\boldsymbol{C}_T)\f$, and its partial derivative w.r.t.
     * temperature:
     * \f$\frac{\partial \mathbf{S}_T}{\partial T} =
     * \frac{1}{2}\left.\mathbb{C}_\text{he}\right|_{\boldsymbol{C}_T} : \frac{\partial
     * \boldsymbol{C}_T}{\partial T}\f$
     *
     * @param[in] delta_temperature current absolute temperature minus reference temperature
     * @param[in] expansion_coefficient isotropic thermal expansion coefficient
     * @param[in] gp Gauss point
     * @param[in] eleGID element global ID
     * @param[in] potsumel isotropic elastic summands used to evaluate invariant derivatives
     */
    inline TensorAndTemperatureDerivative compute_thermoelastic_stress_contribution(
        const double delta_temperature, const double expansion_coefficient,
        const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& potsumel, const int gp,
        const int eleGID)
    {
      auto thermoelastic_stress_contribution = TensorAndTemperatureDerivative{};
      if (expansion_coefficient == 0.0)
      {
        // no thermal expansion, default initialized zero values are correct
        return thermoelastic_stress_contribution;
      }

      /// thermal right Cauchy-Green deformation tensor due to thermal expansion,
      /// \f$\mathbf{C}_T = (1 + 2 \alpha_T \Delta T)\mathbf{I}\f$, and its derivative w.r.t.
      /// temperature, \f$\frac{\partial \mathbf{C}_T}{\partial T} = 2 \alpha_T \mathbf{I}\f$
      const TensorAndTemperatureDerivative thermal_cauchy_green{
          .value = (1 + 2 * expansion_coefficient * delta_temperature) *
                   Core::LinAlg::TensorGenerators::identity<double, 3, 3>,
          .temperature_derivative =
              2 * expansion_coefficient * Core::LinAlg::TensorGenerators::identity<double, 3, 3>};

      // compute principal invariants of the thermal stretch
      Core::LinAlg::Matrix<3, 1> prinv_of_thermal_cauchy_green{Core::LinAlg::Initialization::zero};
      Core::LinAlg::Voigt::Stresses::invariants_principal(
          prinv_of_thermal_cauchy_green, make_stress_like_voigt_view(thermal_cauchy_green.value));

      // compute derivatives of the thermal stretch principal invariants
      Core::LinAlg::Matrix<3, 1> dPI{Core::LinAlg::Initialization::zero};
      Core::LinAlg::Matrix<6, 1> ddPII{Core::LinAlg::Initialization::zero};
      for (const auto& p : potsumel)  // only for isotropic components
      {
        p->add_derivatives_principal(dPI, ddPII, prinv_of_thermal_cauchy_green, gp, eleGID);
      }

      /// \f$\left.\mathbb{C}_\text{he}\right|_{\boldsymbol{C}_T}\f$
      Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> hyperelast_stiffness{};
      elast_hyper_add_isotropic_stress_cmat(thermoelastic_stress_contribution.value,
          hyperelast_stiffness, thermal_cauchy_green.value, inv(thermal_cauchy_green.value),
          prinv_of_thermal_cauchy_green, dPI, ddPII);

      thermoelastic_stress_contribution.temperature_derivative =
          0.5 * ddot(hyperelast_stiffness, thermal_cauchy_green.temperature_derivative);

      return thermoelastic_stress_contribution;
    }

    /// contribution to the 2nd Piola-Kirchhoff stress due to thermal expansion,
    /// \f$\det(\boldsymbol{F}_\text{in}) \boldsymbol{F}_\text{in}^{-1} \boldsymbol{S}_T
    /// \boldsymbol{F}_\text{in}^{-T}\f$, and its partial derivative w.r.t. temperature,
    /// \f$\det(\boldsymbol{F}_\text{in}) \boldsymbol{F}_\text{in}^{-1} \cdot \frac{\partial
    /// \mathbf{S}_T}{\partial T} \cdot \boldsymbol{F}_\text{in}^{-T}\f$
    [[nodiscard]] inline TensorAndTemperatureDerivative compute_pk2_stress_contribution(
        const TensorAndTemperatureDerivative& thermoelastic_stress_contribution,
        const Core::LinAlg::Matrix<3, 3>& iFinM)
    {
      const auto iFin = make_tensor_view(iFinM);
      const double detFin = 1.0 / det(iFin);

      return {
          .value = detFin * assume_symmetry(dot(dot(iFin, thermoelastic_stress_contribution.value),
                                transpose(iFin))),
          .temperature_derivative =
              detFin *
              assume_symmetry(
                  dot(dot(iFin, get_full(thermoelastic_stress_contribution.temperature_derivative)),
                      transpose(iFin)))};
    }
  };  // namespace ThermalExpansion

  inline void evaluate_ce(const Core::LinAlg::Matrix<3, 3>& F,
      const Core::LinAlg::Matrix<3, 3>& iFin, Core::LinAlg::Matrix<3, 3>& Ce)
  {
    static Core::LinAlg::Matrix<3, 3> FiFin(Core::LinAlg::Initialization::uninitialized);
    FiFin.multiply_nn(F, iFin);
    Ce.multiply_tn(FiFin, FiFin);
  }

  inline void evaluatei_cin_ci_cin(const Core::LinAlg::Matrix<3, 3>& C,
      const Core::LinAlg::Matrix<3, 3>& iCin, Core::LinAlg::Matrix<3, 3>& iCinCiCin)
  {
    static Core::LinAlg::Matrix<3, 3> CiCin(Core::LinAlg::Initialization::uninitialized);
    CiCin.multiply_nn(C, iCin);
    iCinCiCin.multiply_nn(iCin, CiCin);
  }

  inline void elast_hyper_evaluate_elastic_part(const Core::LinAlg::Matrix<3, 3>& F,
      const Core::LinAlg::Matrix<3, 3>& iFin, Core::LinAlg::Matrix<6, 1>& S_stress,
      Core::LinAlg::Matrix<6, 6>& cmat,
      const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& potsum,
      Mat::SummandProperties summandProperties, const int gp, const int eleGID)
  {
    if (summandProperties.anisomod or summandProperties.anisoprinc)
    {
      FOUR_C_THROW(
          "An additional inelastic part is not yet implemented for anisotropic materials.");
    }

    S_stress.clear();
    cmat.clear();

    // Variables needed for the computation of the stress resultants
    static Core::LinAlg::Matrix<3, 3> C(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 3> Ce(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 3> iC(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 3> iCin(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 3> iCinCiCin(Core::LinAlg::Initialization::zero);

    static Core::LinAlg::Matrix<6, 1> iCinv(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<6, 1> iCinCiCinv(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<6, 1> iCv(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 1> principleInvariantsCe(Core::LinAlg::Initialization::zero);

    // Compute right Cauchy-Green tensor C=F^TF
    C.multiply_tn(F, F);

    // Compute inverse right Cauchy-Green tensor C^-1
    iC.invert(C);

    // Compute inverse inelastic right Cauchy-Green Tensor
    iCin.multiply_nt(iFin, iFin);

    // Compute iCin * C * iCin
    Mat::evaluatei_cin_ci_cin(C, iCin, iCinCiCin);

    // Compute Ce
    Mat::evaluate_ce(F, iFin, Ce);

    // Compute principal invariants
    Mat::invariants_principal(principleInvariantsCe, Ce);

    Core::LinAlg::Matrix<3, 1> dPIe(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> ddPIIe(Core::LinAlg::Initialization::zero);

    Mat::elast_hyper_evaluate_invariant_derivatives(
        principleInvariantsCe, dPIe, ddPIIe, potsum, summandProperties, gp, eleGID);

    // 2nd Piola Kirchhoff stress factors (according to Holzapfel-Nonlinear Solid Mechanics p. 216)
    static Core::LinAlg::Matrix<3, 1> gamma(Core::LinAlg::Initialization::zero);
    // constitutive tensor factors (according to Holzapfel-Nonlinear Solid Mechanics p. 261)
    static Core::LinAlg::Matrix<8, 1> delta(Core::LinAlg::Initialization::zero);

    Mat::calculate_gamma_delta(gamma, delta, principleInvariantsCe, dPIe, ddPIIe);

    // Convert necessary tensors to stress-like Voigt-Notation
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCin, iCinv);
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCinCiCin, iCinCiCinv);
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(iC, iCv);

    // Contribution to 2nd Piola-Kirchhoff stress tensor
    S_stress.update(gamma(0), iCinv, 1.0);
    S_stress.update(gamma(1), iCinCiCinv, 1.0);
    S_stress.update(gamma(2), iCv, 1.0);

    // Contribution to the linearization
    cmat.multiply_nt(delta(0), iCinv, iCinv, 1.);
    cmat.multiply_nt(delta(1), iCinCiCinv, iCinv, 1.);
    cmat.multiply_nt(delta(1), iCinv, iCinCiCinv, 1.);
    cmat.multiply_nt(delta(2), iCinv, iCv, 1.);
    cmat.multiply_nt(delta(2), iCv, iCinv, 1.);
    cmat.multiply_nt(delta(3), iCinCiCinv, iCinCiCinv, 1.);
    cmat.multiply_nt(delta(4), iCinCiCinv, iCv, 1.);
    cmat.multiply_nt(delta(4), iCv, iCinCiCinv, 1.);
    cmat.multiply_nt(delta(5), iCv, iCv, 1.);
    Core::LinAlg::FourTensorOperations::add_holzapfel_product(cmat, iCv, delta(6));
    Core::LinAlg::FourTensorOperations::add_holzapfel_product(cmat, iCinv, delta(7));
  }

}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif
