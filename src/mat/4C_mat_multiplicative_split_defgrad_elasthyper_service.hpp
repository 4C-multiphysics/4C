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
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
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

  /// Thermal stretch quantities used by multiplicative-split thermoelastic stress evaluation.
  struct ThermalQuantities
  {
    // ----- variables of thermal quantities ----- //
    /// thermal right Cauchy-Green deformation tensor \f$ \mathbf{C}_T \f$ stored as 6x1
    /// vector (stress-form!)
    Core::LinAlg::Matrix<6, 1> CTV{Core::LinAlg::Initialization::zero};
    /// inverse thermal right Cauchy-Green deformation tensor \f$ \mathbf{C}_T^{-1} \f$ stored as
    /// 6x1 vector (stress-form)
    Core::LinAlg::Matrix<6, 1> iCTV{Core::LinAlg::Initialization::zero};
    /// \f$ \mathbf{F}_{\text{in}}^{-1} \mathbf{C}_T \mathbf{F}_{\text{in}}^{-T} \f$ stored as 6x1
    /// vector (stress-form!)
    Core::LinAlg::Matrix<6, 1> iFinCTiFinTV{Core::LinAlg::Initialization::zero};
    /// \f$ \mathbf{F}_{\text{in}}^{-1} \mathbf{C}_T^{-1} \mathbf{F}_{\text{in}}^{-T} \f$ stored
    /// as 6x1 vector (stress-form!)
    Core::LinAlg::Matrix<6, 1> iFiniCTiFinTV{Core::LinAlg::Initialization::zero};
    /// derivative of thermal right Cauchy-Green deformation tensor wrt temperature \f$ \mathrm{d}
    /// \mathbf{C}_T / \mathrm{d} T \f$ stored as 6x1 vector (strain-form!)
    Core::LinAlg::Matrix<6, 1> dCTdTV{Core::LinAlg::Initialization::zero};

    /// principal invariants of the thermal right Cauchy-Green tensor
    Core::LinAlg::Matrix<3, 1> prinv{Core::LinAlg::Initialization::zero};

    // ----- derivatives of principal invariants ----- //

    /// first derivatives of principal invariants
    Core::LinAlg::Matrix<3, 1> dPI{Core::LinAlg::Initialization::zero};
    /// second derivatives of principal invariants
    Core::LinAlg::Matrix<6, 1> ddPII{Core::LinAlg::Initialization::zero};
  };

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

  /*!
   * @brief Subtracts the thermal contribution from the 2nd Piola-Kirchhoff stress
   *
   * \f[\texttt{stress} \mathrel{{-}{=}} \det(\boldsymbol{F}_\text{in})
   * \boldsymbol{F}_\text{in}^{-1}
   * \boldsymbol{S}_\text{he}[\boldsymbol{C}_\text{he} = \boldsymbol{C}_T]
   * \boldsymbol{F}_\text{in}^{-T}\f]
   * where \f$\boldsymbol{S}_\text{he}[\boldsymbol{C}_\text{he} = \boldsymbol{C}_T]\f$ denotes
   * the hyperelastic 2nd Piola-Kirchhoff stress evaluated
   * at the thermal right Cauchy-Green tensor \f$\boldsymbol{C}_T\f$.
   *
   * @param[in] thermal_quant Thermal stretch quantities and invariants
   * @param[in] thermal_stress_fact Holzapfel stress factors for the thermal stretch
   * @param[in] iCinV inverse inelastic right Cauchy-Green tensor in stress-like Voigt notation
   * @param[in] detFin determinant of the inelastic deformation gradient
   * @param[in,out] stress 2nd Piola-Kirchhoff stress in stress-like Voigt notation to be updated by
   * the thermal contribution
   */
  inline void add_thermal_stress_contribution(Core::LinAlg::Matrix<6, 1>& stress,
      const ThermalQuantities& thermal_quant, const StressFactors& thermal_stress_fact,
      const Core::LinAlg::Matrix<6, 1>& iCinV, const double detFin)
  {
    const Core::LinAlg::Matrix<3, 1>& thermal_gamma = thermal_stress_fact.gamma;

    stress.update(-detFin * thermal_gamma(0), iCinV, 1.0);
    stress.update(-detFin * thermal_gamma(1), thermal_quant.iFinCTiFinTV, 1.0);
    stress.update(-detFin * thermal_gamma(2), thermal_quant.iFiniCTiFinTV, 1.0);
  }



  /*!
   * @brief Evaluate the partial derivative of the 2nd Piola-Kirchhoff stress wrt. temperature:
   *
   * \f[\frac{\partial \boldsymbol{S}}{\partial T}
   * = -\det(\boldsymbol{F}_\text{in}) \boldsymbol{F}_{\text{in}}^{-1}
   * \left(\frac{1}{2}\left.\mathbb{C}_\text{he}\right|_{\boldsymbol{C}_T}
   * : \frac{\partial \boldsymbol{C}_T}{\partial T} \right)
   * \boldsymbol{F}_{\text{in}}^{-T}\f]
   *
   * where \f$\left.\mathbb{C}_\text{he}\right|_{\boldsymbol{C}_T}\f$ denotes the hyperelastic
   * stiffness evaluated at the thermal right Cauchy-Green tensor \f$\boldsymbol{C}_T\f$.
   *
   * @param[in] iFinM inverse inelastic deformation gradient
   * @param[in] thermal_quant Thermal stretch quantities and temperature derivative
   * @return derivative of the thermal 2nd Piola-Kirchhoff stress w.r.t. temperature
   */
  inline Core::LinAlg::Matrix<6, 1> compute_partial_d_stress_d_temperature(
      const Core::LinAlg::Matrix<3, 3>& iFinM, const ThermalQuantities& thermal_quant)
  {
    Core::LinAlg::Matrix<6, 1> thermal_stress_deriv{Core::LinAlg::Initialization::zero};

    const Core::LinAlg::Matrix<6, 1>& dCTdTV = thermal_quant.dCTdTV;
    const double detFin = 1.0 / iFinM.determinant();

    // evaluate purely hyperelastic stiffness with the thermal right CG tensor as input
    Core::LinAlg::SymmetricTensor<double, 3, 3> hyperelast_stress{};
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> hyperelast_stiffness{};
    elast_hyper_add_isotropic_stress_cmat(hyperelast_stress, hyperelast_stiffness,
        Core::LinAlg::make_symmetric_tensor_from_stress_like_voigt_matrix(thermal_quant.CTV),
        Core::LinAlg::make_symmetric_tensor_from_stress_like_voigt_matrix(thermal_quant.iCTV),
        thermal_quant.prinv, thermal_quant.dPI, thermal_quant.ddPII);


    /// compute derivative \f$ \frac{\partial \mathbf{S}_{\theta}}{\partial T} \f$

    Core::LinAlg::Matrix<6, 1> pStheta_pT_stress{Core::LinAlg::Initialization::zero};
    pStheta_pT_stress.multiply_nn(
        0.5, Core::LinAlg::make_stress_like_voigt_view(hyperelast_stiffness), dCTdTV, 0.0);
    Core::LinAlg::Matrix<3, 3> pStheta_pT{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Voigt::Stresses::vector_to_matrix(pStheta_pT_stress, pStheta_pT);

    /// compute product \f$ \mathbf{F}_{\text{in}}^{-1}  \frac{\partial
    /// \mathbf{S}_{\theta}}{\partial T} \mathbf{F}_{\text{in}}^{-T} \f$
    Core::LinAlg::Matrix<3, 3> iFin_pStheta_pT{Core::LinAlg::Initialization::zero};
    iFin_pStheta_pT.multiply_nn(1.0, iFinM, pStheta_pT, 0.0);
    Core::LinAlg::Matrix<3, 3> iFin_pStheta_pT_iFinT{Core::LinAlg::Initialization::zero};
    iFin_pStheta_pT_iFinT.multiply_nt(1.0, iFin_pStheta_pT, iFinM, 0.0);
    Core::LinAlg::Matrix<6, 1> iFin_pStheta_pT_iFinT_V{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(iFin_pStheta_pT_iFinT, iFin_pStheta_pT_iFinT_V);

    // thermal derivative
    thermal_stress_deriv.update(-detFin, iFin_pStheta_pT_iFinT_V, 0.0);

    return thermal_stress_deriv;
  }

  /*!
   * @brief Compute thermal stretch quantities for isotropic thermal expansion.
   *
   * @param[in] delta_temperature current absolute temperature minus reference temperature
   * @param[in] thermal_expansion_coefficient isotropic thermal expansion coefficient
   * @param[in] iFinM inverse inelastic deformation gradient
   * @param[in] gp Gauss point
   * @param[in] eleGID element global ID
   * @param[in] potsumel isotropic elastic summands used to evaluate invariant derivatives
   * @return collected thermal kinematic quantities and invariant derivatives
   */
  inline ThermalQuantities evaluate_thermal_quantities(const double delta_temperature,
      const double thermal_expansion_coefficient, const Core::LinAlg::Matrix<3, 3>& iFinM,
      const int gp, const int eleGID,
      const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& potsumel)
  {
    ThermalQuantities quantities{};

    // compute the thermal stretch, along with its temperature
    // derivative
    Core::LinAlg::SymmetricTensor<double, 3, 3> thermal_right_cg_tensor{
        Core::LinAlg::TensorGenerators::identity<double, 3, 3>};
    Core::LinAlg::SymmetricTensor<double, 3, 3> thermal_right_cg_temp_deriv_tensor{};
    thermal_right_cg_tensor += 2 * thermal_expansion_coefficient * delta_temperature *
                               Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
    thermal_right_cg_temp_deriv_tensor +=
        2 * thermal_expansion_coefficient * Core::LinAlg::TensorGenerators::identity<double, 3, 3>;

    // compute inverse of the thermal stretch
    Core::LinAlg::SymmetricTensor<double, 3, 3> inv_thermal_right_cg_tensor =
        inv(thermal_right_cg_tensor);

    // get matrices for the thermal stretch
    const Core::LinAlg::Matrix<3, 3> CTM =
        Core::LinAlg::make_matrix(get_full(thermal_right_cg_tensor));
    const Core::LinAlg::Matrix<3, 3> iCTM =
        Core::LinAlg::make_matrix(Core::LinAlg::get_full(inv_thermal_right_cg_tensor));

    // compute terms with iFin
    Core::LinAlg::Matrix<3, 3> iFinCT{};
    iFinCT.multiply(1.0, iFinM, CTM, 0.0);
    Core::LinAlg::Matrix<3, 3> iFinCTiFinT{};
    iFinCTiFinT.multiply_nt(1.0, iFinCT, iFinM, 0.0);
    Core::LinAlg::Matrix<3, 3> iFiniCT{};
    iFiniCT.multiply(1.0, iFinM, iCTM, 0.0);
    Core::LinAlg::Matrix<3, 3> iFiniCTiFinT{};
    iFiniCTiFinT.multiply_nt(1.0, iFiniCT, iFinM, 0.0);

    // add computed tensors to quantities in the specified form
    quantities.CTV = Core::LinAlg::make_stress_like_voigt_view(thermal_right_cg_tensor);
    quantities.iCTV = Core::LinAlg::make_stress_like_voigt_view(inv_thermal_right_cg_tensor);
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(iFinCTiFinT, quantities.iFinCTiFinTV);
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(iFiniCTiFinT, quantities.iFiniCTiFinTV);
    quantities.dCTdTV = Core::LinAlg::make_strain_like_voigt_matrix(
        thermal_right_cg_temp_deriv_tensor);  // must be in strain-form for contraction afterwards!

    // compute principal invariants of the thermal stretch
    Core::LinAlg::Voigt::Stresses::invariants_principal(quantities.prinv, quantities.CTV);

    // compute derivatives of the thermal stretch principal invariants
    quantities.dPI.clear();
    quantities.ddPII.clear();
    for (const auto& p : potsumel)  // only for isotropic components
    {
      p->add_derivatives_principal(quantities.dPI, quantities.ddPII, quantities.prinv, gp, eleGID);
    }


    return quantities;
  }

}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif