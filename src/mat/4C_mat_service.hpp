// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_SERVICE_HPP
#define FOUR_C_MAT_SERVICE_HPP


#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  template <int dim>
  class FourTensor;
}

namespace Core::Mat::PAR
{
  class Parameter;
}

namespace Mat
{

  /*!
   * @brief Returns the second invariant of the deviatoric stress tensor,
   * \f$J_2 = \dfrac{1}{2} \mathbf(s) : \mathbf{s}\f$
   */
  double second_invariant_of_deviatoric_stress(const Core::LinAlg::Matrix<3, 3>& stress);



  /*!
   * @brief Put apart the volumetric and isochoric part of the 2nd Piola--Kirchhoff stress and the
   * material constitutive tensor
   *
   * The volumetric 2nd Piola--Kirchhof stress \f$S^{AB}_\text{vol}\f$
   * are obtained by
   * \f[
   *   S^{AB}_\text{vol}
   *   = \frac{1}{3} \left( S^{CD} \, C_{CD} \right) \, {C^{-1}}^{AB}
   * \f]
   *
   * and its corresponding contribution to the constitutive tensor
   * \f$C^{ABCD}_\text{vol}\f$ is
   * \f[
   *   C^{ABCD}_\text{vol}
   *   = \frac{2}{3} {C^{-1}}^{AB} \, S^{CD}_\text{lin}
   *   + \frac{2}{3} \left(S^{EF} C_{EF}\right) \, \left( -\frac{1}{2} \left(
   *     {C^{-1}}^{AC} {C^{-1}}^{BD} + {C^{-1}}^{AD} {C^{-1}}^{BC}
   *   \right) \right)
   * \f]
   *
   * with the 'linearised' 2nd Piola--Kirchhoff stresses
   * \f[
   *   S^{CD}_\text{lin} = S^{CD} + \frac{1}{2} C_{AB} C^{ABCD} .
   * \f]
   *
   * The isochoric 2nd Piola--Kirchhoff stress \f$S^{AB}_\text{iso}\f$
   * and and the isochoric constribution \f$C^{ABCD}_\text{iso}\f$
   * to the constitutive tensor are
   * \f[
   *   S^{AB}_\text{iso} = S^{AB} - S^{AB}_\text{vol}
   * \f]
   *
   * and
   * \f[
   *   C^{ABCD}_\text{iso} = C^{ABCD} - C^{ABCD}_\text{vol}
   * \f].
   *
   * @param[in,out] pk2vol  Volumetric 2nd Piola--Kirchhoff 6-Voigt stress, i.e.
   *                        \f$[ S^{11}\; S^{22}\; S^{33}\; S^{12}\; S^{23}\; S^{31}
   *                        ]_\text{vol}\f$, if != nullptr
   * @param[in] cvol        Volumetric contribution to constitutive tensor, if != nullptr
   * @param[in] pk2iso      Isochoric 2nd Piola--Kirchhoff 6-Voigt stress, i.e.
   *                        \f$[ S^{11}\; S^{22}\; S^{33}\; S^{12}\; S^{23}\; S^{31}
   *                        ]_\text{iso}\f$, if != nullptr
   * @param[in,out] ciso    Isochoric contribution to constitutive tensor, if != nullptr
   * @param[in] gl          Green--Lagrange 6-Voigt strain, i.e. \f$[ E_{11}\; E_{22}\; E_{33}\;
   *                        2E_{12}\; 2E_{23}\; 2E_{31} ]\f$
   * @param[in] pk2         2nd Piola--Kirchhoff 6-Voigt stress, i.e. \f$[ S^{11}\; S^{22}\;
   *                        S^{33}\; S^{12}\; S^{23}\; S^{31} ]\f$
   * @param[in] cmat        Constitutive tensor
   */
  void volumetrify_and_isochorify(Core::LinAlg::Matrix<6, 1>* pk2vol,
      Core::LinAlg::Matrix<6, 6>* cvol, Core::LinAlg::Matrix<6, 1>* pk2iso,
      Core::LinAlg::Matrix<6, 6>* ciso, const Core::LinAlg::Matrix<6, 1>& gl,
      const Core::LinAlg::Matrix<6, 1>& pk2, const Core::LinAlg::Matrix<6, 6>& cmat);



  /*!
   * @brief Evaluates the principal invariants of any symmetric tensor
   *
   * @param[out] prinv  Principal invariants of the tensor
   * @param[in]  tens   Tensor
   */
  void invariants_principal(
      Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 3>& tens);

  /*!
   * @brief Converts the principal invariants to the modified principal invariants.
   *
   * @param[out] modinv Modified principal invariants of the Right Cauchy-Green strain tensor
   * @param[in] prinv   Principal invariants of the Right Cauchy-Green strain tensor
   */
  void invariants_modified(
      Core::LinAlg::Matrix<3, 1>& modinv, const Core::LinAlg::Matrix<3, 1>& prinv);

  /*!
   * @brief Evaluates principal stretches \f$\lambda_\alpha\f$ and material principal directions
   *
   * @param[out] prstr  Principal stretches
   * @param[out] prdir  Principal directions
   * @param[in] rcg     Right Cauchy-Green strain tensor in strain-like Voigt notation
   */
  void stretches_principal(Core::LinAlg::Matrix<3, 1>& prstr, Core::LinAlg::Matrix<3, 3>& prdir,
      const Core::LinAlg::Matrix<6, 1>& rcg);

  /*!
   * @brief Evaluates the modified principal stretches \f$\bar{\lambda}_\alpha\f$
   *
   * @param[out] modstr Modified principal stretches
   * @param[in] prstr   Principal stretches
   */
  void stretches_modified(
      Core::LinAlg::Matrix<3, 1>& modstr, const Core::LinAlg::Matrix<3, 1>& prstr);


  template <class T>
  T* create_material_parameter_instance(Core::Mat::PAR::Parameter* curmat)
  {
    auto* params = dynamic_cast<T*>(curmat);
    return params;
  }

  /*!
   * @brief Pull back of a symmetric elastic 4th order tensor (in matrix/voigt notation) via the 2nd
   * order deformation gradient (also in matrix notation)
   */
  Core::LinAlg::Matrix<6, 6> pull_back_four_tensor(const double det_F,
      const Core::LinAlg::Matrix<3, 3>& F_inv, const Core::LinAlg::Matrix<6, 6>& cmat_voigt);

  /*!
   * @brief Pull back the ijkl-th entry of a symmetric elastic 4th order tensor (in matrix/voigt
   * notation) via the 2nd order deformation gradient (also in matrix notation)
   */
  double get_pull_back_four_tensor_entry(const double det_F,
      const Core::LinAlg::Matrix<3, 3>& F_inv, const Core::LinAlg::FourTensor<3>& four_tensor,
      const int i, const int j, const int k, const int l);

  /*!
   * @brief Push forward operation on a stress-like voigt tensor with a deformation gradient
   *
   * @return Core::LinAlg::Matrix<6, 1>
   */
  Core::LinAlg::Matrix<6, 1> push_forward_stress_tensor_voigt(
      const Core::LinAlg::Matrix<6, 1>& stress_elastic,
      const Core::LinAlg::Matrix<3, 3>& deformation_gradient);

  /*!
   * @brief Push forward of a symmetric elastic 4th order tensor (in matrix/voigt notation) via
   * the 2nd order deformation gradient (also in matrix notation)
   */
  Core::LinAlg::Matrix<6, 6> push_forward_four_tensor(const double det_F,
      const Core::LinAlg::Matrix<3, 3>& defgrd, const Core::LinAlg::Matrix<6, 6>& cmat_lagr_voigt);

  /*!
   * @brief Compute the fourth order linear isotropic elastic tensor
   *
   * @param[out] elasticity_tensor  The resulting elastic tensor
   * @param[in]  youngs_modulus     Young's modulus
   * @param[in]  poisson_ratio      Poisson ratio
   */
  void setup_linear_isotropic_elastic_tensor(Core::LinAlg::FourTensor<3>& elasticity_tensor,
      const double youngs_modulus, const double poisson_ratio);


}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
