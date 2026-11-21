// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MEMBRANE_ELASTHYPER_SERVICE_HPP
#define FOUR_C_MAT_MEMBRANE_ELASTHYPER_SERVICE_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_elast_summand.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  // Forward declaration
  class SummandProperties;

  /*!
   * @brief hyperelastic stress response plus elasticity tensor for membranes assuming
   * incompressibility and plane stress
   *
   * @param cauchygreen (in) : right Cauchy-Green tensor
   * @param params (in) : Container for additional information
   * @param Q_trafo (in) : Trafo from local membrane orthonormal coordinates to global coordinates
   * @param stress (out) : 2nd Piola-Kirchhoff stresses in stress-like voigt notation
   * @param cmat (out) : Constitutive matrix
   * @param gp (in) : Gauss point
   * @param eleGID (in) : Global element id
   * @param potsum (in) : Vector of summands of the strain energy function
   * @param properties (in) : Properties of the summands of the strain energy function
   */
  void membrane_elast_hyper_evaluate_isotropic_stress_cmat(
      const Core::LinAlg::Matrix<3, 3>& cauchygreen, const Teuchos::ParameterList& params,
      const Core::LinAlg::Matrix<3, 3>& Q_trafo, Core::LinAlg::Matrix<3, 1>& stress,
      Core::LinAlg::Matrix<3, 3>& cmat, int gp, int eleGID,
      const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& potsum,
      const SummandProperties& properties);


  /*!
   * @brief calculates the kinematic quantities and tensors used afterwards
   *
   * @param cauchygreen (in) : right Cauchy-Green tensor
   * @param id2 (out) : cartesian identity 2-tensor I_{AB}
   * @param id4sharp (out) : cartesian identity 4-tensor
   * @param rcg (out) : right Cauchy-Green in stress-like 3-Voigt notation
   * @param rcg33 (out) : principal stretch in thickness direction
   * @param icg (out) : inverse right Cauchy-Green in stress-like 3-Voigt notation
   */
  void membrane_elast_hyper_evaluate_kin_quant(const Core::LinAlg::Matrix<3, 3>& cauchygreen,
      Core::LinAlg::Matrix<3, 1>& id2, Core::LinAlg::Matrix<3, 3>& id4sharp,
      Core::LinAlg::Matrix<3, 1>& rcg, double& rcg33, Core::LinAlg::Matrix<3, 1>& icg);

  /*!
   * @brief Computes the isotropic stress response and the linearization
   *
   * @param stress_iso (out) : Isotropic stress response in local membrane coordinate system (plane
   * stress)
   * @param cmat_iso (out) : Linearization of stress response
   * @param id2 (in) : 2D identity tensor
   * @param id4sharp (in) : Fourth order identity tensor
   * @param rcg (in) : Right Cauchy-Green tensor in local membrane coordinate system (strain like)
   * @param rcg33 (in) : Right Cauchy-Green tensor in membrane thickness direction
   * @param icg (in) : Inverse Right Cauchy-Green tensor in local membrane coordinate system (strain
   * like)
   * @param gp (in) : Gauss point
   * @param eleGID (in) : global element id
   * @param potsum (in) : Vector of summands of the strain energy function
   * @param properties (in) : Properties of the summands of the strain energy function
   */
  void membrane_elast_hyper_evaluate_isotropic_stress_cmat(Core::LinAlg::Matrix<3, 1>& stress_iso,
      Core::LinAlg::Matrix<3, 3>& cmat_iso, const Core::LinAlg::Matrix<3, 1>& id2,
      const Core::LinAlg::Matrix<3, 3>& id4sharp, const Core::LinAlg::Matrix<3, 1>& rcg,
      const double& rcg33, const Core::LinAlg::Matrix<3, 1>& icg, int gp, int eleGID,
      const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& potsum,
      const SummandProperties& properties);

  /*!
   * @brief calculate principal invariants
   *
   * @param prinv (out) : principal invariants
   * @param rcg (in) : right cauchy-green in stress-like 3-Voigt notation
   * @param rcg33 (in) :principal stretch in thickness direction
   */
  void membrane_elast_hyper_invariants_principal(Core::LinAlg::Matrix<3, 1>& prinv,
      const Core::LinAlg::Matrix<3, 1>& rcg, const double& rcg33);

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif