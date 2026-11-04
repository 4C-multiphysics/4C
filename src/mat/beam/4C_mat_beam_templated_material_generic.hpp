// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_BEAM_TEMPLATED_MATERIAL_GENERIC_HPP
#define FOUR_C_MAT_BEAM_TEMPLATED_MATERIAL_GENERIC_HPP

#include "4C_config.hpp"

#include "4C_mat_beam_material_generic.hpp"

FOUR_C_NAMESPACE_OPEN


// forward declaration
namespace Discret
{
  class ParObject;
}

namespace Mat
{
  // forward declaration
  namespace PAR
  {
    class BeamElastHyperMaterialParameterGeneric;
  }

  /*---------------------------------------------------------------------------------------------*/
  /// constitutive relations for beam cross-section resultants (hyperelastic stored energy function)
  template <typename T>
  class BeamMaterialTemplated : public BeamMaterial
  {
   public:
    /*
     * \brief Compute axial stress contributions
     *
     *\param[out] stressM axial stress
     *
     *\param[in] CM constitutive matrix
     *
     *\param[in] Cur curvature
     */
    virtual void evaluate_moment_contributions_to_stress(Core::LinAlg::Matrix<3, 1, T>& stressM,
        const Core::LinAlg::Matrix<3, 3, T>& CM, const Core::LinAlg::Matrix<3, 1, T>& Cur,
        const unsigned int gp) = 0;

    /*
     * \brief Compute axial stress contributions
     *
     *\param[out] stressN axial stress
     *
     *\param[in] CN constitutive matrix
     *
     *\param[in] Gamma triad
     */

    virtual void evaluate_force_contributions_to_stress(Core::LinAlg::Matrix<3, 1, T>& stressN,
        const Core::LinAlg::Matrix<3, 3, T>& CN, const Core::LinAlg::Matrix<3, 1, T>& Gamma,
        const unsigned int gp) = 0;

    /*
     * \brief Update material-dependent variables
     */
    virtual void compute_constitutive_parameter(
        Core::LinAlg::Matrix<3, 3, T>& C_N, Core::LinAlg::Matrix<3, 3, T>& C_M) = 0;

    /** \brief get constitutive matrix relating stress force resultants and translational strain
     *         measures, expressed w.r.t. material frame
     *
     */
    virtual void get_constitutive_matrix_of_forces_material_frame(
        Core::LinAlg::Matrix<3, 3, T>& C_N) const = 0;

    /** \brief get constitutive matrix relating stress moment resultants and rotational strain
     *         measures, expressed w.r.t. material frame
     *
     */
    virtual void get_constitutive_matrix_of_moments_material_frame(
        Core::LinAlg::Matrix<3, 3, T>& C_M) const = 0;

    /** \brief get linearization of the constitutive law relating stress moment resultants and
     * rotational strain measures, expressed w.r.t. material frame
     *
     *\param[in] C_M constitutive matrix
     *\param[out] stiffness_matrix
     */
    virtual void get_stiffness_matrix_of_moments(Core::LinAlg::Matrix<3, 3, T>& stiffness_matrix,
        const Core::LinAlg::Matrix<3, 3, T>& C_M, const int gp) = 0;

    /** \brief get linearization of the constitutive law relating stress force resultants and
     * translational strain measures, expressed w.r.t. material frame
     *
     *\param[in] C_N constitutive matrix
     *\param[out] stiffness_matrix
     */
    virtual void get_stiffness_matrix_of_forces(Core::LinAlg::Matrix<3, 3, T>& stiffness_matrix,
        const Core::LinAlg::Matrix<3, 3, T>& C_N, const int gp) = 0;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
