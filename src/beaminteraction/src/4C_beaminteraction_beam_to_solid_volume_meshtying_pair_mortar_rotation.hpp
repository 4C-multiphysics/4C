// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_VOLUME_MESHTYING_PAIR_MORTAR_ROTATION_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_VOLUME_MESHTYING_PAIR_MORTAR_ROTATION_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_mortar.hpp"

FOUR_C_NAMESPACE_OPEN


// Forward declarations.
namespace Inpar
{
  namespace BeamToSolid
  {
    enum class BeamToSolidRotationCoupling;
  }
}  // namespace Inpar
namespace LargeRotations
{
  template <unsigned int numnodes, typename T>
  class TriadInterpolationLocalRotationVectors;
}  // namespace LargeRotations


namespace BeamInteraction
{
  /**
   * \brief Class for beam to solid rotational meshtying.
   * @param beam Type from GeometryPair::ElementDiscretization... representing the beam.
   * @param solid Type from GeometryPair::ElementDiscretization... representing the solid.
   * @param mortar Type from BeamInteraction::ElementDiscretization... representing the mortar shape
   * functions for displacement coupling.
   * @param mortar_rot Type from BeamInteraction::ElementDiscretization... representing the mortar
   * shape functions for rotational coupling.
   */
  template <typename Beam, typename Solid, typename Mortar, typename MortarRot>
  class BeamToSolidVolumeMeshtyingPairMortarRotation
      : public BeamToSolidVolumeMeshtyingPairMortar<Beam, Solid, Mortar>
  {
   protected:
    //! Shortcut to the base class.
    using base_class = BeamToSolidVolumeMeshtyingPairMortar<Beam, Solid, Mortar>;

    //! Type to be used for scalar AD variables.
    using scalar_type = typename base_class::scalar_type;

    //! FAD type to evaluate the rotational coupling terms. The first 3 entries are the values of
    //! psi_beam, the following entries are the discrete solid DOFs.
    using scalar_type_rot_1st = typename Sacado::Fad::SLFad<double, 3 + Solid::n_dof_>;
    using scalar_type_rot_2nd =
        typename Core::FADUtils::HigherOrderFadType<2, scalar_type_rot_1st>::type;

    //! Number of rotational DOF for the SR beams;
    static constexpr unsigned int n_dof_rot_ = 9;
    static constexpr unsigned int n_dof_pair_ = n_dof_rot_ + Solid::n_dof_;

   public:
    /**
     * \brief Standard Constructor
     */
    BeamToSolidVolumeMeshtyingPairMortarRotation();

    /**
     * \brief Evaluate the global matrices and vectors resulting from mortar coupling. (derived)
     */
    void evaluate_and_assemble_mortar_contributions(const Core::FE::Discretization& discret,
        const BeamToSolidMortarManager* mortar_manager,
        Core::LinAlg::SparseMatrix& global_constraint_lin_beam,
        Core::LinAlg::SparseMatrix& global_constraint_lin_solid,
        Core::LinAlg::SparseMatrix& global_force_beam_lin_lambda,
        Core::LinAlg::SparseMatrix& global_force_solid_lin_lambda,
        Core::LinAlg::FEVector<double>& global_constraint,
        Core::LinAlg::FEVector<double>& global_kappa,
        Core::LinAlg::SparseMatrix& global_kappa_lin_beam,
        Core::LinAlg::SparseMatrix& global_kappa_lin_solid,
        Core::LinAlg::FEVector<double>& global_lambda_active,
        const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector) override;

    /**
     * \brief Evaluate the pair and directly assemble it into the global force vector and stiffness
     * matrix (derived).
     */
    void evaluate_and_assemble(const Core::FE::Discretization& discret,
        const BeamToSolidMortarManager* mortar_manager,
        const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
        const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
        const Core::LinAlg::Vector<double>& global_lambda,
        const Core::LinAlg::Vector<double>& displacement_vector) override;

   private:
    /**
     * \brief Evaluate the constraint vector and the coupling matrices.
     */
    void evaluate_rotational_coupling_terms(
        const Inpar::BeamToSolid::BeamToSolidRotationCoupling& rot_coupling_type,
        const GeometryPair::ElementData<Solid, scalar_type_rot_1st>& q_solid,
        const LargeRotations::TriadInterpolationLocalRotationVectors<3, double>&
            triad_interpolation_scheme,
        const LargeRotations::TriadInterpolationLocalRotationVectors<3, double>&
            ref_triad_interpolation_scheme,
        Core::LinAlg::Matrix<MortarRot::n_dof_, 1, double>& local_g,
        Core::LinAlg::Matrix<MortarRot::n_dof_, n_dof_rot_, double>& local_G_B,
        Core::LinAlg::Matrix<MortarRot::n_dof_, Solid::n_dof_, double>& local_G_S,
        Core::LinAlg::Matrix<n_dof_rot_, MortarRot::n_dof_, double>& local_FB_L,
        Core::LinAlg::Matrix<Solid::n_dof_, MortarRot::n_dof_, double>& local_FS_L,
        Core::LinAlg::Matrix<MortarRot::n_dof_, 1, double>& local_kappa) const;

    /**
     * \brief Evaluate the stiffness contributions of this pair.
     */
    void evaluate_rotational_coupling_stiff_terms(
        const Inpar::BeamToSolid::BeamToSolidRotationCoupling& rot_coupling_type,
        const GeometryPair::ElementData<Solid, scalar_type_rot_2nd>& q_solid,
        Core::LinAlg::Matrix<MortarRot::n_dof_, 1, double>& lambda_rot,
        const LargeRotations::TriadInterpolationLocalRotationVectors<3, double>&
            triad_interpolation_scheme,
        const LargeRotations::TriadInterpolationLocalRotationVectors<3, double>&
            ref_triad_interpolation_scheme,
        Core::LinAlg::Matrix<n_dof_rot_, n_dof_rot_, double>& local_stiff_BB,
        Core::LinAlg::Matrix<n_dof_rot_, Solid::n_dof_, double>& local_stiff_BS,
        Core::LinAlg::Matrix<Solid::n_dof_, n_dof_rot_, double>& local_stiff_SB,
        Core::LinAlg::Matrix<Solid::n_dof_, Solid::n_dof_, double>& local_stiff_SS) const;
  };
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
