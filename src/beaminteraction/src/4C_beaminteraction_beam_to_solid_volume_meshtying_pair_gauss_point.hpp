// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_VOLUME_MESHTYING_PAIR_GAUSS_POINT_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_VOLUME_MESHTYING_PAIR_GAUSS_POINT_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_base.hpp"

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
   * \brief Class for beam to solid meshtying using Gauss point projection.
   * @param beam Type from GeometryPair::ElementDiscretization... representing the beam.
   * @param solid Type from GeometryPair::ElementDiscretization... representing the solid.
   */
  template <typename Beam, typename Solid>
  class BeamToSolidVolumeMeshtyingPairGaussPoint
      : public BeamToSolidVolumeMeshtyingPairBase<
            GeometryPair::line_to_volume_scalar_type<Beam, Solid>, Beam, Solid>
  {
   protected:
    //! Shortcut to the base class.
    using base_class =
        BeamToSolidVolumeMeshtyingPairBase<GeometryPair::line_to_volume_scalar_type<Beam, Solid>,
            Beam, Solid>;

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
    BeamToSolidVolumeMeshtyingPairGaussPoint();


    /**
     * \brief Evaluate this contact element pair.
     * @param forcevec1 (out) Force vector on element 1.
     * @param forcevec2 (out) Force vector on element 2.
     * @param stiffmat11 (out) Stiffness contributions on element 1 - element 1.
     * @param stiffmat12 (out) Stiffness contributions on element 1 - element 2.
     * @param stiffmat21 (out) Stiffness contributions on element 2 - element 1.
     * @param stiffmat22 (out) Stiffness contributions on element 2 - element 2.
     * @return True if pair is in contact.
     */
    bool evaluate(Core::LinAlg::SerialDenseVector* forcevec1,
        Core::LinAlg::SerialDenseVector* forcevec2, Core::LinAlg::SerialDenseMatrix* stiffmat11,
        Core::LinAlg::SerialDenseMatrix* stiffmat12, Core::LinAlg::SerialDenseMatrix* stiffmat21,
        Core::LinAlg::SerialDenseMatrix* stiffmat22) override;

    /**
     * \brief Evaluate the pair and directly assemble it into the global force vector and stiffness
     * matrix (derived).
     *
     * Rotational coupling contributions will be added in this method.
     */
    void evaluate_and_assemble(const std::shared_ptr<const Core::FE::Discretization>& discret,
        const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
        const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
        const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector) override;

   private:
    /**
     * \brief Evaluate rotational coupling terms.
     * @param rot_coupling_type (in) Type of rotational coupling.
     * @param q_solid (in) Displacements of the solid.
     * @param triad_interpolation_scheme (in) Interpolation scheme for deformed beam triads.
     * @param ref_triad_interpolation_scheme (in) Interpolation scheme for reference beam triads.
     * @param local_force (in/out) Local pair force vector.
     * @param local_stiff (in/out) Local pair stiffness matrix.
     */
    void evaluate_rotational_coupling_terms(
        const Inpar::BeamToSolid::BeamToSolidRotationCoupling& rot_coupling_type,
        const GeometryPair::ElementData<Solid, scalar_type_rot_2nd>& q_solid,
        const LargeRotations::TriadInterpolationLocalRotationVectors<3, double>&
            triad_interpolation_scheme,
        const LargeRotations::TriadInterpolationLocalRotationVectors<3, double>&
            ref_triad_interpolation_scheme,
        Core::LinAlg::Matrix<n_dof_pair_, 1, double>& local_force,
        Core::LinAlg::Matrix<n_dof_pair_, n_dof_pair_, double>& local_stiff) const;
  };
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
