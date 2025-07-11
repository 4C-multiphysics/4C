// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_SURFACE_MESHTYING_PAIR_MORTAR_FAD_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_SURFACE_MESHTYING_PAIR_MORTAR_FAD_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_solid_surface_meshtying_pair_mortar_base.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_scalar_types.hpp"

FOUR_C_NAMESPACE_OPEN


// Forward declaration.
namespace Inpar
{
  namespace BeamToSolid
  {
    enum class BeamToSolidMortarShapefunctions;
    enum class BeamToSolidSurfaceRotationCoupling;
  }  // namespace BeamToSolid
  namespace GeometryPair
  {
    enum class SurfaceNormals;
  }
}  // namespace Inpar
namespace Core::LargeRotations
{
  template <unsigned int numnodes, typename T>
  class TriadInterpolationLocalRotationVectors;
}  // namespace Core::LargeRotations


namespace BeamInteraction
{
  /**
   * \brief Class for Mortar beam to surface surface mesh tying.
   * @tparam scalar_type Type for scalar variables.
   * @tparam beam Type from GeometryPair::ElementDiscretization... representing the beam.
   * @tparam surface Type from GeometryPair::ElementDiscretization... representing the surface.
   * @tparam mortar Type from BeamInteraction::ElementDiscretization... representing the mortar
   * shape functions.
   */
  template <typename ScalarType, typename Beam, typename Surface, typename Mortar>
  class BeamToSolidSurfaceMeshtyingPairMortarFAD
      : public BeamToSolidSurfaceMeshtyingPairMortarBase<ScalarType, Beam, Surface, Mortar>
  {
   private:
    //! Shortcut to the base class.
    using base_class = BeamToSolidSurfaceMeshtyingPairMortarBase<ScalarType, Beam, Surface, Mortar>;

   public:
    /**
     * \brief Standard Constructor
     */
    BeamToSolidSurfaceMeshtyingPairMortarFAD();


    /**
     * \brief Evaluate the pair and directly assemble it into the global force vector and stiffness
     * matrix. (derived)
     */
    void evaluate_and_assemble(const Core::FE::Discretization& discret,
        const BeamToSolidMortarManager* mortar_manager,
        const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
        const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
        const Core::LinAlg::Vector<double>& global_lambda,
        const Core::LinAlg::Vector<double>& displacement_vector) override;

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
  };


  /**
   * \brief Class for beam to solid surface rotational mesh tying.
   * @param beam Type from GeometryPair::ElementDiscretization... representing the beam.
   * @param surface Type from GeometryPair::ElementDiscretization... representing the solid.
   * @param mortar Type from BeamInteraction::ElementDiscretization... representing the mortar shape
   * functions for displacement coupling.
   *
   * @tparam scalar_type Scalar type for position DOFs.
   * @tparam beam Type of used line element.
   * @tparam scalar_type Type of used surface element.
   * @tparam scalar_type Type of used mortar interpolation.
   */
  template <typename ScalarType, typename Beam, typename Surface, typename Mortar>
  class BeamToSolidSurfaceMeshtyingPairMortarRotationFAD
      : public BeamToSolidSurfaceMeshtyingPairMortarFAD<ScalarType, Beam, Surface, Mortar>
  {
   protected:
    //! Shortcut to the base class.
    using base_class = BeamToSolidSurfaceMeshtyingPairMortarFAD<ScalarType, Beam, Surface, Mortar>;

    //! FAD type to evaluate the rotational coupling terms. The first 3 entries are the values of
    //! psi_beam, the following entries are the discrete solid DOFs.
    using scalar_type_rot_1st = typename Sacado::Fad::SLFad<double, 3 + Surface::n_dof_>;
    using scalar_type_rot_2nd =
        typename Core::FADUtils::HigherOrderFadType<2, scalar_type_rot_1st>::type;

    //! Number of rotational DOF for the SR beams;
    static constexpr unsigned int n_dof_rot_ = 9;
    static constexpr unsigned int n_dof_pair_ = n_dof_rot_ + Surface::n_dof_;

   public:
    /**
     * \brief Standard Constructor
     */
    BeamToSolidSurfaceMeshtyingPairMortarRotationFAD() : base_class()
    {
      this->n_mortar_rot_ = Mortar::n_dof_;
    };

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

   private:
    /**
     * \brief Calculate the solid rotation vector on the surface.
     * @param xi (in) Parameter coordinate in the surface
     * @param q_solid_ref (in) Solid position vector in the reference configuration
     * @param q_solid (in) Solid position vector in the current configuration
     * @param quaternion_beam_ref (in) Reference rotation of the beam.
     * @param surface_triad_type (in) How the surface triad should be constructed.
     * @param psi_solid (out) Rotation vector on solid surface.
     */
    template <typename ScalarTypeRotVec>
    void get_surface_rotation_vector(const Core::LinAlg::Matrix<3, 1, double>& xi,
        const GeometryPair::ElementData<Surface, double>& q_solid_ref,
        const GeometryPair::ElementData<Surface, ScalarTypeRotVec>& q_solid,
        const Core::LinAlg::Matrix<4, 1, double>& quaternion_beam_ref,
        const Inpar::BeamToSolid::BeamToSolidSurfaceRotationCoupling surface_triad_type,
        Core::LinAlg::Matrix<3, 1, ScalarTypeRotVec>& psi_solid) const;

    /**
     * \brief Get the rotational GIDs for the beam and surface.
     * @param discret (in) discretization.
     * @param gid_surface (out) GIDs for the surface that influence the rotation.
     * @param gid_rot (out) Rotational GIDs for the beam
     */
    void get_pair_rotational_gid(const Core::FE::Discretization& discret,
        std::vector<int>& gid_surface, Core::LinAlg::Matrix<n_dof_rot_, 1, int>& gid_rot) const;
  };

  /**
   * \brief Factory function for beam-to-solid mortar FAD pairs.
   * @param surface_shape (in) Type of surface element.
   * @param mortar_shapefunction (in) Type of mortar shape function.
   * @param surface_normal_strategy (in) Strategy for surface normal evaluation.
   * @return Pointer to the created pair.
   */
  std::shared_ptr<BeamInteraction::BeamContactPair>
  beam_to_solid_surface_meshtying_pair_mortar_fad_factory(const Core::FE::CellType surface_shape,
      const Inpar::BeamToSolid::BeamToSolidMortarShapefunctions mortar_shapefunction,
      const bool rotational_coupling,
      const Inpar::GeometryPair::SurfaceNormals surface_normal_strategy);
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
