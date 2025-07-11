// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_UTILS_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_UTILS_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_faces.hpp"
#include "4C_linalg_fevector.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

// Forward declarations.
namespace BeamInteraction
{
  class BeamToSolidMortarManager;
  class BeamContactPair;
}  // namespace BeamInteraction
namespace Inpar
{
  namespace BeamToSolid
  {
    enum class BeamToSolidRotationCoupling;
    enum class BeamToSolidMortarShapefunctions;
  }  // namespace BeamToSolid
}  // namespace Inpar
namespace Core::LinAlg
{
  template <unsigned int rows, unsigned int cols, class ValueType>
  class Matrix;
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace LargeRotations
{
  template <unsigned int numnodes, typename T>
  class TriadInterpolationLocalRotationVectors;
}
namespace BeamInteraction
{
  class BeamContactPair;
  class BeamToSolidMortarManager;
  class BeamToSolidSurfaceContactParams;
  class BeamToSolidParamsBase;
}  // namespace BeamInteraction

namespace BeamInteraction
{
  // FAD type templated for two variables with a maximum derivative order of 1
  using fad_type_1st_order_2_variables =
      typename Core::FADUtils::HigherOrderFadType<1, Sacado::ELRFad::SLFad<double, 2>>::type;

  /**
   * \brief Evaluate the penalty force depending on the gap function.
   * @param gap (in) Gap function value.
   * @return Penalty force.
   */
  template <typename ScalarType>
  ScalarType penalty_force(
      const ScalarType& gap, const BeamToSolidSurfaceContactParams& contact_params);

  /**
   * \brief Evaluate the penalty potential depending on the gap function.
   * @param gap (in) Gap function value.
   * @return Penalty potential.
   */
  template <typename ScalarType>
  ScalarType penalty_potential(
      const ScalarType& gap, const BeamToSolidSurfaceContactParams& contact_params);

  /**
   * \brief Get the number of Lagrange multiplicator values corresponding to the beam nodes and beam
   * element.
   * @param beam_to_solid_params (in) Beam-to-solid parameters
   * @param shape_function (in) Mortar shape function.
   * @param n_dim (in) Spatial dimension of Lagrange multiplicator field.
   * @return {n_lambda_node, n_lambda_element} Number of Lagrange multiplicators per node and per
   * element.
   */
  [[nodiscard]] std::pair<unsigned int, unsigned int>
  mortar_shape_functions_to_number_of_lagrange_values(
      const std::shared_ptr<const BeamToSolidParamsBase>& beam_to_solid_params,
      const Inpar::BeamToSolid::BeamToSolidMortarShapefunctions shape_function,
      const unsigned int n_dim);

  /**
   * \brief Setup the triad interpolation scheme for the current triad and reference triad of the
   * given beam element.
   * @param discret (in) discretization.
   * @param displacement_vector (in) Global displacement vector.
   * @param ele (in) Pointer to the beam element.
   * @param triad_interpolation_scheme (out) Interpolation of current triad field..
   * @param ref_triad_interpolation_scheme (out) Interpolation of reference triad field.
   */
  void get_beam_triad_interpolation_scheme(const Core::FE::Discretization& discret,
      const Core::LinAlg::Vector<double>& displacement_vector, const Core::Elements::Element* ele,
      LargeRotations::TriadInterpolationLocalRotationVectors<3, double>& triad_interpolation_scheme,
      LargeRotations::TriadInterpolationLocalRotationVectors<3, double>&
          ref_triad_interpolation_scheme);

  /**
   * \brief Get the rotation vector of a triad constructed in the solid.
   * @param rot_coupling_type (in) Type of triad construction.
   * @param xi (in) Parameter coordinates in the solid.
   * @param q_solid_ref (in) Reference position of the solid.
   * @param q_solid (in) Displacement of the solid.
   * @param quaternion_beam_ref (in) Beam reference quaternion at the solid point.
   * @param psi_solid (out) Rotation vector of the constructed solid triad.
   */
  template <typename Solid, typename ScalarType>
  void get_solid_rotation_vector(
      const Inpar::BeamToSolid::BeamToSolidRotationCoupling& rot_coupling_type,
      const Core::LinAlg::Matrix<3, 1, double>& xi,
      const GeometryPair::ElementData<Solid, double>& q_solid_ref,
      const GeometryPair::ElementData<Solid, ScalarType>& q_solid,
      const Core::LinAlg::Matrix<4, 1, double>& quaternion_beam_ref,
      Core::LinAlg::Matrix<3, 1, ScalarType>& psi_solid);

  /**
   * \brief Construct a solid triad depending on the deformation gradient and return the rotation
   * vector of said triad. The construction is based on the average vector of the deformed triad.
   *
   * @param xi (in) Parameter coordinates in the solid.
   * @param q_solid_ref (in) Reference position of the solid.
   * @param q_solid (in) Displacement of the solid.
   * @param quaternion_beam_ref (in) Beam reference quaternion at the solid point.
   * @param psi_solid (out) Rotation vector of the constructed solid triad.
   */
  template <typename Solid, typename ScalarType>
  void get_solid_rotation_vector_deformation_gradient_3d_general(
      const Core::LinAlg::Matrix<3, 1, double>& xi,
      const GeometryPair::ElementData<Solid, double>& q_solid_ref,
      const GeometryPair::ElementData<Solid, ScalarType>& q_solid,
      const Core::LinAlg::Matrix<4, 1, double>& quaternion_beam_ref,
      Core::LinAlg::Matrix<3, 1, ScalarType>& psi_solid);

  /**
   * \brief Construct a solid triad depending on the deformation gradient and return the rotation
   * vector of said triad. The construction is based on cross section basis vectors.
   *
   * @param xi (in) Parameter coordinates in the solid.
   * @param q_solid_ref (in) Reference position of the solid.
   * @param q_solid (in) Displacement of the solid.
   * @param quaternion_beam_ref (in) Beam reference quaternion at the solid point.
   * @param psi_solid (out) Rotation vector of the constructed solid triad.
   */
  template <typename Solid, typename ScalarType>
  void get_solid_rotation_vector_deformation_gradient_3d_general_in_cross_section_plane(
      const Core::LinAlg::Matrix<3, 1, double>& xi,
      const GeometryPair::ElementData<Solid, double>& q_solid_ref,
      const GeometryPair::ElementData<Solid, ScalarType>& q_solid,
      const Core::LinAlg::Matrix<4, 1, double>& quaternion_beam_ref,
      Core::LinAlg::Matrix<3, 1, ScalarType>& psi_solid);

  /**
   * \brief Construct a solid triad depending on the deformation gradient and return the rotation
   * vector of said triad. The construction is based on cross section basis vectors.
   *
   * @param F (in) Deformation gradient.
   * @param beam_ref_triad (in) Reference triad of the beam.
   * @param psi_solid (out) Rotation vector of the constructed solid triad.
   */
  template <typename ScalarType>
  void get_solid_rotation_vector_deformation_gradient_3d_general_in_cross_section_plane(
      const Core::LinAlg::Matrix<3, 3, ScalarType>& F,
      const Core::LinAlg::Matrix<3, 3, double>& beam_ref_triad,
      Core::LinAlg::Matrix<3, 1, ScalarType>& psi_solid);

  /**
   * \brief Construct a solid triad depending on the deformation gradient and return the rotation
   * vector of said triad. The construction is based on the first basis vector of the deformed
   * triad.
   *
   * @param xi (in) Parameter coordinates in the solid.
   * @param q_solid_ref (in) Reference position of the solid.
   * @param q_solid (in) Displacement of the solid.
   * @param quaternion_beam_ref (in) Beam reference quaternion at the solid point.
   * @param psi_solid (out) Rotation vector of the constructed solid triad.
   */
  template <typename Solid, typename ScalarType>
  void get_solid_rotation_vector_deformation_gradient_3d_base1(
      const Core::LinAlg::Matrix<3, 1, double>& xi,
      const GeometryPair::ElementData<Solid, double>& q_solid_ref,
      const GeometryPair::ElementData<Solid, ScalarType>& q_solid,
      const Core::LinAlg::Matrix<4, 1, double>& quaternion_beam_ref,
      Core::LinAlg::Matrix<3, 1, ScalarType>& psi_solid);

  /**
   * \brief Construct a solid triad depending on the deformation gradient and return the rotation
   * vector of said triad. The construction starts with a user-given base vector.
   *
   * @param rot_coupling_type (in) Type of triad construction.
   * @param xi (in) Parameter coordinates in the solid.
   * @param q_solid_ref (in) Reference position of the solid.
   * @param q_solid (in) Displacement of the solid.
   * @param quaternion_beam_ref (in) Beam reference quaternion at the solid point.
   * @param psi_solid (out) Rotation vector of the constructed solid triad.
   */
  template <typename Solid, typename ScalarType>
  void get_solid_rotation_vector_deformation_gradient3_d(
      const Inpar::BeamToSolid::BeamToSolidRotationCoupling& rot_coupling_type,
      const Core::LinAlg::Matrix<3, 1, double>& xi,
      const GeometryPair::ElementData<Solid, double>& q_solid_ref,
      const GeometryPair::ElementData<Solid, ScalarType>& q_solid,
      const Core::LinAlg::Matrix<4, 1, double>& quaternion_beam_ref,
      Core::LinAlg::Matrix<3, 1, ScalarType>& psi_solid);

  /**
   * \brief Perform a 2D polar decomposition of the deformation gradient and return the rotation
   * vector (2d) of R.
   *
   * @param xi (in) Parameter coordinates in the solid.
   * @param q_solid_ref (in) Reference position of the solid.
   * @param q_solid (in) Displacement of the solid.
   * @param quaternion_beam_ref (in) Beam reference quaternion at the solid point.
   * @param psi_solid (out) Rotation vector of the constructed solid triad.
   */
  template <typename Solid, typename ScalarType>
  void get_solid_rotation_vector_polar_decomposition2_d(
      const Core::LinAlg::Matrix<3, 1, double>& xi,
      const GeometryPair::ElementData<Solid, double>& q_solid_ref,
      const GeometryPair::ElementData<Solid, ScalarType>& q_solid,
      const Core::LinAlg::Matrix<4, 1, double>& quaternion_beam_ref,
      Core::LinAlg::Matrix<3, 1, ScalarType>& psi_solid);

  /**
   * \brief Construct a solid triad depending on a 2d deformation gradient and return the rotation
   * vector (2d) of said triad.
   *
   * @param rot_coupling_type (in) Type of triad construction.
   * @param xi (in) Parameter coordinates in the solid.
   * @param q_solid_ref (in) Reference position of the solid.
   * @param q_solid (in) Displacement of the solid.
   * @param quaternion_beam_ref (in) Beam reference quaternion at the solid point.
   * @param psi_solid (out) Rotation vector of the constructed solid triad.
   */
  template <typename Solid, typename ScalarType>
  void get_solid_rotation_vector_deformation_gradient2_d(
      const Inpar::BeamToSolid::BeamToSolidRotationCoupling& rot_coupling_type,
      const Core::LinAlg::Matrix<3, 1, double>& xi,
      const GeometryPair::ElementData<Solid, double>& q_solid_ref,
      const GeometryPair::ElementData<Solid, ScalarType>& q_solid,
      const Core::LinAlg::Matrix<4, 1, double>& quaternion_beam_ref,
      Core::LinAlg::Matrix<3, 1, ScalarType>& psi_solid);

  /**
   * \brief Check if the given solid deformation gradient as well as the given beam cross section
   * quaternion are plane with respect to the y-z plane.
   * @param deformation_gradient (in) Deformation gradient at a solid point solid.
   * @param quaternion_beam_ref (in) Quaternion of a beam cross section.
   */
  template <typename ScalarType>
  void check_plane_rotations(const Core::LinAlg::Matrix<3, 3, ScalarType> deformation_gradient,
      const Core::LinAlg::Matrix<4, 1, double>& quaternion_beam_ref);

  /**
   * \brief Assemble local mortar contributions from the classical mortar matrices D and M into the
   * global matrices.
   *
   * This function assumes that the mortar contributions are symmetric, i.e.
   * \p global_constraint_lin_beam = \p global_force_beam_lin_lambda^T and \p
   * global_constraint_lin_solid = \p global_force_solid_lin_lambda^T.
   *
   * @param pair (in) The beam-to-solid pair.
   * @param discret (in) discretization
   * @param mortar_manager (in) Mortar manager for the beam-to-solid condition
   * @param global_constraint_lin_beam (in/out) Constraint equations derived w.r.t the beam DOFs
   * @param global_constraint_lin_solid (in/out) Constraint equations derived w.r.t the solid DOFs
   * @param global_force_beam_lin_lambda (in/out) Beam force vector derived w.r.t the Lagrange
   * multipliers
   * @param global_force_solid_lin_lambda (in/out) Solid force vector derived w.r.t the Lagrange
   * multipliers
   * @param global_constraint (in/out) Global constraint equations
   * @param global_kappa (in/out) Global penalty scaling vector equations
   * @param global_lambda_active (in/out) Global vector keeping track of active lagrange multipliers
   * @param local_D (in) Local D matrix of the pair.
   * @param local_M (in) Local M matrix of the pair.
   * @param local_kappa (in) Local scaling vector of the pair.
   * @param local_constraint (in) Local constraint contributions of the pair.
   * @param n_mortar_rot (int) Number of total rotational Lagrange multiplier DOFs per beam.
   */
  template <typename Beam, typename Other, typename Mortar>
  void assemble_local_mortar_contributions(const BeamInteraction::BeamContactPair* pair,
      const Core::FE::Discretization& discret, const BeamToSolidMortarManager* mortar_manager,
      Core::LinAlg::SparseMatrix& global_constraint_lin_beam,
      Core::LinAlg::SparseMatrix& global_constraint_lin_solid,
      Core::LinAlg::SparseMatrix& global_force_beam_lin_lambda,
      Core::LinAlg::SparseMatrix& global_force_solid_lin_lambda,
      Core::LinAlg::FEVector<double>& global_constraint,
      Core::LinAlg::FEVector<double>& global_kappa,
      Core::LinAlg::FEVector<double>& global_lambda_active,
      const Core::LinAlg::Matrix<Mortar::n_dof_, Beam::n_dof_, double>& local_D,
      const Core::LinAlg::Matrix<Mortar::n_dof_, Other::n_dof_, double>& local_M,
      const Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_kappa,
      const Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_constraint,
      const unsigned int n_mortar_rot = 0);

  /**
   * \brief Get the GIDs of the beam and a surface in a beam-to-surface pair.
   * @param discret (in) discretization.
   * @param beam_element (in) Beam element.
   * @param face_element (in) Face element for the surface.
   * @return Vectors with the GIDs of the beam and the solid.
   */
  template <typename Beam, typename Surface, typename ScalarType>
  std::tuple<Core::LinAlg::Matrix<Beam::n_dof_, 1, int>, const std::vector<int>&>
  get_beam_to_surface_pair_gid(const Core::FE::Discretization& discret,
      const Core::Elements::Element& beam_element,
      const GeometryPair::FaceElementTemplate<Surface, ScalarType>& face_element)
  {
    // Get the beam centerline GIDs.
    Core::LinAlg::Matrix<Beam::n_dof_, 1, int> beam_centerline_gid;
    Utils::get_element_centerline_gid_indices(discret, &beam_element, beam_centerline_gid);

    // Get the patch (in this case just the one face element) GIDs.
    const std::vector<int>& patch_gid = face_element.get_patch_gid();
    return {beam_centerline_gid, patch_gid};
  }

  /**
   * \brief Get the combined GIDs of the a beam-to-surface pair, i.e. first the beam GIDs and then
   * the surface GIDs.
   * @param discret (in) discretization.
   * @param beam_element (in) Beam element.
   * @param face_element (in) Face element for the surface.
   * @return Vector with the GIDs of this pair.
   */
  template <typename Beam, typename Surface, typename ScalarType>
  std::vector<int> get_beam_to_surface_pair_gid_combined(const Core::FE::Discretization& discret,
      const Core::Elements::Element& beam_element,
      const GeometryPair::FaceElementTemplate<Surface, ScalarType>& face_element)
  {
    const auto [beam_centerline_gid, patch_gid] =
        get_beam_to_surface_pair_gid<Beam>(discret, beam_element, face_element);
    std::vector<int> pair_gid(Beam::n_dof_ + patch_gid.size(), -1);

    for (unsigned int i_dof_beam = 0; i_dof_beam < Beam::n_dof_; i_dof_beam++)
      pair_gid[i_dof_beam] = beam_centerline_gid(i_dof_beam);
    for (unsigned int i_dof_patch = 0; i_dof_patch < patch_gid.size(); i_dof_patch++)
      pair_gid[Beam::n_dof_ + i_dof_patch] = patch_gid[i_dof_patch];

    return pair_gid;
  }

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
