/*----------------------------------------------------------------------*/
/*! \file

\brief Meshtying element for meshtying between a 3D beam and a 3D solid element.

\level 3
*/


#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_gauss_point.hpp"

#include "4C_beam3_reissner.hpp"
#include "4C_beam3_triad_interpolation_local_rotation_vectors.hpp"
#include "4C_beaminteraction_beam_to_solid_utils.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_params.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_line_to_volume.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"

#include <Epetra_FEVector.h>

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
template <typename beam, typename solid>
BEAMINTERACTION::BeamToSolidVolumeMeshtyingPairGaussPoint<beam,
    solid>::BeamToSolidVolumeMeshtyingPairGaussPoint()
    : BeamToSolidVolumeMeshtyingPairBase<beam, solid>()
{
  // Empty constructor.
}


/**
 *
 */
template <typename beam, typename solid>
bool BEAMINTERACTION::BeamToSolidVolumeMeshtyingPairGaussPoint<beam, solid>::Evaluate(
    CORE::LINALG::SerialDenseVector* forcevec1, CORE::LINALG::SerialDenseVector* forcevec2,
    CORE::LINALG::SerialDenseMatrix* stiffmat11, CORE::LINALG::SerialDenseMatrix* stiffmat12,
    CORE::LINALG::SerialDenseMatrix* stiffmat21, CORE::LINALG::SerialDenseMatrix* stiffmat22)
{
  // Call Evaluate on the geometry Pair. Only do this once for meshtying.
  if (!this->meshtying_is_evaluated_)
  {
    GEOMETRYPAIR::ElementData<beam, double> beam_coupling_ref;
    GEOMETRYPAIR::ElementData<solid, double> solid_coupling_ref;
    this->get_coupling_reference_position(beam_coupling_ref, solid_coupling_ref);
    this->cast_geometry_pair()->Evaluate(
        beam_coupling_ref, solid_coupling_ref, this->line_to_3D_segments_);
    this->meshtying_is_evaluated_ = true;
  }

  // If there are no intersection segments, return no contact status.
  if (this->line_to_3D_segments_.size() == 0) return false;

  // Initialize variables for position and force vectors.
  CORE::LINALG::Matrix<3, 1, double> dr_beam_ref;
  CORE::LINALG::Matrix<3, 1, scalar_type> r_beam;
  CORE::LINALG::Matrix<3, 1, scalar_type> r_solid;
  CORE::LINALG::Matrix<3, 1, scalar_type> force;
  CORE::LINALG::Matrix<beam::n_dof_, 1, scalar_type> force_element_1(true);
  CORE::LINALG::Matrix<solid::n_dof_, 1, scalar_type> force_element_2(true);

  // Initialize scalar variables.
  double segment_jacobian = 0.0;
  double beam_segmentation_factor = 0.0;
  double penalty_parameter =
      this->Params()->beam_to_solid_volume_meshtying_params()->GetPenaltyParameter();

  // Calculate the meshtying forces.
  // Loop over segments.
  const unsigned int n_segments = this->line_to_3D_segments_.size();
  for (unsigned int i_segment = 0; i_segment < n_segments; i_segment++)
  {
    // Factor to account for the integration segment length.
    beam_segmentation_factor = 0.5 * this->line_to_3D_segments_[i_segment].GetSegmentLength();

    // Gauss point loop.
    const unsigned int n_gp = this->line_to_3D_segments_[i_segment].GetProjectionPoints().size();
    for (unsigned int i_gp = 0; i_gp < n_gp; i_gp++)
    {
      // Get the current Gauss point.
      const GEOMETRYPAIR::ProjectionPoint1DTo3D<double>& projected_gauss_point =
          this->line_to_3D_segments_[i_segment].GetProjectionPoints()[i_gp];

      // Get the jacobian in the reference configuration.
      GEOMETRYPAIR::EvaluatePositionDerivative1<beam>(
          projected_gauss_point.GetEta(), this->ele1posref_, dr_beam_ref);

      // Jacobian including the segment length.
      segment_jacobian = dr_beam_ref.Norm2() * beam_segmentation_factor;

      // Get the current positions on beam and solid.
      GEOMETRYPAIR::EvaluatePosition<beam>(projected_gauss_point.GetEta(), this->ele1pos_, r_beam);
      GEOMETRYPAIR::EvaluatePosition<solid>(projected_gauss_point.GetXi(), this->ele2pos_, r_solid);

      // Calculate the force in this Gauss point. The sign of the force calculated here is the one
      // that acts on the beam.
      force = r_solid;
      force -= r_beam;
      force.Scale(penalty_parameter);

      // The force vector is in R3, we need to calculate the equivalent nodal forces on the element
      // dof. This is done with the virtual work equation $F \delta r = f \delta q$.
      for (unsigned int i_dof = 0; i_dof < beam::n_dof_; i_dof++)
        for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
          force_element_1(i_dof) += force(i_dir) * r_beam(i_dir).dx(i_dof) *
                                    projected_gauss_point.GetGaussWeight() * segment_jacobian;
      for (unsigned int i_dof = 0; i_dof < solid::n_dof_; i_dof++)
        for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
          force_element_2(i_dof) -= force(i_dir) * r_solid(i_dir).dx(i_dof + beam::n_dof_) *
                                    projected_gauss_point.GetGaussWeight() * segment_jacobian;
    }
  }


  // Fill in the entries for the local matrices and vectors.
  {
    // Resize and initialize the return variables.
    if (forcevec1 != nullptr) forcevec1->size(beam::n_dof_);
    if (forcevec2 != nullptr) forcevec2->size(solid::n_dof_);
    if (stiffmat11 != nullptr) stiffmat11->shape(beam::n_dof_, beam::n_dof_);
    if (stiffmat12 != nullptr) stiffmat12->shape(beam::n_dof_, solid::n_dof_);
    if (stiffmat21 != nullptr) stiffmat21->shape(solid::n_dof_, beam::n_dof_);
    if (stiffmat22 != nullptr) stiffmat22->shape(solid::n_dof_, solid::n_dof_);

    if (forcevec1 != nullptr && forcevec2 != nullptr)
    {
      // $f_1$
      for (unsigned int i_dof = 0; i_dof < beam::n_dof_; i_dof++)
        (*forcevec1)(i_dof) = CORE::FADUTILS::CastToDouble(force_element_1(i_dof));
      // $f_2$
      for (unsigned int i_dof = 0; i_dof < solid::n_dof_; i_dof++)
        (*forcevec2)(i_dof) = CORE::FADUTILS::CastToDouble(force_element_2(i_dof));
    }

    if (stiffmat11 != nullptr && stiffmat12 != nullptr && stiffmat21 != nullptr &&
        stiffmat22 != nullptr)
    {
      // $k_{11}$
      for (unsigned int i_dof_1 = 0; i_dof_1 < beam::n_dof_; i_dof_1++)
        for (unsigned int i_dof_2 = 0; i_dof_2 < beam::n_dof_; i_dof_2++)
          (*stiffmat11)(i_dof_1, i_dof_2) =
              -CORE::FADUTILS::CastToDouble(force_element_1(i_dof_1).dx(i_dof_2));

      // $k_{12}, k_{21}$
      for (unsigned int i_dof_1 = 0; i_dof_1 < beam::n_dof_; i_dof_1++)
      {
        for (unsigned int i_dof_2 = 0; i_dof_2 < solid::n_dof_; i_dof_2++)
        {
          (*stiffmat12)(i_dof_1, i_dof_2) =
              -CORE::FADUTILS::CastToDouble(force_element_1(i_dof_1).dx(beam::n_dof_ + i_dof_2));
          (*stiffmat21)(i_dof_2, i_dof_1) =
              -CORE::FADUTILS::CastToDouble(force_element_2(i_dof_2).dx(i_dof_1));
        }
      }

      // $k_{22}$
      for (unsigned int i_dof_1 = 0; i_dof_1 < solid::n_dof_; i_dof_1++)
        for (unsigned int i_dof_2 = 0; i_dof_2 < solid::n_dof_; i_dof_2++)
          (*stiffmat22)(i_dof_1, i_dof_2) =
              -CORE::FADUTILS::CastToDouble(force_element_2(i_dof_1).dx(beam::n_dof_ + i_dof_2));
    }
  }

  // Return true as there are meshtying contributions.
  return true;
}


/**
 *
 */
template <typename beam, typename solid>
void BEAMINTERACTION::BeamToSolidVolumeMeshtyingPairGaussPoint<beam, solid>::EvaluateAndAssemble(
    const Teuchos::RCP<const DRT::Discretization>& discret,
    const Teuchos::RCP<Epetra_FEVector>& force_vector,
    const Teuchos::RCP<CORE::LINALG::SparseMatrix>& stiffness_matrix,
    const Teuchos::RCP<const Epetra_Vector>& displacement_vector)
{
  // This function only gives contributions for rotational coupling.
  auto rot_coupling_type =
      this->Params()->beam_to_solid_volume_meshtying_params()->get_rotational_coupling_type();
  if (rot_coupling_type == INPAR::BEAMTOSOLID::BeamToSolidRotationCoupling::none) return;

  // Call Evaluate on the geometry Pair. Only do this once for meshtying.
  if (!this->meshtying_is_evaluated_)
  {
    GEOMETRYPAIR::ElementData<beam, double> beam_coupling_ref;
    GEOMETRYPAIR::ElementData<solid, double> solid_coupling_ref;
    this->get_coupling_reference_position(beam_coupling_ref, solid_coupling_ref);
    this->cast_geometry_pair()->Evaluate(
        beam_coupling_ref, solid_coupling_ref, this->line_to_3D_segments_);
    this->meshtying_is_evaluated_ = true;
  }

  // If there are no intersection segments, return no contact status.
  if (this->line_to_3D_segments_.size() == 0) return;

  // Get the beam triad interpolation schemes.
  LARGEROTATIONS::TriadInterpolationLocalRotationVectors<3, double> triad_interpolation_scheme;
  LARGEROTATIONS::TriadInterpolationLocalRotationVectors<3, double> ref_triad_interpolation_scheme;
  GetBeamTriadInterpolationScheme(*discret, displacement_vector, this->Element1(),
      triad_interpolation_scheme, ref_triad_interpolation_scheme);

  // Set the FAD variables for the solid DOFs.
  auto q_solid =
      GEOMETRYPAIR::InitializeElementData<solid, scalar_type_rot_2nd>::Initialize(this->Element2());
  for (unsigned int i_solid = 0; i_solid < solid::n_dof_; i_solid++)
    q_solid.element_position_(i_solid) =
        CORE::FADUTILS::HigherOrderFadValue<scalar_type_rot_2nd>::apply(3 + solid::n_dof_,
            3 + i_solid, CORE::FADUTILS::CastToDouble(this->ele2pos_.element_position_(i_solid)));


  // Initialize local matrices.
  CORE::LINALG::Matrix<n_dof_pair_, 1, double> local_force(true);
  CORE::LINALG::Matrix<n_dof_pair_, n_dof_pair_, double> local_stiff(true);


  if (rot_coupling_type == INPAR::BEAMTOSOLID::BeamToSolidRotationCoupling::fix_triad_2d)
  {
    // In the case of "fix_triad_2d" we couple both, the ey and ez direction to the beam. Therefore,
    // we have to evaluate the coupling terms w.r.t both of those coupling types.
    evaluate_rotational_coupling_terms(
        INPAR::BEAMTOSOLID::BeamToSolidRotationCoupling::deformation_gradient_y_2d, q_solid,
        triad_interpolation_scheme, ref_triad_interpolation_scheme, local_force, local_stiff);
    evaluate_rotational_coupling_terms(
        INPAR::BEAMTOSOLID::BeamToSolidRotationCoupling::deformation_gradient_z_2d, q_solid,
        triad_interpolation_scheme, ref_triad_interpolation_scheme, local_force, local_stiff);
  }
  else
    evaluate_rotational_coupling_terms(rot_coupling_type, q_solid, triad_interpolation_scheme,
        ref_triad_interpolation_scheme, local_force, local_stiff);


  // Get the GIDs of this pair.
  // The first 9 entries in the vector will be the rotational DOFs of the beam, the other entries
  // are the solid DOFs.
  std::vector<int> lm_beam, lm_solid, lmowner, lmstride;
  this->Element1()->LocationVector(*discret, lm_beam, lmowner, lmstride);
  this->Element2()->LocationVector(*discret, lm_solid, lmowner, lmstride);
  std::array<int, 9> rot_dof_indices = {3, 4, 5, 12, 13, 14, 18, 19, 20};
  CORE::LINALG::Matrix<n_dof_pair_, 1, int> gid_pair;
  for (unsigned int i = 0; i < n_dof_rot_; i++) gid_pair(i) = lm_beam[rot_dof_indices[i]];
  for (unsigned int i = 0; i < solid::n_dof_; i++) gid_pair(i + n_dof_rot_) = lm_solid[i];


  // If given, assemble force terms into the global force vector.
  if (force_vector != Teuchos::null)
    force_vector->SumIntoGlobalValues(gid_pair.numRows(), gid_pair.A(), local_force.A());

  // If given, assemble force terms into the global stiffness matrix.
  if (stiffness_matrix != Teuchos::null)
    for (unsigned int i_dof = 0; i_dof < n_dof_pair_; i_dof++)
      for (unsigned int j_dof = 0; j_dof < n_dof_pair_; j_dof++)
        stiffness_matrix->FEAssemble(CORE::FADUTILS::CastToDouble(local_stiff(i_dof, j_dof)),
            gid_pair(i_dof), gid_pair(j_dof));
}

/**
 *
 */
template <typename beam, typename solid>
void BEAMINTERACTION::BeamToSolidVolumeMeshtyingPairGaussPoint<beam,
    solid>::evaluate_rotational_coupling_terms(  //
    const INPAR::BEAMTOSOLID::BeamToSolidRotationCoupling& rot_coupling_type,
    const GEOMETRYPAIR::ElementData<solid, scalar_type_rot_2nd>& q_solid,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<3, double>&
        triad_interpolation_scheme,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<3, double>&
        ref_triad_interpolation_scheme,
    CORE::LINALG::Matrix<n_dof_pair_, 1, double>& local_force,
    CORE::LINALG::Matrix<n_dof_pair_, n_dof_pair_, double>& local_stiff) const
{
  // Initialize variables.
  CORE::LINALG::Matrix<3, 1, double> dr_beam_ref;
  CORE::LINALG::Matrix<4, 1, double> quaternion_beam_double;
  CORE::LINALG::Matrix<3, 1, double> psi_beam_double;
  CORE::LINALG::Matrix<3, 1, scalar_type_rot_1st> psi_beam;
  CORE::LINALG::Matrix<3, 1, scalar_type_rot_2nd> psi_solid;
  CORE::LINALG::Matrix<3, 1, scalar_type_rot_1st> psi_solid_val;
  CORE::LINALG::Matrix<3, 1, scalar_type_rot_1st> psi_rel;
  CORE::LINALG::Matrix<4, 1, scalar_type_rot_1st> quaternion_beam;
  CORE::LINALG::Matrix<4, 1, scalar_type_rot_1st> quaternion_beam_inv;
  CORE::LINALG::Matrix<4, 1, double> quaternion_beam_ref;
  CORE::LINALG::Matrix<4, 1, scalar_type_rot_1st> quaternion_solid;
  CORE::LINALG::Matrix<4, 1, scalar_type_rot_1st> quaternion_rel;
  CORE::LINALG::Matrix<3, 3, double> T_beam;
  CORE::LINALG::Matrix<3, 3, scalar_type_rot_1st> T_solid;
  CORE::LINALG::Matrix<3, 3, scalar_type_rot_1st> T_solid_inv;
  CORE::LINALG::Matrix<3, 1, scalar_type_rot_1st> potential_variation;
  CORE::LINALG::Matrix<n_dof_rot_, 1, scalar_type_rot_1st> fc_beam_gp;
  CORE::LINALG::Matrix<3, solid::n_dof_, scalar_type_rot_1st> d_psi_solid_d_q_solid;
  CORE::LINALG::Matrix<3, 1, scalar_type_rot_1st> Tinv_solid_times_potential_variation;
  CORE::LINALG::Matrix<solid::n_dof_, 1, scalar_type_rot_1st> fc_solid_gp;
  CORE::LINALG::SerialDenseVector L_i(3);
  CORE::LINALG::Matrix<n_dof_rot_, 3, double> d_fc_beam_d_psi_beam;
  CORE::LINALG::Matrix<solid::n_dof_, 3, double> d_fc_solid_d_psi_beam;
  std::vector<CORE::LINALG::Matrix<3, 3, double>> I_beam_tilde;
  CORE::LINALG::Matrix<3, n_dof_rot_, double> I_beam_tilde_full;
  CORE::LINALG::Matrix<3, n_dof_rot_, double> T_beam_times_I_beam_tilde_full;
  CORE::LINALG::Matrix<n_dof_rot_, n_dof_rot_, double> stiff_beam_beam_gp;
  CORE::LINALG::Matrix<solid::n_dof_, n_dof_rot_, double> stiff_solid_beam_gp;
  CORE::LINALG::Matrix<n_dof_rot_, solid::n_dof_, double> stiff_beam_solid_gp;

  // Initialize scalar variables.
  double segment_jacobian = 0.0;
  double beam_segmentation_factor = 0.0;
  double rotational_penalty_parameter = this->Params()
                                            ->beam_to_solid_volume_meshtying_params()
                                            ->get_rotational_coupling_penalty_parameter();

  // Calculate the meshtying forces.
  // Loop over segments.
  const unsigned int n_segments = this->line_to_3D_segments_.size();
  for (unsigned int i_segment = 0; i_segment < n_segments; i_segment++)
  {
    // Factor to account for the integration segment length.
    beam_segmentation_factor = 0.5 * this->line_to_3D_segments_[i_segment].GetSegmentLength();

    // Gauss point loop.
    const unsigned int n_gp = this->line_to_3D_segments_[i_segment].GetProjectionPoints().size();
    for (unsigned int i_gp = 0; i_gp < n_gp; i_gp++)
    {
      // Get the current Gauss point.
      const GEOMETRYPAIR::ProjectionPoint1DTo3D<double>& projected_gauss_point =
          this->line_to_3D_segments_[i_segment].GetProjectionPoints()[i_gp];

      // Get the jacobian in the reference configuration.
      GEOMETRYPAIR::EvaluatePositionDerivative1<beam>(
          projected_gauss_point.GetEta(), this->ele1posref_, dr_beam_ref);

      // Jacobian including the segment length.
      segment_jacobian = dr_beam_ref.Norm2() * beam_segmentation_factor;

      // Calculate the rotation vector of this cross section.
      triad_interpolation_scheme.get_interpolated_quaternion_at_xi(
          quaternion_beam_double, projected_gauss_point.GetEta());
      CORE::LARGEROTATIONS::quaterniontoangle(quaternion_beam_double, psi_beam_double);
      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        psi_beam(i_dim) = CORE::FADUTILS::HigherOrderFadValue<scalar_type_rot_1st>::apply(
            3 + solid::n_dof_, i_dim, psi_beam_double(i_dim));
      CORE::LARGEROTATIONS::angletoquaternion(psi_beam, quaternion_beam);
      quaternion_beam_inv = CORE::LARGEROTATIONS::inversequaternion(quaternion_beam);

      // Get the solid rotation vector.
      ref_triad_interpolation_scheme.get_interpolated_quaternion_at_xi(
          quaternion_beam_ref, projected_gauss_point.GetEta());
      GetSolidRotationVector<solid>(rot_coupling_type, projected_gauss_point.GetXi(),
          this->ele2posref_, q_solid, quaternion_beam_ref, psi_solid);
      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        psi_solid_val(i_dim) = psi_solid(i_dim).val();
      CORE::LARGEROTATIONS::angletoquaternion(psi_solid_val, quaternion_solid);

      // Calculate the relative rotation vector.
      CORE::LARGEROTATIONS::quaternionproduct(
          quaternion_beam_inv, quaternion_solid, quaternion_rel);
      CORE::LARGEROTATIONS::quaterniontoangle(quaternion_rel, psi_rel);

      // Calculate the transformation matrices.
      T_beam = CORE::LARGEROTATIONS::Tmatrix(CORE::FADUTILS::CastToDouble(psi_beam));
      T_solid = CORE::LARGEROTATIONS::Tmatrix(psi_solid_val);

      // Force terms.
      CORE::FE::shape_function_1D(L_i, projected_gauss_point.GetEta(), CORE::FE::CellType::line3);
      potential_variation = psi_rel;
      potential_variation.Scale(rotational_penalty_parameter);
      for (unsigned int i_node = 0; i_node < 3; i_node++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          fc_beam_gp(3 * i_node + i_dim) = -1.0 * L_i(i_node) * potential_variation(i_dim) *
                                           projected_gauss_point.GetGaussWeight() *
                                           segment_jacobian;
      for (unsigned int i_dof = 0; i_dof < n_dof_rot_; i_dof++)
        local_force(i_dof) += CORE::FADUTILS::CastToDouble(fc_beam_gp(i_dof));

      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        for (unsigned int i_solid = 0; i_solid < solid::n_dof_; i_solid++)
          d_psi_solid_d_q_solid(i_dim, i_solid) = psi_solid(i_dim).dx(3 + i_solid);
      T_solid_inv = T_solid;
      CORE::LINALG::Inverse(T_solid_inv);
      Tinv_solid_times_potential_variation.MultiplyTN(T_solid_inv, potential_variation);
      fc_solid_gp.MultiplyTN(d_psi_solid_d_q_solid, Tinv_solid_times_potential_variation);
      fc_solid_gp.Scale(projected_gauss_point.GetGaussWeight() * segment_jacobian);
      for (unsigned int i_dof = 0; i_dof < solid::n_dof_; i_dof++)
        local_force(n_dof_rot_ + i_dof) += CORE::FADUTILS::CastToDouble(fc_solid_gp(i_dof));


      // Stiffness terms.
      for (unsigned int i_beam_dof = 0; i_beam_dof < n_dof_rot_; i_beam_dof++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          d_fc_beam_d_psi_beam(i_beam_dof, i_dim) = fc_beam_gp(i_beam_dof).dx(i_dim);
      for (unsigned int i_solid_dof = 0; i_solid_dof < solid::n_dof_; i_solid_dof++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          d_fc_solid_d_psi_beam(i_solid_dof, i_dim) = fc_solid_gp(i_solid_dof).dx(i_dim);

      triad_interpolation_scheme.get_nodal_generalized_rotation_interpolation_matrices_at_xi(
          I_beam_tilde, projected_gauss_point.GetEta());
      for (unsigned int i_node = 0; i_node < 3; i_node++)
        for (unsigned int i_dim_0 = 0; i_dim_0 < 3; i_dim_0++)
          for (unsigned int i_dim_1 = 0; i_dim_1 < 3; i_dim_1++)
            I_beam_tilde_full(i_dim_0, i_node * 3 + i_dim_1) =
                I_beam_tilde[i_node](i_dim_0, i_dim_1);

      T_beam_times_I_beam_tilde_full.Multiply(
          CORE::FADUTILS::CastToDouble(T_beam), I_beam_tilde_full);
      stiff_beam_beam_gp.Multiply(d_fc_beam_d_psi_beam, T_beam_times_I_beam_tilde_full);
      for (unsigned int i_dof = 0; i_dof < n_dof_rot_; i_dof++)
        for (unsigned int j_dof = 0; j_dof < n_dof_rot_; j_dof++)
          local_stiff(i_dof, j_dof) += stiff_beam_beam_gp(i_dof, j_dof);

      stiff_solid_beam_gp.Multiply(d_fc_solid_d_psi_beam, T_beam_times_I_beam_tilde_full);
      for (unsigned int i_dof = 0; i_dof < solid::n_dof_; i_dof++)
        for (unsigned int j_dof = 0; j_dof < n_dof_rot_; j_dof++)
          local_stiff(i_dof + n_dof_rot_, j_dof) += stiff_solid_beam_gp(i_dof, j_dof);

      for (unsigned int i_dof = 0; i_dof < n_dof_rot_; i_dof++)
        for (unsigned int j_dof = 0; j_dof < solid::n_dof_; j_dof++)
          local_stiff(i_dof, j_dof + n_dof_rot_) += fc_beam_gp(i_dof).dx(3 + j_dof);

      for (unsigned int i_dof = 0; i_dof < solid::n_dof_; i_dof++)
        for (unsigned int j_dof = 0; j_dof < solid::n_dof_; j_dof++)
          local_stiff(i_dof + n_dof_rot_, j_dof + n_dof_rot_) += fc_solid_gp(i_dof).dx(3 + j_dof);
    }
  }
}


/**
 * Explicit template initialization of template class.
 */
namespace BEAMINTERACTION
{
  using namespace GEOMETRYPAIR;

  template class BeamToSolidVolumeMeshtyingPairGaussPoint<t_hermite, t_hex8>;
  template class BeamToSolidVolumeMeshtyingPairGaussPoint<t_hermite, t_hex20>;
  template class BeamToSolidVolumeMeshtyingPairGaussPoint<t_hermite, t_hex27>;
  template class BeamToSolidVolumeMeshtyingPairGaussPoint<t_hermite, t_tet4>;
  template class BeamToSolidVolumeMeshtyingPairGaussPoint<t_hermite, t_tet10>;
  template class BeamToSolidVolumeMeshtyingPairGaussPoint<t_hermite, t_nurbs27>;
}  // namespace BEAMINTERACTION

FOUR_C_NAMESPACE_CLOSE