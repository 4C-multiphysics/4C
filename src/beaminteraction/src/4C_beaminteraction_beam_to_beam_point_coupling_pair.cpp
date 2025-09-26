// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_beam_point_coupling_pair.hpp"

#include "4C_beam3_reissner.hpp"
#include "4C_beam3_triad_interpolation_local_rotation_vectors.hpp"
#include "4C_beaminteraction_beam_to_solid_utils.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_geometry_pair_access_traits.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparsematrix.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
template <typename Beam>
BeamInteraction::BeamToBeamPointCouplingPair<Beam>::BeamToBeamPointCouplingPair(
    double penalty_parameter_rot, double penalty_parameter_pos,
    std::array<double, 2> pos_in_parameterspace)
    : BeamContactPair(),
      penalty_parameter_pos_(penalty_parameter_pos),
      penalty_parameter_rot_(penalty_parameter_rot),
      position_in_parameterspace_(pos_in_parameterspace)
{
  // Empty constructor.
}

/**
 *
 */
template <typename Beam>
void BeamInteraction::BeamToBeamPointCouplingPair<Beam>::setup()
{
  // This pair only works for Simo Reissner beam elements.
  const auto check_simo_reissner_beam = [](auto element)
  {
    const bool is_sr_beam = dynamic_cast<const Discret::Elements::Beam3r*>(element) != nullptr;
    if (!is_sr_beam)
      FOUR_C_THROW("The BeamToBeamPointCouplingPair only works for Simo Reissner beams");
  };
  check_simo_reissner_beam(this->element1());
  check_simo_reissner_beam(this->element2());

  this->issetup_ = true;
}

/**
 *
 */
template <typename Beam>
void BeamInteraction::BeamToBeamPointCouplingPair<Beam>::evaluate_and_assemble(
    const std::shared_ptr<const Core::FE::Discretization>& discret,
    const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
    const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector)
{
  const std::array<const Core::Elements::Element*, 2> beam_ele = {
      this->element1(), this->element2()};

  // Initialize pair values that we will fill while evaluating the cross-section kinematics.
  std::array<int, 2 * (Beam::n_dof_ + n_dof_rot_)> pair_gid{-1};
  Core::LinAlg::Matrix<2 * (Beam::n_dof_ + n_dof_rot_), 12> left_transformation_matrix(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<12, 2 * (Beam::n_dof_ + n_dof_rot_)> right_transformation_matrix(
      Core::LinAlg::Initialization::zero);

  // Initialize variables for evaluation of the positions.
  std::array<Core::LinAlg::Matrix<3, 1>, 2> r_ref;
  std::array<Core::LinAlg::Matrix<3, 1>, 2> r;

  // Evaluate positional kinematics
  {
    for (unsigned int i_beam = 0; i_beam < 2; i_beam++)
    {
      // Get shape function data
      GeometryPair::ElementData<Beam, double> beam_data =
          GeometryPair::InitializeElementData<Beam, double>::initialize(beam_ele[i_beam]);

      // Get GIDs of the beams positional DOF.
      std::vector<int> lm_beam, lm_solid, lmowner, lmstride;
      beam_ele[i_beam]->location_vector(*discret, lm_beam, lmowner, lmstride);
      const std::array<int, 12> pos_dof_indices = {0, 1, 2, 6, 7, 8, 9, 10, 11, 15, 16, 17};
      for (unsigned int i_dof = 0; i_dof < Beam::n_dof_; i_dof++)
        pair_gid[i_dof + i_beam * (Beam::n_dof_ + n_dof_rot_)] = lm_beam[pos_dof_indices[i_dof]];

      // Set current nodal positions (and tangents) for beam element
      std::vector<double> element_posdofvec_values(Beam::n_dof_, 0.0);
      BeamInteraction::Utils::extract_pos_dof_vec_values(
          *discret, beam_ele[i_beam], *displacement_vector, element_posdofvec_values);
      std::vector<double> element_posdofvec_absolutevalues(Beam::n_dof_, 0.0);
      BeamInteraction::Utils::extract_pos_dof_vec_absolute_values(
          *discret, beam_ele[i_beam], *displacement_vector, element_posdofvec_absolutevalues);

      // Evaluate the current position of the coupling point.
      for (unsigned int i_dof = 0; i_dof < Beam::n_dof_; i_dof++)
        beam_data.element_position_(i_dof) = element_posdofvec_absolutevalues[i_dof];
      GeometryPair::evaluate_position<Beam>(
          position_in_parameterspace_[i_beam], beam_data, r[i_beam]);

      // Evaluate the reference position of the coupling point.
      for (unsigned int i_dof = 0; i_dof < Beam::n_dof_; i_dof++)
        beam_data.element_position_(i_dof) -= element_posdofvec_values[i_dof];
      GeometryPair::evaluate_position<Beam>(
          position_in_parameterspace_[i_beam], beam_data, r_ref[i_beam]);

      // Shape function matrices
      Core::LinAlg::Matrix<3, Beam::n_dof_> H_full;
      GeometryPair::evaluate_shape_function_matrix<Beam>(
          H_full, position_in_parameterspace_[i_beam], beam_data.shape_function_data_);
      for (unsigned int i_dof = 0; i_dof < Beam::n_dof_; i_dof++)
      {
        for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
        {
          left_transformation_matrix(i_dof + i_beam * (Beam::n_dof_ + n_dof_rot_),
              i_dir + i_beam * 6) = H_full(i_dir, i_dof);
          right_transformation_matrix(i_dir + i_beam * 6,
              i_dof + i_beam * (Beam::n_dof_ + n_dof_rot_)) = H_full(i_dir, i_dof);
        }
      }
    }
  }

  // Initialize variables for evaluation of the rotations.
  std::array<Core::LinAlg::Matrix<4, 1, scalar_type_rot>, 2> cross_section_quaternion;
  std::array<Core::LinAlg::Matrix<4, 1, double>, 2> cross_section_quaternion_ref;

  // Evaluate rotational kinematics
  {
    for (unsigned int i_beam = 0; i_beam < 2; i_beam++)
    {
      // Get GIDs of the beams rotational DOF.
      const auto rot_gid = Utils::get_element_rot_gid_indices(*discret, beam_ele[i_beam]);
      for (unsigned int i_dof = 0; i_dof < n_dof_rot_; i_dof++)
        pair_gid[i_dof + Beam::n_dof_ + i_beam * (Beam::n_dof_ + n_dof_rot_)] = rot_gid[i_dof];

      // Get the triad interpolation schemes for the two beams.
      LargeRotations::TriadInterpolationLocalRotationVectors<3, double> triad_interpolation_scheme;
      LargeRotations::TriadInterpolationLocalRotationVectors<3, double>
          ref_triad_interpolation_scheme;
      BeamInteraction::get_beam_triad_interpolation_scheme(*discret, *displacement_vector,
          beam_ele[i_beam], triad_interpolation_scheme, ref_triad_interpolation_scheme);

      // Calculate the rotation vector of the beam cross section and its FAD representation.
      Core::LinAlg::Matrix<4, 1, double> quaternion_double;
      Core::LinAlg::Matrix<3, 1, double> psi_double;
      Core::LinAlg::Matrix<3, 1, scalar_type_rot> psi;
      triad_interpolation_scheme.get_interpolated_quaternion_at_xi(
          quaternion_double, position_in_parameterspace_[i_beam]);
      Core::LargeRotations::quaterniontoangle(quaternion_double, psi_double);
      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        psi(i_dim) = Core::FADUtils::HigherOrderFadValue<scalar_type_rot>::apply(
            6, i_beam * 3 + i_dim, psi_double(i_dim));
      Core::LargeRotations::angletoquaternion(psi, cross_section_quaternion[i_beam]);
      ref_triad_interpolation_scheme.get_interpolated_quaternion_at_xi(
          cross_section_quaternion_ref[i_beam], position_in_parameterspace_[i_beam]);

      // Linearization interpolation matrices
      std::vector<Core::LinAlg::Matrix<3, 3, double>> I_tilde;
      Core::LinAlg::Matrix<3, n_dof_rot_, double> I_tilde_full;
      triad_interpolation_scheme.get_nodal_generalized_rotation_interpolation_matrices_at_xi(
          I_tilde, position_in_parameterspace_[i_beam]);
      for (unsigned int i_node = 0; i_node < 3; i_node++)
        for (unsigned int i_dim_0 = 0; i_dim_0 < 3; i_dim_0++)
          for (unsigned int i_dim_1 = 0; i_dim_1 < 3; i_dim_1++)
            I_tilde_full(i_dim_0, i_node * 3 + i_dim_1) = I_tilde[i_node](i_dim_0, i_dim_1);

      // Spin shape function matrices
      auto L_beam = Core::LinAlg::SerialDenseVector(3);
      Core::FE::shape_function_1d(
          L_beam, position_in_parameterspace_[i_beam], Core::FE::CellType::line3);
      Core::LinAlg::Matrix<3, n_dof_rot_, double> L_beam_full{Core::LinAlg::Initialization::zero};
      for (unsigned int i_node_rot = 0; i_node_rot < 3; i_node_rot++)
      {
        for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
        {
          L_beam_full(i_dir, i_dir + 3 * i_node_rot) = L_beam(i_node_rot);
        }
      }
      for (unsigned int i_dof = 0; i_dof < n_dof_rot_; i_dof++)
      {
        for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
        {
          left_transformation_matrix(i_dof + Beam::n_dof_ + i_beam * (Beam::n_dof_ + n_dof_rot_),
              i_dir + 3 + i_beam * 6) = L_beam_full(i_dir, i_dof);
          right_transformation_matrix(
              i_dir + 3 + i_beam * 6, i_dof + Beam::n_dof_ + i_beam * (Beam::n_dof_ + n_dof_rot_)) =
              I_tilde_full(i_dir, i_dof);
        }
      }
    }
  }

  // Positional coupling terms
  const auto [constraint_position, constraint_position_lin_kinematic, residuum_position_lin_lambda,
      evaluation_data_position] = evaluate_positional_coupling(r_ref, r,
      cross_section_quaternion_ref, cross_section_quaternion);

  // Rotational coupling terms
  const auto [constraint_rotation, constraint_rotation_lin_kinematic, residuum_rotation_lin_lambda,
      evaluation_data_rotation] =
      evaluate_rotational_coupling(cross_section_quaternion_ref, cross_section_quaternion);

  // Coupling residuum and stiffness
  Core::LinAlg::Matrix<12, 1> residuum(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<12, 12> stiffness(Core::LinAlg::Initialization::zero);

  // Penalty regularization positions
  Core::LinAlg::Matrix<3, 1> lambda_position = constraint_position;
  lambda_position.scale(penalty_parameter_pos_);
  Core::LinAlg::Matrix<12, 1> residuum_position;
  residuum_position.multiply_nn(residuum_position_lin_lambda, lambda_position);
  residuum += residuum_position;

  Core::LinAlg::Matrix<12, 12> stiffness_position;
  stiffness_position.multiply_nn(residuum_position_lin_lambda, constraint_position_lin_kinematic);
  stiffness_position.scale(penalty_parameter_pos_);
  for (unsigned int i_beam = 0; i_beam < 2; i_beam++)
  {
    Core::LinAlg::Matrix<3, 3, double> skew_lambda;
    Core::LargeRotations::computespin(skew_lambda, lambda_position);
    Core::LinAlg::Matrix<3, 3> temp_matrix;
    temp_matrix.multiply_nn(skew_lambda, evaluation_data_position[i_beam]);
    for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
    {
      for (unsigned int j_dir = 0; j_dir < 3; j_dir++)
      {
        stiffness_position(i_dir + 3 + 6 * i_beam, j_dir + 3 + 6 * i_beam) -=
            temp_matrix(i_dir, j_dir);
      }
    }
  }
  stiffness += stiffness_position;

  // Penalty regularization rotations
  Core::LinAlg::Matrix<3, 1> lambda_rotation = constraint_rotation;
  lambda_rotation.scale(penalty_parameter_rot_);
  Core::LinAlg::Matrix<12, 1> residuum_rotation;
  residuum_rotation.multiply_nn(residuum_rotation_lin_lambda, lambda_rotation);
  residuum += residuum_rotation;

  Core::LinAlg::Matrix<12, 12> stiffness_rot;
  stiffness_rot.multiply_nn(residuum_rotation_lin_lambda, constraint_rotation_lin_kinematic);
  stiffness_rot.scale(penalty_parameter_rot_);
  for (unsigned int i_beam = 0; i_beam < 2; i_beam++)
  {
    for (unsigned int i = 0; i < 3; i++)
    {
      for (unsigned int j = 0; j < 3; j++)
      {
        for (unsigned int l = 0; l < 3; l++)
        {
          stiffness_rot(i + 3, l + 3 + 6 * i_beam) -=
              evaluation_data_rotation[i_beam][i][j][l] * lambda_rotation(j);
          stiffness_rot(i + 9, l + 3 + 6 * i_beam) +=
              evaluation_data_rotation[i_beam][i][j][l] * lambda_rotation(j);
        }
      }
    }
  }
  stiffness += stiffness_rot;

  // Map residuum and stiffness to element DOFs
  Core::LinAlg::Matrix<2 * (Beam::n_dof_ + n_dof_rot_), 1> residuum_pair{
      Core::LinAlg::Initialization::zero};
  Core::LinAlg::Matrix<2 * (Beam::n_dof_ + n_dof_rot_), 2 * (Beam::n_dof_ + n_dof_rot_)>
      stiffness_pair{Core::LinAlg::Initialization::zero};
  residuum_pair.multiply(left_transformation_matrix, residuum);
  Core::LinAlg::Matrix<2 * (Beam::n_dof_ + n_dof_rot_), 12> temp_matrix;
  temp_matrix.multiply(left_transformation_matrix, stiffness);
  stiffness_pair.multiply(temp_matrix, right_transformation_matrix);

  // Add the coupling terms into the global vector ana matrix.
  if (force_vector != nullptr)
    force_vector->sum_into_global_values(pair_gid.size(), pair_gid.data(), residuum_pair.data());
  if (stiffness_matrix != nullptr)
  {
    for (unsigned int i_dof = 0; i_dof < pair_gid.size(); i_dof++)
    {
      for (unsigned int j_dof = 0; j_dof < pair_gid.size(); j_dof++)
      {
        if (pair_gid[i_dof] == -1 or pair_gid[j_dof] == -1) continue;
        stiffness_matrix->fe_assemble(
            stiffness_pair(i_dof, j_dof), pair_gid[i_dof], pair_gid[j_dof]);
      }
    }
  }
}

/**
 *
 */
template <typename Beam>
std::tuple<Core::LinAlg::Matrix<3, 1>, Core::LinAlg::Matrix<3, 12>, Core::LinAlg::Matrix<12, 3>,
    std::array<Core::LinAlg::Matrix<3, 3, double>, 2>>
BeamInteraction::BeamToBeamPointCouplingPair<Beam>::evaluate_positional_coupling(
    const std::array<Core::LinAlg::Matrix<3, 1>, 2>& r_ref,
    const std::array<Core::LinAlg::Matrix<3, 1>, 2>& r,
    const std::array<Core::LinAlg::Matrix<4, 1, double>, 2>& cross_section_quaternion_ref,
    const std::array<Core::LinAlg::Matrix<4, 1, scalar_type_rot>, 2>& cross_section_quaternion)
{
  // Coupling vectors and matrices
  Core::LinAlg::Matrix<3, 1> constraint_position(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 12> constraint_position_lin_kinematic(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<12, 3> residuum_position_lin_lambda(Core::LinAlg::Initialization::zero);
  std::array<Core::LinAlg::Matrix<3, 3, double>, 2> evaluation_data_position;

  // Note: If the two points are not at the same position in the reference configuration, then even
  // for the positional coupling terms there will be contributions to the rotational equilibrium,
  // i.e., moments. To keep things simple, we first evaluate the terms that will affect the
  // positional DOFs and then we compute the terms that will affect the rotational DOFs.

  // Constraint (pure positional terms)
  constraint_position = r[1];
  constraint_position -= r[0];

  // Linearizations (pure positional terms)
  for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
  {
    constraint_position_lin_kinematic(i_dir, i_dir) = -1.0;
    constraint_position_lin_kinematic(i_dir, 6 + i_dir) = 1.0;

    residuum_position_lin_lambda(i_dir, i_dir) = -1.0;
    residuum_position_lin_lambda(6 + i_dir, i_dir) = 1.0;
  }

  // Terms arising due to non matching points in the reference configurtaion.
  {
    // Evaluate reference offset in the material and spatial frames.
    Core::LinAlg::Matrix<3, 1, double> director_21_ref_spatial = r_ref[1];
    director_21_ref_spatial -= r_ref[0];
    std::array<Core::LinAlg::Matrix<3, 1, double>, 2> director_21_ref_material;
    for (unsigned int i_beam = 0; i_beam < 2; i_beam++)
    {
      Core::LinAlg::Matrix<3, 3, double> rotation_matrix;
      Core::LargeRotations::quaterniontotriad(
          cross_section_quaternion_ref[i_beam], rotation_matrix);
      director_21_ref_material[i_beam].multiply_tn(rotation_matrix, director_21_ref_spatial);
    }

    // Add to constraint vector.
    // We add a half of the reference offset vector in each frame of the two beams. By doing it this
    // way, we obtain a coupling formulation invariant to the ordering of the cross-sections in this
    // pair.
    std::array<Core::LinAlg::Matrix<3, 1, double>, 2> director_21_spatial_half;
    for (unsigned int i_beam = 0; i_beam < 2; i_beam++)
    {
      Core::LinAlg::Matrix<3, 3, double> rotation_matrix;
      Core::LargeRotations::quaterniontotriad(
          Core::FADUtils::cast_to_double(cross_section_quaternion[i_beam]), rotation_matrix);
      director_21_spatial_half[i_beam].multiply_nn(
          rotation_matrix, director_21_ref_material[i_beam]);
      director_21_spatial_half[i_beam].scale(0.5);
      constraint_position -= director_21_spatial_half[i_beam];
    }

    // Add the linearizations of the previous constraint terms.
    for (unsigned int i_beam = 0; i_beam < 2; i_beam++)
    {
      Core::LinAlg::Matrix<3, 3, double> skew_director_21_spatial_half;
      Core::LargeRotations::computespin(
          skew_director_21_spatial_half, director_21_spatial_half[i_beam]);
      evaluation_data_position[i_beam] = skew_director_21_spatial_half;

      for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
      {
        for (unsigned int j_dir = 0; j_dir < 3; j_dir++)
        {
          constraint_position_lin_kinematic(i_dir, j_dir + 3 + 6 * i_beam) +=
              skew_director_21_spatial_half(i_dir, j_dir);

          residuum_position_lin_lambda(j_dir + 3 + 6 * i_beam, i_dir) -=
              skew_director_21_spatial_half(j_dir, i_dir);
        }
      }
    }
  }

  return {constraint_position, constraint_position_lin_kinematic, residuum_position_lin_lambda,
      evaluation_data_position};
}

/**
 *
 */
template <typename Beam>
std::tuple<Core::LinAlg::Matrix<3, 1>, Core::LinAlg::Matrix<3, 12>, Core::LinAlg::Matrix<12, 3>,
    std::array<std::array<std::array<std::array<double, 3>, 3>, 3>, 2>>
BeamInteraction::BeamToBeamPointCouplingPair<Beam>::evaluate_rotational_coupling(
    const std::array<Core::LinAlg::Matrix<4, 1, double>, 2>& cross_section_quaternion_ref,
    const std::array<Core::LinAlg::Matrix<4, 1, scalar_type_rot>, 2>& cross_section_quaternion)
{
  // Coupling vectors and matrices
  Core::LinAlg::Matrix<3, 1> constraint_rotation(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 12> constraint_rotation_lin_kinematic(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<12, 3> residuum_rotation_lin_lambda(Core::LinAlg::Initialization::zero);
  std::array<std::array<std::array<std::array<double, 3>, 3>, 3>, 2> evaluation_data_rotation{};

  // Get the relative rotation vector between the two cross sections.
  Core::LinAlg::Matrix<4, 1, scalar_type_rot> temp_quaternion_1, temp_quaternion_2, quaternion_rel;
  Core::LinAlg::Matrix<3, 1, scalar_type_rot> psi_rel;
  Core::LinAlg::Matrix<4, 1, scalar_type_rot> quaternion_0_inv =
      Core::LargeRotations::inversequaternion(cross_section_quaternion[0]);
  Core::LinAlg::Matrix<4, 1, double> quaternion_1_ref_inv =
      Core::LargeRotations::inversequaternion(cross_section_quaternion_ref[1]);
  Core::LargeRotations::quaternionproduct(
      quaternion_0_inv, cross_section_quaternion_ref[0], temp_quaternion_1);
  Core::LargeRotations::quaternionproduct(
      temp_quaternion_1, quaternion_1_ref_inv, temp_quaternion_2);
  Core::LargeRotations::quaternionproduct(
      temp_quaternion_2, cross_section_quaternion[1], quaternion_rel);
  Core::LargeRotations::quaterniontoangle(quaternion_rel, psi_rel);

  // Transformation matrix
  const Core::LinAlg::Matrix<3, 3, scalar_type_rot> T_psi_rel =
      Core::LargeRotations::tmatrix(psi_rel);

  constraint_rotation = Core::FADUtils::cast_to_double(psi_rel);
  for (unsigned int i_beam = 0; i_beam < 2; i_beam++)
  {
    const double load_factor = i_beam == 0 ? -1.0 : 1.0;

    // Get the rotation angle
    Core::LinAlg::Matrix<3, 1, double> psi_beam;
    Core::LargeRotations::quaterniontoangle(
        Core::FADUtils::cast_to_double(cross_section_quaternion[i_beam]), psi_beam);

    const Core::LinAlg::Matrix<3, 3, double> T_psi_beam = Core::LargeRotations::tmatrix(psi_beam);
    Core::LinAlg::Matrix<3, 3, double> d_psi_rel_d_psi_beam;
    for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
    {
      for (unsigned int j_dim = 0; j_dim < 3; j_dim++)
      {
        d_psi_rel_d_psi_beam(i_dim, j_dim) = psi_rel(i_dim).dx(j_dim + i_beam * 3);
      }
    }

    Core::LinAlg::Matrix<3, 3, double> constraint_lin_psi_beam;
    constraint_lin_psi_beam.multiply_nn(d_psi_rel_d_psi_beam, T_psi_beam);
    for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
    {
      for (unsigned int j_dim = 0; j_dim < 3; j_dim++)
      {
        constraint_rotation_lin_kinematic(i_dim, j_dim + 3 + i_beam * 6) =
            constraint_lin_psi_beam(i_dim, j_dim);
        // Note: Here we insert the transposed, thus the switched ordering in the target but not
        // the source.
        residuum_rotation_lin_lambda(j_dim + 3 + i_beam * 6, i_dim) =
            load_factor * Core::FADUtils::cast_to_double(T_psi_rel(i_dim, j_dim));
      }
    }
    for (unsigned int i = 0; i < 3; i++)
    {
      for (unsigned int j = 0; j < 3; j++)
      {
        for (unsigned int l = 0; l < 3; l++)
        {
          for (unsigned int k = 0; k < 3; k++)
          {
            evaluation_data_rotation[i_beam][i][j][l] +=
                T_psi_rel(j, i).dx(k + 3 * i_beam) * T_psi_beam(k, l);
          }
        }
      }
    }
  }

  return {constraint_rotation, constraint_rotation_lin_kinematic, residuum_rotation_lin_lambda,
      evaluation_data_rotation};
}

/**
 *
 */
template <typename Beam>
void BeamInteraction::BeamToBeamPointCouplingPair<Beam>::print(std::ostream& out) const
{
  check_init_setup();

  // Print some general information: Element IDs and dofvecs.
  out << "\n------------------------------------------------------------------------";
  out << "\nInstance of BeamToBeamPenaltyPointCouplingPair"
      << "\nBeam1 EleGID:  " << element1()->id() << "\nBeam2 EleGID: " << element2()->id();
  out << "------------------------------------------------------------------------\n";
}

/**
 *
 */
template <typename Beam>
void BeamInteraction::BeamToBeamPointCouplingPair<
    Beam>::print_summary_one_line_per_active_segment_pair(std::ostream& out) const
{
  check_init_setup();

  out << "Beam-to-beam point coupling pair, beam1 gid: " << element1()->id()
      << " beam2 gid: " << element2()->id() << ", position in parameter space: ["
      << position_in_parameterspace_[0] << ", " << position_in_parameterspace_[1] << "]\n";
}

/**
 * Explicit template initialization of template class.
 */
namespace BeamInteraction
{
  using namespace GeometryPair;

  template class BeamToBeamPointCouplingPair<t_hermite>;
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE
