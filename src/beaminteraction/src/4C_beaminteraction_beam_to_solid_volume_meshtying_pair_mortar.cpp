// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_mortar.hpp"

#include "4C_beaminteraction_beam_to_solid_mortar_manager.hpp"
#include "4C_beaminteraction_beam_to_solid_utils.hpp"
#include "4C_beaminteraction_beam_to_solid_visualization_output_writer_base.hpp"
#include "4C_beaminteraction_beam_to_solid_visualization_output_writer_visualization.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_visualization_output_params.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_line_to_volume.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"

#include <unordered_set>

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
template <typename Beam, typename Solid, typename Mortar>
BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<Beam, Solid,
    Mortar>::BeamToSolidVolumeMeshtyingPairMortar()
    : base_class(), n_mortar_rot_(0)
{
  // Empty constructor.
}

/**
 *
 */
template <typename Beam, typename Solid, typename Mortar>
void BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<Beam, Solid,
    Mortar>::evaluate_and_assemble_mortar_contributions(const Core::FE::Discretization& discret,
    const BeamToSolidMortarManager* mortar_manager,
    Core::LinAlg::SparseMatrix& global_constraint_lin_beam,
    Core::LinAlg::SparseMatrix& global_constraint_lin_solid,
    Core::LinAlg::SparseMatrix& global_force_beam_lin_lambda,
    Core::LinAlg::SparseMatrix& global_force_solid_lin_lambda,
    Core::LinAlg::FEVector<double>& global_constraint, Core::LinAlg::FEVector<double>& global_kappa,
    Core::LinAlg::SparseMatrix& global_kappa_lin_beam,
    Core::LinAlg::SparseMatrix& global_kappa_lin_solid,
    Core::LinAlg::FEVector<double>& global_lambda_active,
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector)
{
  // Call Evaluate on the geometry Pair. Only do this once for meshtying.
  if (!this->meshtying_is_evaluated_)
  {
    GeometryPair::ElementData<Beam, double> beam_coupling_ref;
    GeometryPair::ElementData<Solid, double> solid_coupling_ref;
    this->get_coupling_reference_position(beam_coupling_ref, solid_coupling_ref);
    this->cast_geometry_pair()->evaluate(
        beam_coupling_ref, solid_coupling_ref, this->line_to_3D_segments_);
    this->meshtying_is_evaluated_ = true;
  }

  // If there are no intersection segments, return no contact status.
  if (this->line_to_3D_segments_.size() == 0) return;

  // Initialize variables for local mortar matrices.
  Core::LinAlg::Matrix<Mortar::n_dof_, Beam::n_dof_, double> local_D(
      Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<Mortar::n_dof_, Solid::n_dof_, double> local_M(
      Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<Mortar::n_dof_, 1, double> local_kappa(
      Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<Mortar::n_dof_, 1, double> local_constraint(
      Core::LinAlg::Initialization::uninitialized);

  // Evaluate the local mortar contributions.
  evaluate_dm(local_D, local_M, local_kappa, local_constraint);

  // Assemble into global matrices.
  assemble_local_mortar_contributions<Beam, Solid, Mortar>(this, discret, mortar_manager,
      global_constraint_lin_beam, global_constraint_lin_solid, global_force_beam_lin_lambda,
      global_force_solid_lin_lambda, global_constraint, global_kappa, global_lambda_active, local_D,
      local_M, local_kappa, local_constraint, n_mortar_rot_);
}

/**
 *
 */
template <typename Beam, typename Solid, typename Mortar>
void BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<Beam, Solid,
    Mortar>::get_pair_visualization(std::shared_ptr<BeamToSolidVisualizationOutputWriterBase>
                                        visualization_writer,
    Teuchos::ParameterList& visualization_params) const
{
  // Get visualization of base method.
  base_class::get_pair_visualization(visualization_writer, visualization_params);

  std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization> visualization_discret =
      visualization_writer->get_visualization_writer("btsv-mortar");
  std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization> visualization_continuous =
      visualization_writer->get_visualization_writer("btsv-mortar-continuous");
  if (!visualization_discret and visualization_continuous == nullptr) return;

  const std::shared_ptr<const BeamToSolidVolumeMeshtyingVisualizationOutputParams>&
      output_params_ptr =
          visualization_params
              .get<std::shared_ptr<const BeamToSolidVolumeMeshtyingVisualizationOutputParams>>(
                  "btsv-output_params_ptr");
  const bool write_unique_ids = output_params_ptr->get_write_unique_ids_flag();

  if (visualization_discret != nullptr || visualization_continuous != nullptr)
  {
    // Setup variables.
    GeometryPair::ElementData<Mortar, double> element_data_lambda;
    Core::LinAlg::Matrix<3, 1, scalar_type> X;
    Core::LinAlg::Matrix<3, 1, scalar_type> r;
    Core::LinAlg::Matrix<3, 1, scalar_type> u;
    Core::LinAlg::Matrix<3, 1, double> lambda_discret;
    Core::LinAlg::Matrix<3, 1, double> xi_mortar_node;

    // Get the mortar manager and the global lambda vector, those objects will be used to get the
    // discrete Lagrange multiplier values for this pair.
    std::shared_ptr<const BeamInteraction::BeamToSolidMortarManager> mortar_manager =
        visualization_params.get<std::shared_ptr<const BeamInteraction::BeamToSolidMortarManager>>(
            "mortar_manager");
    std::shared_ptr<Core::LinAlg::Vector<double>> lambda =
        visualization_params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("lambda");

    // Get the lambda GIDs of this pair.
    const auto& [lambda_row_pos, _] = mortar_manager->location_vector(*this);

    std::vector<double> lambda_pair = Core::FE::extract_values(*lambda, lambda_row_pos);
    for (unsigned int i_dof = 0; i_dof < Mortar::n_dof_; i_dof++)
      element_data_lambda.element_position_(i_dof) = lambda_pair[i_dof];

    // Add the discrete values of the Lagrange multipliers.
    if (visualization_discret != nullptr)
    {
      // Check if data for this beam was already written.
      std::shared_ptr<std::unordered_set<int>> beam_tracker =
          visualization_params.get<std::shared_ptr<std::unordered_set<int>>>("beam_tracker");

      auto it = beam_tracker->find(this->element1()->id());
      if (it == beam_tracker->end())
      {
        // Only do something if this beam element did not write any output yet.

        // Add this element Id to the tracker.
        beam_tracker->insert(this->element1()->id());

        // Get the visualization vectors.
        auto& visualization_data = visualization_discret->get_visualization_data();
        std::vector<double>& point_coordinates = visualization_data.get_point_coordinates();
        std::vector<double>& displacement =
            visualization_data.get_point_data<double>("displacement");
        std::vector<double>& lambda_vis = visualization_data.get_point_data<double>("lambda");

        std::vector<int>* pair_beam_id = nullptr;
        std::vector<int>* pair_solid_id = nullptr;
        if (write_unique_ids)
        {
          pair_beam_id = &(visualization_data.get_point_data<int>("uid_0_pair_beam_id"));
          pair_solid_id = &(visualization_data.get_point_data<int>("uid_1_pair_solid_id"));
        }

        for (unsigned int i_node = 0; i_node < Mortar::n_nodes_; i_node++)
        {
          // Get the local coordinate of this node.
          xi_mortar_node = Core::FE::get_node_coordinates(i_node, Mortar::discretization_);

          // Get position and displacement of the mortar node.
          GeometryPair::evaluate_position<Beam>(xi_mortar_node(0), this->ele1pos_, r);
          GeometryPair::evaluate_position<Beam>(xi_mortar_node(0), this->ele1posref_, X);
          u = r;
          u -= X;

          // Get the discrete Lagrangian multiplier.
          GeometryPair::evaluate_position<Mortar>(
              xi_mortar_node(0), element_data_lambda, lambda_discret);

          // Add to output data.
          for (unsigned int dim = 0; dim < 3; dim++)
          {
            point_coordinates.push_back(Core::FADUtils::cast_to_double(X(dim)));
            displacement.push_back(Core::FADUtils::cast_to_double(u(dim)));
            lambda_vis.push_back(Core::FADUtils::cast_to_double(lambda_discret(dim)));
          }

          if (write_unique_ids)
          {
            pair_beam_id->push_back(this->element1()->id());
            pair_solid_id->push_back(this->element2()->id());
          }
        }
      }
    }


    // Add the continuous values for the Lagrange multipliers.
    if (visualization_continuous != nullptr and this->line_to_3D_segments_.size() > 0)
    {
      const unsigned int mortar_segments =
          visualization_params
              .get<std::shared_ptr<const BeamToSolidVolumeMeshtyingVisualizationOutputParams>>(
                  "btsv-output_params_ptr")
              ->get_mortar_lambda_continuous_segments();
      double xi;
      auto& visualization_data = visualization_continuous->get_visualization_data();
      std::vector<double>& point_coordinates = visualization_data.get_point_coordinates(
          (mortar_segments + 1) * 3 * this->line_to_3D_segments_.size());
      std::vector<double>& displacement = visualization_data.get_point_data<double>(
          "displacement", (mortar_segments + 1) * 3 * this->line_to_3D_segments_.size());
      std::vector<double>& lambda_vis = visualization_data.get_point_data<double>(
          "lambda", (mortar_segments + 1) * 3 * this->line_to_3D_segments_.size());
      std::vector<uint8_t>& cell_types = visualization_data.get_cell_types();
      std::vector<int32_t>& cell_offsets = visualization_data.get_cell_offsets();

      std::vector<int>* pair_point_beam_id = nullptr;
      std::vector<int>* pair_point_solid_id = nullptr;
      std::vector<int>* pair_cell_beam_id = nullptr;
      std::vector<int>* pair_cell_solid_id = nullptr;
      if (write_unique_ids)
      {
        pair_point_beam_id = &(visualization_data.get_point_data<int>("uid_0_pair_beam_id"));
        pair_point_solid_id = &(visualization_data.get_point_data<int>("uid_1_pair_solid_id"));
        pair_cell_beam_id = &(visualization_data.get_cell_data<int>("uid_0_pair_beam_id"));
        pair_cell_solid_id = &(visualization_data.get_cell_data<int>("uid_1_pair_solid_id"));
      }

      for (const auto& segment : this->line_to_3D_segments_)
      {
        for (unsigned int i_curve_segment = 0; i_curve_segment <= mortar_segments;
            i_curve_segment++)
        {
          // Get the position, displacement and lambda value at the current point.
          xi = segment.get_eta_a() + i_curve_segment * (segment.get_eta_b() - segment.get_eta_a()) /
                                         (double)mortar_segments;
          GeometryPair::evaluate_position<Beam>(xi, this->ele1pos_, r);
          GeometryPair::evaluate_position<Beam>(xi, this->ele1posref_, X);
          u = r;
          u -= X;
          GeometryPair::evaluate_position<Mortar>(xi, element_data_lambda, lambda_discret);

          // Add to output data.
          for (unsigned int dim = 0; dim < 3; dim++)
          {
            point_coordinates.push_back(Core::FADUtils::cast_to_double(X(dim)));
            displacement.push_back(Core::FADUtils::cast_to_double(u(dim)));
            lambda_vis.push_back(Core::FADUtils::cast_to_double(lambda_discret(dim)));
          }
        }

        // Add the cell for this segment (poly line).
        cell_types.push_back(4);
        cell_offsets.push_back(point_coordinates.size() / 3);

        if (write_unique_ids)
        {
          pair_cell_beam_id->push_back(this->element1()->id());
          pair_cell_solid_id->push_back(this->element2()->id());
          for (unsigned int i_curve_segment = 0; i_curve_segment <= mortar_segments;
              i_curve_segment++)
          {
            pair_point_beam_id->push_back(this->element1()->id());
            pair_point_solid_id->push_back(this->element2()->id());
          }
        }
      }
    }
  }
}

/**
 *
 */
template <typename Beam, typename Solid, typename Mortar>
void BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<Beam, Solid, Mortar>::evaluate_dm(
    Core::LinAlg::Matrix<Mortar::n_dof_, Beam::n_dof_, double>& local_D,
    Core::LinAlg::Matrix<Mortar::n_dof_, Solid::n_dof_, double>& local_M,
    Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_kappa,
    Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_constraint) const
{
  // Initialize the local mortar matrices.
  local_D.put_scalar(0.0);
  local_M.put_scalar(0.0);
  local_kappa.put_scalar(0.0);
  local_constraint.put_scalar(0.0);

  // Initialize variables for shape function values.
  Core::LinAlg::Matrix<1, Mortar::n_nodes_ * Mortar::n_val_, double> N_mortar(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, Beam::n_nodes_ * Beam::n_val_, double> N_beam(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, Solid::n_nodes_ * Solid::n_val_, double> N_solid(
      Core::LinAlg::Initialization::zero);

  // Initialize variable for beam position derivative.
  Core::LinAlg::Matrix<3, 1, double> dr_beam_ref(Core::LinAlg::Initialization::zero);

  // Initialize scalar variables.Clear
  double segment_jacobian = 0.0;
  double beam_segmentation_factor = 0.0;

  // Calculate the mortar matrices.
  // Loop over segments.
  const unsigned int n_segments = this->line_to_3D_segments_.size();
  for (unsigned int i_segment = 0; i_segment < n_segments; i_segment++)
  {
    // Factor to account for the integration segment length.
    beam_segmentation_factor = 0.5 * this->line_to_3D_segments_[i_segment].get_segment_length();

    // Gauss point loop.
    const unsigned int n_gp = this->line_to_3D_segments_[i_segment].get_projection_points().size();
    for (unsigned int i_gp = 0; i_gp < n_gp; i_gp++)
    {
      // Get the current Gauss point.
      const GeometryPair::ProjectionPoint1DTo3D<double>& projected_gauss_point =
          this->line_to_3D_segments_[i_segment].get_projection_points()[i_gp];

      // Get the jacobian in the reference configuration.
      GeometryPair::evaluate_position_derivative1<Beam>(
          projected_gauss_point.get_eta(), this->ele1posref_, dr_beam_ref);

      // Jacobian including the segment length.
      segment_jacobian = dr_beam_ref.norm2() * beam_segmentation_factor;

      // Get the shape function matrices.
      N_mortar.clear();
      N_beam.clear();
      N_solid.clear();
      GeometryPair::EvaluateShapeFunction<Mortar>::evaluate(
          N_mortar, projected_gauss_point.get_eta());
      GeometryPair::EvaluateShapeFunction<Beam>::evaluate(
          N_beam, projected_gauss_point.get_eta(), this->ele1pos_.shape_function_data_);
      GeometryPair::EvaluateShapeFunction<Solid>::evaluate(
          N_solid, projected_gauss_point.get_xi(), this->ele2pos_.shape_function_data_);

      // Fill in the local templated mortar matrix D.
      for (unsigned int i_mortar_node = 0; i_mortar_node < Mortar::n_nodes_; i_mortar_node++)
        for (unsigned int i_mortar_val = 0; i_mortar_val < Mortar::n_val_; i_mortar_val++)
          for (unsigned int i_beam_node = 0; i_beam_node < Beam::n_nodes_; i_beam_node++)
            for (unsigned int i_beam_val = 0; i_beam_val < Beam::n_val_; i_beam_val++)
              for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
                local_D(i_mortar_node * Mortar::n_val_ * 3 + i_mortar_val * 3 + i_dim,
                    i_beam_node * Beam::n_val_ * 3 + i_beam_val * 3 + i_dim) +=
                    N_mortar(i_mortar_node * Mortar::n_val_ + i_mortar_val) *
                    N_beam(i_beam_node * Beam::n_val_ + i_beam_val) *
                    projected_gauss_point.get_gauss_weight() * segment_jacobian;

      // Fill in the local templated mortar matrix M.
      for (unsigned int i_mortar_node = 0; i_mortar_node < Mortar::n_nodes_; i_mortar_node++)
        for (unsigned int i_mortar_val = 0; i_mortar_val < Mortar::n_val_; i_mortar_val++)
          for (unsigned int i_solid_node = 0; i_solid_node < Solid::n_nodes_; i_solid_node++)
            for (unsigned int i_solid_val = 0; i_solid_val < Solid::n_val_; i_solid_val++)
              for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
                local_M(i_mortar_node * Mortar::n_val_ * 3 + i_mortar_val * 3 + i_dim,
                    i_solid_node * Solid::n_val_ * 3 + i_solid_val * 3 + i_dim) +=
                    N_mortar(i_mortar_node * Mortar::n_val_ + i_mortar_val) *
                    N_solid(i_solid_node * Solid::n_val_ + i_solid_val) *
                    projected_gauss_point.get_gauss_weight() * segment_jacobian;

      // Fill in the local templated mortar scaling vector kappa.
      for (unsigned int i_mortar_node = 0; i_mortar_node < Mortar::n_nodes_; i_mortar_node++)
        for (unsigned int i_mortar_val = 0; i_mortar_val < Mortar::n_val_; i_mortar_val++)
          for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
            local_kappa(i_mortar_node * Mortar::n_val_ * 3 + i_mortar_val * 3 + i_dim) +=
                N_mortar(i_mortar_node * Mortar::n_val_ + i_mortar_val) *
                projected_gauss_point.get_gauss_weight() * segment_jacobian;
    }
  }

  // Add the local constraint contributions.
  for (unsigned int i_lambda = 0; i_lambda < Mortar::n_dof_; i_lambda++)
  {
    for (unsigned int i_beam = 0; i_beam < Beam::n_dof_; i_beam++)
      local_constraint(i_lambda) +=
          local_D(i_lambda, i_beam) *
          Core::FADUtils::cast_to_double(this->ele1pos_.element_position_(i_beam));
    for (unsigned int i_solid = 0; i_solid < Solid::n_dof_; i_solid++)
      local_constraint(i_lambda) -=
          local_M(i_lambda, i_solid) *
          Core::FADUtils::cast_to_double(this->ele2pos_.element_position_(i_solid));
  }
}

/**
 *
 */
template <typename Beam, typename Solid, typename Mortar>
void BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<Beam, Solid,
    Mortar>::evaluate_penalty_force_double(const Core::LinAlg::Matrix<3, 1, double>& r_beam,
    const Core::LinAlg::Matrix<3, 1, double>& r_solid,
    Core::LinAlg::Matrix<3, 1, double>& force) const
{
  force.put_scalar(0.0);
}


/**
 * Explicit template initialization of template class.
 */
namespace BeamInteraction
{
  using namespace GeometryPair;

  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex8, t_line2>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex20, t_line2>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex27, t_line2>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_tet4, t_line2>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_tet10, t_line2>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_nurbs27, t_line2>;

  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex8, t_line3>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex20, t_line3>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex27, t_line3>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_tet4, t_line3>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_tet10, t_line3>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_nurbs27, t_line3>;

  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex8, t_line4>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex20, t_line4>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex27, t_line4>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_tet4, t_line4>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_tet10, t_line4>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_nurbs27, t_line4>;
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE
