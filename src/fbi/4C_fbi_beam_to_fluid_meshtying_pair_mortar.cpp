// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fbi_beam_to_fluid_meshtying_pair_mortar.hpp"

#include "4C_beaminteraction_beam_to_solid_visualization_output_writer_base.hpp"
#include "4C_beaminteraction_beam_to_solid_visualization_output_writer_visualization.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_output_params.hpp"
#include "4C_fbi_beam_to_fluid_mortar_manager.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_line_to_volume.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
template <typename Beam, typename Fluid, typename Mortar>
BeamInteraction::BeamToFluidMeshtyingPairMortar<Beam, Fluid,
    Mortar>::BeamToFluidMeshtyingPairMortar()
    : BeamToFluidMeshtyingPairBase<Beam, Fluid>()
{
  // Empty constructor.
}


/**
 *
 */
template <typename Beam, typename Fluid, typename Mortar>
bool BeamInteraction::BeamToFluidMeshtyingPairMortar<Beam, Fluid, Mortar>::evaluate_dm(
    Core::LinAlg::SerialDenseMatrix& local_D, Core::LinAlg::SerialDenseMatrix& local_M,
    Core::LinAlg::SerialDenseVector& local_kappa,
    Core::LinAlg::SerialDenseVector& local_constraint_offset)
{
  if (!this->meshtying_is_evaluated_)
  {
    this->cast_geometry_pair()->evaluate(
        this->ele1poscur_, this->ele2poscur_, this->line_to_3D_segments_);
    this->meshtying_is_evaluated_ = true;
  }

  // If there are no intersection segments, return no contact status.
  if (this->line_to_3D_segments_.size() == 0) return false;

  // Initialize variables for local mortar matrices.
  local_D.shape(Mortar::n_dof_, Beam::n_dof_);
  local_M.shape(Mortar::n_dof_, Fluid::n_dof_);
  local_kappa.size(Mortar::n_dof_);


  // Initialize variables for shape function values.
  Core::LinAlg::Matrix<1, Mortar::n_nodes_ * Mortar::n_val_, double> N_mortar(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, Beam::n_nodes_ * Beam::n_val_, double> N_beam(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, Fluid::n_nodes_ * Fluid::n_val_, double> N_fluid(
      Core::LinAlg::Initialization::zero);

  // Initialize variable for beam position derivative.
  Core::LinAlg::Matrix<3, 1, double> dr_beam_ref(Core::LinAlg::Initialization::zero);

  // Initialize scalar variables.Clear
  double segment_jacobian, beam_segmentation_factor;

  // Calculate the mortar matrices.
  // Loop over segments.
  for (unsigned int i_segment = 0; i_segment < this->line_to_3D_segments_.size(); i_segment++)
  {
    // Factor to account for the integration segment length.
    beam_segmentation_factor = 0.5 * this->line_to_3D_segments_[i_segment].get_segment_length();

    // Gauss point loop.
    for (unsigned int i_gp = 0;
        i_gp < this->line_to_3D_segments_[i_segment].get_projection_points().size(); i_gp++)
    {
      // Get the current Gauss point.
      const GEOMETRYPAIR::ProjectionPoint1DTo3D<double>& projected_gauss_point =
          this->line_to_3D_segments_[i_segment].get_projection_points()[i_gp];

      // Get the jacobian in the reference configuration.
      GEOMETRYPAIR::evaluate_position_derivative1<Beam>(
          projected_gauss_point.get_eta(), this->ele1posref_, dr_beam_ref);

      // Jacobian including the segment length.
      segment_jacobian = dr_beam_ref.norm2() * beam_segmentation_factor;

      // Get the shape function matrices.
      N_mortar.clear();
      N_beam.clear();
      N_fluid.clear();
      GEOMETRYPAIR::EvaluateShapeFunction<Mortar>::evaluate(
          N_mortar, projected_gauss_point.get_eta());
      GEOMETRYPAIR::EvaluateShapeFunction<Beam>::evaluate(
          N_beam, projected_gauss_point.get_eta(), this->ele1pos_.shape_function_data_);
      GEOMETRYPAIR::EvaluateShapeFunction<Fluid>::evaluate(N_fluid, projected_gauss_point.get_xi());

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
          for (unsigned int i_fluid_node = 0; i_fluid_node < Fluid::n_nodes_; i_fluid_node++)
            for (unsigned int i_fluid_val = 0; i_fluid_val < Fluid::n_val_; i_fluid_val++)
              for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
                local_M(i_mortar_node * Mortar::n_val_ * 3 + i_mortar_val * 3 + i_dim,
                    i_fluid_node * Fluid::n_val_ * 3 + i_fluid_val * 3 + i_dim) +=
                    N_mortar(i_mortar_node * Mortar::n_val_ + i_mortar_val) *
                    N_fluid(i_fluid_node * Fluid::n_val_ + i_fluid_val) *
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

  // If we get to this point, the pair has a mortar contribution.
  return true;
}

/**
 *
 */
template <typename Beam, typename Fluid, typename Mortar>
void BeamInteraction::BeamToFluidMeshtyingPairMortar<Beam, Fluid, Mortar>::get_pair_visualization(
    std::shared_ptr<BeamToSolidVisualizationOutputWriterBase> visualization_writer,
    Teuchos::ParameterList& visualization_params) const
{
  // Get visualization of base method.
  BeamToFluidMeshtyingPairBase<Beam, Fluid>::get_pair_visualization(
      visualization_writer, visualization_params);


  std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization> visualization_discret =
      visualization_writer->get_visualization_writer("mortar");
  std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization> visualization_continuous =
      visualization_writer->get_visualization_writer("mortar-continuous");
  if (visualization_discret != nullptr || visualization_continuous != nullptr)
  {
    // Setup variables.
    auto q_lambda = GEOMETRYPAIR::InitializeElementData<Mortar, double>::initialize(nullptr);
    Core::LinAlg::Matrix<3, 1, scalar_type> current_beamposition;
    Core::LinAlg::Matrix<3, 1, scalar_type> ref_beamposition;
    Core::LinAlg::Matrix<3, 1, scalar_type> beamdisplacement;
    Core::LinAlg::Matrix<3, 1, double> lambda_discret;
    Core::LinAlg::Matrix<3, 1, double> xi_mortar_node;

    // Get the mortar manager and the global lambda vector, those objects will be used to get the
    // discrete Lagrange multiplier values for this pair.
    std::shared_ptr<const BeamInteraction::BeamToFluidMortarManager> mortar_manager =
        visualization_params.get<std::shared_ptr<const BeamInteraction::BeamToFluidMortarManager>>(
            "mortar_manager");
    std::shared_ptr<Core::LinAlg::Vector<double>> lambda =
        visualization_params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("lambda");

    // Get the lambda GIDs of this pair.
    std::shared_ptr<const BeamContactPair> this_rcp = Core::Utils::shared_ptr_from_ref(*this);
    std::vector<int> lambda_row;
    mortar_manager->location_vector(*this_rcp, lambda_row);
    std::vector<double> lambda_pair = Core::FE::extract_values(*lambda, lambda_row);
    for (unsigned int i_dof = 0; i_dof < Mortar::n_dof_; i_dof++)
      q_lambda.element_position_(i_dof) = lambda_pair[i_dof];

    // Add the discrete values of the Lagrange multipliers.
    if (visualization_discret != nullptr)
    {
      // Get the visualization vectors.
      auto& visualization_data = visualization_discret->get_visualization_data();
      std::vector<double>& point_coordinates = visualization_data.get_point_coordinates();
      std::vector<double>& displacement = visualization_data.get_point_data<double>("displacement");
      std::vector<double>& lambda_vis = visualization_data.get_point_data<double>("lambda");

      for (unsigned int i_node = 0; i_node < Mortar::n_nodes_; i_node++)
      {
        // Get the local coordinate of this node.
        xi_mortar_node = Core::FE::get_node_coordinates(i_node, Mortar::discretization_);

        // Get position and displacement of the mortar node.
        GEOMETRYPAIR::evaluate_position<Beam>(
            xi_mortar_node(0), this->ele1pos_, current_beamposition);
        GEOMETRYPAIR::evaluate_position<Beam>(
            xi_mortar_node(0), this->ele1posref_, ref_beamposition);
        beamdisplacement = current_beamposition;
        beamdisplacement -= ref_beamposition;

        // Get the discrete Lagrangian multiplier.
        GEOMETRYPAIR::evaluate_position<Mortar>(xi_mortar_node(0), q_lambda, lambda_discret);

        // Add to output data.
        for (unsigned int dim = 0; dim < 3; dim++)
        {
          point_coordinates.push_back(Core::FADUtils::cast_to_double(current_beamposition(dim)));
          displacement.push_back(Core::FADUtils::cast_to_double(beamdisplacement(dim)));
          lambda_vis.push_back(Core::FADUtils::cast_to_double(lambda_discret(dim)));
        }
      }
    }


    // Add the continuous values for the Lagrange multipliers.
    if (visualization_continuous != nullptr)
    {
      const unsigned int mortar_segments =
          visualization_params
              .get<std::shared_ptr<const BeamToSolidVolumeMeshtyingVisualizationOutputParams>>(
                  "output_params_ptr")
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

      for (const auto& segment : this->line_to_3D_segments_)
      {
        for (unsigned int i_curve_segment = 0; i_curve_segment <= mortar_segments;
            i_curve_segment++)
        {
          // Get the position, displacement and lambda value at the current point.
          xi = segment.get_eta_a() + i_curve_segment * (segment.get_eta_b() - segment.get_eta_a()) /
                                         (double)mortar_segments;
          GEOMETRYPAIR::evaluate_position<Beam>(xi, this->ele1pos_, current_beamposition);
          GEOMETRYPAIR::evaluate_position<Beam>(xi, this->ele1posref_, ref_beamposition);
          beamdisplacement = current_beamposition;
          beamdisplacement -= ref_beamposition;
          GEOMETRYPAIR::evaluate_position<Mortar>(xi, q_lambda, lambda_discret);

          // Add to output data.
          for (unsigned int dim = 0; dim < 3; dim++)
          {
            point_coordinates.push_back(Core::FADUtils::cast_to_double(current_beamposition(dim)));
            displacement.push_back(Core::FADUtils::cast_to_double(beamdisplacement(dim)));
            lambda_vis.push_back(Core::FADUtils::cast_to_double(lambda_discret(dim)));
          }
        }

        // Add the cell for this segment (poly line).
        cell_types.push_back(4);
        cell_offsets.push_back(point_coordinates.size() / 3);
      }
    }
  }
}

template <typename Beam, typename Fluid, typename Mortar>
void BeamInteraction::BeamToFluidMeshtyingPairMortar<Beam, Fluid, Mortar>::evaluate_penalty_force(
    Core::LinAlg::Matrix<3, 1, scalar_type>& force,
    const GEOMETRYPAIR::ProjectionPoint1DTo3D<double>& projected_gauss_point,
    Core::LinAlg::Matrix<3, 1, scalar_type> v_beam) const
{
  force.put_scalar(0.);
}
/**
 * Explicit template initialization of template class.
 */
template class BeamInteraction::BeamToFluidMeshtyingPairMortar<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_hex8, GEOMETRYPAIR::t_line2>;
template class BeamInteraction::BeamToFluidMeshtyingPairMortar<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_hex20, GEOMETRYPAIR::t_line2>;
template class BeamInteraction::BeamToFluidMeshtyingPairMortar<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_hex27, GEOMETRYPAIR::t_line2>;
template class BeamInteraction::BeamToFluidMeshtyingPairMortar<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_tet4, GEOMETRYPAIR::t_line2>;
template class BeamInteraction::BeamToFluidMeshtyingPairMortar<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_tet10, GEOMETRYPAIR::t_line2>;

template class BeamInteraction::BeamToFluidMeshtyingPairMortar<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_hex8, GEOMETRYPAIR::t_line3>;
template class BeamInteraction::BeamToFluidMeshtyingPairMortar<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_hex20, GEOMETRYPAIR::t_line3>;
template class BeamInteraction::BeamToFluidMeshtyingPairMortar<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_hex27, GEOMETRYPAIR::t_line3>;
template class BeamInteraction::BeamToFluidMeshtyingPairMortar<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_tet4, GEOMETRYPAIR::t_line3>;
template class BeamInteraction::BeamToFluidMeshtyingPairMortar<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_tet10, GEOMETRYPAIR::t_line3>;

template class BeamInteraction::BeamToFluidMeshtyingPairMortar<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_hex8, GEOMETRYPAIR::t_line4>;
template class BeamInteraction::BeamToFluidMeshtyingPairMortar<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_hex20, GEOMETRYPAIR::t_line4>;
template class BeamInteraction::BeamToFluidMeshtyingPairMortar<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_hex27, GEOMETRYPAIR::t_line4>;
template class BeamInteraction::BeamToFluidMeshtyingPairMortar<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_tet4, GEOMETRYPAIR::t_line4>;
template class BeamInteraction::BeamToFluidMeshtyingPairMortar<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_tet10, GEOMETRYPAIR::t_line4>;

FOUR_C_NAMESPACE_CLOSE
