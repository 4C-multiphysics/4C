// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_surface_meshtying_pair_mortar_base.hpp"

#include "4C_beaminteraction_beam_to_solid_mortar_manager.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_visualization_output_params.hpp"
#include "4C_beaminteraction_beam_to_solid_utils.hpp"
#include "4C_beaminteraction_beam_to_solid_visualization_output_writer_base.hpp"
#include "4C_beaminteraction_beam_to_solid_visualization_output_writer_visualization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_line_to_surface.hpp"

#include <unordered_set>

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface, typename Mortar>
BeamInteraction::BeamToSolidSurfaceMeshtyingPairMortarBase<ScalarType, Beam, Surface,
    Mortar>::BeamToSolidSurfaceMeshtyingPairMortarBase()
    : base_class(), n_mortar_rot_(0)
{
  // Empty constructor.
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface, typename Mortar>
void BeamInteraction::BeamToSolidSurfaceMeshtyingPairMortarBase<ScalarType, Beam, Surface,
    Mortar>::get_pair_visualization(std::shared_ptr<BeamToSolidVisualizationOutputWriterBase>
                                        visualization_writer,
    Teuchos::ParameterList& visualization_params) const
{
  // Get visualization of base method.
  base_class::get_pair_visualization(visualization_writer, visualization_params);

  std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization> visualization_discret =
      visualization_writer->get_visualization_writer("btss-coupling-mortar");
  std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization> visualization_continuous =
      visualization_writer->get_visualization_writer("btss-coupling-mortar-continuous");
  std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization>
      visualization_nodal_forces =
          visualization_writer->get_visualization_writer("btss-coupling-nodal-forces");
  if (!visualization_discret and !visualization_continuous and
      visualization_nodal_forces == nullptr)
    return;

  const std::shared_ptr<const BeamToSolidSurfaceVisualizationOutputParams>& output_params_ptr =
      visualization_params.get<std::shared_ptr<const BeamToSolidSurfaceVisualizationOutputParams>>(
          "btss-output_params_ptr");
  const bool write_unique_ids = output_params_ptr->get_write_unique_ids_flag();

  if (visualization_discret != nullptr or visualization_continuous != nullptr or
      visualization_nodal_forces != nullptr)
  {
    // Setup variables.
    Core::LinAlg::Matrix<3, 1, ScalarType> X;
    Core::LinAlg::Matrix<3, 1, ScalarType> r;
    Core::LinAlg::Matrix<3, 1, ScalarType> u;
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
    auto q_lambda = GEOMETRYPAIR::InitializeElementData<Mortar, double>::initialize(nullptr);
    const auto& [lambda_row_pos, _] = mortar_manager->location_vector(*this);
    std::vector<double> lambda_pair = Core::FE::extract_values(*lambda, lambda_row_pos);
    for (unsigned int i_dof = 0; i_dof < Mortar::n_dof_; i_dof++)
      q_lambda.element_position_(i_dof) = lambda_pair[i_dof];

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
          GEOMETRYPAIR::evaluate_position<Beam>(xi_mortar_node(0), this->ele1pos_, r);
          GEOMETRYPAIR::evaluate_position<Beam>(xi_mortar_node(0), this->ele1posref_, X);
          u = r;
          u -= X;

          // Get the discrete Lagrangian multiplier.
          GEOMETRYPAIR::evaluate_position<Mortar>(xi_mortar_node(0), q_lambda, lambda_discret);

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
              .get<std::shared_ptr<const BeamToSolidSurfaceVisualizationOutputParams>>(
                  "btss-output_params_ptr")
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
          GEOMETRYPAIR::evaluate_position<Beam>(xi, this->ele1pos_, r);
          GEOMETRYPAIR::evaluate_position<Beam>(xi, this->ele1posref_, X);
          u = r;
          u -= X;
          GEOMETRYPAIR::evaluate_position<Mortar>(xi, q_lambda, lambda_discret);

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


    // Calculate the global moment of the coupling load.
    if (visualization_nodal_forces != nullptr)
    {
      // Get the global moment vector.
      auto line_load_moment_origin =
          visualization_params.get<std::shared_ptr<Core::LinAlg::Matrix<3, 1, double>>>(
              "global_coupling_moment_origin");

      // Initialize variables for local values.
      Core::LinAlg::Matrix<3, 1, double> dr_beam_ref(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<3, 1, double> lambda_gauss_point(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<3, 1, double> r_gauss_point(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<3, 1, double> temp_moment(Core::LinAlg::Initialization::zero);

      // Initialize scalar variables.
      double segment_jacobian = 0.0;
      double beam_segmentation_factor = 0.0;

      // Loop over segments to evaluate the coupling potential.
      const unsigned int n_segments = this->line_to_3D_segments_.size();
      for (unsigned int i_segment = 0; i_segment < n_segments; i_segment++)
      {
        // Factor to account for the integration segment length.
        beam_segmentation_factor = 0.5 * this->line_to_3D_segments_[i_segment].get_segment_length();

        // Gauss point loop.
        const unsigned int n_gp =
            this->line_to_3D_segments_[i_segment].get_projection_points().size();
        for (unsigned int i_gp = 0; i_gp < n_gp; i_gp++)
        {
          // Get the current Gauss point.
          const GEOMETRYPAIR::ProjectionPoint1DTo3D<double>& projected_gauss_point =
              this->line_to_3D_segments_[i_segment].get_projection_points()[i_gp];

          // Get the jacobian in the reference configuration.
          GEOMETRYPAIR::evaluate_position_derivative1<Beam>(
              projected_gauss_point.get_eta(), this->ele1posref_, dr_beam_ref);

          // Jacobian including the segment length.
          segment_jacobian = dr_beam_ref.norm2() * beam_segmentation_factor;

          // Evaluate the coupling load at this point.
          GEOMETRYPAIR::evaluate_position<Mortar>(
              projected_gauss_point.get_eta(), q_lambda, lambda_gauss_point);

          // Get the position at this Gauss point.
          GEOMETRYPAIR::evaluate_position<Beam>(projected_gauss_point.get_eta(),
              GEOMETRYPAIR::ElementDataToDouble<Beam>::to_double(this->ele1pos_), r_gauss_point);

          // Calculate moment around origin.
          temp_moment.cross_product(r_gauss_point, lambda_gauss_point);
          temp_moment.scale(projected_gauss_point.get_gauss_weight() * segment_jacobian);
          (*line_load_moment_origin) += temp_moment;
        }
      }
    }
  }
}


/**
 * Explicit template initialization of template class.
 */
namespace BeamInteraction
{
  using namespace GEOMETRYPAIR;

  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_tri3>, t_hermite, t_tri3, t_line2>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_tri6>, t_hermite, t_tri6, t_line2>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_quad4>, t_hermite, t_quad4, t_line2>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_quad8>, t_hermite, t_quad8, t_line2>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_quad9>, t_hermite, t_quad9, t_line2>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_nurbs9>, t_hermite, t_nurbs9, t_line2>;

  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_tri3>, t_hermite, t_tri3, t_line3>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_tri6>, t_hermite, t_tri6, t_line3>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_quad4>, t_hermite, t_quad4, t_line3>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_quad8>, t_hermite, t_quad8, t_line3>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_quad9>, t_hermite, t_quad9, t_line3>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_nurbs9>, t_hermite, t_nurbs9, t_line3>;

  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_tri3>, t_hermite, t_tri3, t_line4>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_tri6>, t_hermite, t_tri6, t_line4>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_quad4>, t_hermite, t_quad4, t_line4>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_quad8>, t_hermite, t_quad8, t_line4>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_quad9>, t_hermite, t_quad9, t_line4>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_scalar_type<t_hermite, t_nurbs9>, t_hermite, t_nurbs9, t_line4>;


  template class BeamToSolidSurfaceMeshtyingPairMortarBase<line_to_surface_patch_scalar_type,
      t_hermite, t_tri3, t_line2>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<line_to_surface_patch_scalar_type,
      t_hermite, t_tri6, t_line2>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<line_to_surface_patch_scalar_type,
      t_hermite, t_quad4, t_line2>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<line_to_surface_patch_scalar_type,
      t_hermite, t_quad8, t_line2>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<line_to_surface_patch_scalar_type,
      t_hermite, t_quad9, t_line2>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_nurbs9>, t_hermite, t_nurbs9,
      t_line2>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex8>, t_hermite, t_quad4, t_line2>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex20>, t_hermite, t_quad8,
      t_line2>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex27>, t_hermite, t_quad9,
      t_line2>;

  template class BeamToSolidSurfaceMeshtyingPairMortarBase<line_to_surface_patch_scalar_type,
      t_hermite, t_tri3, t_line3>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<line_to_surface_patch_scalar_type,
      t_hermite, t_tri6, t_line3>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<line_to_surface_patch_scalar_type,
      t_hermite, t_quad4, t_line3>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<line_to_surface_patch_scalar_type,
      t_hermite, t_quad8, t_line3>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<line_to_surface_patch_scalar_type,
      t_hermite, t_quad9, t_line3>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_nurbs9>, t_hermite, t_nurbs9,
      t_line3>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex8>, t_hermite, t_quad4, t_line3>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex20>, t_hermite, t_quad8,
      t_line3>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex27>, t_hermite, t_quad9,
      t_line3>;

  template class BeamToSolidSurfaceMeshtyingPairMortarBase<line_to_surface_patch_scalar_type,
      t_hermite, t_tri3, t_line4>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<line_to_surface_patch_scalar_type,
      t_hermite, t_tri6, t_line4>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<line_to_surface_patch_scalar_type,
      t_hermite, t_quad4, t_line4>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<line_to_surface_patch_scalar_type,
      t_hermite, t_quad8, t_line4>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<line_to_surface_patch_scalar_type,
      t_hermite, t_quad9, t_line4>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_nurbs9>, t_hermite, t_nurbs9,
      t_line4>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex8>, t_hermite, t_quad4, t_line4>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex20>, t_hermite, t_quad8,
      t_line4>;
  template class BeamToSolidSurfaceMeshtyingPairMortarBase<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex27>, t_hermite, t_quad9,
      t_line4>;
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE
