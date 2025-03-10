// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_geometry_pair.hpp"

#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
void Inpar::GEOMETRYPAIR::set_valid_parameters_line_to3_d(Core::Utils::SectionSpecs& list)
{
  using namespace Core::IO::InputSpecBuilders;

  // Add the input parameters for line to 3D coupling.

  // Segmentation strategy.
  Core::Utils::string_to_integral_parameter<LineTo3DStrategy>("GEOMETRY_PAIR_STRATEGY",
      "segmentation", "Type of employed segmentation strategy",
      Teuchos::tuple<std::string>("none", "segmentation",
          "gauss_point_projection_without_boundary_segmentation",
          "gauss_point_projection_boundary_segmentation", "gauss_point_projection_cross_section"),
      Teuchos::tuple<LineTo3DStrategy>(LineTo3DStrategy::none, LineTo3DStrategy::segmentation,
          LineTo3DStrategy::gauss_point_projection_without_boundary_segmentation,
          LineTo3DStrategy::gauss_point_projection_boundary_segmentation,
          LineTo3DStrategy::gauss_point_projection_cross_section),
      list);

  // Number of search points for segmentation.
  list.specs.emplace_back(parameter<int>("GEOMETRY_PAIR_SEGMENTATION_SEARCH_POINTS",
      {.description = "Number of search points for segmentation", .default_value = 6}));

  // What to do if not all Gauss points of a segment project valid
  Core::Utils::string_to_integral_parameter<NotAllGaussPointsProjectValidAction>(
      "GEOMETRY_PAIR_SEGMENTATION_NOT_ALL_GAUSS_POINTS_PROJECT_VALID_ACTION", "fail",
      "What to do if not all Gauss points of a segment project valid",
      Teuchos::tuple<std::string>("fail", "warning"),
      Teuchos::tuple<NotAllGaussPointsProjectValidAction>(
          NotAllGaussPointsProjectValidAction::fail, NotAllGaussPointsProjectValidAction::warning),
      list);

  // Number of integration points on the line.
  list.specs.emplace_back(parameter<int>("GAUSS_POINTS",
      {.description = "Number of Gauss Points for the integral evaluations", .default_value = 6}));

  // Number of integration along the circumference in cross section coupling.
  list.specs.emplace_back(parameter<int>("INTEGRATION_POINTS_CIRCUMFERENCE",
      {.description = "Number of Integration points along the circumferential direction of the "
                      "beam. This is parameter is only used in beam to cylinder meshtying. No "
                      "gauss integration is used along the circumferential direction, equally "
                      "spaced integration points are used.",
          .default_value = 6}));
}

/**
 *
 */
void Inpar::GEOMETRYPAIR::set_valid_parameters_line_to_surface(Core::Utils::SectionSpecs& list)
{
  // Add the input parameters for line to surface coupling.

  // Add the surface normal option.
  Core::Utils::string_to_integral_parameter<GEOMETRYPAIR::SurfaceNormals>(
      "GEOMETRY_PAIR_SURFACE_NORMALS", "standard", "How the surface normals are evaluated",
      Teuchos::tuple<std::string>("standard", "extended_volume"),
      Teuchos::tuple<GEOMETRYPAIR::SurfaceNormals>(
          GEOMETRYPAIR::SurfaceNormals::standard, GEOMETRYPAIR::SurfaceNormals::extended_volume),
      list);
}

/**
 *
 */
Core::FE::GaussRule1D Inpar::GEOMETRYPAIR::int_to_gauss_rule1_d(const int n_gauss_points)
{
  switch (n_gauss_points)
  {
    case 1:
      return Core::FE::GaussRule1D::line_1point;
    case 2:
      return Core::FE::GaussRule1D::line_2point;
    case 3:
      return Core::FE::GaussRule1D::line_3point;
    case 4:
      return Core::FE::GaussRule1D::line_4point;
    case 5:
      return Core::FE::GaussRule1D::line_5point;
    case 6:
      return Core::FE::GaussRule1D::line_6point;
    case 7:
      return Core::FE::GaussRule1D::line_7point;
    case 8:
      return Core::FE::GaussRule1D::line_8point;
    case 9:
      return Core::FE::GaussRule1D::line_9point;
    case 10:
      return Core::FE::GaussRule1D::line_10point;
    case 20:
      return Core::FE::GaussRule1D::line_20point;
    case 32:
      return Core::FE::GaussRule1D::line_32point;
    case 50:
      return Core::FE::GaussRule1D::line_50point;
    default:
    {
      FOUR_C_THROW("No Gauss rule defined for %d points", n_gauss_points);
      return Core::FE::GaussRule1D::undefined;
    }
  }
};

FOUR_C_NAMESPACE_CLOSE
