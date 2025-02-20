// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_geometry_pair_line_to_volume_segmentation.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beam3_reissner.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_line_to_3D_evaluation_data.hpp"
#include "4C_geometry_pair_line_to_volume_segmentation_geometry_functions_test.hpp"
#include "4C_geometry_pair_utility_classes.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

#include <mpi.h>


namespace
{

  void set_up_default_parameters_line_to_3d(Teuchos::ParameterList& list)
  {
    list.set("GEOMETRY_PAIR_STRATEGY", Inpar::GEOMETRYPAIR::LineTo3DStrategy::segmentation);
    list.set("GEOMETRY_PAIR_SEGMENTATION_SEARCH_POINTS", 6);
    list.set("GEOMETRY_PAIR_SEGMENTATION_NOT_ALL_GAUSS_POINTS_PROJECT_VALID_ACTION",
        Inpar::GEOMETRYPAIR::NotAllGaussPointsProjectValidAction::fail);
    list.set("GAUSS_POINTS", 6);
    list.set("INTEGRATION_POINTS_CIRCUMFERENCE", 6);
  }

  /**
   * Class to test the line to volume geometry pair segmentation algorithm.
   */
  class GeometryPairLineToVolumeSegmentationTest : public ::testing::Test
  {
   protected:
    /**
     * Set up the testing environment.
     */
    GeometryPairLineToVolumeSegmentationTest()
    {
      // Set up the evaluation data container for the geometry pairs.
      Teuchos::ParameterList line_to_volume_params_list;
      set_up_default_parameters_line_to_3d(line_to_volume_params_list);
      evaluation_data_ =
          std::make_shared<GEOMETRYPAIR::LineTo3DEvaluationData>(line_to_volume_params_list);
    }

    /**
     * This function set up the geometry pairs, line elements and calls evaluate on the pairs. The
     * geometry of the test case is given in the various input parameters. The calculated segments
     * are returned which can be checked for results.
     * @param geometry_pairs (out) Vector with the geometry pairs.
     * @param q_line_elements (in) Vector of DOF vectors for the positions and tangents of the line
     * element.
     * @param line_ref_lengths (in) Reference length of the beam elements.
     * @param q_volume_elements (in) Vector of DOF vectors for the positions of the volume element.
     * @param segments_vector (out) Vector with found segments for each pair.
     */
    template <typename El1, typename El2>
    void create_evaluate_pairs(
        std::vector<std::shared_ptr<
            GEOMETRYPAIR::GeometryPairLineToVolumeSegmentation<double, El1, El2>>>& geometry_pairs,
        const std::vector<Core::LinAlg::Matrix<El1::n_dof_, 1, double>>& q_line_elements,
        const std::vector<double>& line_ref_lengths,
        const std::vector<Core::LinAlg::Matrix<El2::n_dof_, 1, double>>& q_volume_elements,
        std::vector<std::vector<GEOMETRYPAIR::LineSegment<double>>>& segments_vector)
    {
      // Check that the vectors have the right size.
      if (line_elements_.size() != q_line_elements.size())
        FOUR_C_THROW("Size for line elements and line q does not match!");
      if (volume_elements_.size() != q_volume_elements.size())
        FOUR_C_THROW("Size for volume elements and volume q does not match!");

      // Get the element data containers
      std::vector<GEOMETRYPAIR::ElementData<El1, double>> q_line(q_line_elements.size());
      for (unsigned int i_beam = 0; i_beam < line_elements_.size(); i_beam++)
      {
        q_line[i_beam].element_position_ = q_line_elements[i_beam];
        q_line[i_beam].shape_function_data_.ref_length_ = line_ref_lengths[i_beam];
      }
      std::vector<GEOMETRYPAIR::ElementData<El2, double>> q_volume(q_volume_elements.size());
      for (unsigned int i_volume = 0; i_volume < volume_elements_.size(); i_volume++)
      {
        q_volume[i_volume] = GEOMETRYPAIR::InitializeElementData<El2, double>::initialize(
            volume_elements_[i_volume].get());
        q_volume[i_volume].element_position_ = q_volume_elements[i_volume];
      }

      // Create the geometry pairs.
      for (auto& line : line_elements_)
      {
        // Loop over each solid with this beam and create a pair.
        for (auto& volume : volume_elements_)
        {
          geometry_pairs.push_back(std::make_shared<
              GEOMETRYPAIR::GeometryPairLineToVolumeSegmentation<double, El1, El2>>(
              line.get(), volume.get(), evaluation_data_));
        }
      }
      segments_vector.resize(geometry_pairs.size());

      // Evaluate the segmentation.
      unsigned int counter = 0;
      for (unsigned int i_line = 0; i_line < q_line_elements.size(); i_line++)
      {
        for (unsigned int i_volume = 0; i_volume < q_volume_elements.size(); i_volume++)
        {
          geometry_pairs[counter]->evaluate(
              q_line[i_line], q_volume[i_volume], segments_vector[counter]);
          counter++;
        }
      }
    }

    //! Evaluation data container for geometry pairs.
    std::shared_ptr<GEOMETRYPAIR::LineTo3DEvaluationData> evaluation_data_;

    //! Vector of line elements.
    std::vector<std::shared_ptr<Core::Elements::Element>> line_elements_;

    //! Vector of volume elements.
    std::vector<std::shared_ptr<Core::Elements::Element>> volume_elements_;
  };

  /**
   * Test a non straight beam that lies exactly between two solid. The segmentation should only be
   * done on one of the pairs.
   */
  TEST_F(GeometryPairLineToVolumeSegmentationTest, TestLineAlongElementSurface)
  {
    // Definition of variables for this test case.
    std::vector<Core::LinAlg::Matrix<12, 1, double>> q_line_elements;
    std::vector<double> line_ref_lengths;
    std::vector<Core::LinAlg::Matrix<24, 1, double>> q_volume_elements;
    std::vector<std::shared_ptr<GEOMETRYPAIR::GeometryPairLineToVolumeSegmentation<double,
        GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex8>>>
        geometry_pairs;

    // Get the geometry.
    xtest_line_along_element_surface_geometry(
        line_elements_, volume_elements_, q_line_elements, line_ref_lengths, q_volume_elements);

    // Vector with vector of segments for Evaluate.
    std::vector<std::vector<GEOMETRYPAIR::LineSegment<double>>> segments_vector;

    // Create and evaluate the geometry pairs.
    create_evaluate_pairs(
        geometry_pairs, q_line_elements, line_ref_lengths, q_volume_elements, segments_vector);

    // Check results.
    {
      // The segment is found on both pairs, but only evaluated on the first one that found it.
      EXPECT_EQ(segments_vector[0].size(), 1);
      EXPECT_EQ(segments_vector[1].size(), 0);

      // The first pair contains the full beam.
      EXPECT_NEAR(
          -1., segments_vector[0][0].get_eta_a(), GEOMETRYPAIR::Constants::projection_xi_eta_tol);
      EXPECT_NEAR(
          1., segments_vector[0][0].get_eta_b(), GEOMETRYPAIR::Constants::projection_xi_eta_tol);
    }
  }

  /**
   * Test a non straight beam that lies between two volume elements. The elements are
   relatively
   * small. This is to check that the parameter coordinates are converged in the local Newton
   * iterations.
   */
  TEST_F(GeometryPairLineToVolumeSegmentationTest, TestLineInSmallElements)
  {
    // Definition of variables for this test case.
    std::vector<Core::LinAlg::Matrix<12, 1, double>> q_line_elements;
    std::vector<double> line_ref_lengths;
    std::vector<Core::LinAlg::Matrix<24, 1, double>> q_volume_elements;
    std::vector<std::shared_ptr<GEOMETRYPAIR::GeometryPairLineToVolumeSegmentation<double,
        GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex8>>>
        geometry_pairs;

    // Get the geometry.
    xtest_line_in_small_elements_geometry(
        line_elements_, volume_elements_, q_line_elements, line_ref_lengths, q_volume_elements);

    // Vector with vector of segments for Evaluate.
    std::vector<std::vector<GEOMETRYPAIR::LineSegment<double>>> segments_vector;

    // Create and evaluate the geometry pairs.
    create_evaluate_pairs(
        geometry_pairs, q_line_elements, line_ref_lengths, q_volume_elements, segments_vector);

    // Check results.
    {
      EXPECT_EQ(segments_vector[0].size(), 1);
      EXPECT_EQ(segments_vector[1].size(), 1);

      // Check the parameter coordinates.
      EXPECT_NEAR(
          -1., segments_vector[0][0].get_eta_a(), GEOMETRYPAIR::Constants::projection_xi_eta_tol);
      EXPECT_NEAR(0.36285977578126655, segments_vector[0][0].get_eta_b(),
          GEOMETRYPAIR::Constants::projection_xi_eta_tol);
      EXPECT_NEAR(0.36285977578126744, segments_vector[1][0].get_eta_a(),
          GEOMETRYPAIR::Constants::projection_xi_eta_tol);
      EXPECT_NEAR(
          1.0, segments_vector[1][0].get_eta_b(), GEOMETRYPAIR::Constants::projection_xi_eta_tol);
    }
  }

  /**
   * Test a beam that has multiple intersections with a HEX27 element.
   */
  TEST_F(GeometryPairLineToVolumeSegmentationTest, TestMultipleIntersectionsHex27)
  {
    // Definition of variables for this test case.
    std::vector<Core::LinAlg::Matrix<12, 1, double>> q_line_elements;
    std::vector<double> line_ref_lengths;
    std::vector<Core::LinAlg::Matrix<81, 1, double>> q_volume_elements;
    std::vector<std::shared_ptr<GEOMETRYPAIR::GeometryPairLineToVolumeSegmentation<double,
        GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex27>>>
        geometry_pairs;

    // Get the geometry.
    xtest_multiple_intersections_hex27_geometry(
        line_elements_, volume_elements_, q_line_elements, line_ref_lengths, q_volume_elements);

    // Vector with vector of segments for Evaluate.
    std::vector<std::vector<GEOMETRYPAIR::LineSegment<double>>> segments_vector;

    // Create and evaluate the geometry pairs.
    create_evaluate_pairs(
        geometry_pairs, q_line_elements, line_ref_lengths, q_volume_elements, segments_vector);

    // Check results.
    {
      // Two segments should be found.
      EXPECT_EQ(segments_vector[0].size(), 2);

      // Check the segment coordinates on the line.
      EXPECT_NEAR(-0.7495456134309243, segments_vector[0][0].get_eta_a(),
          GEOMETRYPAIR::Constants::projection_xi_eta_tol);
      EXPECT_NEAR(-0.44451080329628256, segments_vector[0][0].get_eta_b(),
          GEOMETRYPAIR::Constants::projection_xi_eta_tol);
      EXPECT_NEAR(0.076870238957896297, segments_vector[0][1].get_eta_a(),
          GEOMETRYPAIR::Constants::projection_xi_eta_tol);
      EXPECT_NEAR(0.95200105410689462, segments_vector[0][1].get_eta_b(),
          GEOMETRYPAIR::Constants::projection_xi_eta_tol);
    }
  }

  /**
   * Test a beam that has multiple intersections with a TET10 element.
   */
  TEST_F(GeometryPairLineToVolumeSegmentationTest, TestMultipleIntersectionsTet10)
  {
    // Definition of variables for this test case.
    std::vector<Core::LinAlg::Matrix<12, 1, double>> q_line_elements;
    std::vector<double> line_ref_lengths;
    std::vector<Core::LinAlg::Matrix<30, 1, double>> q_volume_elements;
    std::vector<std::shared_ptr<GEOMETRYPAIR::GeometryPairLineToVolumeSegmentation<double,
        GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_tet10>>>
        geometry_pairs;

    // Get the geometry.
    xtest_multiple_intersections_tet10_geometry(
        line_elements_, volume_elements_, q_line_elements, line_ref_lengths, q_volume_elements);

    // Vector with vector of segments for Evaluate.
    std::vector<std::vector<GEOMETRYPAIR::LineSegment<double>>> segments_vector;

    // Create and evaluate the geometry pairs.
    create_evaluate_pairs(
        geometry_pairs, q_line_elements, line_ref_lengths, q_volume_elements, segments_vector);

    // Check results.
    {
      // Two segments should be found.
      EXPECT_EQ(segments_vector[0].size(), 2);

      // Check the segment coordinates on the line.
      EXPECT_NEAR(-0.40853230756138476, segments_vector[0][0].get_eta_a(),
          GEOMETRYPAIR::Constants::projection_xi_eta_tol);
      EXPECT_NEAR(-0.079518054716933712, segments_vector[0][0].get_eta_b(),
          GEOMETRYPAIR::Constants::projection_xi_eta_tol);
      EXPECT_NEAR(0.88282786408473413, segments_vector[0][1].get_eta_a(),
          GEOMETRYPAIR::Constants::projection_xi_eta_tol);
      EXPECT_NEAR(0.97917616983415101, segments_vector[0][1].get_eta_b(),
          GEOMETRYPAIR::Constants::projection_xi_eta_tol);
    }
  }

  /**
   * Test a beam that has multiple intersections with a NURBS27 element.
   */
  TEST_F(GeometryPairLineToVolumeSegmentationTest, TestMultipleIntersectionsNurbs27)
  {
    // Definition of variables for this test case.
    std::vector<Core::LinAlg::Matrix<12, 1, double>> q_line_elements;
    std::vector<double> line_ref_lengths;
    std::vector<Core::LinAlg::Matrix<81, 1, double>> q_volume_elements;
    std::vector<std::shared_ptr<GEOMETRYPAIR::GeometryPairLineToVolumeSegmentation<double,
        GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_nurbs27>>>
        geometry_pairs;

    // Vector with vector of segments for Evaluate.
    std::vector<std::vector<GEOMETRYPAIR::LineSegment<double>>> segments_vector;

    // Add the relevant nurbs information to the discretization.
    std::shared_ptr<Core::FE::Nurbs::NurbsDiscretization> structdis =
        std::make_shared<Core::FE::Nurbs::NurbsDiscretization>("structure", MPI_COMM_NULL, 3);
    Global::Problem::instance()->add_dis("structure", structdis);

    // Get the geometry.
    xtest_multiple_intersections_nurbs27_geometry(line_elements_, volume_elements_, q_line_elements,
        line_ref_lengths, q_volume_elements, structdis);

    // Create and evaluate the geometry pairs.
    create_evaluate_pairs(
        geometry_pairs, q_line_elements, line_ref_lengths, q_volume_elements, segments_vector);

    // Check results.
    {
      // Two segments should be found.
      EXPECT_EQ(segments_vector[0].size(), 1);

      // Check the segment coordinates on the line.
      EXPECT_NEAR(-0.40769440465702655, segments_vector[0][0].get_eta_a(),
          GEOMETRYPAIR::Constants::projection_xi_eta_tol);
      EXPECT_NEAR(-0.0090552523537554153, segments_vector[0][0].get_eta_b(),
          GEOMETRYPAIR::Constants::projection_xi_eta_tol);
    }
  }

  /**
   * Test intersections between a he8 element and a single line element
   */
  TEST_F(GeometryPairLineToVolumeSegmentationTest, TestHex8WithLine)
  {
    // Definition of variables for this test case.
    std::vector<Core::LinAlg::Matrix<12, 1, double>> q_line_elements;
    std::vector<double> line_ref_lengths;
    std::vector<Core::LinAlg::Matrix<24, 1, double>> q_volume_elements;
    std::vector<std::shared_ptr<GEOMETRYPAIR::GeometryPairLineToVolumeSegmentation<double,
        GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex8>>>
        geometry_pairs;

    // Get the geometry.
    xtest_create_geometry_single_hex8_with_pre_curved_line(
        line_elements_, volume_elements_, q_line_elements, line_ref_lengths, q_volume_elements);

    // Vector with vector of segments for Evaluate.
    std::vector<std::vector<GEOMETRYPAIR::LineSegment<double>>> segments_vector;

    // Create and evaluate the geometry pairs.
    create_evaluate_pairs(
        geometry_pairs, q_line_elements, line_ref_lengths, q_volume_elements, segments_vector);

    // Check results.
    {
      // Two segments should be found.
      EXPECT_EQ(segments_vector[0].size(), 2);

      // Check the segment coordinates on the line.
      EXPECT_NEAR(
          -1.0, segments_vector[0][0].get_eta_a(), GEOMETRYPAIR::Constants::projection_xi_eta_tol);
      EXPECT_NEAR(-0.9435487116990338, segments_vector[0][0].get_eta_b(),
          GEOMETRYPAIR::Constants::projection_xi_eta_tol);
      EXPECT_NEAR(-0.03868932051359714, segments_vector[0][1].get_eta_a(),
          GEOMETRYPAIR::Constants::projection_xi_eta_tol);
      EXPECT_NEAR(0.9822380322126314, segments_vector[0][1].get_eta_b(),
          GEOMETRYPAIR::Constants::projection_xi_eta_tol);
    }
  }

  /**
   * Test that non valid projected Gauss points can be handled correctly
   */
  TEST_F(GeometryPairLineToVolumeSegmentationTest, TestNonValidGaussPointsHex8)
  {
    // Definition of variables for this test case.
    std::vector<Core::LinAlg::Matrix<12, 1, double>> q_line_elements;
    std::vector<double> line_ref_lengths;
    std::vector<Core::LinAlg::Matrix<24, 1, double>> q_volume_elements;
    std::vector<std::shared_ptr<GEOMETRYPAIR::GeometryPairLineToVolumeSegmentation<double,
        GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex8>>>
        geometry_pairs;

    // Get the geometry.
    xtest_create_geometry_single_hex8_with_pre_curved_line(
        line_elements_, volume_elements_, q_line_elements, line_ref_lengths, q_volume_elements);

    // We change the default settings here, to run into the case where the obtained segment has
    // Gauss points that do not project valid
    Teuchos::ParameterList line_to_volume_params_list;
    set_up_default_parameters_line_to_3d(line_to_volume_params_list);
    line_to_volume_params_list.set("GEOMETRY_PAIR_SEGMENTATION_SEARCH_POINTS", 2);
    line_to_volume_params_list.set(
        "GEOMETRY_PAIR_SEGMENTATION_NOT_ALL_GAUSS_POINTS_PROJECT_VALID_ACTION",
        Inpar::GEOMETRYPAIR::NotAllGaussPointsProjectValidAction::warning);
    evaluation_data_ =
        std::make_shared<GEOMETRYPAIR::LineTo3DEvaluationData>(line_to_volume_params_list);

    // Vector with vector of segments for Evaluate.
    std::vector<std::vector<GEOMETRYPAIR::LineSegment<double>>> segments_vector;

    // Create and evaluate the geometry pairs.
    create_evaluate_pairs(
        geometry_pairs, q_line_elements, line_ref_lengths, q_volume_elements, segments_vector);

    // Check results.
    {
      EXPECT_EQ(segments_vector[0].size(), 1);
      EXPECT_NEAR(
          -1.0, segments_vector[0][0].get_eta_a(), GEOMETRYPAIR::Constants::projection_xi_eta_tol);
      EXPECT_NEAR(0.9822380322126314, segments_vector[0][0].get_eta_b(),
          GEOMETRYPAIR::Constants::projection_xi_eta_tol);
    }
  }

  /**
   * Test that non valid projected Gauss points throw an error
   */
  TEST_F(GeometryPairLineToVolumeSegmentationTest, TestNonValidGaussPointsThrowHex8)
  {
    // Definition of variables for this test case.
    std::vector<Core::LinAlg::Matrix<12, 1, double>> q_line_elements;
    std::vector<double> line_ref_lengths;
    std::vector<Core::LinAlg::Matrix<24, 1, double>> q_volume_elements;
    std::vector<std::shared_ptr<GEOMETRYPAIR::GeometryPairLineToVolumeSegmentation<double,
        GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex8>>>
        geometry_pairs;

    // Get the geometry.
    xtest_create_geometry_single_hex8_with_pre_curved_line(
        line_elements_, volume_elements_, q_line_elements, line_ref_lengths, q_volume_elements);

    // We change the default settings here, to run into the case where the obtained segment has
    // Gauss points that do not project valid
    Teuchos::ParameterList line_to_volume_params_list;
    set_up_default_parameters_line_to_3d(line_to_volume_params_list);
    line_to_volume_params_list.set("GEOMETRY_PAIR_SEGMENTATION_SEARCH_POINTS", 2);
    evaluation_data_ =
        std::make_shared<GEOMETRYPAIR::LineTo3DEvaluationData>(line_to_volume_params_list);

    // Vector with vector of segments for Evaluate.
    std::vector<std::vector<GEOMETRYPAIR::LineSegment<double>>> segments_vector;

    // Create and evaluate the geometry pairs.
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(create_evaluate_pairs(geometry_pairs, q_line_elements,
                                         line_ref_lengths, q_volume_elements, segments_vector),
        Core::Exception, "Error when projecting the Gauss points.");
  }
}  // namespace
