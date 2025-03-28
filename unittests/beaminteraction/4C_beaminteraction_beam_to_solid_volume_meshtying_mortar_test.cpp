// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_beam3_reissner.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_mortar.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_line_to_3D_evaluation_data.hpp"
#include "4C_geometry_pair_line_to_volume_segmentation.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_solid_3D_ele.hpp"



namespace
{
  using namespace FourC;


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
   * Class to test the local mortar matrices calculated by the beam to volume mesh tying mortar
   * pair.
   */
  class BeamToSolidVolumeMeshtyingPairMortarTest : public ::testing::Test
  {
   protected:
    /**
     * \brief Set up the testing environment.
     */
    BeamToSolidVolumeMeshtyingPairMortarTest()
    {
      // Set up the evaluation data container for the geometry pairs.
      Teuchos::ParameterList line_to_volume_params_list;
      set_up_default_parameters_line_to_3d(line_to_volume_params_list);
      evaluation_data_ =
          std::make_shared<GEOMETRYPAIR::LineTo3DEvaluationData>(line_to_volume_params_list);
    }

    /**
     * \brief Set up the contact pair so it can be evaluated and compare the results.
     */
    template <typename BeamType, typename SolidType, typename LambdaType>
    void perform_mortar_pair_unit_test(
        BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<BeamType, SolidType, LambdaType>&
            contact_pair,
        const GEOMETRYPAIR::ElementData<BeamType, double>& q_beam,
        const Core::LinAlg::Matrix<9, 1, double>& q_beam_rot,
        const GEOMETRYPAIR::ElementData<SolidType, double>& q_solid,
        const Core::LinAlg::Matrix<LambdaType::n_dof_, BeamType::n_dof_, double>& result_local_D,
        const Core::LinAlg::Matrix<LambdaType::n_dof_, SolidType::n_dof_, double>& result_local_M,
        const Core::LinAlg::Matrix<LambdaType::n_dof_, 1, double>& result_local_kappa)
    {
      // Create the elements.
      const int dummy_node_ids[2] = {0, 1};
      std::shared_ptr<Core::Elements::Element> beam_element =
          std::make_shared<Discret::Elements::Beam3r>(0, 0);
      beam_element->set_node_ids(2, dummy_node_ids);
      std::shared_ptr<Core::Elements::Element> solid_element =
          std::make_shared<Discret::Elements::Solid>(1, 0);

      // Set up the beam element.
      std::vector<double> xrefe(6);
      for (unsigned int j = 0; j < 2; j++)
        for (unsigned int i = 0; i < 3; i++) xrefe[i + 3 * j] = q_beam.element_position_(i + j * 6);

      // Get the rotational vector.
      std::vector<double> rotrefe(9);
      for (unsigned int i = 0; i < 9; i++) rotrefe[i] = q_beam_rot(i);

      // Cast beam element and set the hermitian interpolation.
      std::shared_ptr<Discret::Elements::Beam3r> beam_element_cast =
          std::dynamic_pointer_cast<Discret::Elements::Beam3r>(beam_element);
      beam_element_cast->set_centerline_hermite(true);
      beam_element_cast->set_up_reference_geometry<3, 2, 2>(xrefe, rotrefe);

      // Call Init on the beam contact pair.
      std::vector<const Core::Elements::Element*> pair_elements;
      pair_elements.push_back(&(*beam_element));
      pair_elements.push_back(&(*solid_element));
      contact_pair.create_geometry_pair(pair_elements[0], pair_elements[1], evaluation_data_);
      contact_pair.init(nullptr, pair_elements);

      // Evaluate the local matrices.
      Core::LinAlg::Matrix<LambdaType::n_dof_, BeamType::n_dof_, double> local_D(
          Core::LinAlg::Initialization::uninitialized);
      Core::LinAlg::Matrix<LambdaType::n_dof_, SolidType::n_dof_, double> local_M(
          Core::LinAlg::Initialization::uninitialized);
      Core::LinAlg::Matrix<LambdaType::n_dof_, 1, double> local_kappa(
          Core::LinAlg::Initialization::uninitialized);
      Core::LinAlg::Matrix<LambdaType::n_dof_, 1, double> local_constraint(
          Core::LinAlg::Initialization::uninitialized);
      contact_pair.ele1posref_ = q_beam;
      contact_pair.ele1pos_.shape_function_data_ = q_beam.shape_function_data_;
      contact_pair.ele2posref_ = q_solid;
      contact_pair.ele2pos_.shape_function_data_ = q_solid.shape_function_data_;
      contact_pair.cast_geometry_pair()->evaluate(
          contact_pair.ele1posref_, contact_pair.ele2posref_, contact_pair.line_to_3D_segments_);
      contact_pair.evaluate_dm(local_D, local_M, local_kappa, local_constraint);

      // Check the results for D.
      for (unsigned int i_row = 0; i_row < LambdaType::n_dof_; i_row++)
        for (unsigned int i_col = 0; i_col < BeamType::n_dof_; i_col++)
          EXPECT_NEAR(local_D(i_row, i_col), result_local_D(i_row, i_col), 1e-11);

      // Check the results for M.
      for (unsigned int i_row = 0; i_row < LambdaType::n_dof_; i_row++)
        for (unsigned int i_col = 0; i_col < SolidType::n_dof_; i_col++)
          EXPECT_NEAR(local_M(i_row, i_col), result_local_M(i_row, i_col), 1e-11);

      // Check the results for kappa.
      for (unsigned int i_row = 0; i_row < LambdaType::n_dof_; i_row++)
        EXPECT_NEAR(local_kappa(i_row), result_local_kappa(i_row), 1e-11);

      // Check the results for the local constraint offset vector.
      for (unsigned int i_row = 0; i_row < LambdaType::n_dof_; i_row++)
        EXPECT_NEAR(local_constraint(i_row), 0.0, 1e-11);
    }


    //! Evaluation data container for geometry pairs.
    std::shared_ptr<GEOMETRYPAIR::LineTo3DEvaluationData> evaluation_data_;
  };

  /**
   * \brief Test a non straight beam in a hex8 element, with line2 mortar shape functions.
   */
  TEST_F(BeamToSolidVolumeMeshtyingPairMortarTest, TestBeamToSolidMeshtyingMortarHermite2Hex8Line2)
  {
    // Element types.
    typedef GEOMETRYPAIR::t_hermite beam_type;
    typedef GEOMETRYPAIR::t_hex8 solid_type;
    typedef GEOMETRYPAIR::t_line2 lambda_type;

    // Create the mesh tying mortar pair.
    BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type, lambda_type>
        contact_pair = BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type,
            lambda_type>();

    // Definition of variables for this test case.
    GEOMETRYPAIR::ElementData<beam_type, double> q_beam;
    GEOMETRYPAIR::ElementData<solid_type, double> q_solid;
    Core::LinAlg::Matrix<9, 1> q_beam_rot;
    Core::LinAlg::SerialDenseMatrix local_D;
    Core::LinAlg::SerialDenseMatrix local_M;
    Core::LinAlg::SerialDenseVector local_kappa;

    // Matrices for the results.
    Core::LinAlg::Matrix<lambda_type::n_dof_, beam_type::n_dof_> result_local_D(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, solid_type::n_dof_> result_local_M(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, 1> result_local_kappa(
        Core::LinAlg::Initialization::zero);

    // Define the geometry of the two elements.
    q_beam.shape_function_data_.ref_length_ = 0.6192043571449604711;
    q_beam.element_position_(0) = 0.15;
    q_beam.element_position_(1) = 0.2;
    q_beam.element_position_(2) = 0.3;
    q_beam.element_position_(3) = 0.5773502691896255;
    q_beam.element_position_(4) = 0.5773502691896258;
    q_beam.element_position_(5) = 0.577350269189626;
    q_beam.element_position_(6) = 0.65;
    q_beam.element_position_(7) = 0.1;
    q_beam.element_position_(8) = 0.1;
    q_beam.element_position_(9) = 0.8017837257372733;
    q_beam.element_position_(10) = -0.5345224838248488;
    q_beam.element_position_(11) = 0.2672612419124244;

    q_beam_rot(0) = 1.674352746442651;
    q_beam_rot(1) = 0.1425949677148126;
    q_beam_rot(2) = 1.0831163124736984;
    q_beam_rot(3) = 1.4331999091513161;
    q_beam_rot(4) = -0.6560404572957742;
    q_beam_rot(5) = -0.2491376152457331;
    q_beam_rot(6) = 0.0;
    q_beam_rot(7) = 0.0;
    q_beam_rot(8) = 0.0;

    // Positional DOFs of the solid.
    q_solid.element_position_(0) = -0.954193516126594;
    q_solid.element_position_(1) = -0.975482672114534;
    q_solid.element_position_(2) = -1.00311316662815;
    q_solid.element_position_(3) = 0.923762409575691;
    q_solid.element_position_(4) = -1.010615727466753;
    q_solid.element_position_(5) = -1.014160992102648;
    q_solid.element_position_(6) = 0.905563488894572;
    q_solid.element_position_(7) = 1.069973754602123;
    q_solid.element_position_(8) = -0.935488788632622;
    q_solid.element_position_(9) = -1.047838992508937;
    q_solid.element_position_(10) = 1.08888017769104;
    q_solid.element_position_(11) = -1.036911970151991;
    q_solid.element_position_(12) = -1.089167378981223;
    q_solid.element_position_(13) = -1.063820485514786;
    q_solid.element_position_(14) = 1.081000671652452;
    q_solid.element_position_(15) = 0.972887477248942;
    q_solid.element_position_(16) = -1.008995039455167;
    q_solid.element_position_(17) = 0.923459231565508;
    q_solid.element_position_(18) = 1.089192110164642;
    q_solid.element_position_(19) = 1.026798931538616;
    q_solid.element_position_(20) = 0.964761788314679;
    q_solid.element_position_(21) = -0.941489453299989;
    q_solid.element_position_(22) = 0.954327975326486;
    q_solid.element_position_(23) = 0.963168067937965;

    // Results for D.
    result_local_D(0, 0) = 0.1954215935447702;
    result_local_D(0, 3) = 0.01819251719410916;
    result_local_D(0, 6) = 0.0989414627904878;
    result_local_D(0, 9) = -0.01347747062937554;
    result_local_D(1, 1) = 0.1954215935447702;
    result_local_D(1, 4) = 0.01819251719410916;
    result_local_D(1, 7) = 0.0989414627904878;
    result_local_D(1, 10) = -0.01347747062937554;
    result_local_D(2, 2) = 0.1954215935447702;
    result_local_D(2, 5) = 0.01819251719410916;
    result_local_D(2, 8) = 0.0989414627904878;
    result_local_D(2, 11) = -0.01347747062937554;
    result_local_D(3, 0) = 0.0946778368171451;
    result_local_D(3, 3) = 0.01347747062937554;
    result_local_D(3, 6) = 0.2300774860559777;
    result_local_D(3, 9) = -0.02083257297403942;
    result_local_D(4, 1) = 0.0946778368171451;
    result_local_D(4, 4) = 0.01347747062937554;
    result_local_D(4, 7) = 0.2300774860559777;
    result_local_D(4, 10) = -0.02083257297403942;
    result_local_D(5, 2) = 0.0946778368171451;
    result_local_D(5, 5) = 0.01347747062937554;
    result_local_D(5, 8) = 0.2300774860559777;
    result_local_D(5, 11) = -0.02083257297403942;

    // Results for M.
    result_local_M(0, 0) = 0.01401807776555987;
    result_local_M(0, 3) = 0.02820630234659772;
    result_local_M(0, 6) = 0.04334313657153085;
    result_local_M(0, 9) = 0.02180811652516679;
    result_local_M(0, 12) = 0.02499407921495017;
    result_local_M(0, 15) = 0.048221643722761;
    result_local_M(0, 18) = 0.07468795643554835;
    result_local_M(0, 21) = 0.03908374375314327;
    result_local_M(1, 1) = 0.01401807776555987;
    result_local_M(1, 4) = 0.02820630234659772;
    result_local_M(1, 7) = 0.04334313657153085;
    result_local_M(1, 10) = 0.02180811652516679;
    result_local_M(1, 13) = 0.02499407921495017;
    result_local_M(1, 16) = 0.048221643722761;
    result_local_M(1, 19) = 0.07468795643554835;
    result_local_M(1, 22) = 0.03908374375314327;
    result_local_M(2, 2) = 0.01401807776555987;
    result_local_M(2, 5) = 0.02820630234659772;
    result_local_M(2, 8) = 0.04334313657153085;
    result_local_M(2, 11) = 0.02180811652516679;
    result_local_M(2, 14) = 0.02499407921495017;
    result_local_M(2, 17) = 0.048221643722761;
    result_local_M(2, 20) = 0.07468795643554835;
    result_local_M(2, 23) = 0.03908374375314327;
    result_local_M(3, 0) = 0.01374032046076183;
    result_local_M(3, 3) = 0.04167222214705426;
    result_local_M(3, 6) = 0.05814720173702535;
    result_local_M(3, 9) = 0.01976961944249095;
    result_local_M(3, 12) = 0.02025971859166802;
    result_local_M(3, 15) = 0.05866143799154906;
    result_local_M(3, 18) = 0.0829388844532102;
    result_local_M(3, 21) = 0.02956591804936312;
    result_local_M(4, 1) = 0.01374032046076183;
    result_local_M(4, 4) = 0.04167222214705426;
    result_local_M(4, 7) = 0.05814720173702535;
    result_local_M(4, 10) = 0.01976961944249095;
    result_local_M(4, 13) = 0.02025971859166802;
    result_local_M(4, 16) = 0.05866143799154906;
    result_local_M(4, 19) = 0.0829388844532102;
    result_local_M(4, 22) = 0.02956591804936312;
    result_local_M(5, 2) = 0.01374032046076183;
    result_local_M(5, 5) = 0.04167222214705426;
    result_local_M(5, 8) = 0.05814720173702535;
    result_local_M(5, 11) = 0.01976961944249095;
    result_local_M(5, 14) = 0.02025971859166802;
    result_local_M(5, 17) = 0.05866143799154906;
    result_local_M(5, 20) = 0.0829388844532102;
    result_local_M(5, 23) = 0.02956591804936312;

    // Results for Kappa.
    result_local_kappa(0) = 0.294363056335258;
    result_local_kappa(1) = 0.294363056335258;
    result_local_kappa(2) = 0.294363056335258;
    result_local_kappa(3) = 0.3247553228731228;
    result_local_kappa(4) = 0.3247553228731228;
    result_local_kappa(5) = 0.3247553228731228;

    // Perform the unit tests.
    perform_mortar_pair_unit_test(contact_pair, q_beam, q_beam_rot, q_solid, result_local_D,
        result_local_M, result_local_kappa);
  }

  /**
   * \brief Test a non straight beam in a hex8 element, with line3 mortar shape functions.
   */
  TEST_F(BeamToSolidVolumeMeshtyingPairMortarTest, TestBeamToSolidMeshtyingMortarHermite2Hex8Line3)
  {
    // Element types.
    typedef GEOMETRYPAIR::t_hermite beam_type;
    typedef GEOMETRYPAIR::t_hex8 solid_type;
    typedef GEOMETRYPAIR::t_line3 lambda_type;

    // Create the mesh tying mortar pair.
    BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type, lambda_type>
        contact_pair = BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type,
            lambda_type>();

    // Definition of variables for this test case.
    GEOMETRYPAIR::ElementData<beam_type, double> q_beam;
    GEOMETRYPAIR::ElementData<solid_type, double> q_solid;
    Core::LinAlg::Matrix<9, 1> q_beam_rot;
    Core::LinAlg::SerialDenseMatrix local_D;
    Core::LinAlg::SerialDenseMatrix local_M;
    Core::LinAlg::SerialDenseVector local_kappa;

    // Matrices for the results.
    Core::LinAlg::Matrix<lambda_type::n_dof_, beam_type::n_dof_> result_local_D(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, solid_type::n_dof_> result_local_M(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, 1> result_local_kappa(
        Core::LinAlg::Initialization::zero);

    // Define the geometry of the two elements.
    q_beam.shape_function_data_.ref_length_ = 0.6192043571449604711;
    q_beam.element_position_(0) = 0.15;
    q_beam.element_position_(1) = 0.2;
    q_beam.element_position_(2) = 0.3;
    q_beam.element_position_(3) = 0.5773502691896255;
    q_beam.element_position_(4) = 0.5773502691896258;
    q_beam.element_position_(5) = 0.577350269189626;
    q_beam.element_position_(6) = 0.65;
    q_beam.element_position_(7) = 0.1;
    q_beam.element_position_(8) = 0.1;
    q_beam.element_position_(9) = 0.8017837257372733;
    q_beam.element_position_(10) = -0.5345224838248488;
    q_beam.element_position_(11) = 0.2672612419124244;

    q_beam_rot(0) = 1.674352746442651;
    q_beam_rot(1) = 0.1425949677148126;
    q_beam_rot(2) = 1.0831163124736984;
    q_beam_rot(3) = 1.4331999091513161;
    q_beam_rot(4) = -0.6560404572957742;
    q_beam_rot(5) = -0.2491376152457331;
    q_beam_rot(6) = 0.0;
    q_beam_rot(7) = 0.0;
    q_beam_rot(8) = 0.0;

    // Positional DOFs of the solid.
    q_solid.element_position_(0) = -0.954193516126594;
    q_solid.element_position_(1) = -0.975482672114534;
    q_solid.element_position_(2) = -1.00311316662815;
    q_solid.element_position_(3) = 0.923762409575691;
    q_solid.element_position_(4) = -1.010615727466753;
    q_solid.element_position_(5) = -1.014160992102648;
    q_solid.element_position_(6) = 0.905563488894572;
    q_solid.element_position_(7) = 1.069973754602123;
    q_solid.element_position_(8) = -0.935488788632622;
    q_solid.element_position_(9) = -1.047838992508937;
    q_solid.element_position_(10) = 1.08888017769104;
    q_solid.element_position_(11) = -1.036911970151991;
    q_solid.element_position_(12) = -1.089167378981223;
    q_solid.element_position_(13) = -1.063820485514786;
    q_solid.element_position_(14) = 1.081000671652452;
    q_solid.element_position_(15) = 0.972887477248942;
    q_solid.element_position_(16) = -1.008995039455167;
    q_solid.element_position_(17) = 0.923459231565508;
    q_solid.element_position_(18) = 1.089192110164642;
    q_solid.element_position_(19) = 1.026798931538616;
    q_solid.element_position_(20) = 0.964761788314679;
    q_solid.element_position_(21) = -0.941489453299989;
    q_solid.element_position_(22) = 0.954327975326486;
    q_solid.element_position_(23) = 0.963168067937965;

    // Results for D.
    result_local_D(0, 0) = 0.0945173766014223;
    result_local_D(0, 3) = 0.005144868995407634;
    result_local_D(0, 6) = -0.0132666172503834;
    result_local_D(0, 9) = 0.0004298224306740067;
    result_local_D(1, 1) = 0.0945173766014223;
    result_local_D(1, 4) = 0.005144868995407634;
    result_local_D(1, 7) = -0.0132666172503834;
    result_local_D(1, 10) = 0.0004298224306740067;
    result_local_D(2, 2) = 0.0945173766014223;
    result_local_D(2, 5) = 0.005144868995407634;
    result_local_D(2, 8) = -0.0132666172503834;
    result_local_D(2, 11) = 0.0004298224306740067;
    result_local_D(3, 0) = -0.006226380126202878;
    result_local_D(3, 3) = 0.0004298224306740081;
    result_local_D(3, 6) = 0.1178694060151065;
    result_local_D(3, 9) = -0.006925279913989872;
    result_local_D(4, 1) = -0.006226380126202878;
    result_local_D(4, 4) = 0.0004298224306740081;
    result_local_D(4, 7) = 0.1178694060151065;
    result_local_D(4, 10) = -0.006925279913989872;
    result_local_D(5, 2) = -0.006226380126202878;
    result_local_D(5, 5) = 0.0004298224306740081;
    result_local_D(5, 8) = 0.1178694060151065;
    result_local_D(5, 11) = -0.006925279913989872;
    result_local_D(6, 0) = 0.2018084338866959;
    result_local_D(6, 3) = 0.02609529639740305;
    result_local_D(6, 6) = 0.2244161600817424;
    result_local_D(6, 9) = -0.02781458612009909;
    result_local_D(7, 1) = 0.2018084338866959;
    result_local_D(7, 4) = 0.02609529639740305;
    result_local_D(7, 7) = 0.2244161600817424;
    result_local_D(7, 10) = -0.02781458612009909;
    result_local_D(8, 2) = 0.2018084338866959;
    result_local_D(8, 5) = 0.02609529639740305;
    result_local_D(8, 8) = 0.2244161600817424;
    result_local_D(8, 11) = -0.02781458612009909;

    // Results for M.
    result_local_M(0, 0) = 0.004453675076985303;
    result_local_M(0, 3) = 0.004861262560233103;
    result_local_M(0, 6) = 0.00843432104202533;
    result_local_M(0, 9) = 0.007216752579710933;
    result_local_M(0, 12) = 0.00945658949498672;
    result_local_M(0, 15) = 0.01190337131970094;
    result_local_M(0, 18) = 0.01977316716623557;
    result_local_M(0, 21) = 0.01515162011116097;
    result_local_M(1, 1) = 0.004453675076985303;
    result_local_M(1, 4) = 0.004861262560233103;
    result_local_M(1, 7) = 0.00843432104202533;
    result_local_M(1, 10) = 0.007216752579710933;
    result_local_M(1, 13) = 0.00945658949498672;
    result_local_M(1, 16) = 0.01190337131970094;
    result_local_M(1, 19) = 0.01977316716623557;
    result_local_M(1, 22) = 0.01515162011116097;
    result_local_M(2, 2) = 0.004453675076985303;
    result_local_M(2, 5) = 0.004861262560233103;
    result_local_M(2, 8) = 0.00843432104202533;
    result_local_M(2, 11) = 0.007216752579710933;
    result_local_M(2, 14) = 0.00945658949498672;
    result_local_M(2, 17) = 0.01190337131970094;
    result_local_M(2, 20) = 0.01977316716623557;
    result_local_M(2, 23) = 0.01515162011116097;
    result_local_M(3, 0) = 0.004175917772187271;
    result_local_M(3, 3) = 0.01832718236068964;
    result_local_M(3, 6) = 0.02323838620751983;
    result_local_M(3, 9) = 0.00517825549703509;
    result_local_M(3, 12) = 0.004722228871704564;
    result_local_M(3, 15) = 0.02234316558848898;
    result_local_M(3, 18) = 0.02802409518389743;
    result_local_M(3, 21) = 0.005633794407380818;
    result_local_M(4, 1) = 0.004175917772187271;
    result_local_M(4, 4) = 0.01832718236068964;
    result_local_M(4, 7) = 0.02323838620751983;
    result_local_M(4, 10) = 0.00517825549703509;
    result_local_M(4, 13) = 0.004722228871704564;
    result_local_M(4, 16) = 0.02234316558848898;
    result_local_M(4, 19) = 0.02802409518389743;
    result_local_M(4, 22) = 0.005633794407380818;
    result_local_M(5, 2) = 0.004175917772187271;
    result_local_M(5, 5) = 0.01832718236068964;
    result_local_M(5, 8) = 0.02323838620751983;
    result_local_M(5, 11) = 0.00517825549703509;
    result_local_M(5, 14) = 0.004722228871704564;
    result_local_M(5, 17) = 0.02234316558848898;
    result_local_M(5, 20) = 0.02802409518389743;
    result_local_M(5, 23) = 0.005633794407380818;
    result_local_M(6, 0) = 0.01912880537714913;
    result_local_M(6, 3) = 0.04669007957272923;
    result_local_M(6, 6) = 0.06981763105901104;
    result_local_M(6, 9) = 0.02918272789091171;
    result_local_M(6, 12) = 0.03107497943992692;
    result_local_M(6, 15) = 0.07263654480612013;
    result_local_M(6, 18) = 0.1098295785386256;
    result_local_M(6, 21) = 0.0478642472839646;
    result_local_M(7, 1) = 0.01912880537714913;
    result_local_M(7, 4) = 0.04669007957272923;
    result_local_M(7, 7) = 0.06981763105901104;
    result_local_M(7, 10) = 0.02918272789091171;
    result_local_M(7, 13) = 0.03107497943992692;
    result_local_M(7, 16) = 0.07263654480612013;
    result_local_M(7, 19) = 0.1098295785386256;
    result_local_M(7, 22) = 0.0478642472839646;
    result_local_M(8, 2) = 0.01912880537714913;
    result_local_M(8, 5) = 0.04669007957272923;
    result_local_M(8, 8) = 0.06981763105901104;
    result_local_M(8, 11) = 0.02918272789091171;
    result_local_M(8, 14) = 0.03107497943992692;
    result_local_M(8, 17) = 0.07263654480612013;
    result_local_M(8, 20) = 0.1098295785386256;
    result_local_M(8, 23) = 0.0478642472839646;

    // Results for Kappa.
    result_local_kappa(0) = 0.0812507593510389;
    result_local_kappa(1) = 0.0812507593510389;
    result_local_kappa(2) = 0.0812507593510389;
    result_local_kappa(3) = 0.1116430258889036;
    result_local_kappa(4) = 0.1116430258889036;
    result_local_kappa(5) = 0.1116430258889036;
    result_local_kappa(6) = 0.4262245939684383;
    result_local_kappa(7) = 0.4262245939684383;
    result_local_kappa(8) = 0.4262245939684383;

    // Perform the unit tests.
    perform_mortar_pair_unit_test(contact_pair, q_beam, q_beam_rot, q_solid, result_local_D,
        result_local_M, result_local_kappa);
  }

  /**
   * \brief Test a non straight beam in a hex8 element, with line4 mortar shape functions.
   */
  TEST_F(BeamToSolidVolumeMeshtyingPairMortarTest, TestBeamToSolidMeshtyingMortarHermite2Hex8Line4)
  {
    // Element types.
    typedef GEOMETRYPAIR::t_hermite beam_type;
    typedef GEOMETRYPAIR::t_hex8 solid_type;
    typedef GEOMETRYPAIR::t_line4 lambda_type;

    // Create the mesh tying mortar pair.
    BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type, lambda_type>
        contact_pair = BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type,
            lambda_type>();

    // Definition of variables for this test case.
    GEOMETRYPAIR::ElementData<beam_type, double> q_beam;
    GEOMETRYPAIR::ElementData<solid_type, double> q_solid;
    Core::LinAlg::Matrix<9, 1> q_beam_rot;
    Core::LinAlg::SerialDenseMatrix local_D;
    Core::LinAlg::SerialDenseMatrix local_M;
    Core::LinAlg::SerialDenseVector local_kappa;

    // Matrices for the results.
    Core::LinAlg::Matrix<lambda_type::n_dof_, beam_type::n_dof_> result_local_D(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, solid_type::n_dof_> result_local_M(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, 1> result_local_kappa(
        Core::LinAlg::Initialization::zero);

    // Define the geometry of the two elements.
    q_beam.shape_function_data_.ref_length_ = 0.6192043571449604711;
    q_beam.element_position_(0) = 0.15;
    q_beam.element_position_(1) = 0.2;
    q_beam.element_position_(2) = 0.3;
    q_beam.element_position_(3) = 0.5773502691896255;
    q_beam.element_position_(4) = 0.5773502691896258;
    q_beam.element_position_(5) = 0.577350269189626;
    q_beam.element_position_(6) = 0.65;
    q_beam.element_position_(7) = 0.1;
    q_beam.element_position_(8) = 0.1;
    q_beam.element_position_(9) = 0.8017837257372733;
    q_beam.element_position_(10) = -0.5345224838248488;
    q_beam.element_position_(11) = 0.2672612419124244;

    q_beam_rot(0) = 1.674352746442651;
    q_beam_rot(1) = 0.1425949677148126;
    q_beam_rot(2) = 1.0831163124736984;
    q_beam_rot(3) = 1.4331999091513161;
    q_beam_rot(4) = -0.6560404572957742;
    q_beam_rot(5) = -0.2491376152457331;
    q_beam_rot(6) = 0.0;
    q_beam_rot(7) = 0.0;
    q_beam_rot(8) = 0.0;

    // Positional DOFs of the solid.
    q_solid.element_position_(0) = -0.954193516126594;
    q_solid.element_position_(1) = -0.975482672114534;
    q_solid.element_position_(2) = -1.00311316662815;
    q_solid.element_position_(3) = 0.923762409575691;
    q_solid.element_position_(4) = -1.010615727466753;
    q_solid.element_position_(5) = -1.014160992102648;
    q_solid.element_position_(6) = 0.905563488894572;
    q_solid.element_position_(7) = 1.069973754602123;
    q_solid.element_position_(8) = -0.935488788632622;
    q_solid.element_position_(9) = -1.047838992508937;
    q_solid.element_position_(10) = 1.08888017769104;
    q_solid.element_position_(11) = -1.036911970151991;
    q_solid.element_position_(12) = -1.089167378981223;
    q_solid.element_position_(13) = -1.063820485514786;
    q_solid.element_position_(14) = 1.081000671652452;
    q_solid.element_position_(15) = 0.972887477248942;
    q_solid.element_position_(16) = -1.008995039455167;
    q_solid.element_position_(17) = 0.923459231565508;
    q_solid.element_position_(18) = 1.089192110164642;
    q_solid.element_position_(19) = 1.026798931538616;
    q_solid.element_position_(20) = 0.964761788314679;
    q_solid.element_position_(21) = -0.941489453299989;
    q_solid.element_position_(22) = 0.954327975326486;
    q_solid.element_position_(23) = 0.963168067937965;

    // Results for D.
    result_local_D(0, 0) = 0.05837517950584759;
    result_local_D(0, 3) = 0.002017208305318981;
    result_local_D(0, 6) = 0.005829701162185071;
    result_local_D(0, 9) = -0.0002955710710872768;
    result_local_D(1, 1) = 0.05837517950584759;
    result_local_D(1, 4) = 0.002017208305318981;
    result_local_D(1, 7) = 0.005829701162185071;
    result_local_D(1, 10) = -0.0002955710710872768;
    result_local_D(2, 2) = 0.05837517950584759;
    result_local_D(2, 5) = 0.002017208305318981;
    result_local_D(2, 8) = 0.005829701162185071;
    result_local_D(2, 11) = -0.0002955710710872768;
    result_local_D(3, 0) = 0.004689762733534804;
    result_local_D(3, 3) = 0.0002955710710872767;
    result_local_D(3, 6) = 0.07072106759232025;
    result_local_D(3, 9) = -0.002723063147216203;
    result_local_D(4, 1) = 0.004689762733534804;
    result_local_D(4, 4) = 0.0002955710710872767;
    result_local_D(4, 7) = 0.07072106759232025;
    result_local_D(4, 10) = -0.002723063147216203;
    result_local_D(5, 2) = 0.004689762733534804;
    result_local_D(5, 5) = 0.0002955710710872767;
    result_local_D(5, 8) = 0.07072106759232025;
    result_local_D(5, 11) = -0.002723063147216203;
    result_local_D(6, 0) = 0.184104753994235;
    result_local_D(6, 3) = 0.0191687182192921;
    result_local_D(6, 6) = 0.02686710479294802;
    result_local_D(6, 9) = -0.00825428928975331;
    result_local_D(7, 1) = 0.184104753994235;
    result_local_D(7, 4) = 0.0191687182192921;
    result_local_D(7, 7) = 0.02686710479294802;
    result_local_D(7, 10) = -0.00825428928975331;
    result_local_D(8, 2) = 0.184104753994235;
    result_local_D(8, 5) = 0.0191687182192921;
    result_local_D(8, 8) = 0.02686710479294802;
    result_local_D(8, 11) = -0.00825428928975331;
    result_local_D(9, 0) = 0.04292973412829792;
    result_local_D(9, 3) = 0.01018849022778634;
    result_local_D(9, 6) = 0.2256010752990122;
    result_local_D(9, 9) = -0.02303712009535817;
    result_local_D(10, 1) = 0.04292973412829792;
    result_local_D(10, 4) = 0.01018849022778634;
    result_local_D(10, 7) = 0.2256010752990122;
    result_local_D(10, 10) = -0.02303712009535817;
    result_local_D(11, 2) = 0.04292973412829792;
    result_local_D(11, 5) = 0.01018849022778634;
    result_local_D(11, 8) = 0.2256010752990122;
    result_local_D(11, 11) = -0.02303712009535817;

    // Results for M.
    result_local_M(0, 0) = 0.003390111296476561;
    result_local_M(0, 3) = 0.005426070011455455;
    result_local_M(0, 6) = 0.00807974885858712;
    result_local_M(0, 9) = 0.005161647312433724;
    result_local_M(0, 12) = 0.006606367843876448;
    result_local_M(0, 15) = 0.01013180548765833;
    result_local_M(0, 18) = 0.0152895800274908;
    result_local_M(0, 21) = 0.01011954983005423;
    result_local_M(1, 1) = 0.003390111296476561;
    result_local_M(1, 4) = 0.005426070011455455;
    result_local_M(1, 7) = 0.00807974885858712;
    result_local_M(1, 10) = 0.005161647312433724;
    result_local_M(1, 13) = 0.006606367843876448;
    result_local_M(1, 16) = 0.01013180548765833;
    result_local_M(1, 19) = 0.0152895800274908;
    result_local_M(1, 22) = 0.01011954983005423;
    result_local_M(2, 2) = 0.003390111296476561;
    result_local_M(2, 5) = 0.005426070011455455;
    result_local_M(2, 8) = 0.00807974885858712;
    result_local_M(2, 11) = 0.005161647312433724;
    result_local_M(2, 14) = 0.006606367843876448;
    result_local_M(2, 17) = 0.01013180548765833;
    result_local_M(2, 20) = 0.0152895800274908;
    result_local_M(2, 23) = 0.01011954983005423;
    result_local_M(3, 0) = 0.00284838088055237;
    result_local_M(3, 3) = 0.01192611496287614;
    result_local_M(3, 6) = 0.01486575450858167;
    result_local_M(3, 9) = 0.003585519777948337;
    result_local_M(3, 12) = 0.003688078092823968;
    result_local_M(3, 15) = 0.01503516331976658;
    result_local_M(3, 18) = 0.018778985005314;
    result_local_M(3, 21) = 0.004682833777991986;
    result_local_M(4, 1) = 0.00284838088055237;
    result_local_M(4, 4) = 0.01192611496287614;
    result_local_M(4, 7) = 0.01486575450858167;
    result_local_M(4, 10) = 0.003585519777948337;
    result_local_M(4, 13) = 0.003688078092823968;
    result_local_M(4, 16) = 0.01503516331976658;
    result_local_M(4, 19) = 0.018778985005314;
    result_local_M(4, 22) = 0.004682833777991986;
    result_local_M(5, 2) = 0.00284838088055237;
    result_local_M(5, 5) = 0.01192611496287614;
    result_local_M(5, 8) = 0.01486575450858167;
    result_local_M(5, 11) = 0.003585519777948337;
    result_local_M(5, 14) = 0.003688078092823968;
    result_local_M(5, 17) = 0.01503516331976658;
    result_local_M(5, 20) = 0.018778985005314;
    result_local_M(5, 23) = 0.004682833777991986;
    result_local_M(6, 0) = 0.01036399335795715;
    result_local_M(6, 3) = 0.01581435748610641;
    result_local_M(6, 6) = 0.02724532819744379;
    result_local_M(6, 9) = 0.01710883876092353;
    result_local_M(6, 12) = 0.0202037822433034;
    result_local_M(6, 15) = 0.03255340179842287;
    result_local_M(6, 18) = 0.05463685336821889;
    result_local_M(6, 21) = 0.03304530357480695;
    result_local_M(7, 1) = 0.01036399335795715;
    result_local_M(7, 4) = 0.01581435748610641;
    result_local_M(7, 7) = 0.02724532819744379;
    result_local_M(7, 10) = 0.01710883876092353;
    result_local_M(7, 13) = 0.0202037822433034;
    result_local_M(7, 16) = 0.03255340179842287;
    result_local_M(7, 19) = 0.05463685336821889;
    result_local_M(7, 22) = 0.03304530357480695;
    result_local_M(8, 2) = 0.01036399335795715;
    result_local_M(8, 5) = 0.01581435748610641;
    result_local_M(8, 8) = 0.02724532819744379;
    result_local_M(8, 11) = 0.01710883876092353;
    result_local_M(8, 14) = 0.0202037822433034;
    result_local_M(8, 17) = 0.03255340179842287;
    result_local_M(8, 20) = 0.05463685336821889;
    result_local_M(8, 23) = 0.03304530357480695;
    result_local_M(9, 0) = 0.01115591269133562;
    result_local_M(9, 3) = 0.03671198203321397;
    result_local_M(9, 6) = 0.05129950674394363;
    result_local_M(9, 9) = 0.01572173011635215;
    result_local_M(9, 12) = 0.01475556962661438;
    result_local_M(9, 15) = 0.04916271110846227;
    result_local_M(9, 18) = 0.06892142248773486;
    result_local_M(9, 21) = 0.02080197461965322;
    result_local_M(10, 1) = 0.01115591269133562;
    result_local_M(10, 4) = 0.03671198203321397;
    result_local_M(10, 7) = 0.05129950674394363;
    result_local_M(10, 10) = 0.01572173011635215;
    result_local_M(10, 13) = 0.01475556962661438;
    result_local_M(10, 16) = 0.04916271110846227;
    result_local_M(10, 19) = 0.06892142248773486;
    result_local_M(10, 22) = 0.02080197461965322;
    result_local_M(11, 2) = 0.01115591269133562;
    result_local_M(11, 5) = 0.03671198203321397;
    result_local_M(11, 8) = 0.05129950674394363;
    result_local_M(11, 11) = 0.01572173011635215;
    result_local_M(11, 14) = 0.01475556962661438;
    result_local_M(11, 17) = 0.04916271110846227;
    result_local_M(11, 20) = 0.06892142248773486;
    result_local_M(11, 23) = 0.02080197461965322;

    // Results for Kappa.
    result_local_kappa(0) = 0.06420488066803266;
    result_local_kappa(1) = 0.06420488066803266;
    result_local_kappa(2) = 0.06420488066803266;
    result_local_kappa(3) = 0.07541083032585504;
    result_local_kappa(4) = 0.07541083032585504;
    result_local_kappa(5) = 0.07541083032585504;
    result_local_kappa(6) = 0.210971858787183;
    result_local_kappa(7) = 0.210971858787183;
    result_local_kappa(8) = 0.210971858787183;
    result_local_kappa(9) = 0.2685308094273101;
    result_local_kappa(10) = 0.2685308094273101;
    result_local_kappa(11) = 0.2685308094273101;

    // Perform the unit tests.
    perform_mortar_pair_unit_test(contact_pair, q_beam, q_beam_rot, q_solid, result_local_D,
        result_local_M, result_local_kappa);
  }

  /**
   * \brief Test a non straight beam in a hex20 element, with line2 mortar shape functions.
   */
  TEST_F(BeamToSolidVolumeMeshtyingPairMortarTest, TestBeamToSolidMeshtyingMortarHermite2Hex20Line2)
  {
    // Element types.
    typedef GEOMETRYPAIR::t_hermite beam_type;
    typedef GEOMETRYPAIR::t_hex20 solid_type;
    typedef GEOMETRYPAIR::t_line2 lambda_type;

    // Create the mesh tying mortar pair.
    BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type, lambda_type>
        contact_pair = BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type,
            lambda_type>();

    // Definition of variables for this test case.
    GEOMETRYPAIR::ElementData<beam_type, double> q_beam;
    GEOMETRYPAIR::ElementData<solid_type, double> q_solid;
    Core::LinAlg::Matrix<9, 1> q_beam_rot;
    Core::LinAlg::SerialDenseMatrix local_D;
    Core::LinAlg::SerialDenseMatrix local_M;
    Core::LinAlg::SerialDenseVector local_kappa;

    // Matrices for the results.
    Core::LinAlg::Matrix<lambda_type::n_dof_, beam_type::n_dof_> result_local_D(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, solid_type::n_dof_> result_local_M(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, 1> result_local_kappa(
        Core::LinAlg::Initialization::zero);

    // Define the geometry of the two elements.
    q_beam.shape_function_data_.ref_length_ = 0.6192043571449604711;
    q_beam.element_position_(0) = 0.15;
    q_beam.element_position_(1) = 0.2;
    q_beam.element_position_(2) = 0.3;
    q_beam.element_position_(3) = 0.5773502691896255;
    q_beam.element_position_(4) = 0.5773502691896258;
    q_beam.element_position_(5) = 0.577350269189626;
    q_beam.element_position_(6) = 0.65;
    q_beam.element_position_(7) = 0.1;
    q_beam.element_position_(8) = 0.1;
    q_beam.element_position_(9) = 0.8017837257372733;
    q_beam.element_position_(10) = -0.5345224838248488;
    q_beam.element_position_(11) = 0.2672612419124244;

    q_beam_rot(0) = 1.674352746442651;
    q_beam_rot(1) = 0.1425949677148126;
    q_beam_rot(2) = 1.0831163124736984;
    q_beam_rot(3) = 1.4331999091513161;
    q_beam_rot(4) = -0.6560404572957742;
    q_beam_rot(5) = -0.2491376152457331;
    q_beam_rot(6) = 0.0;
    q_beam_rot(7) = 0.0;
    q_beam_rot(8) = 0.0;

    // Positional DOFs of the solid.
    q_solid.element_position_(0) = -0.954193516126594;
    q_solid.element_position_(1) = -0.975482672114534;
    q_solid.element_position_(2) = -1.00311316662815;
    q_solid.element_position_(3) = 0.923762409575691;
    q_solid.element_position_(4) = -1.010615727466753;
    q_solid.element_position_(5) = -1.014160992102648;
    q_solid.element_position_(6) = 0.905563488894572;
    q_solid.element_position_(7) = 1.069973754602123;
    q_solid.element_position_(8) = -0.935488788632622;
    q_solid.element_position_(9) = -1.047838992508937;
    q_solid.element_position_(10) = 1.08888017769104;
    q_solid.element_position_(11) = -1.036911970151991;
    q_solid.element_position_(12) = -1.089167378981223;
    q_solid.element_position_(13) = -1.063820485514786;
    q_solid.element_position_(14) = 1.081000671652452;
    q_solid.element_position_(15) = 0.972887477248942;
    q_solid.element_position_(16) = -1.008995039455167;
    q_solid.element_position_(17) = 0.923459231565508;
    q_solid.element_position_(18) = 1.089192110164642;
    q_solid.element_position_(19) = 1.026798931538616;
    q_solid.element_position_(20) = 0.964761788314679;
    q_solid.element_position_(21) = -0.941489453299989;
    q_solid.element_position_(22) = 0.954327975326486;
    q_solid.element_position_(23) = 0.963168067937965;
    q_solid.element_position_(24) = 0.0925155012591516;
    q_solid.element_position_(25) = -0.973383316878771;
    q_solid.element_position_(26) = -1.065800877068667;
    q_solid.element_position_(27) = 0.993751379524698;
    q_solid.element_position_(28) = -0.089078411148891;
    q_solid.element_position_(29) = -1.080589211553427;
    q_solid.element_position_(30) = 0.0823626211382376;
    q_solid.element_position_(31) = 1.033905181036807;
    q_solid.element_position_(32) = -0.952741283444911;
    q_solid.element_position_(33) = -1.024714325285187;
    q_solid.element_position_(34) = -0.0951797166281465;
    q_solid.element_position_(35) = -0.917694055170081;
    q_solid.element_position_(36) = -0.94955183268814;
    q_solid.element_position_(37) = -0.936209915275568;
    q_solid.element_position_(38) = -0.07531489653912402;
    q_solid.element_position_(39) = 0.977045420134747;
    q_solid.element_position_(40) = -0.915831507095275;
    q_solid.element_position_(41) = -0.03604489795829757;
    q_solid.element_position_(42) = 1.044996295051512;
    q_solid.element_position_(43) = 0.912268002554735;
    q_solid.element_position_(44) = 0.026884640088015;
    q_solid.element_position_(45) = -1.003774079332873;
    q_solid.element_position_(46) = 1.016364821141699;
    q_solid.element_position_(47) = -0.04547583423590318;
    q_solid.element_position_(48) = -0.07606458957105064;
    q_solid.element_position_(49) = -0.970767004162723;
    q_solid.element_position_(50) = 1.09197908262736;
    q_solid.element_position_(51) = 1.000582069178839;
    q_solid.element_position_(52) = 0.02495199075528687;
    q_solid.element_position_(53) = 0.942442567644017;
    q_solid.element_position_(54) = -0.0840408739597993;
    q_solid.element_position_(55) = 1.082175787959178;
    q_solid.element_position_(56) = 1.016887685734179;
    q_solid.element_position_(57) = -1.02950388437141;
    q_solid.element_position_(58) = -0.03189318909487643;
    q_solid.element_position_(59) = 0.962850907596607;

    // Results for D.
    result_local_D(0, 0) = 0.1954215935447702;
    result_local_D(0, 3) = 0.01819251719410916;
    result_local_D(0, 6) = 0.0989414627904878;
    result_local_D(0, 9) = -0.01347747062937554;
    result_local_D(1, 1) = 0.1954215935447702;
    result_local_D(1, 4) = 0.01819251719410916;
    result_local_D(1, 7) = 0.0989414627904878;
    result_local_D(1, 10) = -0.01347747062937554;
    result_local_D(2, 2) = 0.1954215935447702;
    result_local_D(2, 5) = 0.01819251719410916;
    result_local_D(2, 8) = 0.0989414627904878;
    result_local_D(2, 11) = -0.01347747062937554;
    result_local_D(3, 0) = 0.0946778368171451;
    result_local_D(3, 3) = 0.01347747062937554;
    result_local_D(3, 6) = 0.2300774860559777;
    result_local_D(3, 9) = -0.02083257297403942;
    result_local_D(4, 1) = 0.0946778368171451;
    result_local_D(4, 4) = 0.01347747062937554;
    result_local_D(4, 7) = 0.2300774860559777;
    result_local_D(4, 10) = -0.02083257297403942;
    result_local_D(5, 2) = 0.0946778368171451;
    result_local_D(5, 5) = 0.01347747062937554;
    result_local_D(5, 8) = 0.2300774860559777;
    result_local_D(5, 11) = -0.02083257297403942;

    // Results for M.
    result_local_M(0, 0) = -0.03991016878403244;
    result_local_M(0, 3) = -0.05903555807314021;
    result_local_M(0, 6) = -0.07421571099636162;
    result_local_M(0, 9) = -0.05323297222187619;
    result_local_M(0, 12) = -0.05618363930341419;
    result_local_M(0, 15) = -0.07724836533263818;
    result_local_M(0, 18) = -0.0891113900382783;
    result_local_M(0, 21) = -0.07125249054387619;
    result_local_M(0, 24) = 0.03698784242998645;
    result_local_M(0, 27) = 0.06716878812140782;
    result_local_M(0, 30) = 0.05898961370672467;
    result_local_M(0, 33) = 0.03509553737468334;
    result_local_M(0, 36) = 0.0362988303599596;
    result_local_M(0, 39) = 0.06864610258453398;
    result_local_M(0, 42) = 0.109326337819849;
    result_local_M(0, 45) = 0.05794382265618636;
    result_local_M(0, 48) = 0.06451772141630999;
    result_local_M(0, 51) = 0.1146044812669878;
    result_local_M(0, 54) = 0.1030869401682464;
    result_local_M(0, 57) = 0.06188733372399985;
    result_local_M(1, 1) = -0.03991016878403244;
    result_local_M(1, 4) = -0.05903555807314021;
    result_local_M(1, 7) = -0.07421571099636162;
    result_local_M(1, 10) = -0.05323297222187619;
    result_local_M(1, 13) = -0.05618363930341419;
    result_local_M(1, 16) = -0.07724836533263818;
    result_local_M(1, 19) = -0.0891113900382783;
    result_local_M(1, 22) = -0.07125249054387619;
    result_local_M(1, 25) = 0.03698784242998645;
    result_local_M(1, 28) = 0.06716878812140782;
    result_local_M(1, 31) = 0.05898961370672467;
    result_local_M(1, 34) = 0.03509553737468334;
    result_local_M(1, 37) = 0.0362988303599596;
    result_local_M(1, 40) = 0.06864610258453398;
    result_local_M(1, 43) = 0.109326337819849;
    result_local_M(1, 46) = 0.05794382265618636;
    result_local_M(1, 49) = 0.06451772141630999;
    result_local_M(1, 52) = 0.1146044812669878;
    result_local_M(1, 55) = 0.1030869401682464;
    result_local_M(1, 58) = 0.06188733372399985;
    result_local_M(2, 2) = -0.03991016878403244;
    result_local_M(2, 5) = -0.05903555807314021;
    result_local_M(2, 8) = -0.07421571099636162;
    result_local_M(2, 11) = -0.05323297222187619;
    result_local_M(2, 14) = -0.05618363930341419;
    result_local_M(2, 17) = -0.07724836533263818;
    result_local_M(2, 20) = -0.0891113900382783;
    result_local_M(2, 23) = -0.07125249054387619;
    result_local_M(2, 26) = 0.03698784242998645;
    result_local_M(2, 29) = 0.06716878812140782;
    result_local_M(2, 32) = 0.05898961370672467;
    result_local_M(2, 35) = 0.03509553737468334;
    result_local_M(2, 38) = 0.0362988303599596;
    result_local_M(2, 41) = 0.06864610258453398;
    result_local_M(2, 44) = 0.109326337819849;
    result_local_M(2, 47) = 0.05794382265618636;
    result_local_M(2, 50) = 0.06451772141630999;
    result_local_M(2, 53) = 0.1146044812669878;
    result_local_M(2, 56) = 0.1030869401682464;
    result_local_M(2, 59) = 0.06188733372399985;
    result_local_M(3, 0) = -0.0402934741749864;
    result_local_M(3, 3) = -0.07356829221529184;
    result_local_M(3, 6) = -0.0876882981416882;
    result_local_M(3, 9) = -0.05238470260640529;
    result_local_M(3, 12) = -0.05123291169323879;
    result_local_M(3, 15) = -0.0864945483523926;
    result_local_M(3, 18) = -0.0970118691086502;
    result_local_M(3, 21) = -0.06477473021468533;
    result_local_M(3, 24) = 0.04100694021001125;
    result_local_M(3, 27) = 0.092888897389302;
    result_local_M(3, 30) = 0.06221616473234391;
    result_local_M(3, 33) = 0.03427414023464743;
    result_local_M(3, 36) = 0.03371986529649411;
    result_local_M(3, 39) = 0.0908696736321025;
    result_local_M(3, 42) = 0.1363443363126134;
    result_local_M(3, 45) = 0.05161917330377516;
    result_local_M(3, 48) = 0.0599074805417769;
    result_local_M(3, 51) = 0.1328384525089864;
    result_local_M(3, 54) = 0.0917178272669011;
    result_local_M(3, 57) = 0.0508011979515074;
    result_local_M(4, 1) = -0.0402934741749864;
    result_local_M(4, 4) = -0.07356829221529184;
    result_local_M(4, 7) = -0.0876882981416882;
    result_local_M(4, 10) = -0.05238470260640529;
    result_local_M(4, 13) = -0.05123291169323879;
    result_local_M(4, 16) = -0.0864945483523926;
    result_local_M(4, 19) = -0.0970118691086502;
    result_local_M(4, 22) = -0.06477473021468533;
    result_local_M(4, 25) = 0.04100694021001125;
    result_local_M(4, 28) = 0.092888897389302;
    result_local_M(4, 31) = 0.06221616473234391;
    result_local_M(4, 34) = 0.03427414023464743;
    result_local_M(4, 37) = 0.03371986529649411;
    result_local_M(4, 40) = 0.0908696736321025;
    result_local_M(4, 43) = 0.1363443363126134;
    result_local_M(4, 46) = 0.05161917330377516;
    result_local_M(4, 49) = 0.0599074805417769;
    result_local_M(4, 52) = 0.1328384525089864;
    result_local_M(4, 55) = 0.0917178272669011;
    result_local_M(4, 58) = 0.0508011979515074;
    result_local_M(5, 2) = -0.0402934741749864;
    result_local_M(5, 5) = -0.07356829221529184;
    result_local_M(5, 8) = -0.0876882981416882;
    result_local_M(5, 11) = -0.05238470260640529;
    result_local_M(5, 14) = -0.05123291169323879;
    result_local_M(5, 17) = -0.0864945483523926;
    result_local_M(5, 20) = -0.0970118691086502;
    result_local_M(5, 23) = -0.06477473021468533;
    result_local_M(5, 26) = 0.04100694021001125;
    result_local_M(5, 29) = 0.092888897389302;
    result_local_M(5, 32) = 0.06221616473234391;
    result_local_M(5, 35) = 0.03427414023464743;
    result_local_M(5, 38) = 0.03371986529649411;
    result_local_M(5, 41) = 0.0908696736321025;
    result_local_M(5, 44) = 0.1363443363126134;
    result_local_M(5, 47) = 0.05161917330377516;
    result_local_M(5, 50) = 0.0599074805417769;
    result_local_M(5, 53) = 0.1328384525089864;
    result_local_M(5, 56) = 0.0917178272669011;
    result_local_M(5, 59) = 0.0508011979515074;

    // Results for Kappa.
    result_local_kappa(0) = 0.294363056335258;
    result_local_kappa(1) = 0.294363056335258;
    result_local_kappa(2) = 0.294363056335258;
    result_local_kappa(3) = 0.3247553228731228;
    result_local_kappa(4) = 0.3247553228731228;
    result_local_kappa(5) = 0.3247553228731228;

    // Perform the unit tests.
    perform_mortar_pair_unit_test(contact_pair, q_beam, q_beam_rot, q_solid, result_local_D,
        result_local_M, result_local_kappa);
  }

  /**
   * \brief Test a non straight beam in a hex20 element, with line3 mortar shape functions.
   */
  TEST_F(BeamToSolidVolumeMeshtyingPairMortarTest, TestBeamToSolidMeshtyingMortarHermite2Hex20Line3)
  {
    // Element types.
    typedef GEOMETRYPAIR::t_hermite beam_type;
    typedef GEOMETRYPAIR::t_hex20 solid_type;
    typedef GEOMETRYPAIR::t_line3 lambda_type;

    // Create the mesh tying mortar pair.
    BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type, lambda_type>
        contact_pair = BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type,
            lambda_type>();

    // Definition of variables for this test case.
    GEOMETRYPAIR::ElementData<beam_type, double> q_beam;
    GEOMETRYPAIR::ElementData<solid_type, double> q_solid;
    Core::LinAlg::Matrix<9, 1> q_beam_rot;
    Core::LinAlg::SerialDenseMatrix local_D;
    Core::LinAlg::SerialDenseMatrix local_M;
    Core::LinAlg::SerialDenseVector local_kappa;

    // Matrices for the results.
    Core::LinAlg::Matrix<lambda_type::n_dof_, beam_type::n_dof_> result_local_D(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, solid_type::n_dof_> result_local_M(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, 1> result_local_kappa(
        Core::LinAlg::Initialization::zero);

    // Define the geometry of the two elements.
    q_beam.shape_function_data_.ref_length_ = 0.6192043571449604711;
    q_beam.element_position_(0) = 0.15;
    q_beam.element_position_(1) = 0.2;
    q_beam.element_position_(2) = 0.3;
    q_beam.element_position_(3) = 0.5773502691896255;
    q_beam.element_position_(4) = 0.5773502691896258;
    q_beam.element_position_(5) = 0.577350269189626;
    q_beam.element_position_(6) = 0.65;
    q_beam.element_position_(7) = 0.1;
    q_beam.element_position_(8) = 0.1;
    q_beam.element_position_(9) = 0.8017837257372733;
    q_beam.element_position_(10) = -0.5345224838248488;
    q_beam.element_position_(11) = 0.2672612419124244;

    q_beam_rot(0) = 1.674352746442651;
    q_beam_rot(1) = 0.1425949677148126;
    q_beam_rot(2) = 1.0831163124736984;
    q_beam_rot(3) = 1.4331999091513161;
    q_beam_rot(4) = -0.6560404572957742;
    q_beam_rot(5) = -0.2491376152457331;
    q_beam_rot(6) = 0.0;
    q_beam_rot(7) = 0.0;
    q_beam_rot(8) = 0.0;

    // Positional DOFs of the solid.
    q_solid.element_position_(0) = -0.954193516126594;
    q_solid.element_position_(1) = -0.975482672114534;
    q_solid.element_position_(2) = -1.00311316662815;
    q_solid.element_position_(3) = 0.923762409575691;
    q_solid.element_position_(4) = -1.010615727466753;
    q_solid.element_position_(5) = -1.014160992102648;
    q_solid.element_position_(6) = 0.905563488894572;
    q_solid.element_position_(7) = 1.069973754602123;
    q_solid.element_position_(8) = -0.935488788632622;
    q_solid.element_position_(9) = -1.047838992508937;
    q_solid.element_position_(10) = 1.08888017769104;
    q_solid.element_position_(11) = -1.036911970151991;
    q_solid.element_position_(12) = -1.089167378981223;
    q_solid.element_position_(13) = -1.063820485514786;
    q_solid.element_position_(14) = 1.081000671652452;
    q_solid.element_position_(15) = 0.972887477248942;
    q_solid.element_position_(16) = -1.008995039455167;
    q_solid.element_position_(17) = 0.923459231565508;
    q_solid.element_position_(18) = 1.089192110164642;
    q_solid.element_position_(19) = 1.026798931538616;
    q_solid.element_position_(20) = 0.964761788314679;
    q_solid.element_position_(21) = -0.941489453299989;
    q_solid.element_position_(22) = 0.954327975326486;
    q_solid.element_position_(23) = 0.963168067937965;
    q_solid.element_position_(24) = 0.0925155012591516;
    q_solid.element_position_(25) = -0.973383316878771;
    q_solid.element_position_(26) = -1.065800877068667;
    q_solid.element_position_(27) = 0.993751379524698;
    q_solid.element_position_(28) = -0.089078411148891;
    q_solid.element_position_(29) = -1.080589211553427;
    q_solid.element_position_(30) = 0.0823626211382376;
    q_solid.element_position_(31) = 1.033905181036807;
    q_solid.element_position_(32) = -0.952741283444911;
    q_solid.element_position_(33) = -1.024714325285187;
    q_solid.element_position_(34) = -0.0951797166281465;
    q_solid.element_position_(35) = -0.917694055170081;
    q_solid.element_position_(36) = -0.94955183268814;
    q_solid.element_position_(37) = -0.936209915275568;
    q_solid.element_position_(38) = -0.07531489653912402;
    q_solid.element_position_(39) = 0.977045420134747;
    q_solid.element_position_(40) = -0.915831507095275;
    q_solid.element_position_(41) = -0.03604489795829757;
    q_solid.element_position_(42) = 1.044996295051512;
    q_solid.element_position_(43) = 0.912268002554735;
    q_solid.element_position_(44) = 0.026884640088015;
    q_solid.element_position_(45) = -1.003774079332873;
    q_solid.element_position_(46) = 1.016364821141699;
    q_solid.element_position_(47) = -0.04547583423590318;
    q_solid.element_position_(48) = -0.07606458957105064;
    q_solid.element_position_(49) = -0.970767004162723;
    q_solid.element_position_(50) = 1.09197908262736;
    q_solid.element_position_(51) = 1.000582069178839;
    q_solid.element_position_(52) = 0.02495199075528687;
    q_solid.element_position_(53) = 0.942442567644017;
    q_solid.element_position_(54) = -0.0840408739597993;
    q_solid.element_position_(55) = 1.082175787959178;
    q_solid.element_position_(56) = 1.016887685734179;
    q_solid.element_position_(57) = -1.02950388437141;
    q_solid.element_position_(58) = -0.03189318909487643;
    q_solid.element_position_(59) = 0.962850907596607;

    // Results for D.
    result_local_D(0, 0) = 0.0945173766014223;
    result_local_D(0, 3) = 0.005144868995407634;
    result_local_D(0, 6) = -0.0132666172503834;
    result_local_D(0, 9) = 0.0004298224306740067;
    result_local_D(1, 1) = 0.0945173766014223;
    result_local_D(1, 4) = 0.005144868995407634;
    result_local_D(1, 7) = -0.0132666172503834;
    result_local_D(1, 10) = 0.0004298224306740067;
    result_local_D(2, 2) = 0.0945173766014223;
    result_local_D(2, 5) = 0.005144868995407634;
    result_local_D(2, 8) = -0.0132666172503834;
    result_local_D(2, 11) = 0.0004298224306740067;
    result_local_D(3, 0) = -0.006226380126202878;
    result_local_D(3, 3) = 0.0004298224306740081;
    result_local_D(3, 6) = 0.1178694060151065;
    result_local_D(3, 9) = -0.006925279913989872;
    result_local_D(4, 1) = -0.006226380126202878;
    result_local_D(4, 4) = 0.0004298224306740081;
    result_local_D(4, 7) = 0.1178694060151065;
    result_local_D(4, 10) = -0.006925279913989872;
    result_local_D(5, 2) = -0.006226380126202878;
    result_local_D(5, 5) = 0.0004298224306740081;
    result_local_D(5, 8) = 0.1178694060151065;
    result_local_D(5, 11) = -0.006925279913989872;
    result_local_D(6, 0) = 0.2018084338866959;
    result_local_D(6, 3) = 0.02609529639740305;
    result_local_D(6, 6) = 0.2244161600817424;
    result_local_D(6, 9) = -0.02781458612009909;
    result_local_D(7, 1) = 0.2018084338866959;
    result_local_D(7, 4) = 0.02609529639740305;
    result_local_D(7, 7) = 0.2244161600817424;
    result_local_D(7, 10) = -0.02781458612009909;
    result_local_D(8, 2) = 0.2018084338866959;
    result_local_D(8, 5) = 0.02609529639740305;
    result_local_D(8, 8) = 0.2244161600817424;
    result_local_D(8, 11) = -0.02781458612009909;

    // Results for M.
    result_local_M(0, 0) = -0.01226827105298108;
    result_local_M(0, 3) = -0.01378747721722018;
    result_local_M(0, 6) = -0.01823241440998341;
    result_local_M(0, 9) = -0.01632131700557335;
    result_local_M(0, 12) = -0.01911991891586063;
    result_local_M(0, 15) = -0.021109606069842;
    result_local_M(0, 18) = -0.02484931389746807;
    result_local_M(0, 21) = -0.02370519581091836;
    result_local_M(0, 24) = 0.01018739177710672;
    result_local_M(0, 27) = 0.01292504870835167;
    result_local_M(0, 30) = 0.01639128248243293;
    result_local_M(0, 33) = 0.01109080347487935;
    result_local_M(0, 36) = 0.0122748411587571;
    result_local_M(0, 39) = 0.01483657252162607;
    result_local_M(0, 42) = 0.02425227186848702;
    result_local_M(0, 45) = 0.01957001695892268;
    result_local_M(0, 48) = 0.02169479305253372;
    result_local_M(0, 51) = 0.02972947706576897;
    result_local_M(0, 54) = 0.03464249146518235;
    result_local_M(0, 57) = 0.02304928319683736;
    result_local_M(1, 1) = -0.01226827105298108;
    result_local_M(1, 4) = -0.01378747721722018;
    result_local_M(1, 7) = -0.01823241440998341;
    result_local_M(1, 10) = -0.01632131700557335;
    result_local_M(1, 13) = -0.01911991891586063;
    result_local_M(1, 16) = -0.021109606069842;
    result_local_M(1, 19) = -0.02484931389746807;
    result_local_M(1, 22) = -0.02370519581091836;
    result_local_M(1, 25) = 0.01018739177710672;
    result_local_M(1, 28) = 0.01292504870835167;
    result_local_M(1, 31) = 0.01639128248243293;
    result_local_M(1, 34) = 0.01109080347487935;
    result_local_M(1, 37) = 0.0122748411587571;
    result_local_M(1, 40) = 0.01483657252162607;
    result_local_M(1, 43) = 0.02425227186848702;
    result_local_M(1, 46) = 0.01957001695892268;
    result_local_M(1, 49) = 0.02169479305253372;
    result_local_M(1, 52) = 0.02972947706576897;
    result_local_M(1, 55) = 0.03464249146518235;
    result_local_M(1, 58) = 0.02304928319683736;
    result_local_M(2, 2) = -0.01226827105298108;
    result_local_M(2, 5) = -0.01378747721722018;
    result_local_M(2, 8) = -0.01823241440998341;
    result_local_M(2, 11) = -0.01632131700557335;
    result_local_M(2, 14) = -0.01911991891586063;
    result_local_M(2, 17) = -0.021109606069842;
    result_local_M(2, 20) = -0.02484931389746807;
    result_local_M(2, 23) = -0.02370519581091836;
    result_local_M(2, 26) = 0.01018739177710672;
    result_local_M(2, 29) = 0.01292504870835167;
    result_local_M(2, 32) = 0.01639128248243293;
    result_local_M(2, 35) = 0.01109080347487935;
    result_local_M(2, 38) = 0.0122748411587571;
    result_local_M(2, 41) = 0.01483657252162607;
    result_local_M(2, 44) = 0.02425227186848702;
    result_local_M(2, 47) = 0.01957001695892268;
    result_local_M(2, 50) = 0.02169479305253372;
    result_local_M(2, 53) = 0.02972947706576897;
    result_local_M(2, 56) = 0.03464249146518235;
    result_local_M(2, 59) = 0.02304928319683736;
    result_local_M(3, 0) = -0.01265157644393503;
    result_local_M(3, 3) = -0.02832021135937182;
    result_local_M(3, 6) = -0.03170500155531004;
    result_local_M(3, 9) = -0.01547304739010245;
    result_local_M(3, 12) = -0.01416919130568522;
    result_local_M(3, 15) = -0.03035578908959643;
    result_local_M(3, 18) = -0.03274979296783998;
    result_local_M(3, 21) = -0.0172274354817275;
    result_local_M(3, 24) = 0.01420648955713154;
    result_local_M(3, 27) = 0.03864515797624585;
    result_local_M(3, 30) = 0.01961783350805217;
    result_local_M(3, 33) = 0.01026940633484345;
    result_local_M(3, 36) = 0.0096958760952916;
    result_local_M(3, 39) = 0.03706014356919462;
    result_local_M(3, 42) = 0.05127027036125136;
    result_local_M(3, 45) = 0.01324536760651148;
    result_local_M(3, 48) = 0.01708455217800063;
    result_local_M(3, 51) = 0.0479634483077675;
    result_local_M(3, 54) = 0.02327337856383697;
    result_local_M(3, 57) = 0.01196314742434492;
    result_local_M(4, 1) = -0.01265157644393503;
    result_local_M(4, 4) = -0.02832021135937182;
    result_local_M(4, 7) = -0.03170500155531004;
    result_local_M(4, 10) = -0.01547304739010245;
    result_local_M(4, 13) = -0.01416919130568522;
    result_local_M(4, 16) = -0.03035578908959643;
    result_local_M(4, 19) = -0.03274979296783998;
    result_local_M(4, 22) = -0.0172274354817275;
    result_local_M(4, 25) = 0.01420648955713154;
    result_local_M(4, 28) = 0.03864515797624585;
    result_local_M(4, 31) = 0.01961783350805217;
    result_local_M(4, 34) = 0.01026940633484345;
    result_local_M(4, 37) = 0.0096958760952916;
    result_local_M(4, 40) = 0.03706014356919462;
    result_local_M(4, 43) = 0.05127027036125136;
    result_local_M(4, 46) = 0.01324536760651148;
    result_local_M(4, 49) = 0.01708455217800063;
    result_local_M(4, 52) = 0.0479634483077675;
    result_local_M(4, 55) = 0.02327337856383697;
    result_local_M(4, 58) = 0.01196314742434492;
    result_local_M(5, 2) = -0.01265157644393503;
    result_local_M(5, 5) = -0.02832021135937182;
    result_local_M(5, 8) = -0.03170500155531004;
    result_local_M(5, 11) = -0.01547304739010245;
    result_local_M(5, 14) = -0.01416919130568522;
    result_local_M(5, 17) = -0.03035578908959643;
    result_local_M(5, 20) = -0.03274979296783998;
    result_local_M(5, 23) = -0.0172274354817275;
    result_local_M(5, 26) = 0.01420648955713154;
    result_local_M(5, 29) = 0.03864515797624585;
    result_local_M(5, 32) = 0.01961783350805217;
    result_local_M(5, 35) = 0.01026940633484345;
    result_local_M(5, 38) = 0.0096958760952916;
    result_local_M(5, 41) = 0.03706014356919462;
    result_local_M(5, 44) = 0.05127027036125136;
    result_local_M(5, 47) = 0.01324536760651148;
    result_local_M(5, 50) = 0.01708455217800063;
    result_local_M(5, 53) = 0.0479634483077675;
    result_local_M(5, 56) = 0.02327337856383697;
    result_local_M(5, 59) = 0.01196314742434492;
    result_local_M(6, 0) = -0.05528379546210273;
    result_local_M(6, 3) = -0.0904961617118401;
    result_local_M(6, 6) = -0.1119665931727564;
    result_local_M(6, 9) = -0.07382331043260568;
    result_local_M(6, 12) = -0.07412744077510714;
    result_local_M(6, 15) = -0.1122775185255924;
    result_local_M(6, 18) = -0.1285241522816205;
    result_local_M(6, 21) = -0.0950945894659157;
    result_local_M(6, 24) = 0.05360090130575944;
    result_local_M(6, 27) = 0.1084874788261123;
    result_local_M(6, 30) = 0.0851966624485835;
    result_local_M(6, 33) = 0.04800946779960798;
    result_local_M(6, 36) = 0.048047978402405;
    result_local_M(6, 39) = 0.1076190601258158;
    result_local_M(6, 42) = 0.170148131902724;
    result_local_M(6, 45) = 0.07674761139452736;
    result_local_M(6, 48) = 0.0856458567275525;
    result_local_M(6, 51) = 0.1697500084024377;
    result_local_M(6, 54) = 0.1368888974061282;
    result_local_M(6, 57) = 0.07767610105432496;
    result_local_M(7, 1) = -0.05528379546210273;
    result_local_M(7, 4) = -0.0904961617118401;
    result_local_M(7, 7) = -0.1119665931727564;
    result_local_M(7, 10) = -0.07382331043260568;
    result_local_M(7, 13) = -0.07412744077510714;
    result_local_M(7, 16) = -0.1122775185255924;
    result_local_M(7, 19) = -0.1285241522816205;
    result_local_M(7, 22) = -0.0950945894659157;
    result_local_M(7, 25) = 0.05360090130575944;
    result_local_M(7, 28) = 0.1084874788261123;
    result_local_M(7, 31) = 0.0851966624485835;
    result_local_M(7, 34) = 0.04800946779960798;
    result_local_M(7, 37) = 0.048047978402405;
    result_local_M(7, 40) = 0.1076190601258158;
    result_local_M(7, 43) = 0.170148131902724;
    result_local_M(7, 46) = 0.07674761139452736;
    result_local_M(7, 49) = 0.0856458567275525;
    result_local_M(7, 52) = 0.1697500084024377;
    result_local_M(7, 55) = 0.1368888974061282;
    result_local_M(7, 58) = 0.07767610105432496;
    result_local_M(8, 2) = -0.05528379546210273;
    result_local_M(8, 5) = -0.0904961617118401;
    result_local_M(8, 8) = -0.1119665931727564;
    result_local_M(8, 11) = -0.07382331043260568;
    result_local_M(8, 14) = -0.07412744077510714;
    result_local_M(8, 17) = -0.1122775185255924;
    result_local_M(8, 20) = -0.1285241522816205;
    result_local_M(8, 23) = -0.0950945894659157;
    result_local_M(8, 26) = 0.05360090130575944;
    result_local_M(8, 29) = 0.1084874788261123;
    result_local_M(8, 32) = 0.0851966624485835;
    result_local_M(8, 35) = 0.04800946779960798;
    result_local_M(8, 38) = 0.048047978402405;
    result_local_M(8, 41) = 0.1076190601258158;
    result_local_M(8, 44) = 0.170148131902724;
    result_local_M(8, 47) = 0.07674761139452736;
    result_local_M(8, 50) = 0.0856458567275525;
    result_local_M(8, 53) = 0.1697500084024377;
    result_local_M(8, 56) = 0.1368888974061282;
    result_local_M(8, 59) = 0.07767610105432496;

    // Results for Kappa.
    result_local_kappa(0) = 0.0812507593510389;
    result_local_kappa(1) = 0.0812507593510389;
    result_local_kappa(2) = 0.0812507593510389;
    result_local_kappa(3) = 0.1116430258889036;
    result_local_kappa(4) = 0.1116430258889036;
    result_local_kappa(5) = 0.1116430258889036;
    result_local_kappa(6) = 0.4262245939684383;
    result_local_kappa(7) = 0.4262245939684383;
    result_local_kappa(8) = 0.4262245939684383;

    // Perform the unit tests.
    perform_mortar_pair_unit_test(contact_pair, q_beam, q_beam_rot, q_solid, result_local_D,
        result_local_M, result_local_kappa);
  }

  /**
   * \brief Test a non straight beam in a hex20 element, with line4 mortar shape functions.
   */
  TEST_F(BeamToSolidVolumeMeshtyingPairMortarTest, TestBeamToSolidMeshtyingMortarHermite2Hex20Line4)
  {
    // Element types.
    typedef GEOMETRYPAIR::t_hermite beam_type;
    typedef GEOMETRYPAIR::t_hex20 solid_type;
    typedef GEOMETRYPAIR::t_line4 lambda_type;

    // Create the mesh tying mortar pair.
    BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type, lambda_type>
        contact_pair = BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type,
            lambda_type>();

    // Definition of variables for this test case.
    GEOMETRYPAIR::ElementData<beam_type, double> q_beam;
    GEOMETRYPAIR::ElementData<solid_type, double> q_solid;
    Core::LinAlg::Matrix<9, 1> q_beam_rot;
    Core::LinAlg::SerialDenseMatrix local_D;
    Core::LinAlg::SerialDenseMatrix local_M;
    Core::LinAlg::SerialDenseVector local_kappa;

    // Matrices for the results.
    Core::LinAlg::Matrix<lambda_type::n_dof_, beam_type::n_dof_> result_local_D(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, solid_type::n_dof_> result_local_M(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, 1> result_local_kappa(
        Core::LinAlg::Initialization::zero);

    // Define the geometry of the two elements.
    q_beam.shape_function_data_.ref_length_ = 0.6192043571449604711;
    q_beam.element_position_(0) = 0.15;
    q_beam.element_position_(1) = 0.2;
    q_beam.element_position_(2) = 0.3;
    q_beam.element_position_(3) = 0.5773502691896255;
    q_beam.element_position_(4) = 0.5773502691896258;
    q_beam.element_position_(5) = 0.577350269189626;
    q_beam.element_position_(6) = 0.65;
    q_beam.element_position_(7) = 0.1;
    q_beam.element_position_(8) = 0.1;
    q_beam.element_position_(9) = 0.8017837257372733;
    q_beam.element_position_(10) = -0.5345224838248488;
    q_beam.element_position_(11) = 0.2672612419124244;

    q_beam_rot(0) = 1.674352746442651;
    q_beam_rot(1) = 0.1425949677148126;
    q_beam_rot(2) = 1.0831163124736984;
    q_beam_rot(3) = 1.4331999091513161;
    q_beam_rot(4) = -0.6560404572957742;
    q_beam_rot(5) = -0.2491376152457331;
    q_beam_rot(6) = 0.0;
    q_beam_rot(7) = 0.0;
    q_beam_rot(8) = 0.0;

    // Positional DOFs of the solid.
    q_solid.element_position_(0) = -0.954193516126594;
    q_solid.element_position_(1) = -0.975482672114534;
    q_solid.element_position_(2) = -1.00311316662815;
    q_solid.element_position_(3) = 0.923762409575691;
    q_solid.element_position_(4) = -1.010615727466753;
    q_solid.element_position_(5) = -1.014160992102648;
    q_solid.element_position_(6) = 0.905563488894572;
    q_solid.element_position_(7) = 1.069973754602123;
    q_solid.element_position_(8) = -0.935488788632622;
    q_solid.element_position_(9) = -1.047838992508937;
    q_solid.element_position_(10) = 1.08888017769104;
    q_solid.element_position_(11) = -1.036911970151991;
    q_solid.element_position_(12) = -1.089167378981223;
    q_solid.element_position_(13) = -1.063820485514786;
    q_solid.element_position_(14) = 1.081000671652452;
    q_solid.element_position_(15) = 0.972887477248942;
    q_solid.element_position_(16) = -1.008995039455167;
    q_solid.element_position_(17) = 0.923459231565508;
    q_solid.element_position_(18) = 1.089192110164642;
    q_solid.element_position_(19) = 1.026798931538616;
    q_solid.element_position_(20) = 0.964761788314679;
    q_solid.element_position_(21) = -0.941489453299989;
    q_solid.element_position_(22) = 0.954327975326486;
    q_solid.element_position_(23) = 0.963168067937965;
    q_solid.element_position_(24) = 0.0925155012591516;
    q_solid.element_position_(25) = -0.973383316878771;
    q_solid.element_position_(26) = -1.065800877068667;
    q_solid.element_position_(27) = 0.993751379524698;
    q_solid.element_position_(28) = -0.089078411148891;
    q_solid.element_position_(29) = -1.080589211553427;
    q_solid.element_position_(30) = 0.0823626211382376;
    q_solid.element_position_(31) = 1.033905181036807;
    q_solid.element_position_(32) = -0.952741283444911;
    q_solid.element_position_(33) = -1.024714325285187;
    q_solid.element_position_(34) = -0.0951797166281465;
    q_solid.element_position_(35) = -0.917694055170081;
    q_solid.element_position_(36) = -0.94955183268814;
    q_solid.element_position_(37) = -0.936209915275568;
    q_solid.element_position_(38) = -0.07531489653912402;
    q_solid.element_position_(39) = 0.977045420134747;
    q_solid.element_position_(40) = -0.915831507095275;
    q_solid.element_position_(41) = -0.03604489795829757;
    q_solid.element_position_(42) = 1.044996295051512;
    q_solid.element_position_(43) = 0.912268002554735;
    q_solid.element_position_(44) = 0.026884640088015;
    q_solid.element_position_(45) = -1.003774079332873;
    q_solid.element_position_(46) = 1.016364821141699;
    q_solid.element_position_(47) = -0.04547583423590318;
    q_solid.element_position_(48) = -0.07606458957105064;
    q_solid.element_position_(49) = -0.970767004162723;
    q_solid.element_position_(50) = 1.09197908262736;
    q_solid.element_position_(51) = 1.000582069178839;
    q_solid.element_position_(52) = 0.02495199075528687;
    q_solid.element_position_(53) = 0.942442567644017;
    q_solid.element_position_(54) = -0.0840408739597993;
    q_solid.element_position_(55) = 1.082175787959178;
    q_solid.element_position_(56) = 1.016887685734179;
    q_solid.element_position_(57) = -1.02950388437141;
    q_solid.element_position_(58) = -0.03189318909487643;
    q_solid.element_position_(59) = 0.962850907596607;

    // Results for D.
    result_local_D(0, 0) = 0.05837517950584759;
    result_local_D(0, 3) = 0.002017208305318981;
    result_local_D(0, 6) = 0.005829701162185071;
    result_local_D(0, 9) = -0.0002955710710872768;
    result_local_D(1, 1) = 0.05837517950584759;
    result_local_D(1, 4) = 0.002017208305318981;
    result_local_D(1, 7) = 0.005829701162185071;
    result_local_D(1, 10) = -0.0002955710710872768;
    result_local_D(2, 2) = 0.05837517950584759;
    result_local_D(2, 5) = 0.002017208305318981;
    result_local_D(2, 8) = 0.005829701162185071;
    result_local_D(2, 11) = -0.0002955710710872768;
    result_local_D(3, 0) = 0.004689762733534804;
    result_local_D(3, 3) = 0.0002955710710872767;
    result_local_D(3, 6) = 0.07072106759232025;
    result_local_D(3, 9) = -0.002723063147216203;
    result_local_D(4, 1) = 0.004689762733534804;
    result_local_D(4, 4) = 0.0002955710710872767;
    result_local_D(4, 7) = 0.07072106759232025;
    result_local_D(4, 10) = -0.002723063147216203;
    result_local_D(5, 2) = 0.004689762733534804;
    result_local_D(5, 5) = 0.0002955710710872767;
    result_local_D(5, 8) = 0.07072106759232025;
    result_local_D(5, 11) = -0.002723063147216203;
    result_local_D(6, 0) = 0.184104753994235;
    result_local_D(6, 3) = 0.0191687182192921;
    result_local_D(6, 6) = 0.02686710479294802;
    result_local_D(6, 9) = -0.00825428928975331;
    result_local_D(7, 1) = 0.184104753994235;
    result_local_D(7, 4) = 0.0191687182192921;
    result_local_D(7, 7) = 0.02686710479294802;
    result_local_D(7, 10) = -0.00825428928975331;
    result_local_D(8, 2) = 0.184104753994235;
    result_local_D(8, 5) = 0.0191687182192921;
    result_local_D(8, 8) = 0.02686710479294802;
    result_local_D(8, 11) = -0.00825428928975331;
    result_local_D(9, 0) = 0.04292973412829792;
    result_local_D(9, 3) = 0.01018849022778634;
    result_local_D(9, 6) = 0.2256010752990122;
    result_local_D(9, 9) = -0.02303712009535817;
    result_local_D(10, 1) = 0.04292973412829792;
    result_local_D(10, 4) = 0.01018849022778634;
    result_local_D(10, 7) = 0.2256010752990122;
    result_local_D(10, 10) = -0.02303712009535817;
    result_local_D(11, 2) = 0.04292973412829792;
    result_local_D(11, 5) = 0.01018849022778634;
    result_local_D(11, 8) = 0.2256010752990122;
    result_local_D(11, 11) = -0.02303712009535817;

    // Results for M.
    result_local_M(0, 0) = -0.00940616606269854;
    result_local_M(0, 3) = -0.01219599638326402;
    result_local_M(0, 6) = -0.01516293324429006;
    result_local_M(0, 9) = -0.01212105569382873;
    result_local_M(0, 12) = -0.01381967365926134;
    result_local_M(0, 15) = -0.01690200348318517;
    result_local_M(0, 18) = -0.01942881172992854;
    result_local_M(0, 21) = -0.01683160660350284;
    result_local_M(0, 24) = 0.00828878641531752;
    result_local_M(0, 27) = 0.01295115066697093;
    result_local_M(0, 30) = 0.01257148076877198;
    result_local_M(0, 33) = 0.00836596192112849;
    result_local_M(0, 36) = 0.00909305162387093;
    result_local_M(0, 39) = 0.01393819733011963;
    result_local_M(0, 42) = 0.02102298674024643;
    result_local_M(0, 45) = 0.0138522488701854;
    result_local_M(0, 48) = 0.01577736083803085;
    result_local_M(0, 51) = 0.02409783078783365;
    result_local_M(0, 54) = 0.02405031506135321;
    result_local_M(0, 57) = 0.01606375650416286;
    result_local_M(1, 1) = -0.00940616606269854;
    result_local_M(1, 4) = -0.01219599638326402;
    result_local_M(1, 7) = -0.01516293324429006;
    result_local_M(1, 10) = -0.01212105569382873;
    result_local_M(1, 13) = -0.01381967365926134;
    result_local_M(1, 16) = -0.01690200348318517;
    result_local_M(1, 19) = -0.01942881172992854;
    result_local_M(1, 22) = -0.01683160660350284;
    result_local_M(1, 25) = 0.00828878641531752;
    result_local_M(1, 28) = 0.01295115066697093;
    result_local_M(1, 31) = 0.01257148076877198;
    result_local_M(1, 34) = 0.00836596192112849;
    result_local_M(1, 37) = 0.00909305162387093;
    result_local_M(1, 40) = 0.01393819733011963;
    result_local_M(1, 43) = 0.02102298674024643;
    result_local_M(1, 46) = 0.0138522488701854;
    result_local_M(1, 49) = 0.01577736083803085;
    result_local_M(1, 52) = 0.02409783078783365;
    result_local_M(1, 55) = 0.02405031506135321;
    result_local_M(1, 58) = 0.01606375650416286;
    result_local_M(2, 2) = -0.00940616606269854;
    result_local_M(2, 5) = -0.01219599638326402;
    result_local_M(2, 8) = -0.01516293324429006;
    result_local_M(2, 11) = -0.01212105569382873;
    result_local_M(2, 14) = -0.01381967365926134;
    result_local_M(2, 17) = -0.01690200348318517;
    result_local_M(2, 20) = -0.01942881172992854;
    result_local_M(2, 23) = -0.01683160660350284;
    result_local_M(2, 26) = 0.00828878641531752;
    result_local_M(2, 29) = 0.01295115066697093;
    result_local_M(2, 32) = 0.01257148076877198;
    result_local_M(2, 35) = 0.00836596192112849;
    result_local_M(2, 38) = 0.00909305162387093;
    result_local_M(2, 41) = 0.01393819733011963;
    result_local_M(2, 44) = 0.02102298674024643;
    result_local_M(2, 47) = 0.0138522488701854;
    result_local_M(2, 50) = 0.01577736083803085;
    result_local_M(2, 53) = 0.02409783078783365;
    result_local_M(2, 56) = 0.02405031506135321;
    result_local_M(2, 59) = 0.01606375650416286;
    result_local_M(3, 0) = -0.00860320700145474;
    result_local_M(3, 3) = -0.01859967197934799;
    result_local_M(3, 6) = -0.02077865857440884;
    result_local_M(3, 9) = -0.01044539489777136;
    result_local_M(3, 12) = -0.01020350646539612;
    result_local_M(3, 15) = -0.02052870186055422;
    result_local_M(3, 18) = -0.02210477610017696;
    result_local_M(3, 21) = -0.01221420100590356;
    result_local_M(3, 24) = 0.00940498225570082;
    result_local_M(3, 27) = 0.02505812116436255;
    result_local_M(3, 30) = 0.01278805241564019;
    result_local_M(3, 33) = 0.006993064413643316;
    result_local_M(3, 36) = 0.006871668329877146;
    result_local_M(3, 39) = 0.02450613624497408;
    result_local_M(3, 42) = 0.03323103900165144;
    result_local_M(3, 45) = 0.00936968427093284;
    result_local_M(3, 48) = 0.01229625230155944;
    result_local_M(3, 51) = 0.03237634353539811;
    result_local_M(3, 54) = 0.0167544427919001;
    result_local_M(3, 57) = 0.00923916148522879;
    result_local_M(4, 1) = -0.00860320700145474;
    result_local_M(4, 4) = -0.01859967197934799;
    result_local_M(4, 7) = -0.02077865857440884;
    result_local_M(4, 10) = -0.01044539489777136;
    result_local_M(4, 13) = -0.01020350646539612;
    result_local_M(4, 16) = -0.02052870186055422;
    result_local_M(4, 19) = -0.02210477610017696;
    result_local_M(4, 22) = -0.01221420100590356;
    result_local_M(4, 25) = 0.00940498225570082;
    result_local_M(4, 28) = 0.02505812116436255;
    result_local_M(4, 31) = 0.01278805241564019;
    result_local_M(4, 34) = 0.006993064413643316;
    result_local_M(4, 37) = 0.006871668329877146;
    result_local_M(4, 40) = 0.02450613624497408;
    result_local_M(4, 43) = 0.03323103900165144;
    result_local_M(4, 46) = 0.00936968427093284;
    result_local_M(4, 49) = 0.01229625230155944;
    result_local_M(4, 52) = 0.03237634353539811;
    result_local_M(4, 55) = 0.0167544427919001;
    result_local_M(4, 58) = 0.00923916148522879;
    result_local_M(5, 2) = -0.00860320700145474;
    result_local_M(5, 5) = -0.01859967197934799;
    result_local_M(5, 8) = -0.02077865857440884;
    result_local_M(5, 11) = -0.01044539489777136;
    result_local_M(5, 14) = -0.01020350646539612;
    result_local_M(5, 17) = -0.02052870186055422;
    result_local_M(5, 20) = -0.02210477610017696;
    result_local_M(5, 23) = -0.01221420100590356;
    result_local_M(5, 26) = 0.00940498225570082;
    result_local_M(5, 29) = 0.02505812116436255;
    result_local_M(5, 32) = 0.01278805241564019;
    result_local_M(5, 35) = 0.006993064413643316;
    result_local_M(5, 38) = 0.006871668329877146;
    result_local_M(5, 41) = 0.02450613624497408;
    result_local_M(5, 44) = 0.03323103900165144;
    result_local_M(5, 47) = 0.00936968427093284;
    result_local_M(5, 50) = 0.01229625230155944;
    result_local_M(5, 53) = 0.03237634353539811;
    result_local_M(5, 56) = 0.0167544427919001;
    result_local_M(5, 59) = 0.00923916148522879;
    result_local_M(6, 0) = -0.02931773826913616;
    result_local_M(6, 3) = -0.03871050314380851;
    result_local_M(6, 6) = -0.0511959159368637;
    result_local_M(6, 9) = -0.04028452534746098;
    result_local_M(6, 12) = -0.04369852606046304;
    result_local_M(6, 15) = -0.05472687720706764;
    result_local_M(6, 18) = -0.0644580636082263;
    result_local_M(6, 21) = -0.05628123867196493;
    result_local_M(6, 24) = 0.02579615407502742;
    result_local_M(6, 27) = 0.04060449868393431;
    result_local_M(6, 30) = 0.04340815355920165;
    result_local_M(6, 33) = 0.02617807508610559;
    result_local_M(6, 36) = 0.02756336050556038;
    result_local_M(6, 39) = 0.04305227312170029;
    result_local_M(6, 42) = 0.07349340484824324;
    result_local_M(6, 45) = 0.04593365853915961;
    result_local_M(6, 48) = 0.0498694929163408;
    result_local_M(6, 51) = 0.0805511919847201;
    result_local_M(6, 54) = 0.0831098657387854;
    result_local_M(6, 57) = 0.05008511797339535;
    result_local_M(7, 1) = -0.02931773826913616;
    result_local_M(7, 4) = -0.03871050314380851;
    result_local_M(7, 7) = -0.0511959159368637;
    result_local_M(7, 10) = -0.04028452534746098;
    result_local_M(7, 13) = -0.04369852606046304;
    result_local_M(7, 16) = -0.05472687720706764;
    result_local_M(7, 19) = -0.0644580636082263;
    result_local_M(7, 22) = -0.05628123867196493;
    result_local_M(7, 25) = 0.02579615407502742;
    result_local_M(7, 28) = 0.04060449868393431;
    result_local_M(7, 31) = 0.04340815355920165;
    result_local_M(7, 34) = 0.02617807508610559;
    result_local_M(7, 37) = 0.02756336050556038;
    result_local_M(7, 40) = 0.04305227312170029;
    result_local_M(7, 43) = 0.07349340484824324;
    result_local_M(7, 46) = 0.04593365853915961;
    result_local_M(7, 49) = 0.0498694929163408;
    result_local_M(7, 52) = 0.0805511919847201;
    result_local_M(7, 55) = 0.0831098657387854;
    result_local_M(7, 58) = 0.05008511797339535;
    result_local_M(8, 2) = -0.02931773826913616;
    result_local_M(8, 5) = -0.03871050314380851;
    result_local_M(8, 8) = -0.0511959159368637;
    result_local_M(8, 11) = -0.04028452534746098;
    result_local_M(8, 14) = -0.04369852606046304;
    result_local_M(8, 17) = -0.05472687720706764;
    result_local_M(8, 20) = -0.0644580636082263;
    result_local_M(8, 23) = -0.05628123867196493;
    result_local_M(8, 26) = 0.02579615407502742;
    result_local_M(8, 29) = 0.04060449868393431;
    result_local_M(8, 32) = 0.04340815355920165;
    result_local_M(8, 35) = 0.02617807508610559;
    result_local_M(8, 38) = 0.02756336050556038;
    result_local_M(8, 41) = 0.04305227312170029;
    result_local_M(8, 44) = 0.07349340484824324;
    result_local_M(8, 47) = 0.04593365853915961;
    result_local_M(8, 50) = 0.0498694929163408;
    result_local_M(8, 53) = 0.0805511919847201;
    result_local_M(8, 56) = 0.0831098657387854;
    result_local_M(8, 59) = 0.05008511797339535;
    result_local_M(9, 0) = -0.0328765316257294;
    result_local_M(9, 3) = -0.06309767878201154;
    result_local_M(9, 6) = -0.07476650138248726;
    result_local_M(9, 9) = -0.04276669888922041;
    result_local_M(9, 12) = -0.03969484481153248;
    result_local_M(9, 15) = -0.07158533113422378;
    result_local_M(9, 18) = -0.0801316077085967;
    result_local_M(9, 21) = -0.05070017447719018;
    result_local_M(9, 24) = 0.03450485989395196;
    result_local_M(9, 27) = 0.081443914995442;
    result_local_M(9, 30) = 0.05243809169545476;
    result_local_M(9, 33) = 0.02783257618845338;
    result_local_M(9, 36) = 0.02649061519714526;
    result_local_M(9, 39) = 0.07801916951984252;
    result_local_M(9, 42) = 0.1179232435423212;
    result_local_M(9, 45) = 0.04040740427968368;
    result_local_M(9, 48) = 0.04648209590215579;
    result_local_M(9, 51) = 0.1104175674680224;
    result_local_M(9, 54) = 0.0708901438431087;
    result_local_M(9, 57) = 0.03730049571272023;
    result_local_M(10, 1) = -0.0328765316257294;
    result_local_M(10, 4) = -0.06309767878201154;
    result_local_M(10, 7) = -0.07476650138248726;
    result_local_M(10, 10) = -0.04276669888922041;
    result_local_M(10, 13) = -0.03969484481153248;
    result_local_M(10, 16) = -0.07158533113422378;
    result_local_M(10, 19) = -0.0801316077085967;
    result_local_M(10, 22) = -0.05070017447719018;
    result_local_M(10, 25) = 0.03450485989395196;
    result_local_M(10, 28) = 0.081443914995442;
    result_local_M(10, 31) = 0.05243809169545476;
    result_local_M(10, 34) = 0.02783257618845338;
    result_local_M(10, 37) = 0.02649061519714526;
    result_local_M(10, 40) = 0.07801916951984252;
    result_local_M(10, 43) = 0.1179232435423212;
    result_local_M(10, 46) = 0.04040740427968368;
    result_local_M(10, 49) = 0.04648209590215579;
    result_local_M(10, 52) = 0.1104175674680224;
    result_local_M(10, 55) = 0.0708901438431087;
    result_local_M(10, 58) = 0.03730049571272023;
    result_local_M(11, 2) = -0.0328765316257294;
    result_local_M(11, 5) = -0.06309767878201154;
    result_local_M(11, 8) = -0.07476650138248726;
    result_local_M(11, 11) = -0.04276669888922041;
    result_local_M(11, 14) = -0.03969484481153248;
    result_local_M(11, 17) = -0.07158533113422378;
    result_local_M(11, 20) = -0.0801316077085967;
    result_local_M(11, 23) = -0.05070017447719018;
    result_local_M(11, 26) = 0.03450485989395196;
    result_local_M(11, 29) = 0.081443914995442;
    result_local_M(11, 32) = 0.05243809169545476;
    result_local_M(11, 35) = 0.02783257618845338;
    result_local_M(11, 38) = 0.02649061519714526;
    result_local_M(11, 41) = 0.07801916951984252;
    result_local_M(11, 44) = 0.1179232435423212;
    result_local_M(11, 47) = 0.04040740427968368;
    result_local_M(11, 50) = 0.04648209590215579;
    result_local_M(11, 53) = 0.1104175674680224;
    result_local_M(11, 56) = 0.0708901438431087;
    result_local_M(11, 59) = 0.03730049571272023;

    // Results for Kappa.
    result_local_kappa(0) = 0.06420488066803266;
    result_local_kappa(1) = 0.06420488066803266;
    result_local_kappa(2) = 0.06420488066803266;
    result_local_kappa(3) = 0.07541083032585504;
    result_local_kappa(4) = 0.07541083032585504;
    result_local_kappa(5) = 0.07541083032585504;
    result_local_kappa(6) = 0.210971858787183;
    result_local_kappa(7) = 0.210971858787183;
    result_local_kappa(8) = 0.210971858787183;
    result_local_kappa(9) = 0.2685308094273101;
    result_local_kappa(10) = 0.2685308094273101;
    result_local_kappa(11) = 0.2685308094273101;

    // Perform the unit tests.
    perform_mortar_pair_unit_test(contact_pair, q_beam, q_beam_rot, q_solid, result_local_D,
        result_local_M, result_local_kappa);
  }

  /**
   * \brief Test a non straight beam in a hex27 element, with line2 mortar shape functions.
   */
  TEST_F(BeamToSolidVolumeMeshtyingPairMortarTest, TestBeamToSolidMeshtyingMortarHermite2Hex27Line2)
  {
    // Element types.
    typedef GEOMETRYPAIR::t_hermite beam_type;
    typedef GEOMETRYPAIR::t_hex27 solid_type;
    typedef GEOMETRYPAIR::t_line2 lambda_type;

    // Create the mesh tying mortar pair.
    BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type, lambda_type>
        contact_pair = BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type,
            lambda_type>();

    // Definition of variables for this test case.
    GEOMETRYPAIR::ElementData<beam_type, double> q_beam;
    GEOMETRYPAIR::ElementData<solid_type, double> q_solid;
    Core::LinAlg::Matrix<9, 1> q_beam_rot;
    Core::LinAlg::SerialDenseMatrix local_D;
    Core::LinAlg::SerialDenseMatrix local_M;
    Core::LinAlg::SerialDenseVector local_kappa;

    // Matrices for the results.
    Core::LinAlg::Matrix<lambda_type::n_dof_, beam_type::n_dof_> result_local_D(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, solid_type::n_dof_> result_local_M(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, 1> result_local_kappa(
        Core::LinAlg::Initialization::zero);

    // Define the geometry of the two elements.
    q_beam.shape_function_data_.ref_length_ = 0.6192043571449604711;
    q_beam.element_position_(0) = 0.15;
    q_beam.element_position_(1) = 0.2;
    q_beam.element_position_(2) = 0.3;
    q_beam.element_position_(3) = 0.5773502691896255;
    q_beam.element_position_(4) = 0.5773502691896258;
    q_beam.element_position_(5) = 0.577350269189626;
    q_beam.element_position_(6) = 0.65;
    q_beam.element_position_(7) = 0.1;
    q_beam.element_position_(8) = 0.1;
    q_beam.element_position_(9) = 0.8017837257372733;
    q_beam.element_position_(10) = -0.5345224838248488;
    q_beam.element_position_(11) = 0.2672612419124244;

    q_beam_rot(0) = 1.674352746442651;
    q_beam_rot(1) = 0.1425949677148126;
    q_beam_rot(2) = 1.0831163124736984;
    q_beam_rot(3) = 1.4331999091513161;
    q_beam_rot(4) = -0.6560404572957742;
    q_beam_rot(5) = -0.2491376152457331;
    q_beam_rot(6) = 0.0;
    q_beam_rot(7) = 0.0;
    q_beam_rot(8) = 0.0;

    // Positional DOFs of the solid.
    q_solid.element_position_(0) = -0.954193516126594;
    q_solid.element_position_(1) = -0.975482672114534;
    q_solid.element_position_(2) = -1.00311316662815;
    q_solid.element_position_(3) = 0.923762409575691;
    q_solid.element_position_(4) = -1.010615727466753;
    q_solid.element_position_(5) = -1.014160992102648;
    q_solid.element_position_(6) = 0.905563488894572;
    q_solid.element_position_(7) = 1.069973754602123;
    q_solid.element_position_(8) = -0.935488788632622;
    q_solid.element_position_(9) = -1.047838992508937;
    q_solid.element_position_(10) = 1.08888017769104;
    q_solid.element_position_(11) = -1.036911970151991;
    q_solid.element_position_(12) = -1.089167378981223;
    q_solid.element_position_(13) = -1.063820485514786;
    q_solid.element_position_(14) = 1.081000671652452;
    q_solid.element_position_(15) = 0.972887477248942;
    q_solid.element_position_(16) = -1.008995039455167;
    q_solid.element_position_(17) = 0.923459231565508;
    q_solid.element_position_(18) = 1.089192110164642;
    q_solid.element_position_(19) = 1.026798931538616;
    q_solid.element_position_(20) = 0.964761788314679;
    q_solid.element_position_(21) = -0.941489453299989;
    q_solid.element_position_(22) = 0.954327975326486;
    q_solid.element_position_(23) = 0.963168067937965;
    q_solid.element_position_(24) = 0.0925155012591516;
    q_solid.element_position_(25) = -0.973383316878771;
    q_solid.element_position_(26) = -1.065800877068667;
    q_solid.element_position_(27) = 0.993751379524698;
    q_solid.element_position_(28) = -0.089078411148891;
    q_solid.element_position_(29) = -1.080589211553427;
    q_solid.element_position_(30) = 0.0823626211382376;
    q_solid.element_position_(31) = 1.033905181036807;
    q_solid.element_position_(32) = -0.952741283444911;
    q_solid.element_position_(33) = -1.024714325285187;
    q_solid.element_position_(34) = -0.0951797166281465;
    q_solid.element_position_(35) = -0.917694055170081;
    q_solid.element_position_(36) = -0.94955183268814;
    q_solid.element_position_(37) = -0.936209915275568;
    q_solid.element_position_(38) = -0.07531489653912402;
    q_solid.element_position_(39) = 0.977045420134747;
    q_solid.element_position_(40) = -0.915831507095275;
    q_solid.element_position_(41) = -0.03604489795829757;
    q_solid.element_position_(42) = 1.044996295051512;
    q_solid.element_position_(43) = 0.912268002554735;
    q_solid.element_position_(44) = 0.026884640088015;
    q_solid.element_position_(45) = -1.003774079332873;
    q_solid.element_position_(46) = 1.016364821141699;
    q_solid.element_position_(47) = -0.04547583423590318;
    q_solid.element_position_(48) = -0.07606458957105064;
    q_solid.element_position_(49) = -0.970767004162723;
    q_solid.element_position_(50) = 1.09197908262736;
    q_solid.element_position_(51) = 1.000582069178839;
    q_solid.element_position_(52) = 0.02495199075528687;
    q_solid.element_position_(53) = 0.942442567644017;
    q_solid.element_position_(54) = -0.0840408739597993;
    q_solid.element_position_(55) = 1.082175787959178;
    q_solid.element_position_(56) = 1.016887685734179;
    q_solid.element_position_(57) = -1.02950388437141;
    q_solid.element_position_(58) = -0.03189318909487643;
    q_solid.element_position_(59) = 0.962850907596607;
    q_solid.element_position_(60) = 0.06298633006639215;
    q_solid.element_position_(61) = -0.06929279965773389;
    q_solid.element_position_(62) = -0.963099037764495;
    q_solid.element_position_(63) = 0.005860067798445501;
    q_solid.element_position_(64) = -1.087116820781742;
    q_solid.element_position_(65) = 0.01143091251576034;
    q_solid.element_position_(66) = 0.990119207129434;
    q_solid.element_position_(67) = -0.07554386166147545;
    q_solid.element_position_(68) = 0.02213195187583161;
    q_solid.element_position_(69) = -0.0892155216099889;
    q_solid.element_position_(70) = 1.051578415322994;
    q_solid.element_position_(71) = 0.01644372121101889;
    q_solid.element_position_(72) = -1.084952712553014;
    q_solid.element_position_(73) = -0.06218066414480983;
    q_solid.element_position_(74) = 0.03589340707026345;
    q_solid.element_position_(75) = -0.01441522951114998;
    q_solid.element_position_(76) = 0.03853468741828525;
    q_solid.element_position_(77) = 0.974176595435944;
    q_solid.element_position_(78) = 0.07916144913106371;
    q_solid.element_position_(79) = 0.0891334591996562;
    q_solid.element_position_(80) = 0.0835795633628921;

    // Results for D.
    result_local_D(0, 0) = 0.1954215935447702;
    result_local_D(0, 3) = 0.01819251719410916;
    result_local_D(0, 6) = 0.0989414627904878;
    result_local_D(0, 9) = -0.01347747062937554;
    result_local_D(1, 1) = 0.1954215935447702;
    result_local_D(1, 4) = 0.01819251719410916;
    result_local_D(1, 7) = 0.0989414627904878;
    result_local_D(1, 10) = -0.01347747062937554;
    result_local_D(2, 2) = 0.1954215935447702;
    result_local_D(2, 5) = 0.01819251719410916;
    result_local_D(2, 8) = 0.0989414627904878;
    result_local_D(2, 11) = -0.01347747062937554;
    result_local_D(3, 0) = 0.0946778368171451;
    result_local_D(3, 3) = 0.01347747062937554;
    result_local_D(3, 6) = 0.2300774860559777;
    result_local_D(3, 9) = -0.02083257297403942;
    result_local_D(4, 1) = 0.0946778368171451;
    result_local_D(4, 4) = 0.01347747062937554;
    result_local_D(4, 7) = 0.2300774860559777;
    result_local_D(4, 10) = -0.02083257297403942;
    result_local_D(5, 2) = 0.0946778368171451;
    result_local_D(5, 5) = 0.01347747062937554;
    result_local_D(5, 8) = 0.2300774860559777;
    result_local_D(5, 11) = -0.02083257297403942;

    // Results for M.
    result_local_M(0, 0) = -0.0001077032778170685;
    result_local_M(0, 3) = 0.000183859156967552;
    result_local_M(0, 6) = -0.0002467975479637281;
    result_local_M(0, 9) = 0.0001446044575337007;
    result_local_M(0, 12) = 0.0001617701029583933;
    result_local_M(0, 15) = -0.000269549696779558;
    result_local_M(0, 18) = 0.0003621891655451537;
    result_local_M(0, 21) = -0.0002172585422474238;
    result_local_M(0, 24) = 0.001238588218574907;
    result_local_M(0, 27) = -0.002921335051479947;
    result_local_M(0, 30) = -0.001653113026976068;
    result_local_M(0, 33) = 0.00170946593418483;
    result_local_M(0, 36) = 0.001504962200223149;
    result_local_M(0, 39) = -0.002831921874933576;
    result_local_M(0, 42) = 0.003762831816042866;
    result_local_M(0, 45) = -0.002008123600619574;
    result_local_M(0, 48) = -0.001918456871968761;
    result_local_M(0, 51) = 0.00426731777854126;
    result_local_M(0, 54) = 0.002559354116382863;
    result_local_M(0, 57) = -0.002564951069744772;
    result_local_M(0, 60) = -0.02007351297941973;
    result_local_M(0, 63) = -0.01557977776397687;
    result_local_M(0, 66) = 0.04678061508189529;
    result_local_M(0, 69) = 0.02074778821960167;
    result_local_M(0, 72) = -0.02444964709875288;
    result_local_M(0, 75) = 0.0311369566485057;
    result_local_M(0, 78) = 0.2546449018409807;
    result_local_M(1, 1) = -0.0001077032778170685;
    result_local_M(1, 4) = 0.000183859156967552;
    result_local_M(1, 7) = -0.0002467975479637281;
    result_local_M(1, 10) = 0.0001446044575337007;
    result_local_M(1, 13) = 0.0001617701029583933;
    result_local_M(1, 16) = -0.000269549696779558;
    result_local_M(1, 19) = 0.0003621891655451537;
    result_local_M(1, 22) = -0.0002172585422474238;
    result_local_M(1, 25) = 0.001238588218574907;
    result_local_M(1, 28) = -0.002921335051479947;
    result_local_M(1, 31) = -0.001653113026976068;
    result_local_M(1, 34) = 0.00170946593418483;
    result_local_M(1, 37) = 0.001504962200223149;
    result_local_M(1, 40) = -0.002831921874933576;
    result_local_M(1, 43) = 0.003762831816042866;
    result_local_M(1, 46) = -0.002008123600619574;
    result_local_M(1, 49) = -0.001918456871968761;
    result_local_M(1, 52) = 0.00426731777854126;
    result_local_M(1, 55) = 0.002559354116382863;
    result_local_M(1, 58) = -0.002564951069744772;
    result_local_M(1, 61) = -0.02007351297941973;
    result_local_M(1, 64) = -0.01557977776397687;
    result_local_M(1, 67) = 0.04678061508189529;
    result_local_M(1, 70) = 0.02074778821960167;
    result_local_M(1, 73) = -0.02444964709875288;
    result_local_M(1, 76) = 0.0311369566485057;
    result_local_M(1, 79) = 0.2546449018409807;
    result_local_M(2, 2) = -0.0001077032778170685;
    result_local_M(2, 5) = 0.000183859156967552;
    result_local_M(2, 8) = -0.0002467975479637281;
    result_local_M(2, 11) = 0.0001446044575337007;
    result_local_M(2, 14) = 0.0001617701029583933;
    result_local_M(2, 17) = -0.000269549696779558;
    result_local_M(2, 20) = 0.0003621891655451537;
    result_local_M(2, 23) = -0.0002172585422474238;
    result_local_M(2, 26) = 0.001238588218574907;
    result_local_M(2, 29) = -0.002921335051479947;
    result_local_M(2, 32) = -0.001653113026976068;
    result_local_M(2, 35) = 0.00170946593418483;
    result_local_M(2, 38) = 0.001504962200223149;
    result_local_M(2, 41) = -0.002831921874933576;
    result_local_M(2, 44) = 0.003762831816042866;
    result_local_M(2, 47) = -0.002008123600619574;
    result_local_M(2, 50) = -0.001918456871968761;
    result_local_M(2, 53) = 0.00426731777854126;
    result_local_M(2, 56) = 0.002559354116382863;
    result_local_M(2, 59) = -0.002564951069744772;
    result_local_M(2, 62) = -0.02007351297941973;
    result_local_M(2, 65) = -0.01557977776397687;
    result_local_M(2, 68) = 0.04678061508189529;
    result_local_M(2, 71) = 0.02074778821960167;
    result_local_M(2, 74) = -0.02444964709875288;
    result_local_M(2, 77) = 0.0311369566485057;
    result_local_M(2, 80) = 0.2546449018409807;
    result_local_M(3, 0) = -0.000083610419001416;
    result_local_M(3, 3) = 0.0001792401922406897;
    result_local_M(3, 6) = -0.0002353130326138415;
    result_local_M(3, 9) = 0.0001108948450888726;
    result_local_M(3, 12) = 0.0001116217159995857;
    result_local_M(3, 15) = -0.0002312477600850804;
    result_local_M(3, 18) = 0.0003052455177640425;
    result_local_M(3, 21) = -0.0001487178334209422;
    result_local_M(3, 24) = 0.0007062062957066386;
    result_local_M(3, 27) = -0.003139813555240989;
    result_local_M(3, 30) = -0.000942639790380131;
    result_local_M(3, 33) = 0.001402663953243143;
    result_local_M(3, 36) = 0.001926858595529986;
    result_local_M(3, 39) = -0.004827798114983534;
    result_local_M(3, 42) = 0.006146882310466189;
    result_local_M(3, 45) = -0.002486538220402141;
    result_local_M(3, 48) = -0.000976907060923827;
    result_local_M(3, 51) = 0.003965332884094244;
    result_local_M(3, 54) = 0.001308647252788366;
    result_local_M(3, 57) = -0.001839594893596401;
    result_local_M(3, 60) = -0.01153915211054991;
    result_local_M(3, 63) = -0.01418870168320633;
    result_local_M(3, 66) = 0.0954035661922705;
    result_local_M(3, 69) = 0.0184898081975646;
    result_local_M(3, 72) = -0.03607951579754085;
    result_local_M(3, 75) = 0.01573577563841683;
    result_local_M(3, 78) = 0.2556821295538945;
    result_local_M(4, 1) = -0.000083610419001416;
    result_local_M(4, 4) = 0.0001792401922406897;
    result_local_M(4, 7) = -0.0002353130326138415;
    result_local_M(4, 10) = 0.0001108948450888726;
    result_local_M(4, 13) = 0.0001116217159995857;
    result_local_M(4, 16) = -0.0002312477600850804;
    result_local_M(4, 19) = 0.0003052455177640425;
    result_local_M(4, 22) = -0.0001487178334209422;
    result_local_M(4, 25) = 0.0007062062957066386;
    result_local_M(4, 28) = -0.003139813555240989;
    result_local_M(4, 31) = -0.000942639790380131;
    result_local_M(4, 34) = 0.001402663953243143;
    result_local_M(4, 37) = 0.001926858595529986;
    result_local_M(4, 40) = -0.004827798114983534;
    result_local_M(4, 43) = 0.006146882310466189;
    result_local_M(4, 46) = -0.002486538220402141;
    result_local_M(4, 49) = -0.000976907060923827;
    result_local_M(4, 52) = 0.003965332884094244;
    result_local_M(4, 55) = 0.001308647252788366;
    result_local_M(4, 58) = -0.001839594893596401;
    result_local_M(4, 61) = -0.01153915211054991;
    result_local_M(4, 64) = -0.01418870168320633;
    result_local_M(4, 67) = 0.0954035661922705;
    result_local_M(4, 70) = 0.0184898081975646;
    result_local_M(4, 73) = -0.03607951579754085;
    result_local_M(4, 76) = 0.01573577563841683;
    result_local_M(4, 79) = 0.2556821295538945;
    result_local_M(5, 2) = -0.000083610419001416;
    result_local_M(5, 5) = 0.0001792401922406897;
    result_local_M(5, 8) = -0.0002353130326138415;
    result_local_M(5, 11) = 0.0001108948450888726;
    result_local_M(5, 14) = 0.0001116217159995857;
    result_local_M(5, 17) = -0.0002312477600850804;
    result_local_M(5, 20) = 0.0003052455177640425;
    result_local_M(5, 23) = -0.0001487178334209422;
    result_local_M(5, 26) = 0.0007062062957066386;
    result_local_M(5, 29) = -0.003139813555240989;
    result_local_M(5, 32) = -0.000942639790380131;
    result_local_M(5, 35) = 0.001402663953243143;
    result_local_M(5, 38) = 0.001926858595529986;
    result_local_M(5, 41) = -0.004827798114983534;
    result_local_M(5, 44) = 0.006146882310466189;
    result_local_M(5, 47) = -0.002486538220402141;
    result_local_M(5, 50) = -0.000976907060923827;
    result_local_M(5, 53) = 0.003965332884094244;
    result_local_M(5, 56) = 0.001308647252788366;
    result_local_M(5, 59) = -0.001839594893596401;
    result_local_M(5, 62) = -0.01153915211054991;
    result_local_M(5, 65) = -0.01418870168320633;
    result_local_M(5, 68) = 0.0954035661922705;
    result_local_M(5, 71) = 0.0184898081975646;
    result_local_M(5, 74) = -0.03607951579754085;
    result_local_M(5, 77) = 0.01573577563841683;
    result_local_M(5, 80) = 0.2556821295538945;

    // Results for Kappa.
    result_local_kappa(0) = 0.294363056335258;
    result_local_kappa(1) = 0.294363056335258;
    result_local_kappa(2) = 0.294363056335258;
    result_local_kappa(3) = 0.3247553228731228;
    result_local_kappa(4) = 0.3247553228731228;
    result_local_kappa(5) = 0.3247553228731228;

    // Perform the unit tests.
    perform_mortar_pair_unit_test(contact_pair, q_beam, q_beam_rot, q_solid, result_local_D,
        result_local_M, result_local_kappa);
  }

  /**
   * \brief Test a non straight beam in a hex27 element, with line3 mortar shape functions.
   */
  TEST_F(BeamToSolidVolumeMeshtyingPairMortarTest, TestBeamToSolidMeshtyingMortarHermite2Hex27Line3)
  {
    // Element types.
    typedef GEOMETRYPAIR::t_hermite beam_type;
    typedef GEOMETRYPAIR::t_hex27 solid_type;
    typedef GEOMETRYPAIR::t_line3 lambda_type;

    // Create the mesh tying mortar pair.
    BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type, lambda_type>
        contact_pair = BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type,
            lambda_type>();

    // Definition of variables for this test case.
    GEOMETRYPAIR::ElementData<beam_type, double> q_beam;
    GEOMETRYPAIR::ElementData<solid_type, double> q_solid;
    Core::LinAlg::Matrix<9, 1> q_beam_rot;
    Core::LinAlg::SerialDenseMatrix local_D;
    Core::LinAlg::SerialDenseMatrix local_M;
    Core::LinAlg::SerialDenseVector local_kappa;

    // Matrices for the results.
    Core::LinAlg::Matrix<lambda_type::n_dof_, beam_type::n_dof_> result_local_D(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, solid_type::n_dof_> result_local_M(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, 1> result_local_kappa(
        Core::LinAlg::Initialization::zero);

    // Define the geometry of the two elements.
    q_beam.shape_function_data_.ref_length_ = 0.6192043571449604711;
    q_beam.element_position_(0) = 0.15;
    q_beam.element_position_(1) = 0.2;
    q_beam.element_position_(2) = 0.3;
    q_beam.element_position_(3) = 0.5773502691896255;
    q_beam.element_position_(4) = 0.5773502691896258;
    q_beam.element_position_(5) = 0.577350269189626;
    q_beam.element_position_(6) = 0.65;
    q_beam.element_position_(7) = 0.1;
    q_beam.element_position_(8) = 0.1;
    q_beam.element_position_(9) = 0.8017837257372733;
    q_beam.element_position_(10) = -0.5345224838248488;
    q_beam.element_position_(11) = 0.2672612419124244;

    q_beam_rot(0) = 1.674352746442651;
    q_beam_rot(1) = 0.1425949677148126;
    q_beam_rot(2) = 1.0831163124736984;
    q_beam_rot(3) = 1.4331999091513161;
    q_beam_rot(4) = -0.6560404572957742;
    q_beam_rot(5) = -0.2491376152457331;
    q_beam_rot(6) = 0.0;
    q_beam_rot(7) = 0.0;
    q_beam_rot(8) = 0.0;

    // Positional DOFs of the solid.
    q_solid.element_position_(0) = -0.954193516126594;
    q_solid.element_position_(1) = -0.975482672114534;
    q_solid.element_position_(2) = -1.00311316662815;
    q_solid.element_position_(3) = 0.923762409575691;
    q_solid.element_position_(4) = -1.010615727466753;
    q_solid.element_position_(5) = -1.014160992102648;
    q_solid.element_position_(6) = 0.905563488894572;
    q_solid.element_position_(7) = 1.069973754602123;
    q_solid.element_position_(8) = -0.935488788632622;
    q_solid.element_position_(9) = -1.047838992508937;
    q_solid.element_position_(10) = 1.08888017769104;
    q_solid.element_position_(11) = -1.036911970151991;
    q_solid.element_position_(12) = -1.089167378981223;
    q_solid.element_position_(13) = -1.063820485514786;
    q_solid.element_position_(14) = 1.081000671652452;
    q_solid.element_position_(15) = 0.972887477248942;
    q_solid.element_position_(16) = -1.008995039455167;
    q_solid.element_position_(17) = 0.923459231565508;
    q_solid.element_position_(18) = 1.089192110164642;
    q_solid.element_position_(19) = 1.026798931538616;
    q_solid.element_position_(20) = 0.964761788314679;
    q_solid.element_position_(21) = -0.941489453299989;
    q_solid.element_position_(22) = 0.954327975326486;
    q_solid.element_position_(23) = 0.963168067937965;
    q_solid.element_position_(24) = 0.0925155012591516;
    q_solid.element_position_(25) = -0.973383316878771;
    q_solid.element_position_(26) = -1.065800877068667;
    q_solid.element_position_(27) = 0.993751379524698;
    q_solid.element_position_(28) = -0.089078411148891;
    q_solid.element_position_(29) = -1.080589211553427;
    q_solid.element_position_(30) = 0.0823626211382376;
    q_solid.element_position_(31) = 1.033905181036807;
    q_solid.element_position_(32) = -0.952741283444911;
    q_solid.element_position_(33) = -1.024714325285187;
    q_solid.element_position_(34) = -0.0951797166281465;
    q_solid.element_position_(35) = -0.917694055170081;
    q_solid.element_position_(36) = -0.94955183268814;
    q_solid.element_position_(37) = -0.936209915275568;
    q_solid.element_position_(38) = -0.07531489653912402;
    q_solid.element_position_(39) = 0.977045420134747;
    q_solid.element_position_(40) = -0.915831507095275;
    q_solid.element_position_(41) = -0.03604489795829757;
    q_solid.element_position_(42) = 1.044996295051512;
    q_solid.element_position_(43) = 0.912268002554735;
    q_solid.element_position_(44) = 0.026884640088015;
    q_solid.element_position_(45) = -1.003774079332873;
    q_solid.element_position_(46) = 1.016364821141699;
    q_solid.element_position_(47) = -0.04547583423590318;
    q_solid.element_position_(48) = -0.07606458957105064;
    q_solid.element_position_(49) = -0.970767004162723;
    q_solid.element_position_(50) = 1.09197908262736;
    q_solid.element_position_(51) = 1.000582069178839;
    q_solid.element_position_(52) = 0.02495199075528687;
    q_solid.element_position_(53) = 0.942442567644017;
    q_solid.element_position_(54) = -0.0840408739597993;
    q_solid.element_position_(55) = 1.082175787959178;
    q_solid.element_position_(56) = 1.016887685734179;
    q_solid.element_position_(57) = -1.02950388437141;
    q_solid.element_position_(58) = -0.03189318909487643;
    q_solid.element_position_(59) = 0.962850907596607;
    q_solid.element_position_(60) = 0.06298633006639215;
    q_solid.element_position_(61) = -0.06929279965773389;
    q_solid.element_position_(62) = -0.963099037764495;
    q_solid.element_position_(63) = 0.005860067798445501;
    q_solid.element_position_(64) = -1.087116820781742;
    q_solid.element_position_(65) = 0.01143091251576034;
    q_solid.element_position_(66) = 0.990119207129434;
    q_solid.element_position_(67) = -0.07554386166147545;
    q_solid.element_position_(68) = 0.02213195187583161;
    q_solid.element_position_(69) = -0.0892155216099889;
    q_solid.element_position_(70) = 1.051578415322994;
    q_solid.element_position_(71) = 0.01644372121101889;
    q_solid.element_position_(72) = -1.084952712553014;
    q_solid.element_position_(73) = -0.06218066414480983;
    q_solid.element_position_(74) = 0.03589340707026345;
    q_solid.element_position_(75) = -0.01441522951114998;
    q_solid.element_position_(76) = 0.03853468741828525;
    q_solid.element_position_(77) = 0.974176595435944;
    q_solid.element_position_(78) = 0.07916144913106371;
    q_solid.element_position_(79) = 0.0891334591996562;
    q_solid.element_position_(80) = 0.0835795633628921;

    // Results for D.
    result_local_D(0, 0) = 0.0945173766014223;
    result_local_D(0, 3) = 0.005144868995407634;
    result_local_D(0, 6) = -0.0132666172503834;
    result_local_D(0, 9) = 0.0004298224306740067;
    result_local_D(1, 1) = 0.0945173766014223;
    result_local_D(1, 4) = 0.005144868995407634;
    result_local_D(1, 7) = -0.0132666172503834;
    result_local_D(1, 10) = 0.0004298224306740067;
    result_local_D(2, 2) = 0.0945173766014223;
    result_local_D(2, 5) = 0.005144868995407634;
    result_local_D(2, 8) = -0.0132666172503834;
    result_local_D(2, 11) = 0.0004298224306740067;
    result_local_D(3, 0) = -0.006226380126202878;
    result_local_D(3, 3) = 0.0004298224306740081;
    result_local_D(3, 6) = 0.1178694060151065;
    result_local_D(3, 9) = -0.006925279913989872;
    result_local_D(4, 1) = -0.006226380126202878;
    result_local_D(4, 4) = 0.0004298224306740081;
    result_local_D(4, 7) = 0.1178694060151065;
    result_local_D(4, 10) = -0.006925279913989872;
    result_local_D(5, 2) = -0.006226380126202878;
    result_local_D(5, 5) = 0.0004298224306740081;
    result_local_D(5, 8) = 0.1178694060151065;
    result_local_D(5, 11) = -0.006925279913989872;
    result_local_D(6, 0) = 0.2018084338866959;
    result_local_D(6, 3) = 0.02609529639740305;
    result_local_D(6, 6) = 0.2244161600817424;
    result_local_D(6, 9) = -0.02781458612009909;
    result_local_D(7, 1) = 0.2018084338866959;
    result_local_D(7, 4) = 0.02609529639740305;
    result_local_D(7, 7) = 0.2244161600817424;
    result_local_D(7, 10) = -0.02781458612009909;
    result_local_D(8, 2) = 0.2018084338866959;
    result_local_D(8, 5) = 0.02609529639740305;
    result_local_D(8, 8) = 0.2244161600817424;
    result_local_D(8, 11) = -0.02781458612009909;

    // Results for M.
    result_local_M(0, 0) = -0.00003151528397381564;
    result_local_M(0, 3) = 0.00003870378441565258;
    result_local_M(0, 6) = -0.00005217861132586865;
    result_local_M(0, 9) = 0.00004202133471259778;
    result_local_M(0, 12) = 0.00005393155948004919;
    result_local_M(0, 15) = -0.00006933008603500568;
    result_local_M(0, 18) = 0.0000929720860375371;
    result_local_M(0, 21) = -0.0000717483673992343;
    result_local_M(0, 24) = 0.0005220391763269595;
    result_local_M(0, 27) = -0.0006036502013382263;
    result_local_M(0, 30) = -0.0006866010798555642;
    result_local_M(0, 33) = 0.0005121111573061156;
    result_local_M(0, 36) = 0.0002091594663718008;
    result_local_M(0, 39) = -0.0000820086386055577;
    result_local_M(0, 42) = 0.0001395239598001106;
    result_local_M(0, 45) = -0.000287154447193771;
    result_local_M(0, 48) = -0.000873043701142296;
    result_local_M(0, 51) = 0.001103000371781354;
    result_local_M(0, 54) = 0.001147408230354483;
    result_local_M(0, 57) = -0.000882929111797573;
    result_local_M(0, 60) = -0.00888806582170079;
    result_local_M(0, 63) = -0.004610492425046012;
    result_local_M(0, 66) = -0.0001352539147201764;
    result_local_M(0, 69) = 0.006102195089193629;
    result_local_M(0, 72) = -0.003002610976073077;
    result_local_M(0, 75) = 0.01489399552684807;
    result_local_M(0, 78) = 0.07667028027461747;
    result_local_M(1, 1) = -0.00003151528397381564;
    result_local_M(1, 4) = 0.00003870378441565258;
    result_local_M(1, 7) = -0.00005217861132586865;
    result_local_M(1, 10) = 0.00004202133471259778;
    result_local_M(1, 13) = 0.00005393155948004919;
    result_local_M(1, 16) = -0.00006933008603500568;
    result_local_M(1, 19) = 0.0000929720860375371;
    result_local_M(1, 22) = -0.0000717483673992343;
    result_local_M(1, 25) = 0.0005220391763269595;
    result_local_M(1, 28) = -0.0006036502013382263;
    result_local_M(1, 31) = -0.0006866010798555642;
    result_local_M(1, 34) = 0.0005121111573061156;
    result_local_M(1, 37) = 0.0002091594663718008;
    result_local_M(1, 40) = -0.0000820086386055577;
    result_local_M(1, 43) = 0.0001395239598001106;
    result_local_M(1, 46) = -0.000287154447193771;
    result_local_M(1, 49) = -0.000873043701142296;
    result_local_M(1, 52) = 0.001103000371781354;
    result_local_M(1, 55) = 0.001147408230354483;
    result_local_M(1, 58) = -0.000882929111797573;
    result_local_M(1, 61) = -0.00888806582170079;
    result_local_M(1, 64) = -0.004610492425046012;
    result_local_M(1, 67) = -0.0001352539147201764;
    result_local_M(1, 70) = 0.006102195089193629;
    result_local_M(1, 73) = -0.003002610976073077;
    result_local_M(1, 76) = 0.01489399552684807;
    result_local_M(1, 79) = 0.07667028027461747;
    result_local_M(2, 2) = -0.00003151528397381564;
    result_local_M(2, 5) = 0.00003870378441565258;
    result_local_M(2, 8) = -0.00005217861132586865;
    result_local_M(2, 11) = 0.00004202133471259778;
    result_local_M(2, 14) = 0.00005393155948004919;
    result_local_M(2, 17) = -0.00006933008603500568;
    result_local_M(2, 20) = 0.0000929720860375371;
    result_local_M(2, 23) = -0.0000717483673992343;
    result_local_M(2, 26) = 0.0005220391763269595;
    result_local_M(2, 29) = -0.0006036502013382263;
    result_local_M(2, 32) = -0.0006866010798555642;
    result_local_M(2, 35) = 0.0005121111573061156;
    result_local_M(2, 38) = 0.0002091594663718008;
    result_local_M(2, 41) = -0.0000820086386055577;
    result_local_M(2, 44) = 0.0001395239598001106;
    result_local_M(2, 47) = -0.000287154447193771;
    result_local_M(2, 50) = -0.000873043701142296;
    result_local_M(2, 53) = 0.001103000371781354;
    result_local_M(2, 56) = 0.001147408230354483;
    result_local_M(2, 59) = -0.000882929111797573;
    result_local_M(2, 62) = -0.00888806582170079;
    result_local_M(2, 65) = -0.004610492425046012;
    result_local_M(2, 68) = -0.0001352539147201764;
    result_local_M(2, 71) = 0.006102195089193629;
    result_local_M(2, 74) = -0.003002610976073077;
    result_local_M(2, 77) = 0.01489399552684807;
    result_local_M(2, 80) = 0.07667028027461747;
    result_local_M(3, 0) = -0.000007422425158163163;
    result_local_M(3, 3) = 0.0000340848196887903;
    result_local_M(3, 6) = -0.00004069409597598205;
    result_local_M(3, 9) = 0.0000083117222677697;
    result_local_M(3, 12) = 0.000003783172521241587;
    result_local_M(3, 15) = -0.00003102814934052809;
    result_local_M(3, 18) = 0.00003602843825642592;
    result_local_M(3, 21) = -0.000003207658572752683;
    result_local_M(3, 24) = -0.00001034274654130886;
    result_local_M(3, 27) = -0.000822128705099268;
    result_local_M(3, 30) = 0.00002387215674037215;
    result_local_M(3, 33) = 0.0002053091763644281;
    result_local_M(3, 36) = 0.0006310558616786379;
    result_local_M(3, 39) = -0.002077884878655517;
    result_local_M(3, 42) = 0.002523574454223434;
    result_local_M(3, 45) = -0.0007655690669763382;
    result_local_M(3, 48) = 0.00006850610990263961;
    result_local_M(3, 51) = 0.000801015477334338;
    result_local_M(3, 54) = -0.0001032986332400147;
    result_local_M(3, 57) = -0.000157572935649202;
    result_local_M(3, 60) = -0.0003537049528309615;
    result_local_M(3, 63) = -0.003219416344275473;
    result_local_M(3, 66) = 0.04848769719565501;
    result_local_M(3, 69) = 0.003844215067156553;
    result_local_M(3, 72) = -0.01463247967486104;
    result_local_M(3, 75) = -0.0005071854832407931;
    result_local_M(3, 78) = 0.07770750798753133;
    result_local_M(4, 1) = -0.000007422425158163163;
    result_local_M(4, 4) = 0.0000340848196887903;
    result_local_M(4, 7) = -0.00004069409597598205;
    result_local_M(4, 10) = 0.0000083117222677697;
    result_local_M(4, 13) = 0.000003783172521241587;
    result_local_M(4, 16) = -0.00003102814934052809;
    result_local_M(4, 19) = 0.00003602843825642592;
    result_local_M(4, 22) = -0.000003207658572752683;
    result_local_M(4, 25) = -0.00001034274654130886;
    result_local_M(4, 28) = -0.000822128705099268;
    result_local_M(4, 31) = 0.00002387215674037215;
    result_local_M(4, 34) = 0.0002053091763644281;
    result_local_M(4, 37) = 0.0006310558616786379;
    result_local_M(4, 40) = -0.002077884878655517;
    result_local_M(4, 43) = 0.002523574454223434;
    result_local_M(4, 46) = -0.0007655690669763382;
    result_local_M(4, 49) = 0.00006850610990263961;
    result_local_M(4, 52) = 0.000801015477334338;
    result_local_M(4, 55) = -0.0001032986332400147;
    result_local_M(4, 58) = -0.000157572935649202;
    result_local_M(4, 61) = -0.0003537049528309615;
    result_local_M(4, 64) = -0.003219416344275473;
    result_local_M(4, 67) = 0.04848769719565501;
    result_local_M(4, 70) = 0.003844215067156553;
    result_local_M(4, 73) = -0.01463247967486104;
    result_local_M(4, 76) = -0.0005071854832407931;
    result_local_M(4, 79) = 0.07770750798753133;
    result_local_M(5, 2) = -0.000007422425158163163;
    result_local_M(5, 5) = 0.0000340848196887903;
    result_local_M(5, 8) = -0.00004069409597598205;
    result_local_M(5, 11) = 0.0000083117222677697;
    result_local_M(5, 14) = 0.000003783172521241587;
    result_local_M(5, 17) = -0.00003102814934052809;
    result_local_M(5, 20) = 0.00003602843825642592;
    result_local_M(5, 23) = -0.000003207658572752683;
    result_local_M(5, 26) = -0.00001034274654130886;
    result_local_M(5, 29) = -0.000822128705099268;
    result_local_M(5, 32) = 0.00002387215674037215;
    result_local_M(5, 35) = 0.0002053091763644281;
    result_local_M(5, 38) = 0.0006310558616786379;
    result_local_M(5, 41) = -0.002077884878655517;
    result_local_M(5, 44) = 0.002523574454223434;
    result_local_M(5, 47) = -0.0007655690669763382;
    result_local_M(5, 50) = 0.00006850610990263961;
    result_local_M(5, 53) = 0.000801015477334338;
    result_local_M(5, 56) = -0.0001032986332400147;
    result_local_M(5, 59) = -0.000157572935649202;
    result_local_M(5, 62) = -0.0003537049528309615;
    result_local_M(5, 65) = -0.003219416344275473;
    result_local_M(5, 68) = 0.04848769719565501;
    result_local_M(5, 71) = 0.003844215067156553;
    result_local_M(5, 74) = -0.01463247967486104;
    result_local_M(5, 77) = -0.0005071854832407931;
    result_local_M(5, 80) = 0.07770750798753133;
    result_local_M(6, 0) = -0.0001523759876865057;
    result_local_M(6, 3) = 0.0002903107451037987;
    result_local_M(6, 6) = -0.0003892378732757189;
    result_local_M(6, 9) = 0.0002051662456422059;
    result_local_M(6, 12) = 0.0002156770869566883;
    result_local_M(6, 15) = -0.0004004392214891046;
    result_local_M(6, 18) = 0.0005384341590152331;
    result_local_M(6, 21) = -0.000291020349696379;
    result_local_M(6, 24) = 0.001433098084495895;
    result_local_M(6, 27) = -0.004635369700283444;
    result_local_M(6, 30) = -0.001933023894241007;
    result_local_M(6, 33) = 0.002394709553757429;
    result_local_M(6, 36) = 0.002591605467702696;
    result_local_M(6, 39) = -0.005499826472656036;
    result_local_M(6, 42) = 0.007246615712485509;
    result_local_M(6, 45) = -0.003441938306851605;
    result_local_M(6, 48) = -0.002090826341652932;
    result_local_M(6, 51) = 0.006328634813519812;
    result_local_M(6, 54) = 0.002823891772056761;
    result_local_M(6, 57) = -0.003364043915894399;
    result_local_M(6, 60) = -0.0223708943154379;
    result_local_M(6, 63) = -0.02193857067786171;
    result_local_M(6, 66) = 0.0938317379932309;
    result_local_M(6, 69) = 0.02929118626081608;
    result_local_M(6, 72) = -0.04289407224535963;
    result_local_M(6, 75) = 0.03248592224331525;
    result_local_M(6, 78) = 0.3559492431327263;
    result_local_M(7, 1) = -0.0001523759876865057;
    result_local_M(7, 4) = 0.0002903107451037987;
    result_local_M(7, 7) = -0.0003892378732757189;
    result_local_M(7, 10) = 0.0002051662456422059;
    result_local_M(7, 13) = 0.0002156770869566883;
    result_local_M(7, 16) = -0.0004004392214891046;
    result_local_M(7, 19) = 0.0005384341590152331;
    result_local_M(7, 22) = -0.000291020349696379;
    result_local_M(7, 25) = 0.001433098084495895;
    result_local_M(7, 28) = -0.004635369700283444;
    result_local_M(7, 31) = -0.001933023894241007;
    result_local_M(7, 34) = 0.002394709553757429;
    result_local_M(7, 37) = 0.002591605467702696;
    result_local_M(7, 40) = -0.005499826472656036;
    result_local_M(7, 43) = 0.007246615712485509;
    result_local_M(7, 46) = -0.003441938306851605;
    result_local_M(7, 49) = -0.002090826341652932;
    result_local_M(7, 52) = 0.006328634813519812;
    result_local_M(7, 55) = 0.002823891772056761;
    result_local_M(7, 58) = -0.003364043915894399;
    result_local_M(7, 61) = -0.0223708943154379;
    result_local_M(7, 64) = -0.02193857067786171;
    result_local_M(7, 67) = 0.0938317379932309;
    result_local_M(7, 70) = 0.02929118626081608;
    result_local_M(7, 73) = -0.04289407224535963;
    result_local_M(7, 76) = 0.03248592224331525;
    result_local_M(7, 79) = 0.3559492431327263;
    result_local_M(8, 2) = -0.0001523759876865057;
    result_local_M(8, 5) = 0.0002903107451037987;
    result_local_M(8, 8) = -0.0003892378732757189;
    result_local_M(8, 11) = 0.0002051662456422059;
    result_local_M(8, 14) = 0.0002156770869566883;
    result_local_M(8, 17) = -0.0004004392214891046;
    result_local_M(8, 20) = 0.0005384341590152331;
    result_local_M(8, 23) = -0.000291020349696379;
    result_local_M(8, 26) = 0.001433098084495895;
    result_local_M(8, 29) = -0.004635369700283444;
    result_local_M(8, 32) = -0.001933023894241007;
    result_local_M(8, 35) = 0.002394709553757429;
    result_local_M(8, 38) = 0.002591605467702696;
    result_local_M(8, 41) = -0.005499826472656036;
    result_local_M(8, 44) = 0.007246615712485509;
    result_local_M(8, 47) = -0.003441938306851605;
    result_local_M(8, 50) = -0.002090826341652932;
    result_local_M(8, 53) = 0.006328634813519812;
    result_local_M(8, 56) = 0.002823891772056761;
    result_local_M(8, 59) = -0.003364043915894399;
    result_local_M(8, 62) = -0.0223708943154379;
    result_local_M(8, 65) = -0.02193857067786171;
    result_local_M(8, 68) = 0.0938317379932309;
    result_local_M(8, 71) = 0.02929118626081608;
    result_local_M(8, 74) = -0.04289407224535963;
    result_local_M(8, 77) = 0.03248592224331525;
    result_local_M(8, 80) = 0.3559492431327263;

    // Results for Kappa.
    result_local_kappa(0) = 0.0812507593510389;
    result_local_kappa(1) = 0.0812507593510389;
    result_local_kappa(2) = 0.0812507593510389;
    result_local_kappa(3) = 0.1116430258889036;
    result_local_kappa(4) = 0.1116430258889036;
    result_local_kappa(5) = 0.1116430258889036;
    result_local_kappa(6) = 0.4262245939684383;
    result_local_kappa(7) = 0.4262245939684383;
    result_local_kappa(8) = 0.4262245939684383;

    // Perform the unit tests.
    perform_mortar_pair_unit_test(contact_pair, q_beam, q_beam_rot, q_solid, result_local_D,
        result_local_M, result_local_kappa);
  }

  /**
   * \brief Test a non straight beam in a hex27 element, with line4 mortar shape functions.
   */
  TEST_F(BeamToSolidVolumeMeshtyingPairMortarTest, TestBeamToSolidMeshtyingMortarHermite2Hex27Line4)
  {
    // Element types.
    typedef GEOMETRYPAIR::t_hermite beam_type;
    typedef GEOMETRYPAIR::t_hex27 solid_type;
    typedef GEOMETRYPAIR::t_line4 lambda_type;

    // Create the mesh tying mortar pair.
    BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type, lambda_type>
        contact_pair = BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type,
            lambda_type>();

    // Definition of variables for this test case.
    GEOMETRYPAIR::ElementData<beam_type, double> q_beam;
    GEOMETRYPAIR::ElementData<solid_type, double> q_solid;
    Core::LinAlg::Matrix<9, 1> q_beam_rot;
    Core::LinAlg::SerialDenseMatrix local_D;
    Core::LinAlg::SerialDenseMatrix local_M;
    Core::LinAlg::SerialDenseVector local_kappa;

    // Matrices for the results.
    Core::LinAlg::Matrix<lambda_type::n_dof_, beam_type::n_dof_> result_local_D(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, solid_type::n_dof_> result_local_M(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, 1> result_local_kappa(
        Core::LinAlg::Initialization::zero);

    // Define the geometry of the two elements.
    q_beam.shape_function_data_.ref_length_ = 0.6192043571449604711;
    q_beam.element_position_(0) = 0.15;
    q_beam.element_position_(1) = 0.2;
    q_beam.element_position_(2) = 0.3;
    q_beam.element_position_(3) = 0.5773502691896255;
    q_beam.element_position_(4) = 0.5773502691896258;
    q_beam.element_position_(5) = 0.577350269189626;
    q_beam.element_position_(6) = 0.65;
    q_beam.element_position_(7) = 0.1;
    q_beam.element_position_(8) = 0.1;
    q_beam.element_position_(9) = 0.8017837257372733;
    q_beam.element_position_(10) = -0.5345224838248488;
    q_beam.element_position_(11) = 0.2672612419124244;

    q_beam_rot(0) = 1.674352746442651;
    q_beam_rot(1) = 0.1425949677148126;
    q_beam_rot(2) = 1.0831163124736984;
    q_beam_rot(3) = 1.4331999091513161;
    q_beam_rot(4) = -0.6560404572957742;
    q_beam_rot(5) = -0.2491376152457331;
    q_beam_rot(6) = 0.0;
    q_beam_rot(7) = 0.0;
    q_beam_rot(8) = 0.0;

    // Positional DOFs of the solid.
    q_solid.element_position_(0) = -0.954193516126594;
    q_solid.element_position_(1) = -0.975482672114534;
    q_solid.element_position_(2) = -1.00311316662815;
    q_solid.element_position_(3) = 0.923762409575691;
    q_solid.element_position_(4) = -1.010615727466753;
    q_solid.element_position_(5) = -1.014160992102648;
    q_solid.element_position_(6) = 0.905563488894572;
    q_solid.element_position_(7) = 1.069973754602123;
    q_solid.element_position_(8) = -0.935488788632622;
    q_solid.element_position_(9) = -1.047838992508937;
    q_solid.element_position_(10) = 1.08888017769104;
    q_solid.element_position_(11) = -1.036911970151991;
    q_solid.element_position_(12) = -1.089167378981223;
    q_solid.element_position_(13) = -1.063820485514786;
    q_solid.element_position_(14) = 1.081000671652452;
    q_solid.element_position_(15) = 0.972887477248942;
    q_solid.element_position_(16) = -1.008995039455167;
    q_solid.element_position_(17) = 0.923459231565508;
    q_solid.element_position_(18) = 1.089192110164642;
    q_solid.element_position_(19) = 1.026798931538616;
    q_solid.element_position_(20) = 0.964761788314679;
    q_solid.element_position_(21) = -0.941489453299989;
    q_solid.element_position_(22) = 0.954327975326486;
    q_solid.element_position_(23) = 0.963168067937965;
    q_solid.element_position_(24) = 0.0925155012591516;
    q_solid.element_position_(25) = -0.973383316878771;
    q_solid.element_position_(26) = -1.065800877068667;
    q_solid.element_position_(27) = 0.993751379524698;
    q_solid.element_position_(28) = -0.089078411148891;
    q_solid.element_position_(29) = -1.080589211553427;
    q_solid.element_position_(30) = 0.0823626211382376;
    q_solid.element_position_(31) = 1.033905181036807;
    q_solid.element_position_(32) = -0.952741283444911;
    q_solid.element_position_(33) = -1.024714325285187;
    q_solid.element_position_(34) = -0.0951797166281465;
    q_solid.element_position_(35) = -0.917694055170081;
    q_solid.element_position_(36) = -0.94955183268814;
    q_solid.element_position_(37) = -0.936209915275568;
    q_solid.element_position_(38) = -0.07531489653912402;
    q_solid.element_position_(39) = 0.977045420134747;
    q_solid.element_position_(40) = -0.915831507095275;
    q_solid.element_position_(41) = -0.03604489795829757;
    q_solid.element_position_(42) = 1.044996295051512;
    q_solid.element_position_(43) = 0.912268002554735;
    q_solid.element_position_(44) = 0.026884640088015;
    q_solid.element_position_(45) = -1.003774079332873;
    q_solid.element_position_(46) = 1.016364821141699;
    q_solid.element_position_(47) = -0.04547583423590318;
    q_solid.element_position_(48) = -0.07606458957105064;
    q_solid.element_position_(49) = -0.970767004162723;
    q_solid.element_position_(50) = 1.09197908262736;
    q_solid.element_position_(51) = 1.000582069178839;
    q_solid.element_position_(52) = 0.02495199075528687;
    q_solid.element_position_(53) = 0.942442567644017;
    q_solid.element_position_(54) = -0.0840408739597993;
    q_solid.element_position_(55) = 1.082175787959178;
    q_solid.element_position_(56) = 1.016887685734179;
    q_solid.element_position_(57) = -1.02950388437141;
    q_solid.element_position_(58) = -0.03189318909487643;
    q_solid.element_position_(59) = 0.962850907596607;
    q_solid.element_position_(60) = 0.06298633006639215;
    q_solid.element_position_(61) = -0.06929279965773389;
    q_solid.element_position_(62) = -0.963099037764495;
    q_solid.element_position_(63) = 0.005860067798445501;
    q_solid.element_position_(64) = -1.087116820781742;
    q_solid.element_position_(65) = 0.01143091251576034;
    q_solid.element_position_(66) = 0.990119207129434;
    q_solid.element_position_(67) = -0.07554386166147545;
    q_solid.element_position_(68) = 0.02213195187583161;
    q_solid.element_position_(69) = -0.0892155216099889;
    q_solid.element_position_(70) = 1.051578415322994;
    q_solid.element_position_(71) = 0.01644372121101889;
    q_solid.element_position_(72) = -1.084952712553014;
    q_solid.element_position_(73) = -0.06218066414480983;
    q_solid.element_position_(74) = 0.03589340707026345;
    q_solid.element_position_(75) = -0.01441522951114998;
    q_solid.element_position_(76) = 0.03853468741828525;
    q_solid.element_position_(77) = 0.974176595435944;
    q_solid.element_position_(78) = 0.07916144913106371;
    q_solid.element_position_(79) = 0.0891334591996562;
    q_solid.element_position_(80) = 0.0835795633628921;

    // Results for D.
    result_local_D(0, 0) = 0.05837517950584759;
    result_local_D(0, 3) = 0.002017208305318981;
    result_local_D(0, 6) = 0.005829701162185071;
    result_local_D(0, 9) = -0.0002955710710872768;
    result_local_D(1, 1) = 0.05837517950584759;
    result_local_D(1, 4) = 0.002017208305318981;
    result_local_D(1, 7) = 0.005829701162185071;
    result_local_D(1, 10) = -0.0002955710710872768;
    result_local_D(2, 2) = 0.05837517950584759;
    result_local_D(2, 5) = 0.002017208305318981;
    result_local_D(2, 8) = 0.005829701162185071;
    result_local_D(2, 11) = -0.0002955710710872768;
    result_local_D(3, 0) = 0.004689762733534804;
    result_local_D(3, 3) = 0.0002955710710872767;
    result_local_D(3, 6) = 0.07072106759232025;
    result_local_D(3, 9) = -0.002723063147216203;
    result_local_D(4, 1) = 0.004689762733534804;
    result_local_D(4, 4) = 0.0002955710710872767;
    result_local_D(4, 7) = 0.07072106759232025;
    result_local_D(4, 10) = -0.002723063147216203;
    result_local_D(5, 2) = 0.004689762733534804;
    result_local_D(5, 5) = 0.0002955710710872767;
    result_local_D(5, 8) = 0.07072106759232025;
    result_local_D(5, 11) = -0.002723063147216203;
    result_local_D(6, 0) = 0.184104753994235;
    result_local_D(6, 3) = 0.0191687182192921;
    result_local_D(6, 6) = 0.02686710479294802;
    result_local_D(6, 9) = -0.00825428928975331;
    result_local_D(7, 1) = 0.184104753994235;
    result_local_D(7, 4) = 0.0191687182192921;
    result_local_D(7, 7) = 0.02686710479294802;
    result_local_D(7, 10) = -0.00825428928975331;
    result_local_D(8, 2) = 0.184104753994235;
    result_local_D(8, 5) = 0.0191687182192921;
    result_local_D(8, 8) = 0.02686710479294802;
    result_local_D(8, 11) = -0.00825428928975331;
    result_local_D(9, 0) = 0.04292973412829792;
    result_local_D(9, 3) = 0.01018849022778634;
    result_local_D(9, 6) = 0.2256010752990122;
    result_local_D(9, 9) = -0.02303712009535817;
    result_local_D(10, 1) = 0.04292973412829792;
    result_local_D(10, 4) = 0.01018849022778634;
    result_local_D(10, 7) = 0.2256010752990122;
    result_local_D(10, 10) = -0.02303712009535817;
    result_local_D(11, 2) = 0.04292973412829792;
    result_local_D(11, 5) = 0.01018849022778634;
    result_local_D(11, 8) = 0.2256010752990122;
    result_local_D(11, 11) = -0.02303712009535817;

    // Results for M.
    result_local_M(0, 0) = -0.00001461194246558386;
    result_local_M(0, 3) = 0.00001833669925923309;
    result_local_M(0, 6) = -0.00002316894446404601;
    result_local_M(0, 9) = 0.00001862385852622348;
    result_local_M(0, 12) = 0.00002478183538899803;
    result_local_M(0, 15) = -0.00003092822270269711;
    result_local_M(0, 18) = 0.00003933141761646087;
    result_local_M(0, 21) = -0.00003170470826846835;
    result_local_M(0, 24) = 0.000289611154012035;
    result_local_M(0, 27) = -0.000354193612385874;
    result_local_M(0, 30) = -0.0003700376031547919;
    result_local_M(0, 33) = 0.0002739128564842182;
    result_local_M(0, 36) = 0.0001669058985750177;
    result_local_M(0, 39) = -0.0002662387446394654;
    result_local_M(0, 42) = 0.0003280532723283626;
    result_local_M(0, 45) = -0.0002092686733702037;
    result_local_M(0, 48) = -0.0004852236293687247;
    result_local_M(0, 51) = 0.0005843160727609264;
    result_local_M(0, 54) = 0.0006210828465263479;
    result_local_M(0, 57) = -0.0004587608617178904;
    result_local_M(0, 60) = -0.005394220007674447;
    result_local_M(0, 63) = -0.003004256452075587;
    result_local_M(0, 66) = 0.005660137909039662;
    result_local_M(0, 69) = 0.003812944696864679;
    result_local_M(0, 72) = -0.003327732605285794;
    result_local_M(0, 75) = 0.00898500798502095;
    result_local_M(0, 78) = 0.05735218017320313;
    result_local_M(1, 1) = -0.00001461194246558386;
    result_local_M(1, 4) = 0.00001833669925923309;
    result_local_M(1, 7) = -0.00002316894446404601;
    result_local_M(1, 10) = 0.00001862385852622348;
    result_local_M(1, 13) = 0.00002478183538899803;
    result_local_M(1, 16) = -0.00003092822270269711;
    result_local_M(1, 19) = 0.00003933141761646087;
    result_local_M(1, 22) = -0.00003170470826846835;
    result_local_M(1, 25) = 0.000289611154012035;
    result_local_M(1, 28) = -0.000354193612385874;
    result_local_M(1, 31) = -0.0003700376031547919;
    result_local_M(1, 34) = 0.0002739128564842182;
    result_local_M(1, 37) = 0.0001669058985750177;
    result_local_M(1, 40) = -0.0002662387446394654;
    result_local_M(1, 43) = 0.0003280532723283626;
    result_local_M(1, 46) = -0.0002092686733702037;
    result_local_M(1, 49) = -0.0004852236293687247;
    result_local_M(1, 52) = 0.0005843160727609264;
    result_local_M(1, 55) = 0.0006210828465263479;
    result_local_M(1, 58) = -0.0004587608617178904;
    result_local_M(1, 61) = -0.005394220007674447;
    result_local_M(1, 64) = -0.003004256452075587;
    result_local_M(1, 67) = 0.005660137909039662;
    result_local_M(1, 70) = 0.003812944696864679;
    result_local_M(1, 73) = -0.003327732605285794;
    result_local_M(1, 76) = 0.00898500798502095;
    result_local_M(1, 79) = 0.05735218017320313;
    result_local_M(2, 2) = -0.00001461194246558386;
    result_local_M(2, 5) = 0.00001833669925923309;
    result_local_M(2, 8) = -0.00002316894446404601;
    result_local_M(2, 11) = 0.00001862385852622348;
    result_local_M(2, 14) = 0.00002478183538899803;
    result_local_M(2, 17) = -0.00003092822270269711;
    result_local_M(2, 20) = 0.00003933141761646087;
    result_local_M(2, 23) = -0.00003170470826846835;
    result_local_M(2, 26) = 0.000289611154012035;
    result_local_M(2, 29) = -0.000354193612385874;
    result_local_M(2, 32) = -0.0003700376031547919;
    result_local_M(2, 35) = 0.0002739128564842182;
    result_local_M(2, 38) = 0.0001669058985750177;
    result_local_M(2, 41) = -0.0002662387446394654;
    result_local_M(2, 44) = 0.0003280532723283626;
    result_local_M(2, 47) = -0.0002092686733702037;
    result_local_M(2, 50) = -0.0004852236293687247;
    result_local_M(2, 53) = 0.0005843160727609264;
    result_local_M(2, 56) = 0.0006210828465263479;
    result_local_M(2, 59) = -0.0004587608617178904;
    result_local_M(2, 62) = -0.005394220007674447;
    result_local_M(2, 65) = -0.003004256452075587;
    result_local_M(2, 68) = 0.005660137909039662;
    result_local_M(2, 71) = 0.003812944696864679;
    result_local_M(2, 74) = -0.003327732605285794;
    result_local_M(2, 77) = 0.00898500798502095;
    result_local_M(2, 80) = 0.05735218017320313;
    result_local_M(3, 0) = -0.000005278768205581725;
    result_local_M(3, 3) = 0.00001816306170723496;
    result_local_M(3, 6) = -0.00002104902867833983;
    result_local_M(3, 9) = 0.000006063417748868262;
    result_local_M(3, 12) = 0.000005973260742706684;
    result_local_M(3, 15) = -0.0000193751099866986;
    result_local_M(3, 18) = 0.00002236483680059799;
    result_local_M(3, 21) = -0.000006873773991471258;
    result_local_M(3, 24) = 0.00004294801521162862;
    result_local_M(3, 27) = -0.0004921640815161895;
    result_local_M(3, 30) = -0.00005106333318027401;
    result_local_M(3, 33) = 0.0001441687829666467;
    result_local_M(3, 36) = 0.000349358746012584;
    result_local_M(3, 39) = -0.001206176463539605;
    result_local_M(3, 42) = 0.001429218177634494;
    result_local_M(3, 45) = -0.000413212552443455;
    result_local_M(3, 48) = -0.0000579606691643149;
    result_local_M(3, 51) = 0.0005286204246647887;
    result_local_M(3, 54) = 0.00007004027908102453;
    result_local_M(3, 57) = -0.0001612356962420854;
    result_local_M(3, 60) = -0.001051188977427561;
    result_local_M(3, 63) = -0.002083330982513185;
    result_local_M(3, 66) = 0.03096333812274131;
    result_local_M(3, 69) = 0.002472067176883492;
    result_local_M(3, 72) = -0.00894559901497837;
    result_local_M(3, 75) = 0.001341061778171925;
    result_local_M(3, 78) = 0.05253195269735488;
    result_local_M(4, 1) = -0.000005278768205581725;
    result_local_M(4, 4) = 0.00001816306170723496;
    result_local_M(4, 7) = -0.00002104902867833983;
    result_local_M(4, 10) = 0.000006063417748868262;
    result_local_M(4, 13) = 0.000005973260742706684;
    result_local_M(4, 16) = -0.0000193751099866986;
    result_local_M(4, 19) = 0.00002236483680059799;
    result_local_M(4, 22) = -0.000006873773991471258;
    result_local_M(4, 25) = 0.00004294801521162862;
    result_local_M(4, 28) = -0.0004921640815161895;
    result_local_M(4, 31) = -0.00005106333318027401;
    result_local_M(4, 34) = 0.0001441687829666467;
    result_local_M(4, 37) = 0.000349358746012584;
    result_local_M(4, 40) = -0.001206176463539605;
    result_local_M(4, 43) = 0.001429218177634494;
    result_local_M(4, 46) = -0.000413212552443455;
    result_local_M(4, 49) = -0.0000579606691643149;
    result_local_M(4, 52) = 0.0005286204246647887;
    result_local_M(4, 55) = 0.00007004027908102453;
    result_local_M(4, 58) = -0.0001612356962420854;
    result_local_M(4, 61) = -0.001051188977427561;
    result_local_M(4, 64) = -0.002083330982513185;
    result_local_M(4, 67) = 0.03096333812274131;
    result_local_M(4, 70) = 0.002472067176883492;
    result_local_M(4, 73) = -0.00894559901497837;
    result_local_M(4, 76) = 0.001341061778171925;
    result_local_M(4, 79) = 0.05253195269735488;
    result_local_M(5, 2) = -0.000005278768205581725;
    result_local_M(5, 5) = 0.00001816306170723496;
    result_local_M(5, 8) = -0.00002104902867833983;
    result_local_M(5, 11) = 0.000006063417748868262;
    result_local_M(5, 14) = 0.000005973260742706684;
    result_local_M(5, 17) = -0.0000193751099866986;
    result_local_M(5, 20) = 0.00002236483680059799;
    result_local_M(5, 23) = -0.000006873773991471258;
    result_local_M(5, 26) = 0.00004294801521162862;
    result_local_M(5, 29) = -0.0004921640815161895;
    result_local_M(5, 32) = -0.00005106333318027401;
    result_local_M(5, 35) = 0.0001441687829666467;
    result_local_M(5, 38) = 0.000349358746012584;
    result_local_M(5, 41) = -0.001206176463539605;
    result_local_M(5, 44) = 0.001429218177634494;
    result_local_M(5, 47) = -0.000413212552443455;
    result_local_M(5, 50) = -0.0000579606691643149;
    result_local_M(5, 53) = 0.0005286204246647887;
    result_local_M(5, 56) = 0.00007004027908102453;
    result_local_M(5, 59) = -0.0001612356962420854;
    result_local_M(5, 62) = -0.001051188977427561;
    result_local_M(5, 65) = -0.002083330982513185;
    result_local_M(5, 68) = 0.03096333812274131;
    result_local_M(5, 71) = 0.002472067176883492;
    result_local_M(5, 74) = -0.00894559901497837;
    result_local_M(5, 77) = 0.001341061778171925;
    result_local_M(5, 80) = 0.05253195269735488;
    result_local_M(6, 0) = -0.000107851019907135;
    result_local_M(6, 3) = 0.0001699677848831831;
    result_local_M(6, 6) = -0.0002329932030638625;
    result_local_M(6, 9) = 0.0001471297706749501;
    result_local_M(6, 12) = 0.0001683280798819115;
    result_local_M(6, 15) = -0.0002653702980553399;
    result_local_M(6, 18) = 0.0003628348148939411;
    result_local_M(6, 21) = -0.00022926360852844;
    result_local_M(6, 24) = 0.001234695848630734;
    result_local_M(6, 27) = -0.002486633404463348;
    result_local_M(6, 30) = -0.001674574390442694;
    result_local_M(6, 33) = 0.001612610985124728;
    result_local_M(6, 36) = 0.00109861275377886;
    result_local_M(6, 39) = -0.001509744609144291;
    result_local_M(6, 42) = 0.002151892954597309;
    result_local_M(6, 45) = -0.001524384186540054;
    result_local_M(6, 48) = -0.001947520093440563;
    result_local_M(6, 51) = 0.003929290952131212;
    result_local_M(6, 54) = 0.002637935566005689;
    result_local_M(6, 57) = -0.002534021218699449;
    result_local_M(6, 60) = -0.01887062281036822;
    result_local_M(6, 63) = -0.01304567192310941;
    result_local_M(6, 66) = 0.0178007262761821;
    result_local_M(6, 69) = 0.01785194602479288;
    result_local_M(6, 72) = -0.0151099122043717;
    result_local_M(6, 75) = 0.02990918346672458;
    result_local_M(6, 78) = 0.1914352664790154;
    result_local_M(7, 1) = -0.000107851019907135;
    result_local_M(7, 4) = 0.0001699677848831831;
    result_local_M(7, 7) = -0.0002329932030638625;
    result_local_M(7, 10) = 0.0001471297706749501;
    result_local_M(7, 13) = 0.0001683280798819115;
    result_local_M(7, 16) = -0.0002653702980553399;
    result_local_M(7, 19) = 0.0003628348148939411;
    result_local_M(7, 22) = -0.00022926360852844;
    result_local_M(7, 25) = 0.001234695848630734;
    result_local_M(7, 28) = -0.002486633404463348;
    result_local_M(7, 31) = -0.001674574390442694;
    result_local_M(7, 34) = 0.001612610985124728;
    result_local_M(7, 37) = 0.00109861275377886;
    result_local_M(7, 40) = -0.001509744609144291;
    result_local_M(7, 43) = 0.002151892954597309;
    result_local_M(7, 46) = -0.001524384186540054;
    result_local_M(7, 49) = -0.001947520093440563;
    result_local_M(7, 52) = 0.003929290952131212;
    result_local_M(7, 55) = 0.002637935566005689;
    result_local_M(7, 58) = -0.002534021218699449;
    result_local_M(7, 61) = -0.01887062281036822;
    result_local_M(7, 64) = -0.01304567192310941;
    result_local_M(7, 67) = 0.0178007262761821;
    result_local_M(7, 70) = 0.01785194602479288;
    result_local_M(7, 73) = -0.0151099122043717;
    result_local_M(7, 76) = 0.02990918346672458;
    result_local_M(7, 79) = 0.1914352664790154;
    result_local_M(8, 2) = -0.000107851019907135;
    result_local_M(8, 5) = 0.0001699677848831831;
    result_local_M(8, 8) = -0.0002329932030638625;
    result_local_M(8, 11) = 0.0001471297706749501;
    result_local_M(8, 14) = 0.0001683280798819115;
    result_local_M(8, 17) = -0.0002653702980553399;
    result_local_M(8, 20) = 0.0003628348148939411;
    result_local_M(8, 23) = -0.00022926360852844;
    result_local_M(8, 26) = 0.001234695848630734;
    result_local_M(8, 29) = -0.002486633404463348;
    result_local_M(8, 32) = -0.001674574390442694;
    result_local_M(8, 35) = 0.001612610985124728;
    result_local_M(8, 38) = 0.00109861275377886;
    result_local_M(8, 41) = -0.001509744609144291;
    result_local_M(8, 44) = 0.002151892954597309;
    result_local_M(8, 47) = -0.001524384186540054;
    result_local_M(8, 50) = -0.001947520093440563;
    result_local_M(8, 53) = 0.003929290952131212;
    result_local_M(8, 56) = 0.002637935566005689;
    result_local_M(8, 59) = -0.002534021218699449;
    result_local_M(8, 62) = -0.01887062281036822;
    result_local_M(8, 65) = -0.01304567192310941;
    result_local_M(8, 68) = 0.0178007262761821;
    result_local_M(8, 71) = 0.01785194602479288;
    result_local_M(8, 74) = -0.0151099122043717;
    result_local_M(8, 77) = 0.02990918346672458;
    result_local_M(8, 80) = 0.1914352664790154;
    result_local_M(9, 0) = -0.00006357196624018395;
    result_local_M(9, 3) = 0.0001566318033585906;
    result_local_M(9, 6) = -0.0002048994043713213;
    result_local_M(9, 9) = 0.0000836822556725315;
    result_local_M(9, 12) = 0.00007430864294436283;
    result_local_M(9, 15) = -0.0001851238261199026;
    result_local_M(9, 18) = 0.0002429036139981963;
    result_local_M(9, 21) = -0.0000981342848799864;
    result_local_M(9, 24) = 0.0003775394964271483;
    result_local_M(9, 27) = -0.002728157508355525;
    result_local_M(9, 30) = -0.0005000774905784391;
    result_local_M(9, 33) = 0.00108143726285238;
    result_local_M(9, 36) = 0.001816943397386673;
    result_local_M(9, 39) = -0.004677560172593749;
    result_local_M(9, 42) = 0.006000549721948888;
    result_local_M(9, 45) = -0.002347796408668001;
    result_local_M(9, 48) = -0.0004046595409189861;
    result_local_M(9, 51) = 0.003190423213078577;
    result_local_M(9, 54) = 0.0005389426775581679;
    result_local_M(9, 57) = -0.001250528186681749;
    result_local_M(9, 60) = -0.006296633294499414;
    result_local_M(9, 63) = -0.01163522008948501;
    result_local_M(9, 66) = 0.0877599789662027;
    result_local_M(9, 69) = 0.01510063851862521;
    result_local_M(9, 72) = -0.03314591907165786;
    result_local_M(9, 75) = 0.006637479057005068;
    result_local_M(9, 78) = 0.2090076320453018;
    result_local_M(10, 1) = -0.00006357196624018395;
    result_local_M(10, 4) = 0.0001566318033585906;
    result_local_M(10, 7) = -0.0002048994043713213;
    result_local_M(10, 10) = 0.0000836822556725315;
    result_local_M(10, 13) = 0.00007430864294436283;
    result_local_M(10, 16) = -0.0001851238261199026;
    result_local_M(10, 19) = 0.0002429036139981963;
    result_local_M(10, 22) = -0.0000981342848799864;
    result_local_M(10, 25) = 0.0003775394964271483;
    result_local_M(10, 28) = -0.002728157508355525;
    result_local_M(10, 31) = -0.0005000774905784391;
    result_local_M(10, 34) = 0.00108143726285238;
    result_local_M(10, 37) = 0.001816943397386673;
    result_local_M(10, 40) = -0.004677560172593749;
    result_local_M(10, 43) = 0.006000549721948888;
    result_local_M(10, 46) = -0.002347796408668001;
    result_local_M(10, 49) = -0.0004046595409189861;
    result_local_M(10, 52) = 0.003190423213078577;
    result_local_M(10, 55) = 0.0005389426775581679;
    result_local_M(10, 58) = -0.001250528186681749;
    result_local_M(10, 61) = -0.006296633294499414;
    result_local_M(10, 64) = -0.01163522008948501;
    result_local_M(10, 67) = 0.0877599789662027;
    result_local_M(10, 70) = 0.01510063851862521;
    result_local_M(10, 73) = -0.03314591907165786;
    result_local_M(10, 76) = 0.006637479057005068;
    result_local_M(10, 79) = 0.2090076320453018;
    result_local_M(11, 2) = -0.00006357196624018395;
    result_local_M(11, 5) = 0.0001566318033585906;
    result_local_M(11, 8) = -0.0002048994043713213;
    result_local_M(11, 11) = 0.0000836822556725315;
    result_local_M(11, 14) = 0.00007430864294436283;
    result_local_M(11, 17) = -0.0001851238261199026;
    result_local_M(11, 20) = 0.0002429036139981963;
    result_local_M(11, 23) = -0.0000981342848799864;
    result_local_M(11, 26) = 0.0003775394964271483;
    result_local_M(11, 29) = -0.002728157508355525;
    result_local_M(11, 32) = -0.0005000774905784391;
    result_local_M(11, 35) = 0.00108143726285238;
    result_local_M(11, 38) = 0.001816943397386673;
    result_local_M(11, 41) = -0.004677560172593749;
    result_local_M(11, 44) = 0.006000549721948888;
    result_local_M(11, 47) = -0.002347796408668001;
    result_local_M(11, 50) = -0.0004046595409189861;
    result_local_M(11, 53) = 0.003190423213078577;
    result_local_M(11, 56) = 0.0005389426775581679;
    result_local_M(11, 59) = -0.001250528186681749;
    result_local_M(11, 62) = -0.006296633294499414;
    result_local_M(11, 65) = -0.01163522008948501;
    result_local_M(11, 68) = 0.0877599789662027;
    result_local_M(11, 71) = 0.01510063851862521;
    result_local_M(11, 74) = -0.03314591907165786;
    result_local_M(11, 77) = 0.006637479057005068;
    result_local_M(11, 80) = 0.2090076320453018;

    // Results for Kappa.
    result_local_kappa(0) = 0.06420488066803266;
    result_local_kappa(1) = 0.06420488066803266;
    result_local_kappa(2) = 0.06420488066803266;
    result_local_kappa(3) = 0.07541083032585504;
    result_local_kappa(4) = 0.07541083032585504;
    result_local_kappa(5) = 0.07541083032585504;
    result_local_kappa(6) = 0.210971858787183;
    result_local_kappa(7) = 0.210971858787183;
    result_local_kappa(8) = 0.210971858787183;
    result_local_kappa(9) = 0.2685308094273101;
    result_local_kappa(10) = 0.2685308094273101;
    result_local_kappa(11) = 0.2685308094273101;

    // Perform the unit tests.
    perform_mortar_pair_unit_test(contact_pair, q_beam, q_beam_rot, q_solid, result_local_D,
        result_local_M, result_local_kappa);
  }

  /**
   * \brief Test a non straight beam in a tet4 element, with line2 mortar shape functions.
   */
  TEST_F(BeamToSolidVolumeMeshtyingPairMortarTest, TestBeamToSolidMeshtyingMortarHermite2Tet4Line2)
  {
    // Element types.
    typedef GEOMETRYPAIR::t_hermite beam_type;
    typedef GEOMETRYPAIR::t_tet4 solid_type;
    typedef GEOMETRYPAIR::t_line2 lambda_type;

    // Create the mesh tying mortar pair.
    BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type, lambda_type>
        contact_pair = BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type,
            lambda_type>();

    // Definition of variables for this test case.
    GEOMETRYPAIR::ElementData<beam_type, double> q_beam;
    GEOMETRYPAIR::ElementData<solid_type, double> q_solid;
    Core::LinAlg::Matrix<9, 1> q_beam_rot;
    Core::LinAlg::SerialDenseMatrix local_D;
    Core::LinAlg::SerialDenseMatrix local_M;
    Core::LinAlg::SerialDenseVector local_kappa;

    // Matrices for the results.
    Core::LinAlg::Matrix<lambda_type::n_dof_, beam_type::n_dof_> result_local_D(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, solid_type::n_dof_> result_local_M(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, 1> result_local_kappa(
        Core::LinAlg::Initialization::zero);

    // Define the geometry of the two elements.
    q_beam.shape_function_data_.ref_length_ = 0.6192043571449604711;
    q_beam.element_position_(0) = 0.15;
    q_beam.element_position_(1) = 0.2;
    q_beam.element_position_(2) = 0.3;
    q_beam.element_position_(3) = 0.5773502691896255;
    q_beam.element_position_(4) = 0.5773502691896258;
    q_beam.element_position_(5) = 0.577350269189626;
    q_beam.element_position_(6) = 0.65;
    q_beam.element_position_(7) = 0.1;
    q_beam.element_position_(8) = 0.1;
    q_beam.element_position_(9) = 0.8017837257372733;
    q_beam.element_position_(10) = -0.5345224838248488;
    q_beam.element_position_(11) = 0.2672612419124244;

    q_beam_rot(0) = 1.674352746442651;
    q_beam_rot(1) = 0.1425949677148126;
    q_beam_rot(2) = 1.0831163124736984;
    q_beam_rot(3) = 1.4331999091513161;
    q_beam_rot(4) = -0.6560404572957742;
    q_beam_rot(5) = -0.2491376152457331;
    q_beam_rot(6) = 0.0;
    q_beam_rot(7) = 0.0;
    q_beam_rot(8) = 0.0;

    // Positional DOFs of the solid.
    q_solid.element_position_(0) = 0.04580648387340619;
    q_solid.element_position_(1) = 0.02451732788546579;
    q_solid.element_position_(2) = -0.0031131666281497;
    q_solid.element_position_(3) = 0.923762409575691;
    q_solid.element_position_(4) = -0.0106157274667526;
    q_solid.element_position_(5) = -0.0141609921026476;
    q_solid.element_position_(6) = -0.0944365111054278;
    q_solid.element_position_(7) = 1.069973754602123;
    q_solid.element_position_(8) = 0.06451121136737775;
    q_solid.element_position_(9) = -0.04783899250893692;
    q_solid.element_position_(10) = 0.0888801776910401;
    q_solid.element_position_(11) = 0.963088029848009;

    // Results for D.
    result_local_D(0, 0) = 0.1954215935447702;
    result_local_D(0, 3) = 0.01819251719410916;
    result_local_D(0, 6) = 0.0989414627904878;
    result_local_D(0, 9) = -0.01347747062937554;
    result_local_D(1, 1) = 0.1954215935447702;
    result_local_D(1, 4) = 0.01819251719410916;
    result_local_D(1, 7) = 0.0989414627904878;
    result_local_D(1, 10) = -0.01347747062937554;
    result_local_D(2, 2) = 0.1954215935447702;
    result_local_D(2, 5) = 0.01819251719410916;
    result_local_D(2, 8) = 0.0989414627904878;
    result_local_D(2, 11) = -0.01347747062937554;
    result_local_D(3, 0) = 0.0946778368171451;
    result_local_D(3, 3) = 0.01347747062937554;
    result_local_D(3, 6) = 0.2300774860559777;
    result_local_D(3, 9) = -0.02083257297403942;
    result_local_D(4, 1) = 0.0946778368171451;
    result_local_D(4, 4) = 0.01347747062937554;
    result_local_D(4, 7) = 0.2300774860559777;
    result_local_D(4, 10) = -0.02083257297403942;
    result_local_D(5, 2) = 0.0946778368171451;
    result_local_D(5, 5) = 0.01347747062937554;
    result_local_D(5, 8) = 0.2300774860559777;
    result_local_D(5, 11) = -0.02083257297403942;

    // Results for M.
    result_local_M(0, 0) = 0.05421782284139923;
    result_local_M(0, 3) = 0.1079950516613436;
    result_local_M(0, 6) = 0.05581278922376477;
    result_local_M(0, 9) = 0.07633739260875041;
    result_local_M(1, 1) = 0.05421782284139923;
    result_local_M(1, 4) = 0.1079950516613436;
    result_local_M(1, 7) = 0.05581278922376477;
    result_local_M(1, 10) = 0.07633739260875041;
    result_local_M(2, 2) = 0.05421782284139923;
    result_local_M(2, 5) = 0.1079950516613436;
    result_local_M(2, 8) = 0.05581278922376477;
    result_local_M(2, 11) = 0.07633739260875041;
    result_local_M(3, 0) = 0.0431422441025867;
    result_local_M(3, 3) = 0.1737322126891698;
    result_local_M(3, 6) = 0.05306130615011368;
    result_local_M(3, 9) = 0.05481955993125258;
    result_local_M(4, 1) = 0.0431422441025867;
    result_local_M(4, 4) = 0.1737322126891698;
    result_local_M(4, 7) = 0.05306130615011368;
    result_local_M(4, 10) = 0.05481955993125258;
    result_local_M(5, 2) = 0.0431422441025867;
    result_local_M(5, 5) = 0.1737322126891698;
    result_local_M(5, 8) = 0.05306130615011368;
    result_local_M(5, 11) = 0.05481955993125258;

    // Results for Kappa.
    result_local_kappa(0) = 0.294363056335258;
    result_local_kappa(1) = 0.294363056335258;
    result_local_kappa(2) = 0.294363056335258;
    result_local_kappa(3) = 0.3247553228731228;
    result_local_kappa(4) = 0.3247553228731228;
    result_local_kappa(5) = 0.3247553228731228;

    // Perform the unit tests.
    perform_mortar_pair_unit_test(contact_pair, q_beam, q_beam_rot, q_solid, result_local_D,
        result_local_M, result_local_kappa);
  }

  /**
   * \brief Test a non straight beam in a tet4 element, with line3 mortar shape functions.
   */
  TEST_F(BeamToSolidVolumeMeshtyingPairMortarTest, TestBeamToSolidMeshtyingMortarHermite2Tet4Line3)
  {
    // Element types.
    typedef GEOMETRYPAIR::t_hermite beam_type;
    typedef GEOMETRYPAIR::t_tet4 solid_type;
    typedef GEOMETRYPAIR::t_line3 lambda_type;

    // Create the mesh tying mortar pair.
    BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type, lambda_type>
        contact_pair = BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type,
            lambda_type>();

    // Definition of variables for this test case.
    GEOMETRYPAIR::ElementData<beam_type, double> q_beam;
    GEOMETRYPAIR::ElementData<solid_type, double> q_solid;
    Core::LinAlg::Matrix<9, 1> q_beam_rot;
    Core::LinAlg::SerialDenseMatrix local_D;
    Core::LinAlg::SerialDenseMatrix local_M;
    Core::LinAlg::SerialDenseVector local_kappa;

    // Matrices for the results.
    Core::LinAlg::Matrix<lambda_type::n_dof_, beam_type::n_dof_> result_local_D(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, solid_type::n_dof_> result_local_M(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, 1> result_local_kappa(
        Core::LinAlg::Initialization::zero);

    // Define the geometry of the two elements.
    q_beam.shape_function_data_.ref_length_ = 0.6192043571449604711;
    q_beam.element_position_(0) = 0.15;
    q_beam.element_position_(1) = 0.2;
    q_beam.element_position_(2) = 0.3;
    q_beam.element_position_(3) = 0.5773502691896255;
    q_beam.element_position_(4) = 0.5773502691896258;
    q_beam.element_position_(5) = 0.577350269189626;
    q_beam.element_position_(6) = 0.65;
    q_beam.element_position_(7) = 0.1;
    q_beam.element_position_(8) = 0.1;
    q_beam.element_position_(9) = 0.8017837257372733;
    q_beam.element_position_(10) = -0.5345224838248488;
    q_beam.element_position_(11) = 0.2672612419124244;

    q_beam_rot(0) = 1.674352746442651;
    q_beam_rot(1) = 0.1425949677148126;
    q_beam_rot(2) = 1.0831163124736984;
    q_beam_rot(3) = 1.4331999091513161;
    q_beam_rot(4) = -0.6560404572957742;
    q_beam_rot(5) = -0.2491376152457331;
    q_beam_rot(6) = 0.0;
    q_beam_rot(7) = 0.0;
    q_beam_rot(8) = 0.0;

    // Positional DOFs of the solid.
    q_solid.element_position_(0) = 0.04580648387340619;
    q_solid.element_position_(1) = 0.02451732788546579;
    q_solid.element_position_(2) = -0.0031131666281497;
    q_solid.element_position_(3) = 0.923762409575691;
    q_solid.element_position_(4) = -0.0106157274667526;
    q_solid.element_position_(5) = -0.0141609921026476;
    q_solid.element_position_(6) = -0.0944365111054278;
    q_solid.element_position_(7) = 1.069973754602123;
    q_solid.element_position_(8) = 0.06451121136737775;
    q_solid.element_position_(9) = -0.04783899250893692;
    q_solid.element_position_(10) = 0.0888801776910401;
    q_solid.element_position_(11) = 0.963088029848009;

    // Results for D.
    result_local_D(0, 0) = 0.0945173766014223;
    result_local_D(0, 3) = 0.005144868995407634;
    result_local_D(0, 6) = -0.0132666172503834;
    result_local_D(0, 9) = 0.0004298224306740067;
    result_local_D(1, 1) = 0.0945173766014223;
    result_local_D(1, 4) = 0.005144868995407634;
    result_local_D(1, 7) = -0.0132666172503834;
    result_local_D(1, 10) = 0.0004298224306740067;
    result_local_D(2, 2) = 0.0945173766014223;
    result_local_D(2, 5) = 0.005144868995407634;
    result_local_D(2, 8) = -0.0132666172503834;
    result_local_D(2, 11) = 0.0004298224306740067;
    result_local_D(3, 0) = -0.006226380126202878;
    result_local_D(3, 3) = 0.0004298224306740081;
    result_local_D(3, 6) = 0.1178694060151065;
    result_local_D(3, 9) = -0.006925279913989872;
    result_local_D(4, 1) = -0.006226380126202878;
    result_local_D(4, 4) = 0.0004298224306740081;
    result_local_D(4, 7) = 0.1178694060151065;
    result_local_D(4, 10) = -0.006925279913989872;
    result_local_D(5, 2) = -0.006226380126202878;
    result_local_D(5, 5) = 0.0004298224306740081;
    result_local_D(5, 8) = 0.1178694060151065;
    result_local_D(5, 11) = -0.006925279913989872;
    result_local_D(6, 0) = 0.2018084338866959;
    result_local_D(6, 3) = 0.02609529639740305;
    result_local_D(6, 6) = 0.2244161600817424;
    result_local_D(6, 9) = -0.02781458612009909;
    result_local_D(7, 1) = 0.2018084338866959;
    result_local_D(7, 4) = 0.02609529639740305;
    result_local_D(7, 7) = 0.2244161600817424;
    result_local_D(7, 10) = -0.02781458612009909;
    result_local_D(8, 2) = 0.2018084338866959;
    result_local_D(8, 5) = 0.02609529639740305;
    result_local_D(8, 8) = 0.2244161600817424;
    result_local_D(8, 11) = -0.02781458612009909;

    // Results for M.
    result_local_M(0, 0) = 0.02309134137849664;
    result_local_M(0, 3) = 0.01167312298826655;
    result_local_M(0, 6) = 0.01604691566430098;
    result_local_M(0, 9) = 0.0304393793199747;
    result_local_M(1, 1) = 0.02309134137849664;
    result_local_M(1, 4) = 0.01167312298826655;
    result_local_M(1, 7) = 0.01604691566430098;
    result_local_M(1, 10) = 0.0304393793199747;
    result_local_M(2, 2) = 0.02309134137849664;
    result_local_M(2, 5) = 0.01167312298826655;
    result_local_M(2, 8) = 0.01604691566430098;
    result_local_M(2, 11) = 0.0304393793199747;
    result_local_M(3, 0) = 0.01201576263968411;
    result_local_M(3, 3) = 0.07741028401609277;
    result_local_M(3, 6) = 0.01329543259064989;
    result_local_M(3, 9) = 0.00892154664247686;
    result_local_M(4, 1) = 0.01201576263968411;
    result_local_M(4, 4) = 0.07741028401609277;
    result_local_M(4, 7) = 0.01329543259064989;
    result_local_M(4, 10) = 0.00892154664247686;
    result_local_M(5, 2) = 0.01201576263968411;
    result_local_M(5, 5) = 0.07741028401609277;
    result_local_M(5, 8) = 0.01329543259064989;
    result_local_M(5, 11) = 0.00892154664247686;
    result_local_M(6, 0) = 0.06225296292580519;
    result_local_M(6, 3) = 0.1926438573461542;
    result_local_M(6, 6) = 0.07953174711892758;
    result_local_M(6, 9) = 0.0917960265775514;
    result_local_M(7, 1) = 0.06225296292580519;
    result_local_M(7, 4) = 0.1926438573461542;
    result_local_M(7, 7) = 0.07953174711892758;
    result_local_M(7, 10) = 0.0917960265775514;
    result_local_M(8, 2) = 0.06225296292580519;
    result_local_M(8, 5) = 0.1926438573461542;
    result_local_M(8, 8) = 0.07953174711892758;
    result_local_M(8, 11) = 0.0917960265775514;

    // Results for Kappa.
    result_local_kappa(0) = 0.0812507593510389;
    result_local_kappa(1) = 0.0812507593510389;
    result_local_kappa(2) = 0.0812507593510389;
    result_local_kappa(3) = 0.1116430258889036;
    result_local_kappa(4) = 0.1116430258889036;
    result_local_kappa(5) = 0.1116430258889036;
    result_local_kappa(6) = 0.4262245939684383;
    result_local_kappa(7) = 0.4262245939684383;
    result_local_kappa(8) = 0.4262245939684383;

    // Perform the unit tests.
    perform_mortar_pair_unit_test(contact_pair, q_beam, q_beam_rot, q_solid, result_local_D,
        result_local_M, result_local_kappa);
  }

  /**
   * \brief Test a non straight beam in a tet4 element, with line4 mortar shape functions.
   */
  TEST_F(BeamToSolidVolumeMeshtyingPairMortarTest, TestBeamToSolidMeshtyingMortarHermite2Tet4Line4)
  {
    // Element types.
    typedef GEOMETRYPAIR::t_hermite beam_type;
    typedef GEOMETRYPAIR::t_tet4 solid_type;
    typedef GEOMETRYPAIR::t_line4 lambda_type;

    // Create the mesh tying mortar pair.
    BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type, lambda_type>
        contact_pair = BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type,
            lambda_type>();

    // Definition of variables for this test case.
    GEOMETRYPAIR::ElementData<beam_type, double> q_beam;
    GEOMETRYPAIR::ElementData<solid_type, double> q_solid;
    Core::LinAlg::Matrix<9, 1> q_beam_rot;
    Core::LinAlg::SerialDenseMatrix local_D;
    Core::LinAlg::SerialDenseMatrix local_M;
    Core::LinAlg::SerialDenseVector local_kappa;

    // Matrices for the results.
    Core::LinAlg::Matrix<lambda_type::n_dof_, beam_type::n_dof_> result_local_D(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, solid_type::n_dof_> result_local_M(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, 1> result_local_kappa(
        Core::LinAlg::Initialization::zero);

    // Define the geometry of the two elements.
    q_beam.shape_function_data_.ref_length_ = 0.6192043571449604711;
    q_beam.element_position_(0) = 0.15;
    q_beam.element_position_(1) = 0.2;
    q_beam.element_position_(2) = 0.3;
    q_beam.element_position_(3) = 0.5773502691896255;
    q_beam.element_position_(4) = 0.5773502691896258;
    q_beam.element_position_(5) = 0.577350269189626;
    q_beam.element_position_(6) = 0.65;
    q_beam.element_position_(7) = 0.1;
    q_beam.element_position_(8) = 0.1;
    q_beam.element_position_(9) = 0.8017837257372733;
    q_beam.element_position_(10) = -0.5345224838248488;
    q_beam.element_position_(11) = 0.2672612419124244;

    q_beam_rot(0) = 1.674352746442651;
    q_beam_rot(1) = 0.1425949677148126;
    q_beam_rot(2) = 1.0831163124736984;
    q_beam_rot(3) = 1.4331999091513161;
    q_beam_rot(4) = -0.6560404572957742;
    q_beam_rot(5) = -0.2491376152457331;
    q_beam_rot(6) = 0.0;
    q_beam_rot(7) = 0.0;
    q_beam_rot(8) = 0.0;

    // Positional DOFs of the solid.
    q_solid.element_position_(0) = 0.04580648387340619;
    q_solid.element_position_(1) = 0.02451732788546579;
    q_solid.element_position_(2) = -0.0031131666281497;
    q_solid.element_position_(3) = 0.923762409575691;
    q_solid.element_position_(4) = -0.0106157274667526;
    q_solid.element_position_(5) = -0.0141609921026476;
    q_solid.element_position_(6) = -0.0944365111054278;
    q_solid.element_position_(7) = 1.069973754602123;
    q_solid.element_position_(8) = 0.06451121136737775;
    q_solid.element_position_(9) = -0.04783899250893692;
    q_solid.element_position_(10) = 0.0888801776910401;
    q_solid.element_position_(11) = 0.963088029848009;

    // Results for D.
    result_local_D(0, 0) = 0.05837517950584759;
    result_local_D(0, 3) = 0.002017208305318981;
    result_local_D(0, 6) = 0.005829701162185071;
    result_local_D(0, 9) = -0.0002955710710872768;
    result_local_D(1, 1) = 0.05837517950584759;
    result_local_D(1, 4) = 0.002017208305318981;
    result_local_D(1, 7) = 0.005829701162185071;
    result_local_D(1, 10) = -0.0002955710710872768;
    result_local_D(2, 2) = 0.05837517950584759;
    result_local_D(2, 5) = 0.002017208305318981;
    result_local_D(2, 8) = 0.005829701162185071;
    result_local_D(2, 11) = -0.0002955710710872768;
    result_local_D(3, 0) = 0.004689762733534804;
    result_local_D(3, 3) = 0.0002955710710872767;
    result_local_D(3, 6) = 0.07072106759232025;
    result_local_D(3, 9) = -0.002723063147216203;
    result_local_D(4, 1) = 0.004689762733534804;
    result_local_D(4, 4) = 0.0002955710710872767;
    result_local_D(4, 7) = 0.07072106759232025;
    result_local_D(4, 10) = -0.002723063147216203;
    result_local_D(5, 2) = 0.004689762733534804;
    result_local_D(5, 5) = 0.0002955710710872767;
    result_local_D(5, 8) = 0.07072106759232025;
    result_local_D(5, 11) = -0.002723063147216203;
    result_local_D(6, 0) = 0.184104753994235;
    result_local_D(6, 3) = 0.0191687182192921;
    result_local_D(6, 6) = 0.02686710479294802;
    result_local_D(6, 9) = -0.00825428928975331;
    result_local_D(7, 1) = 0.184104753994235;
    result_local_D(7, 4) = 0.0191687182192921;
    result_local_D(7, 7) = 0.02686710479294802;
    result_local_D(7, 10) = -0.00825428928975331;
    result_local_D(8, 2) = 0.184104753994235;
    result_local_D(8, 5) = 0.0191687182192921;
    result_local_D(8, 8) = 0.02686710479294802;
    result_local_D(8, 11) = -0.00825428928975331;
    result_local_D(9, 0) = 0.04292973412829792;
    result_local_D(9, 3) = 0.01018849022778634;
    result_local_D(9, 6) = 0.2256010752990122;
    result_local_D(9, 9) = -0.02303712009535817;
    result_local_D(10, 1) = 0.04292973412829792;
    result_local_D(10, 4) = 0.01018849022778634;
    result_local_D(10, 7) = 0.2256010752990122;
    result_local_D(10, 10) = -0.02303712009535817;
    result_local_D(11, 2) = 0.04292973412829792;
    result_local_D(11, 5) = 0.01018849022778634;
    result_local_D(11, 8) = 0.2256010752990122;
    result_local_D(11, 11) = -0.02303712009535817;

    // Results for M.
    result_local_M(0, 0) = 0.01810400494829576;
    result_local_M(0, 3) = 0.01580213277555223;
    result_local_M(0, 6) = 0.01081605573735045;
    result_local_M(0, 9) = 0.01948268720683424;
    result_local_M(1, 1) = 0.01810400494829576;
    result_local_M(1, 4) = 0.01580213277555223;
    result_local_M(1, 7) = 0.01081605573735045;
    result_local_M(1, 10) = 0.01948268720683424;
    result_local_M(2, 2) = 0.01810400494829576;
    result_local_M(2, 5) = 0.01580213277555223;
    result_local_M(2, 8) = 0.01081605573735045;
    result_local_M(2, 11) = 0.01948268720683424;
    result_local_M(3, 0) = 0.00922147870415935;
    result_local_M(3, 3) = 0.04920079206053781;
    result_local_M(3, 6) = 0.00858482412773449;
    result_local_M(3, 9) = 0.00840373543342339;
    result_local_M(4, 1) = 0.00922147870415935;
    result_local_M(4, 4) = 0.04920079206053781;
    result_local_M(4, 7) = 0.00858482412773449;
    result_local_M(4, 10) = 0.00840373543342339;
    result_local_M(5, 2) = 0.00922147870415935;
    result_local_M(5, 5) = 0.04920079206053781;
    result_local_M(5, 8) = 0.00858482412773449;
    result_local_M(5, 11) = 0.00840373543342339;
    result_local_M(6, 0) = 0.03830687038777961;
    result_local_M(6, 3) = 0.05985441714295076;
    result_local_M(6, 6) = 0.04551698495044945;
    result_local_M(6, 9) = 0.06729358630600315;
    result_local_M(7, 1) = 0.03830687038777961;
    result_local_M(7, 4) = 0.05985441714295076;
    result_local_M(7, 7) = 0.04551698495044945;
    result_local_M(7, 10) = 0.06729358630600315;
    result_local_M(8, 2) = 0.03830687038777961;
    result_local_M(8, 5) = 0.05985441714295076;
    result_local_M(8, 8) = 0.04551698495044945;
    result_local_M(8, 11) = 0.06729358630600315;
    result_local_M(9, 0) = 0.03172771290375121;
    result_local_M(9, 3) = 0.1568699223714726;
    result_local_M(9, 6) = 0.04395623055834406;
    result_local_M(9, 9) = 0.03597694359374219;
    result_local_M(10, 1) = 0.03172771290375121;
    result_local_M(10, 4) = 0.1568699223714726;
    result_local_M(10, 7) = 0.04395623055834406;
    result_local_M(10, 10) = 0.03597694359374219;
    result_local_M(11, 2) = 0.03172771290375121;
    result_local_M(11, 5) = 0.1568699223714726;
    result_local_M(11, 8) = 0.04395623055834406;
    result_local_M(11, 11) = 0.03597694359374219;

    // Results for Kappa.
    result_local_kappa(0) = 0.06420488066803266;
    result_local_kappa(1) = 0.06420488066803266;
    result_local_kappa(2) = 0.06420488066803266;
    result_local_kappa(3) = 0.07541083032585504;
    result_local_kappa(4) = 0.07541083032585504;
    result_local_kappa(5) = 0.07541083032585504;
    result_local_kappa(6) = 0.210971858787183;
    result_local_kappa(7) = 0.210971858787183;
    result_local_kappa(8) = 0.210971858787183;
    result_local_kappa(9) = 0.2685308094273101;
    result_local_kappa(10) = 0.2685308094273101;
    result_local_kappa(11) = 0.2685308094273101;

    // Perform the unit tests.
    perform_mortar_pair_unit_test(contact_pair, q_beam, q_beam_rot, q_solid, result_local_D,
        result_local_M, result_local_kappa);
  }

  /**
   * \brief Test a non straight beam in a tet10 element, with line2 mortar shape functions.
   */
  TEST_F(BeamToSolidVolumeMeshtyingPairMortarTest, TestBeamToSolidMeshtyingMortarHermite2Tet10Line2)
  {
    // Element types.
    typedef GEOMETRYPAIR::t_hermite beam_type;
    typedef GEOMETRYPAIR::t_tet10 solid_type;
    typedef GEOMETRYPAIR::t_line2 lambda_type;

    // Create the mesh tying mortar pair.
    BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type, lambda_type>
        contact_pair = BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type,
            lambda_type>();

    // Definition of variables for this test case.
    GEOMETRYPAIR::ElementData<beam_type, double> q_beam;
    GEOMETRYPAIR::ElementData<solid_type, double> q_solid;
    Core::LinAlg::Matrix<9, 1> q_beam_rot;
    Core::LinAlg::SerialDenseMatrix local_D;
    Core::LinAlg::SerialDenseMatrix local_M;
    Core::LinAlg::SerialDenseVector local_kappa;

    // Matrices for the results.
    Core::LinAlg::Matrix<lambda_type::n_dof_, beam_type::n_dof_> result_local_D(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, solid_type::n_dof_> result_local_M(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, 1> result_local_kappa(
        Core::LinAlg::Initialization::zero);

    // Define the geometry of the two elements.
    q_beam.shape_function_data_.ref_length_ = 0.6192043571449604711;
    q_beam.element_position_(0) = 0.15;
    q_beam.element_position_(1) = 0.2;
    q_beam.element_position_(2) = 0.3;
    q_beam.element_position_(3) = 0.5773502691896255;
    q_beam.element_position_(4) = 0.5773502691896258;
    q_beam.element_position_(5) = 0.577350269189626;
    q_beam.element_position_(6) = 0.65;
    q_beam.element_position_(7) = 0.1;
    q_beam.element_position_(8) = 0.1;
    q_beam.element_position_(9) = 0.8017837257372733;
    q_beam.element_position_(10) = -0.5345224838248488;
    q_beam.element_position_(11) = 0.2672612419124244;

    q_beam_rot(0) = 1.674352746442651;
    q_beam_rot(1) = 0.1425949677148126;
    q_beam_rot(2) = 1.0831163124736984;
    q_beam_rot(3) = 1.4331999091513161;
    q_beam_rot(4) = -0.6560404572957742;
    q_beam_rot(5) = -0.2491376152457331;
    q_beam_rot(6) = 0.0;
    q_beam_rot(7) = 0.0;
    q_beam_rot(8) = 0.0;

    // Positional DOFs of the solid.
    q_solid.element_position_(0) = 0.04580648387340619;
    q_solid.element_position_(1) = 0.02451732788546579;
    q_solid.element_position_(2) = -0.0031131666281497;
    q_solid.element_position_(3) = 0.923762409575691;
    q_solid.element_position_(4) = -0.0106157274667526;
    q_solid.element_position_(5) = -0.0141609921026476;
    q_solid.element_position_(6) = -0.0944365111054278;
    q_solid.element_position_(7) = 1.069973754602123;
    q_solid.element_position_(8) = 0.06451121136737775;
    q_solid.element_position_(9) = -0.04783899250893692;
    q_solid.element_position_(10) = 0.0888801776910401;
    q_solid.element_position_(11) = 0.963088029848009;
    q_solid.element_position_(12) = 0.4108326210187769;
    q_solid.element_position_(13) = -0.06382048551478623;
    q_solid.element_position_(14) = 0.0810006716524518;
    q_solid.element_position_(15) = 0.4728874772489418;
    q_solid.element_position_(16) = 0.4910049605448334;
    q_solid.element_position_(17) = -0.07654076843449248;
    q_solid.element_position_(18) = 0.0891921101646426;
    q_solid.element_position_(19) = 0.5267989315386162;
    q_solid.element_position_(20) = -0.03523821168532098;
    q_solid.element_position_(21) = 0.05851054670001138;
    q_solid.element_position_(22) = -0.04567202467351366;
    q_solid.element_position_(23) = 0.4631680679379652;
    q_solid.element_position_(24) = 0.5925155012591516;
    q_solid.element_position_(25) = 0.02661668312122867;
    q_solid.element_position_(26) = 0.4341991229313329;
    q_solid.element_position_(27) = -0.006248620475302225;
    q_solid.element_position_(28) = 0.4109215888511089;
    q_solid.element_position_(29) = 0.4194107884465731;

    // Results for D.
    result_local_D(0, 0) = 0.1954215935447702;
    result_local_D(0, 3) = 0.01819251719410916;
    result_local_D(0, 6) = 0.0989414627904878;
    result_local_D(0, 9) = -0.01347747062937554;
    result_local_D(1, 1) = 0.1954215935447702;
    result_local_D(1, 4) = 0.01819251719410916;
    result_local_D(1, 7) = 0.0989414627904878;
    result_local_D(1, 10) = -0.01347747062937554;
    result_local_D(2, 2) = 0.1954215935447702;
    result_local_D(2, 5) = 0.01819251719410916;
    result_local_D(2, 8) = 0.0989414627904878;
    result_local_D(2, 11) = -0.01347747062937554;
    result_local_D(3, 0) = 0.0946778368171451;
    result_local_D(3, 3) = 0.01347747062937554;
    result_local_D(3, 6) = 0.2300774860559777;
    result_local_D(3, 9) = -0.02083257297403942;
    result_local_D(4, 1) = 0.0946778368171451;
    result_local_D(4, 4) = 0.01347747062937554;
    result_local_D(4, 7) = 0.2300774860559777;
    result_local_D(4, 10) = -0.02083257297403942;
    result_local_D(5, 2) = 0.0946778368171451;
    result_local_D(5, 5) = 0.01347747062937554;
    result_local_D(5, 8) = 0.2300774860559777;
    result_local_D(5, 11) = -0.02083257297403942;

    // Results for M.
    result_local_M(0, 0) = -0.02397686687227069;
    result_local_M(0, 3) = -0.02687214629590493;
    result_local_M(0, 6) = -0.03550651396519254;
    result_local_M(0, 9) = -0.02921012580297817;
    result_local_M(0, 12) = 0.03049976077490606;
    result_local_M(0, 15) = 0.0855301478932878;
    result_local_M(0, 18) = 0.03790662409203156;
    result_local_M(0, 21) = 0.0464904897110114;
    result_local_M(0, 24) = 0.0947375007994866;
    result_local_M(0, 27) = 0.1147641860008809;
    result_local_M(1, 1) = -0.02397686687227069;
    result_local_M(1, 4) = -0.02687214629590493;
    result_local_M(1, 7) = -0.03550651396519254;
    result_local_M(1, 10) = -0.02921012580297817;
    result_local_M(1, 13) = 0.03049976077490606;
    result_local_M(1, 16) = 0.0855301478932878;
    result_local_M(1, 19) = 0.03790662409203156;
    result_local_M(1, 22) = 0.0464904897110114;
    result_local_M(1, 25) = 0.0947375007994866;
    result_local_M(1, 28) = 0.1147641860008809;
    result_local_M(2, 2) = -0.02397686687227069;
    result_local_M(2, 5) = -0.02687214629590493;
    result_local_M(2, 8) = -0.03550651396519254;
    result_local_M(2, 11) = -0.02921012580297817;
    result_local_M(2, 14) = 0.03049976077490606;
    result_local_M(2, 17) = 0.0855301478932878;
    result_local_M(2, 20) = 0.03790662409203156;
    result_local_M(2, 23) = 0.0464904897110114;
    result_local_M(2, 26) = 0.0947375007994866;
    result_local_M(2, 29) = 0.1147641860008809;
    result_local_M(3, 0) = -0.02381982755340511;
    result_local_M(3, 3) = -0.005026396151256262;
    result_local_M(3, 6) = -0.038481801381297;
    result_local_M(3, 9) = -0.03584624386261232;
    result_local_M(3, 12) = 0.05217634405892973;
    result_local_M(3, 15) = 0.124988716336588;
    result_local_M(3, 18) = 0.02706335987386299;
    result_local_M(3, 21) = 0.02697061629963811;
    result_local_M(3, 24) = 0.1200998076511035;
    result_local_M(3, 27) = 0.07663074760157119;
    result_local_M(4, 1) = -0.02381982755340511;
    result_local_M(4, 4) = -0.005026396151256262;
    result_local_M(4, 7) = -0.038481801381297;
    result_local_M(4, 10) = -0.03584624386261232;
    result_local_M(4, 13) = 0.05217634405892973;
    result_local_M(4, 16) = 0.124988716336588;
    result_local_M(4, 19) = 0.02706335987386299;
    result_local_M(4, 22) = 0.02697061629963811;
    result_local_M(4, 25) = 0.1200998076511035;
    result_local_M(4, 28) = 0.07663074760157119;
    result_local_M(5, 2) = -0.02381982755340511;
    result_local_M(5, 5) = -0.005026396151256262;
    result_local_M(5, 8) = -0.038481801381297;
    result_local_M(5, 11) = -0.03584624386261232;
    result_local_M(5, 14) = 0.05217634405892973;
    result_local_M(5, 17) = 0.124988716336588;
    result_local_M(5, 20) = 0.02706335987386299;
    result_local_M(5, 23) = 0.02697061629963811;
    result_local_M(5, 26) = 0.1200998076511035;
    result_local_M(5, 29) = 0.07663074760157119;

    // Results for Kappa.
    result_local_kappa(0) = 0.294363056335258;
    result_local_kappa(1) = 0.294363056335258;
    result_local_kappa(2) = 0.294363056335258;
    result_local_kappa(3) = 0.3247553228731228;
    result_local_kappa(4) = 0.3247553228731228;
    result_local_kappa(5) = 0.3247553228731228;

    // Perform the unit tests.
    perform_mortar_pair_unit_test(contact_pair, q_beam, q_beam_rot, q_solid, result_local_D,
        result_local_M, result_local_kappa);
  }

  /**
   * \brief Test a non straight beam in a tet10 element, with line3 mortar shape functions.
   */
  TEST_F(BeamToSolidVolumeMeshtyingPairMortarTest, TestBeamToSolidMeshtyingMortarHermite2Tet10Line3)
  {
    // Element types.
    typedef GEOMETRYPAIR::t_hermite beam_type;
    typedef GEOMETRYPAIR::t_tet10 solid_type;
    typedef GEOMETRYPAIR::t_line3 lambda_type;

    // Create the mesh tying mortar pair.
    BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type, lambda_type>
        contact_pair = BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type,
            lambda_type>();

    // Definition of variables for this test case.
    GEOMETRYPAIR::ElementData<beam_type, double> q_beam;
    GEOMETRYPAIR::ElementData<solid_type, double> q_solid;
    Core::LinAlg::Matrix<9, 1> q_beam_rot;
    Core::LinAlg::SerialDenseMatrix local_D;
    Core::LinAlg::SerialDenseMatrix local_M;
    Core::LinAlg::SerialDenseVector local_kappa;

    // Matrices for the results.
    Core::LinAlg::Matrix<lambda_type::n_dof_, beam_type::n_dof_> result_local_D(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, solid_type::n_dof_> result_local_M(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, 1> result_local_kappa(
        Core::LinAlg::Initialization::zero);

    // Define the geometry of the two elements.
    q_beam.shape_function_data_.ref_length_ = 0.6192043571449604711;
    q_beam.element_position_(0) = 0.15;
    q_beam.element_position_(1) = 0.2;
    q_beam.element_position_(2) = 0.3;
    q_beam.element_position_(3) = 0.5773502691896255;
    q_beam.element_position_(4) = 0.5773502691896258;
    q_beam.element_position_(5) = 0.577350269189626;
    q_beam.element_position_(6) = 0.65;
    q_beam.element_position_(7) = 0.1;
    q_beam.element_position_(8) = 0.1;
    q_beam.element_position_(9) = 0.8017837257372733;
    q_beam.element_position_(10) = -0.5345224838248488;
    q_beam.element_position_(11) = 0.2672612419124244;

    q_beam_rot(0) = 1.674352746442651;
    q_beam_rot(1) = 0.1425949677148126;
    q_beam_rot(2) = 1.0831163124736984;
    q_beam_rot(3) = 1.4331999091513161;
    q_beam_rot(4) = -0.6560404572957742;
    q_beam_rot(5) = -0.2491376152457331;
    q_beam_rot(6) = 0.0;
    q_beam_rot(7) = 0.0;
    q_beam_rot(8) = 0.0;

    // Positional DOFs of the solid.
    q_solid.element_position_(0) = 0.04580648387340619;
    q_solid.element_position_(1) = 0.02451732788546579;
    q_solid.element_position_(2) = -0.0031131666281497;
    q_solid.element_position_(3) = 0.923762409575691;
    q_solid.element_position_(4) = -0.0106157274667526;
    q_solid.element_position_(5) = -0.0141609921026476;
    q_solid.element_position_(6) = -0.0944365111054278;
    q_solid.element_position_(7) = 1.069973754602123;
    q_solid.element_position_(8) = 0.06451121136737775;
    q_solid.element_position_(9) = -0.04783899250893692;
    q_solid.element_position_(10) = 0.0888801776910401;
    q_solid.element_position_(11) = 0.963088029848009;
    q_solid.element_position_(12) = 0.4108326210187769;
    q_solid.element_position_(13) = -0.06382048551478623;
    q_solid.element_position_(14) = 0.0810006716524518;
    q_solid.element_position_(15) = 0.4728874772489418;
    q_solid.element_position_(16) = 0.4910049605448334;
    q_solid.element_position_(17) = -0.07654076843449248;
    q_solid.element_position_(18) = 0.0891921101646426;
    q_solid.element_position_(19) = 0.5267989315386162;
    q_solid.element_position_(20) = -0.03523821168532098;
    q_solid.element_position_(21) = 0.05851054670001138;
    q_solid.element_position_(22) = -0.04567202467351366;
    q_solid.element_position_(23) = 0.4631680679379652;
    q_solid.element_position_(24) = 0.5925155012591516;
    q_solid.element_position_(25) = 0.02661668312122867;
    q_solid.element_position_(26) = 0.4341991229313329;
    q_solid.element_position_(27) = -0.006248620475302225;
    q_solid.element_position_(28) = 0.4109215888511089;
    q_solid.element_position_(29) = 0.4194107884465731;

    // Results for D.
    result_local_D(0, 0) = 0.0945173766014223;
    result_local_D(0, 3) = 0.005144868995407634;
    result_local_D(0, 6) = -0.0132666172503834;
    result_local_D(0, 9) = 0.0004298224306740067;
    result_local_D(1, 1) = 0.0945173766014223;
    result_local_D(1, 4) = 0.005144868995407634;
    result_local_D(1, 7) = -0.0132666172503834;
    result_local_D(1, 10) = 0.0004298224306740067;
    result_local_D(2, 2) = 0.0945173766014223;
    result_local_D(2, 5) = 0.005144868995407634;
    result_local_D(2, 8) = -0.0132666172503834;
    result_local_D(2, 11) = 0.0004298224306740067;
    result_local_D(3, 0) = -0.006226380126202878;
    result_local_D(3, 3) = 0.0004298224306740081;
    result_local_D(3, 6) = 0.1178694060151065;
    result_local_D(3, 9) = -0.006925279913989872;
    result_local_D(4, 1) = -0.006226380126202878;
    result_local_D(4, 4) = 0.0004298224306740081;
    result_local_D(4, 7) = 0.1178694060151065;
    result_local_D(4, 10) = -0.006925279913989872;
    result_local_D(5, 2) = -0.006226380126202878;
    result_local_D(5, 5) = 0.0004298224306740081;
    result_local_D(5, 8) = 0.1178694060151065;
    result_local_D(5, 11) = -0.006925279913989872;
    result_local_D(6, 0) = 0.2018084338866959;
    result_local_D(6, 3) = 0.02609529639740305;
    result_local_D(6, 6) = 0.2244161600817424;
    result_local_D(6, 9) = -0.02781458612009909;
    result_local_D(7, 1) = 0.2018084338866959;
    result_local_D(7, 4) = 0.02609529639740305;
    result_local_D(7, 7) = 0.2244161600817424;
    result_local_D(7, 10) = -0.02781458612009909;
    result_local_D(8, 2) = 0.2018084338866959;
    result_local_D(8, 5) = 0.02609529639740305;
    result_local_D(8, 8) = 0.2244161600817424;
    result_local_D(8, 11) = -0.02781458612009909;

    // Results for M.
    result_local_M(0, 0) = -0.00863147348892433;
    result_local_M(0, 3) = -0.01096586628742749;
    result_local_M(0, 6) = -0.00974646045882031;
    result_local_M(0, 9) = -0.006134875867851393;
    result_local_M(0, 12) = 0.004047099946688711;
    result_local_M(0, 15) = 0.00889150049994481;
    result_local_M(0, 18) = 0.01764375010921981;
    result_local_M(0, 21) = 0.02452116137597196;
    result_local_M(0, 24) = 0.01618322273526407;
    result_local_M(0, 27) = 0.04544270078697304;
    result_local_M(1, 1) = -0.00863147348892433;
    result_local_M(1, 4) = -0.01096586628742749;
    result_local_M(1, 7) = -0.00974646045882031;
    result_local_M(1, 10) = -0.006134875867851393;
    result_local_M(1, 13) = 0.004047099946688711;
    result_local_M(1, 16) = 0.00889150049994481;
    result_local_M(1, 19) = 0.01764375010921981;
    result_local_M(1, 22) = 0.02452116137597196;
    result_local_M(1, 25) = 0.01618322273526407;
    result_local_M(1, 28) = 0.04544270078697304;
    result_local_M(2, 2) = -0.00863147348892433;
    result_local_M(2, 5) = -0.01096586628742749;
    result_local_M(2, 8) = -0.00974646045882031;
    result_local_M(2, 11) = -0.006134875867851393;
    result_local_M(2, 14) = 0.004047099946688711;
    result_local_M(2, 17) = 0.00889150049994481;
    result_local_M(2, 20) = 0.01764375010921981;
    result_local_M(2, 23) = 0.02452116137597196;
    result_local_M(2, 26) = 0.01618322273526407;
    result_local_M(2, 29) = 0.04544270078697304;
    result_local_M(3, 0) = -0.00847443417005874;
    result_local_M(3, 3) = 0.01087988385722117;
    result_local_M(3, 6) = -0.01272174787492477;
    result_local_M(3, 9) = -0.01277099392748555;
    result_local_M(3, 12) = 0.02572368323071238;
    result_local_M(3, 15) = 0.04835006894324498;
    result_local_M(3, 18) = 0.006800485891051233;
    result_local_M(3, 21) = 0.005001287964598679;
    result_local_M(3, 24) = 0.0415455295868809;
    result_local_M(3, 27) = 0.007309262387663336;
    result_local_M(4, 1) = -0.00847443417005874;
    result_local_M(4, 4) = 0.01087988385722117;
    result_local_M(4, 7) = -0.01272174787492477;
    result_local_M(4, 10) = -0.01277099392748555;
    result_local_M(4, 13) = 0.02572368323071238;
    result_local_M(4, 16) = 0.04835006894324498;
    result_local_M(4, 19) = 0.006800485891051233;
    result_local_M(4, 22) = 0.005001287964598679;
    result_local_M(4, 25) = 0.0415455295868809;
    result_local_M(4, 28) = 0.007309262387663336;
    result_local_M(5, 2) = -0.00847443417005874;
    result_local_M(5, 5) = 0.01087988385722117;
    result_local_M(5, 8) = -0.01272174787492477;
    result_local_M(5, 11) = -0.01277099392748555;
    result_local_M(5, 14) = 0.02572368323071238;
    result_local_M(5, 17) = 0.04835006894324498;
    result_local_M(5, 20) = 0.006800485891051233;
    result_local_M(5, 23) = 0.005001287964598679;
    result_local_M(5, 26) = 0.0415455295868809;
    result_local_M(5, 29) = 0.007309262387663336;
    result_local_M(6, 0) = -0.03069078676669274;
    result_local_M(6, 3) = -0.03181256001695487;
    result_local_M(6, 6) = -0.05152010701274446;
    result_local_M(6, 9) = -0.04615049987025355;
    result_local_M(6, 12) = 0.05290532165643471;
    result_local_M(6, 15) = 0.153277294786686;
    result_local_M(6, 18) = 0.04052574796562352;
    result_local_M(6, 21) = 0.04393865667007887;
    result_local_M(6, 24) = 0.1571085561284451;
    result_local_M(6, 27) = 0.1386429704278157;
    result_local_M(7, 1) = -0.03069078676669274;
    result_local_M(7, 4) = -0.03181256001695487;
    result_local_M(7, 7) = -0.05152010701274446;
    result_local_M(7, 10) = -0.04615049987025355;
    result_local_M(7, 13) = 0.05290532165643471;
    result_local_M(7, 16) = 0.153277294786686;
    result_local_M(7, 19) = 0.04052574796562352;
    result_local_M(7, 22) = 0.04393865667007887;
    result_local_M(7, 25) = 0.1571085561284451;
    result_local_M(7, 28) = 0.1386429704278157;
    result_local_M(8, 2) = -0.03069078676669274;
    result_local_M(8, 5) = -0.03181256001695487;
    result_local_M(8, 8) = -0.05152010701274446;
    result_local_M(8, 11) = -0.04615049987025355;
    result_local_M(8, 14) = 0.05290532165643471;
    result_local_M(8, 17) = 0.153277294786686;
    result_local_M(8, 20) = 0.04052574796562352;
    result_local_M(8, 23) = 0.04393865667007887;
    result_local_M(8, 26) = 0.1571085561284451;
    result_local_M(8, 29) = 0.1386429704278157;

    // Results for Kappa.
    result_local_kappa(0) = 0.0812507593510389;
    result_local_kappa(1) = 0.0812507593510389;
    result_local_kappa(2) = 0.0812507593510389;
    result_local_kappa(3) = 0.1116430258889036;
    result_local_kappa(4) = 0.1116430258889036;
    result_local_kappa(5) = 0.1116430258889036;
    result_local_kappa(6) = 0.4262245939684383;
    result_local_kappa(7) = 0.4262245939684383;
    result_local_kappa(8) = 0.4262245939684383;

    // Perform the unit tests.
    perform_mortar_pair_unit_test(contact_pair, q_beam, q_beam_rot, q_solid, result_local_D,
        result_local_M, result_local_kappa);
  }

  /**
   * \brief Test a non straight beam in a tet10 element, with line4 mortar shape functions.
   */
  TEST_F(BeamToSolidVolumeMeshtyingPairMortarTest, TestBeamToSolidMeshtyingMortarHermite2Tet10Line4)
  {
    // Element types.
    typedef GEOMETRYPAIR::t_hermite beam_type;
    typedef GEOMETRYPAIR::t_tet10 solid_type;
    typedef GEOMETRYPAIR::t_line4 lambda_type;

    // Create the mesh tying mortar pair.
    BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type, lambda_type>
        contact_pair = BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type,
            lambda_type>();

    // Definition of variables for this test case.
    GEOMETRYPAIR::ElementData<beam_type, double> q_beam;
    GEOMETRYPAIR::ElementData<solid_type, double> q_solid;
    Core::LinAlg::Matrix<9, 1> q_beam_rot;
    Core::LinAlg::SerialDenseMatrix local_D;
    Core::LinAlg::SerialDenseMatrix local_M;
    Core::LinAlg::SerialDenseVector local_kappa;

    // Matrices for the results.
    Core::LinAlg::Matrix<lambda_type::n_dof_, beam_type::n_dof_> result_local_D(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, solid_type::n_dof_> result_local_M(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, 1> result_local_kappa(
        Core::LinAlg::Initialization::zero);

    // Define the geometry of the two elements.
    q_beam.shape_function_data_.ref_length_ = 0.6192043571449604711;
    q_beam.element_position_(0) = 0.15;
    q_beam.element_position_(1) = 0.2;
    q_beam.element_position_(2) = 0.3;
    q_beam.element_position_(3) = 0.5773502691896255;
    q_beam.element_position_(4) = 0.5773502691896258;
    q_beam.element_position_(5) = 0.577350269189626;
    q_beam.element_position_(6) = 0.65;
    q_beam.element_position_(7) = 0.1;
    q_beam.element_position_(8) = 0.1;
    q_beam.element_position_(9) = 0.8017837257372733;
    q_beam.element_position_(10) = -0.5345224838248488;
    q_beam.element_position_(11) = 0.2672612419124244;

    q_beam_rot(0) = 1.674352746442651;
    q_beam_rot(1) = 0.1425949677148126;
    q_beam_rot(2) = 1.0831163124736984;
    q_beam_rot(3) = 1.4331999091513161;
    q_beam_rot(4) = -0.6560404572957742;
    q_beam_rot(5) = -0.2491376152457331;
    q_beam_rot(6) = 0.0;
    q_beam_rot(7) = 0.0;
    q_beam_rot(8) = 0.0;

    // Positional DOFs of the solid.
    q_solid.element_position_(0) = 0.04580648387340619;
    q_solid.element_position_(1) = 0.02451732788546579;
    q_solid.element_position_(2) = -0.0031131666281497;
    q_solid.element_position_(3) = 0.923762409575691;
    q_solid.element_position_(4) = -0.0106157274667526;
    q_solid.element_position_(5) = -0.0141609921026476;
    q_solid.element_position_(6) = -0.0944365111054278;
    q_solid.element_position_(7) = 1.069973754602123;
    q_solid.element_position_(8) = 0.06451121136737775;
    q_solid.element_position_(9) = -0.04783899250893692;
    q_solid.element_position_(10) = 0.0888801776910401;
    q_solid.element_position_(11) = 0.963088029848009;
    q_solid.element_position_(12) = 0.4108326210187769;
    q_solid.element_position_(13) = -0.06382048551478623;
    q_solid.element_position_(14) = 0.0810006716524518;
    q_solid.element_position_(15) = 0.4728874772489418;
    q_solid.element_position_(16) = 0.4910049605448334;
    q_solid.element_position_(17) = -0.07654076843449248;
    q_solid.element_position_(18) = 0.0891921101646426;
    q_solid.element_position_(19) = 0.5267989315386162;
    q_solid.element_position_(20) = -0.03523821168532098;
    q_solid.element_position_(21) = 0.05851054670001138;
    q_solid.element_position_(22) = -0.04567202467351366;
    q_solid.element_position_(23) = 0.4631680679379652;
    q_solid.element_position_(24) = 0.5925155012591516;
    q_solid.element_position_(25) = 0.02661668312122867;
    q_solid.element_position_(26) = 0.4341991229313329;
    q_solid.element_position_(27) = -0.006248620475302225;
    q_solid.element_position_(28) = 0.4109215888511089;
    q_solid.element_position_(29) = 0.4194107884465731;

    // Results for D.
    result_local_D(0, 0) = 0.05837517950584759;
    result_local_D(0, 3) = 0.002017208305318981;
    result_local_D(0, 6) = 0.005829701162185071;
    result_local_D(0, 9) = -0.0002955710710872768;
    result_local_D(1, 1) = 0.05837517950584759;
    result_local_D(1, 4) = 0.002017208305318981;
    result_local_D(1, 7) = 0.005829701162185071;
    result_local_D(1, 10) = -0.0002955710710872768;
    result_local_D(2, 2) = 0.05837517950584759;
    result_local_D(2, 5) = 0.002017208305318981;
    result_local_D(2, 8) = 0.005829701162185071;
    result_local_D(2, 11) = -0.0002955710710872768;
    result_local_D(3, 0) = 0.004689762733534804;
    result_local_D(3, 3) = 0.0002955710710872767;
    result_local_D(3, 6) = 0.07072106759232025;
    result_local_D(3, 9) = -0.002723063147216203;
    result_local_D(4, 1) = 0.004689762733534804;
    result_local_D(4, 4) = 0.0002955710710872767;
    result_local_D(4, 7) = 0.07072106759232025;
    result_local_D(4, 10) = -0.002723063147216203;
    result_local_D(5, 2) = 0.004689762733534804;
    result_local_D(5, 5) = 0.0002955710710872767;
    result_local_D(5, 8) = 0.07072106759232025;
    result_local_D(5, 11) = -0.002723063147216203;
    result_local_D(6, 0) = 0.184104753994235;
    result_local_D(6, 3) = 0.0191687182192921;
    result_local_D(6, 6) = 0.02686710479294802;
    result_local_D(6, 9) = -0.00825428928975331;
    result_local_D(7, 1) = 0.184104753994235;
    result_local_D(7, 4) = 0.0191687182192921;
    result_local_D(7, 7) = 0.02686710479294802;
    result_local_D(7, 10) = -0.00825428928975331;
    result_local_D(8, 2) = 0.184104753994235;
    result_local_D(8, 5) = 0.0191687182192921;
    result_local_D(8, 8) = 0.02686710479294802;
    result_local_D(8, 11) = -0.00825428928975331;
    result_local_D(9, 0) = 0.04292973412829792;
    result_local_D(9, 3) = 0.01018849022778634;
    result_local_D(9, 6) = 0.2256010752990122;
    result_local_D(9, 9) = -0.02303712009535817;
    result_local_D(10, 1) = 0.04292973412829792;
    result_local_D(10, 4) = 0.01018849022778634;
    result_local_D(10, 7) = 0.2256010752990122;
    result_local_D(10, 10) = -0.02303712009535817;
    result_local_D(11, 2) = 0.04292973412829792;
    result_local_D(11, 5) = 0.01018849022778634;
    result_local_D(11, 8) = 0.2256010752990122;
    result_local_D(11, 11) = -0.02303712009535817;

    // Results for M.
    result_local_M(0, 0) = -0.007348545771047808;
    result_local_M(0, 3) = -0.004457638338736185;
    result_local_M(0, 6) = -0.007741862001406275;
    result_local_M(0, 9) = -0.005418544507234328;
    result_local_M(0, 12) = 0.006784040014523343;
    result_local_M(0, 15) = 0.00985263171062604;
    result_local_M(0, 18) = 0.01410694721550188;
    result_local_M(0, 21) = 0.0189106461746733;
    result_local_M(0, 24) = 0.01231411171699897;
    result_local_M(0, 27) = 0.02720309445413372;
    result_local_M(1, 1) = -0.007348545771047808;
    result_local_M(1, 4) = -0.004457638338736185;
    result_local_M(1, 7) = -0.007741862001406275;
    result_local_M(1, 10) = -0.005418544507234328;
    result_local_M(1, 13) = 0.006784040014523343;
    result_local_M(1, 16) = 0.00985263171062604;
    result_local_M(1, 19) = 0.01410694721550188;
    result_local_M(1, 22) = 0.0189106461746733;
    result_local_M(1, 25) = 0.01231411171699897;
    result_local_M(1, 28) = 0.02720309445413372;
    result_local_M(2, 2) = -0.007348545771047808;
    result_local_M(2, 5) = -0.004457638338736185;
    result_local_M(2, 8) = -0.007741862001406275;
    result_local_M(2, 11) = -0.005418544507234328;
    result_local_M(2, 14) = 0.006784040014523343;
    result_local_M(2, 17) = 0.00985263171062604;
    result_local_M(2, 20) = 0.01410694721550188;
    result_local_M(2, 23) = 0.0189106461746733;
    result_local_M(2, 26) = 0.01231411171699897;
    result_local_M(2, 29) = 0.02720309445413372;
    result_local_M(3, 0) = -0.005921013542098668;
    result_local_M(3, 3) = 0.00834822591064923;
    result_local_M(3, 6) = -0.00828633295574575;
    result_local_M(3, 9) = -0.007718512804320918;
    result_local_M(3, 12) = 0.01637357795582341;
    result_local_M(3, 15) = 0.028229275884228;
    result_local_M(3, 18) = 0.00527157028906622;
    result_local_M(3, 21) = 0.005119471082137474;
    result_local_M(3, 24) = 0.02577607108909037;
    result_local_M(3, 27) = 0.00821849741702568;
    result_local_M(4, 1) = -0.005921013542098668;
    result_local_M(4, 4) = 0.00834822591064923;
    result_local_M(4, 7) = -0.00828633295574575;
    result_local_M(4, 10) = -0.007718512804320918;
    result_local_M(4, 13) = 0.01637357795582341;
    result_local_M(4, 16) = 0.028229275884228;
    result_local_M(4, 19) = 0.00527157028906622;
    result_local_M(4, 22) = 0.005119471082137474;
    result_local_M(4, 25) = 0.02577607108909037;
    result_local_M(4, 28) = 0.00821849741702568;
    result_local_M(5, 2) = -0.005921013542098668;
    result_local_M(5, 5) = 0.00834822591064923;
    result_local_M(5, 8) = -0.00828633295574575;
    result_local_M(5, 11) = -0.007718512804320918;
    result_local_M(5, 14) = 0.01637357795582341;
    result_local_M(5, 17) = 0.028229275884228;
    result_local_M(5, 20) = 0.00527157028906622;
    result_local_M(5, 23) = 0.005119471082137474;
    result_local_M(5, 26) = 0.02577607108909037;
    result_local_M(5, 29) = 0.00821849741702568;
    result_local_M(6, 0) = -0.01535782819113933;
    result_local_M(6, 3) = -0.031454393852432;
    result_local_M(6, 6) = -0.02533383550202128;
    result_local_M(6, 9) = -0.01945543153319626;
    result_local_M(6, 12) = 0.01162867541765913;
    result_local_M(6, 15) = 0.05459559191296358;
    result_local_M(6, 18) = 0.02580756416826259;
    result_local_M(6, 21) = 0.03330854185517554;
    result_local_M(6, 24) = 0.07052304160296225;
    result_local_M(6, 27) = 0.1067099329089488;
    result_local_M(7, 1) = -0.01535782819113933;
    result_local_M(7, 4) = -0.031454393852432;
    result_local_M(7, 7) = -0.02533383550202128;
    result_local_M(7, 10) = -0.01945543153319626;
    result_local_M(7, 13) = 0.01162867541765913;
    result_local_M(7, 16) = 0.05459559191296358;
    result_local_M(7, 19) = 0.02580756416826259;
    result_local_M(7, 22) = 0.03330854185517554;
    result_local_M(7, 25) = 0.07052304160296225;
    result_local_M(7, 28) = 0.1067099329089488;
    result_local_M(8, 2) = -0.01535782819113933;
    result_local_M(8, 5) = -0.031454393852432;
    result_local_M(8, 8) = -0.02533383550202128;
    result_local_M(8, 11) = -0.01945543153319626;
    result_local_M(8, 14) = 0.01162867541765913;
    result_local_M(8, 17) = 0.05459559191296358;
    result_local_M(8, 20) = 0.02580756416826259;
    result_local_M(8, 23) = 0.03330854185517554;
    result_local_M(8, 26) = 0.07052304160296225;
    result_local_M(8, 29) = 0.1067099329089488;
    result_local_M(9, 0) = -0.01916930692138999;
    result_local_M(9, 3) = -0.004334736166642239;
    result_local_M(9, 6) = -0.03262628488731623;
    result_local_M(9, 9) = -0.03246388082083898;
    result_local_M(9, 12) = 0.04788981144582992;
    result_local_M(9, 15) = 0.1178413647220582;
    result_local_M(9, 18) = 0.01978390229306386;
    result_local_M(9, 21) = 0.01612244689866319;
    result_local_M(9, 24) = 0.1062240840415385;
    result_local_M(9, 27) = 0.04926340882234387;
    result_local_M(10, 1) = -0.01916930692138999;
    result_local_M(10, 4) = -0.004334736166642239;
    result_local_M(10, 7) = -0.03262628488731623;
    result_local_M(10, 10) = -0.03246388082083898;
    result_local_M(10, 13) = 0.04788981144582992;
    result_local_M(10, 16) = 0.1178413647220582;
    result_local_M(10, 19) = 0.01978390229306386;
    result_local_M(10, 22) = 0.01612244689866319;
    result_local_M(10, 25) = 0.1062240840415385;
    result_local_M(10, 28) = 0.04926340882234387;
    result_local_M(11, 2) = -0.01916930692138999;
    result_local_M(11, 5) = -0.004334736166642239;
    result_local_M(11, 8) = -0.03262628488731623;
    result_local_M(11, 11) = -0.03246388082083898;
    result_local_M(11, 14) = 0.04788981144582992;
    result_local_M(11, 17) = 0.1178413647220582;
    result_local_M(11, 20) = 0.01978390229306386;
    result_local_M(11, 23) = 0.01612244689866319;
    result_local_M(11, 26) = 0.1062240840415385;
    result_local_M(11, 29) = 0.04926340882234387;

    // Results for Kappa.
    result_local_kappa(0) = 0.06420488066803266;
    result_local_kappa(1) = 0.06420488066803266;
    result_local_kappa(2) = 0.06420488066803266;
    result_local_kappa(3) = 0.07541083032585504;
    result_local_kappa(4) = 0.07541083032585504;
    result_local_kappa(5) = 0.07541083032585504;
    result_local_kappa(6) = 0.210971858787183;
    result_local_kappa(7) = 0.210971858787183;
    result_local_kappa(8) = 0.210971858787183;
    result_local_kappa(9) = 0.2685308094273101;
    result_local_kappa(10) = 0.2685308094273101;
    result_local_kappa(11) = 0.2685308094273101;

    // Perform the unit tests.
    perform_mortar_pair_unit_test(contact_pair, q_beam, q_beam_rot, q_solid, result_local_D,
        result_local_M, result_local_kappa);
  }

}  // namespace
