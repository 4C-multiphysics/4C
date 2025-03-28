// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_contact_element.hpp"
#include "4C_contact_selfcontact_binarytree_unbiased.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

namespace
{
  using namespace FourC;

  class UtilsRefConfigTest : public testing::Test
  {
   public:
    std::shared_ptr<Core::FE::Discretization> testdis_;

    UtilsRefConfigTest()
    {
      // create a discretization, to store the created elements and nodes
      testdis_ = std::make_shared<Core::FE::Discretization>("dummy", MPI_COMM_WORLD, 3);

      // create hex8 element and store it in the test discretization
      const std::array<int, 8> nodeidshex8 = {0, 1, 2, 3, 4, 5, 6, 7};
      const std::vector<std::vector<double>> coordshex8 = {{-0.10, -0.20, -0.50},
          {1.25, 0.23, 0.66}, {1.20, 0.99, 0.50}, {-0.11, 1.20, 0.66}, {-0.10, -0.20, 1.90},
          {1.00, 0.00, 1.90}, {1.20, 0.99, 1.50}, {-0.11, -0.20, 1.66}};
      for (int i = 0; i < 8; ++i)
      {
        testdis_->add_node(std::make_shared<Core::Nodes::Node>(nodeidshex8[i], coordshex8[i], 0));
      }
      std::shared_ptr<Discret::Elements::Solid> testhex8ele =
          std::make_shared<Discret::Elements::Solid>(0, 0);
      testhex8ele->set_node_ids(8, nodeidshex8.data());
      testdis_->add_element(testhex8ele);

      // create corresponding quad4 surface contact element and store it
      std::shared_ptr<CONTACT::Element> testcontactquad4ele =
          std::make_shared<CONTACT::Element>(testhex8ele->id() + 1, testhex8ele->owner(),
              testhex8ele->shape(), testhex8ele->num_node(), testhex8ele->node_ids(), false, false);
      testdis_->add_element(testcontactquad4ele);

      // create tet4 element and store it in the test discretization
      const std::array<int, 4> nodeidstet4 = {8, 9, 10, 11};
      const std::vector<std::vector<double>> coordstet4 = {
          {2.5, -0.5, 0.0}, {1.0, -1.1, 0.1}, {1.1, 0.11, 0.15}, {1.5, -0.5, 2.0}};
      for (int j = 0; j < 4; ++j)
      {
        testdis_->add_node(std::make_shared<Core::Nodes::Node>(nodeidstet4[j], coordstet4[j], 0));
      }
      std::shared_ptr<Discret::Elements::Solid> testtet4ele =
          std::make_shared<Discret::Elements::Solid>(2, 0);
      testtet4ele->set_node_ids(4, nodeidstet4.data());
      testdis_->add_element(testtet4ele);

      // create corresponding tri3 surface contact element and store it
      std::shared_ptr<CONTACT::Element> testcontacttri3ele =
          std::make_shared<CONTACT::Element>(testtet4ele->id() + 1, testtet4ele->owner(),
              testtet4ele->shape(), testtet4ele->num_node(), testtet4ele->node_ids(), false, false);
      testdis_->add_element(testcontacttri3ele);
      testdis_->fill_complete(false, false, false);
    }
  };

  TEST_F(UtilsRefConfigTest, LocalToGlobalPositionAtXiRefConfig)
  {
    // get hex8 element and test it
    const Core::Elements::Element* hex8ele = testdis_->g_element(0);
    Core::LinAlg::Matrix<3, 1> xicenterhex8ele(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> hex8elecoords(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> hex8refsolution(Core::LinAlg::Initialization::zero);
    hex8refsolution(0, 0) = 423.0 / 800.0;
    hex8refsolution(1, 0) = 281.0 / 800.0;
    hex8refsolution(2, 0) = 207.0 / 200.0;
    CONTACT::local_to_global_position_at_xi_ref_config<3, Core::FE::CellType::hex8>(
        hex8ele, xicenterhex8ele, hex8elecoords);

    FOUR_C_EXPECT_NEAR(hex8elecoords, hex8refsolution, 1e-14);

    // get quad4 element and test it
    const Core::Elements::Element* quad4ele = testdis_->g_element(1);
    Core::LinAlg::Matrix<2, 1> xicenterquad4ele(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> quad4elecoords(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> quad4refsolution(Core::LinAlg::Initialization::zero);
    quad4refsolution(0, 0) = 14.0 / 25.0;
    quad4refsolution(1, 0) = 111.0 / 200.0;
    quad4refsolution(2, 0) = 33.0 / 100.0;
    CONTACT::local_to_global_position_at_xi_ref_config<3, Core::FE::CellType::quad4>(
        quad4ele, xicenterquad4ele, quad4elecoords);

    FOUR_C_EXPECT_NEAR(quad4elecoords, quad4refsolution, 1e-14);

    // get tet4 element stuff and test it
    const Core::Elements::Element* tet4ele = testdis_->g_element(2);
    Core::LinAlg::Matrix<3, 1> xicentertet4ele(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> tet4elecoords(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> tet4refsolution(Core::LinAlg::Initialization::zero);
    tet4refsolution(0, 0) = 61.0 / 40.0;
    tet4refsolution(1, 0) = -199.0 / 400.0;
    tet4refsolution(2, 0) = 9.0 / 16.0;
    xicentertet4ele.put_scalar(1.0 / 4.0);
    CONTACT::local_to_global_position_at_xi_ref_config<3, Core::FE::CellType::tet4>(
        tet4ele, xicentertet4ele, tet4elecoords);

    FOUR_C_EXPECT_NEAR(tet4elecoords, tet4refsolution, 1e-14);

    // get tri3 element and test it
    const Core::Elements::Element* tri3ele = testdis_->g_element(3);
    Core::LinAlg::Matrix<2, 1> xicentertri3ele(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> tri3elecoords(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> tri3refsolution(Core::LinAlg::Initialization::zero);
    tri3refsolution(0, 0) = 23.0 / 15.0;
    tri3refsolution(1, 0) = -149.0 / 300.0;
    tri3refsolution(2, 0) = 1.0 / 12.0;
    xicentertri3ele.put_scalar(1.0 / 3.0);
    CONTACT::local_to_global_position_at_xi_ref_config<3, Core::FE::CellType::tri3>(
        tri3ele, xicentertri3ele, tri3elecoords);

    FOUR_C_EXPECT_NEAR(tri3elecoords, tri3refsolution, 1e-14);
  }

  TEST_F(UtilsRefConfigTest, ComputeUnitNormalAtXiRefConfig)
  {
    // get quad4 element and test it
    const Core::Elements::Element* quad4ele = testdis_->g_element(1);
    Core::LinAlg::Matrix<2, 1> xicenterquad4ele(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> quad4elecoords(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> quad4refsolution(Core::LinAlg::Initialization::zero);
    quad4refsolution(0, 0) = -0.29138926578643;
    quad4refsolution(1, 0) = -0.40854577471087;
    quad4refsolution(2, 0) = 0.86497551742829;
    CONTACT::compute_unit_normal_at_xi_ref_config<Core::FE::CellType::quad4>(
        quad4ele, xicenterquad4ele, quad4elecoords);

    FOUR_C_EXPECT_NEAR(quad4elecoords, quad4refsolution, 1e-14);

    // get tri3 element and test it
    const Core::Elements::Element* tri3ele = testdis_->g_element(3);
    Core::LinAlg::Matrix<2, 1> xicentertri3ele(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> tri3elecoords(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> tri3refsolution(Core::LinAlg::Initialization::zero);
    tri3refsolution(0, 0) = -0.085623542490578;
    tri3refsolution(1, 0) = 0.048198682858935;
    tri3refsolution(2, 0) = -0.995161040205065;
    xicentertri3ele.put_scalar(1.0 / 3.0);
    CONTACT::compute_unit_normal_at_xi_ref_config<Core::FE::CellType::tri3>(
        tri3ele, xicentertri3ele, tri3elecoords);

    FOUR_C_EXPECT_NEAR(tri3elecoords, tri3refsolution, 1e-14);
  }
}  // namespace