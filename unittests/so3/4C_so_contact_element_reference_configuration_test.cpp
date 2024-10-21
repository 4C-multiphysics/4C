#include <gtest/gtest.h>

#include "4C_contact_element.hpp"
#include "4C_contact_selfcontact_binarytree_unbiased.hpp"
#include "4C_so3_hex8.hpp"
#include "4C_so3_tet4.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

#include <Epetra_SerialComm.h>

namespace
{
  using namespace FourC;

  class UtilsRefConfigTest : public testing::Test
  {
   public:
    Teuchos::RCP<Core::FE::Discretization> testdis_;

    UtilsRefConfigTest()
    {
      // create a discretization, to store the created elements and nodes
      testdis_ = Teuchos::make_rcp<Core::FE::Discretization>(
          "dummy", Teuchos::make_rcp<Epetra_SerialComm>(), 3);

      // create hex8 element and store it in the test discretization
      const std::array<int, 8> nodeidshex8 = {0, 1, 2, 3, 4, 5, 6, 7};
      const std::vector<std::vector<double>> coordshex8 = {{-0.10, -0.20, -0.50},
          {1.25, 0.23, 0.66}, {1.20, 0.99, 0.50}, {-0.11, 1.20, 0.66}, {-0.10, -0.20, 1.90},
          {1.00, 0.00, 1.90}, {1.20, 0.99, 1.50}, {-0.11, -0.20, 1.66}};
      for (int i = 0; i < 8; ++i)
      {
        testdis_->add_node(Teuchos::make_rcp<Core::Nodes::Node>(nodeidshex8[i], coordshex8[i], 0));
      }
      Teuchos::RCP<Discret::ELEMENTS::SoHex8> testhex8ele =
          Teuchos::make_rcp<Discret::ELEMENTS::SoHex8>(0, 0);
      testhex8ele->set_node_ids(8, nodeidshex8.data());
      testdis_->add_element(testhex8ele);

      // create corresponding quad4 surface contact element and store it
      Teuchos::RCP<CONTACT::Element> testcontactquad4ele =
          Teuchos::make_rcp<CONTACT::Element>(testhex8ele->id() + 1, testhex8ele->owner(),
              testhex8ele->shape(), testhex8ele->num_node(), testhex8ele->node_ids(), false, false);
      testdis_->add_element(testcontactquad4ele);

      // create tet4 element and store it in the test discretization
      const std::array<int, 4> nodeidstet4 = {8, 9, 10, 11};
      const std::vector<std::vector<double>> coordstet4 = {
          {2.5, -0.5, 0.0}, {1.0, -1.1, 0.1}, {1.1, 0.11, 0.15}, {1.5, -0.5, 2.0}};
      for (int j = 0; j < 4; ++j)
      {
        testdis_->add_node(Teuchos::make_rcp<Core::Nodes::Node>(nodeidstet4[j], coordstet4[j], 0));
      }
      Teuchos::RCP<Discret::ELEMENTS::SoTet4> testtet4ele =
          Teuchos::make_rcp<Discret::ELEMENTS::SoTet4>(2, 0);
      testtet4ele->set_node_ids(4, nodeidstet4.data());
      testdis_->add_element(testtet4ele);

      // create corresponding tri3 surface contact element and store it
      Teuchos::RCP<CONTACT::Element> testcontacttri3ele =
          Teuchos::make_rcp<CONTACT::Element>(testtet4ele->id() + 1, testtet4ele->owner(),
              testtet4ele->shape(), testtet4ele->num_node(), testtet4ele->node_ids(), false, false);
      testdis_->add_element(testcontacttri3ele);
      testdis_->fill_complete(false, false, false);
    }
  };

  TEST_F(UtilsRefConfigTest, LocalToGlobalPositionAtXiRefConfig)
  {
    // get hex8 element and test it
    const Core::Elements::Element* hex8ele = testdis_->g_element(0);
    Core::LinAlg::Matrix<3, 1> xicenterhex8ele(true);
    Core::LinAlg::Matrix<3, 1> hex8elecoords(true);
    Core::LinAlg::Matrix<3, 1> hex8refsolution(true);
    hex8refsolution(0, 0) = 423.0 / 800.0;
    hex8refsolution(1, 0) = 281.0 / 800.0;
    hex8refsolution(2, 0) = 207.0 / 200.0;
    CONTACT::local_to_global_position_at_xi_ref_config<3, Core::FE::CellType::hex8>(
        hex8ele, xicenterhex8ele, hex8elecoords);

    FOUR_C_EXPECT_NEAR(hex8elecoords, hex8refsolution, 1e-14);

    // get quad4 element and test it
    const Core::Elements::Element* quad4ele = testdis_->g_element(1);
    Core::LinAlg::Matrix<2, 1> xicenterquad4ele(true);
    Core::LinAlg::Matrix<3, 1> quad4elecoords(true);
    Core::LinAlg::Matrix<3, 1> quad4refsolution(true);
    quad4refsolution(0, 0) = 14.0 / 25.0;
    quad4refsolution(1, 0) = 111.0 / 200.0;
    quad4refsolution(2, 0) = 33.0 / 100.0;
    CONTACT::local_to_global_position_at_xi_ref_config<3, Core::FE::CellType::quad4>(
        quad4ele, xicenterquad4ele, quad4elecoords);

    FOUR_C_EXPECT_NEAR(quad4elecoords, quad4refsolution, 1e-14);

    // get tet4 element stuff and test it
    const Core::Elements::Element* tet4ele = testdis_->g_element(2);
    Core::LinAlg::Matrix<3, 1> xicentertet4ele(true);
    Core::LinAlg::Matrix<3, 1> tet4elecoords(true);
    Core::LinAlg::Matrix<3, 1> tet4refsolution(true);
    tet4refsolution(0, 0) = 61.0 / 40.0;
    tet4refsolution(1, 0) = -199.0 / 400.0;
    tet4refsolution(2, 0) = 9.0 / 16.0;
    xicentertet4ele.put_scalar(1.0 / 4.0);
    CONTACT::local_to_global_position_at_xi_ref_config<3, Core::FE::CellType::tet4>(
        tet4ele, xicentertet4ele, tet4elecoords);

    FOUR_C_EXPECT_NEAR(tet4elecoords, tet4refsolution, 1e-14);

    // get tri3 element and test it
    const Core::Elements::Element* tri3ele = testdis_->g_element(3);
    Core::LinAlg::Matrix<2, 1> xicentertri3ele(true);
    Core::LinAlg::Matrix<3, 1> tri3elecoords(true);
    Core::LinAlg::Matrix<3, 1> tri3refsolution(true);
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
    Core::LinAlg::Matrix<2, 1> xicenterquad4ele(true);
    Core::LinAlg::Matrix<3, 1> quad4elecoords(true);
    Core::LinAlg::Matrix<3, 1> quad4refsolution(true);
    quad4refsolution(0, 0) = -0.29138926578643;
    quad4refsolution(1, 0) = -0.40854577471087;
    quad4refsolution(2, 0) = 0.86497551742829;
    CONTACT::compute_unit_normal_at_xi_ref_config<Core::FE::CellType::quad4>(
        quad4ele, xicenterquad4ele, quad4elecoords);

    FOUR_C_EXPECT_NEAR(quad4elecoords, quad4refsolution, 1e-14);

    // get tri3 element and test it
    const Core::Elements::Element* tri3ele = testdis_->g_element(3);
    Core::LinAlg::Matrix<2, 1> xicentertri3ele(true);
    Core::LinAlg::Matrix<3, 1> tri3elecoords(true);
    Core::LinAlg::Matrix<3, 1> tri3refsolution(true);
    tri3refsolution(0, 0) = -0.085623542490578;
    tri3refsolution(1, 0) = 0.048198682858935;
    tri3refsolution(2, 0) = -0.995161040205065;
    xicentertri3ele.put_scalar(1.0 / 3.0);
    CONTACT::compute_unit_normal_at_xi_ref_config<Core::FE::CellType::tri3>(
        tri3ele, xicentertri3ele, tri3elecoords);

    FOUR_C_EXPECT_NEAR(tri3elecoords, tri3refsolution, 1e-14);
  }
}  // namespace