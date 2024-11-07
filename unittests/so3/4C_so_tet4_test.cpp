// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_so3_tet4.hpp"

#include <Epetra_SerialComm.h>

namespace
{
  using namespace FourC;

  class SoTet4Test : public ::testing::Test
  {
   protected:
    void SetUp() override
    {
      // create a discretization, that creates node to element pointers and keeps the nodes alive
      testdis_ = std::make_shared<Core::FE::Discretization>(
          "dummy", std::make_shared<Epetra_SerialComm>(), 3);

      // create 4 nodes
      const std::array<int, 4> nodeids = {0, 1, 2, 3};
      std::vector<std::vector<double>> coords = {
          {-0.1, -0.2, -0.5}, {1.25, 0.23, 0.66}, {1.20, 0.99, 0.5}, {-0.10, -0.2, 1.96}};
      for (int lid = 0; lid < 4; ++lid)
        testdis_->add_node(std::make_shared<Core::Nodes::Node>(lid, coords[lid], 0));

      // create 1 element
      testele_ = std::make_shared<Discret::Elements::SoTet4>(0, 0);
      testele_->set_node_ids(4, nodeids.data());
      testdis_->add_element(testele_);
      testdis_->fill_complete(false, false, false);

      copytestele_ = std::make_shared<Discret::Elements::SoTet4>(*testele_);
    }

    // Delete pointers.
    void TearDown() override
    {
      copytestele_ = nullptr;
      testele_ = nullptr;
      testdis_ = nullptr;

      // We need to make sure the Global::Problem instance created in setUp is deleted again. If
      // this is not done, some troubles arise where unit tests influence each other on some
      // configurations. We suspect that missing singleton destruction might be the reason for that.
      Global::Problem::done();
    }
    //! dummy discretization for holding element and node pointers
    std::shared_ptr<Core::FE::Discretization> testdis_;
    //! the tet4 element to be tested
    std::shared_ptr<Discret::Elements::SoTet4> testele_;
    //! a copy of the tet element to test the copy constructor
    std::shared_ptr<Discret::Elements::SoTet4> copytestele_;
  };

  /**
   * Test Number of DOFs per node function
   */
  TEST_F(SoTet4Test, TestNumDofPerNode)
  {
    std::vector<double> pd = {1, 2, 3};
    Core::Nodes::Node node_dummy(0, pd, false);
    EXPECT_EQ(testele_->num_dof_per_node(node_dummy), 3);
    EXPECT_EQ(copytestele_->num_dof_per_node(node_dummy), 3);
  }

  /**
   * Test Number of DOFs per element
   */
  TEST_F(SoTet4Test, TestNumDofPerElement)
  {
    EXPECT_EQ(testele_->num_dof_per_element(), 0);
    EXPECT_EQ(copytestele_->num_dof_per_element(), 0);
  }

  /**
   * Test the polynomial degree
   */
  TEST_F(SoTet4Test, TestDegree)
  {
    EXPECT_EQ(testele_->degree(), 1);
    EXPECT_EQ(copytestele_->degree(), 1);
  }

  /**
   * Test the number of volumes the element is composed of
   */
  TEST_F(SoTet4Test, TestNumVolume)
  {
    EXPECT_EQ(testele_->num_volume(), 1);
    EXPECT_EQ(copytestele_->num_volume(), 1);
  }

  /**
   * Test the number of surfaces the element is composed of
   */
  TEST_F(SoTet4Test, TestNumSurface)
  {
    EXPECT_EQ(testele_->num_surface(), 4);
    EXPECT_EQ(copytestele_->num_surface(), 4);
  }

  /**
   * Test the number of lines the element is composed of
   */
  TEST_F(SoTet4Test, TestNumLine)
  {
    EXPECT_EQ(testele_->num_line(), 6);
    EXPECT_EQ(copytestele_->num_line(), 6);
  }

  /**
   * Test the calculation of the element center coordinates
   */
  TEST_F(SoTet4Test, TestElementCenterRefeCoords)
  {
    double midpoint[3] = {0.5625, 0.2050, 0.6550};
    for (int i = 0; i < 3; ++i)
    {
      EXPECT_NEAR(testele_->element_center_refe_coords()[i], midpoint[i], 1e-14);
      EXPECT_NEAR(copytestele_->element_center_refe_coords()[i], midpoint[i], 1e-14);
    }
  }

}  // namespace
