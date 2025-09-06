// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_beam3_euler_bernoulli.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

#include <array>

const double testTolerance = 1e-14;

namespace
{

  using namespace FourC;
  class Beam3eb : public ::testing::Test
  {
   public:
    Beam3eb()
    {
      testdis_ = std::make_shared<Core::FE::Discretization>("Beam3eb", MPI_COMM_WORLD, 3);

      std::vector<std::vector<double>> xrefe{{-0.05, 0.05, 0.3}, {0.45, -0.05, 0.1}};
      std::vector<double> xrefe_full{-0.05, 0.05, 0.3, 0.45, -0.05, 0.1};

      for (int lid = 0; lid < 2; ++lid)
        testdis_->add_node(std::make_shared<Core::Nodes::Node>(lid, xrefe[lid], 0));

      testele_ = std::make_shared<Discret::Elements::Beam3eb>(0, 0);
      std::array<int, 2> node_ids{0, 1};
      testele_->set_node_ids(2, node_ids.data());

      // create 1 element discretization
      testdis_->add_element(testele_);
      testdis_->fill_complete(false, false, false);

      testele_->set_up_reference_geometry(xrefe_full);
    }

   protected:
    //! dummy discretization for holding element and node pointers
    std::shared_ptr<Core::FE::Discretization> testdis_;
    //! the beam3eb element to be tested
    std::shared_ptr<Discret::Elements::Beam3eb> testele_;
  };

  /**
   * Test reference length calculation of Euler-Bernoulli beam
   */
  TEST_F(Beam3eb, RefLength)
  {
    EXPECT_NEAR(testele_->ref_length(), 0.5477225575051661, testTolerance);
  }

  /**
   * Test nodal nullspace calculation of Euler-Bernoulli beam
   */
  TEST_F(Beam3eb, ComputeNullSpace)
  {
    using namespace FourC;

    // nodal nullspace calculation for reference center of discretization at {0.0, 0.0, 0.0}
    // at node {-0.05, 0.05, 0.3}
    {
      Core::LinAlg::SerialDenseMatrix nullspace_ref(6, 5);
      nullspace_ref(0, 0) = 1.0;
      nullspace_ref(0, 3) = -0.273861278752583;
      nullspace_ref(0, 4) = 0.063333333333333;
      nullspace_ref(1, 1) = 1.0;
      nullspace_ref(1, 3) = 0.054772255750517;
      nullspace_ref(1, 4) = 0.143333333333333;
      nullspace_ref(2, 2) = 1.0;
      nullspace_ref(2, 3) = -0.054772255750517;
      nullspace_ref(2, 4) = -0.013333333333333;
      nullspace_ref(3, 3) = 0.333333333333333;
      nullspace_ref(3, 4) = -0.182574185835055;
      nullspace_ref(4, 3) = -0.066666666666667;
      nullspace_ref(4, 4) = -0.912870929175277;
      nullspace_ref(5, 3) = 0.866666666666667;

      const auto node = testele_->nodes()[0];
      int numdof, dimnsp;

      testele_->element_type().nodal_block_information(node->elements()[0], numdof, dimnsp);
      Core::LinAlg::SerialDenseMatrix nullspace = testele_->element_type().compute_null_space(
          *node, std::vector{0.0, 0.0, 0.0}.data(), numdof, dimnsp);

      FOUR_C_EXPECT_NEAR(nullspace, nullspace_ref, testTolerance);
    }

    // nodal nullspace calculation for reference center of discretization at {-0.05, 0.05, 0.3}
    // at node {-0.05, 0.05, 0.3} -> rotational components in displacement vanish
    {
      Core::LinAlg::SerialDenseMatrix nullspace_ref(6, 5);
      nullspace_ref(0, 0) = 1.0;
      nullspace_ref(1, 1) = 1.0;
      nullspace_ref(2, 2) = 1.0;
      nullspace_ref(3, 3) = 0.333333333333333;
      nullspace_ref(3, 4) = -0.182574185835055;
      nullspace_ref(4, 3) = -0.066666666666667;
      nullspace_ref(4, 4) = -0.912870929175277;
      nullspace_ref(5, 3) = 0.866666666666667;

      const auto node = testele_->nodes()[0];
      int numdof, dimnsp;

      testele_->element_type().nodal_block_information(node->elements()[0], numdof, dimnsp);
      Core::LinAlg::SerialDenseMatrix nullspace = testele_->element_type().compute_null_space(
          *node, std::vector{-0.05, 0.05, 0.3}.data(), numdof, dimnsp);

      FOUR_C_EXPECT_NEAR(nullspace, nullspace_ref, testTolerance);
    }
  }

}  // namespace
