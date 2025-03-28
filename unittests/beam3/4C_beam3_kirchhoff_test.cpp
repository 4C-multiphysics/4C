// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_beam3_kirchhoff.hpp"

#include "4C_fem_general_element.hpp"

#include <array>

const double testTolerance = 1e-14;

namespace
{
  using namespace FourC;

  class Beam3k : public ::testing::Test
  {
   public:
    Beam3k()
    {
      testdis_ = std::make_shared<Core::FE::Discretization>("Beam3k", MPI_COMM_WORLD, 3);

      std::vector<std::vector<double>> xrefe{{-0.05, 0.05, 0.3}, {0.45, -0.05, 0.1}};

      for (int lid = 0; lid < 2; ++lid)
        testdis_->add_node(std::make_shared<Core::Nodes::Node>(lid, xrefe[lid], 0));

      testele_ = std::make_shared<Discret::Elements::Beam3k>(0, 0);
      std::array<int, 2> node_ids{0, 1};
      testele_->set_node_ids(2, node_ids.data());

      // create 1 element discretization
      testdis_->add_element(testele_);
      testdis_->fill_complete(false, false, false);

      // setup internal beam element parameters
      // different data layout is necessary to call this method
      Core::LinAlg::Matrix<3, 1> coord1(Core::LinAlg::Initialization::zero);
      coord1(0) = xrefe[0][0];
      coord1(1) = xrefe[0][1];
      coord1(2) = xrefe[0][2];
      Core::LinAlg::Matrix<3, 1> coord2(Core::LinAlg::Initialization::zero);
      coord2(0) = xrefe[1][0];
      coord2(1) = xrefe[1][1];
      coord2(2) = xrefe[1][2];
      std::vector<Core::LinAlg::Matrix<3, 1>> xrefe_setup{coord1, coord2};

      // setup internal beam element parameters
      std::vector<double> rotrefe(9);
      rotrefe[0] = -2.135698785951414;
      rotrefe[1] = -1.1055190408131161;
      rotrefe[2] = -0.45792098016648797;
      rotrefe[3] = 0.09071600605476587;
      rotrefe[4] = -0.31314870676006484;
      rotrefe[5] = -0.5590172175309829;
      rotrefe[6] = -0.44757433200569813;
      rotrefe[7] = -0.14845112617443665;
      rotrefe[8] = -0.628849061811312;

      testele_->set_up_initial_rotations(rotrefe);
      testele_->set_up_reference_geometry(xrefe_setup);
    }

   protected:
    //! dummy discretization for holding element and node pointers
    std::shared_ptr<Core::FE::Discretization> testdis_;
    //! the beam3k element to be tested
    std::shared_ptr<Discret::Elements::Beam3k> testele_;
  };

  /**
   * Test reference length calculation of Kirchhoff-Love beam
   */
  TEST_F(Beam3k, RefLength)
  {
    EXPECT_NEAR(testele_->ref_length(), 0.61920435714496047, testTolerance);
  }

}  // namespace
