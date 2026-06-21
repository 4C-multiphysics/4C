// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_io_gridgenerator.hpp"
#include "4C_io_input_parameter_container.templates.hpp"
#include "4C_io_pstream.hpp"
#include "4C_unittest_utils_create_discretization_helper_test.hpp"


namespace
{
  using namespace FourC;

  // Serial discretization nodal method tests
  class BuildNodeCoordinatesTest : public testing::Test
  {
   public:
    BuildNodeCoordinatesTest()
    {
      comm_ = MPI_COMM_WORLD;
      test_discretization_ = std::make_shared<Core::FE::Discretization>("dummy", comm_, 3);

      TESTING::fill_discretization_hyper_cube(*test_discretization_, 2, comm_);

      Core::IO::cout.setup(false, false, false, Core::IO::standard, comm_, 0, 0, "dummyFilePrefix");

      test_discretization_->fill_complete(Core::FE::OptionsFillComplete::none());
    }

    void TearDown() override { Core::IO::cout.close(); }

   protected:
    std::shared_ptr<Core::FE::Discretization> test_discretization_;
    MPI_Comm comm_;
  };

  TEST_F(BuildNodeCoordinatesTest, NodalCoordinatesDefault)
  {
    // build node coordinates based on the node row map of the whole discretization
    std::shared_ptr<Core::LinAlg::MultiVector<double>> nodal_test_coordinates =
        extract_node_coordinates(*test_discretization_);

    EXPECT_EQ(nodal_test_coordinates->local_length(), test_discretization_->num_my_row_nodes());
    EXPECT_EQ(nodal_test_coordinates->num_vectors(), 3);

    std::array<double, 81> coords;
    nodal_test_coordinates->extract_copy(coords.data(), nodal_test_coordinates->local_length());

    // first coordinate
    EXPECT_NEAR(coords[0], 0.0, 1e-14);
    EXPECT_NEAR(coords[27], 0.0, 1e-14);
    EXPECT_NEAR(coords[54], 0.0, 1e-14);

    // last coordinate
    EXPECT_NEAR(coords[26], 1.0, 1e-14);
    EXPECT_NEAR(coords[53], 1.0, 1e-14);
    EXPECT_NEAR(coords[80], 1.0, 1e-14);
  }

  TEST_F(BuildNodeCoordinatesTest, NodalCoordinatesPartialMap)
  {
    // build node coordinates based on the node row map of first partial discretization
    {
      std::array<int, 4> nodeList{0, 2, 4, 10};  // GID list of first 4 elements
      Core::LinAlg::Map node_row_map(-1, std::span<const int>(nodeList), 0, comm_);
      std::shared_ptr<Core::LinAlg::MultiVector<double>> nodal_test_coordinates =
          extract_node_coordinates(*test_discretization_, node_row_map);

      EXPECT_EQ(nodal_test_coordinates->local_length(), 4);
      EXPECT_EQ(nodal_test_coordinates->num_vectors(), 3);

      std::array<double, 12> coords;
      nodal_test_coordinates->extract_copy(coords.data(), nodal_test_coordinates->local_length());

      // first coordinate
      EXPECT_DOUBLE_EQ(coords[0], 0.0);
      EXPECT_DOUBLE_EQ(coords[4], 0.0);
      EXPECT_DOUBLE_EQ(coords[8], 0.0);

      // last coordinate
      EXPECT_DOUBLE_EQ(coords[3], 0.5);
      EXPECT_DOUBLE_EQ(coords[7], 0.0);
      EXPECT_DOUBLE_EQ(coords[11], 0.5);
    }
  }
}  // namespace
