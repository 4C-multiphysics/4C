// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io_gridgenerator.hpp"
#include "4C_io_input_parameter_container.templates.hpp"
#include "4C_io_pstream.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_singleton_owner.hpp"


namespace
{
  using namespace FourC;

  void create_material_in_global_problem()
  {
    Core::IO::InputParameterContainer mat_stvenant;
    mat_stvenant.add("YOUNG", 1.0);
    mat_stvenant.add("NUE", 0.1);
    mat_stvenant.add("DENS", 2.0);

    Global::Problem::instance()->materials()->insert(
        1, Mat::make_parameter(1, Core::Materials::MaterialType::m_stvenant, mat_stvenant));
  }

  // Serial discretization nodal method tests
  class BuildNodeCoordinatesTest : public testing::Test
  {
   public:
    BuildNodeCoordinatesTest()
    {
      create_material_in_global_problem();

      comm_ = MPI_COMM_WORLD;
      test_discretization_ = std::make_shared<Core::FE::Discretization>("dummy", comm_, 3);

      Core::IO::cout.setup(false, false, false, Core::IO::standard, comm_, 0, 0, "dummyFilePrefix");

      // results in 27 nodes
      inputData_.bottom_corner_point_ = std::array<double, 3>{0.0, 0.0, 0.0};
      inputData_.top_corner_point_ = std::array<double, 3>{1.0, 1.0, 1.0};
      inputData_.interval_ = std::array<int, 3>{2, 2, 2};
      inputData_.node_gid_of_first_new_node_ = 0;

      inputData_.elementtype_ = "SOLID";
      inputData_.cell_type = Core::FE::CellType::hex8;
      inputData_.element_arguments.add("MAT", 1);
      inputData_.element_arguments.add("KINEM", Inpar::Solid::KinemType::nonlinearTotLag);

      Core::IO::GridGenerator::create_rectangular_cuboid_discretization(
          *test_discretization_, inputData_, true);

      test_discretization_->fill_complete(false, false, false);
    }

    void TearDown() override { Core::IO::cout.close(); }

   protected:
    Core::IO::GridGenerator::RectangularCuboidInputs inputData_{};
    std::shared_ptr<Core::FE::Discretization> test_discretization_;
    MPI_Comm comm_;

    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;
  };

  TEST_F(BuildNodeCoordinatesTest, NodalCoordinatesDefault)
  {
    // build node coordinates based on the node row map of the whole discretization
    std::shared_ptr<Core::LinAlg::MultiVector<double>> nodal_test_coordinates =
        test_discretization_->build_node_coordinates();

    EXPECT_EQ(nodal_test_coordinates->MyLength(), test_discretization_->num_my_row_nodes());
    EXPECT_EQ(nodal_test_coordinates->NumVectors(), 3);

    std::array<double, 81> coords;
    nodal_test_coordinates->ExtractCopy(coords.data(), nodal_test_coordinates->MyLength());

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
      std::shared_ptr<Core::LinAlg::Map> node_row_map =
          std::make_shared<Core::LinAlg::Map>(-1, nodeList.size(), nodeList.data(), 0, comm_);
      std::shared_ptr<Core::LinAlg::MultiVector<double>> nodal_test_coordinates =
          test_discretization_->build_node_coordinates(node_row_map);

      EXPECT_EQ(nodal_test_coordinates->MyLength(), 4);
      EXPECT_EQ(nodal_test_coordinates->NumVectors(), 3);

      std::array<double, 12> coords;
      nodal_test_coordinates->ExtractCopy(coords.data(), nodal_test_coordinates->MyLength());

      // first coordinate
      EXPECT_NEAR(coords[0], 0.0, 1e-14);
      EXPECT_NEAR(coords[4], 0.0, 1e-14);
      EXPECT_NEAR(coords[8], 0.0, 1e-14);

      // last coordinate
      EXPECT_NEAR(coords[3], 0.0, 1e-14);
      EXPECT_NEAR(coords[7], 0.5, 1e-14);
      EXPECT_NEAR(coords[11], 0.0, 1e-14);
    }

    // build node coordinates based on the node row map of second partial discretization
    {
      std::array<int, 3> nodeList{50, 62, 114};  // random GIDs
      std::shared_ptr<Core::LinAlg::Map> node_row_map =
          std::make_shared<Core::LinAlg::Map>(-1, nodeList.size(), nodeList.data(), 0, comm_);
      std::shared_ptr<Core::LinAlg::MultiVector<double>> nodal_test_coordinates =
          test_discretization_->build_node_coordinates(node_row_map);

      EXPECT_EQ(nodal_test_coordinates->MyLength(), 3);
      EXPECT_EQ(nodal_test_coordinates->NumVectors(), 3);

      std::array<double, 9> coords;
      nodal_test_coordinates->ExtractCopy(coords.data(), nodal_test_coordinates->MyLength());

      EXPECT_NEAR(coords[0], 0.0, 1e-14);
      EXPECT_NEAR(coords[3], 0.0, 1e-14);
      EXPECT_NEAR(coords[6], 0.5, 1e-14);

      EXPECT_NEAR(coords[1], 0.5, 1e-14);
      EXPECT_NEAR(coords[4], 0.5, 1e-14);
      EXPECT_NEAR(coords[7], 0.5, 1e-14);

      EXPECT_NEAR(coords[2], 1.0, 1e-14);
      EXPECT_NEAR(coords[5], 0.5, 1e-14);
      EXPECT_NEAR(coords[8], 1.0, 1e-14);
    }
  }
}  // namespace
