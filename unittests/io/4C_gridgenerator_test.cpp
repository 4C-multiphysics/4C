// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
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

  class GridGeneratorTest : public ::testing::Test
  {
   public:
    GridGeneratorTest()
    {
      inputData_.bottom_corner_point_ = std::array<double, 3>{-1.0, -2.0, -3.0};
      inputData_.top_corner_point_ = std::array<double, 3>{2.5, 3.5, 4.5};
      inputData_.interval_ = std::array<int, 3>{5, 10, 15};
      inputData_.node_gid_of_first_new_node_ = 17;
    };

   protected:
    void SetUp() override
    {
      create_material_in_global_problem();
      comm_ = MPI_COMM_WORLD;
      Core::IO::cout.setup(false, false, false, Core::IO::standard, comm_, 0, 0, "dummyFilePrefix");
      testdis_ = std::make_shared<Core::FE::Discretization>("dummy", comm_, 3);
    }

    void TearDown() override { Core::IO::cout.close(); }

   public:
    Core::IO::GridGenerator::RectangularCuboidInputs inputData_{};
    std::shared_ptr<Core::FE::Discretization> testdis_;
    MPI_Comm comm_;

    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;
  };

  TEST_F(GridGeneratorTest, TestGridGeneratorWithHex8Elements)
  {
    inputData_.elementtype_ = "SOLID";
    inputData_.cell_type = Core::FE::CellType::hex8;
    inputData_.element_arguments.add("MAT", 1);
    inputData_.element_arguments.add("KINEM", Inpar::Solid::KinemType::nonlinearTotLag);

    Core::IO::GridGenerator::create_rectangular_cuboid_discretization(*testdis_, inputData_, true);

    testdis_->fill_complete(false, false, false);

    Core::Nodes::Node* lastNode = testdis_->l_row_node(testdis_->num_my_row_nodes() - 1);
    const auto nodePosition = lastNode->x();

    EXPECT_NEAR(nodePosition[0], 2.5, 1e-14);
    EXPECT_NEAR(nodePosition[1], 3.5, 1e-14);
    EXPECT_NEAR(nodePosition[2], 4.5, 1e-14);
    EXPECT_EQ(testdis_->num_my_row_nodes(), 1056);
    EXPECT_EQ(testdis_->num_my_row_elements(), 750);
    EXPECT_EQ(lastNode->id(), 7177);
  }

  TEST_F(GridGeneratorTest, TestGridGeneratorWithRotatedHex8Elements)
  {
    inputData_.elementtype_ = "SOLID";
    inputData_.cell_type = Core::FE::CellType::hex8;
    inputData_.element_arguments.add("MAT", 1);
    inputData_.element_arguments.add("KINEM", Inpar::Solid::KinemType::nonlinearTotLag);
    inputData_.rotation_angle_ = std::array<double, 3>{30.0, 10.0, 7.0};

    Core::IO::GridGenerator::create_rectangular_cuboid_discretization(*testdis_, inputData_, true);

    testdis_->fill_complete(false, false, false);

    Core::Nodes::Node* lastNode = testdis_->l_row_node(testdis_->num_my_row_nodes() - 1);
    const auto nodePosition = lastNode->x();

    EXPECT_NEAR(nodePosition[0], 2.6565639116964181, 1e-14);
    EXPECT_NEAR(nodePosition[1], 4.8044393443812901, 1e-14);
    EXPECT_NEAR(nodePosition[2], 2.8980306453470042, 1e-14);
    EXPECT_EQ(testdis_->num_my_row_nodes(), 1056);
    EXPECT_EQ(testdis_->num_my_row_elements(), 750);
    EXPECT_EQ(lastNode->id(), 7177);
  }

  TEST_F(GridGeneratorTest, TestGridGeneratorWithHex27Elements)
  {
    inputData_.elementtype_ = "SOLID";
    inputData_.cell_type = Core::FE::CellType::hex27;
    inputData_.element_arguments.add("MAT", 1);
    inputData_.element_arguments.add("KINEM", Inpar::Solid::KinemType::nonlinearTotLag);

    Core::IO::GridGenerator::create_rectangular_cuboid_discretization(*testdis_, inputData_, true);

    testdis_->fill_complete(false, false, false);

    Core::Nodes::Node* lastNode = testdis_->l_row_node(testdis_->num_my_row_nodes() - 1);
    const auto nodePosition = lastNode->x();

    EXPECT_NEAR(nodePosition[0], 2.5, 1e-14);
    EXPECT_NEAR(nodePosition[1], 3.5, 1e-14);
    EXPECT_NEAR(nodePosition[2], 4.5, 1e-14);
    EXPECT_EQ(testdis_->num_my_row_nodes(), 7161);
    EXPECT_EQ(testdis_->num_my_row_elements(), 750);
    EXPECT_EQ(lastNode->id(), 7177);
  }

  TEST_F(GridGeneratorTest, TestGridGeneratorWithWedge6Elements)
  {
    inputData_.elementtype_ = "SOLID";
    inputData_.cell_type = Core::FE::CellType::wedge6;
    inputData_.element_arguments.add("MAT", 1);
    inputData_.element_arguments.add("KINEM", Inpar::Solid::KinemType::nonlinearTotLag);
    inputData_.autopartition_ = true;

    Core::IO::GridGenerator::create_rectangular_cuboid_discretization(*testdis_, inputData_, true);

    testdis_->fill_complete(false, false, false);

    Core::Nodes::Node* lastNode = testdis_->l_row_node(testdis_->num_my_row_nodes() - 1);
    const auto nodePosition = lastNode->x();

    EXPECT_NEAR(nodePosition[0], 2.5, 1e-14);
    EXPECT_NEAR(nodePosition[1], 3.5, 1e-14);
    EXPECT_NEAR(nodePosition[2], 4.5, 1e-14);
    EXPECT_EQ(testdis_->num_my_row_nodes(), 1056);
    EXPECT_EQ(testdis_->num_my_row_elements(), 1500);
    EXPECT_EQ(lastNode->id(), 7177);
  }

}  // namespace
