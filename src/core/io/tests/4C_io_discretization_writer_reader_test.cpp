// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_builder.hpp"
#include "4C_fem_general_shape_function_type.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_multi_vector.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_rebalance.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <ostream>
#include <span>
#include <string>
#include <vector>

namespace
{
  using namespace FourC;
  namespace fs = std::filesystem;
  class DiscretizationWriterReaderTest : public ::testing::Test
  {
   protected:
    void SetUp() override
    {
      // create test directory
      fs::create_directories(test_directory);

      // setup test discretization
      set_up_two_stacked_hex8_discretization();

      // create output control utilities
      output_control_ = std::make_shared<Core::IO::OutputControl>(comm_, "Structure",
          Core::FE::ShapeFunctionType::polynomial, "dummy.4C.yaml", filename_ + "-restart",
          filename_, 3, 1, 1, true);
    }

    void TearDown() override { fs::remove_all(test_directory); }

    /// setup a discretization consisting of two HEX8 elements stacked onto each other
    void set_up_two_stacked_hex8_discretization()
    {
      discretization_ =
          std::make_shared<Core::FE::Discretization>("test-discretization", MPI_COMM_WORLD, 3);
      Core::FE::DiscretizationBuilder<3> builder(discretization_->get_comm());

      const std::vector<std::array<double, 3>> coords{{0.0, 0.0, 0.0},  // 0
          {1.0, 0.0, 0.0},                                              // 1
          {2.0, 0.0, 0.0},                                              // 2
          {2.0, 1.0, 0.0},                                              // 3
          {1.0, 1.0, 0.0},                                              // 4
          {0.0, 1.0, 0.0},                                              // 5
          {0.0, 0.0, 1.0},                                              // 6
          {1.0, 0.0, 1.0},                                              // 7
          {2.0, 0.0, 1.0},                                              // 8
          {2.0, 1.0, 1.0},                                              // 9
          {1.0, 1.0, 1.0},                                              // 10
          {0.0, 1.0, 1.0}};                                             // 11

      int counter = 0;
      for (const auto& coord : coords) builder.add_node(coord, counter++, nullptr);

      // Add unit hex8 element
      {
        const int ele_id = 0;
        std::array<int, 8> node_ids{0, 1, 4, 5, 6, 7, 10, 11};

        builder.add_element(Core::FE::CellType::hex8, node_ids, ele_id,
            {.num_dof_per_node = 1, .num_dof_per_element = 0});
      }

      // Add unit hex8 element
      {
        const int ele_id = 1;
        std::array<int, 8> node_ids{1, 2, 3, 4, 7, 8, 9, 10};

        builder.add_element(Core::FE::CellType::hex8, node_ids, ele_id,
            {.num_dof_per_node = 1, .num_dof_per_element = 0});
      }

      Core::Rebalance::RebalanceParameters rebalance_parameters;
      builder.build(*discretization_, rebalance_parameters);
      discretization_->fill_complete(Core::FE::OptionsFillComplete::none());
      discretization_->assign_degrees_of_freedom(0);
    }

    /// communicator
    const MPI_Comm comm_ = MPI_COMM_WORLD;
    /// test discretization
    std::shared_ptr<Core::FE::Discretization> discretization_;
    /// test directory to store the output into
    const fs::path test_directory =
        fs::path(FOUR_C_IO_TEST_TMP_DIR) / "io_discretization_writer_reader";
    /// control file name (without suffixes)
    const std::string filename_ = (test_directory / "test-discretization-writing-reading").string();
    /// output control containing control file and write logic
    std::shared_ptr<Core::IO::OutputControl> output_control_;
  };

  /// Tests whether the discretization writer only writes items with unique names, i.e., we should
  /// not be able to write items with the same name twice even if their data types are different.
  /// For each writer function we test re-writing using both the same writer function and a
  /// different function (for a different data type). The test also verifies at the end that none of
  /// the written variable names are still cached after the discretization writer receives a new
  /// step.
  TEST_F(DiscretizationWriterReaderTest, SafeWritingAndNewStepCacheReset)
  {
    auto discretization_writer = Core::IO::DiscretizationWriter(
        *discretization_, *output_control_, Core::FE::ShapeFunctionType::polynomial);
    discretization_writer.write_mesh(0, 0.0);
    discretization_writer.new_step(0, 0.0);


    // writing variable of type int
    EXPECT_FALSE(discretization_writer.is_written("dummy_int"));
    discretization_writer.write_int("dummy_int", 1);
    EXPECT_TRUE(discretization_writer.is_written("dummy_int"));
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(discretization_writer.write_int("dummy_int", 1),
        Core::Exception, "already written; use unique item names");  // cannot write this variable
                                                                     // again using the same type
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(discretization_writer.write_double("dummy_int", 1.0),
        Core::Exception, "already written; use unique item names");  // cannot write this variable
                                                                     // again using a different type

    // writing variable of type double
    EXPECT_FALSE(discretization_writer.is_written("dummy_double"));
    discretization_writer.write_double("dummy_double", 1.0);
    EXPECT_TRUE(discretization_writer.is_written("dummy_double"));
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(discretization_writer.write_double("dummy_double", 1.0),
        Core::Exception, "already written; use unique item names");  // cannot write this variable
                                                                     // again using the same type
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(discretization_writer.write_int("dummy_int", 1),
        Core::Exception, "already written; use unique item names");  // cannot write this variable
    // again using a different type


    // writing variable of type Core::LinAlg::Vector<double>
    const auto test_vector =
        std::make_shared<Core::LinAlg::Vector<double>>(*discretization_->node_row_map());
    EXPECT_FALSE(discretization_writer.is_written("dummy_vector"));
    discretization_writer.write_vector("dummy_vector", test_vector);
    EXPECT_TRUE(discretization_writer.is_written("dummy_vector"));
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        discretization_writer.write_vector("dummy_vector", test_vector), Core::Exception,
        "already written; use unique item names");  // cannot write this variable
                                                    // again using the same type
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        discretization_writer.write_multi_vector("dummy_vector", *test_vector), Core::Exception,
        "already written; use unique item names");  // cannot write this variable
                                                    // again using a different type


    // writing variable of type Core::LinAlg::MultiVector<double>
    const auto test_multivector =
        std::make_shared<Core::LinAlg::MultiVector<double>>(*discretization_->node_row_map(), 3);
    EXPECT_FALSE(discretization_writer.is_written("dummy_multivector"));
    discretization_writer.write_multi_vector("dummy_multivector", *test_multivector);
    EXPECT_TRUE(discretization_writer.is_written("dummy_multivector"));
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        discretization_writer.write_multi_vector("dummy_multivector", *test_multivector),
        Core::Exception, "already written; use unique item names");  // cannot write this variable
                                                                     // again using same type
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        discretization_writer.write_vector("dummy_multivector", test_vector), Core::Exception,
        "already written; use unique item names");  // cannot write this variable again as vector
                                                    // now


    // writing variable of type vector<char>
    auto test_char_vector = std::vector<char>(1, 'T');
    EXPECT_FALSE(discretization_writer.is_written("dummy_char_vector"));
    discretization_writer.write_vector(
        "dummy_char_vector", test_char_vector, *discretization_->element_row_map());
    EXPECT_TRUE(discretization_writer.is_written("dummy_char_vector"));
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(discretization_writer.write_vector("dummy_char_vector",
                                         test_char_vector, *discretization_->element_row_map()),
        Core::Exception,
        "already written; use unique item names");  // cannot write this variable
                                                    // again using the same type
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        discretization_writer.write_char_data("dummy_char_vector", test_char_vector),
        Core::Exception,
        "already written; use unique item names");  // writing using write_char_data also not
                                                    // allowed
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        discretization_writer.write_vector("dummy_char_vector", test_vector), Core::Exception,
        "already written; use unique item names");  // cannot write this variable
                                                    // again using a different type


    // writing variable of type vector<int>
    const auto test_int_vec = std::vector<int>(1, 0);
    EXPECT_FALSE(discretization_writer.is_written("dummy_int_vec"));
    discretization_writer.write_int_vector_on_first_rank("dummy_int_vec", test_int_vec);
    EXPECT_TRUE(discretization_writer.is_written("dummy_int_vec"));
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        discretization_writer.write_int_vector_on_first_rank("dummy_int_vec", test_int_vec),
        Core::Exception,
        "already written; use unique item names");  // cannot write this variable
                                                    // again using the same type
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        discretization_writer.write_multi_vector("dummy_int_vec", *test_multivector),
        Core::Exception,
        "already written; use unique item names");  // cannot write this variable
                                                    // again using a different type


    // writing variable of type vector<double>
    const auto test_double_vec = std::vector<double>(1, 0.0);
    EXPECT_FALSE(discretization_writer.is_written("dummy_double_vec"));
    discretization_writer.write_double_vector_on_first_rank("dummy_double_vec", test_double_vec);
    EXPECT_TRUE(discretization_writer.is_written("dummy_double_vec"));
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(discretization_writer.write_double_vector_on_first_rank(
                                         "dummy_double_vec", test_double_vec),
        Core::Exception,
        "already written; use unique item names");  // cannot write this variable
                                                    // again using the same type
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        discretization_writer.write_multi_vector("dummy_double_vec", *test_multivector),
        Core::Exception,
        "already written; use unique item names");  // cannot write this variable
                                                    // again using a different type


    // test rewrite in the same timestep -> should not work because the name is cached
    discretization_writer.new_step(0, 0.0);
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(discretization_writer.write_int("dummy_int", 1),
        Core::Exception, "already written; use unique item names");
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(discretization_writer.write_double("dummy_double", 1.0),
        Core::Exception, "already written; use unique item names");
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        discretization_writer.write_vector("dummy_vector", test_vector), Core::Exception,
        "already written; use unique item names");
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        discretization_writer.write_multi_vector("dummy_multivector", *test_multivector),
        Core::Exception, "already written; use unique item names");
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(discretization_writer.write_vector("dummy_char_vector",
                                         test_char_vector, *discretization_->element_row_map()),
        Core::Exception, "already written; use unique item names");
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        discretization_writer.write_int_vector_on_first_rank("dummy_int_vec", test_int_vec),
        Core::Exception, "already written; use unique item names");
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(discretization_writer.write_double_vector_on_first_rank(
                                         "dummy_double_vec", test_double_vec),
        Core::Exception, "already written; use unique item names");

    // test that the cached names have been removed for a new step
    discretization_writer.new_step(1, 1.0);
    EXPECT_FALSE(discretization_writer.is_written("dummy_int"));
    EXPECT_FALSE(discretization_writer.is_written("dummy_double"));
    EXPECT_FALSE(discretization_writer.is_written("dummy_vector"));
    EXPECT_FALSE(discretization_writer.is_written("dummy_multivector"));
    EXPECT_FALSE(discretization_writer.is_written("dummy_char_vector"));
    EXPECT_FALSE(discretization_writer.is_written("dummy_int_vec"));
    EXPECT_FALSE(discretization_writer.is_written("dummy_double_vec"));
    discretization_writer.write_int("dummy_int", 1);
    EXPECT_TRUE(discretization_writer.is_written("dummy_int"));
  }

  /// Tests whether specific vectors or multivectors are available to be read-in using the
  /// DiscretizationReader.
  TEST_F(DiscretizationWriterReaderTest, QueryAndReadVectorOrMultiVector)
  {
    // setup control file
    auto discretization_writer = Core::IO::DiscretizationWriter(
        *discretization_, *output_control_, Core::FE::ShapeFunctionType::polynomial);
    discretization_writer.write_mesh(1, 1.0);
    discretization_writer.new_step(1, 1.0);
    const auto test_vector =
        std::make_shared<Core::LinAlg::Vector<double>>(*discretization_->node_row_map());
    test_vector->local_values_as_span()[0] = 1.0;
    discretization_writer.write_vector("dummy_vector", test_vector);
    const auto test_multivector =
        std::make_shared<Core::LinAlg::MultiVector<double>>(*discretization_->node_row_map(), 3);
    test_multivector->get_vector(0).local_values_as_span()[0] = 1.0;
    test_multivector->get_vector(1).local_values_as_span()[0] = 1.0;
    test_multivector->get_vector(2).local_values_as_span()[0] = 1.0;
    discretization_writer.write_multi_vector("dummy_multivector", *test_multivector);

    // test querying and reading of Core::LinAlg::Vector and Core::LinAlg::MultiVector with correct
    // and wrong names, and with correct and wrong number of columns
    auto input_control = std::make_shared<Core::IO::InputControl>(filename_);
    auto discretization_reader = Core::IO::DiscretizationReader(*discretization_, input_control, 1);

    EXPECT_TRUE(discretization_reader.has_vector("dummy_vector"));
    EXPECT_FALSE(discretization_reader.has_vector("dummy_vector_non_existing"));
    EXPECT_TRUE(discretization_reader.has_vector("dummy_vector", 1));
    EXPECT_FALSE(discretization_reader.has_vector("dummy_vector", 2));
    const auto read_in_vector =
        std::make_shared<Core::LinAlg::Vector<double>>(*discretization_->node_row_map());
    discretization_reader.read_vector(read_in_vector, "dummy_vector");
    Core::LinAlg::Vector<double> vector_difference{test_vector->get_map()};
    vector_difference.update(1.0, *test_vector, -1.0, *read_in_vector, 0.0);
    double norm_vector_difference[1] = {1.0e10};
    vector_difference.norm_2(norm_vector_difference);
    EXPECT_LE(norm_vector_difference[0], 1.0e-15);

    EXPECT_TRUE(discretization_reader.has_vector("dummy_multivector"));
    EXPECT_TRUE(discretization_reader.has_vector("dummy_multivector", 3));
    EXPECT_FALSE(discretization_reader.has_vector("dummy_multivector", 1));
    const auto read_in_multivector = discretization_reader.read_multi_vector("dummy_multivector");
    Core::LinAlg::MultiVector<double> multivector_difference{read_in_multivector->get_map(), 3};
    multivector_difference.update(1.0, *test_multivector, -1.0, *read_in_multivector, 0.0);
    double norm_multivector_difference[3] = {1.0e10, 1.0e10, 1.0e10};
    multivector_difference.norm_2(norm_multivector_difference);
    for (double norm_item : norm_multivector_difference)
    {
      EXPECT_LE(norm_item, 1.0e-15);
    }
  }

}  // namespace
