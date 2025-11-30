// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_io_command_line_helpers.hpp"

#include "4C_unittest_utils_assertions_test.hpp"

#include <vector>

namespace
{
  using namespace FourC;

  TEST(BuildIoPairs, PrimaryOnly)
  {
    std::vector<std::string> extra = {};
    auto pairs = build_io_pairs(extra, std::filesystem::path("prim_in.4C.yaml"), "prim_out");
    ASSERT_EQ(pairs.size(), 1);
    EXPECT_EQ(pairs[0].first, std::filesystem::path("prim_in.4C.yaml"));
    EXPECT_EQ(pairs[0].second, "prim_out");
  }

  TEST(BuildIoPairs, WithAdditionalPairs)
  {
    std::vector<std::string> extra = {"inB.4C.yaml", "outB", "inC.4C.yaml", "outC"};
    auto pairs = build_io_pairs(extra, std::filesystem::path("prim_in.4C.yaml"), "prim_out");
    ASSERT_EQ(pairs.size(), 3);
    EXPECT_EQ(pairs[0].first, std::filesystem::path("prim_in.4C.yaml"));
    EXPECT_EQ(pairs[0].second, "prim_out");
    EXPECT_EQ(pairs[1].first, std::filesystem::path("inB.4C.yaml"));
    EXPECT_EQ(pairs[1].second, "outB");
    EXPECT_EQ(pairs[2].first, std::filesystem::path("inC.4C.yaml"));
    EXPECT_EQ(pairs[2].second, "outC");
  }

  TEST(ValidateArgumentCrossCompatibility, ValidArgumentsNoThrow)
  {
    CommandlineArguments args;
    args.n_groups = 2;
    args.nptype = Core::Communication::NestedParallelismType::separate_input_files;
    args.io_pairs = {{std::filesystem::path("a"), "A"}, {std::filesystem::path("b"), "B"}};
    // should not throw for valid configuration
    EXPECT_NO_THROW(validate_argument_cross_compatibility(args));
  }

  TEST(ValidateArgumentCrossCompatibility, ThrowsIfTooManyIoPairs)
  {
    CommandlineArguments args;
    args.group_layout = {2, 2};
    args.n_groups = 3;  // mismatch
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        validate_argument_cross_compatibility(args), Core::Exception, "When --glayout is provided");
  }

  TEST(ValidateArgumentCrossCompatibility, ThrowsIfNoNptypeWithMultipleGroups)
  {
    CommandlineArguments args;
    args.n_groups = 2;
    args.nptype = Core::Communication::NestedParallelismType::no_nested_parallelism;  // invalid
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(validate_argument_cross_compatibility(args), Core::Exception,
        "--ngroup > 1, a nested parallelism type must be specified");
  }

  TEST(ValidateArgumentCrossCompatibility, ThrowsIfTooManyIoPairsForNoNestedParallelism)
  {
    CommandlineArguments args;
    args.nptype = Core::Communication::NestedParallelismType::no_nested_parallelism;
    args.io_pairs = {{std::filesystem::path("a"), "A"}, {std::filesystem::path("b"), "B"}};
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(validate_argument_cross_compatibility(args), Core::Exception,
        "number of <input> <output> pairs must be exactly 1");
  }

  TEST(ValidateArgumentCrossCompatibility, ThrowsIfTooFewIoPairsForSeparateInputFiles)
  {
    CommandlineArguments args;
    args.n_groups = 2;
    args.nptype = Core::Communication::NestedParallelismType::separate_input_files;
    args.io_pairs = {{std::filesystem::path("a"), "A"}};
    // Not enough io pairs for number of groups
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(validate_argument_cross_compatibility(args), Core::Exception,
        "<input> <output> pairs must equal --ngroup 2.");
  }

  TEST(ValidateArgumentCrossCompatibility, ThrowsIfNoNestedParallelismWithTooManyRestarts)
  {
    CommandlineArguments args;
    args.n_groups = 1;
    args.nptype = Core::Communication::NestedParallelismType::no_nested_parallelism;
    args.io_pairs = {{std::filesystem::path("a"), "A"}};
    args.restart_per_group = {1, 2};  // only one restart for no nested parallelism
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(validate_argument_cross_compatibility(args), Core::Exception,
        "only one restart step and one");

    args.restart_per_group = {1};
    args.restart_identifier_per_group = {
        "a", "b"};  // only one restart identifier for no nested parallelism
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(validate_argument_cross_compatibility(args), Core::Exception,
        "only one restart step and one");
  }

  TEST(ValidateArgumentCrossCompatibility, ThrowsIfRestartFromWithoutRestart)
  {
    CommandlineArguments args;
    args.n_groups = 1;
    args.nptype = Core::Communication::NestedParallelismType::no_nested_parallelism;
    args.io_pairs = {{std::filesystem::path("a"), "A"}};
    args.restart_per_group = {};  // empty
    args.restart_identifier_per_group = {"prefix1"};
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(validate_argument_cross_compatibility(args), Core::Exception,
        "You need to specify a restart step");
  };

  TEST(AdaptLegacyCliArguments, ConvertsAndCombines)
  {
    std::vector<std::string> in = {"-ngroup=2", "-glayout=3,2", "-nptype=separateInputFiles",
        "inp1", "out1", "restart=1", "restartfrom=xxx", "inp2", "out2", "restart=2",
        "restartfrom=yyy"};


    LegacyCliOptions legacy_options = {.single_dash_legacy_names = {"ngroup", "glayout", "nptype"},
        .nodash_legacy_names = {"restart", "restartfrom"}};
    std::vector<std::string> out = adapt_legacy_cli_arguments(in, legacy_options);

    // Expected: converted --ngroup, combined --restart, combined --restartfrom
    std::vector<std::string> expect = {"--ngroup=2", "--glayout=3,2", "--nptype=separateInputFiles",
        "inp1", "out1", "inp2", "out2", "--restart=1,2", "--restartfrom=xxx,yyy"};
    EXPECT_EQ(out, expect);
  }

  std::vector<std::pair<std::filesystem::path, std::string>> single_io = {
      {std::filesystem::path("inputA.4C.yaml"), "outA"}};
  CommandlineArguments args_single = {
      .nptype = Core::Communication::NestedParallelismType::no_nested_parallelism,
      .restart_per_group = std::vector<int>{},  // empty
      .restart_identifier_per_group = std::vector<std::string>{},
      .io_pairs = single_io};

  TEST(UpdateIoIdentifiers, NoNestedParallelismNoRestart)
  {
    update_io_identifiers(args_single, 0);

    EXPECT_EQ(args_single.input_file_name, std::filesystem::path("inputA.4C.yaml"));
    EXPECT_EQ(args_single.output_file_identifier, "outA");
    EXPECT_EQ(args_single.restart, 0);
    EXPECT_EQ(args_single.restart_file_identifier, "outA");
  }

  TEST(UpdateIoIdentifiers, NoNestedParallelismWithRestart)
  {
    CommandlineArguments args = args_single;
    args.restart_per_group = std::vector<int>{5};
    args.restart_identifier_per_group = std::vector<std::string>{"restart_prefix"};

    update_io_identifiers(args, 0);

    EXPECT_EQ(args.input_file_name, std::filesystem::path("inputA.4C.yaml"));
    EXPECT_EQ(args.output_file_identifier, "outA");
    EXPECT_EQ(args.restart, 5);
    EXPECT_EQ(args.restart_file_identifier, "restart_prefix");
  }

  TEST(UpdateIoIdentifiers, EveryGroupReadInputFileNoRestart)
  {
    CommandlineArguments args = args_single;
    args.nptype = Core::Communication::NestedParallelismType::every_group_read_input_file;
    update_io_identifiers(args, 1);

    EXPECT_EQ(args.input_file_name, std::filesystem::path("inputA.4C.yaml"));
    EXPECT_EQ(args.output_file_identifier, "outA_group_1");
    EXPECT_EQ(args.restart, 0);
    EXPECT_EQ(args.restart_file_identifier, "outA_group_1");
  }

  TEST(UpdateIoIdentifiers, EveryGroupReadInputFileWithRestart)
  {
    CommandlineArguments args = args_single;
    args.nptype = Core::Communication::NestedParallelismType::every_group_read_input_file;
    args.restart_per_group = std::vector<int>{10};
    args.restart_identifier_per_group = std::vector<std::string>{"restart_prefix"};

    update_io_identifiers(args, 0);

    EXPECT_EQ(args.input_file_name, std::filesystem::path("inputA.4C.yaml"));
    EXPECT_EQ(args.output_file_identifier, "outA_group_0");
    EXPECT_EQ(args.restart, 10);
    EXPECT_EQ(args.restart_file_identifier, "restart_prefix_group_0");
  }

  TEST(UpdateIoIdentifiers, EveryGroupReadInputFileWithRestartAndNumber)
  {
    CommandlineArguments args = args_single;
    args.io_pairs[0].second = "outA-42";
    args.nptype = Core::Communication::NestedParallelismType::every_group_read_input_file;
    args.restart_per_group = std::vector<int>{10};
    args.restart_identifier_per_group = std::vector<std::string>{"restart_prefix-43"};

    update_io_identifiers(args, 0);

    EXPECT_EQ(args.input_file_name, std::filesystem::path("inputA.4C.yaml"));
    EXPECT_EQ(args.output_file_identifier, "outA_group_0_42");
    EXPECT_EQ(args.restart, 10);
    EXPECT_EQ(args.restart_file_identifier, "restart_prefix_group_0-43");
  }

  std::vector<std::pair<std::filesystem::path, std::string>> double_io = {
      {std::filesystem::path("inputA.4C.yaml"), "outA"},
      {std::filesystem::path("inputB.4C.yaml"), "outB"}};
  CommandlineArguments args_double = {
      .nptype = Core::Communication::NestedParallelismType::separate_input_files,
      .restart_per_group = std::vector<int>{},  // empty
      .restart_identifier_per_group = std::vector<std::string>{},
      .io_pairs = double_io};

  TEST(UpdateIoIdentifiers, SeparateInputFilesNoRestart)
  {
    CommandlineArguments args0 = args_double;
    update_io_identifiers(args0, 0);

    EXPECT_EQ(args0.input_file_name, std::filesystem::path("inputA.4C.yaml"));
    EXPECT_EQ(args0.output_file_identifier, "outA");
    EXPECT_EQ(args0.restart, 0);
    EXPECT_EQ(args0.restart_file_identifier, "outA");

    CommandlineArguments args1 = args_double;
    update_io_identifiers(args1, 1);

    EXPECT_EQ(args1.input_file_name, std::filesystem::path("inputB.4C.yaml"));
    EXPECT_EQ(args1.output_file_identifier, "outB");
    EXPECT_EQ(args1.restart, 0);
    EXPECT_EQ(args1.restart_file_identifier, "outB");
  }

  TEST(UpdateIoIdentifiers, SeparateInputFilesWithRestart)
  {
    CommandlineArguments args0 = args_double;
    args0.restart_per_group = std::vector<int>{3, 7};
    args0.restart_identifier_per_group = std::vector<std::string>{"restartA", "restartB"};

    update_io_identifiers(args0, 0);

    EXPECT_EQ(args0.input_file_name, std::filesystem::path("inputA.4C.yaml"));
    EXPECT_EQ(args0.output_file_identifier, "outA");
    EXPECT_EQ(args0.restart, 3);
    EXPECT_EQ(args0.restart_file_identifier, "restartA");

    CommandlineArguments args1 = args_double;
    args1.restart_per_group = std::vector<int>{3, 7};
    args1.restart_identifier_per_group = std::vector<std::string>{"restartA", "restartB"};

    update_io_identifiers(args1, 1);

    EXPECT_EQ(args1.input_file_name, std::filesystem::path("inputB.4C.yaml"));
    EXPECT_EQ(args1.output_file_identifier, "outB");
    EXPECT_EQ(args1.restart, 7);
    EXPECT_EQ(args1.restart_file_identifier, "restartB");
  }
}  // namespace