// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_io_linedefinition.hpp"

#include <sstream>

namespace
{
  using namespace FourC;


  TEST(LineDefinitionTest, AddReadGetTag)
  {
    std::istringstream input("OMEGA");
    auto line_definition = Input::LineDefinition::Builder().add_tag("OMEGA").build();
    auto container = line_definition.read(input);
    ASSERT_TRUE(container.has_value());
    EXPECT_TRUE(container->get<bool>("OMEGA"));
  }

  TEST(LineDefinitionTest, AddReadGetTagNonExistent)
  {
    std::istringstream input("");
    auto line_definition = Input::LineDefinition::Builder().add_tag("OMEGA").build();
    auto container = line_definition.read(input);
    EXPECT_FALSE(container.has_value());
  }

  TEST(LineDefinitionTest, AddReadGetOptionalTag)
  {
    std::istringstream input("OMEGA");
    auto line_definition = Input::LineDefinition::Builder().add_optional_tag("OMEGA").build();
    auto container = line_definition.read(input);
    ASSERT_TRUE(container.has_value());
    EXPECT_TRUE(container->get<bool>("OMEGA"));
  }

  TEST(LineDefinitionTest, AddReadGetOptionalTagNotExistent)
  {
    std::istringstream input("");
    auto line_definition = Input::LineDefinition::Builder().add_optional_tag("OMEGA").build();
    auto container = line_definition.read(input);
    ASSERT_TRUE(container.has_value());
    EXPECT_FALSE(container->get<bool>("OMEGA"));
  }

  TEST(LineDefinitionTest, ReadFalseWhenTagRequiredButIntGiven)
  {
    std::istringstream input("1");
    auto line_definition = Input::LineDefinition::Builder().add_tag("OMEGA").build();
    EXPECT_FALSE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadFalseWhenTagRequiredButDoubleGiven)
  {
    std::istringstream input("1.23");
    auto line_definition = Input::LineDefinition::Builder().add_tag("OMEGA").build();
    EXPECT_FALSE(line_definition.read(input));
  }

  // Named String
  TEST(LineDefinitionTest, add_named_string)
  {
    std::istringstream input("OMEGA TEST");
    auto line_definition = Input::LineDefinition::Builder().add_named_string("OMEGA").build();
    EXPECT_TRUE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadFalseWhenNamedStringRequiredButNothingGiven)
  {
    std::istringstream input("OMEGA ");
    auto line_definition = Input::LineDefinition::Builder().add_named_string("OMEGA").build();
    EXPECT_FALSE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadFalseWhenNamedStringRequiredButNoNameGiven)
  {
    std::istringstream input("TEST");
    auto line_definition = Input::LineDefinition::Builder().add_named_string("OMEGA").build();
    EXPECT_FALSE(line_definition.read(input));
  }

  // Named Int
  TEST(LineDefinitionTest, add_named_int)
  {
    std::istringstream input("OMEGA 1");
    auto line_definition = Input::LineDefinition::Builder().add_named_int("OMEGA").build();
    EXPECT_TRUE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadFalseWhenNamedIntRequiredButNothingGiven)
  {
    std::istringstream input("OMEGA");
    auto line_definition = Input::LineDefinition::Builder().add_named_int("OMEGA").build();
    EXPECT_FALSE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadFalseWhenNamedIntRequiredButDoubleGiven)
  {
    std::istringstream input("OMEGA 1.23");
    auto line_definition = Input::LineDefinition::Builder().add_named_int("OMEGA").build();
    EXPECT_FALSE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadFalseWhenNamedIntRequiredButNoNameGiven)
  {
    std::istringstream input("1");
    auto line_definition = Input::LineDefinition::Builder().add_named_int("OMEGA").build();
    EXPECT_FALSE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, add_named_string_vector)
  {
    std::istringstream input("OMEGA TEST1 TEST2 TEST3 TEST4 TEST5");
    auto line_definition =
        Input::LineDefinition::Builder().add_named_string_vector("OMEGA", 5).build();
    EXPECT_TRUE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadFalseWhenNamedStringVectorRequiredButTooFewVectorEntriesGiven)
  {
    std::istringstream input("OMEGA TEST1 TEST2 TEST3 TEST4");
    auto line_definition =
        Input::LineDefinition::Builder().add_named_string_vector("OMEGA", 5).build();
    EXPECT_FALSE(line_definition.read(input));
  }

  // Named Int Vector
  TEST(LineDefinitionTest, AddNamedIntVector)
  {
    std::istringstream input("OMEGA 1 2 3");
    auto line_definition =
        Input::LineDefinition::Builder().add_named_int_vector("OMEGA", 3).build();
    EXPECT_TRUE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadFalseWhenNamedIntVectorRequiredButTooFewVectorEntriesGiven)
  {
    std::istringstream input("OMEGA 1 2");
    auto line_definition =
        Input::LineDefinition::Builder().add_named_int_vector("OMEGA", 3).build();
    EXPECT_FALSE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadFalseWhenNamedIntVectorRequiredButTooManyVectorEntriesGiven)
  {
    std::istringstream input("OMEGA 1 2 3 4");
    auto line_definition =
        Input::LineDefinition::Builder().add_named_int_vector("OMEGA", 3).build();
    EXPECT_FALSE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadFalseWhenNamedIntVectorRequiredButDoubleVectorEntriesGiven)
  {
    std::istringstream input("OMEGA 1.23 2.34 3.45");
    auto line_definition =
        Input::LineDefinition::Builder().add_named_int_vector("OMEGA", 3).build();
    EXPECT_FALSE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadFalseWhenNamedIntVectorRequiredButNoNameGiven)
  {
    std::istringstream input("1.23 2.34 3.45");
    auto line_definition =
        Input::LineDefinition::Builder().add_named_int_vector("OMEGA", 3).build();
    EXPECT_FALSE(line_definition.read(input));
  }

  // Named Double
  TEST(LineDefinitionTest, add_named_double)
  {
    std::istringstream input("OMEGA 1.23");
    auto line_definition = Input::LineDefinition::Builder().add_named_double("OMEGA").build();
    EXPECT_TRUE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadFalseWhenNamedDoubleRequiredButNothingGiven)
  {
    std::istringstream input("OMEGA");
    auto line_definition = Input::LineDefinition::Builder().add_named_double("OMEGA").build();
    EXPECT_FALSE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadFalseWhenNamedDoubleConcatenatedWithSomething)
  {
    std::istringstream input("OMEGA 123.45*893");
    auto line_definition = Input::LineDefinition::Builder().add_named_double("OMEGA").build();
    EXPECT_FALSE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadFalseWhenNamedDoubleRequiredButNoNameGiven)
  {
    std::istringstream input("1.23");
    auto line_definition = Input::LineDefinition::Builder().add_named_double("OMEGA").build();
    EXPECT_FALSE(line_definition.read(input));
  }

  // Named Double Vector
  TEST(LineDefinitionTest, add_named_double_vector)
  {
    std::istringstream input("OMEGA 1.23 2.34 3.45");
    auto line_definition =
        Input::LineDefinition::Builder().add_named_double_vector("OMEGA", 3).build();
    EXPECT_TRUE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadFalseWhenNamedDoubleVectorRequiredButTooFewVectorEntriesGiven)
  {
    std::istringstream input("OMEGA 1.23 2.34");
    auto line_definition =
        Input::LineDefinition::Builder().add_named_double_vector("OMEGA", 3).build();
    EXPECT_FALSE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadFalseWhenNamedDoubleVectorRequiredButTooManyVectorEntriesGiven)
  {
    std::istringstream input("OMEGA 1.23 2.34 3.45 4.56");
    auto line_definition =
        Input::LineDefinition::Builder().add_named_double_vector("OMEGA", 3).build();
    EXPECT_FALSE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadFalseWhenNamedDoubleVectorRequiredButNoNameGiven)
  {
    std::istringstream input("1.23 2.34 3.45");
    auto line_definition =
        Input::LineDefinition::Builder().add_named_double_vector("OMEGA", 3).build();
    EXPECT_FALSE(line_definition.read(input));
  }

  // Too much in stream except from comments and whitespaces
  TEST(LineDefinitionTest, ReadFalseWhenStreamHasTooMuch)
  {
    std::istringstream input("OMEGA More data that is not in lineDefinition.");
    auto line_definition = Input::LineDefinition::Builder().add_tag("OMEGA").build();
    EXPECT_FALSE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadTrueWhenStreamHasTooMuchButOnlyComments)
  {
    std::istringstream input("OMEGA // Comment");
    auto line_definition = Input::LineDefinition::Builder().add_tag("OMEGA").build();
    EXPECT_TRUE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, ReadTrue_When_StreamHasTooMuchButOnlyWhitespaces)
  {
    std::istringstream input("OMEGA           ");
    auto line_definition = Input::LineDefinition::Builder().add_tag("OMEGA").build();
    EXPECT_TRUE(line_definition.read(input));
  }

  // Empty stream
  TEST(LineDefinitionTest, ReadTrueWhenEmptyStreamIntoEmptyDefinition)
  {
    std::istringstream input("");
    Input::LineDefinition line_definition;
    EXPECT_TRUE(line_definition.read(input));
  }

  TEST(LineDefinitionTest, PathEmpty)
  {
    std::istringstream input("PATH");
    auto line_definition = Input::LineDefinition::Builder().add_named_path("PATH").build();
    auto container = line_definition.read(input);
    EXPECT_FALSE(container.has_value());
  }

  TEST(LineDefinitionTest, PathNoContext)
  {
    std::istringstream input("PATH file.txt");
    auto line_definition = Input::LineDefinition::Builder().add_named_path("PATH").build();
    auto container = line_definition.read(input);
    ASSERT_TRUE(container.has_value());
    EXPECT_EQ(std::filesystem::path("file.txt"), container->get<std::filesystem::path>("PATH"));
  }

  TEST(LineDefinitionTest, PathAbsolute)
  {
    std::istringstream input("PATH /home/my/file.txt");
    auto line_definition = Input::LineDefinition::Builder().add_named_path("PATH").build();
    auto container = line_definition.read(
        input, Input::LineDefinition::ReadContext{.input_file = {"/data/dummy/input.txt"}});
    ASSERT_TRUE(container.has_value());
    EXPECT_EQ(
        std::filesystem::path("/home/my/file.txt"), container->get<std::filesystem::path>("PATH"));
  }

  TEST(LineDefinitionTest, PathRelativeWithContext)
  {
    std::istringstream input("PATH my/file.txt");
    auto line_definition = Input::LineDefinition::Builder().add_named_path("PATH").build();
    auto container = line_definition.read(
        input, Input::LineDefinition::ReadContext{.input_file = {"/data/dummy/input.txt"}});
    ASSERT_TRUE(container.has_value());
    EXPECT_EQ(std::filesystem::path("/data/dummy/my/file.txt"),
        container->get<std::filesystem::path>("PATH"));
  }

  TEST(LineDefinitionTest, PathRelativeWithSingleFileNameContext)
  {
    std::istringstream input("PATH my/file.txt");
    auto line_definition = Input::LineDefinition::Builder().add_named_path("PATH").build();
    auto container = line_definition.read(
        input, Input::LineDefinition::ReadContext{.input_file = {"input.txt"}});
    ASSERT_TRUE(container.has_value());
    EXPECT_EQ(std::filesystem::path("my/file.txt"), container->get<std::filesystem::path>("PATH"));
  }

  TEST(LineDefinitionPrinting, Empty)
  {
    std::ostringstream out;
    Input::LineDefinition().print(out);
    EXPECT_EQ(out.str(), "");
  }

  TEST(LineDefinitionPrinting, ComplicatedLine)
  {
    std::ostringstream out;
    Input::LineDefinition::Builder()
        .add_tag("abc")
        .add_named_double("d")
        .add_named_int_vector("iv", 3)
        .add_optional_named_string_vector("s", 2)
        .add_optional_named_pair_of_string_and_double_vector(
            "pairs", Input::LengthFromIntNamed("i"))
        .build()
        .print(out);

    EXPECT_EQ(out.str(),
        "abc d <double> iv <vector<int>> [s <vector<string>>] [pairs <vector<pair<string, "
        "double>>>] ");
  }
}  // namespace
