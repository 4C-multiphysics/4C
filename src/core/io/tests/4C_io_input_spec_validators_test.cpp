// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_io_input_spec_validators.hpp"

#include <vector>


namespace
{
  using namespace FourC::Core::IO::InputSpecBuilders::Validators;
  TEST(InputSpecValidators, InRangeInt)
  {
    const auto validator = in_range(incl(0), excl(2));
    EXPECT_TRUE(validator(0));
    EXPECT_TRUE(validator(1));
    EXPECT_FALSE(validator(-1));
    EXPECT_FALSE(validator(2));
  }

  TEST(InputSpecValidators, InRangeDouble)
  {
    const auto validator = in_range(excl(0.), 1.);
    EXPECT_FALSE(validator(0.0));
    EXPECT_TRUE(validator(1.0));
    EXPECT_FALSE(validator(-0.1));
    EXPECT_FALSE(validator(1.1));

    std::stringstream ss;
    validator.describe(ss);
    EXPECT_EQ(ss.str(), "in_range(0,1]");
  }

  TEST(InputSpecValidators, PositiveInt)
  {
    const auto validator = positive<int>();
    EXPECT_TRUE(validator(1));
    EXPECT_FALSE(validator(0));
    EXPECT_FALSE(validator(-1));
    EXPECT_FALSE(std::numeric_limits<int>::infinity());
  }


  TEST(InputSpecValidators, EnumSet)
  {
    enum class MyEnum
    {
      A,
      B,
      C
    };
    const auto validator = in_set({MyEnum::A, MyEnum::B});
    EXPECT_TRUE(validator(MyEnum::A));
    EXPECT_TRUE(validator(MyEnum::B));
    EXPECT_FALSE(validator(MyEnum::C));

    std::stringstream ss;
    validator.describe(ss);
    EXPECT_EQ(ss.str(), "in_set{A,B}");
  }

  TEST(InputSpecValidators, AllElements)
  {
    const auto validator = all_elements(in_range(1, 4));
    EXPECT_TRUE(validator(std::vector<int>{1, 2, 3}));
    EXPECT_FALSE(validator(std::vector<int>{1, -2, 3}));
    EXPECT_FALSE(validator(std::array{1, -2, 3}));

    std::stringstream ss;
    validator.describe(ss);
    EXPECT_EQ(ss.str(), "all_elements{in_range[1,4]}");
  }

  TEST(InputSpecValidators, NestedAllElements)
  {
    const auto validator = all_elements(null_or(all_elements(positive<int>())));
    std::vector<std::optional<std::vector<int>>> good{
        std::vector<int>{1, 2, 3}, std::vector<int>{4, 5}, std::nullopt, std::vector<int>{10}};
    EXPECT_TRUE(validator(good));

    std::array<std::optional<std::vector<int>>, 3> bad{
        std::vector<int>{1, 2, 3}, std::vector<int>{4, -5}, std::nullopt};
    EXPECT_FALSE(validator(bad));
  }

  TEST(InputSpecValidators, Pattern)
  {
    using namespace std::string_literals;
    const auto validator = pattern(R"(\d-\d-\d)");
    EXPECT_TRUE(validator("1-2-3"s));
    // This works since the regex performs a search and not a full match
    EXPECT_TRUE(validator("1-2-3-4"s));

    EXPECT_FALSE(validator("1-a-2"s));
  }

  TEST(InputSpecValidators, CopyCtor)
  {
    auto v1 = positive<int>();
    auto v2 = v1;

    EXPECT_TRUE(v2(2));
    EXPECT_FALSE(v2(0));
  }

  TEST(InputSpecValidators, MoveCtor)
  {
    auto v1 = positive<int>();
    auto v2 = std::move(v1);

    EXPECT_TRUE(v2(2));
    EXPECT_FALSE(v2(0));
  }

  TEST(InputSpecValidators, CopyAssign)
  {
    auto v1 = positive<int>();
    auto v2 = in_range(-3, -1);
    v2 = v1;

    EXPECT_TRUE(v2(2));
    EXPECT_FALSE(v2(0));
  }

  TEST(InputSpecValidators, MoveAssign)
  {
    auto v1 = positive<int>();
    auto v2 = in_range(-3, -1);
    v2 = std::move(v1);

    EXPECT_TRUE(v2(2));
    EXPECT_FALSE(v2(0));
  }

}  // namespace
