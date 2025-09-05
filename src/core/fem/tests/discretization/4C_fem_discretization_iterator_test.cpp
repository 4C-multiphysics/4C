// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_discretization_iterator.hpp"

#include <ranges>
#include <vector>


namespace
{
  using namespace FourC;
  // To test the DiscretizationIterator we do not require the actual Discretization.
  // Instead, we define a few dummy types here. The Iterator only requires a reference type
  // and a pointer that is passed to the reference type.
  using DummyDiscretizationType = std::vector<double>;

  struct Ref
  {
    using DiscretizationType = DummyDiscretizationType;

    Ref(DiscretizationType* discretization, int index)
        : discretization_(discretization), index_(index)
    {
    }

    [[nodiscard]] double& value() const { return (*discretization_)[index_]; }

   private:
    DiscretizationType* discretization_;
    int index_;
  };

  struct ConstRef
  {
    using DiscretizationType = const DummyDiscretizationType;

    ConstRef(DiscretizationType* discretization, int index)
        : discretization_(discretization), index_(index)
    {
    }

    [[nodiscard]] const double& value() const { return (*discretization_)[index_]; }

   private:
    DiscretizationType* discretization_;
    int index_;
  };


  auto make_range(DummyDiscretizationType& discretization, const std::vector<int>& indices)
  {
    return Core::FE::Internal::make_iterator_range<Ref>(&discretization, indices);
  }

  auto make_const_range(
      const DummyDiscretizationType& discretization, const std::vector<int>& indices)
  {
    return Core::FE::Internal::make_iterator_range<ConstRef>(&discretization, indices);
  }

  using Iterator = Core::FE::DiscretizationIterator<Ref>;
  using ConstIterator = Core::FE::DiscretizationIterator<ConstRef>;

  using Range = Core::FE::IteratorRange<Iterator>;
  using ConstRange = Core::FE::IteratorRange<ConstIterator>;

  static_assert(std::random_access_iterator<Iterator>);
  static_assert(std::random_access_iterator<ConstIterator>);

  static_assert(std::ranges::random_access_range<Range>);
  static_assert(std::ranges::random_access_range<ConstRange>);

  static_assert(std::ranges::sized_range<Range>);
  static_assert(std::ranges::sized_range<ConstRange>);

  static_assert(std::ranges::view<Range>);
  static_assert(std::ranges::view<ConstRange>);


  TEST(DiscretizationIteratorTest, RangeBasedLoop)
  {
    DummyDiscretizationType discretization{0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<int> indices{0, 2, 4};

    for (auto ref : make_range(discretization, indices))
    {
      ref.value() += 1.0;
    }
    EXPECT_EQ(discretization, (DummyDiscretizationType{1.0, 1.0, 3.0, 3.0, 5.0}));

    for (auto ref : make_const_range(discretization, indices))
    {
      // This should not compile since ref is ConstRef.
      // ref.value() += 1.0;

      EXPECT_TRUE(ref.value() >= 1.0);
    }
  }

  TEST(DiscretizationIteratorTest, ManualIteration)
  {
    DummyDiscretizationType discretization{0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<int> indices{0, 2, 4};

    auto range = make_range(discretization, indices);

    for (auto it = range.begin(); it != range.end(); ++it)
    {
      it->value() += 1.0;
    }
    EXPECT_EQ(discretization, (DummyDiscretizationType{1.0, 1.0, 3.0, 3.0, 5.0}));

    auto const_range = make_const_range(discretization, indices);
    for (auto it = const_range.begin(); it != const_range.end(); ++it)
    {
      // This should not compile since it is ConstRef.
      // it->value() += 1.0;

      EXPECT_TRUE(it->value() >= 1.0);
    }
  }

  TEST(DiscretizationIteratorTest, RandomAccess)
  {
    DummyDiscretizationType discretization{0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<int> indices{0, 2, 4};
    auto range = make_range(discretization, indices);
    EXPECT_EQ(range.size(), 3);
    EXPECT_EQ(range.begin()[0].value(), 0.0);
    EXPECT_EQ((range.begin() + 2)->value(), 4.0);
    EXPECT_EQ((range.end() - 1)->value(), 4.0);

    EXPECT_TRUE(range.begin() < range.end());
    EXPECT_FALSE(range.begin() > range.end());
    EXPECT_TRUE(range.begin() <= range.end());
    EXPECT_FALSE(range.begin() >= range.end());
    EXPECT_FALSE(range.begin() == range.end());
    EXPECT_TRUE(range.begin() != range.end());
  }


  TEST(DiscretizationIteratorTest, Ranges)
  {
    DummyDiscretizationType discretization{0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<int> indices{0, 1, 2, 3, 4};

    std::vector<int> out;
    std::ranges::copy(make_range(discretization, indices) |
                          std::views::filter([](Ref value) { return value.value() > 2.0; }) |
                          std::views::transform([](const Ref& ref) { return -ref.value(); }) |
                          std::views::reverse,
        std::back_inserter(out));

    EXPECT_EQ(out.size(), 2);
    EXPECT_EQ(out[0], -4.0);
    EXPECT_EQ(out[1], -3.0);
  }

}  // namespace
