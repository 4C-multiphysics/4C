// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include <gmock/gmock.h>

#include "4C_mixture_full_constrained_mixture_fiber_adaptive_history.hpp"

#include "4C_mixture_growth_evolution_linear_cauchy_poisson_turnover.hpp"

#include <Sacado.hpp>

#include <bitset>
#include <memory>

namespace
{
  using namespace FourC;

  TEST(TimestepAdaptivityTest, EmplaceBack)
  {
    Mixture::TimestepAdaptivityInfo level{};

    level.emplace_back(3, 5);
    EXPECT_EQ(level.get_number_of_levels(), 1);
    EXPECT_EQ(level.get_total_number_of_simpson_intervals(), 5);

    level.emplace_back(2, 3);
    EXPECT_EQ(level.get_number_of_levels(), 2);
    EXPECT_EQ(level.get_total_number_of_simpson_intervals(), 8);

    level.emplace_back(2, 6);
    EXPECT_EQ(level.get_number_of_levels(), 2);
    EXPECT_EQ(level.get_total_number_of_simpson_intervals(), 14);
  }

  TEST(TimestepAdaptivityTest, EmplaceBackInWrongOrder)
  {
#ifndef FOUR_C_ENABLE_ASSERTIONS
    GTEST_SKIP() << "Skip debug assertion tests in release mode.";
#endif
    Mixture::TimestepAdaptivityInfo level{};

    level.emplace_back(1, 5);

    EXPECT_ANY_THROW(level.emplace_back(2, 1));
  }

  TEST(TimestepAdaptivityTest, SplitLevel)
  {
    Mixture::TimestepAdaptivityInfo level{};
    level.emplace_back(3, 6);
    level.emplace_back(2, 12);
    level.emplace_back(1, 8);

    level.split_level(3, 2);
    EXPECT_EQ(level[0], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(4, 2));
    EXPECT_EQ(level[1], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(3, 2));
    EXPECT_EQ(level[2], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(2, 12));
    EXPECT_EQ(level[3], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(1, 8));
    EXPECT_EQ(level.get_number_of_levels(), 4);

    level.split_level(4, 1);
    EXPECT_EQ(level[0], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(5, 1));
    EXPECT_EQ(level[1], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(3, 2));
    EXPECT_EQ(level[2], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(2, 12));
    EXPECT_EQ(level[3], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(1, 8));
    EXPECT_EQ(level.get_number_of_levels(), 4);

    level.split_level(1, 1);
    EXPECT_EQ(level[0], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(5, 1));
    EXPECT_EQ(level[1], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(3, 2));
    EXPECT_EQ(level[2], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(2, 13));
    EXPECT_EQ(level[3], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(1, 6));
    EXPECT_EQ(level.get_number_of_levels(), 4);

    level.split_level(1, 3);
    EXPECT_EQ(level[0], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(5, 1));
    EXPECT_EQ(level[1], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(3, 2));
    EXPECT_EQ(level[2], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(2, 16));
    EXPECT_EQ(level.get_number_of_levels(), 3);

    EXPECT_ANY_THROW(level.split_level(1, 0));
  }

  TEST(TimestepAdaptivityTest, get_total_number_of_simpson_intervals)
  {
    Mixture::TimestepAdaptivityInfo level{};
    level.emplace_back(3, 2);
    level.emplace_back(2, 1);

    EXPECT_EQ(level.get_total_number_of_simpson_intervals(), 3);
  }

  TEST(TimestepAdaptivityTest, max_level)
  {
    Mixture::TimestepAdaptivityInfo level{};
    level.emplace_back(3, 2);
    level.emplace_back(2, 1);

    EXPECT_EQ(level.max_level(), 3);
  }

  TEST(TimestepAdaptivityTest, get_number_of_simpson_intervals)
  {
    Mixture::TimestepAdaptivityInfo level{};
    level.emplace_back(3, 2);
    level.emplace_back(2, 1);

    EXPECT_EQ(level.get_number_of_simpson_intervals(3), 2);
    EXPECT_EQ(level.get_number_of_simpson_intervals(2), 1);
    EXPECT_EQ(level.get_number_of_simpson_intervals(1), 0);
  }

  TEST(TimestepAdaptivityTest, get_begin_index)
  {
    Mixture::TimestepAdaptivityInfo level{};
    level.emplace_back(3, 2);
    level.emplace_back(2, 1);

    EXPECT_EQ(level.get_begin_index(3), 0);
    EXPECT_EQ(level.get_begin_index(2), 4);
    EXPECT_EQ(level.get_begin_index(1), 6);
    EXPECT_EQ(level.get_begin_index(0), 6);
  }

  TEST(TimestepAdaptivityTest, get_begin_time)
  {
    Mixture::TimestepAdaptivityInfo level{};
    level.emplace_back(3, 2);
    level.emplace_back(2, 1);

    EXPECT_NEAR(level.get_begin_time(3, 1.0, 0.1), 1.0, 1e-15);
    EXPECT_NEAR(level.get_begin_time(2, 1.0, 0.1), 4.2, 1e-15);
    EXPECT_NEAR(level.get_begin_time(1, 1.0, 0.1), 5.0, 1e-15);
    EXPECT_NEAR(level.get_begin_time(0, 1.0, 0.1), 5.0, 1e-15);
  }

  TEST(TimestepAdaptivityTest, get_index_time)
  {
    Mixture::TimestepAdaptivityInfo level{};
    level.emplace_back(3, 2);
    level.emplace_back(2, 1);

    EXPECT_NEAR(level.get_index_time(0, 1.0, 0.1), 1.0, 1e-15);
    EXPECT_NEAR(level.get_index_time(1, 1.0, 0.1), 1.8, 1e-15);
    EXPECT_NEAR(level.get_index_time(2, 1.0, 0.1), 2.6, 1e-15);
    EXPECT_NEAR(level.get_index_time(3, 1.0, 0.1), 3.4, 1e-15);
    EXPECT_NEAR(level.get_index_time(4, 1.0, 0.1), 4.2, 1e-15);
    EXPECT_NEAR(level.get_index_time(5, 1.0, 0.1), 4.6, 1e-15);
    EXPECT_NEAR(level.get_index_time(6, 1.0, 0.1), 5.0, 1e-15);
    EXPECT_NEAR(level.get_index_time(7, 1.0, 0.1), 5.1, 1e-15);
    EXPECT_NEAR(level.get_index_time(8, 1.0, 0.1), 5.2, 1e-15);
  }

  TEST(TimestepAdaptivityTest, get_number_of_levels)
  {
    Mixture::TimestepAdaptivityInfo level{};
    level.emplace_back(3, 2);
    level.emplace_back(2, 1);

    EXPECT_EQ(level.get_number_of_levels(), 2);
  }

  TEST(TimestepAdaptivityTest, OptimizeHistoryFromUnoptimizedHistory)
  {
    Mixture::TimestepAdaptivityInfo level{};
    const Mixture::LinearCauchyGrowthWithPoissonTurnoverGrowthEvolution<double> growth_evolution(
        0.0, 1.0, true);
    const double time = 30;
    const double dt = 1.0;
    const double tolerance = 1e-8 / 24;

    const auto [remove_items, new_level] = Mixture::optimize_history_integration(level, 25,
        [&](const std::array<std::optional<unsigned int>, 5>& indices)
        {
          const double begin_time = level.get_index_time(indices[0].value(), 0.0, dt);
          const double end_time = level.get_index_time(indices[4].value(), 0.0, dt);
          return Mixture::is_model_equation_simpson_rule_integration_below_tolerance(
              growth_evolution, time, begin_time, end_time, tolerance * (end_time - begin_time));
        });

    level = new_level;

    EXPECT_EQ(level[0], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(2, 1));
    EXPECT_EQ(level[1], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(1, 1));

    std::array comparison = {false, true, true, true, false, true, true, true, false, true, false,
        true, false, false, false, false, false, false, false, false, false, false, false, false,
        false};

    EXPECT_THAT(remove_items, testing::ElementsAreArray(comparison));
  }

  TEST(TimestepAdaptivityTest, OptimizeHistoryFromPartiallyOptimizedHistory)
  {
    Mixture::TimestepAdaptivityInfo level{};
    level.emplace_back(1, 2);

    const Mixture::LinearCauchyGrowthWithPoissonTurnoverGrowthEvolution<double> growth_evolution(
        0.0, 1.0, true);
    const double time = 30;
    const double dt = 1.0;
    const double tolerance = 1e-8 / 24;

    const auto [remove_items, new_level] = Mixture::optimize_history_integration(level, 21,
        [&](const std::array<std::optional<unsigned int>, 5>& indices)
        {
          const double begin_time = level.get_index_time(indices[0].value(), 0.0, dt);
          const double end_time = level.get_index_time(indices[4].value(), 0.0, dt);
          return Mixture::is_model_equation_simpson_rule_integration_below_tolerance(
              growth_evolution, time, begin_time, end_time, tolerance * (end_time - begin_time));
        });


    EXPECT_EQ(new_level[0], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(2, 1));
    EXPECT_EQ(new_level[1], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(1, 1));

    std::array comparison = {false, true, false, true, false, true, false, true, false, false,
        false, false, false, false, false, false, false, false, false, false, false};

    EXPECT_THAT(remove_items, testing::ElementsAreArray(comparison));
  }

  TEST(TimestepAdaptivityTest, OptimizeHistoryFromOptimizedHistory)
  {
    Mixture::TimestepAdaptivityInfo level{};
    level.emplace_back(2, 1);
    level.emplace_back(1, 1);
    const Mixture::LinearCauchyGrowthWithPoissonTurnoverGrowthEvolution<double> growth_evolution(
        0.0, 1.0, true);
    const double time = 30;
    const double dt = 1.0;
    const double tolerance = 1e-8 / 24;

    const auto [remove_items, new_level] = Mixture::optimize_history_integration(level, 17,
        [&](const std::array<std::optional<unsigned int>, 5>& indices)
        {
          const double begin_time = level.get_index_time(indices[0].value(), 0.0, dt);
          const double end_time = level.get_index_time(indices[4].value(), 0.0, dt);
          return Mixture::is_model_equation_simpson_rule_integration_below_tolerance(
              growth_evolution, time, begin_time, end_time, tolerance * (end_time - begin_time));
        });

    EXPECT_EQ(level[0], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(2, 1));
    EXPECT_EQ(level[1], Mixture::TimestepAdaptivityInfo::TimestepAdaptivityInfoItem(1, 1));

    std::array comparison = {false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false};

    EXPECT_THAT(remove_items, testing::ElementsAreArray(comparison));
  }

  TEST(TimestepAdaptivityTest, GetBaseTimestep)
  {
    Mixture::TimestepAdaptivityInfo level{};
    level.emplace_back(3, 1);
    level.emplace_back(2, 2);
    level.emplace_back(1, 1);

    EXPECT_EQ(level.get_base_index(0), 0);
    EXPECT_EQ(level.get_base_index(1), 8);
    EXPECT_EQ(level.get_base_index(2), 16);
    EXPECT_EQ(level.get_base_index(3), 16 + 4);
    EXPECT_EQ(level.get_base_index(4), 16 + 8);
    EXPECT_EQ(level.get_base_index(5), 16 + 12);
    EXPECT_EQ(level.get_base_index(6), 16 + 16);
    EXPECT_EQ(level.get_base_index(7), 16 + 16 + 2);
    EXPECT_EQ(level.get_base_index(8), 16 + 16 + 4);
    EXPECT_EQ(level.get_base_index(9), 16 + 16 + 4 + 1);
    EXPECT_EQ(level.get_base_index(10), 16 + 16 + 4 + 2);
  }

  TEST(TimestepAdaptivityTest, get_base_indices)
  {
    Mixture::TimestepAdaptivityInfo level{};
    level.emplace_back(3, 1);
    level.emplace_back(2, 2);
    level.emplace_back(1, 1);

    auto base_indices =
        level.get_base_indices<unsigned int, 11>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    EXPECT_THAT(base_indices, testing::ElementsAre(0, 8, 16, 20, 24, 28, 32, 34, 36, 37, 38));
  }

  TEST(TimestepAdaptivityTest, get_index_from_base)
  {
    Mixture::TimestepAdaptivityInfo level{};
    level.emplace_back(2, 2);
    level.emplace_back(1, 1);

    EXPECT_EQ(level.get_index_from_base(0), 0);
    EXPECT_FALSE(level.get_index_from_base(1).has_value());
    EXPECT_FALSE(level.get_index_from_base(2).has_value());
    EXPECT_FALSE(level.get_index_from_base(3).has_value());
    EXPECT_EQ(level.get_index_from_base(4), 1);
    EXPECT_FALSE(level.get_index_from_base(5).has_value());
    EXPECT_FALSE(level.get_index_from_base(6).has_value());
    EXPECT_FALSE(level.get_index_from_base(7).has_value());
    EXPECT_EQ(level.get_index_from_base(8), 2);
    EXPECT_FALSE(level.get_index_from_base(9).has_value());
    EXPECT_FALSE(level.get_index_from_base(10).has_value());
    EXPECT_FALSE(level.get_index_from_base(11).has_value());
    EXPECT_EQ(level.get_index_from_base(12), 3);
    EXPECT_FALSE(level.get_index_from_base(13).has_value());
    EXPECT_FALSE(level.get_index_from_base(14).has_value());
    EXPECT_FALSE(level.get_index_from_base(15).has_value());
    EXPECT_EQ(level.get_index_from_base(16), 4);
    EXPECT_FALSE(level.get_index_from_base(17).has_value());
    EXPECT_EQ(level.get_index_from_base(18), 5);
    EXPECT_FALSE(level.get_index_from_base(19).has_value());
    EXPECT_EQ(level.get_index_from_base(20), 6);
    EXPECT_EQ(level.get_index_from_base(21), 7);
    EXPECT_EQ(level.get_index_from_base(22), 8);
  }

  TEST(TimestepAdaptivityTest, get_base_indexWithBase)
  {
    Mixture::TimestepAdaptivityInfo level{};
    level.emplace_back(3, 1);
    level.emplace_back(1, 1);

    Mixture::TimestepAdaptivityInfo base{};
    base.emplace_back(1, 4);

    EXPECT_EQ(level.get_base_index(base, 0), 0);
    EXPECT_EQ(level.get_base_index(base, 1), 4);
    EXPECT_EQ(level.get_base_index(base, 2), 8);
    EXPECT_EQ(level.get_base_index(base, 3), 10);
    EXPECT_EQ(level.get_base_index(base, 4), 12);
    EXPECT_EQ(level.get_base_index(base, 5), 13);
  }

  TEST(TimestepAdaptivityTest, get_base_indicesWithBase)
  {
    Mixture::TimestepAdaptivityInfo level{};
    level.emplace_back(3, 1);
    level.emplace_back(1, 1);

    Mixture::TimestepAdaptivityInfo base{};
    base.emplace_back(1, 4);

    auto base_indices = level.get_base_indices<unsigned int, 6>(base, {0, 1, 2, 3, 4, 5});

    EXPECT_THAT(base_indices, testing::ElementsAre(0, 4, 8, 10, 12, 13));
  }

  TEST(TimestepAdaptivityTest, get_indices_from_base)
  {
    Mixture::TimestepAdaptivityInfo level{};
    level.emplace_back(2, 2);
    level.emplace_back(1, 1);

    auto indices = level.get_indices_from_base<unsigned int, 23>(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22});
    EXPECT_THAT(indices,
        testing::ElementsAre(0, std::nullopt, std::nullopt, std::nullopt, 1, std::nullopt,
            std::nullopt, std::nullopt, 2, std::nullopt, std::nullopt, std::nullopt, 3,
            std::nullopt, std::nullopt, std::nullopt, 4, std::nullopt, 5, std::nullopt, 6, 7, 8));
  }

  TEST(TimestepAdaptivityTest, get_indices_from_baseWrongOrder)
  {
#ifndef FOUR_C_ENABLE_ASSERTIONS
    GTEST_SKIP() << "Skip debug assertion tests in release mode.";
#endif
    Mixture::TimestepAdaptivityInfo level{};
    auto get_indices_from_base = [&]()
    { return level.get_indices_from_base<unsigned int, 3>({0, 3, 1}); };
    EXPECT_ANY_THROW(get_indices_from_base());
  }

  TEST(TimestepAdaptivityTest, get_base_indicesWrongOrder)
  {
#ifndef FOUR_C_ENABLE_ASSERTIONS
    GTEST_SKIP() << "Skip debug assertion tests in release mode.";
#endif
    Mixture::TimestepAdaptivityInfo level{};

    auto get_base_indices = [&]() { return level.get_base_indices<unsigned int, 3>({0, 3, 1}); };
    EXPECT_ANY_THROW(get_base_indices());
  }
}  // namespace