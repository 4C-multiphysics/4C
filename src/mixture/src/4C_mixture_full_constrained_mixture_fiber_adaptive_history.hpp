// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_FULL_CONSTRAINED_MIXTURE_FIBER_ADAPTIVE_HISTORY_HPP
#define FOUR_C_MIXTURE_FULL_CONSTRAINED_MIXTURE_FIBER_ADAPTIVE_HISTORY_HPP

#include "4C_config.hpp"

#include "4C_mixture_growth_evolution_linear_cauchy_poisson_turnover.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_fad.hpp"
#include "4C_utils_local_integration.hpp"

#include <algorithm>
#include <numeric>
#include <optional>
#include <string>

FOUR_C_NAMESPACE_OPEN

namespace Core::Communication
{
  class PackBuffer;
  class UnpackBuffer;
}  // namespace Core::Communication
namespace Mixture
{
  namespace Internal
  {
    template <typename Integer>
    [[nodiscard]] Integer integer_power(Integer x, unsigned int p)
    {
      if (p == 0) return 1;
      if (p == 1) return x;

      Integer tmp = integer_power(x, p / 2);
      if (p % 2 == 0)
        return tmp * tmp;
      else
        return x * tmp * tmp;
    }
  }  // namespace Internal
  /*!
   * @brief Container for tracking adapted history integration intervals with providing some
   * convenience functions.
   *
   * It stores the coarsening-levels and the number of coarsening intervals in a list:
   * @f$[(l_1, n_1), (l_2, n_2), \cdots]@f$, whereas
   * @f$l_1 \gt l_2 \gt \cdots @f$.
   *
   * @f$l_i@f$: Level of coarsening. Effective timestep size is @f$t\Delta t_\text{base} *
   * 2^{l_i}@f$.
   * @f$n_i@f$: Number of Simpson intervals following with the same coarsening level.
   *
   * @note Equidistant base timesteps are assumed / required.
   *
   * Suppose you have following distribution of stored timesteps (x: stored, .: removed)
   * @f$x...x...x...x...x.x.x.x.x.x.xxxxx@f$
   * the TimestepAdaptivityInfo would be @f$[(2, 2), (1, 3)]@f$.
   */
  class TimestepAdaptivityInfo
  {
   public:
    struct TimestepAdaptivityInfoItem
    {
      unsigned int level_;
      unsigned int simpson_intervals_;

      TimestepAdaptivityInfoItem(unsigned int level, unsigned int simpson_intervals)
          : level_(level), simpson_intervals_(simpson_intervals)
      {
      }

      bool operator==(const TimestepAdaptivityInfoItem& other) const
      {
        return level_ == other.level_ && simpson_intervals_ == other.simpson_intervals_;
      }
    };

    void pack(Core::Communication::PackBuffer& data) const;

    void unpack(Core::Communication::UnpackBuffer& buffer);

    void emplace_back(unsigned int level, unsigned int num_simpson_intervals);

    unsigned int get_total_number_of_simpson_intervals();

    void split_level(unsigned int level, unsigned int new_num_simpson_intervals);

    unsigned int max_level();

    [[nodiscard]] unsigned int get_base_index(unsigned int index) const;

    [[nodiscard]] std::optional<unsigned int> get_base_index(
        const TimestepAdaptivityInfo& base, unsigned int timestep) const;

    template <typename ValueType, std::size_t size>
    [[nodiscard]] std::array<ValueType, size> get_base_indices(
        const std::array<ValueType, size>& indices) const
    {
      FOUR_C_ASSERT(
          std::is_sorted(indices.begin(), indices.end()), "The input array must be sorted!");

      std::array<ValueType, size> base_indices{};

      std::size_t current_item_index =
          std::distance(indices.begin(), std::find_if(indices.begin(), indices.end(),
                                             [](ValueType base_index) { return base_index != 0; }));
      if (current_item_index == indices.size()) return base_indices;


      unsigned int current_base_index = 0;
      unsigned int current_index = 0;
      unsigned int level_begin_index = 0;
      for (const auto& item : list_)
      {
        for (unsigned int level_step = 1; level_step <= 2 * item.simpson_intervals_; ++level_step)
        {
          current_index += 1;
          current_base_index =
              level_begin_index + level_step * Internal::integer_power(2, item.level_);

          for (; current_item_index < indices.size(); ++current_item_index)
          {
            if (current_index == indices[current_item_index])
            {
              base_indices[current_item_index] = current_base_index;
            }
            else
            {
              break;
            }
          }
          if (current_item_index == indices.size()) return base_indices;
        }

        level_begin_index = current_base_index;
      }

      std::transform(indices.begin() + current_item_index, indices.end(),
          base_indices.begin() + current_item_index,
          [&](ValueType index) { return current_base_index + index - current_index; });
      return base_indices;
    }

    template <typename ValueType, std::size_t size>
    [[nodiscard]] std::array<std::optional<ValueType>, size> get_base_indices(
        const TimestepAdaptivityInfo& base, const std::array<ValueType, size>& indices) const
    {
      return base.get_indices_from_base<ValueType, size>(
          get_base_indices<ValueType, size>(indices));
    }

    [[nodiscard]] std::optional<unsigned int> get_index_from_base(unsigned int base_index) const;

    template <typename ValueType, std::size_t size>
    [[nodiscard]] std::array<std::optional<ValueType>, size> get_indices_from_base(
        const std::array<ValueType, size>& base_indices) const
    {
      FOUR_C_ASSERT(std::is_sorted(base_indices.begin(), base_indices.end()),
          "The input array must be sorted!");

      std::array<std::optional<ValueType>, size> indices{};

      std::size_t current_item_index = std::distance(
          base_indices.begin(), std::find_if(base_indices.begin(), base_indices.end(),
                                    [](ValueType base_index) { return base_index != 0; }));
      std::fill_n(indices.begin(), current_item_index, 0);

      if (current_item_index == indices.size()) return indices;

      unsigned int current_base_index = 0;
      unsigned int current_index = 0;
      unsigned int level_begin_index = 0;
      for (const auto& item : list_)
      {
        for (unsigned int level_step = 1; level_step <= 2 * item.simpson_intervals_; ++level_step)
        {
          current_index += 1;
          current_base_index =
              level_begin_index + level_step * Internal::integer_power(2, item.level_);

          for (; current_item_index < indices.size(); ++current_item_index)
          {
            if (current_base_index == base_indices[current_item_index])
            {
              indices[current_item_index] = current_index;
            }
            else if (current_base_index > base_indices[current_item_index])
            {
              indices[current_item_index] = std::nullopt;
            }
            else
            {
              break;
            }
          }
        }

        level_begin_index = current_base_index;
      }

      std::transform(base_indices.begin() + current_item_index, base_indices.end(),
          indices.begin() + current_item_index,
          [&](ValueType base_index) { return current_index + base_index - current_base_index; });

      return indices;
    }

    [[nodiscard]] unsigned int get_number_of_simpson_intervals(unsigned int level) const;

    [[nodiscard]] unsigned int get_begin_index(unsigned int level) const;

    [[nodiscard]] double get_begin_time(unsigned int level, double base_time, double base_dt) const;

    [[nodiscard]] double get_index_time(unsigned int index, double base_time, double base_dt) const;

    const TimestepAdaptivityInfoItem& operator[](unsigned int i) const { return list_[i]; }

    [[nodiscard]] std::size_t get_number_of_levels() const { return list_.size(); }

    void clear() { list_.clear(); }

    [[nodiscard]] std::vector<TimestepAdaptivityInfoItem>::iterator begin()
    {
      return list_.begin();
    }

    [[nodiscard]] std::vector<TimestepAdaptivityInfoItem>::const_iterator begin() const
    {
      return list_.begin();
    }

    [[nodiscard]] std::vector<TimestepAdaptivityInfoItem>::iterator end() { return list_.end(); }

    [[nodiscard]] std::vector<TimestepAdaptivityInfoItem>::const_iterator end() const
    {
      return list_.end();
    }

   private:
    std::vector<TimestepAdaptivityInfoItem> list_;
  };

  namespace Internal
  {
    void adapt_timestep_adaptivity_info(Mixture::TimestepAdaptivityInfo& timestep_adaptivity_info,
        unsigned int level, unsigned int num_coarsened_intervals);
    void mark_coarsened_timestep_as_to_be_deleted(std::vector<bool>& items_to_delete,
        const unsigned int num_items_to_delete, const unsigned int begin_index);

    template <typename CoarsenableEvaluator>
    unsigned int get_number_coarsenable_intervals(const TimestepAdaptivityInfo& base_info,
        const TimestepAdaptivityInfo& current_info, const unsigned int begin_index,
        const unsigned int max_simpson_intervals, CoarsenableEvaluator corsenable_evaluator)
    {
      unsigned int num_coarsening_level = 0;
      const unsigned int max_coarsened_simpson_intervals = max_simpson_intervals / 2;

      for (unsigned int i = 0; i < max_coarsened_simpson_intervals; ++i)
      {
        if (corsenable_evaluator(current_info.get_base_indices<unsigned int, 5>(
                base_info, {begin_index + i * 4, begin_index + i * 4 + 1, begin_index + i * 4 + 2,
                               begin_index + i * 4 + 3, begin_index + i * 4 + 4})))
        {
          num_coarsening_level += 2;
        }
        else
          break;
      }

      return num_coarsening_level;
    }
  }  // namespace Internal

  /*!
   * @brief Method to optimize the history size for Simpson Integration
   *
   * Returns an boolean-vector with all items. For every item that can be deleted are marked with
   * true. Items that should not be deleted are marked with false. The rule whether two consecutive
   * Simpson-intervals can be coarsened is based on the callable
   * @p coarsenable_evaluator . It will be called for every potentially coarsenable pair of Simpson
   * intervals and must either return true (if interval can be coarsened) or false (if interval
   * cannot be coarsened).
   *
   * @note The given tolerance is meant for the whole interval. The max allowed error of each
   * subinterval is correnspondingly smaller.
   *
   * @note The coarsening level can only decrease.
   *
   * @note This function assumes that the history times are based on the same base different, hence:
   * times = [0.0, 1.0, 2.0, 3.0, 4.0, ...] for an empty coarsening level, base_dt=1.0
   * times = [0.0, 2.0, 4.0, 6.0, 8.0, 9.0, 10.0 ...] for a coarsening level of [(1, 2)]
   * base_dt=1.0
   * times = [0.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0, 17.0, 18.0, ...] for a
   * coarsening level of [(2, 1), (1, 2)], base_dt=1.0
   *
   * @param base_adaptivity_info (in/out) : History data containing information about already
   * removed timesteps (see examples above)
   * @param num_total_steps (in) : number of total timesteps
   * @param coarsenable_evaluator (in) : A callable object that is called for each coarsenable
   * interval that must return true if the interval can be coarsened, otherwise false.
   * @return std::vector<bool> : Vector of bools to mark timestep that are marked to be deleted.
   */
  template <typename CoarsenableEvaluator>
  std::tuple<std::vector<bool>, TimestepAdaptivityInfo> optimize_history_integration(
      const TimestepAdaptivityInfo& base_adaptivity_info, int num_total_steps,
      CoarsenableEvaluator coarsenable_evaluator)
  {
    TimestepAdaptivityInfo current_adaptivity_info = base_adaptivity_info;

    std::vector<bool> items_to_delete(num_total_steps, false);
    for (unsigned int check_level = 0; check_level <= current_adaptivity_info.max_level();
        ++check_level)
    {
      const unsigned int begin_index = current_adaptivity_info.get_begin_index(check_level);
      const unsigned int max_simpson_intervals = std::invoke(
          [&]()
          {
            if (check_level == 0)
            {
              return (num_total_steps - 1) / 2 -
                     current_adaptivity_info.get_total_number_of_simpson_intervals();
            }
            else
            {
              return current_adaptivity_info.get_number_of_simpson_intervals(check_level);
            }
          });

      unsigned int num_coarsenable_intervals =
          Internal::get_number_coarsenable_intervals<CoarsenableEvaluator>(base_adaptivity_info,
              current_adaptivity_info, begin_index, max_simpson_intervals, coarsenable_evaluator);

      if (num_coarsenable_intervals > 0)
      {
        Internal::adapt_timestep_adaptivity_info(
            current_adaptivity_info, check_level, num_coarsenable_intervals);
        Internal::mark_coarsened_timestep_as_to_be_deleted(
            items_to_delete, num_coarsenable_intervals, begin_index);
      }
    }

    return std::make_tuple(items_to_delete, current_adaptivity_info);
  }

  /*!
   * @brief Check interval if coarsening is possible by approximating the integration error by
   * comparing the the integration of a model equation using Simpson and the analytical integration.
   *
   * @param growth_evolution (in) : Information about decay of tissue
   * @param time (in) : current time
   * @param begin_time (in) : Begin time of the coarsenable interval
   * @param end_time (in) : End time of the coarsenable interval
   * @param tolerance (in) : tolerance
   * @return true if error is smaller than tolerance, else false
   */
  template <typename Number>
  bool is_model_equation_simpson_rule_integration_below_tolerance(
      const Mixture::LinearCauchyGrowthWithPoissonTurnoverGrowthEvolution<Number>& growth_evolution,
      const double time, const double begin_time, const double end_time, const Number tolerance)
  {
    const double dt = (end_time - begin_time) / 2;
    const Number numerical_integration =
        Core::Utils::integrate_simpson_step(dt,
            growth_evolution.evaluate_survival_function(time - begin_time),
            growth_evolution.evaluate_survival_function(time - (begin_time + end_time) / 2),
            growth_evolution.evaluate_survival_function(time - end_time)) /
        growth_evolution.decay_time_;
    const Number exact_integration =
        growth_evolution.evaluate_survival_function_integration(time, begin_time, end_time) /
        growth_evolution.decay_time_;

    return Core::FADUtils::norm<Number>(numerical_integration - exact_integration) <= tolerance ||
           (Core::FADUtils::norm<Number>(numerical_integration) <= tolerance &&
               Core::FADUtils::norm<Number>(exact_integration) <= tolerance);
  }
}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif