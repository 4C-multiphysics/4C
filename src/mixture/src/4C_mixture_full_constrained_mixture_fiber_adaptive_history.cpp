// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_full_constrained_mixture_fiber_adaptive_history.hpp"

#include "4C_comm_pack_buffer.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_parobject.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_fad.hpp"

#include <Sacado_Fad_DFad.hpp>

#include <array>
#include <optional>

FOUR_C_NAMESPACE_OPEN

void Mixture::Internal::adapt_timestep_adaptivity_info(
    Mixture::TimestepAdaptivityInfo& timestep_adaptivity_info, unsigned int level,
    unsigned int num_coarsened_intervals)
{
  if (level == 0)
  {
    timestep_adaptivity_info.emplace_back(1, num_coarsened_intervals / 2);
  }
  else
  {
    timestep_adaptivity_info.split_level(level, num_coarsened_intervals / 2);
  }
}

void Mixture::Internal::mark_coarsened_timestep_as_to_be_deleted(std::vector<bool>& items_to_delete,
    const unsigned int num_items_to_delete, const unsigned int begin_index)
{
  bool last_deleted = true;
  unsigned int num_deleted = 0;
  unsigned int current_index = begin_index;

  for (; num_deleted < num_items_to_delete; ++current_index)
  {
    if (items_to_delete[current_index]) continue;  // item is already deleted

    // delete only every second item
    if (!last_deleted)
    {
      ++num_deleted;
      items_to_delete[current_index] = true;  // delete item
    }

    last_deleted = !last_deleted;
  }
}

void Mixture::TimestepAdaptivityInfo::pack(Core::Communication::PackBuffer& data) const
{
  data.add_to_pack(get_number_of_levels());
  for (const auto& item : list_)
  {
    data.add_to_pack(item.level_);
    data.add_to_pack(item.simpson_intervals_);
  }
}

void Mixture::TimestepAdaptivityInfo::unpack(Core::Communication::UnpackBuffer& buffer)
{
  std::size_t size_of_adaptivity_info;
  extract_from_pack(buffer, size_of_adaptivity_info);
  list_.clear();
  list_.reserve(size_of_adaptivity_info);
  for (std::size_t i = 0; i < size_of_adaptivity_info; ++i)
  {
    unsigned int level;
    extract_from_pack(buffer, level);
    unsigned int num_intervals;
    extract_from_pack(buffer, num_intervals);
    list_.emplace_back(level, num_intervals);
  }
}

void Mixture::TimestepAdaptivityInfo::emplace_back(
    unsigned int level, unsigned int num_simpson_intervals)
{
  if (list_.size() > 0 && list_.back().level_ == level)
  {
    list_.back().simpson_intervals_ += num_simpson_intervals;
  }
  else
  {
    FOUR_C_ASSERT(list_.size() == 0 || std::min_element(list_.begin(), list_.end(),
                                           [](const TimestepAdaptivityInfoItem& item1,
                                               const TimestepAdaptivityInfoItem& item2)
                                           {
                                             return item1.level_ < item2.level_;
                                           })->level_ > level,
        "The timestep adaptivity info list contains an item with a smaller level than you want "
        "to add "
        "at the end.");
    list_.emplace_back(level, num_simpson_intervals);
  }
}

unsigned int Mixture::TimestepAdaptivityInfo::get_total_number_of_simpson_intervals()
{
  return std::accumulate(list_.begin(), list_.end(), 0,
      [](unsigned int sum, const TimestepAdaptivityInfoItem& item)
      { return sum + item.simpson_intervals_; });
}

void Mixture::TimestepAdaptivityInfo::split_level(
    unsigned int level, unsigned int new_num_simpson_intervals)
{
  for (std::size_t i = 0; i < list_.size(); ++i)
  {
    if (list_[i].level_ == level)
    {
      int remaining_length = list_[i].simpson_intervals_ - new_num_simpson_intervals * 2;
      FOUR_C_ASSERT(
          remaining_length >= 0, "The remaining length is smaller than 0. This is not allowed");
      if (i > 0 && list_[i - 1].level_ == level + 1)
      {
        list_[i - 1].simpson_intervals_ += new_num_simpson_intervals;

        if (remaining_length == 0)
        {
          list_.erase(std::remove(list_.begin(), list_.end(), list_[i]), list_.end());
        }
        else
        {
          list_[i].simpson_intervals_ = remaining_length;
        }
      }
      else if (remaining_length == 0)
      {
        list_[i].level_ += 1;
        list_[i].simpson_intervals_ = new_num_simpson_intervals;
      }
      else
      {
        list_[i].simpson_intervals_ = remaining_length;
        list_.emplace(list_.begin() + i, level + 1, new_num_simpson_intervals);
      }
      return;
    }
  }
  FOUR_C_THROW("Could not find refinement level {} in the list", level);
}

unsigned int Mixture::TimestepAdaptivityInfo::max_level()
{
  if (list_.size() == 0) return 0;
  return list_[0].level_;
}
unsigned int Mixture::TimestepAdaptivityInfo::get_base_index(const unsigned int index) const
{
  return get_base_indices<unsigned int, 1>({index})[0];
}

std::optional<unsigned int> Mixture::TimestepAdaptivityInfo::get_base_index(
    const TimestepAdaptivityInfo& base, unsigned int timestep) const
{
  return base.get_index_from_base(get_base_index(timestep));
}

std::optional<unsigned int> Mixture::TimestepAdaptivityInfo::get_index_from_base(
    const unsigned int base_index) const
{
  return get_indices_from_base<unsigned int, 1>({base_index})[0];
}


unsigned int Mixture::TimestepAdaptivityInfo::get_number_of_simpson_intervals(
    const unsigned int level) const
{
  for (const auto& item : list_)
  {
    if (item.level_ == level)
      return item.simpson_intervals_;
    else if (item.level_ < level)
      return 0;
  }

  // level not in the list, so it is empty
  if (level > 0) return 0;

  FOUR_C_THROW(
      "Cou can only call this item for a level within 0 < x <= max_level(). You called it with "
      "{}",
      level);
}

unsigned int Mixture::TimestepAdaptivityInfo::get_begin_index(const unsigned int level) const
{
  unsigned int current_index = 0;
  for (const auto& item : list_)
  {
    if (item.level_ <= level) return current_index;

    current_index += 2 * item.simpson_intervals_;
  }

  return current_index;
}

double Mixture::TimestepAdaptivityInfo::get_begin_time(
    const unsigned int level, const double base_time, const double base_dt) const
{
  double begin_time = base_time;
  for (const auto& item : list_)
  {
    if (item.level_ <= level) return begin_time;

    begin_time += 2 * base_dt * std::pow(2, item.level_) * item.simpson_intervals_;
  }
  return begin_time;
}

double Mixture::TimestepAdaptivityInfo::get_index_time(
    const unsigned int index, const double base_time, const double base_dt) const
{
  return base_time + get_base_index(index) * base_dt;
}
FOUR_C_NAMESPACE_CLOSE
