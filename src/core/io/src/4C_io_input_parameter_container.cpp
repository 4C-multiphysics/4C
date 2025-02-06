// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_input_parameter_container.hpp"

#include "4C_linalg_vector.hpp"

FOUR_C_NAMESPACE_OPEN

void Core::IO::InputParameterContainer::print(std::ostream& os) const
{
  for (const auto& [key, entry] : entries_)
  {
    os << key << " : ";
    get_type_actions().at(entry.data.type()).print(os, entry.data);
  }

  for (const auto& [key, group] : groups_)
  {
    os << key << " : ";
    group.print(os);
  }
}


Core::IO::InputParameterContainer& Core::IO::InputParameterContainer::group(const std::string& name)
{
  return groups_[name];
}


const Core::IO::InputParameterContainer& Core::IO::InputParameterContainer::group(
    const std::string& name) const
{
  FOUR_C_ASSERT_ALWAYS(groups_.count(name) > 0, "Group '%s' not found in container.", name.c_str());
  return groups_.at(name);
}

bool Core::IO::InputParameterContainer::has_group(const std::string& name) const
{
  return groups_.count(name) > 0;
}

void Core::IO::InputParameterContainer::merge(const Core::IO::InputParameterContainer& other)
{
  const auto combine_maps = [](auto& map1, const auto& map2)
  {
    for (const auto& [key, value] : map2)
    {
      if (map1.count(key) > 0)
      {
        FOUR_C_THROW("Duplicate key '%s' encountered while merging two containers.", key.c_str());
      }
      map1[key] = value;
    }
  };

  combine_maps(entries_, other.entries_);
  combine_maps(groups_, other.groups_);
}


void Core::IO::InputParameterContainer::clear()
{
  entries_.clear();
  groups_.clear();
}

Teuchos::ParameterList Core::IO::InputParameterContainer::to_teuchos_parameter_list() const
{
  Teuchos::ParameterList pl;

  for (const auto& [key, entry] : entries_)
  {
    get_type_actions().at(entry.data.type()).write_to_pl(pl, key, entry.data);
  }

  for (const auto& [key, group] : groups_)
  {
    pl.sublist(key) = group.to_teuchos_parameter_list();
  }
  return pl;
}

std::map<std::type_index, Core::IO::InputParameterContainer::TypeActions>&
Core::IO::InputParameterContainer::get_type_actions()
{
  static std::map<std::type_index, TypeActions> type_actions;
  return type_actions;
}



FOUR_C_NAMESPACE_CLOSE
