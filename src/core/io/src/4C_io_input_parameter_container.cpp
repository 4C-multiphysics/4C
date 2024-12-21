// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_input_parameter_container.hpp"

#include "4C_linalg_vector.hpp"

FOUR_C_NAMESPACE_OPEN


std::ostream& operator<<(std::ostream& os, const Core::IO::InputParameterContainer& cont)
{
  cont.print(os);
  return os;
}

namespace
{
  //! Print various types that occurr in the Container
  struct PrintHelper
  {
    //! Base case: print the object directly.
    template <typename T>
    void operator()(const T& object)
    {
      os << object << " ";
    }

    //! Print elements of a vector.
    template <typename T>
    void operator()(const std::vector<T>& vector)
    {
      for (const auto& v : vector)
      {
        (*this)(v);
      }
    }

    //! Print elements of a map.
    template <typename Key, typename Value>
    void operator()(const std::map<Key, Value>& map)
    {
      for (const auto& [key, value] : map)
      {
        os << key << " : ";
        (*this)(value);
      }
    }

    //! Print any data.
    void operator()(const std::any& /*unused*/) { os << "non-printable data of type std::any"; }

    std::ostream& os;
  };
}  // namespace


void Core::IO::InputParameterContainer::print(std::ostream& os) const
{
  PrintHelper printer{os};
  printer(intdata_);
  printer(doubledata_);
  printer(booldata_);
  printer(vecintdata_);
  printer(vecdoubledata_);
  printer(mapdata_);
  printer(stringdata_);
  printer(anydata_);
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

void Core::IO::InputParameterContainer::merge(const Core::IO::InputParameterContainer& other)
{
  const auto combine_maps = [](auto& map1, const auto& map2)
  {
    for (const auto& [key, value] : map2)
    {
      if (map1.count(key) > 0)
      {
        FOUR_C_THROW("Key %s already exists in the container!", key.c_str());
      }
      map1[key] = value;
    }
  };

  combine_maps(intdata_, other.intdata_);
  combine_maps(doubledata_, other.doubledata_);
  combine_maps(booldata_, other.booldata_);
  combine_maps(vecintdata_, other.vecintdata_);
  combine_maps(vecdoubledata_, other.vecdoubledata_);
  combine_maps(mapdata_, other.mapdata_);
  combine_maps(stringdata_, other.stringdata_);
  combine_maps(anydata_, other.anydata_);
  combine_maps(groups_, other.groups_);
}


FOUR_C_NAMESPACE_CLOSE
