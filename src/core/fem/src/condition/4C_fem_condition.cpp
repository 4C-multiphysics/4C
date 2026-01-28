// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition.hpp"

#include "4C_fem_general_element.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN


Core::Conditions::Condition::Condition(const int id, const Core::Conditions::ConditionType type,
    const bool buildgeometry, const Core::Conditions::GeometryType gtype, EntityType entity_type,
    std::optional<std::string> node_set_name)
    : id_(id),
      buildgeometry_(buildgeometry),
      type_(type),
      gtype_(gtype),
      entity_type_(entity_type),
      node_set_name_(std::move(node_set_name))
{
  if (entity_type_ == EntityType::node_set_name)
  {
    FOUR_C_ASSERT(node_set_name_.has_value(),
        "Condition {} of ENTITY_TYPE 'node_set_name' requires a NODE_SET_NAME to be provided.",
        id_);
  }
}

std::ostream& operator<<(std::ostream& os, const Core::Conditions::Condition& cond)
{
  cond.print(os);
  return os;
}


void Core::Conditions::Condition::print(std::ostream& os) const
{
  os << "Condition " << id_ << " " << to_string(type_) << ": ";
  container_.print(os);
  os << std::endl;
  if (nodes_.size() != 0)
  {
    os << "Nodes of this condition:";
    for (const auto& node_gid : nodes_) os << " " << node_gid;
    os << std::endl;
  }
  if (!geometry_.empty())
  {
    os << "Elements of this condition:";
    for (const auto& [ele_id, ele] : geometry_) os << " " << ele_id;
    os << std::endl;
  }
}

std::shared_ptr<Core::Conditions::Condition> Core::Conditions::Condition::copy_without_geometry()
    const
{
  std::shared_ptr<Core::Conditions::Condition> copy(new Condition(*this));
  copy->clear_geometry();
  return copy;
}

std::string Core::Conditions::Condition::node_set_name() const
{
  FOUR_C_ASSERT(node_set_name_.has_value(), "Condition {} has no node_set_name assigned!", id_);
  return node_set_name_.value();
}


FOUR_C_NAMESPACE_CLOSE
