#include "4C_fem_general_fiber_node_holder.hpp"

#include "4C_comm_parobject.hpp"
#include "4C_fem_general_fiber_node.hpp"

FOUR_C_NAMESPACE_OPEN

void Core::Nodes::NodalFiberHolder::set_coordinate_system_direction(
    Core::Nodes::CoordinateSystemDirection type,
    const std::vector<Core::LinAlg::Matrix<3, 1>>& fiber)
{
  coordinate_system_directions_.insert(std::pair<Core::Nodes::CoordinateSystemDirection,
      const std::vector<Core::LinAlg::Matrix<3, 1>>>(type, fiber));
}

const std::vector<Core::LinAlg::Matrix<3, 1>>&
Core::Nodes::NodalFiberHolder::get_coordinate_system_direction(
    Core::Nodes::CoordinateSystemDirection type) const
{
  return coordinate_system_directions_.at(type);
}

std::vector<Core::LinAlg::Matrix<3, 1>>&
Core::Nodes::NodalFiberHolder::get_coordinate_system_direction_mutual(
    Core::Nodes::CoordinateSystemDirection type)
{
  return coordinate_system_directions_.at(type);
}

void Core::Nodes::NodalFiberHolder::add_fiber(const std::vector<Core::LinAlg::Matrix<3, 1>>& fiber)
{
  fibers_.emplace_back(fiber);
}

const std::vector<Core::LinAlg::Matrix<3, 1>>& Core::Nodes::NodalFiberHolder::get_fiber(
    std::size_t fiberid) const
{
  return fibers_.at(fiberid);
}

std::vector<Core::LinAlg::Matrix<3, 1>>& Core::Nodes::NodalFiberHolder::get_fiber_mutual(
    std::size_t fiberid)
{
  return fibers_.at(fiberid);
}

void Core::Nodes::NodalFiberHolder::set_angle(AngleType type, const std::vector<double>& angle)
{
  angles_.insert(std::pair<Core::Nodes::AngleType, const std::vector<double>>(type, angle));
}

const std::vector<double>& Core::Nodes::NodalFiberHolder::get_angle(
    Core::Nodes::AngleType type) const
{
  return angles_.at(type);
}

std::size_t Core::Nodes::NodalFiberHolder::fibers_size() const { return fibers_.size(); }

std::size_t Core::Nodes::NodalFiberHolder::coordinate_system_size() const
{
  return coordinate_system_directions_.size();
}

std::size_t Core::Nodes::NodalFiberHolder::angles_size() const { return angles_.size(); }

bool Core::Nodes::NodalFiberHolder::contains_coordinate_system_direction(
    Core::Nodes::CoordinateSystemDirection type) const
{
  return coordinate_system_directions_.find(type) != coordinate_system_directions_.end();
}

FOUR_C_NAMESPACE_CLOSE
