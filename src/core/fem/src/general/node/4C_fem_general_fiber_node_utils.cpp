// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_fiber_node_utils.hpp"

#include "4C_fem_general_fiber_node.hpp"
#include "4C_fem_general_fiber_node_holder.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN

template <Core::FE::CellType distype>
void Core::Nodes::project_fibers_to_gauss_points(const Core::Nodes::Node* const* nodes,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(distype), 1>>& shapefcts,
    NodalFiberHolder& gpFiberHolder)
{
  // number of nodes per element
  constexpr int nen = Core::FE::num_nodes(distype);
  std::array<const Core::Nodes::FiberNode*, nen> fiberNodes;
  std::vector<std::array<std::array<double, 3>, nen>> fibers;
  std::map<Core::Nodes::CoordinateSystemDirection, std::array<std::array<double, 3>, nen>>
      coordinateSystemDirections;
  std::map<Core::Nodes::AngleType, std::array<double, nen>> angles;

  for (int inode = 0; inode < nen; ++inode)
  {
    fiberNodes[inode] = dynamic_cast<const Core::Nodes::FiberNode*>(nodes[inode]);

    if (fiberNodes[inode] == nullptr)
    {
      FOUR_C_THROW("At least one node of the element does not provide fibers.");
    }

    if (inode > 0)
    {
      if (fiberNodes[inode]->fibers().size() != fiberNodes[0]->fibers().size())
      {
        FOUR_C_THROW(
            "All nodes of an element must define the same number of FIBER directions. Node {} "
            "defines {}, but node {} defines {}.",
            fiberNodes[inode]->id(), fiberNodes[inode]->fibers().size(), fiberNodes[0]->id(),
            fiberNodes[0]->fibers().size());
      }

      const auto& reference_directions = fiberNodes[0]->coordinate_system_directions();
      const auto& node_directions = fiberNodes[inode]->coordinate_system_directions();
      const bool same_coordinate_directions =
          reference_directions.size() == node_directions.size() &&
          std::all_of(reference_directions.begin(), reference_directions.end(),
              [&node_directions](const auto& direction)
              { return node_directions.contains(direction.first); });
      if (!same_coordinate_directions)
      {
        FOUR_C_THROW(
            "All nodes of an element must define the same set of coordinate system direction "
            "types (CIR and TAN).");
      }

      const auto& reference_angles = fiberNodes[0]->angles();
      const auto& node_angles = fiberNodes[inode]->angles();
      const bool same_angles =
          reference_angles.size() == node_angles.size() &&
          std::all_of(reference_angles.begin(), reference_angles.end(),
              [&node_angles](const auto& angle) { return node_angles.contains(angle.first); });
      if (!same_angles)
      {
        FOUR_C_THROW("All nodes of an element must define the same angle types.");
      }
    }

    for (const auto& pair : fiberNodes[inode]->coordinate_system_directions())
    {
      coordinateSystemDirections[pair.first][inode] = pair.second;
    }

    for (std::size_t fiberid = 0; fiberid < fiberNodes[inode]->fibers().size(); ++fiberid)
    {
      if (inode == 0)
      {
        // The first node has to create the array of fibers for each node
        fibers.emplace_back(std::array<std::array<double, 3>, nen>());
      }
      fibers[fiberid][inode] = fiberNodes[inode]->fibers()[fiberid];
    }

    for (const auto& pair : fiberNodes[inode]->angles())
    {
      angles[pair.first][inode] = pair.second;
    }
  }


  // project fibers
  for (const auto& fiber : fibers)
  {
    std::vector<Core::LinAlg::Tensor<double, 3>> gpQuantity;
    project_quantity_with_shape_functions<distype, 3>(fiber, shapefcts, gpQuantity);

    // normalize fiber vectors
    for (Core::LinAlg::Tensor<double, 3>& quantity : gpQuantity)
    {
      quantity *= 1.0 / Core::LinAlg::norm2(quantity);
    }

    // add quantity to the fiber holder
    gpFiberHolder.add_fiber(gpQuantity);
  }


  // project coordinate system directions
  for (const auto& pair : coordinateSystemDirections)
  {
    const Core::Nodes::CoordinateSystemDirection type = pair.first;
    std::vector<Core::LinAlg::Tensor<double, 3>> gpQuantity;
    project_quantity_with_shape_functions<distype, 3>(pair.second, shapefcts, gpQuantity);

    // normalize fiber vectors
    for (Core::LinAlg::Tensor<double, 3>& quantity : gpQuantity)
    {
      quantity *= 1.0 / Core::LinAlg::norm2(quantity);
    }

    // add quantity to the fiber holder
    gpFiberHolder.set_coordinate_system_direction(type, gpQuantity);
  }

  // project angles
  for (const auto& pair : angles)
  {
    const Core::Nodes::AngleType type = pair.first;
    std::vector<double> gpAngle;
    project_quantity_with_shape_functions<distype>(pair.second, shapefcts, gpAngle);

    // add quantity to the fiber holder
    gpFiberHolder.set_angle(type, gpAngle);
  }


  // orthogonalize coordinate system
  if (gpFiberHolder.contains_coordinate_system_direction(CoordinateSystemDirection::Circular) &&
      gpFiberHolder.contains_coordinate_system_direction(CoordinateSystemDirection::Tangential))
  {
    const std::vector<Core::LinAlg::Tensor<double, 3>>& cir =
        gpFiberHolder.get_coordinate_system_direction(CoordinateSystemDirection::Circular);
    std::vector<Core::LinAlg::Tensor<double, 3>>& tan =
        gpFiberHolder.get_coordinate_system_direction_mutual(CoordinateSystemDirection::Tangential);

    // orthogonalize tangential vector, preserve circular direction
    for (std::size_t gp = 0; gp < tan.size(); ++gp)
    {
      double tancir = tan[gp] * cir[gp];
      tan[gp] += -tancir * cir[gp];
      tan[gp] *= 1.0 / Core::LinAlg::norm2(tan[gp]);
    }
  }
}

template <Core::FE::CellType distype, std::size_t dim>
void Core::Nodes::project_quantity_with_shape_functions(
    const std::array<std::array<double, dim>, Core::FE::num_nodes(distype)> quantity,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(distype), 1>>& shapefcts,
    std::vector<Core::LinAlg::Tensor<double, dim>>& quantityProjected)
{
  quantityProjected.resize(shapefcts.size());

  // number of nodes per element
  constexpr std::size_t nen = Core::FE::num_nodes(distype);

  // spatial dimension
  constexpr std::size_t nsd = Core::FE::dim<distype>;

  // number of gauss points
  const std::size_t ngp = shapefcts.size();

  for (Core::LinAlg::Tensor<double, dim>& quantity : quantityProjected)
  {
    quantity = {};
  }

  for (std::size_t i = 0; i < nsd; ++i)
  {
    for (std::size_t q = 0; q < ngp; ++q)
    {
      for (std::size_t j = 0; j < nen; ++j)
      {
        quantityProjected[q](i) += shapefcts[q](j) * quantity[j][i];
      }
    }
  }
}

template <Core::FE::CellType distype>
void Core::Nodes::project_quantity_with_shape_functions(
    const std::array<double, Core::FE::num_nodes(distype)> quantity,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(distype), 1>>& shapefcts,
    std::vector<double>& quantityProjected)
{
  quantityProjected.resize(shapefcts.size());

  // number of nodes per element
  constexpr int nen = Core::FE::num_nodes(distype);

  // number of gauss points
  const int ngp = shapefcts.size();

  for (double& quantity : quantityProjected)
  {
    quantity = 0.0;
  }

  for (int q = 0; q < ngp; ++q)
  {
    for (int j = 0; j < nen; ++j)
    {
      quantityProjected[q] += shapefcts[q](j) * quantity[j];
    }
  }
}

template <Core::FE::CellType distype>
bool Core::Nodes::have_nodal_fibers(const Core::Nodes::Node* const* nodes)
{
  constexpr int numberOfNodes = Core::FE::num_nodes(distype);

  // check whether node can be casted to FiberNode
  auto nodeHasFibers = [](const Core::Nodes::Node* n)
  { return dynamic_cast<const Core::Nodes::FiberNode*>(n) != nullptr; };
  return std::all_of(nodes, &nodes[numberOfNodes], nodeHasFibers);
}

template void Core::Nodes::project_quantity_with_shape_functions<Core::FE::CellType::tet4>(
    const std::array<double, Core::FE::num_nodes(Core::FE::CellType::tet4)> quantity,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::tet4), 1>>&
        shapefcts,
    std::vector<double>& quantityProjected);
template void Core::Nodes::project_quantity_with_shape_functions<Core::FE::CellType::tet10>(
    const std::array<double, Core::FE::num_nodes(Core::FE::CellType::tet10)> quantity,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::tet10), 1>>&
        shapefcts,
    std::vector<double>& quantityProjected);
template void Core::Nodes::project_quantity_with_shape_functions<Core::FE::CellType::hex8>(
    const std::array<double, Core::FE::num_nodes(Core::FE::CellType::hex8)> quantity,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::hex8), 1>>&
        shapefcts,
    std::vector<double>& quantityProjected);
template void Core::Nodes::project_quantity_with_shape_functions<Core::FE::CellType::tri3>(
    const std::array<double, Core::FE::num_nodes(Core::FE::CellType::tri3)> quantity,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::tri3), 1>>&
        shapefcts,
    std::vector<double>& quantityProjected);

template void Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::tet4>(
    const Core::Nodes::Node* const*,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::tet4), 1>>&,
    NodalFiberHolder&);
template void Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::tet10>(
    const Core::Nodes::Node* const*,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::tet10), 1>>&,
    NodalFiberHolder&);
template void Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::hex8>(
    const Core::Nodes::Node* const*,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::hex8), 1>>&,
    NodalFiberHolder&);
template void Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::hex18>(
    const Core::Nodes::Node* const*,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::hex18), 1>>&,
    NodalFiberHolder&);
template void Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::hex20>(
    const Core::Nodes::Node* const*,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::hex20), 1>>&,
    NodalFiberHolder&);
template void Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::hex27>(
    const Core::Nodes::Node* const*,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::hex27), 1>>&,
    NodalFiberHolder&);
template void Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::nurbs27>(
    const Core::Nodes::Node* const*,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::nurbs27), 1>>&,
    NodalFiberHolder&);
template void Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::tri3>(
    const Core::Nodes::Node* const*,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::tri3), 1>>&,
    NodalFiberHolder&);
template void Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::tri6>(
    const Core::Nodes::Node* const*,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::tri6), 1>>&,
    NodalFiberHolder&);
template void Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::quad4>(
    const Core::Nodes::Node* const*,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::quad4), 1>>&,
    NodalFiberHolder&);
template void Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::quad8>(
    const Core::Nodes::Node* const*,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::quad8), 1>>&,
    NodalFiberHolder&);
template void Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::quad9>(
    const Core::Nodes::Node* const*,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::quad9), 1>>&,
    NodalFiberHolder&);
template void Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::pyramid5>(
    const Core::Nodes::Node* const*,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::pyramid5), 1>>&,
    NodalFiberHolder&);
template void Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::wedge6>(
    const Core::Nodes::Node* const*,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::wedge6), 1>>&,
    NodalFiberHolder&);
template void Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::nurbs9>(
    const Core::Nodes::Node* const*,
    const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(Core::FE::CellType::nurbs9), 1>>&,
    NodalFiberHolder&);

template bool Core::Nodes::have_nodal_fibers<Core::FE::CellType::tet4>(
    const Core::Nodes::Node* const* nodes);
template bool Core::Nodes::have_nodal_fibers<Core::FE::CellType::tet10>(
    const Core::Nodes::Node* const* nodes);
template bool Core::Nodes::have_nodal_fibers<Core::FE::CellType::hex8>(
    const Core::Nodes::Node* const* nodes);
template bool Core::Nodes::have_nodal_fibers<Core::FE::CellType::hex18>(
    const Core::Nodes::Node* const* nodes);
template bool Core::Nodes::have_nodal_fibers<Core::FE::CellType::hex20>(
    const Core::Nodes::Node* const* nodes);
template bool Core::Nodes::have_nodal_fibers<Core::FE::CellType::hex27>(
    const Core::Nodes::Node* const* nodes);
template bool Core::Nodes::have_nodal_fibers<Core::FE::CellType::nurbs27>(
    const Core::Nodes::Node* const* nodes);
template bool Core::Nodes::have_nodal_fibers<Core::FE::CellType::pyramid5>(
    const Core::Nodes::Node* const* nodes);
template bool Core::Nodes::have_nodal_fibers<Core::FE::CellType::wedge6>(
    const Core::Nodes::Node* const* nodes);
template bool Core::Nodes::have_nodal_fibers<Core::FE::CellType::quad4>(
    const Core::Nodes::Node* const* nodes);
template bool Core::Nodes::have_nodal_fibers<Core::FE::CellType::quad8>(
    const Core::Nodes::Node* const* nodes);
template bool Core::Nodes::have_nodal_fibers<Core::FE::CellType::quad9>(
    const Core::Nodes::Node* const* nodes);
template bool Core::Nodes::have_nodal_fibers<Core::FE::CellType::nurbs9>(
    const Core::Nodes::Node* const* nodes);
template bool Core::Nodes::have_nodal_fibers<Core::FE::CellType::tri3>(
    const Core::Nodes::Node* const* nodes);
template bool Core::Nodes::have_nodal_fibers<Core::FE::CellType::tri6>(
    const Core::Nodes::Node* const* nodes);

FOUR_C_NAMESPACE_CLOSE
