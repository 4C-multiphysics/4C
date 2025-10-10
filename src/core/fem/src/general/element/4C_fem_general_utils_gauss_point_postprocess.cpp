// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_utils_gauss_point_postprocess.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_gauss_point_extrapolation.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_fem_general_utils_integration.hpp"

#include <unordered_map>

FOUR_C_NAMESPACE_OPEN

namespace
{
  template <Core::FE::CellType distype>
  const auto& get_gauss_integration(unsigned numgp)
  {
    static std::unordered_map<unsigned, Core::FE::IntegrationPoints<Core::FE::dim<distype>>>
        gaussIntegrations = {};

    if (gaussIntegrations.find(numgp) == gaussIntegrations.end())
    {
      auto rule = Core::FE::num_gauss_points_to_gauss_rule<distype>(numgp);
      gaussIntegrations.emplace(numgp, rule);
    }

    return gaussIntegrations.at(numgp);
  }
}  // namespace

void Core::FE::assemble_gauss_point_values(
    std::vector<std::shared_ptr<Core::LinAlg::MultiVector<double>>>& global_data,
    const Core::LinAlg::SerialDenseMatrix& gp_data, const Core::Elements::Element& ele)
{
  for (int gp = 0; gp < gp_data.numRows(); ++gp)
  {
    const Core::LinAlg::Map& elemap = global_data[gp]->get_map();
    int lid = elemap.lid(ele.id());
    if (lid != -1)
    {
      for (int i = 0; i < gp_data.numCols(); ++i)
      {
        ((*global_data[gp])(i)).get_values()[lid] += gp_data(gp, i);
      }
    }
  }
}

void Core::FE::assemble_nodal_element_count(
    Core::LinAlg::Vector<int>& global_count, const Core::Elements::Element& ele)
{
  for (int n = 0; n < ele.num_node(); ++n)
  {
    const int lid = global_count.get_map().lid(ele.node_ids()[n]);

    if (lid != -1)
    {
      global_count.get_local_values()[lid] += 1;
    }
  }
}

void Core::FE::extrapolate_gauss_point_quantity_to_nodes(Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& data, const Core::FE::Discretization& dis,
    Core::LinAlg::MultiVector<double>& nodal_data)
{
  switch (ele.shape())
  {
    case Core::FE::CellType::hex8:
    {
      Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::hex8>(ele, data,
          nodal_data, true, get_gauss_integration<Core::FE::CellType::hex8>(data.numRows()));
    }
    break;
    case Core::FE::CellType::hex27:
    {
      Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::hex27>(ele, data,
          nodal_data, true, get_gauss_integration<Core::FE::CellType::hex27>(data.numRows()));
    }
    break;
    case Core::FE::CellType::nurbs27:
    {
      Core::FE::extrapolate_gp_quantity_to_nurbs_knots_and_assemble<Core::FE::CellType::nurbs27>(
          dis, ele, data, nodal_data, true,
          get_gauss_integration<Core::FE::CellType::nurbs27>(data.numRows()));
    }
    break;
    case Core::FE::CellType::tet10:
    {
      Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::tet10>(ele, data,
          nodal_data, true, get_gauss_integration<Core::FE::CellType::tet10>(data.numRows()));
    }
    break;
    case Core::FE::CellType::tet4:
    {
      Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::tet4>(ele, data,
          nodal_data, true, get_gauss_integration<Core::FE::CellType::tet4>(data.numRows()));
    }
    break;
    case Core::FE::CellType::hex20:
    {
      Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::hex20>(ele, data,
          nodal_data, true, get_gauss_integration<Core::FE::CellType::hex20>(data.numRows()));
    }
    break;
    case Core::FE::CellType::wedge6:
    {
      Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::wedge6>(ele, data,
          nodal_data, true, get_gauss_integration<Core::FE::CellType::wedge6>(data.numRows()));
    }
    break;
    case Core::FE::CellType::wedge15:
    {
      Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::wedge15>(ele,
          data, nodal_data, true,
          get_gauss_integration<Core::FE::CellType::wedge15>(data.numRows()));
    }
    break;
    case Core::FE::CellType::pyramid5:
    {
      Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::pyramid5>(ele,
          data, nodal_data, true,
          get_gauss_integration<Core::FE::CellType::pyramid5>(data.numRows()));
    }
    break;
    case Core::FE::CellType::quad4:
    {
      Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::quad4>(ele, data,
          nodal_data, true, get_gauss_integration<Core::FE::CellType::quad4>(data.numRows()));
    }
    break;
    case Core::FE::CellType::quad8:
    {
      Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::quad8>(ele, data,
          nodal_data, true, get_gauss_integration<Core::FE::CellType::quad8>(data.numRows()));
    }
    break;
    case Core::FE::CellType::quad9:
    {
      Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::quad9>(ele, data,
          nodal_data, true, get_gauss_integration<Core::FE::CellType::quad9>(data.numRows()));
    }
    break;
    case Core::FE::CellType::tri3:
    {
      Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::tri3>(ele, data,
          nodal_data, true, get_gauss_integration<Core::FE::CellType::tri3>(data.numRows()));
    }
    break;
    case Core::FE::CellType::tri6:
    {
      Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::tri6>(ele, data,
          nodal_data, true, get_gauss_integration<Core::FE::CellType::tri6>(data.numRows()));
    }
    break;
    default:
      FOUR_C_THROW("Your discretization type ({}) is not yet in the list!",
          Core::FE::cell_type_to_string(ele.shape()));
  }
}

void Core::FE::evaluate_gauss_point_quantity_at_element_center(Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& data, Core::LinAlg::MultiVector<double>& element_data)
{
  assemble_averaged_element_values(element_data, data, ele);
}

FOUR_C_NAMESPACE_CLOSE
