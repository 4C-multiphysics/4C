// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_ELEMENT_APPEND_VISUALIZATION_HPP
#define FOUR_C_IO_ELEMENT_APPEND_VISUALIZATION_HPP

#include "4C_config.hpp"

#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_io_element_vtk_cell_type_register.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  /**
   * \brief Add the element geometry visualization for elements that use Lagrange shape functions
   *
   * @param ele (in) Element
   * @param discret (in) discretization
   * @param cell_types (in/out) cell type data vector
   * @param point_coordinates (in/out) point coordinates for the representation of this element
   * @return Number of added points
   */
  unsigned int append_visualization_geometry_lagrange_ele(const Core::Elements::Element& ele,
      const Core::FE::Discretization& discret, std::vector<uint8_t>& cell_types,
      std::vector<double>& point_coordinates)
  {
    const unsigned int num_spatial_dimensions = 3;
    auto vtk_cell_info = get_vtk_cell_type_from_element_cell_type(ele.shape());
    const std::vector<int>& numbering = vtk_cell_info.second;

    // Add the cell type to the output.
    cell_types.push_back(vtk_cell_info.first);

    // Add each node to the output.
    auto nodes = ele.node_range();

    for (auto node : numbering | std::views::transform([&](int i) { return nodes[i]; }))
      for (unsigned int idim = 0; idim < num_spatial_dimensions; ++idim)
      {
        const auto x = node.x();
        if (idim < x.size())
          point_coordinates.push_back(x[idim]);
        else
          point_coordinates.push_back(0.0);
      }

    // Return the number of added points.
    return ele.num_node();
  }

  /**
   * \brief Helper function to return the proper VTK cell type for nurbs elements.
   */
  template <Core::FE::CellType celltype>
  auto get_vtk_cell_type_for_nurbs_elements()
  {
    switch (celltype)
    {
      case Core::FE::CellType::nurbs9:
        return get_vtk_cell_type_from_element_cell_type(Core::FE::CellType::quad9);
      case Core::FE::CellType::nurbs27:
        return get_vtk_cell_type_from_element_cell_type(Core::FE::CellType::hex27);
      default:
        FOUR_C_THROW("The VTK cell type for the NURBS element {} is not implemented",
            Core::FE::cell_type_to_string(celltype));
    }
  }

  /**
   * \brief Add the element geometry visualization for elements that use NURBS shape functions
   *
   * @param ele (in) Element
   * @param discret (in) discretization
   * @param cell_types (in/out) cell type data vector
   * @param point_coordinates (in/out) point coordinates for the representation of this element
   * @return Number of added points
   */
  template <Core::FE::CellType celltype>
  unsigned int append_visualization_geometry_nurbs(const Core::Elements::Element& ele,
      const Core::FE::Discretization& discret, std::vector<uint8_t>& cell_types,
      std::vector<double>& point_coordinates)
  {
    constexpr int number_of_output_points = Core::FE::num_nodes(celltype);
    constexpr int dim_nurbs = Core::FE::dim<celltype>;
    constexpr int dim_output = 3;

    // Get the vtk cell information
    const auto vtk_cell_info = get_vtk_cell_type_for_nurbs_elements<celltype>();
    const std::vector<int>& numbering = vtk_cell_info.second;

    // Add the cell type to the output.
    cell_types.push_back(vtk_cell_info.first);

    // Create the vertices for the visualization.
    {
      // Get the knots and weights of this element.
      Core::LinAlg::Matrix<number_of_output_points, 1, double> weights(
          Core::LinAlg::Initialization::zero);
      std::vector<Core::LinAlg::SerialDenseVector> knots(true);
      const bool zero_size =
          Core::FE::Nurbs::get_my_nurbs_knots_and_weights(discret, &ele, knots, weights);
      if (zero_size) FOUR_C_THROW("get_my_nurbs_knots_and_weights has to return a non zero size.");

      // Get the position of the control points in the reference configuration.
      Core::LinAlg::Matrix<number_of_output_points * dim_output, 1, double> pos_controlpoints;
      for (unsigned int i_controlpoint = 0; i_controlpoint < (unsigned int)ele.num_node();
          ++i_controlpoint)
      {
        const Core::Nodes::Node* controlpoint = ele.nodes()[i_controlpoint];
        const auto size = controlpoint->x().size();
        for (size_t i_dim = 0; i_dim < size; ++i_dim)
          pos_controlpoints(dim_output * i_controlpoint + i_dim) = controlpoint->x()[i_dim];
      }

      Core::LinAlg::Matrix<dim_output, 1, double> point_result;
      Core::LinAlg::Matrix<dim_nurbs, 1, double> xi;
      for (unsigned int i_node_nurbs = 0; i_node_nurbs < number_of_output_points; i_node_nurbs++)
      {
        for (unsigned int i_dim_nurbs = 0; i_dim_nurbs < dim_nurbs; i_dim_nurbs++)
        {
          switch (celltype)
          {
            case Core::FE::CellType::nurbs9:
              xi(i_dim_nurbs) =
                  Core::FE::eleNodeNumbering_quad9_nodes_reference[numbering[i_node_nurbs]]
                                                                  [i_dim_nurbs];
              break;
            case Core::FE::CellType::nurbs27:
              xi(i_dim_nurbs) =
                  Core::FE::eleNodeNumbering_hex27_nodes_reference[numbering[i_node_nurbs]]
                                                                  [i_dim_nurbs];
              break;
            default:
              FOUR_C_THROW("The node numbering for the nurbs element shape {} is not implemented",
                  Core::FE::cell_type_to_string(ele.shape()));
          }
        }

        // Get the position at the parameter coordinate.
        point_result = Core::FE::Nurbs::eval_nurbs_interpolation<number_of_output_points, dim_nurbs,
            dim_output>(pos_controlpoints, xi, weights, knots, ele.shape());

        for (unsigned int i_dim = 0; i_dim < dim_output; i_dim++)
          point_coordinates.push_back(point_result(i_dim));
      }
    }

    return number_of_output_points;
  }

  /**
   * \brief Helper function to append the coordinates of vertices of NURBS elements.
   */
  unsigned int append_visualization_geometry_nurbs_ele(const Core::Elements::Element& ele,
      const Core::FE::Discretization& discret, std::vector<uint8_t>& cell_types,
      std::vector<double>& point_coordinates)
  {
    using implemented_celltypes =
        Core::FE::CelltypeSequence<Core::FE::CellType::nurbs9, Core::FE::CellType::nurbs27>;
    return Core::FE::cell_type_switch<implemented_celltypes>(ele.shape(),
        [&](auto celltype_t)
        {
          return append_visualization_geometry_nurbs<celltype_t()>(
              ele, discret, cell_types, point_coordinates);
        });
  }

  /**
   * \brief Add dof based results to point data vector for elements that use Lagrange shape
   * functions.
   *
   * @param ele (in) Element
   * @param discret (in) discretization
   * @param result_data_dofbased (in) Global vector with results
   * @param result_num_dofs_per_node (in) Number of scalar values per point.
   * @param read_result_data_from_dofindex (in) Starting DOF index for the nodal DOFs. This is
   * used if not all nodal DOFs should be output, e.g., velocity or pressure in fluid.
   * @param vtu_point_result_data (in/out) Result data vector.
   * @return Number of points added by this element.
   */
  unsigned int append_visualization_dof_based_result_data_vector_lagrange_ele(
      const Core::Elements::Element& ele, const Core::FE::Discretization& discret,
      const Core::LinAlg::Vector<double>& result_data_dofbased,
      const unsigned int result_num_dofs_per_node,
      const unsigned int read_result_data_from_dofindex, std::vector<double>& vtu_point_result_data)
  {
    const std::vector<int>& numbering =
        get_vtk_cell_type_from_element_cell_type(ele.shape()).second;

    for (unsigned int inode = 0; inode < static_cast<unsigned int>(ele.num_node()); ++inode)
    {
      std::vector<int> nodedofs;
      nodedofs.clear();

      // local storage position of desired dof gid in dofset number 0
      discret.dof(0u, ele.nodes()[numbering[inode]], nodedofs);

      for (unsigned int idof = 0; idof < result_num_dofs_per_node; ++idof)
      {
        // We have to make sure that this dof is actually available at this node. It can happen that
        // this is not the case e.g. in multi-scale simulations where different nodes in the mesh
        // have a different number of dofs.
        // Since this is not supported by vtu, we add a NaN in this case, see else branch
        if (nodedofs.size() > read_result_data_from_dofindex + idof)
        {
          const int lid =
              result_data_dofbased.get_map().lid(nodedofs[idof + read_result_data_from_dofindex]);
          vtu_point_result_data.push_back((result_data_dofbased).local_values_as_span()[lid]);
        }
        else
        {
          vtu_point_result_data.push_back(std::numeric_limits<double>::quiet_NaN());
        }
      }
    }

    return ele.num_node();
  }

  /*!
   * \brief Add node based results to point data vector for elements that use Lagrange shape
   * functions.
   *
   * @param ele (in) Element
   * @param result_data_nodebased (in) Global node-based vector with results.
   * @param result_num_components_per_node (in) Number of components per node.
   * @param point_result_data (in/out) Result data vector.
   * @return Number of points added by this element.
   */
  unsigned int append_visualization_node_based_result_data_vector_lagrange_ele(
      const Core::Elements::Element& ele,
      const Core::LinAlg::MultiVector<double>& result_data_nodebased,
      const int result_num_components_per_node, std::vector<double>& point_result_data)
  {
    const std::vector<int>& numbering =
        Core::IO::get_vtk_cell_type_from_element_cell_type(ele.shape()).second;

    for (unsigned int inode = 0; inode < static_cast<unsigned int>(ele.num_node()); ++inode)
    {
      const Core::Nodes::Node* node = ele.nodes()[numbering[inode]];
      const int lid = node->lid();

      for (int component_i = 0; component_i < result_num_components_per_node; ++component_i)
      {
        const auto& column = (result_data_nodebased)(component_i);

        if (lid > -1)
          point_result_data.push_back(column.local_values_as_span()[lid]);
        else
          FOUR_C_THROW("received illegal node local id: {}", lid);
      }
    }

    return ele.num_node();
  }

  /**
   * \brief Add dof based results to point data vector that use second-order NURBS shape functions.
   *
   * @param ele (in) Element
   * @param discret (in) discretization
   * @param result_data_dofbased (in) Global vector with results
   * @param read_result_data_from_dofindex (in) Starting DOF index for the nodal DOFs. This is
   * used if not all nodal DOFs should be output, e.g., velocity or pressure in fluid.
   * @param vtu_point_result_data (in/out) Result data vector.
   * @return Number of points added by this element.
   */
  template <Core::FE::CellType celltype, unsigned int result_num_dofs_per_node>
  unsigned int append_visualization_dof_based_result_data_vector_nurbs(
      const Core::Elements::Element& ele, const Core::FE::Discretization& discret,
      const Core::LinAlg::Vector<double>& result_data_dofbased,
      const unsigned int read_result_data_from_dofindex, std::vector<double>& vtu_point_result_data)
  {
    constexpr int number_of_output_points = Core::FE::num_nodes(celltype);
    constexpr int dim_nurbs = Core::FE::dim<celltype>;

    // Get the vtk cell information
    const auto vtk_cell_info = get_vtk_cell_type_for_nurbs_elements<celltype>();
    const std::vector<int>& numbering = vtk_cell_info.second;

    // Add the data at the nodes of the nurbs visualization.
    {
      // Get the knots and weights for this element.
      Core::LinAlg::Matrix<number_of_output_points, 1, double> weights(
          Core::LinAlg::Initialization::zero);
      std::vector<Core::LinAlg::SerialDenseVector> knots(true);
      const bool zero_size =
          Core::FE::Nurbs::get_my_nurbs_knots_and_weights(discret, &ele, knots, weights);
      if (zero_size) FOUR_C_THROW("get_my_nurbs_knots_and_weights has to return a non zero size.");

      // Get the element result vector.
      Core::LinAlg::Matrix<number_of_output_points * result_num_dofs_per_node, 1, double>
          dof_result;
      std::vector<int> lm, lmowner, lmstride;
      ele.location_vector(discret, lm, lmowner, lmstride);
      std::vector<double> ele_dof_values = Core::FE::extract_values(result_data_dofbased, lm);

      const auto total_dofs_per_node = ele_dof_values.size() / number_of_output_points;
      FOUR_C_ASSERT((ele_dof_values.size() % total_dofs_per_node) == 0,
          "Unequal number of dofs per node within an element! ");

      for (auto i_node_nurbs = 0; i_node_nurbs < number_of_output_points; ++i_node_nurbs)
      {
        for (unsigned int i_result = 0; i_result < result_num_dofs_per_node; ++i_result)
        {
          dof_result(i_node_nurbs * result_num_dofs_per_node + i_result) = ele_dof_values.at(
              i_node_nurbs * total_dofs_per_node + read_result_data_from_dofindex + i_result);
        }
      }

      // Loop over the nodes of the nurbs element.
      Core::LinAlg::Matrix<result_num_dofs_per_node, 1, double> point_result;
      Core::LinAlg::Matrix<dim_nurbs, 1, double> xi;
      for (unsigned int i_node_nurbs = 0; i_node_nurbs < number_of_output_points; i_node_nurbs++)
      {
        for (unsigned int i = 0; i < dim_nurbs; i++)
        {
          switch (celltype)
          {
            case Core::FE::CellType::nurbs9:
              xi(i) = Core::FE::eleNodeNumbering_quad9_nodes_reference[numbering[i_node_nurbs]][i];
              break;
            case Core::FE::CellType::nurbs27:
              xi(i) = Core::FE::eleNodeNumbering_hex27_nodes_reference[numbering[i_node_nurbs]][i];
              break;
            default:
              FOUR_C_THROW("The node numbering for the nurbs element shape {} is not implemented",
                  Core::FE::cell_type_to_string(ele.shape()));
          }
        }

        // Get the field value at the parameter coordinate
        point_result = Core::FE::Nurbs::eval_nurbs_interpolation<number_of_output_points, dim_nurbs,
            result_num_dofs_per_node>(dof_result, xi, weights, knots, ele.shape());

        for (unsigned int i_dim = 0; i_dim < result_num_dofs_per_node; i_dim++)
          vtu_point_result_data.push_back(point_result(i_dim));
      }
    }

    return number_of_output_points;
  }

  /*!
   * @brief Add node based results to point data vector that use second-order NURBS shape functions.
   *
   * @tparam celltype cell type of the element
   * @tparam result_num_components_per_node result vector has this many components per node
   * @param ele (in) element
   * @param discret (in) discretization
   * @param result_data_nodebased (in) Global node based vector with results
   * @param vtu_point_result_data (in/out) Result data vector
   * @return Number of points added by this element
   */
  template <Core::FE::CellType celltype, unsigned int result_num_components_per_node>
  unsigned int append_visualization_node_based_result_data_vector_nurbs(
      const Core::Elements::Element& ele, const Core::FE::Discretization& discret,
      const Core::LinAlg::MultiVector<double>& result_data_nodebased,
      std::vector<double>& vtu_point_result_data)
  {
    constexpr int number_of_output_points = Core::FE::num_nodes(celltype);
    constexpr int dim_nurbs = Core::FE::dim<celltype>;

    // Get the vtk cell information
    const auto vtk_cell_info = get_vtk_cell_type_for_nurbs_elements<celltype>();
    const std::vector<int>& numbering = vtk_cell_info.second;

    // Add the data at the nodes of the nurbs visualization.
    {
      // Get the knots and weights for this element.
      Core::LinAlg::Matrix<number_of_output_points, 1, double> weights(
          Core::LinAlg::Initialization::zero);
      std::vector<Core::LinAlg::SerialDenseVector> knots(true);
      const bool zero_size =
          Core::FE::Nurbs::get_my_nurbs_knots_and_weights(discret, &ele, knots, weights);
      if (zero_size) FOUR_C_THROW("get_my_nurbs_knots_and_weights has to return a non zero size.");

      // Get the element result vector.
      Core::LinAlg::Matrix<number_of_output_points * result_num_components_per_node, 1, double>
          result;
      std::vector<double> ele_node_values;
      Core::FE::extract_my_node_based_values(&ele, ele_node_values, result_data_nodebased);

      for (auto i_node_nurbs = 0; i_node_nurbs < number_of_output_points; ++i_node_nurbs)
      {
        for (unsigned int component_i = 0; component_i < result_num_components_per_node;
            ++component_i)
        {
          result(i_node_nurbs * result_num_components_per_node + component_i) =
              ele_node_values.at(i_node_nurbs * result_num_components_per_node + component_i);
        }
      }

      // Loop over the nodes of the nurbs element.
      Core::LinAlg::Matrix<result_num_components_per_node, 1, double> point_result;
      Core::LinAlg::Matrix<dim_nurbs, 1, double> xi;
      for (unsigned int i_node_nurbs = 0; i_node_nurbs < number_of_output_points; i_node_nurbs++)
      {
        for (unsigned int i = 0; i < dim_nurbs; i++)
        {
          switch (celltype)
          {
            case Core::FE::CellType::nurbs9:
              xi(i) = Core::FE::eleNodeNumbering_quad9_nodes_reference[numbering[i_node_nurbs]][i];
              break;
            case Core::FE::CellType::nurbs27:
              xi(i) = Core::FE::eleNodeNumbering_hex27_nodes_reference[numbering[i_node_nurbs]][i];
              break;
            default:
              FOUR_C_THROW("The node numbering for the nurbs element shape {} is not implemented",
                  Core::FE::cell_type_to_string(ele.shape()));
          }
        }

        // Get the field value at the parameter coordinate
        point_result = Core::FE::Nurbs::eval_nurbs_interpolation<number_of_output_points, dim_nurbs,
            result_num_components_per_node>(result, xi, weights, knots, ele.shape());

        for (unsigned int i_dim = 0; i_dim < result_num_components_per_node; i_dim++)
          vtu_point_result_data.push_back(point_result(i_dim));
      }
    }

    return number_of_output_points;
  }

  /**
   * \brief Add dof based results to point data vector for elements that use NURBS shape functions.
   *
   * @param ele (in) Element
   * @param discret (in) discretization
   * @param result_data_dofbased (in) Global vector with results
   * @param result_num_dofs_per_node (in) Number of scalar values per point.
   * @param read_result_data_from_dofindex (in) Starting DOF index for the nodal DOFs. This is
   * used if not all nodal DOFs should be output, e.g., velocity or pressure in fluid.
   * @param vtu_point_result_data (in/out) Result data vector.
   * @return Number of points added by this element.
   */
  unsigned int append_visualization_dof_based_result_data_vector_nurbs_ele(
      const Core::Elements::Element& ele, const Core::FE::Discretization& discret,
      const Core::LinAlg::Vector<double>& result_data_dofbased,
      const unsigned int result_num_dofs_per_node,
      const unsigned int read_result_data_from_dofindex, std::vector<double>& vtu_point_result_data)
  {
    using implemented_celltypes =
        Core::FE::CelltypeSequence<Core::FE::CellType::nurbs9, Core::FE::CellType::nurbs27>;
    return Core::FE::cell_type_switch<implemented_celltypes>(ele.shape(),
        [&](auto celltype_t)
        {
          switch (result_num_dofs_per_node)
          {
            case 1:
              return append_visualization_dof_based_result_data_vector_nurbs<celltype_t(), 1>(ele,
                  discret, result_data_dofbased, read_result_data_from_dofindex,
                  vtu_point_result_data);
            case 2:
              return append_visualization_dof_based_result_data_vector_nurbs<celltype_t(), 2>(ele,
                  discret, result_data_dofbased, read_result_data_from_dofindex,
                  vtu_point_result_data);
            case 3:
              return append_visualization_dof_based_result_data_vector_nurbs<celltype_t(), 3>(ele,
                  discret, result_data_dofbased, read_result_data_from_dofindex,
                  vtu_point_result_data);
            default:
              FOUR_C_THROW("The case for result_num_dofs_per_node = {} is not implemented",
                  result_num_dofs_per_node);
          }
        });
  }

  /*!
   * @brief Add node based results to point data vector for elements that use NURBS shape functions.
   *
   * @param ele (in) element
   * @param discret (in) discretization
   * @param result_data_nodebased (in) global vector with node based result data
   * @param result_num_components_per_node (in) result vector has this many components per node
   * @param vtu_point_result_data (in/out) result data vector
   * @return number of points added by this element
   */
  unsigned int append_visualization_node_based_result_data_vector_nurbs_ele(
      const Core::Elements::Element& ele, const Core::FE::Discretization& discret,
      const Core::LinAlg::MultiVector<double>& result_data_nodebased,
      const unsigned int result_num_components_per_node, std::vector<double>& vtu_point_result_data)
  {
    using implemented_celltypes =
        Core::FE::CelltypeSequence<Core::FE::CellType::nurbs9, Core::FE::CellType::nurbs27>;
    return Core::FE::cell_type_switch<implemented_celltypes>(ele.shape(),
        [&](auto celltype_t)
        {
          switch (result_num_components_per_node)
          {
            case 2:
              return append_visualization_node_based_result_data_vector_nurbs<celltype_t(), 2>(
                  ele, discret, result_data_nodebased, vtu_point_result_data);
            case 6:
              return append_visualization_node_based_result_data_vector_nurbs<celltype_t(), 6>(
                  ele, discret, result_data_nodebased, vtu_point_result_data);
            default:
              FOUR_C_THROW("The case for result_num_components_per_node = {} is not implemented",
                  result_num_components_per_node);
          }
        });
  }

}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif
