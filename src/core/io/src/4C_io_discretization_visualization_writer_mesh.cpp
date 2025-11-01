// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_discretization_visualization_writer_mesh.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io.hpp"
#include "4C_io_element_vtk_cell_type_register.hpp"
#include "4C_io_visualization_manager.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_multi_vector.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_exceptions.hpp"

#include <utility>
FOUR_C_NAMESPACE_OPEN


namespace Core::IO
{

  /*-----------------------------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------------------------*/
  std::unique_ptr<Core::LinAlg::MultiVector<double>> convert_dof_vector_to_node_based_multi_vector(
      const Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& input_vector,
      const int number_of_dofset, const int number_of_dofs_per_node)
  {
    // initialize multi vector
    auto multi = std::make_unique<Core::LinAlg::MultiVector<double>>(
        *dis.node_row_map(), number_of_dofs_per_node, true);

    // get maps
    const Core::LinAlg::Map& vectormap = input_vector.get_map();

    // loop over nodes of the discretization
    for (int inode = 0; inode < dis.num_my_row_nodes(); ++inode)
    {
      // get current node
      Core::Nodes::Node* node = dis.l_row_node(inode);
      // copy each dof value of node
      for (int idof = 0; idof < number_of_dofs_per_node; ++idof)
        (*multi)(idof).local_values_as_span()[inode] =
            input_vector
                .local_values_as_span()[vectormap.lid(dis.dof(number_of_dofset, node, idof))];
    }

    return multi;
  }


  /*-----------------------------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------------------------*/
  DiscretizationVisualizationWriterMesh::DiscretizationVisualizationWriterMesh(
      const std::shared_ptr<const Core::FE::Discretization>& discretization,
      VisualizationParameters parameters,
      std::function<bool(const Core::Elements::Element* element)> element_filter,
      const std::optional<std::string>& filename)
      : discretization_(discretization),
        visualization_manager_(std::make_shared<VisualizationManager>(std::move(parameters),
            discretization->get_comm(), filename.value_or(discretization->name()))),
        element_filter_(std::move(element_filter))
  {
    set_geometry_from_discretization();
  }

  /*-----------------------------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------------------------*/
  void DiscretizationVisualizationWriterMesh::set_geometry_from_discretization()
  {
    // Todo assume 3D for now
    const unsigned int num_spatial_dimensions = 3;

    // count number of nodes; output is completely independent of the number of processors involved
    unsigned int num_row_elements = discretization_->num_my_row_elements();
    unsigned int num_nodes = 0;
    for (auto ele : discretization_->my_row_element_range())
    {
      num_nodes += ele.num_nodes();
    }

    // do not need to store connectivity indices here because we create a
    // contiguous array by the order in which we fill the coordinates (otherwise
    // need to adjust order of filling in the coordinates).
    auto& visualization_data = visualization_manager_->get_visualization_data();

    // Clear the visualization data, especially the cell connectivity such that it is rebuilt later.
    visualization_data.clear_data();

    std::vector<double>& point_coordinates = visualization_data.get_point_coordinates();
    point_coordinates.reserve(num_spatial_dimensions * num_nodes);

    std::vector<uint8_t>& cell_types = visualization_data.get_cell_types();
    cell_types.reserve(num_row_elements);

    std::vector<int32_t>& cell_offsets = visualization_data.get_cell_offsets();
    cell_offsets.reserve(num_row_elements);


    // loop over my elements and collect the geometry/grid data
    unsigned int pointcounter = 0;
    unsigned int num_skipped_eles = 0;

    for (auto ele : discretization_->my_row_element_range())
    {
      if (element_filter_(ele.user_element()))
      {
        pointcounter += ele.user_element()->append_visualization_geometry(
            *discretization_, cell_types, point_coordinates);
        cell_offsets.push_back(pointcounter);
      }
      else
        ++num_skipped_eles;
    }

    // safety checks
    FOUR_C_ASSERT_ALWAYS(point_coordinates.size() == num_spatial_dimensions * pointcounter,
        "Expected {} coordinate values, but got {}.", num_spatial_dimensions * pointcounter,
        point_coordinates.size());

    FOUR_C_ASSERT_ALWAYS(cell_types.size() == num_row_elements - num_skipped_eles,
        "Expected {} cell type values, but got {}.", num_row_elements, cell_types.size());

    FOUR_C_ASSERT_ALWAYS(cell_offsets.size() == num_row_elements - num_skipped_eles,
        "Expected {} cell offset values, but got {}.", num_row_elements, cell_offsets.size());

    // store node row and col maps (needed to check for changed parallel distribution)
    noderowmap_last_geometry_set_ =
        std::make_shared<Core::LinAlg::Map>(*discretization_->node_row_map());
    nodecolmap_last_geometry_set_ =
        std::make_shared<Core::LinAlg::Map>(*discretization_->node_col_map());
  }

  /*-----------------------------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------------------------*/
  void DiscretizationVisualizationWriterMesh::reset()
  {
    // check if parallel distribution of discretization changed
    int map_changed =
        ((not noderowmap_last_geometry_set_->same_as(*discretization_->node_row_map())) or
            (not nodecolmap_last_geometry_set_->same_as(*discretization_->node_col_map())));
    int map_changed_allproc(0);
    map_changed_allproc = Core::Communication::max_all(map_changed, discretization_->get_comm());

    // reset geometry of visualization writer
    if (map_changed_allproc) set_geometry_from_discretization();
  }

  /*-----------------------------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------------------------*/
  void DiscretizationVisualizationWriterMesh::append_result_data_vector_with_context(
      const Core::LinAlg::MultiVector<double>& result_data, const OutputEntity output_entity,
      const std::vector<std::optional<std::string>>& context)
  {
    std::set<std::string> unique_names;
    std::multimap<std::string, unsigned int> context_map;

    // interpret the given context
    const unsigned int number_context_elements = context.size();
    for (unsigned int context_item = 0; context_item < number_context_elements; ++context_item)
    {
      const auto& item_string = context[context_item];
      if (item_string.has_value())
      {
        unique_names.insert(item_string.value());
        context_map.insert(std::make_pair(item_string.value(), context_item));
      }
    }

    for (const auto& name : unique_names)
    {
      switch (output_entity)
      {
        case OutputEntity::dof:
        {
          // obtain minimum offset index of quantity 'name' we currently want to append to the
          // output
          unsigned int min_index_of_name = context.size();
          auto [first_ele_with_name, last_ele_with_name] = context_map.equal_range(name);
          for (auto item = first_ele_with_name; item != last_ele_with_name; ++item)
          {
            min_index_of_name = std::min(min_index_of_name, item->second);
          }

          // finally append the data
          append_dof_based_result_data_vector(
              result_data(0), context_map.count(name), min_index_of_name, name);
          break;
        }
        case OutputEntity::element:
        {
          if (unique_names.size() != 1)
            FOUR_C_THROW(
                "For now, element based output can only be written if there is only one unique "
                "entry in the context object");
          append_element_based_result_data_vector(result_data, context_map.count(name), name);
          break;
        }
        case OutputEntity::node:
        {
          if (unique_names.size() != 1)
            FOUR_C_THROW(
                "For now, node based output can only be written if there is only one unique "
                "entry in the context object");
          append_node_based_result_data_vector(result_data, context_map.count(name), name);
          break;
        }
        default:
          FOUR_C_THROW("The output entity you try to output is unknown!");
      }
    }
  }

  void DiscretizationVisualizationWriterMesh::append_field_data_vector(
      const std::vector<double>& result_data, const std::string& data_name) const
  {
    visualization_manager_->get_visualization_data().set_field_data_vector(data_name, result_data);
  }
  /*-----------------------------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------------------------*/
  void DiscretizationVisualizationWriterMesh::append_dof_based_result_data_vector(
      const Core::LinAlg::Vector<double>& result_data_dofbased,
      const unsigned int result_num_dofs_per_node,
      const unsigned int read_result_data_from_dofindex, const std::string& resultname)
  {
    /* the idea is to transform the given data to a 'point data vector' and append it to the
     * collected solution data vectors by calling
     * append_visualization_dof_based_result_data_vector() */

    auto convert_to_col_map_if_necessary = [&](const Core::LinAlg::Vector<double>& vector)
    {
      if (discretization_->dof_col_map()->same_as(vector.get_map()))
      {
        return vector;
      }
      else if (discretization_->dof_row_map()->same_as(vector.get_map()))
      {
        auto vector_col_map = Core::LinAlg::Vector<double>(*discretization_->dof_col_map(), true);
        Core::LinAlg::export_to(vector, vector_col_map);
        return vector_col_map;
      }
      FOUR_C_THROW("'dof_vector' is neither in column nor in row map!");
    };

    auto result_data_dofbased_col_map = convert_to_col_map_if_necessary(result_data_dofbased);

    // safety checks
    FOUR_C_ASSERT(discretization_->dof_col_map()->same_as(result_data_dofbased_col_map.get_map()),
        "Received map of dof-based result data vector does not match the discretization's dof "
        "col map.");

    // count number of nodes for this visualization
    unsigned int num_nodes = 0;
    for (auto ele : discretization_->my_row_element_range())
    {
      num_nodes += ele.num_nodes();
    }

    std::vector<double> point_result_data;
    point_result_data.reserve(result_num_dofs_per_node * num_nodes);

    unsigned int pointcounter = 0;

    for (auto ele : discretization_->my_row_element_range())
    {
      if (element_filter_(ele.user_element()))
      {
        pointcounter += ele.user_element()->append_visualization_dof_based_result_data_vector(
            *discretization_, result_data_dofbased_col_map, result_num_dofs_per_node,
            read_result_data_from_dofindex, point_result_data);
      }
    }

    // sanity check
    FOUR_C_ASSERT_ALWAYS(point_result_data.size() == result_num_dofs_per_node * pointcounter,
        "Expected {} result values, but got {}.", result_num_dofs_per_node * pointcounter,
        point_result_data.size());

    visualization_manager_->get_visualization_data().set_point_data_vector(
        resultname, point_result_data, result_num_dofs_per_node);
  }

  /*-----------------------------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------------------------*/
  void DiscretizationVisualizationWriterMesh::append_node_based_result_data_vector(
      const Core::LinAlg::MultiVector<double>& result_data_nodebased,
      const unsigned int result_num_components_per_node, const std::string& resultname)
  {
    /* the idea is to transform the given data to a 'point data vector' and append it to the
     * collected solution data vectors by calling
     * append_visualization_node_based_result_data_vector() */

    auto convert_to_col_map_if_necessary = [&](const Core::LinAlg::MultiVector<double>& vector)
    {
      if (discretization_->node_col_map()->same_as(vector.get_map()))
      {
        return vector;
      }
      else if (discretization_->node_row_map()->same_as(vector.get_map()))
      {
        auto vector_col_map = Core::LinAlg::MultiVector<double>(
            *discretization_->node_col_map(), result_num_components_per_node, true);
        Core::LinAlg::export_to(vector, vector_col_map);
        return vector_col_map;
      }
      FOUR_C_THROW("'node_vector' is neither in column nor in row map!");
    };

    auto result_data_nodebased_col_map = convert_to_col_map_if_necessary(result_data_nodebased);

    // safety checks
    FOUR_C_ASSERT(static_cast<unsigned int>(result_data_nodebased_col_map.num_vectors()) ==
                      result_num_components_per_node,
        "Expected Core::LinAlg::MultiVector<double> with {} columns but got {}.",
        result_num_components_per_node, result_data_nodebased_col_map.num_vectors());

    FOUR_C_ASSERT(discretization_->node_col_map()->same_as(result_data_nodebased_col_map.get_map()),
        "Received map of node-based result data vector does not match the discretization's node "
        "col map.");

    // count number of nodes
    unsigned int num_nodes = 0;
    for (auto ele : discretization_->my_row_element_range())
    {
      num_nodes += ele.num_nodes();
    }

    std::vector<double> point_result_data;
    point_result_data.reserve(result_num_components_per_node * num_nodes);

    unsigned int pointcounter = 0;

    for (auto ele : discretization_->my_row_element_range())
    {
      if (element_filter_(ele.user_element()))
      {
        pointcounter +=
            ele.user_element()->append_visualization_node_based_result_data_vector(*discretization_,
                result_data_nodebased_col_map, result_num_components_per_node, point_result_data);
      }
    }

    // sanity check
    FOUR_C_ASSERT_ALWAYS(point_result_data.size() == result_num_components_per_node * pointcounter,
        "Expected {} result values, but got {}.", result_num_components_per_node * pointcounter,
        point_result_data.size());

    visualization_manager_->get_visualization_data().set_point_data_vector(
        resultname, point_result_data, result_num_components_per_node);
  }

  /*-----------------------------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------------------------*/
  void DiscretizationVisualizationWriterMesh::append_element_based_result_data_vector(
      const Core::LinAlg::MultiVector<double>& result_data_elementbased,
      const unsigned int result_num_components_per_element, const std::string& resultname)
  {
    /* the idea is to transform the given data to a 'cell data vector' and append it to the
     * collected solution data vectors by adding it to the cell_data_vector of the visualization
     * data */

    // safety check
    FOUR_C_ASSERT(static_cast<unsigned int>(result_data_elementbased.num_vectors()) ==
                      result_num_components_per_element,
        "Expected Core::LinAlg::MultiVector<double> with {} columns but got {}.",
        result_num_components_per_element, result_data_elementbased.num_vectors());

    FOUR_C_ASSERT(discretization_->element_row_map()->same_as(result_data_elementbased.get_map()),
        "Received map of element-based result data vector does not match the discretization's "
        "element row map.");

    // count number of elements for each processor
    auto num_row_elements = static_cast<unsigned int>(discretization_->num_my_row_elements());

    std::vector<double> cell_result_data;
    cell_result_data.reserve(result_num_components_per_element * num_row_elements);

    unsigned int cellcounter = 0;

    for (unsigned int iele = 0; iele < num_row_elements; ++iele)
    {
      const Core::Elements::Element* ele = discretization_->l_row_element(iele);

      if (!element_filter_(ele)) continue;

      for (unsigned int icpe = 0; icpe < result_num_components_per_element; ++icpe)
      {
        const auto& column = result_data_elementbased(icpe);

        cell_result_data.push_back(column.local_values_as_span()[iele]);
      }

      ++cellcounter;
    }

    // sanity check
    FOUR_C_ASSERT_ALWAYS(cell_result_data.size() == result_num_components_per_element * cellcounter,
        "Expected {} result values, but got {}.", result_num_components_per_element * cellcounter,
        cell_result_data.size());

    visualization_manager_->get_visualization_data().set_cell_data_vector(
        resultname, cell_result_data, result_num_components_per_element);
  }

  /*-----------------------------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------------------------*/
  void DiscretizationVisualizationWriterMesh::append_element_owner(const std::string& resultname)
  {
    // Vector with element owner for elements in the row map.
    std::vector<double> owner_of_row_elements;
    owner_of_row_elements.reserve(discretization_->num_my_row_elements());

    const int my_pid = Core::Communication::my_mpi_rank(discretization_->get_comm());
    for (auto ele : discretization_->my_row_element_range())
    {
      if (element_filter_(ele.user_element())) owner_of_row_elements.push_back(my_pid);
    }

    // Pass data to the output writer.
    visualization_manager_->get_visualization_data().set_cell_data_vector(
        resultname, owner_of_row_elements, 1);
  }

  /*-----------------------------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------------------------*/
  void DiscretizationVisualizationWriterMesh::append_element_gid(const std::string& resultname)
  {
    // Vector with element IDs for elements in the row map.
    std::vector<double> gid_of_row_elements;
    gid_of_row_elements.reserve(discretization_->num_my_row_elements());

    for (auto ele : discretization_->my_row_element_range())
    {
      if (element_filter_(ele.user_element())) gid_of_row_elements.push_back(ele.global_id());
    }

    // Pass data to the output writer.
    visualization_manager_->get_visualization_data().set_cell_data_vector(
        resultname, gid_of_row_elements, 1);
  }


  /*-----------------------------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------------------------*/
  void DiscretizationVisualizationWriterMesh::append_element_ghosting_information()
  {
    Core::IO::append_element_ghosting_information(
        *discretization_, *visualization_manager_, element_filter_);
  }

  /*-----------------------------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------------------------*/
  void DiscretizationVisualizationWriterMesh::append_element_material_id()
  {
    // vector with material IDs for elements in the row map.
    std::vector<int> material_id_of_row_elements;
    material_id_of_row_elements.reserve(discretization_->num_my_row_elements());

    for (auto ele : discretization_->my_row_element_range())
    {
      if (element_filter_(ele.user_element()))
        material_id_of_row_elements.push_back(ele.user_element()->material(0)->parameter()->id());
    }

    // Pass data to the output writer.
    visualization_manager_->get_visualization_data().set_cell_data_vector(
        "material_id", material_id_of_row_elements, 1);
  }

  /*-----------------------------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------------------------*/
  void DiscretizationVisualizationWriterMesh::append_node_gid(const std::string& resultname)
  {
    // count number of nodes; output is completely independent of the number of processors involved
    int num_nodes = 0;
    for (auto ele : discretization_->my_row_element_range())
    {
      num_nodes += ele.num_nodes();
    }

    // Setup the vector with the GIDs of the nodes.
    std::vector<int> gid_of_nodes;
    gid_of_nodes.reserve(num_nodes);

    // Loop over each element and add the node GIDs.
    for (auto ele : discretization_->my_row_element_range())
    {
      if (!element_filter_(ele.user_element())) continue;

      // Add the node GIDs.
      const std::vector<int>& numbering =
          Core::IO::get_vtk_cell_type_from_element_cell_type(ele.user_element()->shape()).second;
      auto nodes = ele.nodes();
      for (const auto vtk_index : numbering) gid_of_nodes.push_back(nodes[vtk_index].global_id());
    }

    visualization_manager_->get_visualization_data().set_point_data_vector<int>(
        resultname, gid_of_nodes, 1);
  }

  /*-----------------------------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------------------------*/
  void DiscretizationVisualizationWriterMesh::write_to_disk(
      const double visualization_time, const int visualization_step)
  {
    visualization_manager_->write_to_disk(visualization_time, visualization_step);
  }

  /*-----------------------------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------------------------*/
  void append_element_ghosting_information(const Core::FE::Discretization& discretization,
      VisualizationManager& visualization_manager,
      const std::function<bool(const Core::Elements::Element* ele)>& element_predicate)
  {
    // Set up a multivector which will be populated with all ghosting information.
    MPI_Comm comm = discretization.element_col_map()->get_comm();
    const int n_proc = Core::Communication::num_mpi_ranks(comm);
    const int my_proc = Core::Communication::my_mpi_rank(comm);

    // Create Vectors to store the ghosting information.
    LinAlg::FEVector<double> ghosting_information(*discretization.element_row_map(), n_proc, false);

    // Get elements ghosted by this rank.
    std::vector<int> my_ghost_elements;
    my_ghost_elements.clear();
    int count = 0;
    for (auto ele : discretization.my_col_element_range())
    {
      if (element_predicate(ele.user_element()))
      {
        count++;
        if (ele.owner() != my_proc) my_ghost_elements.push_back(ele.global_id());
      }
    }

    // Add to the multi vector.
    std::vector<double> values(my_ghost_elements.size(), 1.0);
    ghosting_information.sum_into_global_values(
        my_ghost_elements.size(), my_ghost_elements.data(), values.data(), my_proc);

    // Assemble over all processors.
    ghosting_information.complete();

    // Output the ghosting data of the elements owned by this proc.
    std::vector<double> ghosted_elements;
    ghosted_elements.reserve(count * n_proc);
    for (auto ele : discretization.my_row_element_range())
    {
      if (element_predicate(ele.user_element()))
      {
        const int local_row = ghosting_information.get_map().lid(ele.global_id());
        if (local_row == -1) FOUR_C_THROW("The element has to exist in the row map.");
        for (int i_proc = 0; i_proc < n_proc; i_proc++)
          ghosted_elements.push_back(
              (ghosting_information.get_ref_of_epetra_fevector())[i_proc][local_row]);
      }
    }

    visualization_manager.get_visualization_data().set_cell_data_vector(
        "element_ghosting", ghosted_elements, n_proc);
  }
}  // namespace Core::IO
FOUR_C_NAMESPACE_CLOSE
