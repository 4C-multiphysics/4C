// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_DISCRETIZATION_VISUALIZATION_WRITER_MESH_HPP
#define FOUR_C_IO_DISCRETIZATION_VISUALIZATION_WRITER_MESH_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_io_visualization_manager.hpp"
#include "4C_io_visualization_parameters.hpp"
#include "4C_linalg_vector.hpp"

#include <memory>
#include <optional>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  enum class OutputEntity
  {
    dof,      ///< The quantity to be written for output is dof-based
    element,  ///< The quantity to be written for output is element-based
    node      ///< The quantity to be written for output is node-based
  };

  /*!
   * \brief This object allows to write visualization output for a discretization, i.e., write the
   * mesh and results on the mesh to disk
   */
  class DiscretizationVisualizationWriterMesh
  {
   public:
    /**
     * @brief Constructor
     *
     * @param discretization (in)  Pointer to the discretization
     * @param parameters (in)      Visualization parameters
     * @param element_filter (in)  An optional function that returns true for all elements that
     *   should be included in the visualization. By default, all elements in the discretization are
     *   included.
     * @param filename (in)  An optional filename. By default, the name of the discretization
     *   is used.
     */
    DiscretizationVisualizationWriterMesh(
        const std::shared_ptr<const Core::FE::Discretization>& discretization,
        VisualizationParameters parameters,
        std::function<bool(const Core::Elements::Element* element)> element_filter =
            [](const Core::Elements::Element*) { return true; },
        const std::optional<std::string>& filename = std::nullopt);

    /**
     * @brief Reset state depending if the maps changed or not
     */
    void reset();

    /*!
     * @brief result data vector appended for output based on the provided output entity and context
     *
     * The result data vector can provide data in several forms, i.e. dof-based, element-based, or
     * node-based. The corresponding form is handed in from outside using the output_entity. Then
     * the method analyses the context vector to check whether the quantity is a vector or scalar
     * quantity.
     *
     * @example Consider your result vector consists of four components per node, i.e. 3 velocities
     * and one pressure (e.g. output of a fluid field). Then your context vector would be:
     * {"velocity", "velocity", "velocity", "pressure"}.
     * You could also decide to only output the pressure by assigning std::nullopt to the respective
     * position of the quantities you do not want to output. Consider you only want to output the
     * pressure the context vector would be: {std::nullopt, std::nullopt, std::nullopt, "pressure"}.
     *
     * @param result_data (in)    result data vector (dof-based, element-based, or node-based)
     * @param output_entity (in)  defining the output entity, i.e. dof-based, element-based, or
     *                            node-based
     * @param context (in)        context vector of size number components per node or element
     *                            defining the output name of the i-th component
     */
    void append_result_data_vector_with_context(
        const Core::LinAlg::MultiVector<double>& result_data, const OutputEntity output_entity,
        const std::vector<std::optional<std::string>>& context);

    /*!
     * @brief Appends a field data vector for output with the given name.
     *
     * The field data vector represents a single quantity in vector form, valid for the entire
     * field, e.g., an averaged stress.
     *
     * @param result_data (in)  field data vector
     * @param data_name (in)    name assigned to the output quantity
     */
    void append_field_data_vector(
        const std::vector<double>& result_data, const std::string& data_name) const;

   private:
    /**
     * @brief Append result vector with num_dof values per node to output data
     *
     * The results are expected to be stored in the following form (for result_num_dofs_per_node =
     * x, y, z):
     * [node1_x, node1_y, node1_z, node2_x, node2_y, node2_z, node3_x, node3_y, node3_z, ...]
     *
     * @param result_data_dofbased (in) Column based result data vector
     * @param result_num_dofs_per_node  (in) Number of scalar values per node
     * @param read_result_data_from_dofindex (in) Starting DOF index for the nodal DOFs. This is
     * used if not all nodal DOFs should be output, e.g., velocity or pressure in fluid.
     * @param resultname (in) Name of the field to be written to the visualization file
     */
    void append_dof_based_result_data_vector(
        const Core::LinAlg::Vector<double>& result_data_dofbased,
        unsigned int result_num_dofs_per_node, unsigned int read_result_data_from_dofindex,
        const std::string& resultname);

    /**
     * @brief Append result vector with num_components values per node to output data
     *
     * The results are expected to be stored in the following form (for result_num_dofs_per_node =
     * x, y, z):
     * [
     *   [node1_x, node1_y, node1_z]
     *   [node2_x, node2_y, node2_z],
     *   [node3_x, node3_y, node3_z],
     *   ...
     * ]
     *
     * @param result_data_nodebased (in) Column node based result data vector
     * @param result_num_components_per_node (in) Number of scalar values per node
     * @param resultname (in) Name of the field to be written to the visualization file
     */
    void append_node_based_result_data_vector(
        const Core::LinAlg::MultiVector<double>& result_data_nodebased,
        unsigned int result_num_components_per_node, const std::string& resultname);

    /**
     * @brief Append result vector with num_components values per element to output data
     *
     * The results are expected to be stored in the following form (for
     * result_num_components_per_element = x, y, z):
     * [ele1_x, ele1_y, ele1_z, ele2_x, ele2_y, ele2_z, ele3_x, ele3_y, ele3_z, ...]
     *
     * @param result_data_elementbased (in) Column based result data vector
     * @param result_num_components_per_element (in) Number of scalar values per element
     * @param resultname (in) Name of the field to be written to the visualization file
     */
    void append_element_based_result_data_vector(
        const Core::LinAlg::MultiVector<double>& result_data_elementbased,
        unsigned int result_num_components_per_element, const std::string& resultname);

   public:
    /**
     * \brief Write owner id for each element
     *
     * In order to process all solid elements (beam elements are handled by a separate output
     * writer), it is sufficient that each processor writes all elements in its row map. This
     * can easily be achieved by appending the PID of the writing processor.
     *
     * @param resultname (in) Name of the owner field in the visualization file
     */
    void append_element_owner(const std::string& resultname);

    /**
     * \brief Write the 4C internal element GIDs for each element
     *
     * @param resultname (in) Name of the field in the visualization file
     */
    void append_element_gid(const std::string& resultname);

    /**
     * \brief Write ghosting information to the elements
     *
     * For more details look at the documentation to Core::IO::append_element_ghosting_information
     */
    void append_element_ghosting_information();

    //! Write the material id for each element to the output
    void append_element_material_id();

    /**
     * \brief Write the 4C internal node GIDs for each node
     *
     * @param resultname (in) Name of the field in the visualization file
     */
    void append_node_gid(const std::string& resultname);

    /**
     * \brief Write the visualization files to disk
     */
    void write_to_disk(const double visualization_time, const int visualization_step);

   private:
    /** \brief Determine and set geometry data from elements based on reference configuration
     *
     * To simplify parallel output, we loop over each row element and let the element write its
     * topology to the the output file. This results in a "discontinuous" visualization, i.e., the
     * nodes of adjacent elements are not connected in the output file. In ParaView the global
     * nodal connectivity can be restored with the CleanToGrid filter.
     */
    void set_geometry_from_discretization();


   private:
    //! discretization containing elements of which geometry and result data shall be visualized
    std::shared_ptr<const Core::FE::Discretization> discretization_;

    //! The actual visualization writer object that additionally stores the geometry and result data
    std::shared_ptr<VisualizationManager> visualization_manager_;

    //! A filter function that returns true for all elements that should be visualized.
    std::function<bool(const Core::Elements::Element* element)> element_filter_;

    //! Node row and col maps the geometry of visualization writer is based on
    std::shared_ptr<Epetra_Map> noderowmap_last_geometry_set_;
    std::shared_ptr<Epetra_Map> nodecolmap_last_geometry_set_;
  };

  /**
   * \brief Add a vector with the length of num_proc to each element, which contains a 1 for the
   * ranks that ghost the element.
   * @param discretization (in) discretization
   * @param visualization_manager (in/out) Visualization writer object
   * @param element_predicate (in) A predicate function which returns whether a given element
   * should be included in the output.
   */
  void append_element_ghosting_information(const Core::FE::Discretization& discretization,
      VisualizationManager& visualization_manager,
      const std::function<bool(const Core::Elements::Element* ele)>& element_predicate);

}  // namespace Core::IO
FOUR_C_NAMESPACE_CLOSE

#endif
