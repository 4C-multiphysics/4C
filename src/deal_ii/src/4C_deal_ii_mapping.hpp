// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_DEAL_II_MAPPING_HPP
#define FOUR_C_DEAL_II_MAPPING_HPP

#include "4C_config.hpp"

#include "4C_deal_ii_context.hpp"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q1.h>
FOUR_C_NAMESPACE_OPEN

namespace DealiiWrappers
{
  template <int dim, int spacedim>
  class Context;

  /**
   * @brief Class holding a mapping collection that is used to handle a collection of
   * dealii::Mapping objects describing the mapping from the reference cell to the real cell.
   *
   * @note This is a geometric mapping, i.e. it describes the coordinate transformation
   * from the reference cell to the real cell.
   * In 4C this mapping is typically not computed or handled explicitly instead
   * a isogeometric mapping is set up from the coordinates of the nodes of the finite element
   * directly and then implicitly uses the shape functions on a given element. The mapping is thus
   * always implicitly given and tied directly to the FiniteElement discretization.
   * Deal.II has a different approach and decouples this geometric mapping from the actual finite
   * element discretization. This can allow for more flexibility as well as lower overhead as the
   * mappings can be optimized to the task and the required geometric accuracy can be chosen
   * independently of the actual finite element space. There are several mapping classes in deal.II,
   * see there for more information.
   *
   * @attention The difference between the 4C and deal.II approach are quite profound as the whole
   * philosophy of the geometric description and meshes is different. It is thus not enough to just
   * use the same "degree" of the mapping (e.g. MappingQ2 and quadratic finite element) to get the
   * actual same real cell descriptions. There are three main remarks that you should keep in mind:
   * 1. If you are using linear shape functions (hex) the deal.II MappingQ1 is sufficient (i.e. use
   * create_linear_mapping()).
   * 2. If you are working on a regular mesh where you know that the cell will never be distorted
   * the dealii::MappingQ1 is also sufficient (i.e. use create_linear_mapping()).
   * 3. Whenever you are using higher order shape functions (e.g. quadratic or cubic) and your
   * geometry might be distorted you will need to use the dealii::MappingFEField class (i.e. use
   * create_isoparametric_mapping()).
   *
   * The MappingContext class provides a way to create and manage these mappings. While most of the
   * managing is done via the hp::MappingCollection object, these mappings sometimes require
   * additional data to work properly. Most often this data is however not stored within the
   * mapping itself but is assumed to be provided and kept alive by the user. The other task of
   * this class is thus to keep any additional data that is required for the mapping alive for the
   * lifetime of the mapping. This is done via a pimpl structure that holds the additional data and
   * is not exposed to the user at all. The user only ever interacts with the already initialized
   * mapping objects.
   */
  template <int dim, int spacedim = dim>
  struct MappingContext
  {
    /**
     * Construct all the necessary mappings for the given context. In this case the mappings
     * constructed will be isoparametric mapping using the dealii::MappingFEField class.
     * For this a dealii::DoFHandler is created and stored within this class together with a
     * vector that holds the positions of the nodes in the real cell.
     */
    static MappingContext create_isoparametric_mapping(const Context<dim, spacedim>& context);

    /**
     * Construct a linear mapping (i.e. a dealii::MappingQ1) for each finite element in the context.
     */
    static MappingContext create_linear_mapping(const Context<dim, spacedim>& context);

    /**
     * Access to the underlying individual mappings. They are stored in accordance with the context
     * this was created from, i.e. get_mapping_collection()[i] should be used with the i-th Finite
     * element in the context (and thus the i-th active_fe_index on a cell).
     * @return
     */
    const dealii::hp::MappingCollection<dim, spacedim>& get_mapping_collection() const;

   private:
    dealii::hp::MappingCollection<dim, spacedim> mapping_collection_;

    struct ImplementationDetails
    {
      // =============================================================
      // Isoparametric mapping data
      dealii::DoFHandler<dim, spacedim> iso_dof_handler;
      dealii::LinearAlgebra::distributed::Vector<double> position_vector;
    };

    std::vector<std::unique_ptr<ImplementationDetails>> pimpl_;
  };


  namespace Internal
  {
    template <int dim, int spacedim = dim,
        typename VectorType = dealii::LinearAlgebra::distributed::Vector<double>>
    dealii::MappingFEField<dim, spacedim, VectorType> create_isoparametric_mapping(
        const Context<dim, spacedim>& context, VectorType& position_vector,
        dealii::DoFHandler<dim>& iso_dof_handler);
  }  // namespace Internal


  // ===========================================================================================
  // Implementation of the MappingContext methods

  template <int dim, int spacedim>
  MappingContext<dim, spacedim> MappingContext<dim, spacedim>::create_isoparametric_mapping(
      const Context<dim, spacedim>& context)
  {
    FOUR_C_ASSERT(context.n_finite_elements() == 1,
        "Currently only supported for the case that there is only one finite element in the "
        "context, since the underlying dealii::MappingFEField does not support multiple finite "
        "elements.");

    auto data_holder = std::make_unique<ImplementationDetails>();
    auto mapping = Internal::create_isoparametric_mapping<dim, spacedim>(
        context, data_holder->position_vector, data_holder->iso_dof_handler);
    MappingContext mapping_context;
    mapping_context.mapping_collection_.push_back(mapping);
    mapping_context.pimpl_.push_back(std::move(data_holder));
    return mapping_context;
  }
  template <int dim, int spacedim>
  MappingContext<dim, spacedim> MappingContext<dim, spacedim>::create_linear_mapping(
      const Context<dim, spacedim>& context)
  {
    MappingContext mapping_context;
    for (unsigned int i = 0; i < context.n_finite_elements(); ++i)
    {
      mapping_context.mapping_collection_.push_back(dealii::MappingQ1<dim, spacedim>());
      mapping_context.pimpl_.push_back(
          std::make_unique<ImplementationDetails>());  // empty implementation details
    }


    return mapping_context;
  }
  template <int dim, int spacedim>
  const dealii::hp::MappingCollection<dim, spacedim>&
  MappingContext<dim, spacedim>::get_mapping_collection() const
  {
    return mapping_collection_;
  }

  namespace Internal
  {
    template <int dim, int spacedim, typename VectorType>
    dealii::MappingFEField<dim, spacedim, VectorType> create_isoparametric_mapping(
        const Context<dim, spacedim>& context, VectorType& position_vector,
        dealii::DoFHandler<dim>& iso_dof_handler)
    {
      FOUR_C_ASSERT(context.n_finite_elements() == 1,
          "Currently only supported for the case that there is only one finite element in the "
          "context, since the underlying dealii::MappingFEField does not support multiple finite "
          "elements.");

      // create an internal dofhandler using the finite element that is provided
      // this is potentially a multicomponent system, so we have to get
      // the scalar valued finite element
      const auto& fe_system = context.get_finite_elements()[0];
      FOUR_C_ASSERT(fe_system.n_base_elements() == 1,
          "The finite element must have exactly one base element, since we are creating an "
          "isoparametric mapping. This is not the case for the finite element '{}'.",
          fe_system.get_name());
      const auto& fe = fe_system.base_element(0);

      // create an FE System object that has the right dimension
      dealii::FESystem<dim, spacedim> isoparametric_fe(fe, spacedim);

      // create a DofHandler for the isoparametric mapping
      iso_dof_handler.reinit(context.get_triangulation());
      iso_dof_handler.distribute_dofs(isoparametric_fe);

      // create ghosted vector for the positions of the nodes
      auto locally_relevant_dofs = dealii::DoFTools::extract_locally_relevant_dofs(iso_dof_handler);
      position_vector.reinit(iso_dof_handler.locally_owned_dofs(), locally_relevant_dofs,
          iso_dof_handler.get_communicator());

      // Now fill the position vector with the positions of the nodes
      for (const auto& cell : iso_dof_handler.active_cell_iterators())
      {
        // skip ghost cells
        if (not cell->is_locally_owned()) continue;

        // get the equivalent element in four_c
        const auto* element = context.to_element(cell);
        const unsigned int n_nodes = element->num_node();
        const auto* nodes = element->nodes();


        // get the dof indices for the cell
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
        cell->get_dof_indices(dof_indices);

        FOUR_C_ASSERT(n_nodes * spacedim == dofs_per_cell,
            "Since this is an isoparametric mapping, the number of nodes x {} must be equal to "
            "the number of dofs per cell.",
            spacedim);


        // we now have to assign the position of the nodes to the dof indices
        dealii::Vector<double> local_position_vector(dofs_per_cell);
        auto reordering =
            ConversionTools::FourCToDeal::reindex_shape_functions_scalar(element->shape());
        for (unsigned int n = 0; n < n_nodes; ++n)
        {
          const auto local_dealii_index = reordering[n];
          for (unsigned int d = 0; d < spacedim; ++d)
          {
            const auto local_vector_index =
                isoparametric_fe.component_to_system_index(d, local_dealii_index);
            local_position_vector[local_vector_index] = nodes[n]->x()[d];
          }
        }
        // now we can assign the local position vector to the global position vector
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          // only add local entries
          if (iso_dof_handler.locally_owned_dofs().is_element(dof_indices[i]))
          {
            position_vector[dof_indices[i]] = local_position_vector[i];
          }
        }
      }
      const dealii::ComponentMask mask(spacedim, true);
      return dealii::MappingFEField<dim, spacedim, VectorType>(
          iso_dof_handler, position_vector, mask);
    }
  }  // namespace Internal

}  // namespace DealiiWrappers
FOUR_C_NAMESPACE_CLOSE

#endif  // FOUR_C_DEAL_II_MAPPING_HPP
